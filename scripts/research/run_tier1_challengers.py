#!/usr/bin/env python
"""Execute E05/E06 inner-only gates with frozen candidate and policy semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.tier1_challenger import (  # noqa: E402
    GYRO5_FEATURE_NAMES, META_FEATURE_NAMES, adaptive_meta_features,
    classify_inner_gate, gyro5_feature_block,
)
from src.pipeline import runner as rr  # noqa: E402
from src.pipeline.challenger.e03 import _candidate_evidence, registered_e03_config  # noqa: E402
from src.pipeline.challenger.protection import assert_snapshot_unchanged, protected_snapshot  # noqa: E402
from src.pipeline.challenger.replay import official_match_indices  # noqa: E402
from src.pipeline.crossfit import crossfit_predict_proba  # noqa: E402

FIXED_RELEASE_WEIGHTS = (0.50, 0.25, 0.25, 0.00, 0.25)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def git_sha(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def metrics_dict(value) -> dict:
    return {"f1": value.f1, "precision": value.ppv, "recall": value.sensitivity,
            "n_tp": value.n_tp, "n_pred": value.n_pred, "n_true": value.n_true,
            "fp": value.n_pred - value.n_tp, "fn": value.n_true - value.n_tp}


class GyroStore:
    def __init__(self, root: Path):
        self.root, self.cache = root, {}
        self.hits = self.misses = 0

    def session(self, sid: str):
        if sid in self.cache:
            self.hits += 1
            return self.cache[sid]
        self.misses += 1
        with np.load(self.root / "cache" / "sessions" / f"{sid}.npz", allow_pickle=False) as z:
            rate = float(z["row_rate"])
            step = max(1, int(round(rate / 10.0)))
            value = (z["gyro"][:, ::step].astype(np.float64), z["t_acc"][::step].astype(np.int64),
                     z["imu_valid"][::step].astype(bool), rate / step)
        self.cache[sid] = value
        return value

    def block(self, event) -> np.ndarray:
        gyro, timestamps, valid, rate = self.session(event.sid)
        centre = (event.start_ms + event.end_ms) // 2
        mask = (timestamps >= centre - 7_500) & (timestamps <= centre + 7_500) & valid
        return gyro5_feature_block(gyro[:, mask], rate)


def meta_block(evidence, lr: np.ndarray, lgbm: np.ndarray) -> np.ndarray:
    durations, macro, micro, ratios = [], [], [], []
    for candidate in evidence.candidates:
        durations.append((candidate.event.end_ms - candidate.event.start_ms) / 1000.0)
        has_macro, has_micro = candidate.macro is not None, candidate.micro is not None
        macro.append(float(has_macro)); micro.append(float(has_micro))
        macro_score = max(candidate.macro.probabilities, default=0.0) if has_macro else 0.0
        micro_score = max(candidate.micro.probabilities, default=0.0) if has_micro else 0.0
        ratios.append((macro_score + 1e-6) / (micro_score + 1e-6))
    return adaptive_meta_features(lr, lgbm, duration_seconds=np.asarray(durations),
                                  has_macro=np.asarray(macro), has_micro=np.asarray(micro),
                                  macro_micro_ratio=np.asarray(ratios))


def _final_indices(evidence, scores, admission, policy):
    rows = np.asarray(rr.admit_candidates(evidence.candidates, scores, evidence.groups, admission), dtype=np.int64)
    return rows[scores[rows] >= policy.threshold]


def select_best(data, config, evidence, scored: list[tuple[str, np.ndarray]]):
    best = None
    admissions = [rr.CandidateAdmissionConfig(n, threshold, cap)
                  for n in config.admission_nms_iou_grid
                  for threshold in config.admission_threshold_grid
                  for cap in config.admission_subject_cap_grid]
    for score_rank, (name, scores) in enumerate(scored):
        for admission_rank, admission in enumerate(admissions):
            rows = np.asarray(rr.admit_candidates(evidence.candidates, scores, evidence.groups, admission), dtype=np.int64)
            events = [evidence.candidates[i].event for i in rows]
            admission_metrics = rr.compute_event_metrics(events, data.train_truths)
            selected = rr.select_event_threshold(events, scores[rows], data.train_truths)
            policy = rr.EventSelectionPolicy(selected.threshold, None, selected.metrics)
            rank = (policy.metrics.f1, policy.metrics.ppv, -len(rows), -score_rank,
                    -admission_rank, -policy.threshold)
            candidate = (admission_metrics.sensitivity >= config.admission_minimum_recall, rank,
                         name, scores, admission, policy)
            if best is None or ((candidate[0], *candidate[1]) > (best[0], *best[1])):
                best = candidate
    assert best is not None
    _, _, name, scores, admission, policy = best
    indices = _final_indices(evidence, scores, admission, policy)
    return {"name": name, "scores": scores, "admission": admission, "policy": policy,
            "indices": indices, "metrics": policy.metrics}


def accounting(evidence, truths, control, challenger):
    def state(arm):
        events = [evidence.candidates[i].event for i in arm["indices"]]
        pairs = official_match_indices(events, truths)
        truth_keys = {(truths[p.truth_index].sid, truths[p.truth_index].start_ms, truths[p.truth_index].end_ms) for p in pairs}
        matched_pred = {p.prediction_index for p in pairs}
        fp_keys = {(e.sid, e.start_ms, e.end_ms) for i, e in enumerate(events) if i not in matched_pred}
        short = {k for k in truth_keys if (k[2] - k[1]) < 600_000}
        return truth_keys, fp_keys, short
    ct, cf, cs = state(control); nt, nf, ns = state(challenger)
    return {"tp_retained": len(ct & nt), "tp_recovered": len(nt - ct), "tp_lost": len(ct - nt),
            "fp_retained": len(cf & nf), "fp_removed": len(cf - nf), "fp_introduced": len(nf - cf),
            "short_tp_recovered": len(ns - cs), "short_tp_lost": len(cs - ns),
            "control_truth_keys": len(ct), "challenger_truth_keys": len(nt)}


def pooled(rows):
    tp = sum(x["n_tp"] for x in rows); pred = sum(x["n_pred"] for x in rows); true = sum(x["n_true"] for x in rows)
    return {"f1": 2 * tp / (pred + true), "precision": tp / pred, "recall": tp / true,
            "n_tp": tp, "n_pred": pred, "n_true": true, "fp": pred - tp, "fn": true - tp}


def prereg(root: Path) -> None:
    sha = git_sha(root)
    e05 = root / "outputs/challenger_70/experiments/E05-001"
    e06 = root / "outputs/challenger_70/experiments/E06-001"
    if e05.exists() or e06.exists():
        raise RuntimeError("E05-001/E06-001 IDs already exist")
    common = {"code_sha": sha, "base_run": "160afaf81debf1ee", "phase": "inner-only",
              "outer_evaluation_authorized": False, "inner_folds": 4, "seeds": [42],
              "candidate_population": "incumbent raw macro/micro union", "matcher": "official IoU>=0.25",
              "gate": {"strong": 0.780, "review": [0.772, 0.780], "majority_folds_required": True},
              "abort_conditions": ["outer access", "candidate identity change", "schema drift", "non-OOF score"]}
    dump(e05 / "plan.json", {**common, "experiment_id": "E05-001",
        "hypothesis": "centred 5 s GYRO information improves event TP/FP trade-off",
        "feature_list": list(GYRO5_FEATURE_NAMES), "feature_definitions": "see committed tier1_challenger.py",
        "window_definition": "15 s cached raw GYRO around candidate midpoint; centred 5 s subwindow",
        "aggregation_definition": "six deterministic scalar proxies", "normalization": "LR fold-contained median imputer + StandardScaler; LGBM fold-contained imputer",
        "lr_grid": [0.1], "lgbm_settings": "incumbent constrained settings", "fusion_procedure": "incumbent weights grid",
        "threshold_procedure": "incumbent inner event threshold/admission grid"})
    dump(e05 / "feature_schema.json", {"version": "gyro5-v1", "names": list(GYRO5_FEATURE_NAMES), "count": len(GYRO5_FEATURE_NAMES)})
    dump(e06 / "plan.json", {**common, "experiment_id": "E06-001",
        "hypothesis": "candidate-dependent LR/LGBM complementarity improves fixed scalar blend",
        "feature_list": list(META_FEATURE_NAMES), "base_probabilities": "subject-OOF only",
        "controls": ["release-fold scalar", "inner-retuned scalar"], "model": "LogisticRegression",
        "lr_grid": [0.1, 1.0, 10.0], "normalization": "fold-contained median imputer + StandardScaler",
        "threshold_procedure": "incumbent inner event threshold/admission grid"})
    dump(e06 / "meta_feature_schema.json", {"version": "adaptive-fusion-v1", "names": list(META_FEATURE_NAMES), "count": len(META_FEATURE_NAMES)})


def execute(root: Path, workers: int) -> dict:
    before = protected_snapshot(root)
    store = GyroStore(root)
    e05_folds, e06_folds, e05_diff, e06_diff = [], [], [], []
    started = time.perf_counter(); extraction_seconds = training_seconds = 0.0
    for fold in range(5):
        config = registered_e03_config(fold, workers=workers)
        data = rr.FilesystemDataSource(root).load_outer_train_only(config)
        evidence = _candidate_evidence(data, config)
        t0 = time.perf_counter()
        gyro = np.vstack([store.block(c.event) for c in evidence.candidates])
        extraction_seconds += time.perf_counter() - t0
        if gyro.shape != (len(evidence.candidates), 6): raise RuntimeError("GYRO5 schema drift")
        t0 = time.perf_counter()
        base_lr = crossfit_predict_proba(evidence.features, evidence.labels, evidence.groups, evidence.features, evidence.groups,
            config.inner_splits, estimator_factory=lambda: rr._verifier_estimator(config.seed + 20, .1)).probabilities
        base_lgbm = crossfit_predict_proba(evidence.features, evidence.labels, evidence.groups, evidence.features, evidence.groups,
            config.inner_splits, estimator_factory=lambda: rr._verifier_lgbm_estimator(config.seed + 21)).probabilities
        control_scored = [(f"blend_{w}", w * base_lr + (1-w) * base_lgbm) for w in config.verifier_blend_weight_grid]
        control = select_best(data, config, evidence, control_scored)
        augmented = np.column_stack([evidence.features, gyro])
        e05_lr = crossfit_predict_proba(augmented, evidence.labels, evidence.groups, augmented, evidence.groups,
            config.inner_splits, estimator_factory=lambda: rr._verifier_estimator(config.seed + 20, .1)).probabilities
        e05_lgbm = crossfit_predict_proba(augmented, evidence.labels, evidence.groups, augmented, evidence.groups,
            config.inner_splits, estimator_factory=lambda: rr._verifier_lgbm_estimator(config.seed + 21)).probabilities
        e05 = select_best(data, config, evidence, [(f"blend_{w}", w*e05_lr+(1-w)*e05_lgbm) for w in config.verifier_blend_weight_grid])
        fixed_w = FIXED_RELEASE_WEIGHTS[fold]
        fixed = select_best(data, config, evidence, [(f"fixed_{fixed_w}", fixed_w*base_lr+(1-fixed_w)*base_lgbm)])
        meta = meta_block(evidence, base_lr, base_lgbm)
        adaptive_scores = []
        for c in (0.1, 1.0, 10.0):
            scores = crossfit_predict_proba(meta, evidence.labels, evidence.groups, meta, evidence.groups, config.inner_splits,
                estimator_factory=lambda c=c: rr._verifier_estimator(config.seed + 30, c)).probabilities
            adaptive_scores.append((f"meta_C_{c}", scores))
        adaptive = select_best(data, config, evidence, adaptive_scores)
        training_seconds += time.perf_counter() - t0
        cm, em, fm, am = map(lambda arm: metrics_dict(arm["metrics"]), (control, e05, fixed, adaptive))
        e05_folds.append({"fold": fold, "control": cm, "challenger": em, "config": e05["name"]})
        e06_folds.append({"fold": fold, "control_a_fixed": fm, "control_b_retuned": cm, "adaptive": am, "config": adaptive["name"]})
        e05_diff.append({"fold": fold, **accounting(evidence, data.train_truths, control, e05)})
        e06_diff.append({"fold": fold, **accounting(evidence, data.train_truths, control, adaptive)})
    e05_control = pooled([x["control"] for x in e05_folds]); e05_result = pooled([x["challenger"] for x in e05_folds])
    e06_a = pooled([x["control_a_fixed"] for x in e06_folds]); e06_b = pooled([x["control_b_retuned"] for x in e06_folds]); e06_result = pooled([x["adaptive"] for x in e06_folds])
    e05_majority = sum(x["challenger"]["f1"] > x["control"]["f1"] for x in e05_folds) >= 3
    e06_majority = sum(x["adaptive"]["f1"] > x["control_b_retuned"]["f1"] for x in e06_folds) >= 3
    e05_gate = classify_inner_gate(e05_result["f1"], majority_improved=e05_majority)
    e06_gate = classify_inner_gate(e06_result["f1"], majority_improved=e06_majority)
    if e06_result["f1"] <= e06_b["f1"] + 1e-12: e06_gate = "FAILED"
    runtime = {"total_seconds": time.perf_counter()-started, "extraction_seconds": extraction_seconds,
               "training_seconds": training_seconds, "cache_hits": store.hits, "cache_misses": store.misses,
               "cache_hit_rate": store.hits/max(1,store.hits+store.misses), "feature_count_e05": 6, "feature_count_e06": 8}
    assert_snapshot_unchanged(before, root)
    return {"e05": (e05_folds,e05_control,e05_result,e05_diff,e05_gate),
            "e06": (e06_folds,e06_a,e06_b,e06_result,e06_diff,e06_gate), "runtime": runtime}


def write_results(root: Path, result: dict) -> None:
    sha = git_sha(root); runtime = result["runtime"]
    e05_dir=root/"outputs/challenger_70/experiments/E05-001"; e06_dir=root/"outputs/challenger_70/experiments/E06-001"
    folds, control, challenger, diffs, gate = result["e05"]
    total_diff={k:sum(x[k] for x in diffs) for k in diffs[0] if k!='fold'}
    dump(e05_dir/"fold_metrics.json",folds); dump(e05_dir/"aggregate_metrics.json",{"control":control,"challenger":challenger})
    dump(e05_dir/"event_diff.json",{"folds":diffs,"aggregate":total_diff}); dump(e05_dir/"short_meal_effect.json",{k:total_diff[k] for k in ["short_tp_recovered","short_tp_lost"]})
    dump(e05_dir/"fp_mode_effect.json",{"status":"not_estimable_inner_only","reason":"predefined sealed-outer FP modes have no leakage-safe inner identity mapping; event-level FP accounting retained"})
    status={"FAILED":"E05_INNER_GATE_FAILED","REVIEW_REQUIRED":"E05_REVIEW_REQUIRED","STRONG_PASS":"E05_READY_FOR_HUMAN_OUTER_REVIEW"}[gate]
    dump(e05_dir/"inner_gate.json",{"status":status,"classification":gate,"majority_folds_improved":sum(x['challenger']['f1']>x['control']['f1'] for x in folds),"outer_run":False})
    dump(e05_dir/"leakage_audit.json",{"outer_validation_accessible":False,"subject_oof":True,"fold_contained_normalization":True,"candidate_population_frozen":True})
    dump(e05_dir/"runtime.json",runtime); dump(e05_dir/"manifest.json",{"experiment_id":"E05-001","code_sha":sha,"feature_count":6,"status":status,"outer_run":False})
    (e05_dir/"report.md").write_text(f"# E05-001 GYRO5 inner gate\n\nStatus: `{status}`\n\nControl F1: `{control['f1']:.10f}`; challenger F1: `{challenger['f1']:.10f}`. No outer evaluation was run.\n",encoding='utf8')
    folds,a,b,challenger,diffs,gate=result['e06']; total_diff={k:sum(x[k] for x in diffs) for k in diffs[0] if k!='fold'}
    status="FUSION_READY_FOR_HUMAN_OUTER_REVIEW" if gate=="STRONG_PASS" else ("E06_REVIEW_REQUIRED" if gate=="REVIEW_REQUIRED" else "ADAPTIVE_FUSION_NOT_JUSTIFIED")
    dump(e06_dir/"fold_metrics.json",folds); dump(e06_dir/"control_comparison.json",{"control_a_fixed":a,"control_b_retuned":b,"adaptive":challenger})
    dump(e06_dir/"inner_gate.json",{"status":status,"classification":gate,"majority_folds_improved":sum(x['adaptive']['f1']>x['control_b_retuned']['f1'] for x in folds),"outer_run":False})
    dump(e06_dir/"event_diff.json",{"folds":diffs,"aggregate":total_diff}); dump(e06_dir/"disagreement_effect.json",{"diagnostic":"adaptive meta-model includes signed/absolute disagreement; no GT subgroup retuning performed"})
    dump(e06_dir/"leakage_audit.json",{"outer_validation_accessible":False,"base_predictions_subject_oof":True,"meta_predictions_subject_oof":True,"fold_contained_normalization":True})
    dump(e06_dir/"runtime.json",runtime); dump(e06_dir/"manifest.json",{"experiment_id":"E06-001","code_sha":sha,"feature_count":8,"status":status,"outer_run":False})
    (e06_dir/"report.md").write_text(f"# E06-001 adaptive fusion inner gate\n\nStatus: `{status}`\n\nFixed scalar F1: `{a['f1']:.10f}`; retuned scalar F1: `{b['f1']:.10f}`; adaptive F1: `{challenger['f1']:.10f}`. No outer evaluation was run.\n",encoding='utf8')
    registry=root/'outputs/challenger_70/experiment_registry.jsonl'
    with registry.open('a',encoding='utf8',newline='\n') as f:
        f.write(json.dumps({"experiment_id":"E05-001","status":result['e05'][-1],"inner_control_f1":control['f1'],"inner_challenger_f1":result['e05'][2]['f1'],"code_sha":sha},sort_keys=True)+'\n')
        f.write(json.dumps({"experiment_id":"E06-001","status":result['e06'][-1],"inner_control_f1":b['f1'],"inner_challenger_f1":result['e06'][3]['f1'],"code_sha":sha},sort_keys=True)+'\n')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--phase',choices=['prereg','run'],required=True); p.add_argument('--workers',type=int,default=5); a=p.parse_args(); root=a.root.resolve()
    if a.phase=='prereg': prereg(root); return
    write_results(root,execute(root,a.workers))


if __name__=='__main__': main()
