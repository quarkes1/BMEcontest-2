#!/usr/bin/env python
"""Recover immutable E05 row evidence and perform the diagnostic-only churn audit."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.e05_churn_audit import residual_support_gate  # noqa: E402
from scripts.research.run_tier1_challengers import GyroStore, select_best  # noqa: E402
from src.pipeline import runner as rr  # noqa: E402
from src.pipeline.challenger.e03 import _candidate_evidence, registered_e03_config  # noqa: E402
from src.pipeline.challenger.protection import assert_snapshot_unchanged, protected_snapshot  # noqa: E402
from src.pipeline.challenger.replay import official_match_indices  # noqa: E402
from src.pipeline.crossfit import crossfit_predict_proba  # noqa: E402

EXPECTED = {"TP_RECOVERED": 11, "TP_LOST": 9, "FP_REMOVED": 16, "FP_INTRODUCED": 20}


def dump(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def arm_state(evidence, truths, arm):
    candidate_indices = list(map(int, arm["indices"]))
    events = [evidence.candidates[i].event for i in candidate_indices]
    pairs = official_match_indices(events, truths)
    by_truth = {p.truth_index: candidate_indices[p.prediction_index] for p in pairs}
    matched_positions = {p.prediction_index for p in pairs}
    fps = {candidate_indices[i] for i in range(len(events)) if i not in matched_positions}
    return by_truth, fps


def evidence_row(category, fold, index, evidence, gyro, control, lr, lgbm, truth=None):
    candidate = evidence.candidates[index]
    macro = candidate.macro is not None; micro = candidate.micro is not None
    macro_score = max(candidate.macro.probabilities, default=0.0) if macro else 0.0
    micro_score = max(candidate.micro.probabilities, default=0.0) if micro else 0.0
    score = float(control["scores"][index]); threshold = float(control["policy"].threshold)
    return {"category": category, "fold": fold, "candidate_index": index,
        "session_id": candidate.event.sid, "start_ms": candidate.event.start_ms, "end_ms": candidate.event.end_ms,
        "truth_start_ms": "" if truth is None else truth.start_ms, "truth_end_ms": "" if truth is None else truth.end_ms,
        "gyro5_variance": float(gyro[index,0]), "gyro5_rms": float(gyro[index,1]),
        "gyro5_energy": float(gyro[index,2]), "gyro5_max": float(gyro[index,3]),
        "gyro5_local_concentration": float(gyro[index,4]), "gyro5_to_gyro15_variance_ratio": float(gyro[index,5]),
        "incumbent_score": score, "incumbent_threshold": threshold, "incumbent_margin": score-threshold,
        "lr_score": float(lr[index]), "lgbm_score": float(lgbm[index]),
        "duration_s": (candidate.event.end_ms-candidate.event.start_ms)/1000,
        "has_macro": int(macro), "has_micro": int(micro), "macro_evidence": macro_score,
        "micro_evidence": micro_score, "macro_micro_ratio": (macro_score+1e-6)/(micro_score+1e-6)}


def summarize(rows, names):
    out={}
    for category in EXPECTED:
        part=[r for r in rows if r['category']==category]
        out[category]={n:{"median":float(np.median([r[n] for r in part])),"q25":float(np.quantile([r[n] for r in part],.25)),"q75":float(np.quantile([r[n] for r in part],.75)),"n":len(part)} for n in names}
    return out


def main():
    root=ROOT; output=root/'outputs/challenger_70/e05_1_churn_audit'
    if output.exists(): raise RuntimeError('E05.1 output already exists')
    before=protected_snapshot(root); store=GyroStore(root); rows=[]; started=time.perf_counter()
    for fold in range(5):
        config=registered_e03_config(fold,workers=5); data=rr.FilesystemDataSource(root).load_outer_train_only(config); evidence=_candidate_evidence(data,config)
        gyro=np.vstack([store.block(c.event) for c in evidence.candidates])
        lr=crossfit_predict_proba(evidence.features,evidence.labels,evidence.groups,evidence.features,evidence.groups,config.inner_splits,estimator_factory=lambda:rr._verifier_estimator(config.seed+20,.1)).probabilities
        lgbm=crossfit_predict_proba(evidence.features,evidence.labels,evidence.groups,evidence.features,evidence.groups,config.inner_splits,estimator_factory=lambda:rr._verifier_lgbm_estimator(config.seed+21)).probabilities
        control=select_best(data,config,evidence,[(f'blend_{w}',w*lr+(1-w)*lgbm) for w in config.verifier_blend_weight_grid])
        augmented=np.column_stack([evidence.features,gyro])
        glr=crossfit_predict_proba(augmented,evidence.labels,evidence.groups,augmented,evidence.groups,config.inner_splits,estimator_factory=lambda:rr._verifier_estimator(config.seed+20,.1)).probabilities
        glgbm=crossfit_predict_proba(augmented,evidence.labels,evidence.groups,augmented,evidence.groups,config.inner_splits,estimator_factory=lambda:rr._verifier_lgbm_estimator(config.seed+21)).probabilities
        challenger=select_best(data,config,evidence,[(f'blend_{w}',w*glr+(1-w)*glgbm) for w in config.verifier_blend_weight_grid])
        ct,cf=arm_state(evidence,data.train_truths,control); nt,nf=arm_state(evidence,data.train_truths,challenger)
        for truth_index in sorted(set(nt)-set(ct)):
            idx=nt[truth_index]; rows.append(evidence_row('TP_RECOVERED',fold,idx,evidence,gyro,control,lr,lgbm,data.train_truths[truth_index]))
        for truth_index in sorted(set(ct)-set(nt)):
            idx=ct[truth_index]; rows.append(evidence_row('TP_LOST',fold,idx,evidence,gyro,control,lr,lgbm,data.train_truths[truth_index]))
        for idx in sorted(cf-nf): rows.append(evidence_row('FP_REMOVED',fold,idx,evidence,gyro,control,lr,lgbm))
        for idx in sorted(nf-cf): rows.append(evidence_row('FP_INTRODUCED',fold,idx,evidence,gyro,control,lr,lgbm))
    counts=Counter(r['category'] for r in rows)
    if dict(counts)!=EXPECTED:
        dump(output/'manifest.json',{"status":"INVALID_EXPERIMENT","reason":"replay identity counts differ","observed":dict(counts),"expected":EXPECTED})
        raise RuntimeError(f'E05 replay mismatch: {counts}')
    output.mkdir(parents=True)
    with (output/'churn_population.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    margin=summarize(rows,['incumbent_score','incumbent_threshold','incumbent_margin','lr_score','lgbm_score'])
    gyro_summary=summarize(rows,['gyro5_variance','gyro5_rms','gyro5_energy','gyro5_to_gyro15_variance_ratio','gyro5_local_concentration'])
    dump(output/'margin_summary.json',margin); dump(output/'gyro5_summary.json',gyro_summary)
    fold_rows={}
    consistent=0
    for fold in range(5):
        part=[r for r in rows if r['fold']==fold]; rec=[r['gyro5_variance'] for r in part if r['category']=='TP_RECOVERED']; intro=[r['gyro5_variance'] for r in part if r['category']=='FP_INTRODUCED']
        direction=None if not rec or not intro else float(np.median(rec)-np.median(intro)); consistent += int(direction is not None and direction>0)
        fold_rows[str(fold)]={"tp_recovered_n":len(rec),"fp_introduced_n":len(intro),"gyro_variance_median_difference":direction}
    h1=margin['TP_RECOVERED']['incumbent_margin']['median'] < margin['TP_LOST']['incumbent_margin']['median']
    h2=(gyro_summary['TP_RECOVERED']['gyro5_variance']['median'] > gyro_summary['FP_INTRODUCED']['gyro5_variance']['median'] and gyro_summary['FP_REMOVED']['gyro5_variance']['median'] < gyro_summary['TP_RECOVERED']['gyro5_variance']['median'])
    status=residual_support_gate(h1=h1,h2=h2,consistent_folds=consistent)
    dump(output/'fold_direction.json',{"folds":fold_rows,"consistent_folds":consistent})
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dump(output/'manifest.json',{"status":status,"diagnostic_only":True,"new_model_trained":False,"outer_f1_computed":False,"recovery_replay":True,"source_commit":sha,"counts":dict(counts),"h1":h1,"h2":h2,"consistent_folds":consistent,"runtime_seconds":time.perf_counter()-started,"cache_hit_rate":store.hits/max(1,store.hits+store.misses)})
    report=["# E05.1 Decision Churn Audit","",f"Final status: `{status}`","",f"Exact replay counts: `{dict(counts)}`.","",f"H1 boundary interaction: **{h1}**. TP-recovered median margin `{margin['TP_RECOVERED']['incumbent_margin']['median']:.6f}` vs TP-lost `{margin['TP_LOST']['incumbent_margin']['median']:.6f}`.",f"H2 monotonic GYRO: **{h2}**. TP-recovered median variance `{gyro_summary['TP_RECOVERED']['gyro5_variance']['median']:.3f}` vs FP-introduced `{gyro_summary['FP_INTRODUCED']['gyro5_variance']['median']:.3f}`.",f"Fold direction consistency: `{consistent}/5`.","",f"Frozen residual supported: **{status == 'GYRO_RESIDUAL_SUPPORTED'}**.","Uncertainty gating is supported only when H1 and the full residual gate pass; no threshold was searched.","No new model, threshold, policy, outer F1, release, or dist artifact was created."]
    (output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    assert_snapshot_unchanged(before,root)
    print(json.dumps({"status":status,"counts":dict(counts),"consistent_folds":consistent}))


if __name__=='__main__': main()
