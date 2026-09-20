"""Preregistered E03 v1 hard-negative inner gate.

This module is deliberately limited to outer-train-only development evidence.
It has no outer-fold execution, release, or distribution entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from src.pipeline.challenger.hard_negative import (
    HardNegativeEvidence, assert_outer_subject_disjoint, hard_negative_spec_descriptor,
    mine_hard_negatives,
)
from src.pipeline.challenger.protection import assert_snapshot_unchanged, protected_snapshot
from src.pipeline.context_features import CONTEXT_V1_COLUMNS
from src.pipeline.crossfit import crossfit_predict_proba
from src.pipeline.event_stack import EventMetrics
from src.pipeline import runner as _runner
from src.pipeline.runner import FilesystemDataSource, InnerOOFResult, OuterTrainOnlyData, RunConfig, run_outer_train_inner_oof


E03_ID = "E03-001"
E03_NAMESPACE = "e03-v1-hard-negative"
E03_BASE_RUN = "160afaf81debf1ee"
E03_NUMPY_VERSION = "2.4.6"


def registered_e03_config(outer_fold: int, *, workers: int) -> RunConfig:
    """Return E03's sole v1 control surface."""
    if outer_fold not in range(5) or workers < 1:
        raise ValueError("E03 requires outer fold 0..4 and positive workers")
    # This is the exact registered v1 Context-v2-control surface.
    from src.pipeline.challenger.context_v2 import registered_e02_config
    return replace(registered_e02_config(outer_fold, workers=workers), context_features_version="v1")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_sha(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _require_e03_environment(root: Path, *, code_sha: str | None = None) -> str:
    if np.__version__ != E03_NUMPY_VERSION:
        raise RuntimeError(f"E03 requires NumPy {E03_NUMPY_VERSION}, found {np.__version__}")
    incumbent = json.loads((Path(root) / "release" / "event_stack_incumbent.json").read_text(encoding="utf-8"))
    if incumbent.get("run_key") != E03_BASE_RUN or incumbent.get("f1") != 0.6514285714285715:
        raise RuntimeError("protected incumbent registry does not identify E03 baseline")
    protected_snapshot(Path(root))
    return code_sha or _git_sha(Path(root))


def _validate_output(root: Path, output: Path) -> Path:
    expected = Path(root).resolve() / "outputs" / "challenger_70" / "experiments" / E03_ID
    if Path(output).resolve(strict=False) != expected:
        raise ValueError("E03 output must be exactly outputs/challenger_70/experiments/E03-001")
    return expected


def _mining_hash() -> str:
    return hashlib.sha256(json.dumps(hard_negative_spec_descriptor(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _plan_payload(code_sha: str) -> dict[str, object]:
    config = registered_e03_config(0, workers=5)
    return {
        "experiment_id": E03_ID,
        "experiment_slug": E03_NAMESPACE,
        "derived_from": "E02-005",
        "derivation_rationale": "E02-005 failed its TP-loss gate; E03 isolates v1 weighting rather than combining a failed v2 representation.",
        "hypothesis": "Inner-OOF weighted v1 hard negatives reduce episode false positives beyond a same-grid threshold-only control.",
        "base_run": E03_BASE_RUN, "code_sha": code_sha, "numpy_version": E03_NUMPY_VERSION,
        "feature_version": "v1", "context_v1_schema_hash": hashlib.sha256("\n".join(CONTEXT_V1_COLUMNS).encode()).hexdigest(),
        "mining_spec": hard_negative_spec_descriptor(), "mining_spec_sha256": _mining_hash(),
        "allowed_hyperparameter_grid": {**asdict(config), "hard_negative_weight": [2.0]},
        "primary_metric": "aggregate_episode_f1", "secondary_metrics": ["tp", "fp", "short_recall", "fold_f1"],
        "selection_protocol": "outer-train inner-OOF only",
        "candidate_order_contract": "canonical raw-union order is retained through mining and fitting",
        "imputer_contract": "incumbent median imputation; no learned transform outside each fit",
        "policy_tie_order_contract": "incumbent runner tuple order",
        "development_evidence_warning": "Inner gate is development evidence only, not independent validation.",
    }


def write_e03_preregistration(root: Path, output: Path, *, code_sha: str | None = None) -> Path:
    """Atomically write E03's immutable preregistration before a gate run."""
    root, output = Path(root).resolve(), _validate_output(root, output)
    bound = _require_e03_environment(root, code_sha=code_sha)
    payload = _plan_payload(bound)
    plan = output / "plan.json"
    if plan.exists():
        if json.loads(plan.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("E03 preregistration is immutable")
        return plan
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("E03 preregistration output must be empty")
    _atomic_json(plan, payload)
    return plan


def _metrics_dict(value: EventMetrics) -> dict[str, int | float]:
    return {"n_tp": int(value.n_tp), "n_pred": int(value.n_pred), "n_true": int(value.n_true), "f1": float(value.f1)}


def e03_gate_passes(control: Sequence[EventMetrics], hard: Sequence[EventMetrics]) -> bool:
    """Apply the fixed four-predicate E03 gate, with no tolerance."""
    if len(control) != 5 or len(hard) != 5:
        return False
    if any(not np.isfinite(item.f1) for item in (*control, *hard)):
        return False
    c_tp, h_tp = sum(x.n_tp for x in control), sum(x.n_tp for x in hard)
    c_pred, h_pred = sum(x.n_pred for x in control), sum(x.n_pred for x in hard)
    c_true, h_true = sum(x.n_true for x in control), sum(x.n_true for x in hard)
    c_f1 = 2 * c_tp / (c_true + c_pred) if c_true + c_pred else 0.0
    h_f1 = 2 * h_tp / (h_true + h_pred) if h_true + h_pred else 0.0
    return h_f1 > c_f1 and (c_pred - c_tp) - (h_pred - h_tp) >= 4 and c_tp - h_tp <= 2 and sum(h.f1 >= c.f1 for c, h in zip(control, hard, strict=True)) >= 3


@dataclass(frozen=True)
class _CandidateEvidence:
    candidates: tuple[object, ...]
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray


def _candidate_evidence(data: OuterTrainOnlyData, config: RunConfig) -> _CandidateEvidence:
    """Rebuild frozen raw-union v1 candidates using outer-train data only."""
    _runner._validate_batch(data.window_train, "window_train")
    _runner._validate_batch(data.candidate_train, "candidate_train")
    window_groups = _runner._groups_for(data.window_train.windows, data.subject_by_session)
    candidate_groups = _runner._groups_for(data.candidate_train.windows, data.subject_by_session)
    keep = np.asarray(data.window_train.labels) >= 0
    labels = np.asarray(data.window_train.labels[keep], dtype=np.int8)
    _runner._validate_binary(labels, "window training")
    windows = crossfit_predict_proba(
        _runner._with_time_prior(data.window_train.features, data.window_train.windows)[keep], labels, window_groups[keep],
        _runner._with_time_prior(data.candidate_train.features, data.candidate_train.windows), candidate_groups,
        config.inner_splits, estimator_factory=lambda: _runner._window_estimator(config.seed),
    )
    macro_windows = _runner._windows_by_session(data.candidate_train.windows, windows.probabilities)
    raw_macro = _runner.density_candidates(macro_windows, config.density)
    if not config.micro_enabled or data.micro_window_train is None or data.micro_candidate_train is None:
        raise ValueError("E03 requires the incumbent micro path")
    micro_groups = _runner._groups_for(data.micro_window_train.windows, data.subject_by_session)
    micro_candidate_groups = _runner._groups_for(data.micro_candidate_train.windows, data.subject_by_session)
    micro_keep = _runner._micro_training_keep(data.micro_window_train, data.train_truths, config.micro_positive_middle_fraction)
    micro_labels = np.asarray(data.micro_window_train.labels[micro_keep], dtype=np.int8)
    _runner._validate_binary(micro_labels, "micro window training")
    micro = crossfit_predict_proba(data.micro_window_train.features[micro_keep], micro_labels, micro_groups[micro_keep],
        data.micro_candidate_train.features, micro_candidate_groups, config.inner_splits,
        estimator_factory=lambda: _runner._micro_window_estimator(config.seed + 10))
    micro_windows = _runner._windows_by_session(data.micro_candidate_train.windows, micro.probabilities)
    micro_selection = _runner.select_micro_candidate_threshold(micro_windows, data.train_truths, config.micro_threshold_grid, config.micro_candidate)
    candidates = tuple(_runner.union_candidates(raw_macro, _runner.micro_candidates(micro_windows, micro_selection.threshold, config.micro_candidate)))
    features = _runner.multiscale_verifier_features(candidates, macro_windows, micro_windows, context_features_version="v1",
        session_bounds_by_sid=_runner._context_session_bounds(data, candidates, data.candidate_train.windows, data.micro_candidate_train.windows))
    if features.shape != (len(candidates), 116):
        raise RuntimeError("E03 requires frozen 116D v1 verifier features")
    candidate_labels = _runner._candidate_labels(candidates, data.train_truths)
    groups = _runner._groups_for([candidate.event for candidate in candidates], data.subject_by_session)
    return _CandidateEvidence(candidates, features, candidate_labels, groups)


def build_e03_inner_weights(data: OuterTrainOnlyData, config: RunConfig):
    """Construct weights from logistic-only subject OOF scores, never outer data.

    Candidate reconstruction remains intentionally in the incumbent runner; the
    E03 helper is a narrow guard until the weighted inner evaluator is added.
    """
    if not isinstance(data, OuterTrainOnlyData):
        raise TypeError("E03 inner weights require OuterTrainOnlyData")
    if config.context_features_version != "v1":
        raise ValueError("E03 accepts frozen Context-v1 only")
    evidence = _candidate_evidence(data, config)
    oof = crossfit_predict_proba(evidence.features, evidence.labels, evidence.groups, evidence.features, evidence.groups,
        config.inner_splits, estimator_factory=lambda: _runner._verifier_estimator(config.seed + 20, .1))
    result = mine_hard_negatives(HardNegativeEvidence(evidence.candidates, evidence.features, evidence.labels, oof.probabilities, evidence.groups))
    return evidence, result


def _select_policy(evidence: _CandidateEvidence, scores: np.ndarray, data: OuterTrainOnlyData, config: RunConfig):
    """Use the incumbent candidate-control grid/tie order on inner OOF scores."""
    candidates = evidence.candidates
    choices = []
    admissions = tuple(_runner.CandidateAdmissionConfig(n, threshold, cap)
        for n in config.admission_nms_iou_grid for threshold in config.admission_threshold_grid for cap in config.admission_subject_cap_grid)
    for admission_rank, admission in enumerate(admissions):
        rows = np.asarray(_runner.admit_candidates(candidates, scores, evidence.groups, admission), dtype=np.int64)
        events = [candidates[i].event for i in rows]
        admitted_scores, groups = scores[rows], evidence.groups[rows]
        admission_metrics = _runner.compute_event_metrics(events, data.train_truths)
        policy = _runner.select_event_policy(events, admitted_scores, data.train_truths, groups, max_events_options=config.subject_cap_grid)
        rank = (policy.metrics.f1, policy.metrics.ppv, -len(rows), 0, 0, -admission_rank, -policy.threshold, -float(policy.max_events_per_group or math.inf))
        choices.append((admission_metrics.sensitivity, rank, policy))
    feasible = [choice for choice in choices if choice[0] >= config.admission_minimum_recall]
    return max(feasible or choices, key=lambda x: x[1] if feasible else (x[0], *x[1]))[2]


def _weighted_inner_result(data: OuterTrainOnlyData, config: RunConfig, weights: np.ndarray | None) -> InnerOOFResult:
    evidence = _candidate_evidence(data, config)
    logistic = crossfit_predict_proba(evidence.features, evidence.labels, evidence.groups, evidence.features, evidence.groups, config.inner_splits,
        estimator_factory=lambda: _runner._verifier_estimator(config.seed + 20, .1), fit_sample_weight=weights).probabilities
    lgbm = crossfit_predict_proba(evidence.features, evidence.labels, evidence.groups, evidence.features, evidence.groups, config.inner_splits,
        estimator_factory=lambda: _runner._verifier_lgbm_estimator(config.seed + 21), fit_sample_weight=weights).probabilities
    best = None
    for blend in config.verifier_blend_weight_grid:
        policy = _select_policy(evidence, blend * logistic + (1 - blend) * lgbm, data, config)
        rank = (policy.metrics.f1, policy.metrics.ppv, -blend)
        if best is None or rank > best[0]: best = (rank, policy)
    assert best is not None
    subjects = frozenset(evidence.groups)
    return InnerOOFResult(best[1].metrics, 116, subjects, subjects, subjects)


def run_e03_inner_gate(root: Path, output: Path, *, workers: int, source_factory: Callable[[Path], FilesystemDataSource] = FilesystemDataSource) -> dict[str, object]:
    """Run only the E03 inner gate; formal outer evaluation is intentionally absent."""
    root, output = Path(root).resolve(), _validate_output(root, output)
    code_sha = _require_e03_environment(root)
    write_e03_preregistration(root, output, code_sha=code_sha)
    before = protected_snapshot(root)
    source = source_factory(root)
    folds, control_rows, hard_rows = [], [], []
    for fold in range(5):
        config = registered_e03_config(fold, workers=workers)
        data = source.load_outer_train_only(config)
        if not isinstance(data, OuterTrainOnlyData):
            raise TypeError("E03 source must return OuterTrainOnlyData")
        # The source mapping is outer-train-only, and is the only subject
        # provenance admitted to mining/fitting in this phase.
        assert_outer_subject_disjoint(data.subject_by_session.values(), ())
        evidence, mined = build_e03_inner_weights(data, config)
        control = _weighted_inner_result(data, config, None)
        hard = _weighted_inner_result(data, config, mined.sample_weight)
        control_rows.append(control.inner_metrics); hard_rows.append(hard.inner_metrics)
        folds.append({"outer_fold": fold, "threshold_only": _metrics_dict(control.inner_metrics), "hard_negative_v1": _metrics_dict(hard.inner_metrics),
            "hard_negative": {"hard_count": int(mined.hard_mask.sum()), "category_counts": {key: int(value.sum()) for key, value in mined.category_mask.items()}, "weight_sum": float(mined.sample_weight.sum())}})
    report: dict[str, object] = {"experiment_id": E03_ID, "phase": "inner-gate", "code_sha": code_sha, "development_evidence": True,
        "not_independent_validation": True, "folds": folds,
        "threshold_only": _pooled_metrics(control_rows), "hard_negative_v1": _pooled_metrics(hard_rows)}
    report["passed"] = e03_gate_passes(control_rows, hard_rows)
    assert_snapshot_unchanged(before, root)
    target = output / "inner_gate.json"
    if target.exists(): raise RuntimeError("E03 inner gate is immutable")
    _atomic_json(target, report)
    return report


def _pooled_metrics(values: Sequence[EventMetrics]) -> dict[str, int | float]:
    tp, pred, true = (sum(getattr(v, name) for v in values) for name in ("n_tp", "n_pred", "n_true"))
    return {"n_tp": int(tp), "n_pred": int(pred), "n_true": int(true), "f1": 2 * tp / (true + pred) if true + pred else 0.0}
