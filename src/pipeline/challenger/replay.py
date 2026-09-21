"""Read-only replay of the sealed outer-fold evidence.

This module deliberately invokes only fitted estimators and the same pure
candidate/policy primitives used by the runner.  It neither imports nor calls
``run_outer_fold``: replay must remain incapable of retraining or caching.
It is intentionally frozen to the incumbent Context-v1 ABI; challenger
Context-v2 has no replay/runtime implementation until human promotion review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.eval.metrics import event_iou
from src.pipeline.artifacts import (
    EventStackBundle,
    load_event_stack_bundle,
    verify_promotion_attestation,
)
from src.pipeline.candidate_control import (
    CandidateAdmissionConfig,
    _multiscale_key,
    admit_candidates,
    suppress_overlapping_candidates,
)
from src.pipeline.challenger.protection import validate_fingerprints
from src.pipeline.context_features import CONTEXT_V1_COLUMNS, CONTEXT_V1_SCHEMA_HASH
from src.pipeline.event_stack import (
    CandidateEvent,
    EventMetrics,
    EventRef,
    MultiScaleCandidate,
    apply_event_policy,
    compute_event_metrics,
    density_candidates,
    micro_candidates,
    multiscale_verifier_features,
    union_candidates,
)
from src.pipeline.runner import (
    DensityConfig,
    FilesystemDataSource,
    MicroCandidateConfig,
    RunConfig,
    _groups_for,
    _positive_probability,
    _windows_by_session,
    _with_time_prior,
    cache_key,
    expected_feature_dimensions,
)


_RUN_KEY = "160afaf81debf1ee"
_IOU = 0.25


@dataclass(frozen=True)
class MatchPair:
    prediction_index: int
    truth_index: int
    iou: float


@dataclass(frozen=True)
class ReplayTrace:
    fold: int
    truths: tuple[EventRef, ...]
    truth_slices: Mapping[str, tuple[EventRef, ...]]
    raw_macro: tuple[CandidateEvent, ...]
    raw_micro: tuple[CandidateEvent, ...]
    raw_union: tuple[MultiScaleCandidate, ...]
    candidate_rows: tuple[Mapping[str, object], ...]
    admitted_indices: tuple[int, ...]
    final_indices: tuple[int, ...]
    predictions: tuple[EventRef, ...]
    metrics: EventMetrics
    match_pairs: tuple[MatchPair, ...]
    # Required by the E01 atlas even for C0 truths that have no candidate row.
    subject_by_session: Mapping[str, str] = field(default_factory=dict)
    # Preserves the attested policy verbatim for diagnostics.  A default keeps
    # historical synthetic trace construction source-compatible.
    policy: Mapping[str, object] = field(default_factory=dict)


def official_match_indices(
    predictions: Sequence[EventRef], truths: Sequence[EventRef], iou_threshold: float = _IOU
) -> tuple[MatchPair, ...]:
    """Return the official greedy one-to-one pairing with stable equal-IoU ties."""

    pairs: list[MatchPair] = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            if prediction.sid != truth.sid:
                continue
            iou = event_iou(prediction.interval, truth.interval)
            if iou >= iou_threshold:
                pairs.append(MatchPair(prediction_index, truth_index, float(iou)))
    # Python's stable sort preserves the original prediction/truth Cartesian
    # order, exactly as compute_event_metrics does for score ties.
    pairs.sort(key=lambda pair: -pair.iou)
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    selected: list[MatchPair] = []
    for pair in pairs:
        if pair.prediction_index in used_predictions or pair.truth_index in used_truths:
            continue
        used_predictions.add(pair.prediction_index)
        used_truths.add(pair.truth_index)
        selected.append(pair)
    return tuple(selected)


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} cannot be read") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be an object")
    return payload


def _config(payload: Mapping[str, object]) -> RunConfig:
    values = dict(payload)
    try:
        values["density"] = DensityConfig(**dict(values["density"]))
        values["micro_candidate"] = MicroCandidateConfig(**dict(values["micro_candidate"]))
        for name in (
            "subject_cap_grid", "verifier_c_grid", "micro_threshold_grid",
            "external_fd_weight_grid", "admission_nms_iou_grid",
            "admission_threshold_grid", "admission_subject_cap_grid",
            "verifier_blend_weight_grid",
        ):
            values[name] = tuple(values[name])
        config = RunConfig(**values)
        if config.context_features_version == "v2":
            raise RuntimeError(
                "Context-v2 challenger requires human promotion/runtime implementation"
            )
        return config
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("outer bundle run_config is invalid") from exc


def _summary_fold(summary: Mapping[str, object], fold: int) -> Mapping[str, object]:
    records = summary.get("folds")
    if not isinstance(records, list):
        raise RuntimeError("attested summary has no fold records")
    rows = [row for row in records if isinstance(row, dict) and row.get("outer_fold") == fold]
    if len(rows) != 1 or "micro_threshold" not in rows[0]:
        raise RuntimeError("attested summary is missing selected micro threshold")
    value = rows[0]["micro_threshold"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise RuntimeError("attested summary selected micro threshold is invalid")
    return rows[0]


def validate_summary_run_configs(
    summary: Mapping[str, object], bundle_configs: Mapping[int, Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    """Bind every attested summary configuration to its matching outer bundle.

    ``promotion_summary.json`` is the only attested source of selected micro
    thresholds, so it must not be allowed to carry an unbound parallel config
    list.  The mapping is deliberately complete and exact rather than relying
    on list position.
    """

    rows = summary.get("run_configs")
    if not isinstance(rows, list):
        raise RuntimeError("attested summary has no run_configs")
    parsed: dict[int, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("attested summary run_configs must be objects")
        fold = row.get("outer_fold")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold in parsed:
            raise RuntimeError("attested summary run_configs must be unique folds 0 through 4")
        parsed[fold] = row
    if set(parsed) != set(range(5)) or set(bundle_configs) != set(range(5)):
        raise RuntimeError("attested summary run_configs must be unique folds 0 through 4")
    for fold in range(5):
        if dict(parsed[fold]) != dict(bundle_configs[fold]):
            raise RuntimeError(f"attested summary run_config disagrees with outer bundle fold {fold}")
    return parsed


def _verify_evidence(root: Path, fold: int) -> tuple[EventStackBundle, RunConfig, float]:
    run_root = root / "models" / "event_stack" / _RUN_KEY
    problems = verify_promotion_attestation(run_root, expected_run_key=_RUN_KEY)
    if problems:
        raise RuntimeError("attested challenger evidence is invalid: " + "; ".join(problems))
    summary = _load_json(run_root / "promotion_summary.json", "attested summary")
    if summary.get("experiment_key") != _RUN_KEY:
        raise RuntimeError("attested summary run key disagrees with frozen evidence")
    record = _summary_fold(summary, fold)
    bundle = load_event_stack_bundle(
        run_root / f"outer-fold-{fold}", expected_role="outer-fold-evidence",
        expected_run_key=f"outer-fold-{fold}",
    )
    config_payload = _load_json(run_root / f"outer-fold-{fold}" / "run_config.json", "outer bundle run_config")
    if dict(bundle.run_config) != dict(config_payload):
        raise RuntimeError("outer bundle run_config disagreement")
    config = _config(config_payload)
    # Bind all five summary configs before trusting even this fold's threshold.
    all_bundle_configs = {
        index: _load_json(run_root / f"outer-fold-{index}" / "run_config.json", "outer bundle run_config")
        for index in range(5)
    }
    validate_summary_run_configs(summary, all_bundle_configs)
    if config.outer_fold != fold or not config.micro_enabled or config.context_features_version != "v1":
        raise RuntimeError("outer bundle does not describe the frozen Context-v1 micro replay")
    schema = bundle.feature_schema
    if schema.get("widths") != {"macro": 63, "micro": 47, "verifier": 116}:
        raise RuntimeError("outer bundle feature widths are not frozen 63/47/116")
    context = schema.get("context")
    if not isinstance(context, Mapping) or context.get("schema_hash") != CONTEXT_V1_SCHEMA_HASH:
        raise RuntimeError("outer bundle Context-v1 schema hash is invalid")
    if list(context.get("columns", ())) != list(CONTEXT_V1_COLUMNS):
        raise RuntimeError("outer bundle Context-v1 columns are invalid")
    if record.get("config_hash") is None:
        raise RuntimeError("attested summary fold has no config hash")
    return bundle, config, float(record["micro_threshold"])


def _validate_source_identity(source: FilesystemDataSource, config: RunConfig, bundle: EventStackBundle) -> None:
    expected = tuple(str(path.resolve()) for path in source.input_files(config))
    actual = tuple(str(row["path"]) for row in bundle.source_fingerprints)
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise RuntimeError("outer evidence source fingerprints do not match canonical input path set")
    validate_fingerprints(bundle.source_fingerprints)


def _event_policy_indices(
    candidates: Sequence[MultiScaleCandidate], scores: np.ndarray, groups: Sequence[str],
    threshold: float, cap: int | None,
) -> tuple[int, ...]:
    """Mirror the index set behind apply_event_policy, retaining provenance coordinates."""
    eligible = [index for index, score in enumerate(scores) if score >= threshold]
    if cap is None:
        return tuple(eligible)
    selected: set[int] = set()
    for group in sorted(set(str(groups[index]) for index in eligible)):
        members = [index for index in eligible if str(groups[index]) == group]
        members.sort(key=lambda index: (-scores[index], index))
        selected.update(members[:cap])
    return tuple(index for index in range(len(candidates)) if index in selected)


def _ledger(
    candidates: Sequence[MultiScaleCandidate], raw_macro: Sequence[CandidateEvent], raw_micro: Sequence[CandidateEvent],
    features: np.ndarray, logistic: np.ndarray, lgbm: np.ndarray, scores: np.ndarray,
    groups: Sequence[str], admission: CandidateAdmissionConfig, admitted: Sequence[int],
    threshold: float, event_cap: int | None, final_raw: Sequence[int], subject_by_session: Mapping[str, str],
) -> tuple[Mapping[str, object], ...]:
    admitted_set, final_set = set(admitted), set(final_raw)
    # Recreate canonical NMS/cap orders for annotations.  The actual selected
    # result still comes from admit_candidates.
    rank = sorted(range(len(candidates)), key=lambda i: (-scores[i], _multiscale_key(candidates[i])))
    nms_indices = set(suppress_overlapping_candidates(candidates, scores, admission.nms_iou))
    suppressor: dict[int, int] = {}
    winners: list[int] = []
    for index in rank:
        if index in nms_indices:
            winners.append(index)
            continue
        hit = next(other for other in winners if candidates[index].event.sid == candidates[other].event.sid and event_iou(candidates[index].event.interval, candidates[other].event.interval) >= admission.nms_iou)
        suppressor[index] = hit
    nms_rank = {index: position for position, index in enumerate(rank)}
    candidate_features = {id(candidate): index for index, candidate in enumerate(raw_macro)}
    micro_features = {id(candidate): index for index, candidate in enumerate(raw_micro)}
    admitted_order = {index: position for position, index in enumerate(admitted)}
    subject_cap_rank: dict[int, int] = {}
    cap_candidates = [index for index in nms_indices if scores[index] >= admission.threshold]
    for group in sorted({str(groups[index]) for index in cap_candidates}):
        members = [index for index in cap_candidates if str(groups[index]) == group]
        members.sort(key=lambda index: (-scores[index], _multiscale_key(candidates[index])))
        subject_cap_rank.update({index: position for position, index in enumerate(members)})
    event_rank: dict[int, int] = {}
    if event_cap is not None:
        for group in sorted(set(str(groups[index]) for index in admitted)):
            rows = [index for index in admitted if str(groups[index]) == group and scores[index] >= threshold]
            rows.sort(key=lambda index: (-scores[index], admitted_order[index]))
            event_rank.update({index: position for position, index in enumerate(rows)})
    result: list[Mapping[str, object]] = []
    for index, candidate in enumerate(candidates):
        subject_id = subject_by_session.get(candidate.event.sid)
        if not isinstance(subject_id, str) or not subject_id:
            raise RuntimeError("replay candidate has no nonempty subject provenance")
        context_values = features[index, 56:]
        if index in suppressor:
            reason = "nms_suppressed"
        elif index not in winners:
            raise RuntimeError("candidate NMS accounting is inconsistent")
        elif scores[index] < admission.threshold:
            reason = "below_admission_threshold"
        elif index not in admitted_set:
            reason = "subject_cap_excluded"
        elif scores[index] < threshold:
            reason = "below_event_threshold"
        elif index not in final_set:
            reason = "event_cap_excluded"
        else:
            reason = "selected"
        row: dict[str, object] = {
            "raw_index": index, "subject_id": subject_id, "session_id": candidate.event.sid,
            "start_ms": candidate.event.start_ms, "end_ms": candidate.event.end_ms,
            "macro_raw_index": candidate_features.get(id(candidate.macro)) if candidate.macro else None,
            "micro_raw_index": micro_features.get(id(candidate.micro)) if candidate.micro else None,
            "nms_rank": nms_rank[index], "nms_suppressed_by_raw_index": suppressor.get(index),
            "logistic_score": float(logistic[index]), "lgbm_score": float(lgbm[index]), "blend_score": float(scores[index]),
            "admission_threshold_pass": None if index in suppressor else bool(scores[index] >= admission.threshold),
            "subject_cap_rank": subject_cap_rank.get(index),
            "admitted_index": admitted_order.get(index),
            "event_threshold_pass": bool(scores[index] >= threshold) if index in admitted_set else None,
            "event_cap_rank": event_rank.get(index) if index in admitted_set and scores[index] >= threshold else None,
            "final_event_index": list(final_raw).index(index) if index in final_set else None,
            "terminal_reason": reason,
        }
        row.update({name: float(value) for name, value in zip(CONTEXT_V1_COLUMNS, context_values, strict=True)})
        result.append(row)
    return tuple(result)


def replay_outer_fold(root: Path, fold: int) -> ReplayTrace:
    """Replay one attested outer validation fold using only immutable models/evidence."""
    if fold not in range(5):
        raise ValueError("outer fold must be in range(5)")
    base = Path(root).resolve()
    bundle, config, micro_threshold = _verify_evidence(base, fold)
    source = FilesystemDataSource(base)
    _validate_source_identity(source, config, bundle)
    expected_key = cache_key(config, expected_feature_dimensions(config), source.input_files(config))
    summary = _load_json(base / "models" / "event_stack" / _RUN_KEY / "promotion_summary.json", "attested summary")
    record = _summary_fold(summary, fold)
    if record["config_hash"] != expected_key:
        raise RuntimeError("attested summary config hash does not match current cache-key semantics")
    data = source.load_outer_fold(config)
    if data.micro_validation is None or data.session_bounds_by_sid is None:
        raise RuntimeError("frozen outer validation lacks micro/session data")
    macro_scores = _positive_probability(bundle.models["macro"], _with_time_prior(data.validation.features, data.validation.windows))
    micro_scores = _positive_probability(bundle.models["micro"], data.micro_validation.features)
    macro_windows = _windows_by_session(data.validation.windows, macro_scores)
    micro_windows = _windows_by_session(data.micro_validation.windows, micro_scores)
    raw_macro = tuple(density_candidates(macro_windows, config.density))
    raw_micro = tuple(micro_candidates(micro_windows, micro_threshold, config.micro_candidate))
    raw_union = tuple(union_candidates(raw_macro, raw_micro))
    features = multiscale_verifier_features(raw_union, macro_windows, micro_windows, "v1", data.session_bounds_by_sid)
    if features.shape != (len(raw_union), 116):
        raise RuntimeError("replay verifier feature surface is not 116 columns")
    logistic = _positive_probability(bundle.models["verifier_logistic"], features)
    lgbm = _positive_probability(bundle.models["verifier_lgbm"], features)
    policy = bundle.policy
    try:
        admission = CandidateAdmissionConfig(float(policy["nms_iou"]), float(policy["admission_threshold"]), policy["max_candidates_per_subject"])
        blend = float(policy["blend_weight"])
        threshold = float(policy["threshold"])
        event_cap = policy["max_events_per_group"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("frozen outer policy is invalid") from exc
    scores = blend * logistic + (1.0 - blend) * lgbm
    groups = _groups_for([candidate.event for candidate in raw_union], data.subject_by_session)
    admitted = admit_candidates(raw_union, scores, groups, admission)
    admitted_candidates = [raw_union[index] for index in admitted]
    admitted_scores = scores[np.asarray(admitted, dtype=np.int64)]
    admitted_groups = groups[np.asarray(admitted, dtype=np.int64)]
    # Call the canonical primitive first; index reconstruction only adds audit coordinates.
    predictions = tuple(apply_event_policy([item.event for item in admitted_candidates], admitted_scores, admitted_groups, threshold, event_cap))
    final_in_admitted = _event_policy_indices(admitted_candidates, admitted_scores, admitted_groups, threshold, event_cap)
    final_raw = tuple(admitted[index] for index in final_in_admitted)
    if tuple(raw_union[index].event for index in final_raw) != predictions:
        raise RuntimeError("replay final-event coordinates disagree with canonical event policy")
    metrics = compute_event_metrics(predictions, data.validation_truths, _IOU)
    expected_metrics = bundle.metrics
    if tuple(int(expected_metrics[name]) for name in ("n_tp", "n_true", "n_pred")) != (metrics.n_tp, metrics.n_true, metrics.n_pred):
        raise RuntimeError("replay metrics disagree with frozen outer bundle")
    rows = _ledger(raw_union, raw_macro, raw_micro, features, logistic, lgbm, scores, groups, admission, admitted, threshold, event_cap, final_raw, data.subject_by_session)
    return ReplayTrace(fold, tuple(data.validation_truths), data.validation_truth_slices, raw_macro, raw_micro, raw_union, rows, tuple(admitted), final_raw, predictions, metrics, official_match_indices(predictions, data.validation_truths), dict(data.subject_by_session), dict(policy))
