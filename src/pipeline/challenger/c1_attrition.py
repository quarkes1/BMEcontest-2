"""Pure, GT-marked E03.5 candidate-control attrition diagnostics.

This module consumes a completed :class:`ReplayTrace`; it neither scores nor
fits a model and has no filesystem effects.
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from src.eval.metrics import event_iou
from src.pipeline.candidate_control import CandidateAdmissionConfig, _multiscale_key, admit_candidates, suppress_overlapping_candidates
from src.pipeline.challenger.atlas import classify_false_negatives
from src.pipeline.challenger.replay import ReplayTrace, _event_policy_indices, official_match_indices
from src.pipeline.event_stack import apply_event_policy

_IOU = 0.25
MODES = ("disable_nms_only", "disable_admission_threshold_only", "disable_subject_cap_only")


def _truth_key(trace: ReplayTrace, truth_index: int) -> str:
    truth = trace.truths[truth_index]
    return f"{trace.fold}:{truth.sid}:{truth.start_ms}:{truth.end_ms}"


def _overlap(candidate_index: int, trace: ReplayTrace, truth_index: int) -> float:
    candidate, truth = trace.raw_union[candidate_index].event, trace.truths[truth_index]
    return float(event_iou(candidate.interval, truth.interval)) if candidate.sid == truth.sid else 0.0


def _frozen_policy(trace: ReplayTrace) -> dict[str, object]:
    required = ("nms_iou", "admission_threshold", "max_candidates_per_subject", "threshold", "max_events_per_group")
    if not isinstance(trace.policy, Mapping) or any(name not in trace.policy for name in required):
        raise RuntimeError("E03.5 requires exact attested ReplayTrace policy metadata")
    policy = dict(trace.policy)
    CandidateAdmissionConfig(float(policy["nms_iou"]), float(policy["admission_threshold"]), policy["max_candidates_per_subject"])
    return policy


def _scores(trace: ReplayTrace) -> np.ndarray:
    rows = {int(row["raw_index"]): row for row in trace.candidate_rows}
    if set(rows) != set(range(len(trace.raw_union))):
        raise RuntimeError("candidate ledger does not provide one row per raw union candidate")
    return np.asarray([float(rows[index]["blend_score"]) for index in range(len(rows))], dtype=np.float64)


def _groups(trace: ReplayTrace) -> tuple[str, ...]:
    result = tuple(trace.subject_by_session.get(item.event.sid, "") for item in trace.raw_union)
    if any(not value for value in result):
        raise RuntimeError("replay candidate has no subject provenance")
    return result


def _ordered(trace: ReplayTrace, indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(indices, key=lambda i: (trace.raw_union[i].event.sid, trace.raw_union[i].event.start_ms, trace.raw_union[i].event.end_ms, _multiscale_key(trace.raw_union[i]))))


def _admitted_for_mode(trace: ReplayTrace, mode: str) -> tuple[int, ...]:
    if mode not in MODES:
        raise ValueError("unknown E03.5 counterfactual mode")
    policy, scores, groups = _frozen_policy(trace), _scores(trace), _groups(trace)
    config = CandidateAdmissionConfig(float(policy["nms_iou"]), float(policy["admission_threshold"]), policy["max_candidates_per_subject"])
    nms = tuple(range(len(trace.raw_union))) if mode == "disable_nms_only" else suppress_overlapping_candidates(trace.raw_union, scores, config.nms_iou)
    thresholded = nms if mode == "disable_admission_threshold_only" else tuple(i for i in nms if scores[i] >= config.threshold)
    if mode == "disable_subject_cap_only" or config.max_candidates_per_subject is None:
        return _ordered(trace, thresholded)
    selected: list[int] = []
    for group in sorted({groups[i] for i in thresholded}, key=str):
        ranked = sorted((i for i in thresholded if groups[i] == group), key=lambda i: (-scores[i], _multiscale_key(trace.raw_union[i])))
        selected.extend(ranked[:config.max_candidates_per_subject])
    return _ordered(trace, selected)


def _final_for_indices(trace: ReplayTrace, indices: Sequence[int]) -> tuple[int, ...]:
    policy, scores, groups = _frozen_policy(trace), _scores(trace), _groups(trace)
    local = tuple(indices)
    selected_events = tuple(apply_event_policy([trace.raw_union[i].event for i in local], scores[np.asarray(local, dtype=np.int64)], [groups[i] for i in local], float(policy["threshold"]), policy["max_events_per_group"]))
    final_local = _event_policy_indices([trace.raw_union[i] for i in local], scores[np.asarray(local, dtype=np.int64)], [groups[i] for i in local], float(policy["threshold"]), policy["max_events_per_group"])
    result = tuple(local[i] for i in final_local)
    if tuple(trace.raw_union[i].event for i in result) != selected_events:
        raise RuntimeError("counterfactual event coordinates disagree with canonical event policy")
    return result


def _matches(trace: ReplayTrace, indices: Sequence[int]):
    return official_match_indices([trace.raw_union[i].event for i in indices], trace.truths)


def _unmatched_event_counter(trace: ReplayTrace, indices: Sequence[int]) -> Counter[tuple[str, int, int]]:
    matched = {pair.prediction_index for pair in _matches(trace, indices)}
    return Counter((trace.raw_union[raw].event.sid, trace.raw_union[raw].event.start_ms, trace.raw_union[raw].event.end_ms) for pos, raw in enumerate(indices) if pos not in matched)


def _oracle(trace: ReplayTrace, truth_index: int, baseline: tuple[int, ...]) -> dict[str, object]:
    reachable = [i for i in range(len(trace.raw_union)) if _overlap(i, trace, truth_index) >= _IOU]
    scores = _scores(trace)
    attempts = []
    for index in reachable:
        final = _final_for_indices(trace, _ordered(trace, set(baseline) | {index}))
        pairs = _matches(trace, final)
        success = any(pair.truth_index == truth_index for pair in pairs)
        attempts.append((success, _overlap(index, trace, truth_index), float(scores[index]), index, final))
    if not attempts:
        return {"truth_key": _truth_key(trace, truth_index), "outcome": "downstream_still_rejects", "witness_raw_index": None, "final_indices": ()}
    attempts.sort(key=lambda item: (not item[0], -item[1], -item[2], item[3]))
    success, _, _, index, final = attempts[0]
    return {"truth_key": _truth_key(trace, truth_index), "outcome": "control_limited_and_downstream_capable" if success else "downstream_still_rejects", "witness_raw_index": index, "final_indices": final, "diagnostic_only": "DIAGNOSTIC ONLY / USES GT"}


def audit_trace(trace: ReplayTrace) -> dict[str, object]:
    """Audit one sealed trace with frozen scores and exactly three ablations."""
    policy, scores = _frozen_policy(trace), _scores(trace)
    groups = _groups(trace)
    baseline = tuple(trace.admitted_indices)
    expected = admit_candidates(trace.raw_union, scores, groups, CandidateAdmissionConfig(float(policy["nms_iou"]), float(policy["admission_threshold"]), policy["max_candidates_per_subject"]))
    if baseline != expected or tuple(trace.final_indices) != _final_for_indices(trace, baseline):
        raise RuntimeError("replay trace does not reproduce frozen admission/event policy")
    c1 = [row for row in classify_false_negatives(trace) if row["fn_stage"] == "FN-C1"]
    target_indices = [int(row["truth_index"]) for row in c1]
    ledger = {int(row["raw_index"]): row for row in trace.candidate_rows}
    truth_rows, candidate_rows = [], []
    for truth_index in target_indices:
        reachable = [i for i in range(len(trace.raw_union)) if _overlap(i, trace, truth_index) >= _IOU]
        stages = {"raw": reachable, "nms": list(suppress_overlapping_candidates(trace.raw_union, scores, float(policy["nms_iou"])))}
        stages["admission_threshold"] = [i for i in stages["nms"] if scores[i] >= float(policy["admission_threshold"])]
        stages["subject_cap"] = [i for i in baseline if i in stages["admission_threshold"]]
        reachable_stages = {name: any(i in values for i in reachable) for name, values in stages.items()}
        loss = next((name for name in ("nms", "admission_threshold", "subject_cap") if not reachable_stages[name]), None)
        reasons = {str(ledger[i]["terminal_reason"]) for i in reachable}
        truth_rows.append({"truth_key": _truth_key(trace, truth_index), "truth_index": truth_index, "first_total_loss_stage": loss, "terminal_reason_set": tuple(sorted(reasons)), "cause": "mixed" if len(reasons) > 1 else (next(iter(reasons)) if reasons else None), "reachability": reachable_stages})
        for raw_index in reachable:
            candidate_rows.append({"truth_key": _truth_key(trace, truth_index), "raw_index": raw_index, "iou": _overlap(raw_index, trace, truth_index), **dict(ledger[raw_index])})
    baseline_fp = _unmatched_event_counter(trace, trace.final_indices)
    ablations: dict[str, object] = {}
    for mode in MODES:
        admitted, final = _admitted_for_mode(trace, mode), _final_for_indices(trace, _admitted_for_mode(trace, mode))
        pairs = _matches(trace, final)
        matched = {_truth_key(trace, pair.truth_index) for pair in pairs}
        new = set(admitted) - set(baseline)
        candidate_nonmatching = sum(not any(_overlap(i, trace, truth_index) >= _IOU for truth_index in range(len(trace.truths))) for i in new)
        current_fp = _unmatched_event_counter(trace, final)
        event_diffs = tuple(
            {"event": key, "kind": "introduced_unmatched_final_event", "count": count}
            for key, count in sorted((current_fp - baseline_fp).items())
        ) + tuple(
            {"event": key, "kind": "removed_incumbent_unmatched_event", "count": count}
            for key, count in sorted((baseline_fp - current_fp).items())
        )
        ablations[mode] = {"admitted_indices": admitted, "final_indices": final, "newly_admitted_indices": tuple(sorted(new)), "extra_nonmatching_candidates": candidate_nonmatching, "newly_matched_c1_truth_keys": tuple(sorted(matched & {_truth_key(trace, i) for i in target_indices})), "gross_introduced_final_fp": sum((current_fp - baseline_fp).values()), "removed_incumbent_final_fp": sum((baseline_fp - current_fp).values()), "net_final_fp_difference": sum(current_fp.values()) - sum(baseline_fp.values())}
        ablations[mode]["event_diffs"] = event_diffs
    return {"diagnostic_only": "DIAGNOSTIC ONLY / USES GT", "frozen_policy": policy, "frozen_blend_scores": scores.copy(), "baseline_admitted_indices": baseline, "truth_rows": truth_rows, "candidate_rows": candidate_rows, "ablations": ablations, "oracle_rows": [_oracle(trace, i, baseline) for i in target_indices]}
