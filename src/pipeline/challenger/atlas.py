"""Pure, deterministic E01 false-negative and false-positive diagnostics.

Nothing in this module trains, ranks, or filters a candidate.  It only
describes the immutable trace produced by :mod:`src.pipeline.challenger.replay`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isnan
from statistics import mean
from typing import Mapping, Sequence

from src.eval.metrics import event_iou
from src.pipeline.challenger.replay import MatchPair, ReplayTrace
from src.pipeline.context_features import CONTEXT_V1_COLUMNS
from src.pipeline.event_stack import EventRef, MultiScaleCandidate


_IOU = 0.25
_SHORT_MS = 600_000
_MEDIUM_MS = 1_200_000
_NEIGHBOUR_MS = 1_200_000


def _truth_key(fold: int, truth: EventRef) -> str:
    return f"{fold}:{truth.sid}:{truth.start_ms}:{truth.end_ms}"


def _duration_slice(event: EventRef) -> str:
    duration = event.end_ms - event.start_ms
    if duration < _SHORT_MS:
        return "short"
    if duration < _MEDIUM_MS:
        return "medium"
    return "long"


def _overlap(left: EventRef, right: EventRef) -> float:
    if left.sid != right.sid:
        return 0.0
    return float(event_iou(left.interval, right.interval))


def _partial_overlap(left: EventRef, right: EventRef) -> bool:
    value = _overlap(left, right)
    return 0.0 < value < _IOU


def _row_by_raw(trace: ReplayTrace) -> dict[int, Mapping[str, object]]:
    rows = {int(row["raw_index"]): row for row in trace.candidate_rows}
    if set(rows) != set(range(len(trace.raw_union))):
        raise RuntimeError("candidate ledger does not provide one row per raw union candidate")
    return rows


def _support(candidates: Sequence[MultiScaleCandidate]) -> dict[str, object]:
    macro = sum(candidate.macro is not None for candidate in candidates)
    micro = sum(candidate.micro is not None for candidate in candidates)
    both = sum(candidate.macro is not None and candidate.micro is not None for candidate in candidates)
    return {
        "reachable_candidate_count": len(candidates),
        "macro_support_count": macro,
        "micro_support_count": micro,
        "macro_micro_support_count": both,
        "macro_only_count": sum(candidate.macro is not None and candidate.micro is None for candidate in candidates),
        "micro_only_count": sum(candidate.macro is None and candidate.micro is not None for candidate in candidates),
    }


def _subject_for(trace: ReplayTrace, sid: str, candidates: Sequence[MultiScaleCandidate]) -> object:
    """Use replay provenance when available; C0 may truly have no candidate."""
    value = trace.subject_by_session.get(sid)
    if value is not None:
        return value
    rows = _row_by_raw(trace)
    for index, candidate in enumerate(trace.raw_union):
        if candidate in candidates:
            value = rows[index].get("subject_id")
            if value is not None:
                return value
    return None


def classify_false_negatives(trace: ReplayTrace) -> list[dict[str, object]]:
    """Classify unmatched GT by earliest observable pipeline failure.

    The fixed C1→C2→C3→C0→C4 order records candidate-control removal,
    then event-policy removal, then final geometry, then absence of any raw
    union candidate; C4 is the remaining greedy-matcher ambiguity.
    """
    rows = _row_by_raw(trace)
    matched_truths = {pair.truth_index for pair in trace.match_pairs}
    result: list[dict[str, object]] = []
    for truth_index, truth in enumerate(trace.truths):
        if truth_index in matched_truths:
            continue
        raw_indices = [index for index, candidate in enumerate(trace.raw_union) if _overlap(candidate.event, truth) >= _IOU]
        macro_raw_indices = [index for index, candidate in enumerate(trace.raw_macro) if _overlap(candidate.event, truth) >= _IOU]
        micro_raw_indices = [index for index, candidate in enumerate(trace.raw_micro) if _overlap(candidate.event, truth) >= _IOU]
        admitted_indices = [index for index in trace.admitted_indices if _overlap(trace.raw_union[index].event, truth) >= _IOU]
        final_indices = [index for index in trace.final_indices if _overlap(trace.raw_union[index].event, truth) >= _IOU]
        partial_final_indices = [index for index, prediction in enumerate(trace.predictions) if _partial_overlap(prediction, truth)]

        if raw_indices and not admitted_indices:
            stage = "FN-C1"
        elif admitted_indices and not final_indices:
            stage = "FN-C2"
        elif partial_final_indices:
            stage = "FN-C3"
        elif not raw_indices:
            stage = "FN-C0"
        else:
            stage = "FN-C4"

        reachable = [trace.raw_union[index] for index in raw_indices]
        entry: dict[str, object] = {
            "outer_fold": trace.fold,
            "truth_index": truth_index,
            "truth_key": _truth_key(trace.fold, truth),
            "subject_id": _subject_for(trace, truth.sid, reachable),
            "session_id": truth.sid,
            "start_ms": truth.start_ms,
            "end_ms": truth.end_ms,
            "duration_ms": truth.end_ms - truth.start_ms,
            "duration_slice": _duration_slice(truth),
            "fn_stage": stage,
            "raw_reachable": bool(raw_indices),
            "union_raw_reachable": bool(raw_indices),
            "macro_raw_reachable": bool(macro_raw_indices),
            "micro_raw_reachable": bool(micro_raw_indices),
            "macro_raw_reachable_indices": macro_raw_indices,
            "micro_raw_reachable_indices": micro_raw_indices,
            "admitted_reachable": bool(admitted_indices),
            "final_reachable": bool(final_indices),
            "partial_final_overlap": bool(partial_final_indices),
            "raw_reachable_indices": raw_indices,
            "admitted_reachable_indices": admitted_indices,
            "final_reachable_indices": final_indices,
            "partial_final_indices": partial_final_indices,
            **_support(reachable),
        }
        if stage == "FN-C4":
            consuming: MatchPair | None = None
            for pair in trace.match_pairs:
                if pair.truth_index == truth_index:
                    continue
                if _overlap(trace.predictions[pair.prediction_index], truth) >= _IOU:
                    consuming = pair
                    break
            if consuming is not None:
                consumed_truth = trace.truths[consuming.truth_index]
                entry.update(
                    consuming_final_event_index=consuming.prediction_index,
                    consuming_truth_key=_truth_key(trace.fold, consumed_truth),
                    consuming_pair_iou=consuming.iou,
                )
            else:
                entry.update(consuming_final_event_index=None, consuming_truth_key=None, consuming_pair_iou=None)
        else:
            entry.update(consuming_final_event_index=None, consuming_truth_key=None, consuming_pair_iou=None)
        result.append(entry)
    return result


def _interval_gap(left: EventRef, right: EventRef) -> int:
    if left.end_ms < right.start_ms:
        return right.start_ms - left.end_ms
    if right.end_ms < left.start_ms:
        return left.start_ms - right.end_ms
    return 0


def _percentile(values: Sequence[float], value: float) -> float | None:
    finite = sorted(item for item in values if item == item)
    if not finite or value != value:
        return None
    return sum(item <= value for item in finite) / len(finite)


def false_positive_rows(trace: ReplayTrace) -> list[dict[str, object]]:
    """Return exactly final predictions which official greedy matching leaves unmatched."""
    rows = _row_by_raw(trace)
    matched = {pair.prediction_index for pair in trace.match_pairs}
    final_to_raw = {int(rows[index]["final_event_index"]): index for index in trace.final_indices}
    if set(final_to_raw) != set(range(len(trace.predictions))):
        raise RuntimeError("candidate ledger final-event coordinates are incomplete")
    by_session: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(trace.raw_union):
        by_session[candidate.event.sid].append(index)
    result: list[dict[str, object]] = []
    for final_index, prediction in enumerate(trace.predictions):
        if final_index in matched:
            continue
        raw_index = final_to_raw[final_index]
        row = rows[raw_index]
        candidate = trace.raw_union[raw_index]
        session_indices = sorted(by_session[prediction.sid], key=lambda index: (trace.raw_union[index].event.start_ms, trace.raw_union[index].event.end_ms, index))
        position = session_indices.index(raw_index)
        neighbours = [index for index in session_indices if index != raw_index]
        nearest = min((_interval_gap(prediction, trace.raw_union[index].event) for index in neighbours), default=None)
        previous = session_indices[position - 1] if position else None
        following = session_indices[position + 1] if position + 1 < len(session_indices) else None
        score = float(row["blend_score"])
        duration = prediction.end_ms - prediction.start_ms
        session_scores = [float(rows[index]["blend_score"]) for index in session_indices]
        session_durations = [float(trace.raw_union[index].event.end_ms - trace.raw_union[index].event.start_ms) for index in session_indices]
        macro_values = [max(item.macro.probabilities) for item in (trace.raw_union[index] for index in session_indices) if item.macro is not None]
        micro_values = [max(item.micro.probabilities) for item in (trace.raw_union[index] for index in session_indices) if item.micro is not None]
        macro_score = max(candidate.macro.probabilities) if candidate.macro is not None else None
        micro_score = max(candidate.micro.probabilities) if candidate.micro is not None else None
        nearby = sum(_interval_gap(prediction, trace.raw_union[index].event) <= _NEIGHBOUR_MS for index in neighbours)
        macro, micro = candidate.macro is not None, candidate.micro is not None
        duplicate = any(_overlap(prediction, truth) >= _IOU for truth in trace.truths)
        entry: dict[str, object] = {
            "outer_fold": trace.fold, "final_index": final_index, "raw_index": raw_index,
            "subject_id": trace.subject_by_session.get(prediction.sid, row.get("subject_id")), "session_id": prediction.sid,
            "start_ms": prediction.start_ms, "end_ms": prediction.end_ms, "duration_ms": duration,
            "macro_support": macro, "micro_support": micro, "macro_micro_support": macro and micro,
            "logistic_score": row["logistic_score"], "lgbm_score": row["lgbm_score"], "blend_score": score,
            "nms_rank": row["nms_rank"], "admitted_index": row["admitted_index"],
            "event_cap_rank": row["event_cap_rank"], "terminal_reason": row["terminal_reason"],
            "nearest_candidate_distance_ms": nearest,
            "previous_raw_index": previous, "next_raw_index": following,
            "previous_evidence_score": rows[previous]["blend_score"] if previous is not None else None,
            "next_evidence_score": rows[following]["blend_score"] if following is not None else None,
            "session_blend_percentile": _percentile(session_scores, score),
            "session_duration_percentile": _percentile(session_durations, float(duration)),
            "macro_score_percentile": _percentile(macro_values, macro_score) if macro_score is not None else None,
            "micro_score_percentile": _percentile(micro_values, micro_score) if micro_score is not None else None,
            "candidate_score_percentile": _percentile(session_scores, score),
            "duration_percentile": _percentile(session_durations, float(duration)),
            "candidate_count_session": len(session_indices),
            "short_spike": duration < _SHORT_MS,
            "long_continuous_activity": duration >= _MEDIUM_MS,
            "macro_high_micro_low": macro and not micro,
            "micro_high_macro_low": micro and not macro,
            "dense_cluster": nearby >= 2,
            "isolated": nearby == 0,
            "same_truth_duplicate": duplicate,
        }
        entry.update({name: row.get(name) for name in CONTEXT_V1_COLUMNS})
        result.append(entry)
    return result


def _numeric_mean(rows: Sequence[Mapping[str, object]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool) and not isnan(float(row[name]))]
    return mean(values) if values else None


def _distribution(rows: Sequence[Mapping[str, object]], name: str) -> dict[str, float | int | None]:
    values = sorted(
        float(row[name]) for row in rows
        if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)
        and float(row[name]) == float(row[name])
    )
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    def at(fraction: float) -> float:
        return values[min(len(values) - 1, int(fraction * (len(values) - 1)))]
    return {"count": len(values), "min": values[0], "p25": at(.25), "median": at(.5), "p75": at(.75), "max": values[-1]}


def _fn_support(row: Mapping[str, object]) -> str:
    macro = bool(row.get("macro_raw_reachable"))
    micro = bool(row.get("micro_raw_reachable"))
    if macro and micro:
        return "macro+micro"
    if macro:
        return "macro-only"
    if micro:
        return "micro-only"
    return "neither"


def atlas_summaries(traces: Sequence[ReplayTrace]) -> dict[str, object]:
    """Aggregate atlas facts without deriving a learned decision rule."""
    fn_rows = [row for trace in traces for row in classify_false_negatives(trace)]
    fp_rows = [row for trace in traces for row in false_positive_rows(trace)]
    keys = [str(row["truth_key"]) for row in fn_rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("false-negative truth keys must be globally unique")
    per_fold: dict[str, dict[str, int]] = {}
    for trace in traces:
        per_fold[str(trace.fold)] = {
            "tp": trace.metrics.n_tp, "eligible": trace.metrics.n_true, "pred": trace.metrics.n_pred,
            "fn": len(classify_false_negatives(trace)), "fp": len(false_positive_rows(trace)),
        }
    tp_rows: list[Mapping[str, object]] = []
    for trace in traces:
        ledger = _row_by_raw(trace)
        for pair in trace.match_pairs:
            raw = next(index for index in trace.final_indices if ledger[index]["final_event_index"] == pair.prediction_index)
            tp_rows.append(ledger[raw])
    def separation(fps: Sequence[Mapping[str, object]], tps: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float | None]]:
        return {
            name: {
                "fp_mean": _numeric_mean(fps, name), "tp_mean": _numeric_mean(tps, name),
                "fp_minus_tp": (None if _numeric_mean(fps, name) is None or _numeric_mean(tps, name) is None else _numeric_mean(fps, name) - _numeric_mean(tps, name)),
            }
            for name in CONTEXT_V1_COLUMNS
        }
    per_fold_context: dict[str, object] = {}
    for trace in traces:
        ledger = _row_by_raw(trace)
        fold_tps = [ledger[next(index for index in trace.final_indices if ledger[index]["final_event_index"] == pair.prediction_index)] for pair in trace.match_pairs]
        per_fold_context[str(trace.fold)] = separation(false_positive_rows(trace), fold_tps)
    sign_consistency: dict[str, dict[str, int]] = {}
    for name in CONTEXT_V1_COLUMNS:
        deltas = [value[name]["fp_minus_tp"] for value in per_fold_context.values()]
        sign_consistency[name] = {
            "positive": sum(value is not None and value > 0 for value in deltas),
            "negative": sum(value is not None and value < 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas if value is not None),
            "noncomparable": sum(value is None for value in deltas),
        }
    fn_cross_stats: dict[str, dict[str, dict[str, dict[str, int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for row in fn_rows:
        fn_cross_stats[str(row["duration_slice"])][str(row["outer_fold"])][str(row["subject_id"])][_fn_support(row)] += 1
    short_rows = [row for row in fn_rows if row["duration_slice"] == "short"]
    fp_subject_profile: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "pred": 0, "eligible": 0})
    for trace in traces:
        matched_predictions = {pair.prediction_index for pair in trace.match_pairs}
        for truth in trace.truths:
            fp_subject_profile[str(trace.subject_by_session.get(truth.sid, truth.sid))]["eligible"] += 1
        for index, prediction in enumerate(trace.predictions):
            subject = str(trace.subject_by_session.get(prediction.sid, prediction.sid))
            fp_subject_profile[subject]["pred"] += 1
            if index in matched_predictions:
                fp_subject_profile[subject]["tp"] += 1
            else:
                fp_subject_profile[subject]["fp"] += 1
    q3_fields = ("session_blend_percentile", "candidate_score_percentile", "duration_percentile", "candidate_count_session", "nearest_candidate_distance_ms")
    feature_report = []
    global_separation = separation(fp_rows, tp_rows)
    for name in CONTEXT_V1_COLUMNS:
        feature_report.append({
            "feature": name,
            "global": global_separation[name],
            "per_fold": {fold: values[name] for fold, values in per_fold_context.items()},
            "sign_consistency": sign_consistency[name],
        })
    stage_counts = dict(sorted(Counter(str(row["fn_stage"]) for row in short_rows).items()))
    recommendations = [
        {"mechanism": "candidate_recall", "observed": f"{sum(row['fn_stage'] == 'FN-C1' for row in fn_rows)} FN-C1 of {len(fn_rows)} FN; short-stage split={stage_counts}", "recommendation": "Prioritize candidate-control reachability diagnostics before changing verifier complexity."},
        {"mechanism": "scale_support", "observed": dict(sorted(Counter(_fn_support(row) for row in fn_rows).items())), "recommendation": "Compare macro-only, micro-only and joint support in a preregistered candidate experiment; this atlas does not select features."},
        {"mechanism": "false_positive_concentration", "observed": f"{len(fp_rows)} unmatched final predictions across {len(fp_subject_profile)} subjects", "recommendation": "Use subject/session-local diagnostics to formulate, but not fit, a later hard-negative hypothesis."},
    ]
    summary: dict[str, object] = {
        "incumbent_metrics": {
            "tp": sum(trace.metrics.n_tp for trace in traces),
            "eligible": sum(trace.metrics.n_true for trace in traces),
            "pred": sum(trace.metrics.n_pred for trace in traces),
            "fp": len(fp_rows), "fn": len(fn_rows),
        },
        "fn_stage_decomposition": dict(sorted(Counter(str(row["fn_stage"]) for row in fn_rows).items())),
        "fn_duration_slice": dict(sorted(Counter(str(row["duration_slice"]) for row in fn_rows).items())),
        "fn_by_fold": dict(sorted(Counter(str(row["outer_fold"]) for row in fn_rows).items())),
        "fn_by_subject": dict(sorted(Counter(str(row["subject_id"]) for row in fn_rows).items())),
        "fn_cross_stats": {duration: {fold: {subject: dict(supports) for subject, supports in subjects.items()} for fold, subjects in folds.items()} for duration, folds in fn_cross_stats.items()},
        "fn_short_stage_decomposition": stage_counts,
        "fp_taxonomy": {name: sum(bool(row[name]) for row in fp_rows) for name in ("short_spike", "long_continuous_activity", "macro_high_micro_low", "micro_high_macro_low", "dense_cluster", "isolated", "same_truth_duplicate")},
        "fp_support": {
            "macro_only": sum(bool(row["macro_support"]) and not bool(row["micro_support"]) for row in fp_rows),
            "micro_only": sum(not bool(row["macro_support"]) and bool(row["micro_support"]) for row in fp_rows),
            "macro_micro": sum(bool(row["macro_micro_support"]) for row in fp_rows),
        },
        "fp_by_fold": dict(sorted(Counter(str(row["outer_fold"]) for row in fp_rows).items())),
        "fp_by_subject": dict(sorted(Counter(str(row["subject_id"]) for row in fp_rows).items())),
        "fp_subject_profile": dict(sorted(fp_subject_profile.items())),
        "fp_q3_distributions": {name: _distribution(fp_rows, name) for name in q3_fields},
        "per_fold_error_profile": per_fold,
        "context_v1_residual_separation": {
            "global": separation(fp_rows, tp_rows), "per_fold": per_fold_context,
            "fold_sign_consistency": sign_consistency,
        },
        "context_v1_feature_report": feature_report,
        "mechanism_recommendations": recommendations,
        "fold_consistency": {
            "fp_counts": [len(false_positive_rows(trace)) for trace in traces],
            "fn_counts": [len(classify_false_negatives(trace)) for trace in traces],
        },
    }
    if {trace.fold for trace in traces} == set(range(5)) and len(traces) == 5:
        metrics = summary["incumbent_metrics"]
        if metrics != {"tp": 114, "eligible": 153, "pred": 197, "fp": 83, "fn": 39}:
            raise RuntimeError("frozen incumbent atlas totals do not match 114/153/197/83/39")
        if sum(row["duration_slice"] == "short" for row in fn_rows) != 19:
            raise RuntimeError("frozen incumbent atlas must contain exactly 19 short false negatives")
    return summary


def render_report(summary: Mapping[str, object]) -> str:
    """Render a small human-readable companion to the machine-readable atlas."""
    metric = summary["incumbent_metrics"]
    lines = [
        "# E01 Error Atlas",
        "", "## Incumbent metrics", "", f"- TP / eligible / pred: {metric['tp']} / {metric['eligible']} / {metric['pred']}",
        f"- FP / FN: {metric['fp']} / {metric['fn']}", "", "## FN stage decomposition", "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary["fn_stage_decomposition"].items())
    lines.extend(["", "## Short FN breakdown", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["fn_duration_slice"].items())
    lines.extend(["", "## Short FN stage split", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["fn_short_stage_decomposition"].items())
    lines.extend(["", "## FP taxonomy", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["fp_taxonomy"].items())
    lines.extend(["", "## Support and geometry", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["fp_support"].items())
    lines.extend(["", "## FP Q3 distributions", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["fp_q3_distributions"].items())
    lines.extend(["", "## FN cross statistics", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["fn_cross_stats"].items())
    lines.extend(["", "## Subject concentration", ""])
    lines.extend(f"- FP subject {key}: {value}" for key, value in summary["fp_by_subject"].items())
    lines.extend(f"- FN subject {key}: {value}" for key, value in summary["fn_by_subject"].items())
    lines.extend(["", "## Subject TP/FP/pred/eligible profile", ""])
    lines.extend(f"- {subject}: TP={values['tp']}, FP={values['fp']}, pred={values['pred']}, eligible={values['eligible']}" for subject, values in summary["fp_subject_profile"].items())
    lines.extend(["", "## per-fold error profile", ""])
    lines.extend(f"- fold {key}: {value}" for key, value in summary["per_fold_error_profile"].items())
    lines.extend(["", "## Context-v1 residual separation", "", "Descriptive FP-versus-TP means only; no classifier or learned transform was used."])
    lines.extend(["", "## Q6 feature separation and sign consistency", ""])
    lines.extend(f"- {row['feature']}: global={row['global']}; sign={row['sign_consistency']}" for row in summary["context_v1_feature_report"])
    lines.extend(["", "## Recommended next experiments", "", "Evidence-based mechanisms only; no outer-derived feature selection was performed."])
    lines.extend(f"- {row['mechanism']}: observed={row['observed']}; recommendation={row['recommendation']}" for row in summary["mechanism_recommendations"])
    return "\n".join(lines) + "\n"
