"""Pure, GT-aware E04 autopsy of frozen verifier scores.

The functions in this module only arrange values which are already present in
a :class:`ReplayTrace`; they never score, fit, or alter a candidate policy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np

from src.eval.metrics import event_iou
from src.pipeline.challenger.atlas import classify_false_negatives
from src.pipeline.challenger.replay import ReplayTrace, official_match_indices
from src.pipeline.challenger.verifier_attribution import explain_fitted_verifiers
from src.pipeline.context_features import CONTEXT_V1_COLUMNS


_IOU = 0.25
_BASE_COLUMNS = tuple(f"base_{index:02d}" for index in range(56))
_FEATURE_COLUMNS = _BASE_COLUMNS + CONTEXT_V1_COLUMNS


def _ledger(trace: ReplayTrace) -> dict[int, Mapping[str, object]]:
    rows = {int(row["raw_index"]): row for row in trace.candidate_rows}
    if set(rows) != set(range(len(trace.raw_union))):
        raise RuntimeError("candidate ledger does not provide one row per raw union candidate")
    return rows


def _finite(value: object) -> float | None:
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        number = float(value)
        return number if np.isfinite(number) else None
    return None


def _row(trace: ReplayTrace, index: int, contributions: Mapping[str, np.ndarray], *, group: str,
         truth: Mapping[str, object] | None = None, iou: float | None = None,
         final_index: int | None = None) -> dict[str, object]:
    if trace.verifier_features is None:
        raise RuntimeError("E04 audit requires frozen 116D verifier features")
    ledger = _ledger(trace)[index]
    features = trace.verifier_features[index]
    threshold = _finite(trace.policy.get("threshold"))
    if threshold is None:
        raise RuntimeError("E04 audit requires a frozen event threshold")
    lr, lgbm, blend = (_finite(ledger.get(name)) for name in ("logistic_score", "lgbm_score", "blend_score"))
    if lr is None or lgbm is None or blend is None:
        raise RuntimeError("candidate ledger has invalid frozen verifier scores")
    blend_weight = _finite(trace.policy.get("blend_weight"))
    if blend_weight is None:
        raise RuntimeError("E04 audit requires a finite frozen blend weight")
    expected_blend = blend_weight * lr + (1.0 - blend_weight) * lgbm
    if abs(blend - expected_blend) > 1e-12:
        raise RuntimeError("candidate ledger blend score disagrees with the frozen blend identity")
    result: dict[str, object] = {
        "fold": trace.fold, "outer_fold": trace.fold, "group": group, "raw_index": index,
        "candidate_key": f"{trace.fold}:{index}", "final_index": final_index,
        "session_id": ledger.get("session_id"), "subject_id": ledger.get("subject_id"),
        "start_ms": ledger.get("start_ms"), "end_ms": ledger.get("end_ms"),
        "logistic_score": lr, "lgbm_score": lgbm, "blend_score": blend,
        "blend_weight": blend_weight, "event_threshold": threshold,
        "margin": blend - threshold, "lr_minus_lgbm": lr - lgbm,
        "absolute_lr_lgbm_difference": abs(lr - lgbm),
        "disagreement_flag": abs(lr - lgbm) >= .20,
        "higher_model": "logistic" if lr > lgbm else "lgbm" if lgbm > lr else "tie",
        "terminal_reason": ledger.get("terminal_reason"),
        "macro_support": trace.raw_union[index].macro is not None,
        "micro_support": trace.raw_union[index].micro is not None,
        "macro_micro_support": trace.raw_union[index].macro is not None and trace.raw_union[index].micro is not None,
        "iou": iou, "diagnostic_only_uses_gt": truth is not None,
    }
    if truth is not None:
        result.update({key: truth.get(key) for key in ("truth_index", "truth_key", "fn_stage", "duration_slice")})
    for name, value in zip(_FEATURE_COLUMNS, features, strict=True):
        result[name] = float(value)  # NaN deliberately remains NaN for CSV empty-cell serialization.
    for name, values in contributions.items():
        result[name] = float(values[index])
    return result


def _references(trace: ReplayTrace, contributions: Mapping[str, np.ndarray]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ledger = _ledger(trace)
    final_keys = [int(ledger[index]["final_event_index"]) for index in trace.final_indices]
    if len(final_keys) != len(set(final_keys)):
        raise RuntimeError("candidate ledger has duplicate final-event indices")
    final_by_prediction = dict(zip(final_keys, trace.final_indices, strict=True))
    if set(final_by_prediction) != set(range(len(trace.predictions))):
        raise RuntimeError("candidate ledger final-event coordinates are incomplete")
    matched_predictions = {pair.prediction_index for pair in trace.match_pairs}
    tp = []
    for pair in trace.match_pairs:
        row = _row(trace, final_by_prediction[pair.prediction_index], contributions, group="tp_reference",
                   final_index=pair.prediction_index)
        truth = trace.truths[pair.truth_index]
        row.update({"matched_truth_index": pair.truth_index,
                    "matched_truth_key": f"{trace.fold}:{truth.sid}:{truth.start_ms}:{truth.end_ms}",
                    "matched_iou": pair.iou})
        tp.append(row)
    fp = [_row(trace, raw, contributions, group="fp_reference", final_index=final)
          for final, raw in sorted(final_by_prediction.items()) if final not in matched_predictions]
    return tp, fp


def audit_verifier_trace(trace: ReplayTrace) -> dict[str, object]:
    """Construct one pure fold's candidate, witness, reference, and example rows."""
    if trace.verifier_features is None:
        raise RuntimeError("E04 audit requires frozen 116D verifier features")
    models = trace.verifier_models
    if set(models) != {"logistic", "lgbm"}:
        raise RuntimeError("E04 audit requires frozen logistic and lgbm verifier models")
    ledger = _ledger(trace)
    if tuple(trace.match_pairs) != official_match_indices(trace.predictions, trace.truths):
        raise RuntimeError("replay match pairs disagree with official matcher")
    contributions = explain_fitted_verifiers(
        trace.verifier_features, models["logistic"], models["lgbm"],
        np.asarray([float(ledger[index]["logistic_score"]) for index in range(len(ledger))]),
        np.asarray([float(ledger[index]["lgbm_score"]) for index in range(len(ledger))]),
    )
    fn_pairs: list[dict[str, object]] = []
    by_truth: dict[str, list[dict[str, object]]] = defaultdict(list)
    for fn in classify_false_negatives(trace):
        if not bool(fn["raw_reachable"]) or fn["fn_stage"] == "FN-C0":
            continue
        truth = trace.truths[int(fn["truth_index"])]
        for index, candidate in enumerate(trace.raw_union):
            iou = float(event_iou(candidate.event.interval, truth.interval)) if candidate.event.sid == truth.sid else 0.0
            if iou >= _IOU:
                row = _row(trace, index, contributions, group="reachable_fn_candidate", truth=fn, iou=iou)
                fn_pairs.append(row)
                by_truth[str(fn["truth_key"])].append(row)
    best_iou = [dict(sorted(rows, key=lambda row: (-float(row["iou"]), -float(row["blend_score"]), int(row["raw_index"])))[0])
                for _, rows in sorted(by_truth.items())]
    best_score = [dict(sorted(rows, key=lambda row: (-float(row["blend_score"]), -float(row["iou"]), int(row["raw_index"])))[0])
                  for _, rows in sorted(by_truth.items())]
    for row in best_iou:
        row["witness_definition"] = "best_iou"
    for row in best_score:
        row["witness_definition"] = "best_score"
    tp, fp = _references(trace, contributions)
    analysis_rows = [dict(row, group="best_iou_fn") for row in best_iou] + [dict(row, group="best_score_fn") for row in best_score] + tp + fp
    negative = [row for row in best_score if float(row["margin"]) < 0]
    examples: list[dict[str, object]] = []
    if negative:
        chosen = [sorted(negative, key=lambda row: (float(row["margin"]), str(row["truth_key"]), int(row["raw_index"])))[0],
                  sorted(negative, key=lambda row: (-float(row["margin"]), str(row["truth_key"]), int(row["raw_index"])))[0]]
        seen: set[tuple[str, int]] = set()
        for row in chosen:
            key = (str(row["truth_key"]), int(row["raw_index"]))
            if key not in seen:
                examples.append(dict(row))
                seen.add(key)
    return {"fold": trace.fold, "reachable_fn_candidates": fn_pairs, "tp_reference": tp, "fp_reference": fp,
            "best_iou_witnesses": best_iou, "best_score_witnesses": best_score,
            "analysis_rows": analysis_rows, "examples": examples}


def _quantile(rows: Sequence[Mapping[str, object]], field: str, fraction: float) -> float | None:
    values = sorted(value for value in (_finite(row.get(field)) for row in rows) if value is not None)
    return values[int(fraction * (len(values) - 1))] if values else None


def _mechanisms(row: Mapping[str, object], tp: Sequence[Mapping[str, object]], fp: Sequence[Mapping[str, object]]) -> list[str]:
    labels: list[str] = []
    margin = float(row["margin"])
    if -.05 <= margin < 0:
        labels.append("A")
    if bool(row.get("disagreement_flag", abs(float(row["logistic_score"]) - float(row["lgbm_score"])) >= .20)):
        labels.append("C")
    tp_lr_base, tp_lgbm_base = _quantile(tp, "lr_base", .5), _quantile(tp, "lgbm_base", .5)
    tp_lr_ctx, tp_lgbm_ctx = _quantile(tp, "lr_context", .25), _quantile(tp, "lgbm_context", .25)
    b_lr = None if None in (tp_lr_base, tp_lr_ctx) else float(row["lr_base"]) >= tp_lr_base and float(row["lr_context"]) < 0 and float(row["lr_context"]) <= tp_lr_ctx
    b_lgbm = None if None in (tp_lgbm_base, tp_lgbm_ctx) else float(row["lgbm_base"]) >= tp_lgbm_base and float(row["lgbm_context"]) < 0 and float(row["lgbm_context"]) <= tp_lgbm_ctx
    if b_lr is True and b_lgbm is True:
        labels.append("B")
    elif b_lr is not None and b_lgbm is not None and b_lr != b_lgbm:
        labels.append("B_model_disagreement")
    elif b_lr is None or b_lgbm is None:
        labels.append("unsupported")
    tp_lr, tp_lgbm = _quantile(tp, "logistic_score", .25), _quantile(tp, "lgbm_score", .25)
    fp_lr_q1, fp_lr_q3 = _quantile(fp, "logistic_score", .25), _quantile(fp, "logistic_score", .75)
    fp_lgbm_q1, fp_lgbm_q3 = _quantile(fp, "lgbm_score", .25), _quantile(fp, "lgbm_score", .75)
    d_values = (tp_lr, tp_lgbm, fp_lr_q1, fp_lr_q3, fp_lgbm_q1, fp_lgbm_q3)
    if all(value is not None for value in d_values):
        if float(row["logistic_score"]) < tp_lr and float(row["lgbm_score"]) < tp_lgbm and fp_lr_q1 <= float(row["logistic_score"]) <= fp_lr_q3 and fp_lgbm_q1 <= float(row["lgbm_score"]) <= fp_lgbm_q3:
            labels.append("D")
    else:
        labels.append("unsupported")
    if not any(label in {"A", "B", "C", "D", "B_model_disagreement"} for label in labels):
        labels.append("unresolved")
    return labels


def _distribution(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, object]:
    """Descriptive numeric distribution preserving unsupported/missing support."""
    values = sorted(value for value in (_finite(row.get(field)) for row in rows) if value is not None)
    missing = len(rows) - len(values)
    if not values:
        return {"count": 0, "missing": missing, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    at = lambda fraction: values[int(fraction * (len(values) - 1))]
    return {"count": len(values), "missing": missing, "min": values[0], "p25": at(.25),
            "median": at(.5), "p75": at(.75), "max": values[-1]}


def _margin_bin(value: float) -> str:
    return ("deep_below" if value < -.10 else "below_0_10_to_0_05" if value < -.05
            else "below_0_05_to_0" if value < 0 else "zero_to_0_05" if value <= .05 else "above_0_05")


def _margin_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    bins = {"deep_below": 0, "below_0_10_to_0_05": 0, "below_0_05_to_0": 0, "zero_to_0_05": 0, "above_0_05": 0}
    for row in rows:
        bins[_margin_bin(float(row["margin"]))] += 1
    return {"bins": bins, "distribution": _distribution(rows, "margin")}


def summarize_verifier_audits(audits: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize descriptive fixed diagnostics without fitting or scoring anything."""
    audit_folds = [audit.get("fold") for audit in audits]
    if len(audits) != 5 or set(audit_folds) != set(range(5)):
        raise ValueError("E04 summary requires five distinct folds 0 through 4")
    analysis = [row for audit in audits for row in audit.get("analysis_rows", ()) if isinstance(row, Mapping)]
    groups = ("best_iou_fn", "best_score_fn", "tp_reference", "fp_reference")
    by_group = {group: [row for row in analysis if row.get("group") == group] for group in groups}
    folds = sorted({int(audit["fold"]) for audit in audits if isinstance(audit.get("fold"), int)})
    margin_analysis = {
        "pooled": {group: _margin_summary(rows) for group, rows in by_group.items()},
        "per_fold": {str(fold): {group: _margin_summary([row for row in rows if int(row["fold"]) == fold])
                                  for group, rows in by_group.items()} for fold in folds},
    }
    contribution_fields = ("lr_base", "lr_context", "lr_bias", "lgbm_base", "lgbm_context", "lgbm_bias")
    contribution = {
        "pooled": {group: {field: _distribution(rows, field) for field in contribution_fields}
                   for group, rows in by_group.items()},
        "per_fold": {str(fold): {group: {field: _distribution([row for row in rows if int(row["fold"]) == fold], field)
                                           for field in contribution_fields} for group, rows in by_group.items()}
                     for fold in folds},
    }
    disagreement = {
        "pooled": {group: {"absolute_difference": _distribution(rows, "absolute_lr_lgbm_difference"),
                            "signed_lr_minus_lgbm": _distribution(rows, "lr_minus_lgbm"),
                            "flagged": sum(bool(row.get("disagreement_flag", abs(float(row["logistic_score"]) - float(row["lgbm_score"])) >= .20)) for row in rows)}
                   for group, rows in by_group.items()},
        "per_fold": {str(fold): {group: {"absolute_difference": _distribution([row for row in rows if int(row["fold"]) == fold], "absolute_lr_lgbm_difference"),
                                             "signed_lr_minus_lgbm": _distribution([row for row in rows if int(row["fold"]) == fold], "lr_minus_lgbm")}
                                  for group, rows in by_group.items()} for fold in folds},
    }
    mechanism_summary: dict[str, object] = {}
    for witness_name, key in (("best_iou", "best_iou_witnesses"), ("best_score", "best_score_witnesses")):
        pooled: Counter[str] = Counter()
        per_fold: dict[str, object] = {}
        denominator = 0
        for audit in audits:
            rows = [row for row in audit.get(key, ()) if isinstance(row, Mapping)]
            counts: Counter[str] = Counter()
            tp = [row for row in audit.get("tp_reference", ()) if isinstance(row, Mapping)]
            fp = [row for row in audit.get("fp_reference", ()) if isinstance(row, Mapping)]
            for row in rows:
                counts.update(_mechanisms(row, tp, fp))
            denominator += len(rows)
            pooled.update(counts)
            per_fold[str(audit.get("fold"))] = {"denominator": len(rows), "counts": dict(counts)}
        mechanism_summary[witness_name] = {"pooled": {"denominator": denominator, "counts": dict(pooled)}, "per_fold": per_fold}
    # Preserve pooled and per-fold feature distributions for both witness definitions.
    separability: dict[str, object] = {}
    for witness_name, key in (("best_iou", "best_iou_witnesses"), ("best_score", "best_score_witnesses")):
        feature_views: dict[str, object] = {}
        for feature in _FEATURE_COLUMNS:
            fn_rows = [row for audit in audits for row in audit.get(key, ()) if isinstance(row, Mapping)]
            fp_rows = [row for audit in audits for row in audit.get("fp_reference", ()) if isinstance(row, Mapping)]
            fn_median, fp_median = _quantile(fn_rows, feature, .5), _quantile(fp_rows, feature, .5)
            per_fold = {}
            signs: list[int] = []
            for audit in audits:
                fn = [row for row in audit.get(key, ()) if isinstance(row, Mapping)]
                fp = [row for row in audit.get("fp_reference", ()) if isinstance(row, Mapping)]
                left, right = _quantile(fn, feature, .5), _quantile(fp, feature, .5)
                sign = None if left is None or right is None else (1 if left > right else -1 if left < right else 0)
                if sign is not None:
                    signs.append(sign)
                per_fold[str(audit.get("fold"))] = {"fn": _distribution(fn, feature), "fp": _distribution(fp, feature),
                                                     "fn_minus_fp_median": None if sign is None else left - right, "sign": sign}
            feature_views[feature] = {"pooled": {"fn": _distribution(fn_rows, feature), "fp": _distribution(fp_rows, feature),
                                                   "fn_minus_fp_median": None if fn_median is None or fp_median is None else fn_median - fp_median},
                                      "per_fold": per_fold, "direction": "stable_descriptive" if max(signs.count(1), signs.count(-1)) >= 4 else "inconsistent_or_unsupported"}
        separability[witness_name] = feature_views
    return {"score_margin_analysis": margin_analysis, "lr_lgbm_disagreement": disagreement,
            "context_contribution": contribution, "mechanisms": mechanism_summary,
            "separability_by_witness": separability}
