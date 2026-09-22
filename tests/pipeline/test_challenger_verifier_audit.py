"""Pure E04 verifier-autopsy contracts."""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from src.pipeline.challenger.replay import MatchPair, ReplayTrace
from src.pipeline.event_stack import CandidateEvent, EventMetrics, EventRef, MultiScaleCandidate


def _candidate(sid: str, start: int, end: int) -> MultiScaleCandidate:
    event = CandidateEvent(EventRef(sid, start, end), (0.8,), 1.0, 0, 0, 1, 1)
    return MultiScaleCandidate(event.event, event, None)


def _trace() -> ReplayTrace:
    # Truth 0 is a matched TP, truth 1 is a C1 FN with two raw candidates,
    # and truth 2 is C0 and must not enter the reachable population.
    raw = (_candidate("tp", 0, 100), _candidate("fn", 0, 100),
           _candidate("fn", 0, 100), _candidate("fp", 200, 300))
    rows = tuple({
        "raw_index": index, "subject_id": f"s-{candidate.event.sid}",
        "session_id": candidate.event.sid, "start_ms": candidate.event.start_ms,
        "end_ms": candidate.event.end_ms, "logistic_score": (0.8, 0.1, 0.2, 0.7)[index],
        "lgbm_score": (0.7, 0.3, 0.2, 0.5)[index],
        "blend_score": (0.75, 0.2, 0.2, 0.6)[index], "terminal_reason": "selected",
        "final_event_index": (0, None, None, 1)[index],
    } for index, candidate in enumerate(raw))
    return ReplayTrace(
        fold=0, truths=(EventRef("tp", 0, 100), EventRef("fn", 0, 100), EventRef("c0", 0, 100)),
        truth_slices={}, raw_macro=tuple(item.macro for item in raw), raw_micro=(), raw_union=raw,
        candidate_rows=rows, admitted_indices=(0, 3), final_indices=(0, 3),
        predictions=(raw[0].event, raw[3].event), metrics=EventMetrics(1, 2, 3, 0.5, 1 / 3, 0.4),
        match_pairs=(MatchPair(0, 0, 1.0),), policy={"blend_weight": 0.5, "threshold": 0.4},
        verifier_features=np.arange(4 * 116, dtype=float).reshape(4, 116),
        verifier_models={"logistic": object(), "lgbm": object()},
    )


def test_audit_keeps_every_reachable_pair_and_deterministic_dual_witnesses(monkeypatch):
    """Dropping a tied raw pair would turn a GT-aware population into cherry-picking."""
    from src.pipeline.challenger import verifier_audit

    monkeypatch.setattr(verifier_audit, "explain_fitted_verifiers", lambda features, *_: {
        name: np.full(len(features), value) for name, value in {
            "lr_base": 1., "lr_context": -1., "lr_bias": 0., "lgbm_base": 1., "lgbm_context": -1., "lgbm_bias": 0.
        }.items()
    })
    result = verifier_audit.audit_verifier_trace(_trace())

    assert [row["raw_index"] for row in result["reachable_fn_candidates"]] == [1, 2]
    assert result["best_iou_witnesses"][0]["raw_index"] == 1
    assert result["best_score_witnesses"][0]["raw_index"] == 1
    assert len(result["tp_reference"]) == len(result["fp_reference"]) == 1
    row = result["reachable_fn_candidates"][0]
    assert row["base_00"] == 116.0 and row["base_55"] == 171.0
    assert row["margin"] == pytest.approx(-0.2)
    assert row["diagnostic_only_uses_gt"] is True


def test_summary_uses_fixed_bins_flags_and_fold_sign_rule():
    """Changing bin edges, flag thresholds, or four-fold sign support hides audit evidence."""
    from src.pipeline.challenger.verifier_audit import summarize_verifier_audits

    rows = [
        {"fold": fold, "group": "best_score_fn", "margin": margin, "logistic_score": .1,
         "lgbm_score": .35, "lr_context": -2., "lgbm_context": -2., "lr_base": 2., "lgbm_base": 2.,
         "base_00": value}
        for fold, (margin, value) in enumerate(zip([-.11, -.075, -.02, .02, .06], range(5), strict=True))
    ]
    audits = [{"fold": row["fold"], "analysis_rows": [row], "tp_reference": [], "fp_reference": [],
               "best_iou_witnesses": [], "best_score_witnesses": []} for row in rows]
    summary = summarize_verifier_audits(audits)

    assert summary["score_margin_analysis"]["pooled"]["best_score_fn"]["bins"]["deep_below"] == 1
    assert summary["score_margin_analysis"]["pooled"]["best_score_fn"]["bins"]["below_0_05_to_0"] == 1
    assert summary["lr_lgbm_disagreement"]["pooled"]["best_score_fn"]["flagged"] == 5
    assert summary["separability_by_witness"]["best_score"]["base_00"]["direction"] == "inconsistent_or_unsupported"


def test_tp_reference_keeps_official_truth_provenance_and_source_support(monkeypatch):
    """Dropping the official pair makes a TP reference indistinguishable from any selected event."""
    from src.pipeline.challenger import verifier_audit
    monkeypatch.setattr(verifier_audit, "explain_fitted_verifiers", lambda features, *_: {
        name: np.zeros(len(features)) for name in ("lr_base", "lr_context", "lr_bias", "lgbm_base", "lgbm_context", "lgbm_bias")
    })

    result = verifier_audit.audit_verifier_trace(_trace())

    tp = result["tp_reference"][0]
    assert (tp["matched_truth_index"], tp["matched_truth_key"], tp["matched_iou"]) == (0, "0:tp:0:100", 1.0)
    assert tp["macro_support"] is True and tp["micro_support"] is False and tp["macro_micro_support"] is False


def test_summary_separates_witness_denominators_and_has_no_mixed_unit_aliases():
    """Combining the two witness definitions doubles the same truth in a mechanism count."""
    from src.pipeline.challenger.verifier_audit import summarize_verifier_audits
    witness = {"fold": 0, "margin": -.02, "logistic_score": .1, "lgbm_score": .35,
               "lr_base": 1., "lr_context": -1., "lgbm_base": 1., "lgbm_context": -1., "base_00": 1.}
    audits = []
    for fold in range(5):
        row = dict(witness, fold=fold)
        audits.append({"fold": fold, "analysis_rows": [], "tp_reference": [], "fp_reference": [],
                       "best_iou_witnesses": [row], "best_score_witnesses": [row]})
    summary = summarize_verifier_audits(audits)

    assert summary["mechanisms"]["best_iou"]["pooled"]["denominator"] == 5
    assert summary["mechanisms"]["best_score"]["pooled"]["denominator"] == 5
    assert "margin_bins" not in summary and "signed_lr_minus_lgbm" not in summary
    assert "feature_separability" not in summary


def test_audit_rejects_nonofficial_pair_or_duplicate_final_ledger_coordinate(monkeypatch):
    """A stale match list or ambiguous final coordinate would corrupt TP/FP references."""
    from src.pipeline.challenger import verifier_audit
    monkeypatch.setattr(verifier_audit, "explain_fitted_verifiers", lambda features, *_: {
        name: np.zeros(len(features)) for name in ("lr_base", "lr_context", "lr_bias", "lgbm_base", "lgbm_context", "lgbm_bias")
    })
    trace = _trace()
    with pytest.raises(RuntimeError, match="official matcher"):
        verifier_audit.audit_verifier_trace(replace(trace, match_pairs=()))
    duplicated = [dict(row) for row in trace.candidate_rows]
    duplicated[3]["final_event_index"] = 0
    with pytest.raises(RuntimeError, match="duplicate final-event"):
        verifier_audit.audit_verifier_trace(replace(trace, candidate_rows=tuple(duplicated)))


def test_audit_rejects_nonfinite_weight_and_frozen_blend_identity_mismatch(monkeypatch):
    """A corrupt frozen blend chain must not be exported as a verifier explanation."""
    from src.pipeline.challenger import verifier_audit
    monkeypatch.setattr(verifier_audit, "explain_fitted_verifiers", lambda features, *_: {
        name: np.zeros(len(features)) for name in ("lr_base", "lr_context", "lr_bias", "lgbm_base", "lgbm_context", "lgbm_bias")
    })
    trace = _trace()
    with pytest.raises(RuntimeError, match="blend weight"):
        verifier_audit.audit_verifier_trace(replace(trace, policy={"blend_weight": float("nan"), "threshold": .4}))
    rows = [dict(row) for row in trace.candidate_rows]
    rows[0]["blend_score"] = .7500000001
    with pytest.raises(RuntimeError, match="blend score"):
        verifier_audit.audit_verifier_trace(replace(trace, candidate_rows=tuple(rows)))


def test_summary_requires_exactly_five_distinct_folds_and_four_fold_sign_gate():
    """A stable descriptive sign is meaningful only across the sealed five-fold population."""
    from src.pipeline.challenger.verifier_audit import summarize_verifier_audits
    with pytest.raises(ValueError, match="five distinct"):
        summarize_verifier_audits([{"fold": 0, "analysis_rows": [], "tp_reference": [], "fp_reference": [],
                                    "best_iou_witnesses": [], "best_score_witnesses": []}])
    audits = []
    for fold in range(5):
        fn = {"fold": fold, "base_00": 2.0, "margin": -.1, "logistic_score": .1, "lgbm_score": .1,
              "lr_base": 0., "lr_context": 0., "lgbm_base": 0., "lgbm_context": 0.}
        fp = {"fold": fold, "base_00": 1.0}
        audits.append({"fold": fold, "analysis_rows": [], "tp_reference": [fp], "fp_reference": [fp],
                       "best_iou_witnesses": [fn], "best_score_witnesses": [fn]})
    result = summarize_verifier_audits(audits)
    assert result["separability_by_witness"]["best_score"]["base_00"]["direction"] == "stable_descriptive"
