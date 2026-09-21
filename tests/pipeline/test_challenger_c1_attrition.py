"""Pure frozen-policy contracts for Amendment A1 E03.5."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.challenger.replay import ReplayTrace, official_match_indices
from src.pipeline.event_stack import CandidateEvent, EventMetrics, EventRef, MultiScaleCandidate


def _candidate(sid: str, start: int, end: int) -> MultiScaleCandidate:
    event = EventRef(sid, start, end)
    return MultiScaleCandidate(event, CandidateEvent(event, (0.8,), 1.0, 0, 0, 1, 1), None)


def _trace() -> ReplayTrace:
    raw = tuple(_candidate(*values) for values in (
        ("n", 0, 100), ("n", 50, 150), ("t", 0, 100),
        ("cap-a", 0, 100), ("cap-b", 0, 100), ("cap", 0, 100),
    ))
    scores = (.9, .8, .4, .99, .98, .7)
    reasons = ("selected", "nms_suppressed", "below_admission_threshold", "selected", "selected", "subject_cap_excluded")
    rows = tuple({
        "raw_index": index, "blend_score": score, "logistic_score": score,
        "lgbm_score": score, "terminal_reason": reasons[index],
        "nms_suppressed_by_raw_index": 0 if index == 1 else None,
        "admission_threshold_pass": score >= .5, "subject_cap_rank": index,
        "admitted_index": (0, None, None, 1, 2, None)[index],
        "final_event_index": (0, None, None, 1, 2, None)[index],
    } for index, score in enumerate(scores))
    truths = (EventRef("n", 100, 200), EventRef("t", 0, 100), EventRef("cap", 0, 100))
    predictions = tuple(raw[index].event for index in (0, 3, 4))
    return ReplayTrace(
        fold=0, truths=truths, truth_slices={}, raw_macro=(), raw_micro=(), raw_union=raw,
        candidate_rows=rows, admitted_indices=(3, 4, 0), final_indices=(3, 4, 0),
        predictions=predictions, metrics=EventMetrics(0, 3, 3, 0., 0., 0.),
        match_pairs=official_match_indices(predictions, truths),
        subject_by_session={"n": "n", "t": "t", "cap-a": "cap", "cap-b": "cap", "cap": "cap"},
        policy={"nms_iou": .3, "admission_threshold": .5, "max_candidates_per_subject": 2,
                "threshold": .5, "max_events_per_group": None},
    )


def test_audit_ablates_exactly_one_frozen_admission_operation():
    """Removing a wrong stage must not implicitly relax either other stage."""
    from src.pipeline.challenger.c1_attrition import audit_trace

    audit = audit_trace(_trace())

    assert audit["baseline_admitted_indices"] == (3, 4, 0)
    assert audit["ablations"]["disable_nms_only"]["admitted_indices"] == (3, 4, 0, 1)
    assert audit["ablations"]["disable_admission_threshold_only"]["admitted_indices"] == (3, 4, 0, 2)
    assert audit["ablations"]["disable_subject_cap_only"]["admitted_indices"] == (5, 3, 4, 0)
    assert audit["ablations"]["disable_nms_only"]["newly_matched_c1_truth_keys"] == ("0:n:100:200",)
    assert audit["ablations"]["disable_admission_threshold_only"]["newly_matched_c1_truth_keys"] == ()


def test_audit_keeps_scores_frozen_and_records_truth_attrition_and_oracle_reset():
    """Re-scoring, truth leakage between oracle runs, or forced threshold bypass breaks this audit."""
    from src.pipeline.challenger.c1_attrition import audit_trace

    trace = _trace()
    audit = audit_trace(trace)

    assert np.array_equal(audit["frozen_blend_scores"], np.asarray([.9, .8, .4, .99, .98, .7]))
    causes = {row["truth_key"]: row["first_total_loss_stage"] for row in audit["truth_rows"]}
    assert causes == {"0:n:100:200": "nms", "0:t:0:100": "admission_threshold", "0:cap:0:100": "subject_cap"}
    oracle = {row["truth_key"]: row for row in audit["oracle_rows"]}
    assert oracle["0:n:100:200"]["outcome"] == "control_limited_and_downstream_capable"
    assert oracle["0:t:0:100"]["outcome"] == "downstream_still_rejects"
    assert oracle["0:cap:0:100"]["outcome"] == "control_limited_and_downstream_capable"


def test_audit_counts_unmatched_final_events_with_official_matcher_and_nonmatching_candidates():
    """Replacing matcher-based FP differences with candidate labels would hide actual event cost."""
    from src.pipeline.challenger.c1_attrition import audit_trace

    audit = audit_trace(_trace())

    nms = audit["ablations"]["disable_nms_only"]
    assert nms["extra_nonmatching_candidates"] == 0
    assert nms["gross_introduced_final_fp"] == 0
    assert nms["net_final_fp_difference"] == 0
    assert nms["event_diffs"] == ()


def test_audit_reports_only_short_c1_targets_and_all_counterfactual_effect_sets():
    """Ablation summaries must not silently omit downstream losses or non-C1 collateral."""
    from src.pipeline.challenger.c1_attrition import audit_trace

    audit = audit_trace(_trace())

    assert audit["target_truth_keys"] == ("0:cap:0:100", "0:n:100:200", "0:t:0:100")
    threshold = audit["ablations"]["disable_admission_threshold_only"]
    assert threshold["newly_reachable_c1_truth_keys"] == ("0:t:0:100",)
    assert threshold["c1_truth_keys_still_lost_downstream"] == ("0:t:0:100",)
    assert threshold["newly_matched_non_c1_truth_keys"] == ()
    assert threshold["lost_non_c1_truth_keys"] == ()


def test_oracle_records_a_downstream_rejection_reason():
    """An unsuccessful GT-aware force must identify the frozen later control that rejects it."""
    from src.pipeline.challenger.c1_attrition import audit_trace

    oracle = {row["truth_key"]: row for row in audit_trace(_trace())["oracle_rows"]}

    assert oracle["0:t:0:100"]["downstream_rejection_reason"] == "below_event_threshold"


def test_replay_trace_policy_is_an_immutable_snapshot():
    """Mutating a trace policy after construction must not rewrite frozen audit evidence."""
    trace = _trace()

    with pytest.raises(TypeError):
        trace.policy["threshold"] = .1  # type: ignore[index]
