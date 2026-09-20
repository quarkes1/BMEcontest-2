"""Deterministic, label-free reporting contracts for the E01 Error Atlas."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.pipeline.challenger.replay import MatchPair, ReplayTrace
from src.pipeline.event_stack import CandidateEvent, EventMetrics, EventRef, MultiScaleCandidate


def _candidate(sid: str, start: int, end: int) -> CandidateEvent:
    return CandidateEvent(EventRef(sid, start, end), (0.8,), 1.0, 0, 0, 1, 1)


def _row(index: int, candidate: MultiScaleCandidate, reason: str, final_index: int | None) -> dict[str, object]:
    return {
        "raw_index": index, "session_id": candidate.event.sid,
        "start_ms": candidate.event.start_ms, "end_ms": candidate.event.end_ms,
        "macro_raw_index": index if candidate.macro else None,
        "micro_raw_index": index if candidate.micro else None,
        "nms_rank": index, "nms_suppressed_by_raw_index": None,
        "logistic_score": 0.7 + index / 100, "lgbm_score": 0.6 + index / 100,
        "blend_score": 0.65 + index / 100, "admission_threshold_pass": reason != "below_admission_threshold",
        "subject_cap_rank": index, "admitted_index": index if reason not in {"below_admission_threshold", "nms_suppressed", "subject_cap_excluded"} else None,
        "event_threshold_pass": reason not in {"below_event_threshold", "below_admission_threshold", "nms_suppressed", "subject_cap_excluded"},
        "event_cap_rank": index if final_index is not None else None,
        "final_event_index": final_index, "terminal_reason": reason,
        "ctx": float(index),
    }


@pytest.fixture
def synthetic_trace() -> ReplayTrace:
    """One hand-derived example for every mutually-exclusive FN stage."""
    # C1 has a raw candidate, C2 has an admitted candidate, C3 has a final
    # partial-overlap event, and C4's one final event is greedily consumed by
    # the first of two identical truths.
    raw = (
        MultiScaleCandidate(EventRef("c1", 0, 100), _candidate("c1", 0, 100), None),
        MultiScaleCandidate(EventRef("c2", 100, 200), None, _candidate("c2", 100, 200)),
        MultiScaleCandidate(EventRef("c3", 300, 320), _candidate("c3", 300, 320), None),
        MultiScaleCandidate(EventRef("c4", 400, 500), _candidate("c4", 400, 500), _candidate("c4", 400, 500)),
        MultiScaleCandidate(EventRef("fp", 700, 800), _candidate("fp", 700, 800), None),
    )
    truths = (
        EventRef("c0", 0, 100), EventRef("c1", 0, 100), EventRef("c2", 100, 200),
        EventRef("c3", 300, 400), EventRef("c4", 400, 500), EventRef("c4", 400, 500),
    )
    # final #0 matches first c4 truth; #1 is unrelated FP; #2 produces C3.
    predictions = (raw[3].event, raw[4].event, raw[2].event)
    rows = (
        _row(0, raw[0], "below_admission_threshold", None),
        _row(1, raw[1], "below_event_threshold", None),
        _row(2, raw[2], "selected", 2),
        _row(3, raw[3], "selected", 0),
        _row(4, raw[4], "selected", 1),
    )
    return ReplayTrace(
        fold=0, truths=truths, truth_slices={}, raw_macro=(raw[0].macro, raw[2].macro, raw[3].macro, raw[4].macro), raw_micro=(raw[1].micro, raw[3].micro), raw_union=raw,
        candidate_rows=rows, admitted_indices=(1, 2, 3, 4), final_indices=(3, 4, 2),
        predictions=predictions, metrics=EventMetrics(1, 3, 6, 1 / 6, 1 / 3, 2 / 9),
        match_pairs=(MatchPair(0, 4, 1.0),),
    )


def test_fn_taxonomy_is_total_and_exclusive(synthetic_trace: ReplayTrace):
    """Dropping a stage branch must leave a real official FN unclassified."""
    from src.pipeline.challenger.atlas import classify_false_negatives

    rows = classify_false_negatives(synthetic_trace)

    assert {row["fn_stage"] for row in rows} == {"FN-C0", "FN-C1", "FN-C2", "FN-C3", "FN-C4"}
    assert len({row["truth_key"] for row in rows}) == 5


def test_c4_records_the_greedy_consuming_pair(synthetic_trace: ReplayTrace):
    """Treating any final reachability as a TP would hide matcher competition."""
    from src.pipeline.challenger.atlas import classify_false_negatives

    c4 = next(row for row in classify_false_negatives(synthetic_trace) if row["fn_stage"] == "FN-C4")

    assert c4["consuming_final_event_index"] == 0
    assert c4["consuming_truth_key"] == "0:c4:400:500"
    assert c4["consuming_pair_iou"] == 1.0


def test_fn_rows_keep_session_subject_provenance_when_no_candidate_exists(synthetic_trace: ReplayTrace):
    """A C0 miss must not lose its subject merely because it has no raw row."""
    from src.pipeline.challenger.atlas import classify_false_negatives

    traced = replace(synthetic_trace, subject_by_session={"c0": "subject-c0"})

    c0 = next(row for row in classify_false_negatives(traced) if row["fn_stage"] == "FN-C0")
    assert c0["subject_id"] == "subject-c0"


def test_fn_rows_expose_pre_union_scale_reachability(synthetic_trace: ReplayTrace):
    """Collapsing macro/micro reachability into union-only hides candidate mechanism."""
    from src.pipeline.challenger.atlas import classify_false_negatives

    rows = {row["session_id"]: row for row in classify_false_negatives(synthetic_trace)}

    assert rows["c1"]["macro_raw_reachable"] is True
    assert rows["c1"]["micro_raw_reachable"] is False
    assert rows["c2"]["macro_raw_reachable"] is False
    assert rows["c2"]["micro_raw_reachable"] is True
    assert rows["c0"]["union_raw_reachable"] is False


def test_fp_rows_are_exactly_unmatched_final_predictions(synthetic_trace: ReplayTrace):
    """Changing unmatched-final filtering would leak non-final candidate errors into FP."""
    from src.pipeline.challenger.atlas import false_positive_rows

    assert [row["final_index"] for row in false_positive_rows(synthetic_trace)] == [1, 2]


def test_report_has_required_error_atlas_sections(synthetic_trace: ReplayTrace):
    """A report missing a required heading would make E01 evidence incomplete."""
    from src.pipeline.challenger.atlas import atlas_summaries, render_report

    report = render_report(atlas_summaries([synthetic_trace]))

    for heading in (
        "Incumbent metrics", "FN stage decomposition", "FP taxonomy",
        "per-fold error profile", "Context-v1 residual separation",
    ):
        assert heading in report


def test_five_fold_incumbent_validation_rejects_wrong_frozen_totals(synthetic_trace: ReplayTrace):
    """Changing the frozen five-fold error accounting must fail closed."""
    from src.pipeline.challenger.atlas import atlas_summaries

    traces = [replace(synthetic_trace, fold=fold) for fold in range(5)]
    with pytest.raises(RuntimeError, match="frozen incumbent atlas totals"):
        atlas_summaries(traces)


@pytest.mark.integration
def test_frozen_five_fold_trace_has_complete_official_error_accounting():
    """An atlas regression must not silently alter the sealed 39 FN / 83 FP universe."""
    from src.pipeline.challenger.atlas import atlas_summaries, classify_false_negatives, false_positive_rows
    from src.pipeline.challenger.replay import replay_outer_fold

    traces = [replay_outer_fold(Path("."), fold) for fold in range(5)]
    fn_rows = [row for trace in traces for row in classify_false_negatives(trace)]
    fp_rows = [row for trace in traces for row in false_positive_rows(trace)]
    summary = atlas_summaries(traces)

    assert len({row["truth_key"] for row in fn_rows}) == len(fn_rows) == 39
    assert sum(row["duration_slice"] == "short" for row in fn_rows) == 19
    assert len(fp_rows) == 83
    assert summary["incumbent_metrics"] == {"tp": 114, "eligible": 153, "pred": 197, "fp": 83, "fn": 39}
