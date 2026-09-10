from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.candidate_control import (
    CandidateAdmissionConfig,
    admit_candidates,
    select_candidate_admission,
    suppress_overlapping_candidates,
)
from src.pipeline.event_stack import CandidateEvent, EventRef, MultiScaleCandidate


def multi(sid: str, start: int, end: int, source: str) -> MultiScaleCandidate:
    evidence = CandidateEvent(
        EventRef(sid, start, end), (0.8,), 1.0, 0, 0, 1, 1
    )
    return MultiScaleCandidate(
        EventRef(sid, start, end),
        evidence if source in {"macro", "both"} else None,
        evidence if source in {"micro", "both"} else None,
    )


def geometry(candidates: tuple[MultiScaleCandidate, ...]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (row.event.sid, row.event.start_ms, row.event.end_ms)
        for row in candidates
    )


def test_same_session_nms_is_score_first_and_input_order_independent():
    candidates = (
        multi("s1", 0, 100, "micro"),
        multi("s1", 10, 110, "macro"),
        multi("s2", 0, 100, "micro"),
    )
    scores = np.array([0.8, 0.9, 0.7])
    expected = (("s1", 10, 110), ("s2", 0, 100))
    kept = suppress_overlapping_candidates(candidates, scores, 0.5)
    reversed_kept = suppress_overlapping_candidates(candidates[::-1], scores[::-1], 0.5)
    assert geometry(tuple(candidates[index] for index in kept)) == expected
    assert geometry(tuple(candidates[::-1][index] for index in reversed_kept)) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nms_iou": -0.1},
        {"nms_iou": 1.1},
        {"threshold": float("nan")},
        {"max_candidates_per_subject": 0},
        {"max_candidates_per_subject": True},
    ],
)
def test_admission_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CandidateAdmissionConfig(**kwargs)


def test_admission_budget_is_per_subject_not_per_session():
    candidates = (
        multi("s1", 0, 100, "micro"),
        multi("s2", 0, 100, "micro"),
        multi("s3", 0, 100, "micro"),
    )
    admitted = admit_candidates(
        candidates,
        np.array([0.9, 0.8, 0.7]),
        groups=np.array(["p1", "p1", "p2"]),
        config=CandidateAdmissionConfig(1.0, 0.0, 1),
    )
    assert [candidates[index].event.sid for index in admitted] == ["s1", "s3"]


def test_admission_rejects_misaligned_non_binary_inputs():
    candidates = (multi("s1", 0, 100, "micro"),)
    config = CandidateAdmissionConfig()
    with pytest.raises(ValueError):
        admit_candidates(candidates, np.array([[0.5]]), np.array(["p1"]), config)
    with pytest.raises(ValueError):
        admit_candidates(candidates, np.array([np.nan]), np.array(["p1"]), config)
    with pytest.raises(ValueError):
        select_candidate_admission(
            candidates,
            np.array([0.5]),
            np.array([2]),
            (EventRef("s1", 0, 100),),
            np.array(["p1"]),
            (config,),
        )


def test_selection_prefers_recall_floor_then_f1_then_lower_burden():
    candidates = (
        multi("s1", 0, 100, "micro"),
        multi("s1", 120, 220, "micro"),
        multi("s2", 0, 100, "micro"),
        multi("s3", 0, 100, "micro"),
    )
    scores = np.array([0.9, 0.35, 0.8, 0.7])
    labels = np.array([1, 0, 1, 1])
    truths = tuple(
        EventRef(candidates[index].event.sid, candidates[index].event.start_ms, candidates[index].event.end_ms)
        for index in (0, 2, 3)
    )
    configs = (
        CandidateAdmissionConfig(0.5, 0.3, 4),
        CandidateAdmissionConfig(0.5, 0.4, 4),
        CandidateAdmissionConfig(0.5, 0.0, None),
    )
    selection = select_candidate_admission(
        candidates, scores, labels, truths, np.array(["p1", "p1", "p2", "p3"]), configs
    )
    assert selection.config == CandidateAdmissionConfig(0.5, 0.4, 4)
    assert selection.candidate_recall >= 0.88
