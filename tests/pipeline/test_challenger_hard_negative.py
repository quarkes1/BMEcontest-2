"""Frozen E03 hard-negative mining rules, exercised without a fitter."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.event_stack import EventRef, MultiScaleCandidate


def _candidate(sid: str, center: float) -> MultiScaleCandidate:
    """Build a hand-derived one-millisecond candidate centred at ``center``."""
    start = int(center - 0.5)
    return MultiScaleCandidate(EventRef(sid, start, start + 1), None, None)


def _evidence(
    *,
    scores: list[float],
    labels: list[int],
    macro: list[float] | None = None,
    micro: list[float] | None = None,
    centers: list[float] | None = None,
    sessions: list[str] | None = None,
):
    from src.pipeline.challenger.hard_negative import HardNegativeEvidence

    size = len(scores)
    values = np.zeros((size, 116), dtype=np.float64)
    values[:, 0] = 0.0 if macro is None else macro
    values[:, 41] = 0.0 if micro is None else micro
    centers = list(range(size)) if centers is None else centers
    sessions = ["session"] * size if sessions is None else sessions
    return HardNegativeEvidence(
        candidates=tuple(_candidate(sid, center) for sid, center in zip(sessions, centers, strict=True)),
        v1_features=values,
        labels=np.asarray(labels),
        oof_logistic_scores=np.asarray(scores),
        groups=np.asarray([f"subject-{index}" for index in range(size)], dtype=object),
    )


def test_negative_union_of_fixed_categories_is_weighted_once():
    """Adding category weights instead of one union weight would over-emphasise overlaps."""
    from src.pipeline.challenger.hard_negative import mine_hard_negatives

    result = mine_hard_negatives(_evidence(
        scores=[.50, .40, .90, .10, .10], labels=[0, 0, 0, 1, 0],
        macro=[.80, .10, .10, .80, .10], micro=[.10, .10, .10, .10, .10],
        centers=[0, 600_000, 1_200_000, 1_210_000, 1_215_000],
    ))

    assert result.sample_weight.tolist() == [2.0, 2.0, 2.0, 1.0, 2.0]
    assert result.hard_mask.tolist() == [True, True, True, False, True]


def test_dense_uses_same_session_inclusive_center_distance_and_two_others():
    """Counting self, another session, or a strict boundary would alter the frozen dense rule."""
    from src.pipeline.challenger.hard_negative import mine_hard_negatives

    result = mine_hard_negatives(_evidence(
        scores=[.10, .10, .10, .10], labels=[0, 0, 0, 0],
        centers=[0, 600_000, 1_200_000, 600_000],
        sessions=["a", "a", "a", "b"],
    ))

    assert result.category_mask["dense"].tolist() == [False, True, False, False]


def test_disagreement_uses_exact_v1_indices_and_missing_zero_semantics():
    """Changing either column index or treating missing evidence as non-hard changes E03."""
    from src.pipeline.challenger.hard_negative import mine_hard_negatives

    result = mine_hard_negatives(_evidence(
        scores=[.10, .10], labels=[0, 0], macro=[np.nan, .35], micro=[.35, np.nan],
    ))

    assert result.category_mask["disagreement"].tolist() == [True, True]


def test_miner_preserves_canonical_input_order_when_rows_are_permuted():
    """Sorting rows inside the miner would sever weights from canonical raw-union rows."""
    from src.pipeline.challenger.hard_negative import HardNegativeEvidence, mine_hard_negatives

    baseline = _evidence(
        scores=[.10, .70, .10], labels=[0, 0, 1], centers=[0, 1, 2],
        sessions=["a", "b", "c"],
    )
    order = np.array([2, 0, 1])
    shuffled = HardNegativeEvidence(
        candidates=tuple(baseline.candidates[index] for index in order),
        v1_features=baseline.v1_features[order], labels=baseline.labels[order],
        oof_logistic_scores=baseline.oof_logistic_scores[order], groups=baseline.groups[order],
    )

    assert mine_hard_negatives(shuffled).sample_weight.tolist() == [1.0, 1.0, 2.0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda item: (item.v1_features[:, :115], item.labels, item.oof_logistic_scores, item.groups), "116"),
        (lambda item: (item.v1_features, np.array([0, 2]), item.oof_logistic_scores, item.groups), "binary"),
        (lambda item: (item.v1_features, item.labels, np.array([.1, np.nan]), item.groups), "finite"),
        (lambda item: (item.v1_features, item.labels, item.oof_logistic_scores, np.array(["s0"])), "align"),
    ],
)
def test_miner_rejects_misaligned_or_nonfinite_training_evidence(mutate, message):
    """Accepting malformed rows could attach a hard weight to the wrong verifier sample."""
    from src.pipeline.challenger.hard_negative import HardNegativeEvidence, mine_hard_negatives

    original = _evidence(scores=[.1, .1], labels=[0, 0])
    features, labels, scores, groups = mutate(original)
    bad = HardNegativeEvidence(original.candidates, features, labels, scores, groups)

    with pytest.raises(ValueError, match=message):
        mine_hard_negatives(bad)


def test_miner_rejects_invalid_candidate_session_and_interval():
    """Session-local geometry requires a real session id and a finite positive interval."""
    from src.pipeline.challenger.hard_negative import HardNegativeEvidence, mine_hard_negatives

    original = _evidence(scores=[.1], labels=[0])
    invalid = MultiScaleCandidate(EventRef("", 3, 3), None, None)
    bad = HardNegativeEvidence((invalid,), original.v1_features, original.labels, original.oof_logistic_scores, original.groups)

    with pytest.raises(ValueError, match="candidate"):
        mine_hard_negatives(bad)


def test_outer_subject_intersection_is_rejected():
    """Mining with a held-out subject would violate E03's outer-train-only contract."""
    from src.pipeline.challenger.hard_negative import assert_outer_subject_disjoint

    with pytest.raises(ValueError, match="outer-validation"):
        assert_outer_subject_disjoint({"s1"}, {"s1", "s2"})


def test_descriptor_exposes_only_frozen_categories_and_constants():
    """A hidden tunable mining parameter would invalidate the preregistered experiment."""
    from src.pipeline.challenger.hard_negative import hard_negative_spec_descriptor

    descriptor = hard_negative_spec_descriptor()

    assert descriptor["categories"] == ("high_score", "near_threshold", "disagreement", "dense")
    assert descriptor["negative_hard_weight"] == 2.0
    assert descriptor["feature_width"] == 116
