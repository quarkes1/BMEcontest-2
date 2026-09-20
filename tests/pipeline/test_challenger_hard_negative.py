"""Frozen E03 hard-negative mining rules, exercised without a fitter."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.event_stack import EventMetrics, EventRef, EventSelectionPolicy, MultiScaleCandidate


class _WeightRecordingEstimator:
    """Small deterministic estimator used to observe the cross-fit boundary."""

    fit_weights: list[np.ndarray | None] = []
    fit_groups: list[np.ndarray] = []

    @classmethod
    def reset(cls) -> None:
        cls.fit_weights = []
        cls.fit_groups = []

    def fit(self, features, labels, **kwargs):
        weights = kwargs.get("model__sample_weight")
        type(self).fit_weights.append(None if weights is None else np.asarray(weights).copy())
        type(self).fit_groups.append(np.asarray(features)[:, 0].copy())
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, features):
        return np.tile(np.asarray([[0.75, 0.25]]), (len(features), 1))


def _crossfit_weight_fixture():
    """Four subject rows with feature zero encoding the hand-checked row id."""
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    labels = np.asarray([0, 1, 0, 1])
    groups = np.asarray(["a", "b", "c", "d"])
    return features, labels, groups


def test_weighted_crossfit_slices_only_training_weights_per_subject_fold():
    """Passing a whole weight vector to a held-out fit would leak validation-row weights."""
    from src.pipeline.crossfit import crossfit_predict_proba

    features, labels, groups = _crossfit_weight_fixture()
    _WeightRecordingEstimator.reset()
    result = crossfit_predict_proba(
        features, labels, groups, features, groups, 2, _WeightRecordingEstimator,
        fit_sample_weight=np.asarray([1.0, 2.0, 3.0, 4.0]),
    )

    assert result.probabilities.tolist() == [0.25, 0.25, 0.25, 0.25]
    assert sorted(tuple(weights.tolist()) for weights in _WeightRecordingEstimator.fit_weights) == [
        (1.0, 3.0), (2.0, 4.0),
    ]
    assert all(len(weights) == 2 for weights in _WeightRecordingEstimator.fit_weights)
    for fold, fitted_row_ids in zip(result.folds, _WeightRecordingEstimator.fit_groups, strict=True):
        assert set(fitted_row_ids) == set(features[fold.train_rows, 0])
        assert set(groups[fold.train_rows]).isdisjoint(groups[fold.validation_rows])


def test_unweighted_crossfit_preserves_the_legacy_fit_call_shape():
    """Adding a sample-weight keyword to normal runs would alter the frozen incumbent path."""
    from src.pipeline.crossfit import crossfit_predict_proba

    features, labels, groups = _crossfit_weight_fixture()
    _WeightRecordingEstimator.reset()
    crossfit_predict_proba(features, labels, groups, features, groups, 2, _WeightRecordingEstimator)

    assert _WeightRecordingEstimator.fit_weights == [None, None]


@pytest.mark.parametrize(
    "weights",
    [
        np.asarray([]),
        np.asarray([1.0, np.nan, 1.0, 1.0]),
        np.asarray([1.0, 0.0, 1.0, 1.0]),
        np.asarray([1, 2, 1, 2], dtype=np.int64),
    ],
)
def test_crossfit_rejects_bad_verifier_weights_before_any_model_is_fit(weights):
    """Malformed weights must fail before they can silently misalign a verifier fit."""
    from src.pipeline.crossfit import crossfit_predict_proba

    features, labels, groups = _crossfit_weight_fixture()
    _WeightRecordingEstimator.reset()
    with pytest.raises(ValueError, match="sample weight"):
        crossfit_predict_proba(
            features, labels, groups, features, groups, 2, _WeightRecordingEstimator,
            fit_sample_weight=weights,
        )
    assert _WeightRecordingEstimator.fit_weights == []


def test_unit_weight_logistic_crossfit_matches_unweighted_incumbent_scores():
    """A vector of ones is semantically the incumbent verifier, not a new arm."""
    from src.pipeline.crossfit import crossfit_predict_proba
    from src.pipeline.runner import _verifier_estimator

    features = np.asarray([
        [0.0, 1.0], [1.0, 0.0], [0.1, 1.1], [1.1, 0.1],
        [0.2, 1.2], [1.2, 0.2], [0.3, 1.3], [1.3, 0.3],
    ])
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.asarray(["a", "b", "c", "d", "e", "f", "g", "h"])
    factory = lambda: _verifier_estimator(20260911, 0.1)

    unweighted = crossfit_predict_proba(features, labels, groups, features, groups, 4, factory)
    unit_weighted = crossfit_predict_proba(
        features, labels, groups, features, groups, 4, factory,
        fit_sample_weight=np.ones(len(labels)),
    )

    np.testing.assert_allclose(unit_weighted.probabilities, unweighted.probabilities, rtol=0.0, atol=1e-12)


def test_final_weighted_fit_supplies_the_same_vector_to_both_verifiers(monkeypatch):
    """Weighting only one ensemble member would make E03 a different, unregistered model."""
    from src.pipeline import runner

    seen: dict[str, list[np.ndarray]] = {"logistic": [], "lgbm": []}

    class _FinalProbe:
        def __init__(self, name):
            self.name = name

        def fit(self, features, labels, **kwargs):
            seen[self.name].append(np.asarray(kwargs["model__sample_weight"]).copy())
            return self

    monkeypatch.setattr(runner, "_verifier_estimator", lambda *_: _FinalProbe("logistic"))
    monkeypatch.setattr(runner, "_verifier_lgbm_estimator", lambda *_: _FinalProbe("lgbm"))
    config = runner.RunConfig(outer_fold=0, micro_enabled=True, candidate_control_enabled=True)
    features = np.asarray([[0.0] * 116, [1.0] * 116, [0.5] * 116, [0.25] * 116])
    weights = np.asarray([1.0, 2.0, 1.0, 2.0])

    runner._fit_final_models(
        config,
        train_features=np.asarray([[0.0], [1.0], [0.1], [0.9]]),
        train_labels=np.asarray([0, 1, 0, 1]),
        micro_train_features=np.asarray([[0.0] * 47, [1.0] * 47, [0.1] * 47, [0.9] * 47]),
        micro_train_labels=np.asarray([0, 1, 0, 1]),
        train_candidate_features=features,
        train_candidate_labels=np.asarray([0, 1, 0, 1]),
        verifier_c=0.1,
        policy=EventSelectionPolicy(0.5, None, EventMetrics(0, 0, 0, 0.0, 0.0, 0.0)),
        admission=None,
        blend_weight=0.5,
        verifier_sample_weight=weights,
    )

    assert len(seen["logistic"]) == len(seen["lgbm"]) == 1
    np.testing.assert_array_equal(seen["logistic"][0], weights)
    np.testing.assert_array_equal(seen["lgbm"][0], weights)


@pytest.mark.parametrize("factory_args", [(20260911, 0.1), (20260911,)])
def test_weighted_verifier_factories_keep_balanced_class_weight(factory_args):
    """Sample weights supplement rather than replace the frozen balanced class weighting."""
    from src.pipeline.runner import _verifier_estimator, _verifier_lgbm_estimator

    factory = _verifier_estimator if len(factory_args) == 2 else _verifier_lgbm_estimator
    pipeline = factory(*factory_args)
    features = np.asarray([[0.0] * 116, [1.0] * 116, [0.1] * 116, [0.9] * 116])
    pipeline.fit(features, np.asarray([0, 1, 0, 1]), model__sample_weight=np.asarray([1.0, 2.0, 1.0, 2.0]))

    assert pipeline.named_steps["model"].get_params()["class_weight"] == "balanced"


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
