import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.pipeline.crossfit import crossfit_predict_proba, make_group_folds


def test_group_folds_never_split_subjects():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])

    for fold in make_group_folds(groups, n_splits=2):
        assert set(groups[fold.train_rows]).isdisjoint(
            set(groups[fold.validation_rows])
        )


def test_crossfit_scores_every_requested_row_once():
    features = np.arange(16, dtype=float).reshape(8, 2)
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])

    result = crossfit_predict_proba(
        features,
        labels,
        groups,
        features,
        groups,
        n_splits=2,
        estimator_factory=lambda: LogisticRegression(random_state=42),
    )

    assert result.probabilities.shape == (8,)
    assert np.isfinite(result.probabilities).all()
    assert (result.score_counts == 1).all()


def test_crossfit_rejects_unknown_score_group():
    with pytest.raises(ValueError, match="score groups absent from training groups"):
        crossfit_predict_proba(
            np.ones((4, 1)),
            np.array([0, 1, 0, 1]),
            np.array(["a", "a", "b", "b"]),
            np.ones((1, 1)),
            np.array(["missing"]),
            2,
            lambda: LogisticRegression(),
        )
