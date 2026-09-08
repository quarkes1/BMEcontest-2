"""Deterministic subject-disjoint cross-fitting primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class GroupFold:
    """Row and subject membership for one group-disjoint fold."""

    index: int
    train_rows: np.ndarray
    validation_rows: np.ndarray
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]


@dataclass(frozen=True)
class OOFProbabilities:
    """Cross-fitted probabilities and their audit trail."""

    probabilities: np.ndarray
    score_counts: np.ndarray
    folds: tuple[GroupFold, ...]


def _one_dimensional_groups(groups: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(groups)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return values.astype(str)


def make_group_folds(groups: np.ndarray, n_splits: int) -> tuple[GroupFold, ...]:
    """Create deterministic folds in which a subject occurs on one side only."""

    group_values = _one_dimensional_groups(groups, "groups")
    unique_groups = np.unique(group_values)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(unique_groups):
        raise ValueError(
            f"n_splits={n_splits} exceeds the {len(unique_groups)} unique groups"
        )

    splitter = GroupKFold(n_splits=n_splits)
    folds: list[GroupFold] = []
    dummy_features = np.zeros((len(group_values), 1), dtype=np.int8)
    for index, (train_rows, validation_rows) in enumerate(
        splitter.split(dummy_features, groups=group_values)
    ):
        train_rows = np.asarray(train_rows, dtype=np.int64)
        validation_rows = np.asarray(validation_rows, dtype=np.int64)
        train_group_names = tuple(sorted(set(group_values[train_rows])))
        validation_group_names = tuple(sorted(set(group_values[validation_rows])))
        if set(train_group_names) & set(validation_group_names):
            raise RuntimeError(f"group leakage detected while building fold {index}")
        train_rows.setflags(write=False)
        validation_rows.setflags(write=False)
        folds.append(
            GroupFold(
                index=index,
                train_rows=train_rows,
                validation_rows=validation_rows,
                train_groups=train_group_names,
                validation_groups=validation_group_names,
            )
        )
    return tuple(folds)


def _positive_probabilities(estimator: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
        raise ValueError("predict_proba returned an invalid shape")
    classes = np.asarray(getattr(estimator, "classes_", ()))
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError("estimator classes_ must contain binary positive class 1")
    return probabilities[:, int(positive_columns[0])]


def crossfit_predict_proba(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    score_features: np.ndarray,
    score_groups: np.ndarray,
    n_splits: int,
    estimator_factory: Callable[[], Any],
) -> OOFProbabilities:
    """Score each requested row with a model that excluded its subject."""

    train_x = np.asarray(train_features)
    labels = np.asarray(train_labels)
    train_group_values = _one_dimensional_groups(train_groups, "train_groups")
    score_x = np.asarray(score_features)
    score_group_values = _one_dimensional_groups(score_groups, "score_groups")
    if train_x.ndim != 2 or score_x.ndim != 2:
        raise ValueError("train_features and score_features must be two-dimensional")
    if train_x.shape[1] != score_x.shape[1]:
        raise ValueError("train_features and score_features must have equal width")
    if labels.ndim != 1:
        raise ValueError("train_labels must be one-dimensional")
    if len(train_x) != len(labels) or len(train_x) != len(train_group_values):
        raise ValueError("training arrays must have equal row counts")
    if len(score_x) != len(score_group_values):
        raise ValueError("scoring arrays must have equal row counts")

    unknown_groups = sorted(set(score_group_values) - set(train_group_values))
    if unknown_groups:
        raise ValueError(
            "score groups absent from training groups: " + ", ".join(unknown_groups)
        )

    folds = make_group_folds(train_group_values, n_splits)
    probabilities = np.full(len(score_x), np.nan, dtype=np.float64)
    score_counts = np.zeros(len(score_x), dtype=np.int8)
    for fold in folds:
        validation_groups = np.asarray(fold.validation_groups)
        score_rows = np.flatnonzero(np.isin(score_group_values, validation_groups))
        if not len(score_rows):
            continue
        estimator = estimator_factory()
        estimator.fit(train_x[fold.train_rows], labels[fold.train_rows])
        fold_probabilities = _positive_probabilities(estimator, score_x[score_rows])
        if not np.isfinite(fold_probabilities).all():
            raise ValueError(f"fold {fold.index} produced non-finite probabilities")
        probabilities[score_rows] = fold_probabilities
        score_counts[score_rows] += 1

    if not np.all(score_counts == 1):
        invalid = np.flatnonzero(score_counts != 1)
        raise RuntimeError(
            f"cross-fit scoring invariant failed for {len(invalid)} rows"
        )
    probabilities.setflags(write=False)
    score_counts.setflags(write=False)
    return OOFProbabilities(probabilities, score_counts, folds)
