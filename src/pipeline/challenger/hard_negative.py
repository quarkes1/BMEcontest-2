"""Pure, frozen E03-v1 hard-negative sample weighting.

This module deliberately has no data-source, truth, fitter, or outer-fold
dependency.  The caller is responsible for constructing grouped inner-OOF
logistic scores from outer-train rows only; this module validates and preserves
that aligned raw-union row order while deriving the registered weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.pipeline.event_stack import MultiScaleCandidate


_FEATURE_WIDTH = 116
_HIGH_SCORE = 0.50
_NEAR_LOW = 0.40
_NEAR_HIGH = 0.60
_MACRO_MEAN_INDEX = 0
_MICRO_MEAN_INDEX = 41
_DISAGREEMENT = 0.35
_DENSE_RADIUS_MS = 600_000.0
_DENSE_OTHER_COUNT = 2
_HARD_NEGATIVE_WEIGHT = 2.0
_CATEGORIES = ("high_score", "near_threshold", "disagreement", "dense")


@dataclass(frozen=True)
class HardNegativeEvidence:
    """Aligned outer-train candidate evidence used by the E03 miner only."""

    candidates: Sequence[MultiScaleCandidate]
    v1_features: np.ndarray
    labels: np.ndarray
    oof_logistic_scores: np.ndarray
    groups: np.ndarray


@dataclass(frozen=True)
class HardNegativeResult:
    """One weight per input raw-union row plus auditable predicate outcomes."""

    sample_weight: np.ndarray
    category_mask: Mapping[str, np.ndarray]
    hard_mask: np.ndarray
    provenance: Mapping[str, tuple[object, ...]]


def hard_negative_spec_descriptor() -> dict[str, object]:
    """Return the complete fixed E03 mining contract (not a tuning surface)."""
    return {
        "version": "e03-v1-hard-negative",
        "feature_width": _FEATURE_WIDTH,
        "categories": _CATEGORIES,
        "high_score_min": _HIGH_SCORE,
        "near_threshold_inclusive": (_NEAR_LOW, _NEAR_HIGH),
        "disagreement_feature_indices": (_MACRO_MEAN_INDEX, _MICRO_MEAN_INDEX),
        "disagreement_abs_min": _DISAGREEMENT,
        "missing_feature_value": 0.0,
        "dense_center_radius_ms_inclusive": _DENSE_RADIUS_MS,
        "dense_other_candidate_count_min": _DENSE_OTHER_COUNT,
        "negative_hard_weight": _HARD_NEGATIVE_WEIGHT,
        "ordinary_weight": 1.0,
        "positive_weight": 1.0,
    }


def assert_outer_subject_disjoint(
    train_subjects: Iterable[str], outer_subjects: Iterable[str],
) -> None:
    """Fail closed if a mined training group belongs to outer validation."""
    train = set(train_subjects)
    outer = set(outer_subjects)
    overlap = train & outer
    if overlap:
        rendered = ", ".join(sorted(map(str, overlap)))
        raise ValueError(f"hard-negative training subjects overlap outer-validation: {rendered}")


def _as_vector(values: object, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != size:
        raise ValueError(f"{name} must be one-dimensional and align with candidates")
    return array


def _validate_candidates(candidates: Sequence[MultiScaleCandidate]) -> tuple[np.ndarray, np.ndarray]:
    sessions: list[str] = []
    centers: list[float] = []
    for candidate in candidates:
        event = getattr(candidate, "event", None)
        sid = getattr(event, "sid", None)
        if not isinstance(sid, str) or not sid:
            raise ValueError("candidate session ids must be non-empty strings")
        try:
            start = float(event.start_ms)
            end = float(event.end_ms)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("candidate intervals must be finite and positive") from exc
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("candidate intervals must be finite and positive")
        sessions.append(sid)
        centers.append((start + end) / 2.0)
    return np.asarray(sessions, dtype=object), np.asarray(centers, dtype=np.float64)


def _validate_features(values: object, *, size: int) -> np.ndarray:
    try:
        features = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("v1_features must be numeric") from exc
    if features.ndim != 2 or features.shape != (size, _FEATURE_WIDTH):
        raise ValueError(f"v1_features must align with candidates and have width {_FEATURE_WIDTH}")
    # Context-v1 deliberately represents unavailable session-local summaries,
    # comparisons and run geometry as NaN.  The frozen incumbent verifier
    # consumes that exact 116D surface with a fold-fitted median imputer; E03
    # must validate the same contract rather than narrowing it to its two
    # legacy mining reads.  Only those two reads use nan_to_num below.
    if np.isinf(features).any():
        raise ValueError("v1_features must be finite or NaN; +/-Inf is invalid")
    if np.isnan(features).all(axis=0).any():
        raise ValueError("v1_features must not contain all-missing columns")
    return features


def _validate_labels(values: object, *, size: int) -> np.ndarray:
    array = _as_vector(values, name="labels", size=size)
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be finite binary values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("labels must be finite binary values")
    return numeric


def _validate_scores(values: object, *, size: int) -> np.ndarray:
    array = _as_vector(values, name="oof_logistic_scores", size=size)
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("oof_logistic_scores must be finite probabilities") from exc
    if not np.isfinite(numeric).all() or np.any(numeric < 0.0) or np.any(numeric > 1.0):
        raise ValueError("oof_logistic_scores must be finite probabilities in [0, 1]")
    return numeric


def _validate_groups(values: object, *, size: int) -> np.ndarray:
    groups = _as_vector(values, name="groups", size=size).astype(object, copy=False)
    for group in groups:
        if group is None or (isinstance(group, (float, np.floating)) and not np.isfinite(group)):
            raise ValueError("groups must not contain missing values")
    return groups


def _dense_mask(sessions: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Mark rows with two other same-session centres in the inclusive radius."""
    dense = np.zeros(len(centers), dtype=bool)
    for session in sorted(set(sessions.tolist())):
        indices = np.flatnonzero(sessions == session)
        ordered = indices[np.argsort(centers[indices], kind="stable")]
        local_centers = centers[ordered]
        left = np.searchsorted(local_centers, local_centers - _DENSE_RADIUS_MS, side="left")
        right = np.searchsorted(local_centers, local_centers + _DENSE_RADIUS_MS, side="right")
        dense[ordered] = (right - left - 1) >= _DENSE_OTHER_COUNT
    return dense


def mine_hard_negatives(evidence: HardNegativeEvidence) -> HardNegativeResult:
    """Apply the exact fixed E03 category union to aligned candidate rows.

    The only label use is selecting negative candidates for weighting.  Inputs
    and outputs retain their given canonical raw-union order; no fitter or
    cross-fold data is observed here.
    """
    candidates = tuple(evidence.candidates)
    if not candidates:
        raise ValueError("candidates must be non-empty")
    sessions, centers = _validate_candidates(candidates)
    size = len(candidates)
    features = _validate_features(evidence.v1_features, size=size)
    labels = _validate_labels(evidence.labels, size=size)
    scores = _validate_scores(evidence.oof_logistic_scores, size=size)
    groups = _validate_groups(evidence.groups, size=size)

    negative = labels == 0.0
    macro_mean = np.nan_to_num(features[:, _MACRO_MEAN_INDEX], nan=0.0)
    micro_mean = np.nan_to_num(features[:, _MICRO_MEAN_INDEX], nan=0.0)
    categories = {
        "high_score": negative & (scores >= _HIGH_SCORE),
        "near_threshold": negative & (scores >= _NEAR_LOW) & (scores <= _NEAR_HIGH),
        "disagreement": negative & (np.abs(macro_mean - micro_mean) >= _DISAGREEMENT),
        "dense": negative & _dense_mask(sessions, centers),
    }
    hard = np.logical_or.reduce(tuple(categories.values()))
    weights = np.where(hard, _HARD_NEGATIVE_WEIGHT, 1.0).astype(np.float64)
    masks = MappingProxyType({name: mask.copy() for name, mask in categories.items()})
    provenance = MappingProxyType({
        "groups": tuple(groups.tolist()),
        "candidate_session_ids": tuple(sessions.tolist()),
    })
    return HardNegativeResult(weights, masks, hard.copy(), provenance)
