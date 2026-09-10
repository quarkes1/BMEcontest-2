"""Deterministic, train-only candidate admission primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.pipeline.event_stack import (
    CandidateEvent,
    EventMetrics,
    EventRef,
    MultiScaleCandidate,
    compute_event_metrics,
)
from src.eval.metrics import event_iou


@dataclass(frozen=True)
class CandidateAdmissionConfig:
    """One registered candidate-admission configuration."""

    nms_iou: float = 0.5
    threshold: float = 0.5
    max_candidates_per_subject: int | None = None

    def __post_init__(self) -> None:
        try:
            nms_iou = float(self.nms_iou)
            threshold = float(self.threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("nms_iou and threshold must be finite numbers") from exc
        if (
            isinstance(self.nms_iou, bool)
            or not np.isfinite(nms_iou)
            or not 0.0 < nms_iou <= 1.0
        ):
            raise ValueError("nms_iou must be finite and in (0, 1]")
        if (
            isinstance(self.threshold, bool)
            or not np.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("threshold must be finite and in [0, 1]")
        cap = self.max_candidates_per_subject
        if cap is not None and (
            isinstance(cap, bool) or not isinstance(cap, int) or cap < 1
        ):
            raise ValueError("max_candidates_per_subject must be a positive integer or None")


@dataclass(frozen=True)
class CandidateAdmissionSelection:
    """The train-only selected admission policy and its event diagnostics."""

    config: CandidateAdmissionConfig
    metrics: EventMetrics
    candidate_count: int
    candidate_recall: float


def _stream_key(candidate: CandidateEvent | None) -> tuple[object, ...]:
    if candidate is None:
        return (0,)
    return (
        1,
        candidate.event.sid,
        candidate.event.start_ms,
        candidate.event.end_ms,
        candidate.probabilities,
        candidate.observed_fraction,
        candidate.bridged_gap_count,
        candidate.bridged_gap_ms,
        candidate.pre_observed_count,
        candidate.post_observed_count,
    )


def _multiscale_key(candidate: MultiScaleCandidate) -> tuple[object, ...]:
    return (
        candidate.event.sid,
        candidate.event.start_ms,
        candidate.event.end_ms,
        _stream_key(candidate.macro),
        _stream_key(candidate.micro),
    )


def _validate_iou_threshold(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("iou threshold must be finite and in (0, 1]") from exc
    if isinstance(value, bool) or not np.isfinite(numeric) or not 0.0 < numeric <= 1.0:
        raise ValueError("iou threshold must be finite and in (0, 1]")
    return numeric


def _validate_scores(
    candidates: Sequence[MultiScaleCandidate], scores: np.ndarray,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(candidates):
        raise ValueError("scores must be one-dimensional and align with candidates")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    return values


def _validate_groups(
    candidates: Sequence[MultiScaleCandidate], groups: Sequence[object],
) -> np.ndarray:
    values = np.asarray(groups, dtype=object)
    if values.ndim != 1 or len(values) != len(candidates):
        raise ValueError("groups must be one-dimensional and align with candidates")
    for value in values:
        if value is None:
            raise ValueError("groups must not contain missing values")
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError("groups must contain finite values")
    return values


def _validate_labels(
    candidates: Sequence[MultiScaleCandidate], labels: Sequence[object],
) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != len(candidates):
        raise ValueError("labels must be one-dimensional and align with candidates")
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be finite binary values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("labels must be finite binary values")
    return numeric


def suppress_overlapping_candidates(
    candidates: Sequence[MultiScaleCandidate],
    scores: np.ndarray,
    iou_threshold: float,
) -> tuple[int, ...]:
    """Return original row indices after stable same-session greedy NMS."""

    threshold = _validate_iou_threshold(iou_threshold)
    values = _validate_scores(candidates, scores)
    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (-values[index], _multiscale_key(candidates[index])),
    )
    kept: list[int] = []
    for index in ranked:
        event = candidates[index].event
        if all(
            event.sid != candidates[other].event.sid
            or event_iou(event.interval, candidates[other].event.interval) < threshold
            for other in kept
        ):
            kept.append(index)
    return tuple(
        sorted(
            kept,
            key=lambda index: (
                candidates[index].event.sid,
                candidates[index].event.start_ms,
                candidates[index].event.end_ms,
                _multiscale_key(candidates[index]),
            ),
        )
    )


def admit_candidates(
    candidates: Sequence[MultiScaleCandidate],
    scores: np.ndarray,
    groups: Sequence[object],
    config: CandidateAdmissionConfig,
) -> tuple[int, ...]:
    """Apply NMS, score threshold, and optional per-subject budget."""

    if not isinstance(config, CandidateAdmissionConfig):
        raise TypeError("config must be a CandidateAdmissionConfig")
    values = _validate_scores(candidates, scores)
    group_values = _validate_groups(candidates, groups)
    nms_indices = suppress_overlapping_candidates(candidates, values, config.nms_iou)
    eligible = [index for index in nms_indices if values[index] >= config.threshold]

    if config.max_candidates_per_subject is None:
        selected = eligible
    else:
        selected = []
        for group in sorted({group_values[index] for index in eligible}, key=str):
            group_indices = [index for index in eligible if group_values[index] == group]
            group_indices.sort(
                key=lambda index: (-values[index], _multiscale_key(candidates[index]))
            )
            selected.extend(group_indices[: config.max_candidates_per_subject])

    return tuple(
        sorted(
            selected,
            key=lambda index: (
                candidates[index].event.sid,
                candidates[index].event.start_ms,
                candidates[index].event.end_ms,
                _multiscale_key(candidates[index]),
            ),
        )
    )


def _config_sort_key(config: CandidateAdmissionConfig) -> tuple[float, float, float]:
    cap_key = float("inf") if config.max_candidates_per_subject is None else float(
        config.max_candidates_per_subject
    )
    return config.nms_iou, config.threshold, cap_key


def select_candidate_admission(
    candidates: Sequence[MultiScaleCandidate],
    scores: np.ndarray,
    labels: Sequence[object],
    truths: Sequence[EventRef],
    groups: Sequence[object],
    configs: Sequence[CandidateAdmissionConfig],
    minimum_recall: float = 0.88,
    iou_threshold: float = 0.25,
) -> CandidateAdmissionSelection:
    """Select one policy using only the supplied train candidates and truths."""

    values = _validate_scores(candidates, scores)
    _validate_labels(candidates, labels)
    _validate_groups(candidates, groups)
    metric_iou = _validate_iou_threshold(iou_threshold)
    try:
        minimum_recall_value = float(minimum_recall)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_recall must be finite and in [0, 1]") from exc
    if (
        isinstance(minimum_recall, bool)
        or not np.isfinite(minimum_recall_value)
        or not 0.0 <= minimum_recall_value <= 1.0
    ):
        raise ValueError("minimum_recall must be finite and in [0, 1]")
    options = tuple(configs)
    if not options:
        raise ValueError("configs must not be empty")
    if any(not isinstance(config, CandidateAdmissionConfig) for config in options):
        raise TypeError("configs must contain CandidateAdmissionConfig values")

    canonical_configs = sorted(enumerate(options), key=lambda item: (_config_sort_key(item[1]), item[0]))
    evaluated: list[tuple[int, CandidateAdmissionSelection]] = []
    for canonical_rank, (_, config) in enumerate(canonical_configs):
        admitted = admit_candidates(candidates, values, groups, config)
        predictions = tuple(candidates[index].event for index in admitted)
        metrics = compute_event_metrics(predictions, truths, metric_iou)
        selection = CandidateAdmissionSelection(
            config=config,
            metrics=metrics,
            candidate_count=len(admitted),
            candidate_recall=metrics.sensitivity,
        )
        evaluated.append((canonical_rank, selection))

    feasible = [
        item for item in evaluated if item[1].candidate_recall >= minimum_recall_value
    ]
    pool = feasible if feasible else evaluated
    truth_count = max(1, len(truths))

    def rank(item: tuple[int, CandidateAdmissionSelection]) -> tuple[float, ...]:
        canonical_rank, selection = item
        burden = selection.candidate_count / truth_count
        base = (
            selection.metrics.f1,
            selection.metrics.ppv,
            -burden,
            -float(canonical_rank),
        )
        return (selection.candidate_recall, *base) if not feasible else base

    return max(pool, key=rank)[1]
