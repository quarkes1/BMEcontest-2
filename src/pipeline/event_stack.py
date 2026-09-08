"""Pure event-level primitives shared by training and evaluation commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.eval.metrics import event_iou


@dataclass(frozen=True)
class EventRef:
    """One event on one sensor session's millisecond time axis."""

    sid: str
    start_ms: int
    end_ms: int

    @property
    def interval(self) -> tuple[int, int]:
        return self.start_ms, self.end_ms


@dataclass(frozen=True)
class EventMetrics:
    """Global event counts and official sensitivity/PPV/F1 ratios."""

    n_tp: int
    n_pred: int
    n_true: int
    sensitivity: float
    ppv: float
    f1: float


@dataclass(frozen=True)
class ThresholdSelection:
    """The deterministic threshold choice and its event-level metrics."""

    threshold: float
    metrics: EventMetrics


def compute_event_metrics(
    predictions: Sequence[EventRef],
    truths: Sequence[EventRef],
    iou_threshold: float = 0.25,
) -> EventMetrics:
    """Compute greedy one-to-one metrics, matching only within a session."""

    pairs: list[tuple[float, int, int]] = []
    for pred_idx, prediction in enumerate(predictions):
        for truth_idx, truth in enumerate(truths):
            if prediction.sid != truth.sid:
                continue
            iou = event_iou(prediction.interval, truth.interval)
            if iou >= iou_threshold:
                pairs.append((iou, pred_idx, truth_idx))

    pairs.sort(key=lambda row: -row[0])
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    for _, pred_idx, truth_idx in pairs:
        if pred_idx in used_predictions or truth_idx in used_truths:
            continue
        used_predictions.add(pred_idx)
        used_truths.add(truth_idx)

    n_tp = len(used_predictions)
    n_pred = len(predictions)
    n_true = len(truths)
    sensitivity = n_tp / n_true if n_true else 0.0
    ppv = n_tp / n_pred if n_pred else 0.0
    f1 = (
        2.0 * sensitivity * ppv / (sensitivity + ppv)
        if sensitivity + ppv
        else 0.0
    )
    return EventMetrics(n_tp, n_pred, n_true, sensitivity, ppv, f1)


def select_event_threshold(
    candidates: Sequence[EventRef],
    scores: np.ndarray,
    truths: Sequence[EventRef],
    iou_threshold: float = 0.25,
) -> ThresholdSelection:
    """Select one threshold by event F1, then PPV, then strictness."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(candidates):
        raise ValueError("scores must be one-dimensional and align with candidates")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")

    thresholds = np.unique(
        np.concatenate((scores, [0.0, 1.0, np.nextafter(1.0, 2.0)]))
    )
    best: ThresholdSelection | None = None
    for threshold in thresholds:
        predictions = [
            candidate
            for candidate, score in zip(candidates, scores)
            if score >= threshold
        ]
        metrics = compute_event_metrics(predictions, truths, iou_threshold)
        choice = ThresholdSelection(float(threshold), metrics)
        if best is None or (
            metrics.f1,
            metrics.ppv,
            choice.threshold,
        ) > (
            best.metrics.f1,
            best.metrics.ppv,
            best.threshold,
        ):
            best = choice

    assert best is not None
    return best
