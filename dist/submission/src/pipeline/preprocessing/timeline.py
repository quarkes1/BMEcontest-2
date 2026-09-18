"""Gap-safe IMU spans and coverage-aware window starts."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TimelineSpan:
    start_ms: int
    end_ms: int
    timestamps_ms: np.ndarray
    row_indices: np.ndarray | None = None


def timeline_regressions(session) -> int:
    """Count timestamp regressions among valid IMU rows without changing the session."""
    indices = np.flatnonzero(np.asarray(session.imu_valid, dtype=bool))
    timestamps = np.asarray(session.t_acc, dtype=np.int64)[indices]
    return int(np.count_nonzero(np.diff(timestamps) < 0))


def valid_imu_spans(session) -> tuple[TimelineSpan, ...]:
    """Split valid IMU rows at acquisition discontinuities and timestamp regressions."""
    indices = np.flatnonzero(np.asarray(session.imu_valid, dtype=bool))
    if not len(indices):
        return ()
    timestamps = np.asarray(session.t_acc, dtype=np.int64)[indices]
    if (timestamps < 0).any():
        raise ValueError("valid IMU timestamps must be non-negative")
    positive = np.diff(timestamps)
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError("valid IMU timestamps require a positive interval")
    period = float(np.median(positive))
    deltas = np.diff(timestamps)
    groups = np.split(np.arange(len(indices)), np.flatnonzero((deltas > period * 2.0) | (deltas < 0)) + 1)
    return tuple(TimelineSpan(int(timestamps[group[0]]), int(timestamps[group[-1]]), timestamps[group].copy(), indices[group].copy()) for group in groups if len(group))


def window_starts(span: TimelineSpan, window_ms: int, stride_ms: int, coverage_min: float) -> np.ndarray:
    """Return coverage-qualified starts entirely inside one timeline span."""
    if window_ms <= 0 or stride_ms <= 0 or not 0 < coverage_min <= 1:
        raise ValueError("invalid window parameters")
    timestamps = np.asarray(span.timestamps_ms, dtype=np.int64)
    positive = np.diff(timestamps)
    positive = positive[positive > 0]
    if not len(positive):
        return np.empty(0, dtype=np.int64)
    period = float(np.median(positive))
    expected = coverage_min * window_ms / period
    end_exclusive = int(round(span.end_ms + period))
    starts = np.arange(span.start_ms, end_exclusive - window_ms + 1, stride_ms, dtype=np.int64)
    return np.asarray([start for start in starts if np.searchsorted(timestamps, start + window_ms) - np.searchsorted(timestamps, start) >= expected], dtype=np.int64)
