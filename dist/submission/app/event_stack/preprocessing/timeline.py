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
    """Split valid IMU rows at acquisition discontinuities and timestamp regressions.

    A rewound clock starts a new span exactly like a gap does.  Because a rewound run
    can overlap the wall-clock interval already owned by an earlier run, the returned
    spans are ordered by start time and never overlap: an overlapping prefix is
    trimmed (those rows duplicate an interval another span already covers).  Ordered
    sessions return the same spans as before, unchanged.
    """
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
    spans = [TimelineSpan(int(timestamps[group[0]]), int(timestamps[group[-1]]),
                          timestamps[group].copy(), indices[group].copy())
             for group in groups if len(group)]
    return _ordered_disjoint_spans(spans)


def _ordered_disjoint_spans(spans: list[TimelineSpan]) -> tuple[TimelineSpan, ...]:
    """Order spans by start time and trim whatever an earlier span already covers."""
    if all(left.end_ms < right.start_ms for left, right in zip(spans, spans[1:])):
        return tuple(spans)                       # already ordered and disjoint (the common case)
    kept: list[TimelineSpan] = []
    frontier: int | None = None
    for span in sorted(spans, key=lambda item: item.start_ms):
        rows, stamps = span.row_indices, span.timestamps_ms
        if frontier is not None:
            if span.end_ms <= frontier:
                continue                          # fully covered by an earlier span
            if span.start_ms <= frontier:
                cut = int(np.searchsorted(stamps, frontier, side="right"))
                if cut >= stamps.size:
                    continue
                stamps, rows = stamps[cut:], None if rows is None else rows[cut:]
        kept.append(TimelineSpan(int(stamps[0]), int(stamps[-1]), stamps.copy(),
                                 None if rows is None else rows.copy()))
        frontier = int(stamps[-1])
    return tuple(kept)


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
