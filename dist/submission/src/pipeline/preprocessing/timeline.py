"""不跨缺口的惯性片段划分及考虑覆盖率的窗口起点生成。"""

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
    """统计有效惯性行中的时间戳回退次数，不修改会话。"""
    indices = np.flatnonzero(np.asarray(session.imu_valid, dtype=bool))
    timestamps = np.asarray(session.t_acc, dtype=np.int64)[indices]
    return int(np.count_nonzero(np.diff(timestamps) < 0))


def valid_imu_spans(session) -> tuple[TimelineSpan, ...]:
    """按采集缺口和时间戳回退切分有效惯性行。

时钟回退与缺口一样触发新片段；回退片段若与已有时间区间重叠，裁去已覆盖的前缀。
返回片段按开始时间排序且互不重叠，原本有序的会话保持不变。"""
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
    """按开始时间排列片段，裁去已被先前片段覆盖的部分。"""
    if all(left.end_ms < right.start_ms for left, right in zip(spans, spans[1:])):
        return tuple(spans)                       # 已排序且互不重叠时直接返回，这是常见情况。
    kept: list[TimelineSpan] = []
    frontier: int | None = None
    for span in sorted(spans, key=lambda item: item.start_ms):
        rows, stamps = span.row_indices, span.timestamps_ms
        if frontier is not None:
            if span.end_ms <= frontier:
                continue                          # 当前片段已被先前片段完全覆盖。
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
    """返回覆盖率达标且完全位于单个时间片段内的窗口起点。"""
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
