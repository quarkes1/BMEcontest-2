"""Canonical gap-safe 47-D micro IMU window producer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_stack.event_stack import EventRef
from event_stack.imu_features import MicroFeatureConfig, extract_micro_features, gravity_rotation
from event_stack.preprocessing.timeline import valid_imu_spans, window_starts


@dataclass(frozen=True)
class MicroWindowBatch:
    features: np.ndarray
    windows: tuple[EventRef, ...]


def extract_micro_windows(session, *, session_id: str, config: MicroFeatureConfig) -> MicroWindowBatch:
    """Create the frozen gravity-aligned micro features without crossing gaps."""
    rows: list[np.ndarray] = []
    windows: list[EventRef] = []
    for span in valid_imu_spans(session):
        timestamps = span.timestamps_ms
        positive_deltas = np.diff(timestamps)
        positive_deltas = positive_deltas[positive_deltas > 0]
        if not len(positive_deltas):
            continue
        sample_rate_hz = 1000.0 / float(np.median(positive_deltas))
        indices = span.row_indices
        if indices is None:
            indices = np.arange(len(timestamps), dtype=np.int64)
        acc = np.asarray(session.acc, dtype=np.float32)[:, indices]
        gyro = np.asarray(session.gyro, dtype=np.float32)[:, indices]
        if config.gravity_align:
            rotation = gravity_rotation(np.median(acc, axis=1))
            acc, gyro = rotation @ acc, rotation @ gyro
        for start in window_starts(span, config.window_ms, config.stride_ms, config.coverage_min):
            end = int(start + config.window_ms)
            left, right = np.searchsorted(timestamps, (start, end))
            rows.append(extract_micro_features(acc[:, left:right], gyro[:, left:right], sample_rate_hz))
            windows.append(EventRef(session_id, int(start), end))
    matrix = np.asarray(rows, dtype=np.float32).reshape((-1, 47)) if rows else np.empty((0, 47), dtype=np.float32)
    return MicroWindowBatch(matrix, tuple(windows))
