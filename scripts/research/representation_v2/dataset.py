"""Timestamp-faithful, session-local 5-second motion token extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.pipeline.imu_features import gravity_rotation
from src.pipeline.preprocessing.timeline import valid_imu_spans


@dataclass(frozen=True)
class PreprocessConfig:
    sampling_hz: int = 25
    token_seconds: int = 5
    stride_seconds: int = 5
    max_gap_ms: int = 200
    min_coverage: float = 0.8
    gravity_align: bool = True

    def __post_init__(self) -> None:
        if (self.sampling_hz, self.token_seconds, self.stride_seconds, self.max_gap_ms, self.min_coverage) != (25, 5, 5, 200, 0.8):
            raise ValueError("v2 first-pass preprocessing contract is frozen")


@dataclass(frozen=True)
class TokenBag:
    candidate_key: tuple[str, int, int]
    values: np.ndarray
    sample_mask: np.ndarray
    token_mask: np.ndarray
    token_center_ms: np.ndarray
    start_ms: int
    end_ms: int
    coverage: np.ndarray


@dataclass(frozen=True)
class PreparedMotionSession:
    spans: tuple[tuple[np.ndarray, np.ndarray], ...]
    first_ms: float
    last_ms: float
    row_rate: float


def _expand_packet_timestamps(timestamps: np.ndarray, row_rate: float) -> np.ndarray:
    """Spread same-stamp packet rows using their order and the source row rate.

    Raw target cache timestamps are recorded once for roughly ten IMU rows.
    Interpolating against those repeated stamps collapses motion to packet edges.
    We retain the recorded anchor and fill only its ordered rows, never across
    a gap to the next packet.
    """
    source = np.asarray(timestamps, dtype=np.int64)
    if np.any(np.diff(source) < 0):
        raise ValueError("timeline span timestamps regress")
    starts = np.r_[0, np.flatnonzero(np.diff(source) != 0) + 1]
    ends = np.r_[starts[1:], len(source)]
    expanded = source.astype(np.float64)
    nominal_period = 1_000.0 / row_rate
    for position, (left, right) in enumerate(zip(starts, ends)):
        size = right - left
        if size <= 1:
            continue
        following = float(source[starts[position + 1]] - source[left]) if position + 1 < len(starts) else float("inf")
        period = min(nominal_period, following / size)
        expanded[left:right] += np.arange(size) * period
    if np.any(np.diff(expanded) <= 0):
        raise ValueError("packet timestamp expansion is not strictly increasing")
    return expanded


def prepare_session(session: object) -> PreparedMotionSession:
    timestamps = np.asarray(getattr(session, "t_acc"))
    acc = np.asarray(getattr(session, "acc"))
    gyro = np.asarray(getattr(session, "gyro"))
    valid = np.asarray(getattr(session, "imu_valid"), dtype=bool)
    rate = float(getattr(session, "row_rate"))
    if timestamps.ndim != 1 or timestamps.dtype.kind not in "iu" or len(timestamps) < 2:
        raise ValueError("session needs integer millisecond IMU timestamps")
    if acc.shape != gyro.shape or acc.shape != (3, len(timestamps)) or valid.shape != timestamps.shape:
        raise ValueError("session ACC/GYRO/validity shapes disagree")
    if not np.isfinite(rate) or rate <= 20:
        raise ValueError("source IMU sampling rate is too low for 10 Hz anti-alias cutoff")
    if not np.isfinite(acc[:, valid]).all() or not np.isfinite(gyro[:, valid]).all():
        raise ValueError("valid IMU samples must be finite")
    source_spans = valid_imu_spans(session)
    if not source_spans:
        raise ValueError("session has no valid IMU spans")
    expanded_spans = []
    for span in source_spans:
        if span.row_indices is None:
            raise ValueError("timeline span lacks source row mapping")
        t = _expand_packet_timestamps(span.timestamps_ms, rate)
        values = np.vstack((acc[:, span.row_indices], gyro[:, span.row_indices])).astype(np.float64)
        expanded_spans.append((t, values))
    return PreparedMotionSession(tuple(expanded_spans), float(expanded_spans[0][0][0]),
                                 float(expanded_spans[-1][0][-1]), rate)


def _event_parts(candidate: object) -> tuple[str, int, int]:
    event = getattr(candidate, "event", candidate)
    sid = getattr(event, "sid", None)
    start = getattr(event, "start_ms", None)
    end = getattr(event, "end_ms", None)
    if not isinstance(sid, str) or not sid or not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
        raise ValueError("candidate must provide sid/start_ms/end_ms")
    if end <= start:
        raise ValueError("candidate end must exceed start")
    return sid, int(start), int(end)


def build_token_bag(session: object, candidate: object, preprocess_config: PreprocessConfig) -> TokenBag:
    sid, start, end = _event_parts(candidate)
    config = preprocess_config
    prepared = session if isinstance(session, PreparedMotionSession) else prepare_session(session)
    if start < prepared.first_ms or end > prepared.last_ms + config.max_gap_ms:
        raise ValueError("candidate extends beyond its session")

    count = int(np.ceil((end - start) / (config.stride_seconds * 1_000)))
    sample_count = config.sampling_hz * config.token_seconds
    token_starts = start + np.arange(count, dtype=np.int64) * config.stride_seconds * 1_000
    target = token_starts[:, None] + np.arange(sample_count, dtype=np.int64)[None, :] * (1_000 // config.sampling_hz)
    inside_candidate = target < end
    flat = target.ravel()
    flat_values = np.zeros((6, flat.size), dtype=np.float32)
    flat_mask = np.zeros(flat.size, dtype=bool)

    # Candidate-local gravity estimate uses only this session's valid samples.
    gravity_sources = []
    for span_t, span_values in prepared.spans:
        candidate_rows = (span_t >= start) & (span_t < end)
        if candidate_rows.any():
            gravity_sources.append(span_values[:3, candidate_rows])
    gravity = np.median(np.concatenate(gravity_sources, axis=1), axis=1) if gravity_sources else np.array([0.0, 0.0, 1.0])
    rotation = gravity_rotation(gravity) if config.gravity_align else np.eye(3)
    sos = butter(4, 10.0, btype="lowpass", fs=prepared.row_rate, output="sos")

    for span_t, span_values in prepared.spans:
        if span_t[-1] < start or span_t[0] >= end:
            continue
        left = max(0, int(np.searchsorted(span_t, start - 1_000)))
        right = min(len(span_t), int(np.searchsorted(span_t, end + 1_000, side="right")))
        if right - left < 32:
            continue
        source_t = span_t[left:right]
        within = (flat >= source_t[0]) & (flat <= source_t[-1]) & inside_candidate.ravel()
        if not within.any():
            continue
        filtered = sosfiltfilt(sos, span_values[:, left:right], axis=1)
        filtered[:3] = rotation @ filtered[:3]
        filtered[3:] = rotation @ filtered[3:]
        for channel in range(6):
            flat_values[channel, within] = np.interp(flat[within], source_t, filtered[channel]).astype(np.float32)
        flat_mask[within] = True

    sample_mask = flat_mask.reshape(count, sample_count)
    values = flat_values.reshape(6, count, sample_count).transpose(1, 0, 2).copy()
    coverage = sample_mask.mean(axis=1).astype(np.float32)
    token_mask = coverage >= config.min_coverage
    values[~sample_mask[:, None, :].repeat(6, axis=1)] = 0.0
    return TokenBag(
        candidate_key=(sid, start, end), values=values, sample_mask=sample_mask,
        token_mask=token_mask, token_center_ms=token_starts + config.token_seconds * 500,
        start_ms=start, end_ms=end, coverage=coverage,
    )


def pool_time_bins(embeddings: np.ndarray, bag: TokenBag, *, bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.shape != (len(bag.token_mask), 16) or bins < 1:
        raise ValueError("embeddings must align with 16D token bag")
    result = np.zeros((bins, 36), dtype=np.float32)
    output_mask = np.zeros(bins, dtype=bool)
    span = bag.end_ms - bag.start_ms
    placement = np.minimum(bins - 1, ((bag.token_center_ms - bag.start_ms) * bins // span).astype(int))
    for bin_index in range(bins):
        assigned = placement == bin_index
        kept = assigned & bag.token_mask
        if not kept.any():
            continue
        result[bin_index, :16] = vectors[kept].mean(axis=0)
        result[bin_index, 16:32] = vectors[kept].max(axis=0)
        result[bin_index, 32] = (bin_index + 0.5) / bins
        result[bin_index, 33] = 1.0 / bins
        result[bin_index, 34] = bag.coverage[assigned].mean()
        result[bin_index, 35] = kept.sum() / max(1, assigned.sum())
        output_mask[bin_index] = True
    return result, output_mask
