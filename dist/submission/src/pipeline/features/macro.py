"""Frozen 62-D macro IMU windows and the separate 63rd time-prior adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.signal

from src.pipeline.event_stack import EventRef, _GLOBAL_PRIOR
from src.pipeline.preprocessing.timeline import valid_imu_spans


@dataclass(frozen=True)
class MacroFeatureConfig:
    window_ms: int = 240_000
    stride_ms: int = 15_000
    coverage_min: float = 0.80
    row_rate_hz: float = 105.0


@dataclass(frozen=True)
class MacroWindowBatch62:
    features: np.ndarray
    windows: tuple[EventRef, ...]


_STATS = ("mean", "std", "median", "p10", "p25", "p75", "p90", "p90_p10", "iqr", "rms", "mad")
_POSE_STATS = ("mean", "median", "p10", "p25", "p75", "p90", "rms")


def _stats(values: np.ndarray, names=_STATS) -> list[float]:
    values = values.astype(np.float64)
    p10, p25, p75, p90 = np.percentile(values, [10, 25, 75, 90])
    median = float(np.median(values))
    full = {
        "mean": float(values.mean()), "std": float(values.std()), "median": median,
        "p10": p10, "p25": p25, "p75": p75, "p90": p90,
        "p90_p10": p90 - p10, "iqr": p75 - p25,
        "rms": float(np.sqrt((values ** 2).mean())),
        "mad": float(np.median(np.abs(values - median))),
    }
    return [full[name] for name in names]


def _temporal_on_env(env: np.ndarray) -> list[float]:
    n = len(env)
    median = float(np.median(env))
    mad = float(np.median(np.abs(env - median)))
    scale = max(mad, 1e-6)
    normalized = (env - median) / scale
    smooth = np.convolve(normalized, np.ones(3) / 3, mode="same")
    active_ratio = float((env > median + 2.0 * scale).mean())
    peaks, _ = scipy.signal.find_peaks(smooth, prominence=1.0, distance=2)
    peak_rate = len(peaks) / (n / 60.0)
    if len(peaks) >= 2:
        gaps = np.diff(peaks)
        peak_interval_median = float(np.median(gaps))
        peak_interval_cv = float(gaps.std() / (gaps.mean() + 1e-9))
    else:
        peak_interval_median = float(n) if len(peaks) == 1 else np.nan
        peak_interval_cv = np.nan
    centered = env - env.mean()
    power = np.abs(np.fft.rfft(centered * np.hanning(n))) ** 2
    if power.size > 1:
        nonzero_power = power[1:]
        entropy = -float((nonzero_power / nonzero_power.sum() * np.log(nonzero_power / nonzero_power.sum() + 1e-12)).sum()) / np.log(nonzero_power.size)
        dominant = float(np.argmax(nonzero_power) / n)
        total = nonzero_power.sum() + 1e-12

        def band_power(low: float, high: float) -> float:
            indices = np.arange(1, n // 2 + 1) / n
            return float(nonzero_power[(indices >= low) & (indices < high)].sum() / total)

        slow, middle, fast = band_power(0.02, 0.08), band_power(0.08, 0.20), band_power(0.20, 0.45)
    else:
        entropy = dominant = slow = middle = fast = np.nan
    return [float(np.median(env)), float(np.percentile(env, 90)), active_ratio, peak_rate,
            peak_interval_median, peak_interval_cv, entropy, dominant, slow, middle, fast]


def _temporal_from_env(segment: np.ndarray, sample_rate_hz: float) -> list[float]:
    samples_per_second = int(sample_rate_hz)
    usable = int(segment.shape[1] // samples_per_second) * samples_per_second
    if usable < 30 * samples_per_second:
        return np.full(11, np.nan)
    blocks = segment[:, :usable].reshape(3, -1, samples_per_second)
    valid = (~np.isnan(blocks)).all(axis=(0, 2))
    axis_range = np.percentile(blocks, 90, axis=2) - np.percentile(blocks, 10, axis=2)
    env = np.linalg.norm(axis_range, axis=0)
    env[~valid] = np.nan
    if (~np.isnan(env)).sum() < 0.8 * len(env):
        return np.full(11, np.nan)
    valid_indices = np.arange(len(env))[~np.isnan(env)]
    env = np.interp(np.arange(len(env)), valid_indices, env[~np.isnan(env)])
    return _temporal_on_env(env)


def extract_macro_features(segment: np.ndarray, sample_rate_hz: float) -> list[float]:
    """Exact frozen raw macro mathematics from the historic slide producer."""
    magnitude = np.linalg.norm(segment, axis=0)
    x_axis, y_axis, z_axis = segment[0], segment[1], segment[2]
    jerk = np.linalg.norm(np.diff(segment, axis=1), axis=0) * sample_rate_hz
    output: list[float] = []
    for channel in (y_axis, z_axis, magnitude):
        output.extend(_stats(channel))
    output.extend(_stats(x_axis, _POSE_STATS))
    output.extend(_stats(jerk))
    output.extend(_temporal_from_env(segment, sample_rate_hz))
    if len(output) != 62:
        raise RuntimeError(f"macro feature contract produced {len(output)}, expected 62")
    return output


def extract_macro_windows(session, *, session_id: str, config: MacroFeatureConfig) -> MacroWindowBatch62:
    """Build gap-safe raw 62-D windows with historic slide boundaries/coverage."""
    features: list[list[float]] = []
    windows: list[EventRef] = []
    for span in valid_imu_spans(session):
        timestamps = span.timestamps_ms
        if len(timestamps) < int(config.window_ms / 1000.0 * config.row_rate_hz):
            continue
        starts = np.arange(
            span.start_ms,
            span.end_ms - config.window_ms + 1,
            config.stride_ms,
            dtype=np.int64,
        )
        indices = span.row_indices
        if indices is None:
            indices = np.arange(len(timestamps), dtype=np.int64)
        acc = np.asarray(session.acc, dtype=np.float32)[:, indices]
        minimum_rows = config.coverage_min * config.window_ms / 1000.0 * config.row_rate_hz
        for start in starts:
            end = int(start + config.window_ms)
            left, right = np.searchsorted(timestamps, (start, end))
            if right - left < minimum_rows:
                continue
            features.append(extract_macro_features(acc[:, left:right], config.row_rate_hz))
            windows.append(EventRef(session_id, int(start), end))
    matrix = np.asarray(features, dtype=np.float32).reshape((-1, 62)) if features else np.empty((0, 62), dtype=np.float32)
    return MacroWindowBatch62(matrix, tuple(windows))


def add_time_prior(features_62: np.ndarray, windows: Sequence[EventRef]) -> np.ndarray:
    """Append the release-frozen hour-of-day prior at the model boundary."""
    matrix = np.asarray(features_62)
    if matrix.ndim != 2 or matrix.shape[1] != 62 or len(matrix) != len(windows):
        raise ValueError("time-prior adapter requires aligned (n, 62) features and windows")
    prior = np.asarray(
        [_GLOBAL_PRIOR[int((window.start_ms / 3.6e6) % 24)] for window in windows],
        dtype=np.float32,
    ).reshape((-1, 1))
    return np.concatenate((matrix, prior), axis=1)
