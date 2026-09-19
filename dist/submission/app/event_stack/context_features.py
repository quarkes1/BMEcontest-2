"""Deterministic Context-v1 features built from complete probability streams."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np


CONTEXT_V1_CONTEXT_MS = 1_200_000
CONTEXT_V1_RUN_GAP_MS = 60_000
CONTEXT_V1_MIN_RUN_WINDOWS = 2
MACRO_STRIDE_MS, MACRO_WINDOW_MS = 15_000, 240_000
MICRO_STRIDE_MS, MICRO_WINDOW_MS = 7_500, 15_000

_SUFFIX_COLUMNS = (
    "pre_mean", "pre_max", "pre_std", "pre_above_fraction",
    "candidate_mean", "candidate_max", "candidate_std", "candidate_above_fraction",
    "post_mean", "post_max", "post_std", "post_above_fraction",
    "candidate_minus_pre_mean", "candidate_minus_pre_max",
    "candidate_minus_pre_std", "candidate_minus_pre_above_fraction",
    "candidate_minus_post_mean", "candidate_minus_post_max",
    "candidate_minus_post_std", "candidate_minus_post_above_fraction",
    "pre_coverage", "candidate_coverage", "post_coverage",
    "neighbor_run_count", "neighbor_run_total_duration_s", "neighbor_run_max_duration_s",
    "preceding_run_distance_s", "following_run_distance_s",
    "candidate_center_half_mass_fraction", "candidate_first_minus_second_mean",
)

_SCALES = (
    ("macro", 0.28838, MACRO_STRIDE_MS, MACRO_WINDOW_MS),
    ("micro", 0.20, MICRO_STRIDE_MS, MICRO_WINDOW_MS),
)

CONTEXT_V1_COLUMNS: tuple[str, ...] = tuple(
    f"{scale}_{suffix}" for scale, _, _, _ in _SCALES for suffix in _SUFFIX_COLUMNS
)
CONTEXT_V1_SCHEMA_HASH = hashlib.sha256(
    json.dumps(CONTEXT_V1_COLUMNS, separators=(",", ":")).encode()
).hexdigest()


def context_v1_features(
    candidates,
    macro_windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    micro_windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    session_bounds_by_sid: Mapping[str, tuple[int, int]],
) -> np.ndarray:
    """Build the fixed-width Context-v1 matrix for candidate events.

    Session bounds are deliberately a required input: sparse probability windows
    cannot be used to infer where a session starts or ends.
    """
    rows = []
    for candidate in candidates:
        event = getattr(candidate, "event", candidate)
        bounds = session_bounds_by_sid[event.sid]
        blocks = [
            _scale_context(event, streams.get(event.sid, ()), bounds,
                           threshold, stride_ms, window_ms)
            for _, threshold, stride_ms, window_ms, streams in (
                (*_SCALES[0], macro_windows_by_sid),
                (*_SCALES[1], micro_windows_by_sid),
            )
        ]
        rows.append(np.concatenate(blocks))
    result = np.asarray(rows, dtype=np.float64).reshape((-1, 60))
    if np.isinf(result).any():
        raise ValueError("Context-v1 features must not contain infinity")
    return result


def _regions(event, session_bounds: tuple[int, int]):
    session_start, session_end = _validated_time_bounds(
        session_bounds, "session bounds"
    )
    event_start, event_end = _validated_time_bounds(
        (event.start_ms, event.end_ms), "event bounds"
    )
    if session_end < session_start:
        raise ValueError("session bounds must be ordered")
    if event_end < event_start:
        raise ValueError("event bounds must be ordered")
    clip = lambda start, end: (max(session_start, start), min(session_end, end))
    return (
        clip(event_start - CONTEXT_V1_CONTEXT_MS, event_start),
        clip(event_start, event_end),
        clip(event_end, event_end + CONTEXT_V1_CONTEXT_MS),
    )


def _validated_time_bounds(bounds, label):
    try:
        start, end = bounds
        start = float(start)
        end = float(end)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite integer-like") from exc
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or not start.is_integer()
        or not end.is_integer()
    ):
        raise ValueError(f"{label} must be finite integer-like")
    return start, end


def _validated_windows(windows):
    result = []
    for window in windows:
        if len(window) != 3:
            raise ValueError("probability windows must be (start, end, probability)")
        start, end, probability = window
        start, end = _validated_time_bounds((start, end), "probability window bounds")
        if end < start:
            raise ValueError("probability window bounds must be ordered")
        probability = float(probability)
        if math.isinf(probability):
            raise ValueError("Context-v1 features must not contain infinity")
        result.append((start, end, probability))
    return result


def _in_region(windows, region):
    start, end = region
    return [window for window in windows if start <= (window[0] + window[1]) / 2.0 < end]


def _summaries(windows, regions, threshold, stride_ms):
    values = []
    coverages = []
    for region in regions:
        region_start, region_end = region
        finite = [window[2] for window in _in_region(windows, region) if np.isfinite(window[2])]
        if finite:
            values.extend((float(np.mean(finite)), float(np.max(finite)),
                           float(np.std(finite)),
                           float(np.mean(np.asarray(finite) >= threshold))))
        else:
            values.extend((np.nan, np.nan, np.nan, np.nan))
        duration = max(0.0, region_end - region_start)
        denominator = max(1, int(math.ceil(duration / stride_ms)))
        coverages.append(float(np.clip(len(finite) / denominator, 0.0, 1.0)))
    return values, coverages


def _candidate_like_runs(windows, context_region, threshold, window_ms):
    context_start, context_end = context_region
    eligible = [
        window for window in windows
        if context_start <= (window[0] + window[1]) / 2.0 < context_end
        and np.isfinite(window[2]) and window[2] >= threshold
    ]
    eligible.sort(key=lambda window: (window[0] + window[1], window[0], window[1], window[2]))
    runs = []
    current = []
    for window in eligible:
        center = (window[0] + window[1]) / 2.0
        if current:
            previous_center = (current[-1][0] + current[-1][1]) / 2.0
            if center - previous_center > CONTEXT_V1_RUN_GAP_MS:
                if len(current) >= CONTEXT_V1_MIN_RUN_WINDOWS:
                    runs.append(_run_span(current, window_ms))
                current = []
        current.append(window)
    if len(current) >= CONTEXT_V1_MIN_RUN_WINDOWS:
        runs.append(_run_span(current, window_ms))
    return runs


def _run_span(windows, window_ms):
    first_center = (windows[0][0] + windows[0][1]) / 2.0
    last_center = (windows[-1][0] + windows[-1][1]) / 2.0
    return first_center - window_ms / 2.0, last_center + window_ms / 2.0


def _scale_context(event, windows, session_bounds, threshold, stride_ms, window_ms):
    windows = _validated_windows(windows)
    pre, candidate, post = _regions(event, session_bounds)
    regions = (pre, candidate, post)
    region_values, coverages = _summaries(windows, regions, threshold, stride_ms)
    pre_values, candidate_values, post_values = (
        region_values[0:4], region_values[4:8], region_values[8:12]
    )
    differences = []
    for left in (pre_values, post_values):
        differences.extend(
            value - reference if np.isfinite(value) and np.isfinite(reference) else np.nan
            for value, reference in zip(candidate_values, left)
        )

    context_region = (pre[0], post[1])
    runs = _candidate_like_runs(windows, context_region, threshold, window_ms)
    candidate_start, candidate_end = candidate
    own_runs = [run for run in runs if run[0] < candidate_end and run[1] > candidate_start]
    neighbors = [run for run in runs if run not in own_runs]
    preceding = [run for run in neighbors if run[1] <= candidate_start]
    following = [run for run in neighbors if run[0] >= candidate_end]
    preceding_run = max(preceding, key=lambda run: run[1], default=None)
    following_run = min(following, key=lambda run: run[0], default=None)
    durations = [(end - start) / 1000.0 for start, end in neighbors]
    if neighbors:
        neighbor_values = [len(neighbors), sum(durations), max(durations)]
    else:
        neighbor_values = [0, 0.0, np.nan]
    neighbor_values.extend((
        (candidate_start - preceding_run[1]) / 1000.0 if preceding_run else np.nan,
        (following_run[0] - candidate_end) / 1000.0 if following_run else np.nan,
    ))

    candidate_windows = _in_region(windows, candidate)
    concentration, half_difference = _candidate_shape(candidate_windows, candidate)
    return np.asarray(
        region_values[:12] + differences + coverages + neighbor_values + [concentration, half_difference],
        dtype=np.float64,
    )


def _candidate_shape(windows, candidate):
    start, end = candidate
    duration = end - start
    if duration <= 0:
        return np.nan, np.nan
    midpoint = start + duration / 2.0
    center_start = start + duration / 4.0
    center_end = start + 3.0 * duration / 4.0
    first = [window[2] for window in windows
             if np.isfinite(window[2]) and (window[0] + window[1]) / 2.0 < midpoint]
    second = [window[2] for window in windows
              if np.isfinite(window[2]) and (window[0] + window[1]) / 2.0 >= midpoint]
    half_difference = (
        float(np.mean(first)) - float(np.mean(second)) if first and second else np.nan
    )
    denominator = 0.0
    numerator = 0.0
    for window_start, window_end, probability in windows:
        center = (window_start + window_end) / 2.0
        if not np.isfinite(probability) or probability < 0:
            continue
        denominator += probability
        if center_start <= center < center_end:
            numerator += probability
    concentration = numerator / denominator if denominator > 0 else np.nan
    return float(concentration), half_difference
