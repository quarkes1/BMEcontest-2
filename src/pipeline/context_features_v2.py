"""Pure, fixed Context-v2 features for raw multiscale candidates.

This module intentionally knows nothing about labels, fitted models, or
verifier scores.  It only summarizes raw candidate evidence and raw window
probabilities within each acquisition session.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


CONTEXT_V2_COLUMNS: tuple[str, ...] = (
    "macro_candidate_mean_percentile",
    "macro_candidate_max_percentile",
    "macro_candidate_mean_robust_z",
    "macro_candidate_mean_relative_to_session_median",
    "macro_candidate_max_relative_to_session_p90",
    "macro_candidate_mean_to_session_max",
    "macro_best_mean_margin",
    "macro_stronger_count_20m",
    "macro_nearest_stronger_distance_s",
    "micro_candidate_mean_percentile",
    "micro_candidate_max_percentile",
    "micro_candidate_mean_robust_z",
    "micro_candidate_mean_relative_to_session_median",
    "micro_candidate_max_relative_to_session_p90",
    "micro_candidate_mean_to_session_max",
    "micro_best_mean_margin",
    "micro_stronger_count_20m",
    "micro_nearest_stronger_distance_s",
    "session_candidate_count",
    "session_candidate_rate_per_hour",
    "candidate_duration_percentile",
    "candidate_duration_robust_z",
    "candidate_support_count",
    "candidate_support_coverage_percentile",
    "candidate_support_coverage_robust_z",
)
CONTEXT_V2_SCHEMA_HASH = hashlib.sha256(
    json.dumps(CONTEXT_V2_COLUMNS, separators=(",", ":")).encode()
).hexdigest()

_MAD_FLOOR = 0.05
_TWENTY_MINUTES_S = 1_200.0
# The sensor index records whole-second bounds, while cached candidate windows
# retain stride-aligned millisecond timestamps.  A macro candidate can therefore
# start up to one 15 s grid step outside the nominal index endpoint without
# leaving its acquisition session.  This is a data-format tolerance, not a
# learned or tuneable Context-v2 parameter.
_SESSION_BOUNDARY_QUANTIZATION_MS = 15_000


def context_schema_descriptor(version: str) -> dict[str, object]:
    """Return the immutable, non-learned Context-v2 schema contract."""

    if version != "v2":
        raise ValueError("only Context-v2 has a descriptor in this module")
    return {
        "name": "context-v2",
        "definition": "session-local raw candidate and probability context",
        "data_dependency": "raw-union candidates and raw probability windows",
        "label_dependency": False,
        "session_local": True,
        "learned": False,
        "width": len(CONTEXT_V2_COLUMNS),
        "columns": list(CONTEXT_V2_COLUMNS),
        "ordered_column_sha256": CONTEXT_V2_SCHEMA_HASH,
    }


def context_v2_features(
    candidates: Sequence[object],
    macro_windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    micro_windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    session_bounds_by_sid: Mapping[str, tuple[int, int]],
) -> np.ndarray:
    """Build the fixed 25D Context-v2 append block in caller candidate order.

    Candidates are validated as complete members of the already-built raw union;
    they are never deduplicated or scored here.  The internal view is sorted only
    to make all comparison populations deterministic.
    """

    records = [_validate_candidate(candidate, index, session_bounds_by_sid)
               for index, candidate in enumerate(candidates)]
    if not records:
        return np.empty((0, len(CONTEXT_V2_COLUMNS)), dtype=np.float64)

    needed_sids = {record["sid"] for record in records}
    macro_streams = _validated_streams(macro_windows_by_sid, needed_sids, "macro")
    micro_streams = _validated_streams(micro_windows_by_sid, needed_sids, "micro")
    by_sid: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_sid[record["sid"]].append(record)
    for values in by_sid.values():
        values.sort(key=_working_key)

    rows: list[np.ndarray | None] = [None] * len(records)
    for sid, population in by_sid.items():
        streams = {"macro": macro_streams[sid], "micro": micro_streams[sid]}
        bounds = _validated_bounds(session_bounds_by_sid[sid])
        durations = np.asarray([record["duration"] for record in population], dtype=np.float64)
        coverages = np.asarray([record["coverage"] for record in population], dtype=np.float64)
        for record in population:
            source_blocks = [
                _source_features(record, population, source, streams[source])
                for source in ("macro", "micro")
            ]
            duration = float(record["duration"])
            rate = len(population) / max((bounds[1] - bounds[0]) / 3_600_000.0, 1.0 / 3600.0)
            row = np.asarray(
                source_blocks[0]
                + source_blocks[1]
                + [
                    float(len(population)),
                    float(rate),
                    _midrank(duration, durations),
                    _robust_z(duration, durations),
                    float(record["support_count"]),
                    _midrank(float(record["coverage"]), coverages),
                    _robust_z(float(record["coverage"]), coverages),
                ],
                dtype=np.float64,
            )
            if np.isinf(row).any():
                raise ValueError("Context-v2 features must not contain infinity")
            rows[int(record["index"])] = row
    return np.asarray(rows, dtype=np.float64).reshape((-1, len(CONTEXT_V2_COLUMNS)))


def _validate_candidate(candidate: object, index: int, bounds_by_sid: Mapping[str, tuple[int, int]]) -> dict[str, object]:
    event = getattr(candidate, "event", None)
    if event is None:
        raise ValueError("raw-union candidate must have an event")
    sid = getattr(event, "sid", None)
    if not isinstance(sid, str) or not sid:
        raise ValueError("raw-union candidate must have one exact session sid")
    start, end = _integer_bounds((getattr(event, "start_ms", None), getattr(event, "end_ms", None)), "candidate bounds")
    if end <= start:
        raise ValueError("raw-union candidates must have positive duration")
    if sid not in bounds_by_sid:
        raise ValueError("raw-union candidate session bounds are required")
    session_start, session_end = _validated_bounds(bounds_by_sid[sid])
    if (
        start < session_start - _SESSION_BOUNDARY_QUANTIZATION_MS
        or end > session_end + _SESSION_BOUNDARY_QUANTIZATION_MS
    ):
        raise ValueError("raw-union candidate must lie inside its session bounds")

    source_values: dict[str, float] = {}
    source_maxima: dict[str, float] = {}
    finite_coverages: list[float] = []
    support_count = 0
    any_finite_probability = False
    canonical_sources: list[tuple[object, ...] | None] = []
    for name in ("macro", "micro"):
        source = getattr(candidate, name, None)
        if source is None:
            canonical_sources.append(None)
            continue
        support_count += 1
        source_event = getattr(source, "event", None)
        if source_event is None or getattr(source_event, "sid", None) != sid:
            raise ValueError("raw-union source must belong to candidate session")
        source_start, source_end = _integer_bounds(
            (getattr(source_event, "start_ms", None), getattr(source_event, "end_ms", None)),
            "source candidate bounds",
        )
        if source_end <= source_start:
            raise ValueError("raw-union source candidates must have positive duration")
        probabilities = _candidate_probabilities(getattr(source, "probabilities", None))
        finite = [value for value in probabilities if math.isfinite(value)]
        if finite:
            any_finite_probability = True
            source_values[name] = float(np.mean(finite))
            source_maxima[name] = float(np.max(finite))
        coverage = _finite_or_nan(getattr(source, "observed_fraction", np.nan), "source observed_fraction")
        if math.isfinite(coverage):
            finite_coverages.append(coverage)
        canonical_sources.append((
            sid, source_start, source_end, tuple(probabilities), coverage,
            getattr(source, "bridged_gap_count", None), getattr(source, "bridged_gap_ms", None),
            getattr(source, "pre_observed_count", None), getattr(source, "post_observed_count", None),
        ))
    if support_count == 0:
        raise ValueError("raw-union candidate needs at least one source")
    if not any_finite_probability:
        raise ValueError("raw-union candidate needs a finite probability stream")
    coverage = float(np.mean(finite_coverages)) if finite_coverages else np.nan
    return {
        "index": index,
        "sid": sid,
        "start": start,
        "end": end,
        "duration": float(end - start),
        "means": source_values,
        "maxima": source_maxima,
        "support_count": support_count,
        "coverage": coverage,
        "identity": tuple(canonical_sources),
    }


def _candidate_probabilities(values: object) -> tuple[float, ...]:
    if values is None:
        return ()
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("source probability stream must be numeric") from exc
    if any(math.isinf(value) for value in result):
        raise ValueError("source probability streams must not contain infinity")
    return result


def _validated_streams(
    windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]], needed_sids: set[str], source: str,
) -> dict[str, np.ndarray]:
    streams: dict[str, np.ndarray] = {}
    for sid in needed_sids:
        values: list[float] = []
        for window in windows_by_sid.get(sid, ()):
            try:
                if len(window) != 3:
                    raise ValueError
                start, end, probability = window
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source} windows must be (start, end, probability)") from exc
            start, end = _integer_bounds((start, end), f"{source} window bounds")
            if end < start:
                raise ValueError(f"{source} window bounds must be ordered")
            value = _finite_or_nan(probability, f"{source} window probability")
            if not math.isfinite(value):
                raise ValueError(f"{source} window probability must be finite")
            values.append(value)
        streams[sid] = np.asarray(values, dtype=np.float64)
    return streams


def _source_features(record: dict[str, object], population: Sequence[dict[str, object]], source: str, stream: np.ndarray) -> list[float]:
    mean = record["means"].get(source)  # type: ignore[union-attr]
    maximum = record["maxima"].get(source)  # type: ignore[union-attr]
    if mean is None or maximum is None:
        return [np.nan] * 9
    means = np.asarray([item["means"][source] for item in population if source in item["means"]], dtype=np.float64)
    maxima = np.asarray([item["maxima"][source] for item in population if source in item["maxima"]], dtype=np.float64)
    stronger = [item for item in population if source in item["means"] and item["means"][source] > mean]
    distances = np.asarray([_distance_seconds(record, item) for item in stronger], dtype=np.float64)
    nearby = int(np.count_nonzero(distances <= _TWENTY_MINUTES_S))
    if stronger:
        margin = float(max(item["means"][source] for item in stronger) - mean)
        nearest = float(np.min(distances))
    else:
        margin, nearest = 0.0, np.nan
    if len(stream):
        median = float(np.median(stream))
        p90 = float(np.percentile(stream, 90, method="linear"))
        session_max = float(np.max(stream))
        relative_median = _relative(mean, median)
        relative_p90 = _relative(maximum, p90)
        ratio_max = mean / max(session_max, _MAD_FLOOR)
    else:
        relative_median = relative_p90 = ratio_max = np.nan
    return [
        _midrank(float(mean), means), _midrank(float(maximum), maxima),
        _robust_z(float(mean), means), relative_median, relative_p90, ratio_max,
        margin, float(nearby), nearest,
    ]


def _working_key(record: dict[str, object]) -> tuple[object, ...]:
    return record["sid"], record["start"], record["end"], record["identity"]


def _distance_seconds(left: dict[str, object], right: dict[str, object]) -> float:
    if right["start"] >= left["end"]:
        return (right["start"] - left["end"]) / 1000.0
    if left["start"] >= right["end"]:
        return (left["start"] - right["end"]) / 1000.0
    return 0.0


def _midrank(value: float, values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan
    return float((np.count_nonzero(finite < value) + .5 * np.count_nonzero(finite == value)) / len(finite))


def _robust_z(value: float, values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return (value - median) / (1.4826 * mad + _MAD_FLOOR)


def _relative(value: float, baseline: float) -> float:
    return (value - baseline) / max(abs(baseline), _MAD_FLOOR)


def _validated_bounds(bounds: object) -> tuple[int, int]:
    start, end = _integer_bounds(bounds, "session bounds")
    if end < start:
        raise ValueError("session bounds must be ordered")
    return start, end


def _integer_bounds(bounds: object, label: str) -> tuple[int, int]:
    try:
        start, end = bounds  # type: ignore[misc]
        start_value, end_value = float(start), float(end)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite integer-like") from exc
    if not (math.isfinite(start_value) and math.isfinite(end_value) and start_value.is_integer() and end_value.is_integer()):
        raise ValueError(f"{label} must be finite integer-like")
    return int(start_value), int(end_value)


def _finite_or_nan(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if math.isinf(result):
        raise ValueError(f"{label} must not contain infinity")
    return result
