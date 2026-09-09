"""Pure event-level primitives shared by training and evaluation commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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


@dataclass(frozen=True)
class EventSelectionPolicy:
    """A score threshold with an optional per-group event budget."""

    threshold: float
    max_events_per_group: int | None
    metrics: EventMetrics


@dataclass(frozen=True)
class MicroCandidateSelection:
    """A registered micro threshold and its inner-fold candidate metrics."""

    threshold: float
    metrics: EventMetrics
    candidate_count: int


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


def _policy_inputs(
    candidates: Sequence[EventRef],
    scores: np.ndarray,
    groups: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    score_values = np.asarray(scores, dtype=np.float64)
    group_values = np.asarray(groups)
    if score_values.ndim != 1 or len(score_values) != len(candidates):
        raise ValueError("scores must be one-dimensional and align with candidates")
    if group_values.ndim != 1 or len(group_values) != len(candidates):
        raise ValueError("groups must be one-dimensional and align with candidates")
    if not np.isfinite(score_values).all():
        raise ValueError("scores must contain only finite values")
    return score_values, group_values.astype(str)


def apply_event_policy(
    candidates: Sequence[EventRef],
    scores: np.ndarray,
    groups: Sequence[str],
    threshold: float,
    max_events_per_group: int | None = None,
) -> list[EventRef]:
    """Apply a threshold and keep only each group's highest-scored K events."""

    score_values, group_values = _policy_inputs(candidates, scores, groups)
    if max_events_per_group is not None and (
        isinstance(max_events_per_group, bool) or max_events_per_group < 1
    ):
        raise ValueError("max_events_per_group must be a positive integer or None")

    eligible = np.flatnonzero(score_values >= threshold)
    if max_events_per_group is None:
        selected_indices = set(int(index) for index in eligible)
    else:
        selected_indices: set[int] = set()
        for group in sorted(set(group_values[eligible])):
            group_indices = [
                int(index) for index in eligible if group_values[index] == group
            ]
            group_indices.sort(key=lambda index: (-score_values[index], index))
            selected_indices.update(group_indices[:max_events_per_group])
    return [
        candidate
        for index, candidate in enumerate(candidates)
        if index in selected_indices
    ]


def select_event_policy(
    candidates: Sequence[EventRef],
    scores: np.ndarray,
    truths: Sequence[EventRef],
    groups: Sequence[str],
    max_events_options: Sequence[int | None],
    iou_threshold: float = 0.25,
) -> EventSelectionPolicy:
    """Select a threshold and registered group budget using only supplied truths."""

    score_values, group_values = _policy_inputs(candidates, scores, groups)
    options = tuple(max_events_options)
    if not options:
        raise ValueError("max_events_options must not be empty")
    for option in options:
        if option is not None and (
            isinstance(option, bool) or not isinstance(option, int) or option < 1
        ):
            raise ValueError("event cap options must be positive integers or None")

    thresholds = np.unique(
        np.concatenate((score_values, [0.0, 1.0, np.nextafter(1.0, 2.0)]))
    )
    best: EventSelectionPolicy | None = None
    best_rank: tuple[float, float, float, float] | None = None
    for max_events in options:
        cap_rank = -float(max_events) if max_events is not None else -float("inf")
        for threshold in thresholds:
            predictions = apply_event_policy(
                candidates,
                score_values,
                group_values,
                float(threshold),
                max_events,
            )
            metrics = compute_event_metrics(predictions, truths, iou_threshold)
            rank = (metrics.f1, metrics.ppv, cap_rank, float(threshold))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = EventSelectionPolicy(
                    float(threshold), max_events, metrics
                )
    assert best is not None
    return best


@dataclass(frozen=True)
class DensityConfig:
    """All parameters that define candidate-generation semantics."""

    stride_ms: int = 15_000
    window_ms: int = 240_000
    bridge_ms: int = 60_000
    density_ms: int = 600_000
    min_positive: int = 10
    coverage_min: float = 0.80
    merge_ms: int = 120_000
    window_threshold: float = 0.28838
    context_ms: int = 1_200_000
    coverage_fix: bool = False


@dataclass(frozen=True)
class MicroCandidateConfig:
    """Parameters for gap-safe, high-resolution candidate construction."""

    stride_ms: int = 7_500
    window_ms: int = 15_000
    smooth_sigma_ms: int = 30_000
    smooth_radius_ms: int = 60_000
    merge_ms: int = 180_000
    min_duration_ms: int = 60_000
    context_ms: int = 1_200_000


@dataclass(frozen=True)
class CandidateEvent:
    """One density candidate and the data-quality context that produced it."""

    event: EventRef
    probabilities: tuple[float, ...]
    observed_fraction: float
    bridged_gap_count: int
    bridged_gap_ms: int
    pre_observed_count: int
    post_observed_count: int


_GLOBAL_PRIOR = np.array(
    [
        0.174,
        0.278,
        0.546,
        0.92,
        0.889,
        0.496,
        0.187,
        0.141,
        0.408,
        0.863,
        1.0,
        0.681,
        0.368,
        0.216,
        0.127,
        0.073,
        0.037,
        0.012,
        0.002,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def _candidate_quality(
    starts: np.ndarray,
    observed: np.ndarray,
    event_start: int,
    event_end: int,
    config: DensityConfig,
) -> tuple[float, int, int]:
    center = (event_start + event_end) // 2
    half = config.density_ms // 2
    mask = (starts >= center - half) & (starts <= center + half)
    local_observed = observed[mask]
    if not len(local_observed):
        return 0.0, 0, 0
    missing = ~local_observed
    gap_count = 0
    in_gap = False
    for value in missing:
        if value and not in_gap:
            gap_count += 1
        in_gap = bool(value)
    return (
        float(local_observed.mean()),
        gap_count,
        int(missing.sum()) * config.stride_ms,
    )


def density_candidates(
    windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    config: DensityConfig | None = None,
) -> list[CandidateEvent]:
    """Convert window probabilities into density candidates without file I/O."""

    config = config or DensityConfig()
    candidates: list[CandidateEvent] = []
    for sid in sorted(windows_by_sid):
        raw_windows = sorted(windows_by_sid[sid])
        if not raw_windows:
            continue
        segments: list[list[tuple[int, float, bool]]] = []
        segment: list[tuple[int, float, bool]] = []
        for start, _, probability in raw_windows:
            if (
                segment
                and start - segment[-1][0]
                > config.stride_ms + config.bridge_ms
            ):
                segments.append(segment)
                segment = []
            if segment and start - segment[-1][0] > config.stride_ms:
                for gap_start in range(
                    segment[-1][0] + config.stride_ms,
                    start,
                    config.stride_ms,
                ):
                    segment.append((gap_start, 0.0, False))
            segment.append((int(start), float(probability), True))
        if segment:
            segments.append(segment)

        sid_candidates: list[CandidateEvent] = []
        for segment in segments:
            starts = np.array([row[0] for row in segment], dtype=np.int64)
            probabilities = np.array([row[1] for row in segment], dtype=np.float64)
            observed = np.array([row[2] for row in segment], dtype=bool)
            if config.coverage_fix:
                half = config.density_ms // 2
                dense = np.zeros(len(segment), dtype=bool)
                for index, start in enumerate(starts):
                    left = int(np.searchsorted(starts, start - half))
                    right = int(np.searchsorted(starts, start + half, side="right"))
                    local_observed = observed[left:right]
                    coverage = (
                        float(local_observed.mean())
                        if len(local_observed)
                        else 0.0
                    )
                    positive = int(
                        (probabilities[left:right] >= config.window_threshold).sum()
                    )
                    dense[index] = (
                        positive >= config.min_positive
                        and coverage >= config.coverage_min
                    )
            else:
                density_size = int(round(config.density_ms / config.stride_ms))
                positive = np.convolve(
                    (probabilities >= config.window_threshold).astype(np.int64),
                    np.ones(density_size, dtype=np.int64),
                    mode="same",
                )[: len(segment)]
                coverage = np.convolve(
                    observed.astype(np.float64),
                    np.ones(density_size, dtype=np.float64) / density_size,
                    mode="same",
                )[: len(segment)]
                dense = (
                    (positive >= config.min_positive)
                    & (coverage >= config.coverage_min)
                )

            index = 0
            while index < len(segment):
                if not dense[index]:
                    index += 1
                    continue
                end_index = index
                while end_index < len(segment) and dense[end_index]:
                    end_index += 1
                support = np.where(
                    probabilities[index:end_index] >= config.window_threshold
                )[0]
                if not len(support):
                    index = end_index
                    continue
                first = index + int(support[0])
                last = index + int(support[-1])
                event_start = int(starts[first])
                event_end = int(starts[last] + config.window_ms)
                quality = _candidate_quality(
                    starts,
                    observed,
                    event_start,
                    event_end,
                    config,
                )
                centers = np.array(
                    [(start + end) // 2 for start, end, _ in raw_windows],
                    dtype=np.int64,
                )
                pre_count = int(
                    ((centers >= event_start - config.context_ms) & (centers < event_start)).sum()
                )
                post_count = int(
                    ((centers >= event_end) & (centers < event_end + config.context_ms)).sum()
                )
                candidate = CandidateEvent(
                    event=EventRef(sid, event_start, event_end),
                    probabilities=tuple(float(x) for x in probabilities[first : last + 1]),
                    observed_fraction=quality[0],
                    bridged_gap_count=quality[1],
                    bridged_gap_ms=quality[2],
                    pre_observed_count=pre_count,
                    post_observed_count=post_count,
                )
                if (
                    sid_candidates
                    and event_start - sid_candidates[-1].event.end_ms
                    <= config.merge_ms
                ):
                    previous = sid_candidates.pop()
                    merged_start = previous.event.start_ms
                    merged_end = max(previous.event.end_ms, event_end)
                    merged_quality = _candidate_quality(
                        starts,
                        observed,
                        merged_start,
                        merged_end,
                        config,
                    )
                    candidate = CandidateEvent(
                        event=EventRef(sid, merged_start, merged_end),
                        probabilities=previous.probabilities + candidate.probabilities,
                        observed_fraction=merged_quality[0],
                        bridged_gap_count=merged_quality[1],
                        bridged_gap_ms=merged_quality[2],
                        pre_observed_count=previous.pre_observed_count,
                        post_observed_count=candidate.post_observed_count,
                    )
                sid_candidates.append(candidate)
                index = end_index
        candidates.extend(sid_candidates)
    return candidates


def _validate_micro_config(config: MicroCandidateConfig) -> None:
    values = {
        "stride_ms": config.stride_ms,
        "window_ms": config.window_ms,
        "smooth_sigma_ms": config.smooth_sigma_ms,
        "smooth_radius_ms": config.smooth_radius_ms,
        "merge_ms": config.merge_ms,
        "min_duration_ms": config.min_duration_ms,
        "context_ms": config.context_ms,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in values.values()
    ):
        raise ValueError("micro candidate durations must be integer milliseconds")
    if config.stride_ms <= 0:
        raise ValueError("micro stride must be positive")
    if config.window_ms <= 0 or config.min_duration_ms <= 0:
        raise ValueError("micro window and minimum duration must be positive")
    if any(
        value < 0
        for name, value in values.items()
        if name not in {"stride_ms", "window_ms", "min_duration_ms"}
    ):
        raise ValueError(
            "micro smoothing, merge, and context durations must be non-negative"
        )
    if any(value % config.stride_ms for value in values.values()):
        raise ValueError("micro candidate durations must be multiples of stride_ms")


def _gaussian_kernel(config: MicroCandidateConfig) -> np.ndarray:
    if config.smooth_sigma_ms == 0 or config.smooth_radius_ms == 0:
        return np.ones(1, dtype=np.float64)
    offsets = np.arange(
        -config.smooth_radius_ms,
        config.smooth_radius_ms + config.stride_ms,
        config.stride_ms,
    )
    kernel = np.exp(-0.5 * (offsets / config.smooth_sigma_ms) ** 2)
    return kernel / kernel.sum()


def _normalized_smooth(values: Sequence[float] | np.ndarray, kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(kernel) == 1:
        return values
    start = (len(kernel) - 1) // 2
    numerator = np.convolve(values, kernel, mode="full")[start : start + len(values)]
    denominator = np.convolve(np.ones(len(values)), kernel, mode="full")[
        start : start + len(values)
    ]
    return numerator / np.maximum(denominator, 1e-12)


def _micro_candidate(
    sid: str,
    starts: np.ndarray,
    ends: np.ndarray,
    smoothed: np.ndarray,
    first: int,
    last: int,
    centers: np.ndarray,
    config: MicroCandidateConfig,
) -> CandidateEvent:
    """Construct one micro candidate from an inclusive support-index range."""

    event_start = int(starts[first])
    event_end = int(ends[last])
    gaps = np.diff(starts[first : last + 1]) - config.stride_ms
    gap_mask = gaps > 0
    return CandidateEvent(
        event=EventRef(sid, event_start, event_end),
        probabilities=tuple(float(value) for value in smoothed[first : last + 1]),
        observed_fraction=1.0,
        bridged_gap_count=int(gap_mask.sum()),
        bridged_gap_ms=int(gaps[gap_mask].sum()),
        pre_observed_count=int(
            ((centers >= event_start - config.context_ms) & (centers < event_start)).sum()
        ),
        post_observed_count=int(
            ((centers >= event_end) & (centers < event_end + config.context_ms)).sum()
        ),
    )


def micro_candidates(
    windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    threshold: float,
    config: MicroCandidateConfig | None = None,
) -> list[CandidateEvent]:
    """Build smoothed micro candidates without crossing acquisition gaps."""

    config = config or MicroCandidateConfig()
    _validate_micro_config(config)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("micro threshold must be a finite value in [0, 1]") from exc
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("micro threshold must be a finite value in [0, 1]")

    kernel = _gaussian_kernel(config)
    candidates: list[CandidateEvent] = []
    for sid in sorted(windows_by_sid):
        raw_windows = sorted(windows_by_sid[sid], key=lambda row: (row[0], row[1]))
        normalized_windows: list[tuple[int, int, float]] = []
        for row in raw_windows:
            if len(row) != 3:
                raise ValueError("micro windows must be (start_ms, end_ms, probability)")
            start, end, probability = row
            try:
                start_ms = int(start)
                end_ms = int(end)
                score = float(probability)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("micro windows must contain numeric values") from exc
            if end_ms <= start_ms:
                raise ValueError("micro windows must have strictly positive durations")
            if not np.isfinite(score):
                raise ValueError("micro probabilities must contain only finite values")
            normalized_windows.append((start_ms, end_ms, score))
        if not normalized_windows:
            continue

        segments: list[list[tuple[int, int, float]]] = []
        segment: list[tuple[int, int, float]] = []
        for row in normalized_windows:
            if segment and row[0] - segment[-1][0] > 2 * config.stride_ms:
                segments.append(segment)
                segment = []
            segment.append(row)
        if segment:
            segments.append(segment)

        centers = np.asarray(
            [(start + end) // 2 for start, end, _ in normalized_windows],
            dtype=np.int64,
        )
        for acquisition_segment in segments:
            starts = np.asarray([row[0] for row in acquisition_segment], dtype=np.int64)
            ends = np.asarray([row[1] for row in acquisition_segment], dtype=np.int64)
            scores = np.asarray([row[2] for row in acquisition_segment], dtype=np.float64)
            smoothed = _normalized_smooth(scores, kernel)
            if not np.isfinite(smoothed).all():
                raise ValueError("micro smoothed probabilities must be finite")
            above = smoothed >= threshold

            retained: list[CandidateEvent] = []
            index = 0
            while index < len(acquisition_segment):
                if not above[index]:
                    index += 1
                    continue
                end_index = index + 1
                while end_index < len(acquisition_segment) and above[end_index]:
                    end_index += 1
                candidate = _micro_candidate(
                    sid,
                    starts,
                    ends,
                    smoothed,
                    index,
                    end_index - 1,
                    centers,
                    config,
                )
                if candidate.event.end_ms - candidate.event.start_ms >= config.min_duration_ms:
                    retained.append(candidate)
                index = end_index

            merged: list[CandidateEvent] = []
            for candidate in retained:
                if (
                    merged
                    and candidate.event.start_ms - merged[-1].event.end_ms
                    <= config.merge_ms
                ):
                    previous = merged.pop()
                    merged.append(
                        CandidateEvent(
                            event=EventRef(
                                sid,
                                previous.event.start_ms,
                                max(previous.event.end_ms, candidate.event.end_ms),
                            ),
                            probabilities=previous.probabilities + candidate.probabilities,
                            observed_fraction=1.0,
                            bridged_gap_count=(
                                previous.bridged_gap_count
                                + candidate.bridged_gap_count
                            ),
                            bridged_gap_ms=(
                                previous.bridged_gap_ms + candidate.bridged_gap_ms
                            ),
                            pre_observed_count=previous.pre_observed_count,
                            post_observed_count=candidate.post_observed_count,
                        )
                    )
                else:
                    merged.append(candidate)
            candidates.extend(merged)
    return candidates


def select_micro_candidate_threshold(
    windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    truths: Sequence[EventRef],
    thresholds: Sequence[float],
    config: MicroCandidateConfig | None = None,
) -> MicroCandidateSelection:
    """Select a registered micro threshold using only supplied inner truths."""

    try:
        options = tuple(float(value) for value in thresholds)
    except (TypeError, ValueError) as exc:
        raise ValueError("micro thresholds must be unique finite values in [0, 1]") from exc
    if (
        not options
        or len(set(options)) != len(options)
        or any(not np.isfinite(value) or value < 0 or value > 1 for value in options)
    ):
        raise ValueError("micro thresholds must be unique finite values in [0, 1]")

    budget = 3 * max(len(truths), 1)
    feasible: list[MicroCandidateSelection] = []
    fallback: list[MicroCandidateSelection] = []
    for threshold in options:
        candidates = micro_candidates(windows_by_sid, threshold, config)
        metrics = compute_event_metrics([candidate.event for candidate in candidates], truths)
        selection = MicroCandidateSelection(threshold, metrics, len(candidates))
        fallback.append(selection)
        if len(candidates) <= budget:
            feasible.append(selection)

    if feasible:
        return max(
            feasible,
            key=lambda item: (
                item.metrics.sensitivity,
                -item.candidate_count,
                item.threshold,
            ),
        )
    return max(
        fallback,
        key=lambda item: (item.metrics.f1, item.metrics.ppv, item.threshold),
    )


def _longest_above(probabilities: np.ndarray, threshold: float) -> int:
    best = current = 0
    for value in probabilities >= threshold:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def verifier_features(
    candidates: Sequence[CandidateEvent],
    windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    include_coverage: bool = False,
    tcn_scores_by_sid: Mapping[str, Sequence[tuple[int, float]]] | None = None,
) -> np.ndarray:
    """Build the existing verifier feature surface plus optional gap features."""

    rows: list[list[float]] = []
    for candidate in candidates:
        probabilities = np.asarray(candidate.probabilities, dtype=np.float64)
        if len(probabilities) < 2:
            continue
        event = candidate.event
        duration = len(probabilities) * 15.0
        features = [
            duration,
            float(len(probabilities)),
            float(probabilities.max() - probabilities.min()),
            float(probabilities.std() / (probabilities.mean() + 1e-9)),
        ]
        positions = np.arange(len(probabilities))
        slope = (
            float(np.polyfit(positions, probabilities, 1)[0])
            if len(probabilities) > 2
            else 0.0
        )
        features.extend(
            [
                slope,
                float(probabilities[0] - probabilities[-1]),
                float((probabilities >= 0.35).mean()),
                float((probabilities >= 0.45).mean()),
                float(_longest_above(probabilities, 0.35)),
                float(_longest_above(probabilities, 0.45)),
                float(np.maximum(probabilities - 0.28838, 0.0).sum()),
                float(probabilities.mean()),
                float(probabilities.max()),
                float(probabilities.std()),
                float(np.percentile(probabilities, 10)),
                float(np.percentile(probabilities, 50)),
                float(np.percentile(probabilities, 90)),
            ]
        )
        session_windows = sorted(windows_by_sid.get(event.sid, ()))
        centers = np.array(
            [(start + end) // 2 for start, end, _ in session_windows],
            dtype=np.int64,
        )
        session_probabilities = np.array(
            [probability for _, _, probability in session_windows],
            dtype=np.float64,
        )
        before = session_probabilities[
            (centers >= event.start_ms - 1_200_000) & (centers < event.start_ms)
        ]
        after = session_probabilities[
            (centers >= event.end_ms) & (centers < event.end_ms + 1_200_000)
        ]
        for context in (before, after):
            if len(context) >= 5:
                features.extend(
                    [
                        float(context.mean()),
                        float(context.max()),
                        float(context.std()),
                        float(np.percentile(context, 10)),
                        float(np.percentile(context, 50)),
                        float(np.percentile(context, 90)),
                    ]
                )
            else:
                features.extend([0.0] * 6)
        before_safe = before if len(before) else np.zeros(1)
        after_safe = after if len(after) else np.zeros(1)
        both = np.concatenate((before_safe, after_safe))
        features.extend(
            [
                float(probabilities.mean() - both.mean()),
                float(probabilities.max() - np.percentile(both, 90)),
                float(probabilities.mean() - before_safe.mean()),
                float(probabilities.mean() - after_safe.mean()),
            ]
        )
        tcn_values: list[float] = []
        if tcn_scores_by_sid is not None:
            tcn_values = [
                float(value)
                for timestamp, value in tcn_scores_by_sid.get(event.sid, ())
                if timestamp >= event.start_ms - 1000 and timestamp < event.end_ms
            ]
        features.extend(
            [
                max(tcn_values) if tcn_values else 0.0,
                float(np.mean(tcn_values)) if tcn_values else 0.0,
            ]
        )
        hour = (event.start_ms / 3.6e6) % 24
        features.extend([float(hour), float(_GLOBAL_PRIOR[int(hour) % 24])])
        if include_coverage:
            features.extend(
                [
                    candidate.observed_fraction,
                    float(candidate.bridged_gap_count),
                    float(candidate.bridged_gap_ms),
                    float(candidate.pre_observed_count),
                    float(candidate.post_observed_count),
                ]
            )
        rows.append(features)
    width = 42 if include_coverage else 37
    return np.asarray(rows, dtype=np.float64).reshape((-1, width))


def aggregate_candidate_features(
    candidates: Sequence[CandidateEvent],
    windows: Sequence[EventRef],
    window_features: np.ndarray,
    context_ms: int = 1_200_000,
) -> np.ndarray:
    """Summarize aligned raw window features inside and around each candidate."""

    features = np.asarray(window_features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("window_features must be two-dimensional")
    if len(features) != len(windows):
        raise ValueError("window_features must align with windows")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must not contain duplicate references")
    if context_ms < 0:
        raise ValueError("context_ms must be non-negative")

    rows_by_sid: dict[str, list[int]] = {}
    for index, window in enumerate(windows):
        rows_by_sid.setdefault(window.sid, []).append(index)

    width = features.shape[1]
    output: list[np.ndarray] = []
    for candidate in candidates:
        event = candidate.event
        session_rows = np.asarray(rows_by_sid.get(event.sid, ()), dtype=np.int64)
        if len(session_rows):
            session_windows = [windows[index] for index in session_rows]
            centers = np.asarray(
                [
                    (window.start_ms + window.end_ms) // 2
                    for window in session_windows
                ],
                dtype=np.int64,
            )
            event_mask = np.asarray(
                [
                    window.start_ms >= event.start_ms
                    and window.end_ms <= event.end_ms
                    for window in session_windows
                ],
                dtype=bool,
            )
            context_mask = (
                (
                    (centers >= event.start_ms - context_ms)
                    & (centers < event.start_ms)
                )
                | (
                    (centers >= event.end_ms)
                    & (centers < event.end_ms + context_ms)
                )
            )
            event_values = features[session_rows[event_mask]]
            context_values = features[session_rows[context_mask]]
        else:
            event_values = np.empty((0, width), dtype=np.float64)
            context_values = np.empty((0, width), dtype=np.float64)

        if len(event_values):
            with np.errstate(invalid="ignore"):
                event_mean = np.nanmean(event_values, axis=0)
                event_std = np.nanstd(event_values, axis=0)
                event_p10 = np.nanpercentile(event_values, 10, axis=0)
                event_p90 = np.nanpercentile(event_values, 90, axis=0)
        else:
            event_mean = event_std = event_p10 = event_p90 = np.full(
                width, np.nan
            )
        if len(context_values):
            with np.errstate(invalid="ignore"):
                contrast = event_mean - np.nanmean(context_values, axis=0)
        else:
            contrast = np.full(width, np.nan)
        output.append(
            np.concatenate(
                (
                    event_mean,
                    event_std,
                    event_p10,
                    event_p90,
                    contrast,
                    [float(len(event_values)), float(len(context_values))],
                )
            )
        )
    return np.asarray(output, dtype=np.float64).reshape((-1, width * 5 + 2))
