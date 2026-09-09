import numpy as np
import pytest

from src.pipeline.event_stack import (
    DensityConfig,
    EventRef,
    CandidateEvent,
    MicroCandidateConfig,
    MultiScaleCandidate,
    aggregate_candidate_features,
    apply_event_policy,
    compute_event_metrics,
    density_candidates,
    micro_candidates,
    select_event_policy,
    select_event_threshold,
    select_micro_candidate_threshold,
    multiscale_verifier_features,
    union_candidates,
    verifier_features,
)


def test_compute_event_metrics_matches_once_per_truth():
    truths = [EventRef("s1", 100, 200)]
    predictions = [EventRef("s1", 100, 200), EventRef("s1", 110, 190)]

    metrics = compute_event_metrics(predictions, truths, iou_threshold=0.25)

    assert (metrics.n_tp, metrics.n_pred, metrics.n_true) == (1, 2, 1)
    assert metrics.f1 == 2 / 3


def test_select_event_threshold_prefers_stricter_equal_f1_choice():
    candidates = [
        EventRef("s1", 100, 200),
        EventRef("s2", 100, 200),
        EventRef("s3", 100, 200),
        EventRef("s4", 100, 200),
    ]
    truths = [EventRef("s1", 100, 200), EventRef("s2", 100, 200)]

    selected = select_event_threshold(
        candidates,
        np.array([0.9, 0.7, 0.7, 0.7]),
        truths,
    )

    assert selected.threshold == 0.9
    assert selected.metrics.n_tp == selected.metrics.n_pred == 1


def test_density_boundaries_use_threshold_support():
    windows = {
        "s1": [
            (
                i * 15_000,
                i * 15_000 + 240_000,
                0.8 if 10 <= i <= 19 else 0.1,
            )
            for i in range(40)
        ]
    }

    result = density_candidates(
        windows,
        DensityConfig(density_ms=600_000, min_positive=10),
    )

    assert [(c.event.start_ms, c.event.end_ms) for c in result] == [
        (180_000, 525_000)
    ]


def test_coverage_fix_does_not_zero_pad_segment_edges():
    windows = {
        "s1": [
            (i * 15_000, i * 15_000 + 240_000, 0.8)
            for i in range(10)
        ]
    }

    legacy = density_candidates(windows, DensityConfig(coverage_fix=False))
    fixed = density_candidates(windows, DensityConfig(coverage_fix=True))

    assert legacy == []
    assert len(fixed) == 1
    assert fixed[0].observed_fraction == 1.0


def test_coverage_metadata_counts_internal_gap():
    starts = [
        0,
        15_000,
        30_000,
        60_000,
        75_000,
        90_000,
        105_000,
        120_000,
        135_000,
        150_000,
    ]

    result = density_candidates(
        {"s1": [(start, start + 240_000, 0.8) for start in starts]},
        DensityConfig(coverage_fix=True, min_positive=10),
    )

    assert result[0].bridged_gap_count == 1
    assert result[0].bridged_gap_ms == 15_000


def test_micro_candidates_never_bridge_more_than_two_strides():
    rows = [
        (index * 7_500, index * 7_500 + 15_000, 0.9)
        for index in range(8)
    ]
    rows += [
        (180_000 + index * 7_500, 195_000 + index * 7_500, 0.9)
        for index in range(8)
    ]
    config = MicroCandidateConfig(
        smooth_sigma_ms=0,
        smooth_radius_ms=0,
        merge_ms=180_000,
        min_duration_ms=60_000,
    )

    result = micro_candidates({"s1": rows}, threshold=0.5, config=config)

    assert [(item.event.start_ms, item.event.end_ms) for item in result] == [
        (0, 67_500),
        (180_000, 247_500),
    ]


def test_micro_candidates_use_normalized_gaussian_smoothing():
    rows = [
        (0, 15_000, 0.0),
        (7_500, 22_500, 1.0),
        (15_000, 30_000, 0.0),
    ]
    config = MicroCandidateConfig(
        smooth_sigma_ms=7_500,
        smooth_radius_ms=7_500,
        merge_ms=0,
        min_duration_ms=15_000,
    )

    result = micro_candidates({"s1": rows}, threshold=0.4, config=config)

    assert [(item.event.start_ms, item.event.end_ms) for item in result] == [
        (7_500, 22_500)
    ]
    np.testing.assert_allclose(result[0].probabilities, [0.45186276])


def test_micro_candidates_smooth_each_acquisition_segment_independently():
    rows = [
        (0, 15_000, 1.0),
        (7_500, 22_500, 1.0),
        (30_000, 45_000, 0.0),
    ]
    config = MicroCandidateConfig(
        smooth_sigma_ms=7_500,
        smooth_radius_ms=7_500,
        merge_ms=0,
        min_duration_ms=15_000,
    )

    result = micro_candidates({"s1": rows}, threshold=0.9, config=config)

    assert [(item.event.start_ms, item.event.end_ms) for item in result] == [
        (0, 22_500)
    ]
    np.testing.assert_allclose(result[0].probabilities, [1.0, 1.0])


def test_micro_candidates_merge_runs_within_registered_gap():
    rows = [
        (
            start,
            start + 15_000,
            0.9 if start < 60_000 or start >= 120_000 else 0.1,
        )
        for start in range(0, 180_000, 7_500)
    ]
    config = MicroCandidateConfig(
        smooth_sigma_ms=0,
        smooth_radius_ms=0,
        merge_ms=60_000,
        min_duration_ms=60_000,
    )

    result = micro_candidates({"s1": rows}, threshold=0.5, config=config)

    assert [(item.event.start_ms, item.event.end_ms) for item in result] == [
        (0, 187_500)
    ]


def test_micro_candidates_discard_short_runs_before_merging():
    starts = list(range(0, 45_000, 7_500)) + list(
        range(90_000, 142_500, 7_500)
    )
    rows = [(start, start + 15_000, 0.9) for start in starts]
    config = MicroCandidateConfig(
        smooth_sigma_ms=0,
        smooth_radius_ms=0,
        merge_ms=60_000,
        min_duration_ms=60_000,
    )

    result = micro_candidates({"s1": rows}, threshold=0.5, config=config)

    assert [(item.event.start_ms, item.event.end_ms) for item in result] == [
        (90_000, 150_000)
    ]


def test_micro_threshold_selection_maximizes_recall_with_candidate_budget():
    rows = {
        "s1": [
            (index * 7_500, index * 7_500 + 15_000, 0.35)
            for index in range(8)
        ],
        "s2": [
            (index * 7_500, index * 7_500 + 15_000, 0.15)
            for index in range(8)
        ],
    }
    truths = (EventRef("s1", 0, 67_500),)
    config = MicroCandidateConfig(smooth_sigma_ms=0, smooth_radius_ms=0)

    selected = select_micro_candidate_threshold(
        rows,
        truths,
        (0.1, 0.2, 0.3, 0.4),
        config,
    )

    assert selected.threshold == 0.3
    assert selected.metrics.sensitivity == 1.0
    assert selected.candidate_count == 1


def test_micro_threshold_selection_prefers_stricter_deterministic_tie():
    rows = {
        "s1": [
            (index * 7_500, index * 7_500 + 15_000, 0.9)
            for index in range(8)
        ]
    }
    truths = (EventRef("s1", 0, 67_500),)
    config = MicroCandidateConfig(smooth_sigma_ms=0, smooth_radius_ms=0)

    selected = select_micro_candidate_threshold(rows, truths, (0.2, 0.1), config)

    assert selected.threshold == 0.2


def test_micro_threshold_selection_uses_f1_when_every_choice_exceeds_budget():
    scores = (0.9, 0.8, 0.7, 0.6, 0.4, 0.3)
    rows = {
        f"s{sid}": [
            (index * 7_500, index * 7_500 + 15_000, score)
            for index in range(8)
        ]
        for sid, score in enumerate(scores, start=1)
    }
    truths = (EventRef("s1", 0, 67_500),)
    config = MicroCandidateConfig(smooth_sigma_ms=0, smooth_radius_ms=0)

    selected = select_micro_candidate_threshold(rows, truths, (0.2, 0.5), config)

    assert selected.threshold == 0.5
    assert selected.candidate_count == 4
    assert selected.metrics.f1 == 0.4


@pytest.mark.parametrize("thresholds", [(), (0.2, 0.2), (float("nan"),), (1.1,)])
def test_micro_threshold_selection_rejects_unregistered_grids(thresholds):
    with pytest.raises(ValueError, match="unique finite values"):
        select_micro_candidate_threshold({}, (), thresholds)


@pytest.mark.parametrize(
    "config",
    [
        MicroCandidateConfig(stride_ms=0),
        MicroCandidateConfig(window_ms=10_000),
        MicroCandidateConfig(smooth_sigma_ms=-7_500),
        MicroCandidateConfig(min_duration_ms=0),
    ],
)
def test_micro_candidates_require_valid_stride_multiple_configuration(config):
    with pytest.raises(ValueError):
        micro_candidates({}, threshold=0.5, config=config)


@pytest.mark.parametrize(
    "row, message",
    [
        ((0, 0, 0.5), "strictly positive durations"),
        ((0, 15_000, float("nan")), "finite values"),
    ],
)
def test_micro_candidates_validate_window_rows(row, message):
    with pytest.raises(ValueError, match=message):
        micro_candidates({"s1": [row]}, threshold=0.5)


def test_verifier_feature_dimensions_are_explicit():
    windows = {
        "s1": [
            (
                i * 15_000,
                i * 15_000 + 240_000,
                0.8 if 10 <= i <= 19 else 0.1,
            )
            for i in range(40)
        ]
    }
    candidates = density_candidates(windows, DensityConfig())

    legacy = verifier_features(candidates, windows, include_coverage=False)
    covered = verifier_features(candidates, windows, include_coverage=True)

    assert legacy.shape == (1, 37)
    assert covered.shape == (1, 42)


def test_union_preserves_macro_geometry_and_keeps_micro_only_rescue():
    macro = [_candidate("s1", 100, 300)]
    micro = [_candidate("s1", 150, 250), _candidate("s1", 500, 620)]

    result = union_candidates(macro, micro, merge_iou=0.25)

    assert [item.event for item in result] == [
        EventRef("s1", 100, 300),
        EventRef("s1", 500, 620),
    ]
    assert result[0].macro is macro[0] and result[0].micro is micro[0]
    assert result[1].macro is None and result[1].micro is micro[1]


def test_union_is_independent_of_input_order():
    macro = [_candidate("s1", 0, 100), _candidate("s1", 200, 300)]
    micro = [_candidate("s1", 10, 90), _candidate("s1", 210, 290)]

    assert union_candidates(macro, micro) == union_candidates(
        tuple(reversed(macro)), tuple(reversed(micro))
    )


def test_union_uses_canonical_evidence_ties_for_duplicate_geometry():
    low_evidence = CandidateEvent(
        EventRef("s1", 0, 100), (0.1,), 1.0, 0, 0, 1, 1
    )
    high_evidence = CandidateEvent(
        EventRef("s1", 0, 100), (0.2,), 1.0, 0, 0, 1, 1
    )
    micro = [_candidate("s1", 10, 90)]

    forward = union_candidates([low_evidence, high_evidence], micro)
    reverse = union_candidates([high_evidence, low_evidence], micro)

    assert forward == reverse
    assert forward[0].macro is high_evidence


def test_multiscale_verifier_has_exact_width_and_distinct_source_missing_flags():
    candidate = MultiScaleCandidate(
        EventRef("s1", 0, 60_000), None, _candidate("s1", 0, 60_000)
    )
    micro_windows = {
        "s1": [
            (index * 7_500, index * 7_500 + 15_000, 0.8)
            for index in range(8)
        ]
    }

    result = multiscale_verifier_features(
        [candidate], macro_windows_by_sid={}, micro_windows_by_sid=micro_windows
    )

    assert result.shape == (1, 56)
    assert np.isfinite(result).all()
    assert result[0, 37:39].tolist() == [0.0, 1.0]
    assert result[0, 54:56].tolist() == [1.0, 0.0]


def test_multiscale_verifier_keeps_macro_only_candidate_without_micro_samples():
    candidate = MultiScaleCandidate(
        EventRef("s1", 0, 240_000), _candidate("s1", 0, 240_000), None
    )
    macro_windows = {
        "s1": [
            (index * 15_000, index * 15_000 + 240_000, 0.7)
            for index in range(4)
        ]
    }

    result = multiscale_verifier_features([candidate], macro_windows, {})

    assert result.shape == (1, 56)
    assert result[0, 37:39].tolist() == [1.0, 0.0]
    assert result[0, 54:56].tolist() == [0.0, 1.0]


def test_multiscale_verifier_zero_sample_micro_stream_zeros_score_features():
    candidate = MultiScaleCandidate(
        EventRef("s1", 0, 60_000), None, _candidate("s1", 0, 60_000)
    )
    micro_windows = {"s1": [(60_000, 75_000, 0.9)]}

    result = multiscale_verifier_features([candidate], {}, micro_windows)

    assert result[0, 37:39].tolist() == [0.0, 1.0]
    assert result[0, 39] == 0.0
    np.testing.assert_allclose(result[0, 41:54], 0.0)
    assert result[0, 54:56].tolist() == [1.0, 1.0]


def test_multiscale_verifier_one_sample_micro_stream_zeros_score_features():
    candidate = MultiScaleCandidate(
        EventRef("s1", 0, 60_000), None, _candidate("s1", 0, 60_000)
    )
    micro_windows = {
        "s1": [(0, 15_000, 0.4), (60_000, 75_000, 0.9)]
    }

    result = multiscale_verifier_features([candidate], {}, micro_windows)

    assert result[0, 37:39].tolist() == [0.0, 1.0]
    assert result[0, 39] == 1.0
    np.testing.assert_allclose(result[0, 41:54], 0.0)
    assert result[0, 54:56].tolist() == [1.0, 1.0]


def test_apply_event_policy_caps_each_subject_after_thresholding():
    candidates = [
        EventRef("session-a", index * 100, index * 100 + 50)
        for index in range(3)
    ] + [EventRef("session-b", 0, 50)]
    scores = np.array([0.7, 0.9, 0.8, 0.6])
    groups = np.array(["subject-a", "subject-a", "subject-a", "subject-b"])

    selected = apply_event_policy(
        candidates,
        scores,
        groups,
        threshold=0.5,
        max_events_per_group=2,
    )

    assert selected == [candidates[1], candidates[2], candidates[3]]


def test_select_event_policy_uses_only_registered_caps():
    candidates = [
        EventRef("session-a", 0, 100),
        EventRef("session-a", 200, 300),
        EventRef("session-b", 0, 100),
        EventRef("session-b", 200, 300),
    ]
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    groups = np.array(["subject-a", "subject-a", "subject-b", "subject-b"])
    truths = [candidates[0], candidates[2]]

    selected = select_event_policy(
        candidates,
        scores,
        truths,
        groups,
        max_events_options=(1, 2),
    )

    assert selected.max_events_per_group == 1
    assert selected.max_events_per_group in (1, 2)
    assert selected.metrics.f1 == 1.0


def _candidate(sid: str, start_ms: int, end_ms: int) -> CandidateEvent:
    return CandidateEvent(
        event=EventRef(sid, start_ms, end_ms),
        probabilities=(0.7, 0.8),
        observed_fraction=1.0,
        bridged_gap_count=0,
        bridged_gap_ms=0,
        pre_observed_count=1,
        post_observed_count=1,
    )


def test_aggregate_candidate_features_uses_only_aligned_session_context():
    windows = (
        EventRef("s1", -100, 0),
        EventRef("s1", 0, 100),
        EventRef("s1", 100, 200),
        EventRef("s1", 200, 300),
        EventRef("s2", 0, 100),
    )
    features = np.array(
        [[-1.0, -10.0], [1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [999.0, 999.0]]
    )

    result = aggregate_candidate_features(
        [_candidate("s1", 0, 200)], windows, features, context_ms=100
    )

    assert result.shape == (1, 12)
    np.testing.assert_allclose(
        result[0],
        [2.0, 20.0, 1.0, 10.0, 1.2, 12.0, 2.8, 28.0, 0.0, 0.0, 2.0, 2.0],
    )


def test_aggregate_candidate_features_marks_missing_context():
    windows = (EventRef("s1", 0, 100), EventRef("s1", 100, 200))
    features = np.array([[1.0], [3.0]])

    result = aggregate_candidate_features(
        [_candidate("s1", 0, 200)], windows, features, context_ms=100
    )

    assert result.shape == (1, 7)
    assert np.isnan(result[0, 4])
    assert result[0, -2:].tolist() == [2.0, 0.0]
