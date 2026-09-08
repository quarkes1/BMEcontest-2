import numpy as np

from src.pipeline.event_stack import (
    DensityConfig,
    EventRef,
    apply_event_policy,
    compute_event_metrics,
    density_candidates,
    select_event_policy,
    select_event_threshold,
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
