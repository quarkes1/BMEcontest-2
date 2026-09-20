import hashlib
import json

import numpy as np
import pytest

from src.pipeline.context_features import (
    CONTEXT_V1_COLUMNS,
    CONTEXT_V1_CONTEXT_MS,
    CONTEXT_V1_RUN_GAP_MS,
    CONTEXT_V1_MIN_RUN_WINDOWS,
    CONTEXT_V1_SCHEMA_HASH,
    MACRO_STRIDE_MS,
    MACRO_WINDOW_MS,
    MICRO_STRIDE_MS,
    MICRO_WINDOW_MS,
    context_v1_features,
)
from src.pipeline.context_features_v2 import (
    CONTEXT_V2_COLUMNS,
    CONTEXT_V2_SCHEMA_HASH,
    context_schema_descriptor,
    context_v2_features,
)
from src.pipeline.event_stack import CandidateEvent, EventRef, MultiScaleCandidate, union_candidates


def _idx(name):
    return CONTEXT_V1_COLUMNS.index(name)


def test_context_v1_schema_is_fixed_and_hashed():
    assert len(CONTEXT_V1_COLUMNS) == 60
    assert CONTEXT_V1_COLUMNS[:4] == (
        "macro_pre_mean", "macro_pre_max", "macro_pre_std",
        "macro_pre_above_fraction",
    )
    assert CONTEXT_V1_COLUMNS[30] == "micro_pre_mean"
    expected = hashlib.sha256(
        json.dumps(CONTEXT_V1_COLUMNS, separators=(",", ":")).encode()
    ).hexdigest()
    assert CONTEXT_V1_SCHEMA_HASH == expected


def test_context_v1_uses_same_session_clipped_real_time_regions():
    candidate = EventRef("s1", 1_200_000, 1_800_000)
    macro = {
        "s1": [(0, 30_000, .1), (1_300_000, 1_330_000, .8), (1_900_000, 1_930_000, .2)],
        "s2": [(1_300_000, 1_330_000, 1.0)],
    }
    row = context_v1_features([candidate], macro, {}, {"s1": (0, 3_000_000)})[0]
    assert row[_idx("macro_candidate_mean")] == pytest.approx(.8)
    assert row[_idx("macro_post_mean")] == pytest.approx(.2)


def test_context_v1_empty_scale_has_nan_statistics_and_zero_coverage():
    event = EventRef("s", 1_000_000, 1_100_000)
    row = context_v1_features([event], {}, {}, {"s": (0, 3_000_000)})[0]
    assert np.isnan(row[:4]).all()
    assert np.isnan(row[12:20]).all()
    assert row[_idx("macro_pre_coverage")] == 0
    assert row[_idx("macro_candidate_coverage")] == 0
    assert row[_idx("macro_post_coverage")] == 0
    assert row[_idx("macro_neighbor_run_count")] == 0
    assert row[_idx("macro_neighbor_run_total_duration_s")] == 0
    assert np.isnan(row[_idx("macro_neighbor_run_max_duration_s")])
    assert np.isnan(row[_idx("macro_pre_coverage") + 9])


def test_context_v1_clips_regions_to_explicit_session_bounds():
    event = EventRef("s", 100_000, 200_000)
    macro = {"s": [(60_000, 70_000, .4), (210_000, 220_000, .6)]}
    row = context_v1_features([event], macro, {}, {"s": (50_000, 220_000)})[0]
    assert row[_idx("macro_pre_mean")] == pytest.approx(.4)
    assert row[_idx("macro_post_mean")] == pytest.approx(.6)


def test_context_v1_score_gap_does_not_redefine_explicit_session_boundary():
    event = EventRef("s", 100_000, 200_000)
    macro = {"s": [(0, 20_000, .4), (500_000, 520_000, .6)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 1_000_000)})[0]
    assert row[_idx("macro_pre_mean")] == pytest.approx(.4)
    assert row[_idx("macro_post_mean")] == pytest.approx(.6)


def test_context_v1_uses_fixed_thresholds_for_above_fraction():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(1_000_000, 1_010_000, .28838), (1_020_000, 1_030_000, .288379)]}
    micro = {"s": [(1_000_000, 1_005_000, .2), (1_010_000, 1_015_000, .19999)]}
    row = context_v1_features([event], macro, micro, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_candidate_above_fraction")] == pytest.approx(.5)
    assert row[_idx("micro_candidate_above_fraction")] == pytest.approx(.5)


def test_context_v1_joins_two_window_run_at_sixty_second_gap():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(810_000, 820_000, .5), (870_000, 880_000, .5), (1_300_000, 1_310_000, .5)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_neighbor_run_count")] == 1
    assert row[_idx("macro_neighbor_run_total_duration_s")] == pytest.approx(300)


def test_context_v1_excludes_candidate_own_intersecting_run_from_neighbors():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(900_000, 910_000, .5), (950_000, 960_000, .5),
                     (1_300_000, 1_310_000, .5), (1_350_000, 1_360_000, .5)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_neighbor_run_count")] == 1
    assert row[_idx("macro_pre_mean")] == pytest.approx(.5)


def test_context_v1_no_neighbor_has_nan_distances_and_max_duration():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(900_000, 910_000, .5), (1_050_000, 1_060_000, .5)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_neighbor_run_count")] == 0
    assert row[_idx("macro_neighbor_run_total_duration_s")] == 0
    assert np.isnan(row[_idx("macro_neighbor_run_max_duration_s")])
    assert np.isnan(row[_idx("macro_preceding_run_distance_s")])
    assert np.isnan(row[_idx("macro_following_run_distance_s")])


def test_context_v1_center_half_mass_and_first_minus_second_mean():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(1_000_000, 1_025_000, 1.0), (1_025_000, 1_050_000, 3.0),
                     (1_050_000, 1_075_000, 5.0), (1_075_000, 1_100_000, 7.0)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_candidate_center_half_mass_fraction")] == pytest.approx(0.5)
    assert row[_idx("macro_candidate_first_minus_second_mean")] == pytest.approx(-4.0)


def test_context_v1_is_stable_under_input_ordering_and_accepts_wrapped_candidates():
    events = [EventRef("s", 1_000_000, 1_100_000), EventRef("s", 1_200_000, 1_300_000)]
    windows = [(1_000_000, 1_010_000, .5), (1_050_000, 1_060_000, .7), (1_200_000, 1_210_000, .4)]
    reverse = list(reversed(windows))
    bounds = {"s": (0, 2_000_000)}
    first = context_v1_features(events, {"s": windows}, {}, bounds)
    second = context_v1_features([type("C", (), {"event": e})() for e in events], {"s": reverse}, {}, bounds)
    np.testing.assert_equal(first, second)


def test_context_v1_exposes_named_window_and_stride_constants():
    assert (MACRO_STRIDE_MS, MACRO_WINDOW_MS) == (15_000, 240_000)
    assert (MICRO_STRIDE_MS, MICRO_WINDOW_MS) == (7_500, 15_000)
    assert CONTEXT_V1_CONTEXT_MS == 1_200_000
    assert CONTEXT_V1_RUN_GAP_MS == 60_000
    assert CONTEXT_V1_MIN_RUN_WINDOWS == 2


def test_context_v1_rejects_infinite_probability():
    event = EventRef("s", 1_000_000, 1_100_000)
    with pytest.raises(ValueError, match="infinity"):
        context_v1_features([event], {"s": [(1_000_000, 1_010_000, np.inf)]}, {}, {"s": (0, 2_000_000)})


def test_context_v1_rejects_nonfinite_window_timestamp():
    event = EventRef("s", 1_000_000, 1_100_000)
    with pytest.raises(ValueError, match="finite integer-like"):
        context_v1_features(
            [event],
            {"s": [(1_000_000, np.inf, .5)]},
            {},
            {"s": (0, 2_000_000)},
        )


def test_context_v1_rejects_nonfinite_event_boundary():
    event = EventRef("s", 1_000_000, np.inf)
    with pytest.raises(ValueError, match="finite integer-like"):
        context_v1_features([event], {}, {}, {"s": (0, 2_000_000)})


def test_context_v1_rejects_nonfinite_session_boundary():
    event = EventRef("s", 1_000_000, 1_100_000)
    with pytest.raises(ValueError, match="finite integer-like"):
        context_v1_features([event], {}, {}, {"s": (0, np.inf)})


def _source(sid, start, end, probabilities, coverage=1.0):
    return CandidateEvent(
        EventRef(sid, start, end), tuple(probabilities), coverage, 0, 0, 0, 0
    )


def _multi(sid, start, end, macro=None, micro=None):
    return MultiScaleCandidate(EventRef(sid, start, end), macro, micro)


def _v2_idx(name):
    return CONTEXT_V2_COLUMNS.index(name)


def test_context_v2_has_fixed_schema_descriptor_and_no_label_dependency():
    assert CONTEXT_V2_COLUMNS == (
        "macro_candidate_mean_percentile", "macro_candidate_max_percentile",
        "macro_candidate_mean_robust_z", "macro_candidate_mean_relative_to_session_median",
        "macro_candidate_max_relative_to_session_p90", "macro_candidate_mean_to_session_max",
        "macro_best_mean_margin", "macro_stronger_count_20m",
        "macro_nearest_stronger_distance_s", "micro_candidate_mean_percentile",
        "micro_candidate_max_percentile", "micro_candidate_mean_robust_z",
        "micro_candidate_mean_relative_to_session_median",
        "micro_candidate_max_relative_to_session_p90", "micro_candidate_mean_to_session_max",
        "micro_best_mean_margin", "micro_stronger_count_20m",
        "micro_nearest_stronger_distance_s", "session_candidate_count",
        "session_candidate_rate_per_hour", "candidate_duration_percentile",
        "candidate_duration_robust_z", "candidate_support_count",
        "candidate_support_coverage_percentile", "candidate_support_coverage_robust_z",
    )
    assert CONTEXT_V2_SCHEMA_HASH == hashlib.sha256(
        json.dumps(CONTEXT_V2_COLUMNS, separators=(",", ":")).encode()
    ).hexdigest()
    descriptor = context_schema_descriptor("v2")
    assert descriptor["width"] == 25
    assert descriptor["ordered_column_sha256"] == CONTEXT_V2_SCHEMA_HASH
    assert descriptor["label_dependency"] is False
    assert descriptor["session_local"] is True
    assert descriptor["learned"] is False


def test_context_v2_computes_hand_checked_source_competition_and_session_baselines():
    low = _source("s", 0, 1000, (.2, .4), .5)
    high = _source("s", 120_000, 121_000, (.8, 1.0), 1.0)
    candidates = [_multi("s", 0, 1000, macro=low), _multi("s", 120_000, 121_000, macro=high)]
    result = context_v2_features(
        candidates,
        {"s": [(0, 1, .1), (1, 2, .3), (2, 3, .5), (3, 4, .9)]},
        {}, {"s": (0, 3_600_000)},
    )
    row = result[0]
    assert row[_v2_idx("macro_candidate_mean_percentile")] == pytest.approx(.25)
    assert row[_v2_idx("macro_candidate_max_percentile")] == pytest.approx(.25)
    assert row[_v2_idx("macro_candidate_mean_robust_z")] == pytest.approx(-.3 / (1.4826 * .3 + .05))
    assert row[_v2_idx("macro_candidate_mean_relative_to_session_median")] == pytest.approx(-.25)
    assert row[_v2_idx("macro_candidate_max_relative_to_session_p90")] == pytest.approx((.4 - .78) / .78)
    assert row[_v2_idx("macro_candidate_mean_to_session_max")] == pytest.approx(1 / 3)
    assert row[_v2_idx("macro_best_mean_margin")] == pytest.approx(.6)
    assert row[_v2_idx("macro_stronger_count_20m")] == 1
    assert row[_v2_idx("macro_nearest_stronger_distance_s")] == pytest.approx(119)
    assert row[_v2_idx("session_candidate_count")] == 2
    assert row[_v2_idx("session_candidate_rate_per_hour")] == pytest.approx(2)
    assert row[_v2_idx("candidate_support_count")] == 1
    assert row[_v2_idx("candidate_support_coverage_percentile")] == pytest.approx(.25)


def test_context_v2_is_session_isolated_order_invariant_and_preserves_caller_rows():
    first = _multi("a", 1_000, 2_000, macro=_source("a", 1_000, 2_000, (.3,), .7))
    second = _multi("a", 4_000, 5_000, micro=_source("a", 4_000, 5_000, (.6,), .8))
    other = _multi("b", 1_000, 2_000, macro=_source("b", 1_000, 2_000, (.99,), .1))
    kwargs = dict(
        macro_windows_by_sid={"a": [(0, 1, .2), (1, 2, .4)], "b": [(0, 1, .99)]},
        micro_windows_by_sid={"a": [(0, 1, .5)], "b": []},
        session_bounds_by_sid={"a": (0, 10_000), "b": (0, 10_000)},
    )
    baseline = context_v2_features([first, second], **kwargs)
    reordered = context_v2_features([second, other, first], **kwargs)
    np.testing.assert_equal(baseline[0], reordered[2])
    np.testing.assert_equal(baseline[1], reordered[0])


@pytest.mark.parametrize(
    "candidate, windows, bounds, message",
    [
        (_multi("s", 0, 0, macro=_source("s", 0, 0, (.4,))), {}, {"s": (0, 1)}, "positive"),
        (_multi("s", 1, 2, macro=None), {}, {"s": (0, 3)}, "source"),
        (_multi("s", 1, 2, macro=_source("s", 1, 2, (np.nan,))), {}, {"s": (0, 3)}, "finite probability"),
        (_multi("s", 1, 2, macro=_source("s", 1, 2, (.4,))), {"s": [(2, 1, .2)]}, {"s": (0, 3)}, "ordered"),
        (_multi("s", 1, 2, macro=_source("s", 1, 2, (.4,))), {"s": [(1, 2, np.nan)]}, {"s": (0, 3)}, "finite"),
        (_multi("s", 1, 2, macro=_source("s", 1, 2, (.4,))), {}, {"s": (2, 3)}, "inside"),
    ],
)
def test_context_v2_fails_closed_for_invalid_raw_union_or_session_data(candidate, windows, bounds, message):
    with pytest.raises(ValueError, match=message):
        context_v2_features([candidate], windows, {}, bounds)


def test_context_v2_missing_source_is_nan_but_common_support_fields_are_defined():
    candidate = _multi("s", 1_000, 2_000, macro=_source("s", 1_000, 2_000, (.4,), 0.0))
    row = context_v2_features([candidate], {"s": []}, {}, {"s": (0, 2_000)})[0]
    assert np.isnan(row[_v2_idx("micro_candidate_mean_percentile"): _v2_idx("micro_nearest_stronger_distance_s") + 1]).all()
    assert row[_v2_idx("macro_candidate_mean_percentile")] == pytest.approx(.5)
    assert row[_v2_idx("macro_candidate_mean_robust_z")] == pytest.approx(0)
    assert np.isnan(row[_v2_idx("macro_candidate_mean_relative_to_session_median")])
    assert row[_v2_idx("macro_best_mean_margin")] == 0
    assert row[_v2_idx("macro_stronger_count_20m")] == 0
    assert np.isnan(row[_v2_idx("macro_nearest_stronger_distance_s")])
    assert row[_v2_idx("candidate_duration_percentile")] == pytest.approx(.5)
    assert row[_v2_idx("candidate_duration_robust_z")] == pytest.approx(0)
    assert row[_v2_idx("candidate_support_coverage_percentile")] == pytest.approx(.5)
    assert row[_v2_idx("candidate_support_coverage_robust_z")] == pytest.approx(0)
    assert not np.isinf(row).any()


def test_context_v2_ties_are_not_stronger_and_twenty_minute_boundary_is_inclusive():
    tied = _multi("s", 0, 1_000, macro=_source("s", 0, 1_000, (.5,)))
    boundary = _multi("s", 1_201_000, 1_202_000, macro=_source("s", 1_201_000, 1_202_000, (.7,)))
    farther = _multi("s", 1_202_001, 1_203_001, macro=_source("s", 1_202_001, 1_203_001, (.9,)))
    rows = context_v2_features([tied, boundary, farther], {}, {}, {"s": (0, 2_000_000)})
    assert rows[0, _v2_idx("macro_candidate_mean_percentile")] == pytest.approx(1 / 6)
    assert rows[0, _v2_idx("macro_stronger_count_20m")] == 1
    assert rows[0, _v2_idx("macro_nearest_stronger_distance_s")] == pytest.approx(1200)
    assert rows[1, _v2_idx("macro_stronger_count_20m")] == 1
    assert rows[2, _v2_idx("macro_best_mean_margin")] == 0
    assert np.isnan(rows[2, _v2_idx("macro_nearest_stronger_distance_s")])


def test_context_v2_overlap_distance_is_zero_and_keeps_coincident_source_identities():
    macro_low = _source("s", 0, 1_000, (.3,), .4)
    macro_high = _source("s", 0, 1_000, (.8,), .6)
    candidates = [
        _multi("s", 0, 1_000, macro=macro_low),
        _multi("s", 0, 1_000, macro=macro_high),
    ]
    rows = context_v2_features(candidates, {}, {}, {"s": (0, 2_000)})
    assert rows[:, _v2_idx("session_candidate_count")].tolist() == [2.0, 2.0]
    assert rows[0, _v2_idx("macro_nearest_stronger_distance_s")] == 0
    assert rows[0, _v2_idx("macro_stronger_count_20m")] == 1


def test_context_v2_accepts_exact_union_population_without_second_deduplication():
    macro = [_source("s", 0, 1_000, (.3,)), _source("s", 0, 1_000, (.8,))]
    micro = [_source("s", 100, 900, (.7,))]
    population = union_candidates(macro, micro)
    rows = context_v2_features(population, {}, {}, {"s": (0, 2_000)})
    assert len(population) == 2
    assert rows.shape == (2, 25)
    assert rows[:, _v2_idx("session_candidate_count")].tolist() == [2.0, 2.0]


def test_context_v2_keeps_touching_sessions_separate_and_ignores_acquisition_gaps():
    left = _multi("left", 900, 1_000, macro=_source("left", 900, 1_000, (.2,)))
    right = _multi("right", 1_000, 1_100, macro=_source("right", 1_000, 1_100, (.9,)))
    rows = context_v2_features(
        [left, right],
        {"left": [(0, 1, .1), (900, 901, .3)], "right": [(1_000, 1_001, .8), (9_000_000, 9_000_001, .9)]},
        {},
        {"left": (0, 1_000), "right": (1_000, 10_000_000)},
    )
    assert rows[:, _v2_idx("session_candidate_count")].tolist() == [1.0, 1.0]
    assert rows[0, _v2_idx("macro_candidate_mean_relative_to_session_median")] == pytest.approx(0.0)
    assert rows[1, _v2_idx("macro_candidate_mean_relative_to_session_median")] > 0
