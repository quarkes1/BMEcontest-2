"""Coverage measures interval unions, not the distance between first and last row."""
import pytest


def test_coverage_does_not_bridge_an_acquisition_gap():
    from scripts.research.official_recovery.coverage_audit import covered_ms
    assert covered_ms([(0, 20), (80, 100)], (0, 100)) == 40
    assert covered_ms([(0, 60), (40, 100)], (0, 100)) == 100
    assert covered_ms([(0, 100)], (30, 50)) == 20
    assert covered_ms([], (0, 100)) == 0


def test_consecutive_files_can_cover_a_meal_without_one_containing_it():
    from scripts.research.official_recovery.coverage_audit import interval_diagnostics
    result = interval_diagnostics([(0, 40), (40, 100)], (10, 90))
    assert result == {'covered_ms': 80, 'coverage_fraction': 1.0,
                      'overlapping_intervals': 2, 'single_interval_contains': False}


def test_union_merges_touching_and_duplicate_intervals_but_not_gaps():
    from scripts.research.official_recovery.coverage_audit import merge_intervals
    assert merge_intervals([(20, 30), (0, 10), (0, 10), (10, 20), (40, 50)]) == [(0, 30), (40, 50)]
    with pytest.raises(ValueError, match='positive'):
        merge_intervals([(2, 2)])


def test_active_day_count_does_not_depend_on_the_host_timezone():
    from scripts.research.official_recovery.coverage_audit import active_days
    # Local midnight at UTC16h; one interval crosses it, with no time in the third day.
    assert active_days([(15*3600000, 17*3600000)], utc_offset_hours=8) == 2
    assert active_days([], utc_offset_hours=8) == 0
    assert active_days([(0, 86400000)], utc_offset_hours=0) == 1


def test_nonoverlap_and_partial_intervals_do_not_claim_complete_coverage():
    from scripts.research.official_recovery.coverage_audit import interval_diagnostics
    result = interval_diagnostics([(0, 10), (30, 60)], (20, 100))
    assert result['covered_ms'] == 30
    assert result['coverage_fraction'] == .375
    assert result['overlapping_intervals'] == 1


def test_invalid_target_is_rejected_even_with_no_recordings():
    from scripts.research.official_recovery.coverage_audit import covered_ms
    with pytest.raises(ValueError, match='positive'):
        covered_ms([], (4, 4))
