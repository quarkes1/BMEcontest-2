import numpy as np
import pandas as pd

from scripts.research.data_sprint import (
    disagreement_counts,
    disagreement_fold_support,
    expected_window_coverage,
    periodicity_metrics,
    prepare_output_dir,
    select_truth_level_fn,
    temporal_metrics,
)


def test_prepare_output_dir_creates_nested_directory(tmp_path):
    target = prepare_output_dir(tmp_path / "subject_relative")
    assert target.is_dir()


def test_select_truth_level_fn_keeps_highest_blend_per_truth():
    rows = pd.DataFrame(
        {
            "truth_key": ["a", "a", "b"],
            "blend_score": [0.2, 0.7, 0.4],
            "candidate_key": ["a0", "a1", "b0"],
        }
    )
    selected = select_truth_level_fn(rows)
    assert selected["candidate_key"].tolist() == ["a1", "b0"]


def test_disagreement_counts_uses_registered_direction_threshold():
    rows = pd.DataFrame(
        {"logistic_score": [0.8, 0.1, 0.5], "lgbm_score": [0.5, 0.4, 0.4]}
    )
    assert disagreement_counts(rows, threshold=0.20) == {
        "lr_minus_lgbm_ge_0_20": 1,
        "lgbm_minus_lr_ge_0_20": 1,
        "neither": 1,
        "population_size": 3,
    }


def test_disagreement_fold_support_requires_repeated_direction():
    by_fold = {
        "reachable_fn": {str(i): {"lr_minus_lgbm_ge_0_20": 2, "population_size": 4} for i in range(5)},
        "final_fp": {str(i): {"lr_minus_lgbm_ge_0_20": 0, "population_size": 4} for i in range(5)},
    }
    result = disagreement_fold_support(by_fold)
    assert result["fn_lr_excess_fold_count"] == 5
    assert result["stable"] is True


def test_expected_window_coverage_is_relative_to_requested_window():
    assert expected_window_coverage(observed_samples=80, sample_rate=10, seconds=10) == 0.8
    assert expected_window_coverage(observed_samples=120, sample_rate=10, seconds=10) == 1.0


def test_temporal_metrics_identifies_bursty_activity():
    sample_rate = 10.0
    n = 200
    acc = np.zeros((3, n), dtype=float)
    gyro = np.zeros((3, n), dtype=float)
    acc[0, 80:100] = np.tile([0.0, 8.0], 10)
    gyro[1, 80:100] = np.tile([0.0, 4.0], 10)
    metrics = temporal_metrics(acc, gyro, sample_rate)
    assert metrics["short_term_energy"] > 0
    assert metrics["peak_count"] > 0
    assert 0 < metrics["active_fraction"] < 1


def test_periodicity_metrics_recovers_one_hz_signal():
    sample_rate = 10.0
    t = np.arange(600) / sample_rate
    signal = np.sin(2 * np.pi * t)
    metrics = periodicity_metrics(signal, sample_rate)
    assert 0.9 <= metrics["dominant_frequency_hz"] <= 1.1
    assert metrics["autocorrelation_peak_strength"] > 0.8
    assert 0.8 <= metrics["autocorrelation_first_peak_s"] <= 1.2
