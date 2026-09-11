"""Leakage and alignment contracts for nested candidate-control stacking."""

from dataclasses import replace

import numpy as np

from src.pipeline import runner
from src.pipeline.runner import RunConfig, run_outer_fold
from tests.pipeline.test_multiscale_runner import multiscale_config, multiscale_dataset


def registered_stacking_config(**kwargs) -> RunConfig:
    return multiscale_config(candidate_control_enabled=True, **kwargs)


def _selected_tuple(result):
    return (
        result.selected_blend_weight,
        result.selected_admission_nms_iou,
        result.selected_admission_threshold,
        result.selected_admission_subject_cap,
        result.verifier_c,
        result.threshold,
        result.max_events_per_subject,
    )


def test_both_verifiers_crossfit_each_candidate_once_without_subject_overlap(monkeypatch):
    real = runner.crossfit_predict_proba
    calls = []

    def recording_crossfit(*args, **kwargs):
        output = real(*args, **kwargs)
        calls.append(output)
        return output

    monkeypatch.setattr(runner, "crossfit_predict_proba", recording_crossfit)
    run_outer_fold(registered_stacking_config(), multiscale_dataset())

    assert len(calls) == 4
    for call in calls[-2:]:
        assert np.all(call.score_counts == 1)
        assert all(
            set(fold.train_groups).isdisjoint(fold.validation_groups)
            for fold in call.folds
        )
    assert calls[-2].probabilities.shape == calls[-1].probabilities.shape


def test_outer_labels_cannot_change_stacking_or_admission_settings():
    dataset = multiscale_dataset()
    config = registered_stacking_config(subject_cap_grid=(1, 2))
    original = run_outer_fold(config, dataset)
    changed = run_outer_fold(
        config,
        replace(
            dataset,
            validation_truths=(),
            validation_truth_slices={},
            validation=replace(
                dataset.validation, labels=1 - dataset.validation.labels
            ),
            micro_validation=replace(
                dataset.micro_validation,
                labels=1 - dataset.micro_validation.labels,
            ),
        ),
    )

    assert _selected_tuple(changed) == _selected_tuple(original)
    assert changed.outer_metrics.n_true == 0


def test_enabled_result_reports_raw_and_admitted_counts_and_stage_timings():
    result = run_outer_fold(registered_stacking_config(), multiscale_dataset())

    assert result.raw_union_candidate_count >= result.admitted_candidate_count
    assert result.candidate_count == result.admitted_candidate_count
    assert result.selected_blend_weight in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert result.selected_admission_nms_iou in (0.3, 0.5, 0.7)
    assert result.selected_admission_threshold in (0.2, 0.35, 0.5, 0.65)
    assert result.selected_admission_subject_cap in (3, 4, 5, 6, 8)
    for name in (
        "verifier_logistic_oof",
        "verifier_lgbm_oof",
        "admission_selection",
        "final_fit",
        "outer_inference",
    ):
        assert np.isfinite(result.timings_seconds[name])
        assert result.timings_seconds[name] >= 0.0


def test_final_verifiers_each_score_outer_candidates_once(monkeypatch):
    real = runner._positive_probability
    scored_widths = []

    def recording_probability(estimator, features):
        scored_widths.append(features.shape[1])
        return real(estimator, features)

    monkeypatch.setattr(runner, "_positive_probability", recording_probability)
    run_outer_fold(registered_stacking_config(), multiscale_dataset())

    assert scored_widths == [63, 47, 56, 56]


def test_registered_c_grid_crossfits_each_logistic_option_before_joint_selection(monkeypatch):
    real = runner.crossfit_predict_proba
    calls = []

    def recording_crossfit(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(runner, "crossfit_predict_proba", recording_crossfit)
    result = run_outer_fold(
        registered_stacking_config(verifier_c_grid=(0.01, 0.1)),
        multiscale_dataset(),
    )

    assert len(calls) == 5
    assert result.verifier_c in (0.01, 0.1)
    assert all(np.all(call.score_counts == 1) for call in calls[-3:])


def test_candidate_control_disabled_keeps_neutral_stacking_fields():
    result = run_outer_fold(multiscale_config(), multiscale_dataset())

    assert result.selected_blend_weight is None
    assert result.selected_admission_nms_iou is None
    assert result.selected_admission_threshold is None
    assert result.selected_admission_subject_cap is None
    assert result.raw_union_candidate_count == 0
    assert result.admitted_candidate_count == 0
    assert result.verifier_logistic_oof_seconds == 0.0
    assert result.verifier_lgbm_oof_seconds == 0.0
    assert result.verifier_logistic_outer_seconds == 0.0
    assert result.verifier_lgbm_outer_seconds == 0.0


def test_admission_falls_back_to_highest_recall_when_floor_is_unreachable():
    config = registered_stacking_config(
        admission_nms_iou_grid=(0.3,),
        admission_threshold_grid=(1.0,),
        admission_subject_cap_grid=(1,),
    )

    result = run_outer_fold(config, multiscale_dataset())

    assert result.selected_admission_nms_iou == 0.3
    assert result.selected_admission_threshold == 1.0
    assert result.selected_admission_subject_cap == 1
    assert result.selected_blend_weight in config.verifier_blend_weight_grid
    assert np.isfinite(result.inner_metrics.f1)
    assert np.isfinite(result.outer_metrics.f1)
