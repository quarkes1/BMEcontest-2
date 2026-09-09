"""Integration contracts for subject-disjoint macro/micro event training."""

from dataclasses import replace

import numpy as np
import pytest

from tests.pipeline.test_runner import synthetic_runner_dataset
from src.pipeline import runner
from src.pipeline.event_stack import DensityConfig, EventRef
from src.pipeline.runner import RunConfig, WindowBatch, run_outer_fold


def multiscale_dataset():
    base = synthetic_runner_dataset(subjects=8, windows_per_subject=40)

    def macro(batch):
        return replace(batch, features=np.pad(batch.features, ((0, 0), (0, 60))))

    def micro(batch, truths):
        features, labels, windows = [], [], []
        for sid in sorted({window.sid for window in batch.windows}):
            sid_windows = [window for window in batch.windows if window.sid == sid]
            first = min(window.start_ms for window in sid_windows)
            last = max(window.end_ms for window in sid_windows)
            for start in range(first, last - 15_000 + 1, 7_500):
                end = start + 15_000
                overlap = max(
                    (min(end, truth.end_ms) - max(start, truth.start_ms)
                     for truth in truths if truth.sid == sid), default=0,
                )
                label = int(overlap >= 7_500)
                features.append([float(label)] * 47)
                labels.append(label)
                windows.append(EventRef(sid, start, end))
        return WindowBatch(np.asarray(features, np.float32), np.asarray(labels, np.int8), tuple(windows))

    return replace(
        base,
        window_train=macro(base.window_train),
        candidate_train=macro(base.candidate_train),
        validation=macro(base.validation),
        micro_window_train=micro(base.window_train, base.train_truths),
        micro_candidate_train=micro(base.candidate_train, base.train_truths),
        micro_validation=micro(base.validation, base.validation_truths),
        validation_truth_slices={"duration_lt10": base.validation_truths},
        micro_cache_extraction_seconds=1.25,
    )


def multiscale_config(**kwargs):
    return RunConfig(
        outer_fold=0, inner_splits=3, micro_enabled=True,
        density=DensityConfig(density_ms=60_000, min_positive=2,
                              coverage_min=0.0, window_threshold=0.05),
        **kwargs,
    )


def test_multiscale_outer_subjects_never_enter_any_fit_set():
    result = run_outer_fold(multiscale_config(), multiscale_dataset())
    assert result.outer_subjects.isdisjoint(result.window_fit_subjects)
    assert result.outer_subjects.isdisjoint(result.micro_window_fit_subjects)
    assert result.outer_subjects.isdisjoint(result.verifier_fit_subjects)
    assert result.micro_window_fit_subjects == result.window_fit_subjects
    assert result.micro_threshold in (0.10, 0.20, 0.30, 0.40, 0.50)
    assert result.macro_window_feature_count == 63
    assert result.micro_window_feature_count == 47
    assert result.verifier_feature_count == 56
    assert result.micro_candidate_count > 0
    assert result.candidate_count >= result.micro_candidate_count
    assert result.micro_candidate_match_recall == 1.0
    assert result.short_meal_candidate_recall == 1.0
    assert set(result.timings_seconds) == {
        "feature_extraction", "macro_window_oof", "micro_window_oof",
        "verifier_oof", "final_fit", "outer_inference", "total",
    }
    assert result.timings_seconds["feature_extraction"] == 1.25
    assert all(value >= 0 for value in result.timings_seconds.values())


def test_outer_truth_changes_do_not_change_any_selected_setting():
    dataset = multiscale_dataset()
    config = multiscale_config(verifier_c_grid=(0.01, 0.1), subject_cap_grid=(1, 2))
    original = run_outer_fold(config, dataset)
    assert original.micro_threshold is not None
    changed = run_outer_fold(config, replace(
        dataset, validation_truths=(), validation_truth_slices={},
        micro_validation=replace(dataset.micro_validation, labels=1 - dataset.micro_validation.labels),
    ))
    assert changed.micro_threshold == original.micro_threshold
    assert changed.verifier_c == original.verifier_c
    assert changed.threshold == original.threshold
    assert changed.max_events_per_subject == original.max_events_per_subject
    assert changed.outer_metrics.n_true == 0
    assert changed.outer_metrics.n_pred == original.outer_metrics.n_pred


def test_real_crossfit_audits_each_scale_and_outer_windows_are_scored_once(monkeypatch):
    dataset = multiscale_dataset()
    real_crossfit = runner.crossfit_predict_proba
    real_probability = runner._positive_probability
    audits = []
    inference_widths = []

    def audited_crossfit(train_x, labels, groups, score_x, score_groups, splits, estimator_factory):
        assert dataset.outer_subjects.isdisjoint(groups)
        assert dataset.outer_subjects.isdisjoint(score_groups)
        result = real_crossfit(train_x, labels, groups, score_x, score_groups, splits, estimator_factory)
        for fold in result.folds:
            assert set(fold.train_groups).isdisjoint(fold.validation_groups)
            assert dataset.outer_subjects.isdisjoint(fold.train_groups)
        assert np.all(result.score_counts == 1)
        audits.append((train_x.shape[1], len(train_x)))
        return result

    def audited_probability(estimator, features):
        inference_widths.append(features.shape[1])
        return real_probability(estimator, features)

    monkeypatch.setattr(runner, "crossfit_predict_proba", audited_crossfit)
    monkeypatch.setattr(runner, "_positive_probability", audited_probability)
    run_outer_fold(multiscale_config(micro_positive_middle_fraction=0.5), dataset)
    assert [width for width, _ in audits] == [63, 47, 56]
    # Six train sessions have 109 micro rows each. Three meals retain only the
    # 23 positive window centers in [266250, 438750], versus 47 before purity.
    assert audits[1][1] == 582
    assert inference_widths == [63, 47, 56]


@pytest.mark.parametrize("name", ["micro_window_train", "micro_candidate_train", "micro_validation"])
def test_micro_batches_are_required_and_must_have_47_columns(name):
    dataset = multiscale_dataset()
    with pytest.raises(ValueError, match=name):
        run_outer_fold(multiscale_config(), replace(dataset, **{name: None}))
    batch = getattr(dataset, name)
    with pytest.raises(ValueError, match="47"):
        run_outer_fold(multiscale_config(), replace(dataset, **{
            name: replace(batch, features=batch.features[:, :46]),
        }))


@pytest.mark.parametrize("name", ["micro_window_train", "micro_candidate_train"])
def test_outer_subject_in_either_micro_training_batch_is_rejected(name):
    dataset = multiscale_dataset()
    with pytest.raises(ValueError, match="outer subject leakage"):
        run_outer_fold(multiscale_config(), replace(dataset, **{name: dataset.micro_validation}))


def test_micro_validation_must_match_outer_subject_manifest():
    dataset = multiscale_dataset()
    with pytest.raises(ValueError, match="micro.*outer subject manifest"):
        run_outer_fold(multiscale_config(), replace(dataset, micro_validation=dataset.micro_candidate_train))


@pytest.mark.parametrize("weights", [(0.0, 0.5), (0.5,), ()])
def test_external_weights_are_not_silently_ignored(weights):
    with pytest.raises(ValueError, match="external_fd_weight_grid"):
        run_outer_fold(multiscale_config(external_fd_weight_grid=weights), multiscale_dataset())


def test_micro_positive_purity_keeps_negatives_and_only_middle_positive_centers():
    batch = WindowBatch(
        np.zeros((6, 47)), np.array([-1, 0, 1, 1, 1, 1]),
        tuple(EventRef("s", start, start + 10) for start in (0, 0, 5, 25, 65, 85)),
    )
    truths = (EventRef("s", 0, 100),)
    assert runner._micro_training_keep(batch, truths, None).tolist() == [False, True, True, True, True, True]
    assert runner._micro_training_keep(batch, truths, 0.5).tolist() == [False, True, False, True, True, False]
    for invalid in (0, -0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="micro_positive_middle_fraction"):
            runner._micro_training_keep(batch, truths, invalid)


def test_macro_only_ignores_optional_micro_batches_and_preserves_predictions():
    dataset = multiscale_dataset()
    config = replace(multiscale_config(), micro_enabled=False)
    original = run_outer_fold(config, dataset)
    changed = run_outer_fold(config, replace(
        dataset, micro_window_train=None, micro_candidate_train=None, micro_validation=None,
    ))
    assert original.outer_metrics == changed.outer_metrics
    assert original.threshold == changed.threshold
    assert original.verifier_feature_count == 37
    assert original.micro_threshold is None
    assert original.micro_window_fit_subjects == frozenset()
