import numpy as np
import pytest

from src.pipeline.event_stack import DensityConfig, EventRef
from src.pipeline.runner import (
    FoldDataset,
    RunConfig,
    WindowBatch,
    cache_key,
    run_outer_fold,
    validate_outer_isolation,
)


def test_cache_key_changes_with_candidate_semantics():
    base = RunConfig(
        outer_fold=0,
        inner_splits=3,
        density=DensityConfig(coverage_fix=False),
    )
    fixed = RunConfig(
        outer_fold=0,
        inner_splits=3,
        density=DensityConfig(coverage_fix=True),
    )

    assert cache_key(base) != cache_key(fixed)


def test_validate_outer_isolation_rejects_overlap():
    with pytest.raises(ValueError, match="outer subject leakage"):
        validate_outer_isolation({"a", "b"}, {"b", "c"})


def synthetic_runner_dataset(
    subjects: int = 8, windows_per_subject: int = 40
) -> FoldDataset:
    subject_names = [f"subject-{index}" for index in range(subjects)]
    train_subjects = subject_names[:-2]
    outer_subjects = subject_names[-2:]
    subject_by_session = {
        f"session-{index}": subject for index, subject in enumerate(subject_names)
    }

    def batch(selected_subjects: list[str]) -> WindowBatch:
        features = []
        labels = []
        windows = []
        for subject in selected_subjects:
            subject_index = subject_names.index(subject)
            sid = f"session-{subject_index}"
            has_meal = subject_index % 2 == 0
            for window_index in range(windows_per_subject):
                in_meal = has_meal and 10 <= window_index <= 19
                features.append([float(in_meal), float(subject_index % 3)])
                labels.append(int(in_meal))
                start = window_index * 15_000
                windows.append(EventRef(sid, start, start + 240_000))
        return WindowBatch(
            np.asarray(features, dtype=np.float32),
            np.asarray(labels, dtype=np.int8),
            tuple(windows),
        )

    truths = {
        subject: EventRef(
            f"session-{subject_names.index(subject)}", 180_000, 525_000
        )
        for subject in subject_names
        if subject_names.index(subject) % 2 == 0
    }
    train_truths = tuple(truths[s] for s in train_subjects if s in truths)
    validation_truths = tuple(truths[s] for s in outer_subjects if s in truths)
    return FoldDataset(
        window_train=batch(train_subjects),
        candidate_train=batch(train_subjects),
        validation=batch(outer_subjects),
        train_truths=train_truths,
        validation_truths=validation_truths,
        subject_by_session=subject_by_session,
        outer_subjects=frozenset(outer_subjects),
        validation_truth_slices={"synthetic_meals": validation_truths},
    )


def test_outer_subjects_never_enter_fit_sets():
    dataset = synthetic_runner_dataset(subjects=8, windows_per_subject=40)
    result = run_outer_fold(
        RunConfig(
            outer_fold=0,
            inner_splits=3,
            density=DensityConfig(
                density_ms=60_000,
                min_positive=2,
                coverage_min=0.0,
                window_threshold=0.05,
            ),
        ),
        data_source=dataset,
    )

    assert result.outer_subjects.isdisjoint(result.window_fit_subjects)
    assert result.outer_subjects.isdisjoint(result.verifier_fit_subjects)


def test_subject_cap_is_learned_inside_and_applied_outside():
    dataset = synthetic_runner_dataset(subjects=8, windows_per_subject=40)
    result = run_outer_fold(
        RunConfig(
            outer_fold=0,
            inner_splits=3,
            subject_cap_grid=(1, 2),
            density=DensityConfig(
                density_ms=60_000,
                min_positive=2,
                coverage_min=0.0,
                window_threshold=0.05,
            ),
        ),
        data_source=dataset,
    )

    assert result.max_events_per_subject in (1, 2)
    assert result.outer_metrics.n_pred <= (
        result.max_events_per_subject * len(result.outer_subjects)
    )
