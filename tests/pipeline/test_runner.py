from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.crossfit_event_stack import parse_subject_cap_grid, parse_verifier_c_grid
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
    assert result.verifier_feature_count == 37
    assert result.verifier_c == 0.1
    assert result.macro_window_feature_count == 3
    assert result.timings_seconds["feature_extraction"] == 0.0


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


def test_subject_cap_grid_parser_is_strict_and_deterministic():
    assert parse_subject_cap_grid("2,3,4,5,6") == (2, 3, 4, 5, 6)
    assert parse_subject_cap_grid("") == ()
    with pytest.raises(ValueError, match="positive unique integers"):
        parse_subject_cap_grid("2,2,0")


def test_verifier_c_grid_parser_rejects_nonfinite_or_duplicate_values():
    assert parse_verifier_c_grid("0.001,0.01,0.1") == (0.001, 0.01, 0.1)
    with pytest.raises(ValueError, match="positive unique finite"):
        parse_verifier_c_grid("0.01,nan,0.01")


def test_raw_summary_verifier_selects_registered_c_and_reports_width():
    dataset = synthetic_runner_dataset(subjects=8, windows_per_subject=40)
    result = run_outer_fold(
        RunConfig(
            outer_fold=0,
            inner_splits=3,
            verifier_feature_mode="raw_summary",
            verifier_c_grid=(0.001, 0.01),
            density=DensityConfig(
                density_ms=60_000,
                min_positive=2,
                coverage_min=0.0,
                window_threshold=0.05,
            ),
        ),
        data_source=dataset,
    )

    assert result.verifier_c in (0.001, 0.01)
    assert result.verifier_feature_count == 49
    assert result.outer_subjects.isdisjoint(result.verifier_fit_subjects)


@pytest.mark.parametrize("changes", [
    {"micro_enabled": True},
    {"micro_gravity_align": False},
    {"micro_threshold_grid": (0.2, 0.4)},
    {"micro_positive_middle_fraction": 0.6},
    {"external_fd_weight_grid": (0.0, 0.5)},
])
def test_cache_key_changes_for_every_micro_behavior_setting(changes):
    base = RunConfig(outer_fold=0)
    assert cache_key(base) != cache_key(replace(base, **changes))


def test_cache_key_changes_for_micro_candidates_and_model_parameters(monkeypatch):
    from src.pipeline import runner
    from src.pipeline.event_stack import MicroCandidateConfig

    base = RunConfig(outer_fold=0, micro_enabled=True)
    original = cache_key(base)
    assert original != cache_key(replace(base, micro_candidate=MicroCandidateConfig(merge_ms=120_000)))
    monkeypatch.setitem(runner.MICRO_WINDOW_MODEL_PARAMETERS, "num_leaves", 15)
    assert original != cache_key(base)


def test_schema_three_prediction_cache_identity_is_not_reused(monkeypatch):
    from src.pipeline import runner

    current = cache_key(RunConfig(outer_fold=0))
    monkeypatch.setattr(runner, "RUNNER_SCHEMA_VERSION", 3)
    assert cache_key(RunConfig(outer_fold=0)) != current


def test_micro_lightgbm_is_single_threaded_and_seeded():
    from src.pipeline import runner

    estimator = runner._micro_window_estimator(20260909)
    parameters = estimator.get_params()
    assert parameters["model__n_jobs"] == 1
    assert parameters["model__random_state"] == 20260909
    assert parameters["model__class_weight"] == "balanced"
    assert parameters["imputer__strategy"] == "median"


def test_macro_fold_dataset_remains_constructible_without_micro_batches():
    dataset = synthetic_runner_dataset()
    assert dataset.micro_window_train is None
    assert dataset.micro_candidate_train is None
    assert dataset.micro_validation is None
    assert dataset.micro_cache_extraction_seconds == 0.0


def test_filesystem_source_rejects_micro_metadata_mismatch(tmp_path):
    from src.pipeline.micro_cache import MicroCacheArrays, write_micro_cache_atomic
    from src.pipeline.runner import FilesystemDataSource

    path = tmp_path / "cache" / "micro15" / "fold0_train.npz"
    arrays = MicroCacheArrays(np.ones((1, 47), np.float32), np.array([0], np.int8), np.array(['["s1", 0, 15000]']))
    write_micro_cache_atomic(path, arrays, {"wrong": True})
    with pytest.raises(ValueError, match="metadata mismatch"):
        FilesystemDataSource(tmp_path)._load_micro_batch(path, {"expected": True})


@pytest.fixture
def filesystem_runner_source(tmp_path, monkeypatch):
    from src.data import manifests
    from src.pipeline.imu_features import MicroFeatureConfig
    from src.pipeline.micro_cache import MicroCacheArrays, cache_metadata, write_micro_cache_atomic
    from src.pipeline.runner import FilesystemDataSource

    index = tmp_path / "index.csv"
    index.write_text(
        "externalid,sensorData,timeStamp.startTime,timeStamp.endTime\n"
        "p1,s1.zip,0,600000\np2,s2.zip,0,600000\np3,s3.zip,0,600000\n",
        encoding="utf-8",
    )
    meals = tmp_path / "meals.csv"
    meals.write_text(
        "externalid,beforeTime,afterTime,wearHand,dietaryHand,dietaryType,tablewareType\n"
        "p1,0,240000,right,right,meal,spoon\np3,0,240000,right,right,meal,spoon\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manifests, "INDEX_CSV", index)
    monkeypatch.setattr(manifests, "MEALS_CSV", meals)
    monkeypatch.setattr(manifests, "BLACKLIST_FILE", tmp_path / "absent-blacklist")
    source = FilesystemDataSource(tmp_path)
    source.slide_dir.mkdir(parents=True)
    source.session_dir.mkdir(parents=True)
    manifest = tmp_path / "cache" / "splits" / "fold0.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"train_sessions": ["s1", "s2"], "val_sessions": ["s3"]}', encoding="utf-8")
    for sid in ("s1", "s2", "s3"):
        np.savez(source.session_dir / f"{sid}.npz", t_acc=np.arange(0, 600001, 1000), imu_valid=np.ones(601, bool))
    split_sessions = {"train": ("s1", "s2"), "meal_train": ("s1",), "no_meal_train": ("s2",), "val": ("s3",)}
    for split, sessions in split_sessions.items():
        labels = np.array([int(sid != "s2") for sid in sessions], np.int8)
        np.savez(source.slide_dir / f"fold0_{split}.npz", feat=np.zeros((len(sessions), 62), np.float32), label=labels,
                 wid=np.array([json.dumps((sid, 0, 240000)) for sid in sessions]))
        metadata = cache_metadata(MicroFeatureConfig(), (*[source.session_dir / f"{sid}.npz" for sid in sessions], manifest))
        metadata["extraction_seconds"] = {"train": 1.0, "meal_train": 2.0, "no_meal_train": 3.0, "val": 4.0}[split]
        arrays = MicroCacheArrays(np.ones((len(sessions), 47), np.float32), labels,
                                 np.array([json.dumps((sid, 0, 15000)) for sid in sessions]))
        write_micro_cache_atomic(tmp_path / "cache" / "micro15" / f"fold0_{split}.npz", arrays, metadata)
    return source


def test_filesystem_source_loads_optional_micro_splits_and_extraction_time(filesystem_runner_source):
    source = filesystem_runner_source
    config = RunConfig(outer_fold=0, micro_enabled=True)
    dataset = source.load_outer_fold(config)
    assert dataset.micro_window_train.features.shape == (2, 47)
    assert dataset.micro_candidate_train.windows == (EventRef("s1", 0, 15000), EventRef("s2", 0, 15000))
    assert dataset.micro_candidate_train.labels.tolist() == [1, 0]
    assert dataset.micro_validation.windows == (EventRef("s3", 0, 15000),)
    assert dataset.micro_cache_extraction_seconds == 10.0
    assert set(source.input_files(config)) - set(source.input_files(replace(config, micro_enabled=False))) == {
        source.root / "cache" / "micro15" / f"fold0_{split}.npz"
        for split in ("train", "meal_train", "no_meal_train", "val")
    }


@pytest.mark.parametrize("change", ["gravity", "session", "manifest"])
def test_filesystem_source_rejects_stale_micro_semantics(filesystem_runner_source, change):
    source = filesystem_runner_source
    config = RunConfig(outer_fold=0, micro_enabled=True)
    if change == "gravity":
        config = replace(config, micro_gravity_align=False)
    else:
        path = source.session_dir / "s1.npz" if change == "session" else source.root / "cache" / "splits" / "fold0.json"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="metadata mismatch"):
        source.load_outer_fold(config)


def test_macro_filesystem_source_does_not_require_micro_files(filesystem_runner_source):
    source = filesystem_runner_source
    for path in (source.root / "cache" / "micro15").glob("*.npz"):
        path.unlink()
    config = RunConfig(outer_fold=0)
    assert source.load_outer_fold(config).micro_validation is None
    assert all(path.parent.name != "micro15" for path in source.input_files(config))
    with pytest.raises(FileNotFoundError, match="missing runner inputs"):
        source.input_files(replace(config, micro_enabled=True))


def historical_result_payload():
    from src.pipeline.event_stack import compute_event_metrics

    metrics = asdict(compute_event_metrics([], []))
    return {"config_hash": "schema3", "threshold": 0.5, "inner_metrics": metrics,
            "outer_metrics": metrics, "candidate_count": 0, "candidate_match_recall": 0.0}


def test_historical_results_receive_micro_defaults():
    from src.pipeline.runner import _fold_result_from_dict

    result = _fold_result_from_dict(historical_result_payload())
    assert result.micro_threshold is None
    assert result.micro_candidate_count == 0
    assert result.micro_candidate_match_recall == 0.0
    assert result.short_meal_candidate_recall == 0.0
    assert result.external_weight == 0.0
    assert result.macro_window_feature_count == 63
    assert result.micro_window_feature_count == 0
    assert result.micro_window_fit_subjects == frozenset()


def test_multiscale_result_json_round_trip_preserves_every_field():
    from src.pipeline.runner import _fold_result_from_dict, fold_result_to_dict

    result = replace(_fold_result_from_dict(historical_result_payload()),
                     micro_threshold=0.3, micro_candidate_count=7, micro_candidate_match_recall=0.8,
                     short_meal_candidate_recall=0.75, external_weight=0.5,
                     macro_window_feature_count=63, micro_window_feature_count=47,
                     micro_window_fit_subjects=frozenset({"p2", "p1"}),
                     timings_seconds={"feature_extraction": 1.0, "macro_window_oof": 2.0, "micro_window_oof": 3.0,
                                      "verifier_oof": 4.0, "final_fit": 5.0, "outer_inference": 6.0, "total": 21.0})
    payload = json.loads(json.dumps(fold_result_to_dict(result)))
    assert set(payload) == set(asdict(result))
    assert payload["micro_window_fit_subjects"] == ["p1", "p2"]
    assert _fold_result_from_dict(payload) == result


@pytest.mark.parametrize("physical_count, logical_count, user_limit, expected", [
    (4, 8, None, 4), (None, 8, None, 4), (4, 8, "2", 2),
])
def test_runner_import_and_use_avoid_joblib_physical_core_warning(physical_count, logical_count, user_limit, expected):
    script = f'''
import os
import sys
from types import SimpleNamespace
os.cpu_count = lambda: {logical_count!r}
physical_count = {physical_count!r}
sys.modules["psutil"] = SimpleNamespace(
    cpu_count=lambda logical=True: physical_count,
    Process=lambda: SimpleNamespace(cpu_affinity=lambda: list(range({logical_count!r}))),
) if physical_count is not None else None
from src.pipeline.runner import _window_estimator
from joblib.externals.loky.backend import context
context._count_physical_cores = lambda: ("not found", FileNotFoundError("WMIC unavailable"))
import numpy as np
_window_estimator(7).fit(np.arange(40).reshape(20, 2), np.array([0, 1] * 10))
print(os.environ["LOKY_MAX_CPU_COUNT"])
'''
    environment = dict(os.environ)
    environment.pop("LOKY_MAX_CPU_COUNT", None)
    if user_limit is not None:
        environment["LOKY_MAX_CPU_COUNT"] = user_limit
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                               env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "Could not find the number of physical cores" not in completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.strip() == str(expected)
