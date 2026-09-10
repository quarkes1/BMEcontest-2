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


@pytest.fixture
def filesystem_source_with_orphan_micro_sessions(filesystem_runner_source):
    from src.data import manifests
    from src.pipeline.imu_features import MicroFeatureConfig
    from src.pipeline.micro_cache import MicroCacheArrays, cache_metadata, write_micro_cache_atomic

    source = filesystem_runner_source
    # Extra sessions share existing subjects; s4 also adds an outer-only subject.
    manifests.INDEX_CSV.write_text(
        manifests.INDEX_CSV.read_text(encoding="utf-8")
        + "p1,s1-extra.zip,0,600000\np2,s2-extra.zip,0,600000\n"
        + "p3,s3-extra.zip,0,600000\np4,s4.zip,0,600000\n",
        encoding="utf-8",
    )
    manifest = source.root / "cache" / "splits" / "fold0.json"
    manifest.write_text(json.dumps({
        "train_sessions": ["s1", "s2", "s1-extra", "s2-extra"],
        "val_sessions": ["s3", "s3-extra", "s4"],
    }), encoding="utf-8")
    for sid in ("s1-extra", "s2-extra", "s3-extra", "s4"):
        np.savez(source.session_dir / f"{sid}.npz", t_acc=np.arange(0, 600001, 1000),
                 imu_valid=np.ones(601, bool))

    # Distinguish the macro training SID universe from candidate training.
    np.savez(source.slide_dir / "fold0_train.npz", feat=np.zeros((1, 62), np.float32),
             label=np.array([1], np.int8), wid=np.array(['["s1", 0, 240000]']))
    split_rows = {
        "train": [("s1-extra", 90, 1), ("s2", 20, 0), ("s2-extra", 91, 0), ("s1", 10, 1)],
        "meal_train": [("s1-extra", 90, 1), ("s1", 10, 1)],
        "no_meal_train": [("s2", 20, 0), ("s2-extra", 91, 0)],
        "val": [("s3-extra", 90, 1), ("s3", 30, 1), ("s4", 91, 0), ("s3", 31, 0)],
    }
    for split, rows in split_rows.items():
        metadata = cache_metadata(MicroFeatureConfig(), (
            *[source.session_dir / f"{sid}.npz" for sid in dict.fromkeys(row[0] for row in rows)],
            manifest,
        ))
        metadata["extraction_seconds"] = {"train": 1.0, "meal_train": 2.0, "no_meal_train": 3.0, "val": 4.0}[split]
        arrays = MicroCacheArrays(
            np.array([[marker] * 47 for _, marker, _ in rows], np.float32),
            np.array([label for _, _, label in rows], np.int8),
            np.array([json.dumps((sid, marker * 1000, marker * 1000 + 15000)) for sid, marker, _ in rows]),
        )
        write_micro_cache_atomic(source.micro_dir / f"fold0_{split}.npz", arrays, metadata)
    return source


@pytest.mark.parametrize("micro_name, macro_name, expected_windows, markers, labels", [
    ("micro_window_train", "window_train", (EventRef("s1", 10000, 25000),), [10], [1]),
    ("micro_candidate_train", "candidate_train",
     (EventRef("s1", 10000, 25000), EventRef("s2", 20000, 35000)), [10, 20], [1, 0]),
    ("micro_validation", "validation",
     (EventRef("s3", 30000, 45000), EventRef("s3", 31000, 46000)), [30, 31], [1, 0]),
])
def test_filesystem_source_filters_micro_orphan_sids_at_each_macro_boundary(
    filesystem_source_with_orphan_micro_sessions, micro_name, macro_name, expected_windows, markers, labels,
):
    from src.pipeline.micro_cache import read_micro_metadata

    source = filesystem_source_with_orphan_micro_sessions
    config = RunConfig(outer_fold=0, micro_enabled=True)
    macro_dataset = source.load_outer_fold(replace(config, micro_enabled=False))
    inputs_before = source.input_files(config)
    fingerprint_before = cache_key(config, input_files=inputs_before)
    metadata_before = {path: read_micro_metadata(path) for path in inputs_before if path.parent == source.micro_dir}
    dataset = source.load_outer_fold(config)
    micro = getattr(dataset, micro_name)
    macro = getattr(dataset, macro_name)

    # SID identity, not subject identity: these orphans share macro subjects.
    assert dataset.subject_by_session["s1-extra"] == dataset.subject_by_session["s1"]
    assert dataset.subject_by_session["s3-extra"] == dataset.subject_by_session["s3"]
    assert {window.sid for window in micro.windows} <= {window.sid for window in macro.windows}
    assert micro.windows == expected_windows
    np.testing.assert_array_equal(micro.features, np.array([[marker] * 47 for marker in markers], np.float32))
    assert micro.labels.tolist() == labels
    assert {window.sid for window in dataset.micro_validation.windows} == {
        window.sid for window in dataset.validation.windows
    }
    assert dataset.train_truths == macro_dataset.train_truths
    assert dataset.validation_truths == macro_dataset.validation_truths
    assert dataset.validation_truth_slices == macro_dataset.validation_truth_slices
    assert dataset.outer_subjects == macro_dataset.outer_subjects == frozenset({"p3"})
    assert dataset.micro_cache_extraction_seconds == 10.0
    assert source.input_files(config) == inputs_before
    assert cache_key(config, input_files=source.input_files(config)) == fingerprint_before
    assert {path: read_micro_metadata(path) for path in metadata_before} == metadata_before


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
    (4, 8, None, 4), (None, 8, None, 4), (4, 8, "2", 2), (8, 8, None, 7),
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


def test_probability_grid_parser_is_strict_and_deterministic():
    from scripts.crossfit_event_stack import parse_probability_grid

    assert parse_probability_grid(" 0.5,0, 1,0.1 ") == (0.0, 0.1, 0.5, 1.0)
    for value in ("", "0.1,nan,0.1", "inf", "-0.1,0.5", "1.1", "0.1,0.10", "bad", "0.1,"):
        with pytest.raises(ValueError, match="unique finite probabilities"):
            parse_probability_grid(value)


def test_middle_fraction_parser_accepts_none_or_unit_interval():
    from scripts.crossfit_event_stack import parse_middle_fraction

    assert parse_middle_fraction(" None ") is None
    assert parse_middle_fraction("0.6") == 0.6
    assert parse_middle_fraction("1") == 1.0
    for value in ("0", "-0.2", "1.1", "nan", "inf", "", "bad"):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            parse_middle_fraction(value)


def aggregate_test_results():
    from src.pipeline.event_stack import EventMetrics
    from src.pipeline.runner import _fold_result_from_dict

    base = _fold_result_from_dict(historical_result_payload())
    return (
        replace(base, config_hash="fold-a", micro_threshold=0.2,
                outer_metrics=EventMetrics(1, 2, 4, 0.25, 0.5, 1 / 3),
                inner_metrics=EventMetrics(1, 2, 4, 0.25, 0.5, 1 / 3),
                candidate_count=10, candidate_match_recall=0.5,
                micro_candidate_count=6, micro_candidate_match_recall=0.25,
                short_meal_candidate_recall=1.0,
                slices={"duration_lt10": EventMetrics(1, 2, 1, 1.0, 0.5, 2 / 3)},
                timings_seconds={"total": 10.0, "micro_window_oof": 3.0}),
        replace(base, config_hash="fold-b", micro_threshold=0.4,
                outer_metrics=EventMetrics(2, 5, 3, 2 / 3, 0.4, 0.5),
                inner_metrics=EventMetrics(2, 5, 3, 2 / 3, 0.4, 0.5),
                candidate_count=7, candidate_match_recall=1.0,
                micro_candidate_count=5, micro_candidate_match_recall=1.0,
                short_meal_candidate_recall=0.5,
                slices={"duration_lt10": EventMetrics(1, 5, 2, 0.5, 0.2, 2 / 7)},
                timings_seconds={"total": 20.0, "micro_window_oof": 4.0}),
    )


def test_aggregate_fold_results_recomputes_counts_and_truth_weighted_recalls():
    from src.pipeline.runner import aggregate_fold_results

    configs = [RunConfig(outer_fold=0), RunConfig(outer_fold=1)]
    summary = aggregate_fold_results(configs, aggregate_test_results())
    for name in ("inner_metrics", "outer_metrics"):
        assert summary[name] == {"n_tp": 3, "n_pred": 7, "n_true": 7,
                                 "sensitivity": 3 / 7, "ppv": 3 / 7, "f1": 3 / 7}
    assert summary["candidate_count"] == 17
    assert summary["micro_candidate_count"] == 11
    assert summary["candidate_match_recall"] == pytest.approx(5 / 7)
    assert summary["micro_candidate_match_recall"] == pytest.approx(4 / 7)
    assert summary["short_meal_candidate_recall"] == pytest.approx(2 / 3)
    assert summary["slices"]["duration_lt10"]["n_true"] == 3
    assert summary["slices"]["duration_lt10"]["sensitivity"] == pytest.approx(2 / 3)
    assert summary["timings_seconds"] == {"total": 30.0, "micro_window_oof": 7.0}
    assert summary["folds"] == [
        {"outer_fold": 0, "config_hash": "fold-a", "micro_threshold": 0.2},
        {"outer_fold": 1, "config_hash": "fold-b", "micro_threshold": 0.4},
    ]


@pytest.mark.parametrize("folds,count", [([], 0), ([0], 0), ([0, 1], 1), ([0, 0], 2)])
def test_aggregate_fold_results_rejects_empty_mismatched_or_duplicate_folds(folds, count):
    from src.pipeline.runner import aggregate_fold_results

    with pytest.raises(ValueError):
        aggregate_fold_results([RunConfig(outer_fold=fold) for fold in folds], aggregate_test_results()[:count])


def test_aggregate_fold_results_handles_no_truths_or_predictions():
    from src.pipeline.runner import aggregate_fold_results, _fold_result_from_dict

    summary = aggregate_fold_results([RunConfig(outer_fold=0)], [_fold_result_from_dict(historical_result_payload())])
    assert summary["outer_metrics"]["f1"] == 0.0
    assert summary["candidate_match_recall"] == 0.0
    assert summary["micro_candidate_match_recall"] == 0.0
    assert summary["short_meal_candidate_recall"] == 0.0


def test_experiment_key_is_configuration_only_and_order_independent():
    from src.pipeline.runner import experiment_key

    a = RunConfig(outer_fold=0, micro_enabled=True)
    b = replace(a, outer_fold=1, micro_gravity_align=False)
    assert experiment_key([a, b]) == experiment_key([b, a])
    assert experiment_key([a, b]) == experiment_key([replace(a, outer_fold=4), replace(b, outer_fold=3)])
    assert experiment_key([a, b]) != experiment_key([a, replace(b, micro_threshold_grid=(0.3,))])


@pytest.mark.parametrize("micro,coverage,dimensions", [
    (True, False, (63, 56, 47)), (True, True, (63, 56, 47)),
    (False, False, (63, 37)), (False, True, (63, 42)),
])
def test_filesystem_fold_cache_uses_registered_feature_dimensions(filesystem_runner_source, monkeypatch, micro, coverage, dimensions):
    from src.pipeline.runner import fold_result_to_dict, _fold_result_from_dict, write_json_atomic

    source = filesystem_runner_source
    config = RunConfig(outer_fold=0, micro_enabled=micro, density=DensityConfig(coverage_fix=coverage))
    key = cache_key(config, dimensions, source.input_files(config))
    result = replace(_fold_result_from_dict(historical_result_payload()), config_hash=key,
                     micro_threshold=0.3 if micro else None, verifier_feature_count=dimensions[1])
    write_json_atomic(source.cache_directory / f"fold0_{key}.json", fold_result_to_dict(result))

    def unexpected_load(config):
        pytest.fail("registered cache identity was missed")

    monkeypatch.setattr(source, "load_outer_fold", unexpected_load)
    restored = run_outer_fold(config, source)
    assert restored.config_hash == key
    assert restored.cache_hits == {"fold_result": True}
    assert restored.micro_threshold == result.micro_threshold


@pytest.mark.parametrize("micro", [False, True])
def test_external_weights_rejected_before_any_file_or_cache_access(tmp_path, monkeypatch, micro):
    from src.pipeline import runner

    source = runner.FilesystemDataSource(tmp_path)
    calls = []

    def forbidden_access(*args, **kwargs):
        calls.append("access")
        pytest.fail("external weight rejection happened after filesystem/cache access")

    monkeypatch.setattr(source, "input_files", forbidden_access)
    monkeypatch.setattr(source, "load_outer_fold", forbidden_access)
    monkeypatch.setattr(runner, "cache_key", forbidden_access)
    with pytest.raises(ValueError, match="external_fd_weight_grid"):
        run_outer_fold(RunConfig(outer_fold=0, micro_enabled=micro, external_fd_weight_grid=(0.0, 0.5)), source)
    assert calls == []


def test_cli_all_writes_multiscale_configs_fold_outputs_and_summary(tmp_path, monkeypatch):
    from scripts import crossfit_event_stack as cli

    monkeypatch.setattr(sys, "argv", ["crossfit", "--fold", "all", "--micro-enabled", "--no-micro-gravity-align",
                                    "--micro-threshold-grid", "0.4,0.2", "--micro-positive-middle-fraction", "0.6"])
    monkeypatch.setattr(cli.project_config, "OUTPUT_DIR", tmp_path)
    recorded_configs = []

    def evaluate(configs, workers, force):
        recorded_configs.extend(configs)
        return [replace(aggregate_test_results()[0], config_hash=f"hash-{config.outer_fold}") for config in configs]

    monkeypatch.setattr(cli, "run_folds", evaluate)
    assert cli.main() == 0
    assert [config.outer_fold for config in recorded_configs] == list(range(5))
    assert all(config.micro_enabled and not config.micro_gravity_align for config in recorded_configs)
    assert all(config.micro_threshold_grid == (0.2, 0.4) and config.micro_positive_middle_fraction == 0.6 for config in recorded_configs)
    output = tmp_path / "crossfit"
    assert len(list(output.glob("fold*.json"))) == 5
    summaries = list(output.glob("summary_*.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["outer_metrics"]["n_tp"] == 5
    assert summary["outer_metrics"]["n_true"] == 20
    assert [config["outer_fold"] for config in summary["run_configs"]] == list(range(5))
    assert summaries[0].stem == f"summary_{cli.experiment_key(recorded_configs)}"
    assert not list(output.glob("*.tmp"))


@pytest.mark.parametrize("arguments,message", [
    (["--device", "cuda"], "cuda is unavailable"),
    (["--micro-enabled", "--verifier-features", "raw_summary"], "raw_summary"),
])
def test_cli_rejects_unsupported_modes_before_training(monkeypatch, arguments, message):
    from scripts import crossfit_event_stack as cli

    monkeypatch.setattr(sys, "argv", ["crossfit", *arguments])
    monkeypatch.setattr(cli, "run_folds", lambda *args, **kwargs: pytest.fail("unsupported mode reached training"))
    with pytest.raises(SystemExit, match=message):
        cli.main()


def test_schema_five_cache_key_changes_for_stacked_candidate_control_settings():
    from src.pipeline import runner

    base = RunConfig(outer_fold=0, candidate_control_enabled=True)
    assert runner.RUNNER_SCHEMA_VERSION == 5
    assert cache_key(base) != cache_key(
        replace(base, verifier_blend_weight_grid=(0.0, 1.0))
    )
    assert cache_key(base) != cache_key(
        replace(base, admission_threshold_grid=(0.2, 0.5))
    )
    assert cache_key(base) != cache_key(
        replace(base, admission_subject_cap_grid=(3, 4))
    )


def test_stacked_result_round_trip_preserves_admission_diagnostics_and_timings():
    from src.pipeline.runner import _fold_result_from_dict, fold_result_to_dict

    base = _fold_result_from_dict(historical_result_payload())
    result = replace(
        base,
        selected_blend_weight=0.25,
        selected_admission_nms_iou=0.5,
        selected_admission_threshold=0.35,
        selected_admission_subject_cap=4,
        raw_union_candidate_count=700,
        admitted_candidate_count=120,
        verifier_logistic_oof_seconds=1.25,
        verifier_lgbm_oof_seconds=2.5,
        verifier_logistic_outer_seconds=0.125,
        verifier_lgbm_outer_seconds=0.25,
    )
    payload = json.loads(json.dumps(fold_result_to_dict(result)))

    assert _fold_result_from_dict(payload) == result
    assert fold_result_to_dict(_fold_result_from_dict(payload)) == payload


def test_schema_four_result_defaults_stacked_admission_diagnostics_to_neutral_values():
    from src.pipeline.runner import _fold_result_from_dict

    result = _fold_result_from_dict(historical_result_payload())

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


def test_verifier_lgbm_factory_is_single_threaded_seeded_and_median_imputed():
    from src.pipeline import runner

    estimator = runner._verifier_lgbm_estimator(20260910)
    parameters = estimator.get_params()

    assert runner.VERIFIER_LGBM_PARAMETERS == {
        "n_estimators": 200,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 40,
        "learning_rate": 0.03,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "class_weight": "balanced",
        "n_jobs": 1,
        "verbosity": -1,
    }
    assert parameters["imputer__strategy"] == "median"
    assert parameters["model__random_state"] == 20260910
    assert parameters["model__n_jobs"] == 1


@pytest.mark.parametrize("parser, value", [
    ("parse_admission_nms_iou_grid", "0.7,0.3,1"),
    ("parse_probability_grid", "0.5,0,1,0.1"),
])
def test_stacked_float_grid_parsers_canonicalize_ascending_unique_finite_values(parser, value):
    from scripts import crossfit_event_stack as cli

    assert getattr(cli, parser)(value) == tuple(sorted(float(item) for item in value.split(",")))


@pytest.mark.parametrize("parser, values", [
    ("parse_admission_nms_iou_grid", ("", "0", "-0.1", "1.1", "nan", "inf", "0.3,0.30")),
    ("parse_probability_grid", ("", "-0.1", "1.1", "nan", "inf", "0.3,0.30")),
])
def test_stacked_float_grid_parsers_reject_noncanonical_domains(parser, values):
    from scripts import crossfit_event_stack as cli

    for value in values:
        with pytest.raises(ValueError):
            getattr(cli, parser)(value)


def test_admission_subject_cap_parser_rejects_bool_and_canonicalizes_order():
    from scripts.crossfit_event_stack import parse_admission_subject_cap_grid

    assert parse_admission_subject_cap_grid("8,3,4") == (3, 4, 8)
    for value in ("", "0", "-1", "3,3", "True", "3,True"):
        with pytest.raises(ValueError, match="positive unique integers"):
            parse_admission_subject_cap_grid(value)


@pytest.mark.parametrize("kwargs", [
    {"admission_nms_iou_grid": (0.7, 0.3)},
    {"admission_threshold_grid": (0.5, float("nan"))},
    {"verifier_blend_weight_grid": (0.0, 0.0)},
    {"admission_subject_cap_grid": (True,)},
])
def test_stacked_run_config_rejects_invalid_or_noncanonical_registered_grids(kwargs):
    # Construction itself must reject invalid settings before they reach a cache key.
    with pytest.raises(ValueError):
        RunConfig(outer_fold=0, **kwargs)


def test_cli_candidate_control_requires_micro_and_propagates_registered_grids(tmp_path, monkeypatch):
    from scripts import crossfit_event_stack as cli

    monkeypatch.setattr(sys, "argv", ["crossfit", "--candidate-control-enabled"])
    with pytest.raises(SystemExit, match="micro-enabled"):
        cli.main()

    monkeypatch.setattr(sys, "argv", [
        "crossfit", "--fold", "0", "--micro-enabled", "--candidate-control-enabled",
        "--admission-nms-iou-grid", "0.7,0.3",
        "--admission-threshold-grid", "0.65,0.2",
        "--admission-subject-cap-grid", "8,3",
        "--verifier-blend-weight-grid", "1,0.25,0",
        "--admission-minimum-recall", "0.9",
    ])
    monkeypatch.setattr(cli.project_config, "OUTPUT_DIR", tmp_path)
    recorded = []
    monkeypatch.setattr(
        cli,
        "run_folds",
        lambda configs, workers, force: recorded.extend(configs) or [
            replace(aggregate_test_results()[0], config_hash="stacked-contract")
        ],
    )

    assert cli.main() == 0
    assert len(recorded) == 1
    config = recorded[0]
    assert config.outer_fold == 0
    assert config.micro_enabled and config.candidate_control_enabled
    assert config.admission_nms_iou_grid == (0.3, 0.7)
    assert config.admission_threshold_grid == (0.2, 0.65)
    assert config.admission_subject_cap_grid == (3, 8)
    assert config.verifier_blend_weight_grid == (0.0, 0.25, 1.0)
    assert config.admission_minimum_recall == 0.9
