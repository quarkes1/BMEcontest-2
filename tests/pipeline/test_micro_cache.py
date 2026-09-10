from pathlib import Path

import numpy as np
import pytest

from src.pipeline.imu_features import MicroFeatureConfig
from src.pipeline.micro_cache import label_micro_window, sample_training_rows


def test_micro_label_boundaries_are_inclusive_and_buffered():
    meals = ((100_000, 200_000),)
    assert label_micro_window(92_500, 107_500, meals) == 1
    assert label_micro_window(200_000, 215_000, meals) == -1
    assert label_micro_window(500_000, 515_000, meals) == 0
    assert label_micro_window(499_999, 514_999, meals) == -1


def test_training_sampler_keeps_all_positives_and_one_pure_negative_session_row():
    labels = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    sessions = np.array(["meal"] * 6 + ["pure-negative"])
    first = sample_training_rows(labels, sessions, seed=17, negative_ratio=3)
    second = sample_training_rows(labels, sessions, seed=17, negative_ratio=3)
    np.testing.assert_array_equal(first, second)
    assert set(np.flatnonzero(labels == 1)).issubset(set(first))
    assert 6 in first


def test_micro_cache_round_trip_and_metadata_rejection(tmp_path: Path):
    from src.pipeline.micro_cache import (
        MicroCacheArrays,
        cache_metadata,
        read_micro_cache,
        write_micro_cache_atomic,
    )

    source = tmp_path / "session.npz"
    np.savez(source, marker=np.array([1]))
    arrays = MicroCacheArrays(
        feat=np.ones((1, 47), dtype=np.float32),
        label=np.array([0], dtype=np.int8),
        wid=np.array(['["s1", 0, 15000]']),
    )
    expected = cache_metadata(MicroFeatureConfig(), (source,))
    destination = tmp_path / "micro.npz"
    write_micro_cache_atomic(destination, arrays, expected)
    loaded = read_micro_cache(destination, expected)
    np.testing.assert_array_equal(loaded.feat, arrays.feat)
    assert not list(tmp_path.glob("*.tmp"))

    changed = dict(expected)
    changed["extraction_version"] += 1
    with pytest.raises(ValueError, match="metadata mismatch"):
        read_micro_cache(destination, changed)


def test_metadata_validation_ignores_recorded_extraction_seconds(tmp_path: Path):
    from src.pipeline.micro_cache import MicroCacheArrays, read_micro_cache, write_micro_cache_atomic

    destination = tmp_path / "micro.npz"
    expected = {"extraction_version": 1, "sources": []}
    stored = {**expected, "extraction_seconds": 0.03}
    arrays = MicroCacheArrays(np.zeros((0, 47), np.float32), np.zeros(0, np.int8), np.empty(0, str))
    write_micro_cache_atomic(destination, arrays, stored)
    read_micro_cache(destination, expected)


def test_build_micro_split_uses_timestamp_coverage_and_session_rotation(tmp_path, monkeypatch):
    from src.pipeline.micro_cache import build_micro_split, read_micro_cache

    session_dir = tmp_path / "cache" / "sessions"
    session_dir.mkdir(parents=True)
    time_ms = np.arange(0, 30_000, 10, dtype=np.int64)
    acc = np.vstack((np.zeros(len(time_ms)), np.zeros(len(time_ms)), np.ones(len(time_ms)))).astype(np.float32)
    gyro = np.zeros_like(acc)
    np.savez(
        session_dir / "s1.npz",
        acc=acc,
        gyro=gyro,
        t_acc=time_ms,
        imu_valid=np.ones(len(time_ms), dtype=bool),
    )
    monkeypatch.setattr("src.pipeline.micro_cache.split_sessions", lambda root, fold, split: ("s1",))
    monkeypatch.setattr("src.pipeline.micro_cache.meals_by_session", lambda root: {"s1": ()})
    output = build_micro_split(tmp_path, fold=0, split="val", config=MicroFeatureConfig(), workers=1)
    arrays = read_micro_cache(output)
    assert arrays.feat.shape == (3, 47)
    assert arrays.label.tolist() == [0, 0, 0]


def test_build_micro_split_skips_unavailable_binary_session_and_invalidates_on_arrival(tmp_path, monkeypatch):
    from src.pipeline.micro_cache import (
        _session_rows,
        build_micro_split,
        cache_metadata,
        read_micro_cache,
        read_micro_metadata,
    )

    session_dir = tmp_path / "cache" / "sessions"
    session_dir.mkdir(parents=True)
    manifest_dir = tmp_path / "cache" / "splits"
    manifest_dir.mkdir()
    manifest = manifest_dir / "fold0.json"
    manifest.write_text('{"train_sessions": ["present", "absent"], "val_sessions": []}', encoding="utf-8")
    present = session_dir / "present.npz"
    absent = session_dir / "absent.npz"
    time_ms = np.arange(0, 30_000, 10, dtype=np.int64)
    acc = np.vstack((np.zeros(len(time_ms)), np.zeros(len(time_ms)), np.ones(len(time_ms)))).astype(np.float32)
    np.savez(
        present,
        acc=acc,
        gyro=np.zeros_like(acc),
        t_acc=time_ms,
        imu_valid=np.ones(len(time_ms), dtype=bool),
    )
    monkeypatch.setattr("src.pipeline.micro_cache.split_sessions", lambda root, fold, split: ("present", "absent"))
    monkeypatch.setattr("src.pipeline.micro_cache.meals_by_session", lambda root: {"present": (), "absent": ()})

    empty = _session_rows((absent, "absent", (), MicroFeatureConfig()))
    assert empty.feat.shape == (0, 47)
    assert empty.label.shape == empty.wid.shape == (0,)

    output = build_micro_split(tmp_path, fold=0, split="val", config=MicroFeatureConfig(), workers=1)
    arrays = read_micro_cache(output)
    assert arrays.feat.shape == (3, 47)
    assert all('"present"' in window for window in arrays.wid)
    old_metadata = read_micro_metadata(output)
    assert {"path": str(absent.resolve()), "missing": True} in old_metadata["sources"]

    np.savez(
        absent,
        acc=acc,
        gyro=np.zeros_like(acc),
        t_acc=time_ms,
        imu_valid=np.ones(len(time_ms), dtype=bool),
    )
    changed_metadata = cache_metadata(MicroFeatureConfig(), (present, absent, manifest))
    assert changed_metadata != {key: value for key, value in old_metadata.items() if key != "extraction_seconds"}
    with pytest.raises(ValueError, match="metadata mismatch"):
        read_micro_cache(output, changed_metadata)


@pytest.mark.parametrize(
    ("split", "expected"),
    (
        ("train", ("train-meal", "train-empty")),
        ("meal_train", ("train-meal",)),
        ("no_meal_train", ("train-empty",)),
        ("val", ("validation",)),
    ),
)
def test_split_sessions_selects_fold_members_by_meal_containment(tmp_path: Path, split, expected):
    from src.pipeline.micro_cache import split_sessions

    manifest = tmp_path / "cache" / "splits"
    manifest.mkdir(parents=True)
    (manifest / "fold0.json").write_text(
        '{"train_sessions": ["train-meal", "train-empty"], "val_sessions": ["validation"]}',
        encoding="utf-8",
    )
    # The actual contained-meal classification is checked through the module-level helper.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "src.pipeline.micro_cache.meals_by_session",
        lambda root: {"train-meal": ((10, 20),), "train-empty": (), "validation": ()},
    )
    try:
        assert split_sessions(tmp_path, 0, split) == expected
    finally:
        monkeypatch.undo()


def test_micro_builder_cli_parses_defaults_and_explicit_smoke_options(monkeypatch):
    from scripts.build_micro_features import parse_args

    monkeypatch.setattr("sys.argv", ["build_micro_features.py"])
    defaults = parse_args()
    assert (defaults.fold, defaults.split, defaults.workers, defaults.limit) == ("all", "all", 8, 0)
    assert not defaults.no_gravity_align
    assert not defaults.force

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_micro_features.py",
            "--fold",
            "2",
            "--split",
            "val",
            "--workers",
            "1",
            "--limit",
            "4",
            "--no-gravity-align",
            "--force",
        ],
    )
    options = parse_args()
    assert (options.fold, options.split, options.workers, options.limit) == ("2", "val", 1, 4)
    assert options.no_gravity_align
    assert options.force
