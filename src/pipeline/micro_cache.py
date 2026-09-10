"""Versioned 15-second IMU feature-cache construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from src.data import manifests
from src.pipeline.imu_features import MicroFeatureConfig
from src.pipeline.imu_features import extract_micro_features, gravity_rotation


MICRO_EXTRACTION_VERSION = 1
MICRO_SAMPLE_SEED = 20260901
MICRO_NEGATIVE_RATIO = 3


@dataclass(frozen=True)
class MicroCacheArrays:
    """The stable, model-facing arrays stored in one micro-window cache."""

    feat: np.ndarray
    label: np.ndarray
    wid: np.ndarray


def cache_metadata(config: MicroFeatureConfig, source_files) -> dict:
    """Fingerprint feature semantics and every source artifact used to build them."""
    sources = []
    for path in sorted((Path(value) for value in source_files), key=str):
        if not path.exists():
            sources.append({"path": str(path.resolve()), "missing": True})
            continue
        stat = path.stat()
        sources.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return {
        "extraction_version": MICRO_EXTRACTION_VERSION,
        "feature_config_hash": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "feature_count": 47,
        "sources": sources,
    }


def write_micro_cache_atomic(path, arrays: MicroCacheArrays, metadata: dict) -> None:
    """Atomically replace a compressed cache, leaving no partial result on errors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                feat=arrays.feat,
                label=arrays.label,
                wid=arrays.wid,
                metadata=json.dumps(metadata, sort_keys=True),
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_micro_metadata(path) -> dict:
    """Read only the JSON metadata from a cache artifact."""
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"].item()))


def read_micro_cache(path, expected_metadata=None) -> MicroCacheArrays:
    """Load a validated cache, ignoring only non-semantic timing metadata."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        actual = json.loads(str(data["metadata"].item()))
        if expected_metadata is not None and any(
            actual.get(key) != value
            for key, value in expected_metadata.items()
            if key != "extraction_seconds"
        ):
            raise ValueError(f"micro cache metadata mismatch: {path}")
        arrays = MicroCacheArrays(
            np.asarray(data["feat"]).copy(),
            np.asarray(data["label"]).copy(),
            np.asarray(data["wid"]).copy(),
        )
    if arrays.feat.ndim != 2 or arrays.feat.shape[1] != 47 or not np.isfinite(arrays.feat).all():
        raise ValueError(f"invalid 47-feature micro cache: {path}")
    if arrays.label.ndim != 1 or arrays.wid.ndim != 1 or not (
        len(arrays.feat) == len(arrays.label) == len(arrays.wid)
    ):
        raise ValueError(f"invalid aligned micro cache arrays: {path}")
    return arrays


def label_micro_window(start_ms, end_ms, meals, positive_fraction=0.5, negative_buffer_ms=300_000):
    """Return a positive, negative, or ambiguous label for one time window."""
    duration = end_ms - start_ms
    overlap = max(
        (
            max(0, min(end_ms, meal_end) - max(start_ms, meal_start))
            for meal_start, meal_end in meals
        ),
        default=0,
    )
    if overlap >= positive_fraction * duration:
        return 1
    distance = min(
        (
            meal_start - end_ms if end_ms <= meal_start else start_ms - meal_end if start_ms >= meal_end else 0
            for meal_start, meal_end in meals
        ),
        default=negative_buffer_ms,
    )
    return 0 if overlap == 0 and distance >= negative_buffer_ms else -1


def sample_training_rows(labels, sessions, seed, negative_ratio=3):
    """Keep positives and bounded, deterministic per-session negatives."""
    labels = np.asarray(labels, dtype=np.int8)
    sessions = np.asarray(sessions).astype(str)
    rng = np.random.default_rng(seed)
    selected = []
    for session_id in sorted(set(sessions)):
        indices = np.flatnonzero(sessions == session_id)
        positives = indices[labels[indices] == 1]
        negatives = indices[labels[indices] == 0]
        allowance = max(negative_ratio * len(positives), 1)
        if len(negatives) > allowance:
            negatives = np.sort(rng.choice(negatives, allowance, replace=False))
        selected.extend(positives.tolist())
        selected.extend(negatives.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def meals_by_session(root) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return only meal intervals fully contained by each indexed session."""
    del root  # Dataset manifests are project-global; the argument keeps this helper replaceable in tests.
    index = manifests.load_sensor_index()
    meals = manifests.load_meals()
    by_external: dict[str, list[tuple[int, int]]] = {}
    for _, meal in meals.iterrows():
        by_external.setdefault(str(meal["externalid"]), []).append(
            (int(meal["before_ms"]), int(meal["after_ms"]))
        )
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for _, session in index.iterrows():
        start = int(session["timeStamp.startTime"])
        end = int(session["timeStamp.endTime"])
        contained = tuple(
            interval
            for interval in by_external.get(str(session["externalid"]), [])
            if interval[0] >= start and interval[1] <= end
        )
        result[str(session["session_id"])] = contained
    return result


def _fold_manifest_path(root: Path, fold: int) -> Path:
    return root / "cache" / "splits" / f"fold{fold}.json"


def split_sessions(root, fold: int, split: str) -> tuple[str, ...]:
    """Resolve one leakage-safe split from its stored fold manifest."""
    root = Path(root)
    with _fold_manifest_path(root, fold).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    training = tuple(str(value) for value in manifest["train_sessions"])
    validation = tuple(str(value) for value in manifest["val_sessions"])
    contained_meals = meals_by_session(root)
    if split == "train":
        return training
    if split == "meal_train":
        return tuple(session for session in training if contained_meals.get(session, ()))
    if split == "no_meal_train":
        return tuple(session for session in training if not contained_meals.get(session, ()))
    if split == "val":
        return validation
    raise ValueError(f"unknown micro-cache split: {split}")


def _empty_micro_arrays() -> MicroCacheArrays:
    return MicroCacheArrays(
        np.empty((0, 47), dtype=np.float32),
        np.empty(0, dtype=np.int8),
        np.empty(0, dtype=str),
    )


def _session_rows(task) -> MicroCacheArrays:
    """Extract complete, covered windows for one session; safe for process workers."""
    path, session_id, meals, config = task
    if not Path(path).exists():
        return _empty_micro_arrays()
    with np.load(path, allow_pickle=False) as data:
        valid = np.asarray(data["imu_valid"], dtype=bool)
        time_ms = np.asarray(data["t_acc"], dtype=np.int64)[valid]
        acc = np.asarray(data["acc"], dtype=np.float32)[:, valid]
        gyro = np.asarray(data["gyro"], dtype=np.float32)[:, valid]
    positive_deltas = np.diff(time_ms)
    positive_deltas = positive_deltas[positive_deltas > 0]
    if not len(positive_deltas):
        return _empty_micro_arrays()
    sample_period_ms = float(np.median(positive_deltas))
    sample_rate_hz = 1000.0 / sample_period_ms
    if config.gravity_align:
        rotation = gravity_rotation(np.median(acc, axis=1))
        acc, gyro = rotation @ acc, rotation @ gyro
    session_end_exclusive = int(round(time_ms[-1] + sample_period_ms))
    starts = np.arange(
        int(time_ms[0]),
        session_end_exclusive - config.window_ms + 1,
        config.stride_ms,
    )
    expected_rows = config.coverage_min * config.window_ms * sample_rate_hz / 1000.0
    features, labels, windows = [], [], []
    for start in starts:
        end = int(start + config.window_ms)
        left, right = np.searchsorted(time_ms, (start, end))
        if right - left < expected_rows:
            continue
        features.append(extract_micro_features(acc[:, left:right], gyro[:, left:right], sample_rate_hz))
        labels.append(label_micro_window(int(start), end, meals))
        windows.append(json.dumps((session_id, int(start), end)))
    if not features:
        return _empty_micro_arrays()
    return MicroCacheArrays(
        np.asarray(features, dtype=np.float32).reshape(-1, 47),
        np.asarray(labels, dtype=np.int8),
        np.asarray(windows),
    )


def _combine_rows(rows: list[MicroCacheArrays]) -> MicroCacheArrays:
    if not rows or not any(len(row.label) for row in rows):
        return _empty_micro_arrays()
    return MicroCacheArrays(
        np.concatenate([row.feat for row in rows]),
        np.concatenate([row.label for row in rows]),
        np.concatenate([row.wid for row in rows]),
    )


def _window_sessions(window_ids: np.ndarray) -> np.ndarray:
    return np.asarray([str(json.loads(value)[0]) for value in window_ids])


def build_micro_split(
    root,
    *,
    fold: int,
    split: str,
    config: MicroFeatureConfig,
    workers: int = 8,
    limit: int = 0,
    force: bool = False,
) -> Path:
    """Build or reuse one versioned fold/split cache and return its artifact path."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    root = Path(root)
    sessions = split_sessions(root, fold, split)
    if limit:
        sessions = sessions[:limit]
    session_paths = tuple(root / "cache" / "sessions" / f"{session_id}.npz" for session_id in sessions)
    manifest_path = _fold_manifest_path(root, fold)
    source_files = (*session_paths, *(() if not manifest_path.exists() else (manifest_path,)))
    expected_metadata = cache_metadata(config, source_files)
    cache_directory = root / "cache" / "micro15"
    if limit:
        cache_directory = cache_directory / "smoke"
    destination = cache_directory / f"fold{fold}_{split}.npz"
    if destination.exists() and not force:
        try:
            read_micro_cache(destination, expected_metadata)
            return destination
        except ValueError:
            pass
    meal_map = meals_by_session(root)
    tasks = [
        (path, session_id, meal_map.get(session_id, ()), config)
        for path, session_id in zip(session_paths, sessions)
    ]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=min(workers, 8)) as executor:
        rows = list(executor.map(_session_rows, tasks))
    arrays = _combine_rows(rows)
    if split == "train" and len(arrays.label):
        keep = sample_training_rows(
            arrays.label,
            _window_sessions(arrays.wid),
            seed=MICRO_SAMPLE_SEED,
            negative_ratio=MICRO_NEGATIVE_RATIO,
        )
        arrays = MicroCacheArrays(arrays.feat[keep], arrays.label[keep], arrays.wid[keep])
    stored_metadata = {**expected_metadata, "extraction_seconds": time.monotonic() - started}
    write_micro_cache_atomic(destination, arrays, stored_metadata)
    return destination
