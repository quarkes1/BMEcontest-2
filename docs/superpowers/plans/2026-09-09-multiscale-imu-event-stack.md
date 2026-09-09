# Multi-scale IMU Event Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a leakage-safe 15-second ACC+GYRO branch that unions short-motion candidates with the existing 240-second macro branch and moves the locked development F1 toward 0.65.

**Architecture:** Extract one reusable 47-feature micro representation from timestamped session caches, score it with subject-disjoint LightGBM cross-fitting, and select a permissive micro-candidate threshold from outer-train OOF data only. Preserve macro event geometry when streams agree, retain micro-only rescues, and train the existing logistic event verifier on an exact 56-feature multi-scale surface.

**Tech Stack:** Python 3.11, NumPy, SciPy, scikit-learn, LightGBM, pytest, NPZ caches, process-level CPU parallelism

**Spec:** `docs/superpowers/specs/2026-09-09-multiscale-imu-event-stack-design.md`

## Global Constraints

- Official scoring is greedy one-to-one event matching at IoU≥0.25.
- No outer-validation subject may influence feature selection, model fitting, score fusion, post-processing, threshold selection, or variant selection inside the reported run.
- Micro windows are exactly 15,000 ms with a 7,500 ms stride and require at least 0.80 expected IMU-row coverage.
- The micro feature vector is exactly 47 finite float32 values and uses ACC+GYRO only.
- The multi-scale verifier feature vector is exactly 56 values (`37 + 19`).
- LightGBM uses one native thread per fold worker; outer folds use bounded process parallelism and never oversubscribe BLAS.
- Every test, extraction, and training command uses `D:/Anaconda3/envs/bme/python.exe` (Python 3.11.15, LightGBM 4.7.0); the system Python 3.13 environment is not a project runtime.
- The first implementation plan is target-domain only. FD-I/FD-II transfer receives a separate implementation plan only after this branch passes its acceptance gate.
- `micro_enabled=False` preserves macro-only prediction semantics and the existing 37/42-dimensional verifier paths.
- Reusable code belongs under `src/`, durable thin commands under `scripts/`, tests under `tests/`, decisions under `docs/` or `README.md`, and compact run evidence under ignored `outputs/crossfit/`.
- Every coherent increment ends with focused tests, cleanup, `git add`, `git commit`, and an empty `git status --short`.

## File Map

- Create `src/pipeline/imu_features.py`: pure gravity alignment and 47-feature extraction.
- Create `src/pipeline/micro_cache.py`: micro-window labeling, deterministic sampling, cache metadata, atomic NPZ I/O, and fold split construction.
- Create `scripts/build_micro_features.py`: thin CLI over `micro_cache.build_micro_split`.
- Modify `src/pipeline/event_stack.py`: micro smoothing/candidates, threshold selection, deterministic macro/micro union, and 56-feature construction.
- Modify `src/pipeline/runner.py`: micro configuration/data/result schema, LightGBM cross-fitting, multi-scale verifier flow, caching, and aggregate reporting.
- Modify `scripts/crossfit_event_stack.py`: registered micro flags and compact aggregate output.
- Create `tests/pipeline/test_imu_features.py`: feature-shape, numerical, gravity, and spectral tests.
- Create `tests/pipeline/test_micro_cache.py`: labels, sampling, metadata, atomic I/O, and split smoke tests.
- Modify `tests/pipeline/test_event_stack.py`: micro candidate, selection, union, and feature-width tests.
- Modify `tests/pipeline/test_runner.py`: config/cache compatibility and CLI parser tests.
- Create `tests/pipeline/test_multiscale_runner.py`: end-to-end subject isolation and outer-label independence tests.
- Modify `README.md`: reproducible commands, accepted/rejected results, timing, and current target gap.
- Modify `docs/三阶段重构设计.md`: experiment record and architecture decision.

---

### Task 1: Pure 47-feature ACC+GYRO representation

**Files:**
- Create: `src/pipeline/imu_features.py`
- Create: `tests/pipeline/test_imu_features.py`

**Interfaces:**
- Consumes: `(3, n)` ACC/GYRO arrays and a positive finite sampling rate.
- Produces: `MicroFeatureConfig`, `gravity_rotation(gravity) -> np.ndarray`, and `extract_micro_features(acc, gyro, sample_rate_hz) -> np.ndarray`.

- [ ] **Step 1: Write failing gravity and validation tests**

```python
# tests/pipeline/test_imu_features.py
import numpy as np
import pytest

from src.pipeline.imu_features import gravity_rotation, extract_micro_features


@pytest.mark.parametrize(
    "gravity",
    (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 0.0]),
    ),
)
def test_gravity_rotation_is_finite_orthogonal(gravity):
    rotation = gravity_rotation(gravity)
    assert rotation.shape == (3, 3)
    assert np.isfinite(rotation).all()
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    if np.linalg.norm(gravity) > 0:
        aligned = rotation @ (gravity / np.linalg.norm(gravity))
        np.testing.assert_allclose(aligned, [0.0, 0.0, 1.0], atol=1e-6)


def test_micro_features_reject_misaligned_inputs():
    with pytest.raises(ValueError, match="shape"):
        extract_micro_features(np.zeros((2, 10)), np.zeros((3, 10)), 100.0)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        extract_micro_features(np.zeros((3, 10)), np.zeros((3, 10)), 0.0)
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_imu_features.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.pipeline.imu_features'`.

- [ ] **Step 3: Implement the configuration, input validation, and robust gravity rotation**

```python
# src/pipeline/imu_features.py
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MicroFeatureConfig:
    window_ms: int = 15_000
    stride_ms: int = 7_500
    coverage_min: float = 0.80
    gravity_align: bool = True


def gravity_rotation(gravity: np.ndarray) -> np.ndarray:
    vector = np.asarray(gravity, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    source = vector / norm
    target = np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if cosine > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(source, target)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)
```

- [ ] **Step 4: Add failing feature-layout and dominant-frequency tests**

```python
def test_micro_features_are_exact_finite_float32_for_constant_channels():
    acc = np.vstack((np.ones(1500), np.zeros(1500), -np.ones(1500)))
    gyro = np.zeros((3, 1500))
    result = extract_micro_features(acc, gyro, 100.0)
    assert result.shape == (47,)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_micro_features_recover_three_hz_acc_magnitude_peak():
    sample_rate = 100.0
    time = np.arange(1500) / sample_rate
    acc = np.vstack((2.0 + np.sin(2 * np.pi * 3.0 * time), np.zeros(1500), np.zeros(1500)))
    gyro = np.zeros_like(acc)
    result = extract_micro_features(acc, gyro, sample_rate)
    assert result[28] == pytest.approx(3.0, abs=0.08)
    assert result[33] > result[34]
```

- [ ] **Step 5: Implement the exact 47-feature order**

```python
def _safe_axis_features(values: np.ndarray) -> list[float]:
    centered = values - values.mean()
    crossings = np.count_nonzero(np.signbit(centered[1:]) != np.signbit(centered[:-1]))
    std = float(values.std())
    return [
        float(values.mean()), std, float(values.min()), float(values.max()),
        float(np.sqrt(np.mean(values ** 2))),
        float(stats.skew(values, bias=False)) if std > 1e-9 else 0.0,
        float(stats.kurtosis(values, fisher=True, bias=False)) if std > 1e-9 else 0.0,
        float(crossings / max(len(values) - 1, 1)),
    ]


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() < 1e-9 or right.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spectrum_features(magnitude: np.ndarray, sample_rate_hz: float) -> list[float]:
    centered = magnitude - magnitude.mean()
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequency = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    keep = frequency >= 0.5
    kept_power, kept_frequency = power[keep], frequency[keep]
    total = float(kept_power.sum())
    if total <= 1e-12:
        return [0.0] * 7
    probability = kept_power / total
    dominant = int(np.argmax(kept_power))
    entropy_denominator = np.log(max(len(probability), 2))
    band = lambda low, high: float(kept_power[(kept_frequency >= low) & (kept_frequency < high)].sum() / total)
    return [
        float(kept_frequency[dominant]), float(probability[dominant]),
        float((kept_frequency * probability).sum()),
        float(-(probability * np.log(probability + 1e-12)).sum() / entropy_denominator),
        band(0.5, 2.0), band(2.0, 5.0), band(5.0, 10.0),
    ]


def extract_micro_features(acc: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    acc_values = np.asarray(acc, dtype=np.float64)
    gyro_values = np.asarray(gyro, dtype=np.float64)
    if acc_values.ndim != 2 or gyro_values.ndim != 2 or acc_values.shape[0] != 3 or acc_values.shape != gyro_values.shape:
        raise ValueError("acc and gyro must have equal shape (3, n)")
    if acc_values.shape[1] < 3 or not np.isfinite(acc_values).all() or not np.isfinite(gyro_values).all():
        raise ValueError("sensor arrays must contain at least three finite samples")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    features = []
    for axis in acc_values:
        features.extend(_safe_axis_features(axis))
    features.extend([
        float(np.mean(np.abs(acc_values).sum(axis=0))),
        _safe_correlation(acc_values[0], acc_values[1]),
        _safe_correlation(acc_values[0], acc_values[2]),
        _safe_correlation(acc_values[1], acc_values[2]),
    ])
    features.extend(_spectrum_features(np.linalg.norm(acc_values, axis=0), sample_rate_hz))
    for axis in gyro_values:
        features.extend([float(axis.mean()), float(axis.std()), float(np.sqrt(np.mean(axis ** 2))), float(np.mean(axis ** 2))])
    result = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if result.shape != (47,):
        raise RuntimeError(f"micro feature contract produced {result.shape}, expected (47,)")
    return result
```

- [ ] **Step 6: Run tests, inspect the diff, and commit**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_imu_features.py -q`

Expected: all tests pass.

Run: `git diff --check`

```bash
git add src/pipeline/imu_features.py tests/pipeline/test_imu_features.py
git commit -m "feat: add micro imu feature extraction"
```

---

### Task 2: Versioned micro-window cache and thin builder CLI

**Files:**
- Create: `src/pipeline/micro_cache.py`
- Create: `scripts/build_micro_features.py`
- Create: `tests/pipeline/test_micro_cache.py`

**Interfaces:**
- Consumes: `MicroFeatureConfig`, `cache/sessions/*.npz`, fold manifests, and meal intervals.
- Produces: `MicroCacheArrays`, `label_micro_window`, `sample_training_rows`, `read_micro_metadata`, `read_micro_cache`, `write_micro_cache_atomic`, and `build_micro_split`.

- [ ] **Step 1: Write failing label and deterministic-sampling tests**

```python
# tests/pipeline/test_micro_cache.py
import numpy as np

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
```

- [ ] **Step 2: Run the focused test and confirm the missing-module failure**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_micro_cache.py -q`

Expected: collection fails because `src.pipeline.micro_cache` does not exist.

- [ ] **Step 3: Implement label geometry and per-session sampling**

```python
# src/pipeline/micro_cache.py
def label_micro_window(start_ms, end_ms, meals, positive_fraction=0.5, negative_buffer_ms=300_000):
    duration = end_ms - start_ms
    overlap = max((max(0, min(end_ms, meal_end) - max(start_ms, meal_start)) for meal_start, meal_end in meals), default=0)
    if overlap >= positive_fraction * duration:
        return 1
    distance = min(
        (meal_start - end_ms if end_ms <= meal_start else start_ms - meal_end if start_ms >= meal_end else 0 for meal_start, meal_end in meals),
        default=negative_buffer_ms,
    )
    return 0 if overlap == 0 and distance >= negative_buffer_ms else -1


def sample_training_rows(labels, sessions, seed, negative_ratio=3):
    labels = np.asarray(labels, dtype=np.int8)
    sessions = np.asarray(sessions).astype(str)
    rng = np.random.default_rng(seed)
    selected = []
    for sid in sorted(set(sessions)):
        indices = np.flatnonzero(sessions == sid)
        positives = indices[labels[indices] == 1]
        negatives = indices[labels[indices] == 0]
        allowance = max(negative_ratio * len(positives), 1)
        if len(negatives) > allowance:
            negatives = np.sort(rng.choice(negatives, allowance, replace=False))
        selected.extend(positives.tolist())
        selected.extend(negatives.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)
```

- [ ] **Step 4: Add failing metadata, atomic-I/O, and mismatch tests**

```python
from pathlib import Path

import pytest

from src.pipeline.imu_features import MicroFeatureConfig
from src.pipeline.micro_cache import (
    MicroCacheArrays,
    cache_metadata,
    read_micro_cache,
    write_micro_cache_atomic,
)


def test_micro_cache_round_trip_and_metadata_rejection(tmp_path: Path):
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
```

- [ ] **Step 5: Implement exact cache metadata and atomic NPZ I/O**

```python
MICRO_EXTRACTION_VERSION = 1
MICRO_SAMPLE_SEED = 20260901
MICRO_NEGATIVE_RATIO = 3


@dataclass(frozen=True)
class MicroCacheArrays:
    feat: np.ndarray
    label: np.ndarray
    wid: np.ndarray


def cache_metadata(config, source_files):
    sources = []
    for path in sorted((Path(value) for value in source_files), key=str):
        stat = path.stat()
        sources.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return {
        "extraction_version": MICRO_EXTRACTION_VERSION,
        "feature_config_hash": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "feature_count": 47,
        "sources": sources,
    }


def write_micro_cache_atomic(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, feat=arrays.feat, label=arrays.label, wid=arrays.wid, metadata=json.dumps(metadata, sort_keys=True))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_micro_metadata(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"].item()))


def read_micro_cache(path, expected_metadata=None):
    with np.load(path, allow_pickle=False) as data:
        actual = json.loads(str(data["metadata"].item()))
        if expected_metadata is not None and any(actual.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"micro cache metadata mismatch: {path}")
        arrays = MicroCacheArrays(np.asarray(data["feat"]).copy(), np.asarray(data["label"]).copy(), np.asarray(data["wid"]).copy())
    if arrays.feat.ndim != 2 or arrays.feat.shape[1] != 47 or not np.isfinite(arrays.feat).all():
        raise ValueError(f"invalid 47-feature micro cache: {path}")
    return arrays
```

- [ ] **Step 6: Add a synthetic-session split-build test**

```python
def test_build_micro_split_uses_timestamp_coverage_and_session_rotation(tmp_path, monkeypatch):
    session_dir = tmp_path / "cache" / "sessions"
    session_dir.mkdir(parents=True)
    time_ms = np.arange(0, 30_000, 10, dtype=np.int64)
    acc = np.vstack((np.zeros(len(time_ms)), np.zeros(len(time_ms)), np.ones(len(time_ms)))).astype(np.float32)
    gyro = np.zeros_like(acc)
    np.savez(session_dir / "s1.npz", acc=acc, gyro=gyro, t_acc=time_ms, imu_valid=np.ones(len(time_ms), dtype=bool))
    monkeypatch.setattr("src.pipeline.micro_cache.split_sessions", lambda root, fold, split: ("s1",))
    monkeypatch.setattr("src.pipeline.micro_cache.meals_by_session", lambda root: {"s1": ()})
    output = build_micro_split(tmp_path, fold=0, split="val", config=MicroFeatureConfig(), workers=1)
    arrays = read_micro_cache(output)
    assert arrays.feat.shape == (3, 47)
    assert arrays.label.tolist() == [0, 0, 0]
```

- [ ] **Step 7: Implement session extraction and fold split construction**

The worker must estimate sampling rate from the median positive timestamp difference, rotate the full valid session once when `gravity_align=True`, reject windows below `coverage_min * window_seconds * sample_rate_hz`, and call `extract_micro_features` on `[start_ms, end_ms)` rows. `train` applies `sample_training_rows`; the other three splits retain every valid window and all `-1` labels.

```python
def _session_rows(task):
    path, sid, meals, config = task
    with np.load(path, allow_pickle=False) as data:
        valid = np.asarray(data["imu_valid"], dtype=bool)
        time_ms = np.asarray(data["t_acc"])[valid]
        acc = np.asarray(data["acc"], dtype=np.float32)[:, valid]
        gyro = np.asarray(data["gyro"], dtype=np.float32)[:, valid]
    positive_deltas = np.diff(time_ms)
    positive_deltas = positive_deltas[positive_deltas > 0]
    if not len(positive_deltas):
        return MicroCacheArrays(np.empty((0, 47), np.float32), np.empty(0, np.int8), np.empty(0, str))
    sample_period_ms = float(np.median(positive_deltas))
    sample_rate_hz = 1000.0 / sample_period_ms
    if config.gravity_align:
        rotation = gravity_rotation(np.median(acc, axis=1))
        acc, gyro = rotation @ acc, rotation @ gyro
    session_end_exclusive = int(round(time_ms[-1] + sample_period_ms))
    starts = np.arange(int(time_ms[0]), session_end_exclusive - config.window_ms + 1, config.stride_ms)
    features, labels, windows = [], [], []
    expected_rows = config.coverage_min * config.window_ms * sample_rate_hz / 1000.0
    for start in starts:
        end = int(start + config.window_ms)
        left, right = np.searchsorted(time_ms, (start, end))
        if right - left < expected_rows:
            continue
        features.append(extract_micro_features(acc[:, left:right], gyro[:, left:right], sample_rate_hz))
        labels.append(label_micro_window(int(start), end, meals))
        windows.append(json.dumps((sid, int(start), end)))
    return MicroCacheArrays(np.asarray(features, np.float32).reshape(-1, 47), np.asarray(labels, np.int8), np.asarray(windows))
```

`split_sessions` must map `train` to all fold training sessions, `meal_train` to training sessions with contained meals, `no_meal_train` to every training session without a contained meal, and `val` to all validation sessions. `build_micro_split` must use `ProcessPoolExecutor(max_workers=min(workers, 8))`, preserve input session order through `executor.map`, use `MICRO_SAMPLE_SEED=20260901` and `MICRO_NEGATIVE_RATIO=3` for `train`, include the exact session NPZ and manifest signatures in `cache_metadata`, add non-semantic `extraction_seconds` to the stored metadata, and place limited-session smoke artifacts under `cache/micro15/smoke/` so they cannot overwrite production caches. Metadata validation compares all expected semantic keys and permits the recorded timing key.

- [ ] **Step 8: Create the thin CLI and test its parsers**

```python
# scripts/build_micro_features.py
def parse_args():
    parser = argparse.ArgumentParser(description="Build versioned 15-second ACC+GYRO feature caches.")
    parser.add_argument("--fold", choices=("0", "1", "2", "3", "4", "all"), default="all")
    parser.add_argument("--split", choices=("train", "meal_train", "no_meal_train", "val", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-gravity-align", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()
```

`main()` expands `all` deterministically, constructs one `MicroFeatureConfig`, and calls `build_micro_split`; it contains no feature or labeling logic.

- [ ] **Step 9: Run tests and commit the cache increment**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_imu_features.py tests/pipeline/test_micro_cache.py -q`

Expected: all tests pass.

Run: `git diff --check`

```bash
git add src/pipeline/micro_cache.py scripts/build_micro_features.py tests/pipeline/test_micro_cache.py
git commit -m "feat: build versioned micro feature caches"
```

---

### Task 3: Gap-safe micro candidates and inner-OOF threshold selection

**Files:**
- Modify: `src/pipeline/event_stack.py:230-470`
- Modify: `tests/pipeline/test_event_stack.py`

**Interfaces:**
- Consumes: `Mapping[str, Sequence[tuple[int, int, float]]]`, inner truths, and a registered threshold grid.
- Produces: `MicroCandidateConfig`, `MicroCandidateSelection`, `micro_candidates`, and `select_micro_candidate_threshold`.

- [ ] **Step 1: Write failing gap, smoothing, merge, and duration tests**

```python
from src.pipeline.event_stack import MicroCandidateConfig, micro_candidates


def test_micro_candidates_never_bridge_more_than_two_strides():
    rows = [(index * 7_500, index * 7_500 + 15_000, 0.9) for index in range(8)]
    rows += [(180_000 + index * 7_500, 195_000 + index * 7_500, 0.9) for index in range(8)]
    config = MicroCandidateConfig(smooth_sigma_ms=0, smooth_radius_ms=0, merge_ms=180_000, min_duration_ms=60_000)
    result = micro_candidates({"s1": rows}, threshold=0.5, config=config)
    assert [(item.event.start_ms, item.event.end_ms) for item in result] == [(0, 67_500), (180_000, 247_500)]


def test_micro_candidates_merge_runs_within_registered_gap():
    starts = list(range(0, 60_000, 7_500)) + list(range(120_000, 180_000, 7_500))
    rows = [(start, start + 15_000, 0.9) for start in starts]
    config = MicroCandidateConfig(smooth_sigma_ms=0, smooth_radius_ms=0, merge_ms=60_000, min_duration_ms=60_000)
    result = micro_candidates({"s1": rows}, threshold=0.5, config=config)
    assert [(item.event.start_ms, item.event.end_ms) for item in result] == [(0, 187_500)]
```

- [ ] **Step 2: Run the focused tests and verify missing symbols fail**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_event_stack.py -q`

Expected: import fails for `MicroCandidateConfig`.

- [ ] **Step 3: Implement deterministic segmented Gaussian smoothing and candidate construction**

```python
@dataclass(frozen=True)
class MicroCandidateConfig:
    stride_ms: int = 7_500
    window_ms: int = 15_000
    smooth_sigma_ms: int = 30_000
    smooth_radius_ms: int = 60_000
    merge_ms: int = 180_000
    min_duration_ms: int = 60_000
    context_ms: int = 1_200_000


def _gaussian_kernel(config):
    if config.smooth_sigma_ms == 0 or config.smooth_radius_ms == 0:
        return np.ones(1, dtype=np.float64)
    offsets = np.arange(-config.smooth_radius_ms, config.smooth_radius_ms + config.stride_ms, config.stride_ms)
    kernel = np.exp(-0.5 * (offsets / config.smooth_sigma_ms) ** 2)
    return kernel / kernel.sum()


def _normalized_smooth(values, kernel):
    if len(kernel) == 1:
        return np.asarray(values, dtype=np.float64)
    start = (len(kernel) - 1) // 2
    numerator = np.convolve(values, kernel, mode="full")[start:start + len(values)]
    denominator = np.convolve(np.ones(len(values)), kernel, mode="full")[start:start + len(values)]
    return numerator / np.maximum(denominator, 1e-12)
```

`micro_candidates` must split raw rows whenever consecutive starts differ by more than `2 * stride_ms`, call `_normalized_smooth` on each segment independently, turn thresholded runs into `[first_window_start, last_window_end)`, discard runs shorter than `min_duration_ms`, merge only retained runs within the same acquisition segment whose event gap is at most `merge_ms`, and populate `CandidateEvent.probabilities` with smoothed supporting scores. Validate finite probabilities, strictly positive durations, and configuration multiples of stride.

- [ ] **Step 4: Write failing threshold-budget and deterministic tie tests**

```python
from src.pipeline.event_stack import EventRef, select_micro_candidate_threshold


def test_micro_threshold_selection_maximizes_recall_with_candidate_budget():
    rows = {
        "s1": [(index * 7_500, index * 7_500 + 15_000, 0.35) for index in range(8)],
        "s2": [(index * 7_500, index * 7_500 + 15_000, 0.15) for index in range(8)],
    }
    truths = (EventRef("s1", 0, 67_500),)
    config = MicroCandidateConfig(smooth_sigma_ms=0, smooth_radius_ms=0)
    selected = select_micro_candidate_threshold(rows, truths, (0.1, 0.2, 0.3, 0.4), config)
    assert selected.threshold == 0.3
    assert selected.metrics.sensitivity == 1.0
    assert selected.candidate_count == 1
```

- [ ] **Step 5: Implement registered-grid selection without outer inputs**

```python
@dataclass(frozen=True)
class MicroCandidateSelection:
    threshold: float
    metrics: EventMetrics
    candidate_count: int


def select_micro_candidate_threshold(windows_by_sid, truths, thresholds, config=None):
    options = tuple(float(value) for value in thresholds)
    if not options or len(set(options)) != len(options) or any(not np.isfinite(value) or value < 0 or value > 1 for value in options):
        raise ValueError("micro thresholds must be unique finite values in [0, 1]")
    budget = 3 * max(len(truths), 1)
    feasible, fallback = [], []
    for threshold in options:
        candidates = micro_candidates(windows_by_sid, threshold, config)
        metrics = compute_event_metrics([item.event for item in candidates], truths)
        selection = MicroCandidateSelection(threshold, metrics, len(candidates))
        fallback.append(selection)
        if len(candidates) <= budget:
            feasible.append(selection)
    if feasible:
        return max(feasible, key=lambda item: (item.metrics.sensitivity, -item.candidate_count, item.threshold))
    return max(fallback, key=lambda item: (item.metrics.f1, item.metrics.ppv, item.threshold))
```

- [ ] **Step 6: Run event-stack tests and commit**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_event_stack.py -q`

Expected: existing macro and new micro tests pass.

```bash
git add src/pipeline/event_stack.py tests/pipeline/test_event_stack.py
git commit -m "feat: add gap-safe micro candidates"
```

---

### Task 4: Deterministic candidate union and exact 56-feature verifier surface

**Files:**
- Modify: `src/pipeline/event_stack.py:246-560`
- Modify: `tests/pipeline/test_event_stack.py`

**Interfaces:**
- Consumes: macro/micro `CandidateEvent` sequences and both window-score streams.
- Produces: `MultiScaleCandidate`, `union_candidates`, and `multiscale_verifier_features`.

- [ ] **Step 1: Write failing one-to-one union tests**

```python
from src.pipeline.event_stack import MultiScaleCandidate, union_candidates


def test_union_preserves_macro_geometry_and_keeps_micro_only_rescue():
    macro = [_candidate("s1", 100, 300)]
    micro = [_candidate("s1", 150, 250), _candidate("s1", 500, 620)]
    result = union_candidates(macro, micro, merge_iou=0.25)
    assert [item.event for item in result] == [EventRef("s1", 100, 300), EventRef("s1", 500, 620)]
    assert result[0].macro is macro[0] and result[0].micro is micro[0]
    assert result[1].macro is None and result[1].micro is micro[1]


def test_union_is_independent_of_input_order():
    macro = [_candidate("s1", 0, 100), _candidate("s1", 200, 300)]
    micro = [_candidate("s1", 10, 90), _candidate("s1", 210, 290)]
    assert union_candidates(macro, micro) == union_candidates(tuple(reversed(macro)), tuple(reversed(micro)))
```

- [ ] **Step 2: Implement deterministic one-to-one matching**

```python
@dataclass(frozen=True)
class MultiScaleCandidate:
    event: EventRef
    macro: CandidateEvent | None
    micro: CandidateEvent | None


def union_candidates(macro, micro, merge_iou=0.25):
    macro_sorted = sorted(macro, key=lambda item: (item.event.sid, item.event.start_ms, item.event.end_ms))
    micro_sorted = sorted(micro, key=lambda item: (item.event.sid, -max(item.probabilities, default=0.0), item.event.start_ms, item.event.end_ms))
    unmatched = set(range(len(macro_sorted)))
    result = []
    for micro_item in micro_sorted:
        matches = []
        for index in unmatched:
            macro_item = macro_sorted[index]
            if macro_item.event.sid != micro_item.event.sid:
                continue
            overlap = event_iou(macro_item.event.interval, micro_item.event.interval)
            if overlap >= merge_iou:
                matches.append((overlap, -macro_item.event.start_ms, -macro_item.event.end_ms, index))
        if matches:
            index = max(matches)[-1]
            unmatched.remove(index)
            macro_item = macro_sorted[index]
            result.append(MultiScaleCandidate(macro_item.event, macro_item, micro_item))
        else:
            result.append(MultiScaleCandidate(micro_item.event, None, micro_item))
    result.extend(MultiScaleCandidate(macro_sorted[index].event, macro_sorted[index], None) for index in sorted(unmatched))
    return sorted(result, key=lambda item: (item.event.sid, item.event.start_ms, item.event.end_ms))
```

- [ ] **Step 3: Write failing 56-feature and missing-stream tests**

```python
from src.pipeline.event_stack import multiscale_verifier_features


def test_multiscale_verifier_has_exact_width_and_distinct_source_missing_flags():
    candidate = MultiScaleCandidate(EventRef("s1", 0, 60_000), None, _candidate("s1", 0, 60_000))
    micro_windows = {"s1": [(index * 7_500, index * 7_500 + 15_000, 0.8) for index in range(8)]}
    result = multiscale_verifier_features([candidate], macro_windows_by_sid={}, micro_windows_by_sid=micro_windows)
    assert result.shape == (1, 56)
    assert np.isfinite(result).all()
    assert result[0, 37:39].tolist() == [0.0, 1.0]
    assert result[0, 54:56].tolist() == [1.0, 0.0]


def test_multiscale_verifier_keeps_macro_only_candidate_without_micro_samples():
    candidate = MultiScaleCandidate(EventRef("s1", 0, 240_000), _candidate("s1", 0, 240_000), None)
    macro_windows = {"s1": [(index * 15_000, index * 15_000 + 240_000, 0.7) for index in range(4)]}
    result = multiscale_verifier_features([candidate], macro_windows, {})
    assert result.shape == (1, 56)
    assert result[0, 37:39].tolist() == [1.0, 0.0]
    assert result[0, 54:56].tolist() == [0.0, 1.0]
```

- [ ] **Step 4: Implement score sampling and the fixed 19-feature micro block**

For each union geometry, sample a stream window when its center lies in `[event.start_ms, event.end_ms)`. Construct the macro 37-vector through `verifier_features` only when at least two macro samples exist; otherwise use 37 zeros. Compute the 20-minute context from the same session and do not synthesize rows across gaps.

```python
def _scores_for_event(event, windows_by_sid):
    return np.asarray([
        float(score)
        for start, end, score in windows_by_sid.get(event.sid, ())
        if event.start_ms <= (start + end) // 2 < event.end_ms
    ], dtype=np.float64)


def _context_scores(event, windows_by_sid, context_ms):
    return np.asarray([
        float(score)
        for start, end, score in windows_by_sid.get(event.sid, ())
        if event.start_ms - context_ms <= (start + end) // 2 < event.start_ms
        or event.end_ms <= (start + end) // 2 < event.end_ms + context_ms
    ], dtype=np.float64)


def multiscale_verifier_features(candidates, macro_windows_by_sid, micro_windows_by_sid):
    rows = []
    for candidate in candidates:
        event = candidate.event
        macro_values = _scores_for_event(event, macro_windows_by_sid)
        micro_values = _scores_for_event(event, micro_windows_by_sid)
        macro_missing = len(macro_values) < 2
        micro_missing = len(micro_values) < 2
        if macro_missing:
            macro_block = np.zeros(37, dtype=np.float64)
        else:
            synthetic = CandidateEvent(event, tuple(macro_values), 1.0, 0, 0, 0, 0)
            macro_block = verifier_features((synthetic,), macro_windows_by_sid, include_coverage=False)[0]
        micro_safe = micro_values if len(micro_values) else np.zeros(1)
        context = _context_scores(event, micro_windows_by_sid, 1_200_000)
        context_safe = context if len(context) else np.zeros(1)
        mean_difference = float(macro_values.mean() - micro_values.mean()) if not macro_missing and not micro_missing else 0.0
        max_difference = float(macro_values.max() - micro_values.max()) if not macro_missing and not micro_missing else 0.0
        micro_block = np.asarray([
            float(candidate.macro is not None), float(candidate.micro is not None),
            float(len(micro_values)), (event.end_ms - event.start_ms) / 1000.0,
            float(micro_safe.mean()), float(micro_safe.max()), float(micro_safe.std()),
            float(np.percentile(micro_safe, 10)), float(np.median(micro_safe)), float(np.percentile(micro_safe, 90)),
            float((micro_values >= 0.30).mean()) if len(micro_values) else 0.0,
            float((micro_values >= 0.50).mean()) if len(micro_values) else 0.0,
            float(_longest_above(micro_values, 0.30)),
            float(micro_safe.mean() - context_safe.mean()), float(micro_safe.max() - np.percentile(context_safe, 90)),
            mean_difference, max_difference, float(macro_missing), float(micro_missing),
        ])
        rows.append(np.concatenate((macro_block, micro_block)))
    result = np.asarray(rows, dtype=np.float64).reshape((-1, 56))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
```

- [ ] **Step 5: Run event-stack tests and commit**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_event_stack.py -q`

Expected: all legacy and multi-scale tests pass.

```bash
git add src/pipeline/event_stack.py tests/pipeline/test_event_stack.py
git commit -m "feat: fuse macro and micro event evidence"
```

---

### Task 5: Runner configuration, data-source, and result contracts

**Files:**
- Modify: `src/pipeline/runner.py:50-140`
- Modify: `src/pipeline/runner.py:545-735`
- Modify: `src/pipeline/runner.py:735-842`
- Modify: `tests/pipeline/test_runner.py`

**Interfaces:**
- Consumes: production micro NPZ caches whose metadata matches `MicroFeatureConfig`.
- Produces: micro-aware `RunConfig`, optional micro batches on `FoldDataset`, versioned `FoldResult`, and `_micro_window_estimator`.

- [ ] **Step 1: Write failing config-hash and LightGBM-contract tests**

```python
from src.pipeline.event_stack import MicroCandidateConfig
from src.pipeline.runner import _micro_window_estimator


def test_cache_key_changes_for_every_micro_behavior_setting():
    base = RunConfig(outer_fold=0)
    assert cache_key(base) != cache_key(replace(base, micro_enabled=True))
    assert cache_key(base) != cache_key(replace(base, micro_gravity_align=False))
    assert cache_key(base) != cache_key(replace(base, micro_threshold_grid=(0.2, 0.4)))
    assert cache_key(base) != cache_key(replace(base, micro_candidate=MicroCandidateConfig(merge_ms=120_000)))
    assert cache_key(base) != cache_key(replace(base, micro_positive_middle_fraction=0.6))


def test_micro_lightgbm_is_single_threaded_and_seeded():
    estimator = _micro_window_estimator(20260909)
    assert estimator.get_params()["model__n_jobs"] == 1
    assert estimator.get_params()["model__random_state"] == 20260909
```

- [ ] **Step 2: Add the exact configuration and estimator**

```python
MICRO_WINDOW_MODEL_PARAMETERS = {
    "n_estimators": 300, "num_leaves": 31, "min_child_samples": 100,
    "learning_rate": 0.05, "colsample_bytree": 0.8, "reg_lambda": 5.0,
    "class_weight": "balanced", "n_jobs": 1, "verbosity": -1,
}
RUNNER_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class RunConfig:
    outer_fold: int
    inner_splits: int = 4
    seed: int = 20260908
    no_tcn: bool = True
    workers: int = 1
    device: str = "auto"
    subject_cap_grid: tuple[int, ...] = ()
    verifier_feature_mode: str = "probability"
    verifier_c_grid: tuple[float, ...] = (0.1,)
    density: DensityConfig = field(default_factory=DensityConfig)
    micro_enabled: bool = False
    micro_gravity_align: bool = True
    micro_threshold_grid: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
    micro_candidate: MicroCandidateConfig = field(default_factory=MicroCandidateConfig)
    micro_positive_middle_fraction: float | None = None
    external_fd_weight_grid: tuple[float, ...] = (0.0,)


def _micro_window_estimator(seed):
    from lightgbm import LGBMClassifier
    parameters = dict(MICRO_WINDOW_MODEL_PARAMETERS)
    parameters["random_state"] = seed
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LGBMClassifier(**parameters))])
```

- [ ] **Step 3: Write failing optional-batch and metadata-loading tests**

```python
def test_macro_fold_dataset_remains_constructible_without_micro_batches():
    dataset = synthetic_runner_dataset()
    assert dataset.micro_window_train is None
    assert dataset.micro_candidate_train is None
    assert dataset.micro_validation is None


def test_filesystem_source_rejects_micro_metadata_mismatch(tmp_path, monkeypatch):
    source = FilesystemDataSource(tmp_path)
    path = tmp_path / "cache" / "micro15" / "fold0_train.npz"
    path.parent.mkdir(parents=True)
    arrays = MicroCacheArrays(np.ones((1, 47), np.float32), np.array([0], np.int8), np.array(['["s1", 0, 15000]']))
    write_micro_cache_atomic(path, arrays, {"wrong": True})
    with pytest.raises(ValueError, match="metadata mismatch"):
        source._load_micro_batch(path, {"expected": True})
```

- [ ] **Step 4: Extend dataset loading without changing macro-only call sites**

Add these defaulted fields after `validation_truth_slices`:

```python
micro_window_train: WindowBatch | None = None
micro_candidate_train: WindowBatch | None = None
micro_validation: WindowBatch | None = None
micro_cache_extraction_seconds: float = 0.0
```

Add `micro_dir = root / "cache" / "micro15"`, `_micro_split_path`, and `_load_micro_batch(path, expected_metadata) -> tuple[WindowBatch, float]`; the float is `read_micro_metadata(path)["extraction_seconds"]`. Change `input_files` and `load_outer_fold` to accept `config: RunConfig`; include and load micro files only when `config.micro_enabled`. Combine `meal_train` and `no_meal_train` for `micro_candidate_train`. Derive each split's expected metadata from `MicroFeatureConfig(gravity_align=config.micro_gravity_align)` plus the exact session/manifests used by the builder, and reject a mismatch. Sum the four stored `extraction_seconds` values into `FoldDataset.micro_cache_extraction_seconds` without treating timing as cache identity.

- [ ] **Step 5: Add all result fields and backward-compatible JSON defaults**

```python
@dataclass(frozen=True)
class FoldResult:
    config_hash: str
    threshold: float
    max_events_per_subject: int | None
    verifier_c: float
    verifier_feature_count: int
    inner_metrics: EventMetrics
    outer_metrics: EventMetrics
    candidate_count: int
    candidate_match_recall: float
    slices: Mapping[str, EventMetrics]
    timings_seconds: Mapping[str, float]
    cache_hits: Mapping[str, bool]
    outer_subjects: frozenset[str]
    window_fit_subjects: frozenset[str]
    verifier_fit_subjects: frozenset[str]
    micro_threshold: float | None = None
    micro_candidate_count: int = 0
    micro_candidate_match_recall: float = 0.0
    short_meal_candidate_recall: float = 0.0
    external_weight: float = 0.0
    macro_window_feature_count: int = 0
    micro_window_feature_count: int = 0
    micro_window_fit_subjects: frozenset[str] = field(default_factory=frozenset)
```

Serialize every field in `fold_result_to_dict`; `_fold_result_from_dict` uses the displayed defaults (`macro_window_feature_count=63` for historical macro results) for schema-3 result inspection. Schema-3 paths are never reused as schema-4 prediction caches. Multi-scale `timings_seconds` contains exact keys `feature_extraction`, `macro_window_oof`, `micro_window_oof`, `verifier_oof`, `final_fit`, `outer_inference`, and `total`; direct in-memory datasets report `feature_extraction=0.0`.

- [ ] **Step 6: Run runner tests and commit**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_runner.py -q`

Expected: macro tests and new schema tests pass.

```bash
git add src/pipeline/runner.py tests/pipeline/test_runner.py
git commit -m "feat: add micro runner data contracts"
```

---

### Task 6: Nested multi-scale training and untouched outer evaluation

**Files:**
- Modify: `src/pipeline/runner.py:300-545`
- Create: `tests/pipeline/test_multiscale_runner.py`

**Interfaces:**
- Consumes: the Task 5 `FoldDataset` contract and Task 3/4 event functions.
- Produces: one outer-fold `FoldResult` with micro diagnostics and disjoint macro/micro/verifier fit-subject sets.

- [ ] **Step 1: Create a synthetic multi-scale dataset helper and failing isolation test**

```python
# tests/pipeline/test_multiscale_runner.py
from dataclasses import replace

import numpy as np

from tests.pipeline.test_runner import synthetic_runner_dataset
from src.pipeline.runner import RunConfig, WindowBatch, run_outer_fold


def multiscale_dataset():
    base = synthetic_runner_dataset(subjects=8, windows_per_subject=40)
    def micro(batch, truths):
        features, labels, windows = [], [], []
        for sid in sorted({window.sid for window in batch.windows}):
            sid_windows = [window for window in batch.windows if window.sid == sid]
            first = min(window.start_ms for window in sid_windows)
            last = max(window.end_ms for window in sid_windows)
            for start in range(first, last - 15_000 + 1, 7_500):
                end = start + 15_000
                overlap = max(
                    (min(end, truth.end_ms) - max(start, truth.start_ms) for truth in truths if truth.sid == sid),
                    default=0,
                )
                label = int(overlap >= 7_500)
                features.append([float(label)] * 47)
                labels.append(label)
                windows.append(type(sid_windows[0])(sid, start, end))
        return WindowBatch(np.asarray(features, np.float32), np.asarray(labels, np.int8), tuple(windows))
    return replace(
        base,
        micro_window_train=micro(base.window_train, base.train_truths),
        micro_candidate_train=micro(base.candidate_train, base.train_truths),
        micro_validation=micro(base.validation, base.validation_truths),
    )


def test_multiscale_outer_subjects_never_enter_any_fit_set():
    result = run_outer_fold(RunConfig(outer_fold=0, inner_splits=3, micro_enabled=True), multiscale_dataset())
    assert result.outer_subjects.isdisjoint(result.window_fit_subjects)
    assert result.outer_subjects.isdisjoint(result.micro_window_fit_subjects)
    assert result.outer_subjects.isdisjoint(result.verifier_fit_subjects)
    assert result.micro_threshold in (0.10, 0.20, 0.30, 0.40, 0.50)
    assert result.macro_window_feature_count == 63
    assert result.micro_window_feature_count == 47
    assert result.verifier_feature_count == 56
```

- [ ] **Step 2: Add micro batch validation and positive-purity filtering**

```python
def _micro_training_keep(batch, truths, middle_fraction):
    keep = np.asarray(batch.labels) >= 0
    if middle_fraction is None:
        return keep
    if not 0 < middle_fraction <= 1:
        raise ValueError("micro_positive_middle_fraction must be in (0, 1]")
    for index in np.flatnonzero(np.asarray(batch.labels) == 1):
        window = batch.windows[index]
        center = (window.start_ms + window.end_ms) // 2
        inside_middle = False
        for truth in truths:
            if truth.sid != window.sid:
                continue
            margin = (truth.end_ms - truth.start_ms) * (1.0 - middle_fraction) / 2.0
            inside_middle |= truth.start_ms + margin <= center <= truth.end_ms - margin
        if not inside_middle:
            keep[index] = False
    return keep
```

When `micro_enabled=True`, require all three micro batches, validate 47 columns, map their sessions to subjects, and call `validate_outer_isolation` for both micro training batches before fitting. Reject `external_fd_weight_grid != (0.0,)` in this target-domain plan so a nonzero external weight cannot be silently ignored.

- [ ] **Step 3: Add subject-disjoint micro OOF scoring and train-only threshold selection**

In `_run_outer_dataset`, keep the current macro OOF path intact. Add a sibling micro path:

```python
micro_keep = _micro_training_keep(data_source.micro_window_train, data_source.train_truths, config.micro_positive_middle_fraction)
micro_train_groups = _groups_for(data_source.micro_window_train.windows, data_source.subject_by_session)[micro_keep]
micro_candidate_groups = _groups_for(data_source.micro_candidate_train.windows, data_source.subject_by_session)
micro_oof = crossfit_predict_proba(
    data_source.micro_window_train.features[micro_keep],
    data_source.micro_window_train.labels[micro_keep].astype(np.int8),
    micro_train_groups,
    data_source.micro_candidate_train.features,
    micro_candidate_groups,
    config.inner_splits,
    estimator_factory=lambda: _micro_window_estimator(config.seed + 10),
)
micro_oof_windows = _windows_by_session(data_source.micro_candidate_train.windows, micro_oof.probabilities)
micro_selection = select_micro_candidate_threshold(
    micro_oof_windows, data_source.train_truths, config.micro_threshold_grid, config.micro_candidate
)
```

Generate macro OOF candidates exactly as before, generate micro OOF candidates with `micro_selection.threshold`, call `union_candidates`, and build 56 training features with `multiscale_verifier_features`. Candidate labels and groups use each `MultiScaleCandidate.event`.

- [ ] **Step 4: Fit final micro model and score outer-validation windows once**

```python
micro_model = _micro_window_estimator(config.seed + 12)
micro_model.fit(data_source.micro_window_train.features[micro_keep], data_source.micro_window_train.labels[micro_keep])
micro_validation_scores = _positive_probability(micro_model, data_source.micro_validation.features)
micro_validation_windows = _windows_by_session(data_source.micro_validation.windows, micro_validation_scores)
micro_validation_candidates = micro_candidates(micro_validation_windows, micro_selection.threshold, config.micro_candidate)
validation_union = union_candidates(raw_validation_candidates, micro_validation_candidates)
validation_candidate_features = multiscale_verifier_features(validation_union, validation_windows, micro_validation_windows)
```

Fit the verifier on the 56-column outer-train candidate matrix, use the already selected inner policy on outer validation, and report union candidate recall plus raw micro candidate recall. Compute `short_meal_candidate_recall` by matching union candidates against `validation_truth_slices.get("duration_lt10", ())`.

- [ ] **Step 5: Write a failing outer-label-independence test**

```python
def test_outer_truth_changes_do_not_change_any_selected_setting():
    dataset = multiscale_dataset()
    config = RunConfig(outer_fold=0, inner_splits=3, micro_enabled=True)
    original = run_outer_fold(config, dataset)
    changed = run_outer_fold(config, replace(dataset, validation_truths=(), validation_truth_slices={}))
    assert changed.micro_threshold == original.micro_threshold
    assert changed.verifier_c == original.verifier_c
    assert changed.threshold == original.threshold
    assert changed.max_events_per_subject == original.max_events_per_subject
```

- [ ] **Step 6: Ensure macro-only execution takes the original branch**

Guard every micro operation behind `config.micro_enabled`. The existing `_candidate_matrix`, 37/42-column probability verifier, raw-summary option, estimators, policy selection, and candidate geometry remain the only executed path when the flag is false.

- [ ] **Step 7: Run focused and full pipeline tests**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_multiscale_runner.py -q`

Expected: all multi-scale integration tests pass.

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline -q`

Expected: all existing and new tests pass; no macro-only regression.

- [ ] **Step 8: Commit the nested runner**

```bash
git add src/pipeline/runner.py tests/pipeline/test_multiscale_runner.py
git commit -m "feat: run nested multiscale event stack"
```

---

### Task 7: Registered CLI, compact aggregation, and cache compatibility

**Files:**
- Modify: `scripts/crossfit_event_stack.py:20-160`
- Modify: `src/pipeline/runner.py:735-880`
- Modify: `tests/pipeline/test_runner.py`

**Interfaces:**
- Consumes: all Task 5/6 `RunConfig` and `FoldResult` fields.
- Produces: strict CLI parsers, per-fold JSON, and one compact five-fold summary JSON.

- [ ] **Step 1: Write failing parser tests**

```python
from scripts.crossfit_event_stack import parse_probability_grid, parse_middle_fraction


def test_probability_grid_parser_is_strict_and_deterministic():
    assert parse_probability_grid("0.1,0.3,0.5") == (0.1, 0.3, 0.5)
    with pytest.raises(ValueError, match="unique finite probabilities"):
        parse_probability_grid("0.1,nan,0.1")
    with pytest.raises(ValueError, match="unique finite probabilities"):
        parse_probability_grid("-0.1,0.5")


def test_middle_fraction_parser_accepts_none_or_unit_interval():
    assert parse_middle_fraction("none") is None
    assert parse_middle_fraction("0.6") == 0.6
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        parse_middle_fraction("0")
```

- [ ] **Step 2: Add explicit CLI options and construct the full config**

```python
parser.add_argument("--micro-enabled", action="store_true")
parser.add_argument("--micro-threshold-grid", type=parse_probability_grid, default=(0.10, 0.20, 0.30, 0.40, 0.50))
parser.add_argument("--micro-gravity-align", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--micro-positive-middle-fraction", type=parse_middle_fraction, default=None)
```

Pass these values into every `RunConfig`. Reject `--verifier-features raw_summary` together with `--micro-enabled` until a separately designed raw multi-scale surface exists. Keep `--device cuda` rejected for this CPU-first tree milestone.

- [ ] **Step 3: Write failing aggregate-count tests**

```python
from dataclasses import replace

from src.pipeline.event_stack import EventMetrics
from src.pipeline.runner import aggregate_fold_results


def test_aggregate_fold_results_sums_counts_before_recomputing_ratios():
    config = RunConfig(outer_fold=0, inner_splits=3)
    base = run_outer_fold(config, synthetic_runner_dataset())
    configs = (config, replace(config, outer_fold=1))
    results = (
        replace(base, outer_metrics=EventMetrics(1, 2, 4, 0.25, 0.5, 1 / 3)),
        replace(base, outer_metrics=EventMetrics(2, 5, 3, 2 / 3, 0.4, 0.5)),
    )
    summary = aggregate_fold_results(configs, results)
    assert summary["outer_metrics"]["n_tp"] == 3
    assert summary["outer_metrics"]["n_pred"] == 7
    assert summary["outer_metrics"]["n_true"] == 7
    assert summary["outer_metrics"]["f1"] == pytest.approx(3 / 7)
```

- [ ] **Step 4: Implement compact aggregate JSON without an experiment helper script**

`aggregate_fold_results(configs, results)` must require equal nonzero lengths, reject duplicate `RunConfig.outer_fold` values, sum `n_tp`, `n_pred`, and `n_true`, recompute ratios from sums, total candidate counts, compute truth-weighted candidate recalls, sum timings, and include each fold's config hash and selected micro threshold.

When `--fold all`, `scripts/crossfit_event_stack.py` writes:

```python
summary = aggregate_fold_results(configs, results)
summary["run_configs"] = [asdict(config) for config in configs]
summary_path = output_directory / f"summary_{experiment_key(configs)}.json"
write_json_atomic(summary_path, summary)
```

`experiment_key` hashes the sorted configs after replacing `outer_fold` with `-1`; it never hashes result metrics.

- [ ] **Step 5: Verify cache inputs and serialization**

Change `run_outer_fold` to call `source.input_files(config)` and `source.load_outer_fold(config)`. The cache feature dimensions are `(63, 56, 47)` for multi-scale probability mode and remain `(63, 37)` or `(63, 42)` for the macro-only probability paths. Confirm every micro result field survives `fold_result_to_dict` followed by `_fold_result_from_dict`.

- [ ] **Step 6: Run CLI and pipeline tests, then commit**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_runner.py tests/pipeline/test_multiscale_runner.py -q`

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline -q`

Expected: all tests pass.

```bash
git add scripts/crossfit_event_stack.py src/pipeline/runner.py tests/pipeline/test_runner.py
git commit -m "feat: expose multiscale crossfit evaluation"
```

---

### Task 8: Smoke test, locked development run, decision record, and cleanup

**Files:**
- Modify: `README.md:159-190`
- Modify: `README.md:280-315`
- Modify: `docs/三阶段重构设计.md`
- Retain: `outputs/crossfit/*.json` as ignored compact evidence
- Remove after smoke: `cache/micro15/smoke/`

**Interfaces:**
- Consumes: the finished target-domain multi-scale implementation and real session caches.
- Produces: verified cache/runtime evidence, fold metrics, a gate decision, documentation, and a clean repository.

- [ ] **Step 1: Verify the clean implementation baseline before generating data**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline -q`

Expected: all tests pass.

Run: `git status --short`

Expected: no tracked or untracked entries.

- [ ] **Step 2: Run a limited-session cache smoke test**

Run: `D:/Anaconda3/envs/bme/python.exe scripts/build_micro_features.py --fold 0 --split train --limit 2 --workers 2 --force`

Expected: a finite `(n, 47)` artifact under `cache/micro15/smoke/`, with positive or negative labels as available, and no production cache overwritten.

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_micro_cache.py -q`

Expected: all cache tests still pass after reading real source shapes.

- [ ] **Step 3: Delete the smoke cache immediately**

Resolve and print `D:\BMEtest\cache\micro15\smoke`, confirm it is inside `D:\BMEtest\cache\micro15`, then remove only that directory with PowerShell `Remove-Item -LiteralPath ... -Recurse -Force`. Confirm `Test-Path` returns `False`.

- [ ] **Step 4: Build all production target-domain micro caches with bounded CPU parallelism**

Run: `D:/Anaconda3/envs/bme/python.exe scripts/build_micro_features.py --fold all --split all --workers 8`

Expected: 20 versioned NPZ artifacts under `cache/micro15/`; each has 47 columns and matching source/config metadata. Record total extraction time and row counts, but do not add caches to Git.

- [ ] **Step 5: Run fold 0 as a runtime and geometry diagnostic**

Run: `D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold 0 --inner-splits 4 --no-tcn --workers 1 --micro-enabled --force`

Expected: the run completes without outer-subject overlap, selects a threshold from `(0.10, 0.20, 0.30, 0.40, 0.50)`, reports 47 micro and 56 verifier features, and writes one compact fold JSON. Inspect candidate count, candidate recall, short-meal candidate recall, PPV, and stage timings. A crash, non-finite feature, or impractical candidate explosion returns execution to the failing task instead of changing thresholds on fold-0 labels.

- [ ] **Step 6: Run the registered five-fold target-domain configuration**

Run: `D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 0 --micro-enabled --force`

Expected: five fold JSON files plus one aggregate summary. Compare aggregate counts against the macro-only baseline `TP=89`, `true=153`, `pred=275`, `F1=0.415888`, candidate recall `121/153=0.791`, and short final recall `16/39=0.410`.

- [ ] **Step 7: Apply the pre-registered acceptance gate**

Accept the branch as the next development default only if all conditions hold:

```text
aggregate F1 >= 0.435888
short-meal final recall >= 0.510
aggregate PPV >= 0.294
candidate volume and total runtime remain practical for five-fold iteration
```

The final line is exact: aggregate union candidates must be at most `4 × 153 = 612`, and the
five-fold crossfit wall clock excluding one-time feature extraction must be at most 600 seconds
on the current machine.

If rejected, retain only the tested reusable modules and compact JSON evidence; do not enable `micro_enabled` by default. If accepted, enable it only in the documented recommended command—the dataclass default remains `False` for backward compatibility. In either case, do not start FD transfer in this plan.

- [ ] **Step 8: Update README and architecture documentation with exact evidence**

Add a dated table containing per-fold config hash, selected micro threshold, TP/true/pred, F1, union candidate recall, micro-only candidate recall, short-meal recall, and runtime. State clearly whether the gate passed, how far aggregate F1 remains from 0.65, and that repeated outer-CV comparisons are development evidence rather than a final untouched score.

Update reproduction commands to include:

```bash
D:/Anaconda3/envs/bme/python.exe scripts/build_micro_features.py --fold all --split all --workers 8
D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 0 --micro-enabled
```

Document that FD-I/FD-II is permitted but deferred to a separate gated plan and that its CC BY-NC-ND terms still require compliance.

- [ ] **Step 9: Run final verification and remove generated debris**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline -q`

Run: `git diff --check`

Remove only generated `__pycache__` directories and the verified `cache/micro15/smoke/` path if recreated. Do not remove production micro caches or compact crossfit evidence. Confirm no one-off `.py`, `.ps1`, log, or temporary NPZ files were created.

- [ ] **Step 10: Commit the evidence and leave the worktree clean**

```bash
git add README.md docs/三阶段重构设计.md
git commit -m "docs: record multiscale imu ablation"
git status --short
```

Expected: `git status --short` prints no entries. The ignored ACL-protected `.pytest_cache` warning may remain, but it must not hide any reported path.
