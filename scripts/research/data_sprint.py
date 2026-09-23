"""Low-cost, read-only signal characterisation for the challenger data sprint."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

SCALES_SECONDS = (5, 10, 15, 30, 60)
ANALYSIS_RATE_HZ = 10.0
SCHEMA_VERSION = "data-sprint-v1"


def prepare_output_dir(path: Path) -> Path:
    """Create and return one analysis output directory before any file write."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_truth_level_fn(rows: pd.DataFrame) -> pd.DataFrame:
    """Choose one deterministic, highest-scoring witness for each reachable FN."""
    required = {"truth_key", "blend_score", "candidate_key"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"reachable FN table is missing {sorted(missing)}")
    ordered = rows.sort_values(
        ["truth_key", "blend_score", "candidate_key"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ordered.drop_duplicates("truth_key", keep="first").reset_index(drop=True)


def disagreement_counts(rows: pd.DataFrame, threshold: float = 0.20) -> dict[str, int]:
    delta = rows["logistic_score"].to_numpy(float) - rows["lgbm_score"].to_numpy(float)
    lr = int(np.sum(delta >= threshold))
    lgbm = int(np.sum(delta <= -threshold))
    return {
        "lr_minus_lgbm_ge_0_20": lr,
        "lgbm_minus_lr_ge_0_20": lgbm,
        "neither": int(len(rows) - lr - lgbm),
        "population_size": int(len(rows)),
    }


def disagreement_fold_support(by_fold: dict) -> dict[str, int | bool]:
    """Count folds where reachable FN has a larger LR-high share than final FP."""
    count = 0
    comparable = 0
    for fold in sorted(set(by_fold["reachable_fn"]) & set(by_fold["final_fp"])):
        fn = by_fold["reachable_fn"][fold]
        fp = by_fold["final_fp"][fold]
        if fn["population_size"] and fp["population_size"]:
            comparable += 1
            fn_share = fn["lr_minus_lgbm_ge_0_20"] / fn["population_size"]
            fp_share = fp["lr_minus_lgbm_ge_0_20"] / fp["population_size"]
            count += int(fn_share > fp_share)
    return {"fn_lr_excess_fold_count": count, "comparable_fold_count": comparable, "stable": bool(comparable >= 4 and count >= 4)}


def expected_window_coverage(observed_samples: int, sample_rate: float, seconds: float) -> float:
    expected = max(1.0, sample_rate * seconds)
    return float(min(1.0, observed_samples / expected))


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _finite_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    indices = np.arange(values.size)
    return np.interp(indices, indices[finite], values[finite])


def temporal_metrics(acc: np.ndarray, gyro: np.ndarray, sample_rate: float) -> dict[str, float]:
    """Compute the fixed cheap proxy set on one uniformly sampled window."""
    if acc.shape[0] != 3 or gyro.shape != acc.shape or acc.shape[1] < 4:
        return {name: float("nan") for name in (
            "activity_magnitude", "acc_variance", "gyro_variance", "short_term_energy",
            "sign_change_rate", "peak_count", "burst_count", "active_fraction", "local_contrast",
        )}
    acc_mag = _finite_signal(np.linalg.norm(acc, axis=0))
    gyro_mag = _finite_signal(np.linalg.norm(gyro, axis=0))
    acc_delta = np.diff(acc_mag, prepend=acc_mag[0])
    gyro_delta = np.diff(gyro_mag, prepend=gyro_mag[0])
    activity = np.sqrt(acc_delta ** 2 + gyro_delta ** 2)
    med = float(np.median(activity))
    mad = robust_mad(activity)
    threshold = med + 2.0 * max(mad, np.finfo(float).eps)
    active = activity > threshold
    changes = np.diff(active.astype(np.int8), prepend=0)
    peak_distance = max(1, int(round(sample_rate * 0.30)))
    prominence = max(2.0 * mad, np.finfo(float).eps)
    peaks, _ = find_peaks(activity, distance=peak_distance, prominence=prominence)
    signed = gyro_mag - np.median(gyro_mag)
    sign_change = np.mean((signed[1:] * signed[:-1]) < 0) if len(signed) > 1 else 0.0
    q1, q3 = len(activity) // 4, (3 * len(activity)) // 4
    centre = np.mean(activity[q1:q3]) if q3 > q1 else np.mean(activity)
    edge = np.mean(np.r_[activity[:q1], activity[q3:]]) if q1 else np.mean(activity)
    return {
        "activity_magnitude": float(np.mean(activity)),
        "acc_variance": float(np.var(acc_mag)),
        "gyro_variance": float(np.var(gyro_mag)),
        "short_term_energy": float(np.mean(acc_delta ** 2 + gyro_delta ** 2)),
        "sign_change_rate": float(sign_change),
        "peak_count": int(len(peaks)),
        "burst_count": int(np.sum(changes == 1)),
        "active_fraction": float(np.mean(active)),
        "local_contrast": float((centre + 1e-12) / (edge + 1e-12)),
    }


def periodicity_metrics(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
    signal = _finite_signal(np.asarray(signal, dtype=float).reshape(-1))
    empty = {
        "autocorrelation_first_peak_s": float("nan"),
        "autocorrelation_peak_strength": float("nan"),
        "dominant_frequency_hz": float("nan"),
        "spectral_entropy": float("nan"),
        "burst_interval_mean_s": float("nan"),
        "burst_interval_variance_s2": float("nan"),
        "burst_count": 0,
        "repetition_regularity": float("nan"),
        "peak_interval_mean_s": float("nan"),
        "peak_interval_variance_s2": float("nan"),
    }
    if len(signal) < max(16, int(sample_rate * 3)):
        return empty
    centered = signal - np.mean(signal)
    scale = np.std(centered)
    if scale <= np.finfo(float).eps:
        return empty
    centered /= scale
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    corr /= np.maximum(np.arange(len(centered), 0, -1), 1)
    corr /= max(corr[0], np.finfo(float).eps)
    lo, hi = max(1, int(0.4 * sample_rate)), min(len(corr), int(10 * sample_rate) + 1)
    ac_peaks, _ = find_peaks(corr[lo:hi])
    first_lag = lo + int(ac_peaks[0]) if len(ac_peaks) else None
    strongest_lag = lo + int(np.argmax(corr[lo:hi])) if hi > lo else None
    fft = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), 1.0 / sample_rate)
    band = (frequencies >= 0.1) & (frequencies <= 2.0)
    band_power = fft[band]
    band_freq = frequencies[band]
    total = float(np.sum(band_power))
    probability = band_power / total if total > 0 else np.array([])
    entropy = -np.sum(probability * np.log(probability + 1e-12)) / np.log(max(len(probability), 2)) if len(probability) else np.nan
    prominence = max(robust_mad(signal) * 2.0, np.finfo(float).eps)
    peaks, _ = find_peaks(signal, distance=max(1, int(0.3 * sample_rate)), prominence=prominence)
    intervals = np.diff(peaks) / sample_rate
    mean_interval = float(np.mean(intervals)) if len(intervals) else np.nan
    interval_var = float(np.var(intervals)) if len(intervals) else np.nan
    regularity = float(1.0 / (1.0 + np.std(intervals) / (mean_interval + 1e-12))) if len(intervals) else np.nan
    return {
        "autocorrelation_first_peak_s": float(first_lag / sample_rate) if first_lag is not None else np.nan,
        "autocorrelation_peak_strength": float(corr[strongest_lag]) if strongest_lag is not None else np.nan,
        "dominant_frequency_hz": float(band_freq[np.argmax(band_power)]) if total > 0 else np.nan,
        "spectral_entropy": float(entropy),
        "burst_interval_mean_s": mean_interval,
        "burst_interval_variance_s2": interval_var,
        "burst_count": int(len(peaks)),
        "repetition_regularity": regularity,
        "peak_interval_mean_s": mean_interval,
        "peak_interval_variance_s2": interval_var,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_sha(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
