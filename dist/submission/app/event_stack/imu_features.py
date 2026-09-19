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
