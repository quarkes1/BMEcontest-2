"""Pure feature and gate contracts for the E05/E06 inner-only challengers."""

from __future__ import annotations

import numpy as np

GYRO5_FEATURE_NAMES = (
    "gyro5_magnitude_variance",
    "gyro5_magnitude_rms",
    "gyro5_delta_energy",
    "gyro5_magnitude_max",
    "gyro5_active_fraction",
    "gyro5_to_gyro15_variance_ratio",
)

META_FEATURE_NAMES = (
    "p_lr", "p_lgbm", "lr_minus_lgbm", "abs_lr_minus_lgbm",
    "duration_seconds", "has_macro", "has_micro", "macro_micro_ratio",
)


def gyro5_feature_block(gyro_15s: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return the preregistered six-feature block from a centred 15 s window."""
    values = np.asarray(gyro_15s, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < 4 or sample_rate <= 0:
        return np.full(len(GYRO5_FEATURE_NAMES), np.nan, dtype=np.float64)
    magnitude_15 = np.linalg.norm(values, axis=0)
    width = min(len(magnitude_15), max(4, int(round(5.0 * sample_rate))))
    offset = (len(magnitude_15) - width) // 2
    magnitude = magnitude_15[offset : offset + width]
    variance = float(np.var(magnitude))
    median = float(np.median(magnitude))
    mad = float(np.median(np.abs(magnitude - median)))
    active = magnitude > median + 2.0 * max(mad, np.finfo(float).eps)
    delta = np.diff(magnitude)
    return np.asarray([
        variance,
        float(np.sqrt(np.mean(magnitude ** 2))),
        float(np.mean(delta ** 2)) if len(delta) else 0.0,
        float(np.max(magnitude)),
        float(np.mean(active)),
        variance / (float(np.var(magnitude_15)) + 1e-9),
    ])


def adaptive_meta_features(
    p_lr: np.ndarray,
    p_lgbm: np.ndarray,
    *,
    duration_seconds: np.ndarray,
    has_macro: np.ndarray,
    has_micro: np.ndarray,
    macro_micro_ratio: np.ndarray,
) -> np.ndarray:
    lr = np.asarray(p_lr, dtype=np.float64)
    lgbm = np.asarray(p_lgbm, dtype=np.float64)
    columns = [lr, lgbm, lr - lgbm, np.abs(lr - lgbm), duration_seconds, has_macro, has_micro, macro_micro_ratio]
    matrix = np.column_stack(columns).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("adaptive meta features must be finite")
    return matrix


def classify_inner_gate(f1: float, *, majority_improved: bool) -> str:
    if not majority_improved or f1 < 0.772:
        return "FAILED"
    if f1 < 0.780:
        return "REVIEW_REQUIRED"
    return "STRONG_PASS"
