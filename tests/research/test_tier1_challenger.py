import numpy as np

from scripts.research.tier1_challenger import (
    adaptive_meta_features,
    classify_inner_gate,
    gyro5_feature_block,
)


def test_gyro5_feature_block_has_frozen_six_column_schema():
    t = np.arange(200) / 10.0
    gyro = np.vstack([np.sin(t), np.cos(t), np.sin(2 * t)])
    block = gyro5_feature_block(gyro, sample_rate=10.0)
    assert block.shape == (6,)
    assert np.isfinite(block).all()
    assert block[0] > 0


def test_adaptive_meta_features_are_low_capacity_and_finite():
    values = adaptive_meta_features(
        np.array([0.8, 0.2]), np.array([0.5, 0.4]),
        duration_seconds=np.array([60.0, 120.0]),
        has_macro=np.array([1.0, 1.0]), has_micro=np.array([1.0, 0.0]),
        macro_micro_ratio=np.array([2.0, 0.5]),
    )
    assert values.shape == (2, 8)
    assert np.allclose(values[:, 2], [0.3, -0.2])
    assert np.isfinite(values).all()


def test_inner_gate_boundaries_are_registered():
    assert classify_inner_gate(0.7719, majority_improved=True) == "FAILED"
    assert classify_inner_gate(0.7750, majority_improved=True) == "REVIEW_REQUIRED"
    assert classify_inner_gate(0.7810, majority_improved=True) == "STRONG_PASS"
    assert classify_inner_gate(0.7900, majority_improved=False) == "FAILED"
