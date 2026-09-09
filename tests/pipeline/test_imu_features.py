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
