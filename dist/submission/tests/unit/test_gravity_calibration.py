import numpy as np

from src.pipeline.inference.local_server import estimate_gravity_calibration
from src.pipeline.io.raw_session import SessionData


def _session(values):
    n = len(values)
    return SessionData(
        acc=np.array([values, np.zeros(n), np.zeros(n)], dtype=np.float32),
        gyro=np.zeros((3, n), dtype=np.float32), ppg=np.zeros((44, n), dtype=np.float32),
        t_acc=np.arange(n, dtype=np.int64) * 10 + 1, t_ppg=np.full(n, -1),
        imu_valid=np.ones(n, dtype=bool), ppg_valid=np.zeros(n, dtype=bool),
    )


def test_three_quiet_intervals_estimate_counts_per_g():
    values = np.full(4300, 16384.0)
    values[1000:1100] = np.linspace(1000, 32000, 100)
    result = estimate_gravity_calibration(_session(values))
    assert result is not None
    assert abs(result["acceleration_counts_per_g"] / 16384 - 1) < .05
    assert "gyroscope_counts_per_rad_s" not in result


def test_insufficient_quiet_intervals_have_no_calibration():
    values = 1000 + (np.arange(2000) % 10) * 1000
    assert estimate_gravity_calibration(_session(values)) is None
