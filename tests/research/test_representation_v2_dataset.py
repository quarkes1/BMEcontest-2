"""Timestamp, masking and full-span contracts for motion tokens."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.research.representation_v2.dataset import PreprocessConfig, build_token_bag, pool_time_bins
from src.pipeline.event_stack import EventRef


def _session(timestamps_ms: np.ndarray, *, valid: np.ndarray | None = None):
    t = timestamps_ms.astype(np.int64)
    seconds = (t - t[0]) / 1000.0
    acc = np.vstack((np.sin(seconds), np.cos(seconds), np.ones(len(t)))).astype(np.float32)
    gyro = np.vstack((np.sin(seconds * 2), np.cos(seconds * 2), np.zeros(len(t)))).astype(np.float32)
    return SimpleNamespace(acc=acc, gyro=gyro, t_acc=t, imu_valid=np.ones(len(t), bool) if valid is None else valid, row_rate=100.0)


def test_token_covers_start_middle_end_of_ten_minute_candidate():
    times = np.arange(0, 600_000, 10, dtype=np.int64)
    bag = build_token_bag(_session(times), EventRef("one", 0, 600_000), PreprocessConfig())
    assert bag.values.shape == (120, 6, 125)
    assert bag.sample_mask.shape == (120, 125)
    assert bag.token_center_ms[0] == 2_500
    assert bag.token_center_ms[60] == 302_500
    assert bag.token_center_ms[-1] == 597_500
    assert bag.token_mask.all()


def test_timestamp_interpolation_uses_elapsed_time_not_row_index():
    times = np.r_[np.arange(0, 2_500, 10), np.arange(2_510, 5_000, 20)].astype(np.int64)
    bag = build_token_bag(_session(times), EventRef("one", 0, 5_000), PreprocessConfig(gravity_align=False))
    assert bag.token_mask[0]
    # A 40 ms target near 4 s should reflect sin(4), regardless of source row density.
    assert abs(float(bag.values[0, 0, 100]) - np.sin(4.0)) < 0.05


def test_gap_is_masked_and_not_interpolated():
    times = np.r_[np.arange(0, 2_000, 10), np.arange(2_400, 5_000, 10)].astype(np.int64)
    bag = build_token_bag(_session(times), EventRef("one", 0, 5_000), PreprocessConfig(gravity_align=False))
    assert not bag.sample_mask[0, 55]
    assert bag.values[0, :, 55].sum() == 0
    assert bag.token_mask[0]  # 400 ms gap is below the 20% invalid-token limit.


def test_short_candidate_is_retained_as_invalid():
    times = np.arange(0, 1_000, 10, dtype=np.int64)
    bag = build_token_bag(_session(times), EventRef("one", 0, 1_000), PreprocessConfig())
    assert bag.values.shape == (1, 6, 125)
    assert not bag.token_mask[0]


def test_repeated_packet_timestamp_is_expanded_without_row_collapse():
    times = np.arange(0, 5_000, 10, dtype=np.int64)
    times[101] = times[100]
    bag = build_token_bag(_session(times), EventRef("one", 0, 5_000), PreprocessConfig())
    assert bag.token_mask[0]


def test_realistic_packet_timestamps_and_invalid_minus_one_rows():
    packet_times = np.repeat(np.arange(0, 5_100, 95, dtype=np.int64), 10)
    packet_times = np.r_[packet_times[:100], [-1] * 5, packet_times[100:]]
    valid = packet_times >= 0
    session = _session(packet_times, valid=valid)
    session.row_rate = 113.0
    bag = build_token_bag(session, EventRef("one", 0, 5_000), PreprocessConfig())
    assert bag.token_mask[0]
    assert np.count_nonzero(bag.sample_mask) >= 100


def test_candidate_outside_session_is_rejected():
    times = np.arange(0, 5_000, 10, dtype=np.int64)
    with pytest.raises(ValueError, match="session"):
        build_token_bag(_session(times), EventRef("one", -1_000, 5_000), PreprocessConfig())


def test_pool_bins_preserves_order_and_empty_bins():
    times = np.arange(0, 10_000, 10, dtype=np.int64)
    bag = build_token_bag(_session(times), EventRef("one", 0, 10_000), PreprocessConfig())
    embeddings = np.stack((np.ones(16), np.ones(16) * 2)).astype(np.float32)
    bins, mask = pool_time_bins(embeddings, bag, bins=4)
    assert bins.shape == (4, 36)
    assert mask.tolist() == [False, True, False, True]
    assert np.allclose(bins[1, :16], 1)
    assert np.allclose(bins[3, :16], 2)


def test_gravity_rotation_is_shared_by_acc_and_gyro():
    times = np.arange(0, 5_000, 10, dtype=np.int64)
    session = _session(times)
    session.acc[:] = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)
    session.gyro[:] = session.acc
    bag = build_token_bag(session, EventRef("one", 0, 5_000), PreprocessConfig())
    assert np.allclose(bag.values[0, :3, bag.sample_mask[0]], bag.values[0, 3:, bag.sample_mask[0]])


def test_low_rate_session_retains_invalid_candidate_for_control_fallback():
    times = np.arange(0, 10_000, 500, dtype=np.int64)
    session = _session(times)
    session.row_rate = 2.0
    bag = build_token_bag(session, EventRef("one", 0, 5_000), PreprocessConfig())
    assert bag.values.shape == (1, 6, 125)
    assert not bag.token_mask.any()
