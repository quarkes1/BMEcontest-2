# -*- coding: utf-8 -*-
"""姿态解算单元测试：合成信号验证 Pitch/Roll 绝对角、相对 Yaw、静止零偏补偿。"""
import numpy as np
from src.data.loader import SessionData
from src.features.pose import estimate_attitude, per_row_tilt, static_mask, GRAV

def _make(acc_fn, gyro_fn, dur_sec=30.0, fs=100.0, gap_ms=None):
    N = int(dur_sec * fs)
    t = (np.arange(N) * (1000.0 / fs)).astype(np.int64)
    if gap_ms:
        t[N // 2:] += gap_ms
    return SessionData(
        acc=acc_fn(N).astype(np.float32),
        gyro=gyro_fn(N).astype(np.float32),
        ppg=np.zeros((44, N), dtype=np.float32),
        t_acc=t, t_ppg=np.full(N, -1, dtype=np.int64),
        imu_valid=np.ones(N, dtype=bool), ppg_valid=np.zeros(N, dtype=bool),
        meta={"row_rate": fs})

def test_static_flat_zero_angles():
    s = _make(lambda n: np.tile([0, 0, GRAV], (n, 1)).T, lambda n: np.zeros((3, n)))
    r = estimate_attitude(s)
    assert np.abs(r["pitch"]).max() < 2.0
    assert np.abs(r["roll"]).max() < 2.0
    assert np.abs(r["yaw"]).max() < 2.0
    assert r["static"].mean() > 0.95

def test_pitch_30deg():
    """传感器 X 轴下倾 30°：重力读数为 [g·sin30, 0, g·cos30] → pitch≈30。"""
    a = np.array([GRAV * np.sin(np.deg2rad(30)), 0, GRAV * np.cos(np.deg2rad(30))])
    s = _make(lambda n: np.tile(a, (n, 1)).T, lambda n: np.zeros((3, n)))
    r = estimate_attitude(s)
    assert abs(np.median(r["pitch"]) - 30) < 2.0
    assert abs(np.median(r["roll"])) < 2.0

def test_roll_30deg():
    a = np.array([0, -GRAV * np.sin(np.deg2rad(30)), GRAV * np.cos(np.deg2rad(30))])
    s = _make(lambda n: np.tile(a, (n, 1)).T, lambda n: np.zeros((3, n)))
    r = estimate_attitude(s)
    assert abs(np.median(r["roll"]) - 30) < 2.0

def test_yaw_relative_zeroing_and_slope():
    """前 10s 静止（yaw 归零基准），后 20s 绕 Z 轴 0.5 rad/s 旋转 → yaw 斜率≈0.5。"""
    def gyro(n):
        g = np.zeros((3, n))
        g[2, int(10 * 100):] = 0.5
        return g
    s = _make(lambda n: np.tile([0, 0, GRAV], (n, 1)).T, gyro, dur_sec=30.0)
    r = estimate_attitude(s)
    t = r["t_sec"]
    assert abs(np.median(r["yaw"][t < 9])) < 2.0               # 静止段 yaw≈0
    slope = np.polyfit(t[t > 12], r["yaw"][t > 12], 1)[0]
    assert abs(slope - 0.5 * 180 / np.pi) < 12.0               # 0.5 rad/s ≈ 28.6°/s

def test_static_gyro_bias_compensated():
    """静止段陀螺零偏 [0.1,0,0]：不补偿则 pitch 漂移 171°；补偿后应 <3°。"""
    s = _make(lambda n: np.tile([0, 0, GRAV], (n, 1)).T,
              lambda n: np.tile([0.1, 0, 0], (n, 1)).T)
    r = estimate_attitude(s)
    assert np.abs(r["pitch"]).max() < 3.0
    assert np.abs(r["roll"]).max() < 3.0

def test_output_rate_10hz():
    s = _make(lambda n: np.tile([0, 0, GRAV], (n, 1)).T, lambda n: np.zeros((3, n)), dur_sec=20.0)
    r = estimate_attitude(s)
    assert abs(len(r["t_sec"]) - 200) <= 3
    assert np.allclose(np.diff(r["t_sec"]), 0.1, atol=1e-6)

def test_time_gap_resilient():
    """时间戳中途跳跃 5s 不得崩溃，输出全有限。"""
    s = _make(lambda n: np.tile([0, 0, GRAV], (n, 1)).T, lambda n: np.zeros((3, n)), gap_ms=5000)
    r = estimate_attitude(s)
    assert np.isfinite(r["pitch"]).all() and np.isfinite(r["yaw"]).all()

def test_per_row_tilt_30deg():
    a = np.array([GRAV * np.sin(np.deg2rad(30)), 0, GRAV * np.cos(np.deg2rad(30))])
    s = _make(lambda n: np.tile(a, (n, 1)).T, lambda n: np.zeros((3, n)), dur_sec=10.0)
    tilt = per_row_tilt(s.acc, 100.0)
    assert abs(np.median(tilt) - 30) < 2.0

def test_static_mask_detects_motion():
    n = 1000
    am = np.full(n, GRAV)
    am[500:700] += 2.0                                     # 明显晃动
    m = static_mask(am, 100.0)
    assert m[:400].all() and m[900:].all()            # 2s 滑窗：影响延伸至 700+200
    assert not m[550:650].any()
