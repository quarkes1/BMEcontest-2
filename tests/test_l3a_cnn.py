# -*- coding: utf-8 -*-
"""L3a CNN 单测：通道构建、镜像增强、前向形状、参数/MACs 约束。"""
import numpy as np
from src.data.loader import SessionData
from src.features.pose import GRAV
from src.models.l3a_cnn import (L3aCNN, build_raw_channels, mirror_channels,
                                count_params, count_macs, N_CHANNELS, WINDOW_LEN)

def _session(n=1100, moving=True, seed=0, fs=100.0):
    rng = np.random.RandomState(seed)
    t = np.arange(n) / fs
    if moving:
        a = np.vstack([0.4 * rng.randn(n) + np.sin(2 * np.pi * 1.5 * t),
                       0.4 * rng.randn(n), GRAV + 0.4 * rng.randn(n)])
        g = rng.randn(3, n)
    else:
        a = np.vstack([0.01 * rng.randn(n), 0.01 * rng.randn(n), GRAV + 0.01 * rng.randn(n)])
        g = 0.01 * rng.randn(3, n)
    return SessionData(
        acc=a.astype(np.float32), gyro=g.astype(np.float32),
        ppg=np.zeros((44, n), dtype=np.float32),
        t_acc=(np.arange(n) * 10).astype(np.int64),
        t_ppg=np.full(n, -1, dtype=np.int64),
        imu_valid=np.ones(n, dtype=bool), ppg_valid=np.zeros(n, dtype=bool),
        meta={"row_rate": fs})

def test_build_raw_channels_shape():
    X = build_raw_channels(_session(), 0, WINDOW_LEN)
    assert X.shape == (N_CHANNELS, WINDOW_LEN)
    assert np.isfinite(X).all()

def test_gravity_separation_static():
    """静止会话：重力分离后线加速度≈0，合值通道≈0。"""
    X = build_raw_channels(_session(moving=False), 0, WINDOW_LEN)
    assert np.abs(X[:3]).max() < 0.15          # la ≈ 0
    assert np.abs(X[9]).max() < 0.15           # 合值 ≈ 0

def test_mirror_only_flips_x_channels():
    X = build_raw_channels(_session(), 0, WINDOW_LEN)
    Y = mirror_channels(X)
    for ch in (0, 3, 6):                       # la_x / gyro_x / la_x 带通
        assert np.allclose(Y[ch], -X[ch])
    for ch in (1, 2, 4, 5, 7, 8, 9, 10):
        assert np.allclose(Y[ch], X[ch])

def test_forward_shapes():
    import torch
    model = L3aCNN(num_tableware=5)
    x = torch.randn(4, N_CHANNELS, WINDOW_LEN)
    eat, tw = model(x)
    assert eat.shape == (4,)
    assert tw.shape == (4, 5)

def test_param_budget():
    model = L3aCNN(num_tableware=5)
    assert count_params(model) <= 300_000

def test_mac_budget():
    model = L3aCNN(num_tableware=5)
    assert count_macs(model) <= 50_000_000

def test_forward_batch_independence():
    import torch
    model = L3aCNN(num_tableware=5).eval()
    x = torch.randn(2, N_CHANNELS, WINDOW_LEN)
    with torch.no_grad():
        e1, _ = model(x[:1])
        e2, _ = model(x[1:2])
        ea, _ = model(x)
    assert torch.allclose(ea, torch.cat([e1, e2]), atol=1e-5)
