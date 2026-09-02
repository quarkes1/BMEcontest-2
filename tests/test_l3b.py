# -*- coding: utf-8 -*-
"""L3b 生理专家网络单测：去噪、HRV 特征、前向形状、参数预算。"""
import numpy as np
from src.models.l3b_ppgnn import (L3bPPGNN, denoise_ppg, hrv_features, build_ppg_window,
                                  count_params, N_PPG_CHANNELS, PPG_WINDOW_ROWS, SEQ_LEN,
                                  EMBED_DIM, HRV_DIMS)
from src.data.loader import SessionData

def test_denoise_removes_common_mode():
    """纯净信号 + 共模伪影 → 去噪后伪影大幅衰减。"""
    rng = np.random.RandomState(0)
    T = 720
    clean = 0.5 * np.sin(2 * np.pi * 1.2 * np.arange(T) / 24.0)[None, :]      # (1, T)
    artifact = 2.0 * np.sin(2 * np.pi * 0.15 * np.arange(T) / 24.0)          # 慢漂移伪影
    ppg = np.repeat(clean, 44, axis=0) + artifact[None, :] + 0.05 * rng.randn(44, T)
    out = denoise_ppg(ppg)
    # 伪影功率应显著下降（共模被回归掉）
    artifact_power_before = np.var(ppg.mean(axis=0))
    artifact_power_after = np.var(out.mean(axis=0))
    assert artifact_power_after < artifact_power_before * 0.01

def test_hrv_features_shape_and_nan_safety():
    rng = np.random.RandomState(1)
    T = 720
    ppg = 0.5 * np.sin(2 * np.pi * 1.1 * np.arange(T) / 24.0) + 0.1 * rng.randn(T)
    ppg = np.repeat(ppg[None, :], 44, axis=0)
    f = hrv_features(ppg)
    assert f.shape == (8,)
    assert np.isfinite(f).all()
    # 无信号（全零）也不得产生 NaN
    f0 = hrv_features(np.zeros((44, T)))
    assert np.isfinite(f0).all()

def test_build_ppg_window():
    n = 1500
    rng = np.random.RandomState(2)
    t = np.arange(n) / 24.0
    ppg = 0.5 * np.sin(2 * np.pi * 1.1 * t)[None, :] + 0.1 * rng.randn(44, n)
    ppg_valid = np.zeros(n, dtype=bool)
    ppg_valid[::5] = True                              # 采样调度：每 5 行 1 个有效行
    s = SessionData(
        acc=np.random.randn(3, n).astype(np.float32),
        gyro=np.random.randn(3, n).astype(np.float32),
        ppg=ppg.astype(np.float32),
        t_acc=(np.arange(n) * 40).astype(np.int64),
        t_ppg=np.full(n, -1, dtype=np.int64),
        imu_valid=np.ones(n, dtype=bool), ppg_valid=ppg_valid,
        meta={"row_rate": 25.0})
    X, hrv = build_ppg_window(s, 0, 750)               # 30s × 25 行/s 跨度
    assert X.shape == (N_PPG_CHANNELS, PPG_WINDOW_ROWS)
    assert X.dtype == np.float16
    assert hrv.shape == (HRV_DIMS,)

def test_forward_shapes():
    import torch
    model = L3bPPGNN()
    x = torch.randn(4, SEQ_LEN, N_PPG_CHANNELS, PPG_WINDOW_ROWS)
    feat = torch.randn(4, SEQ_LEN, HRV_DIMS)
    logits = model(x, feat)
    assert logits.shape == (4, SEQ_LEN)

def test_param_budget():
    model = L3bPPGNN()
    assert count_params(model) <= 150_000

def test_embed_dim_constant():
    assert EMBED_DIM == 64
