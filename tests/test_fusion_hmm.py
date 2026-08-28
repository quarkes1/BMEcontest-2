# -*- coding: utf-8 -*-
"""融合层与 HMM 解码单测。"""
import numpy as np
import torch
from src.models.fusion import FusionMLP, handcrafted_alpha, ema_smooth, fuse
from src.models.hmm_decode import viterbi, decode_events, calibrate

def test_fusion_mlp_shape_and_range():
    model = FusionMLP()
    x = torch.randn(16, 3)
    a = model(x)
    assert a.shape == (16,)
    assert ((a >= 0) & (a <= 1)).all()

def test_handcrafted_alpha_extremes():
    # 高质量 PPG + 安静手腕 → alpha 高（信任 PPG）；差 PPG + 剧烈运动 → alpha 低
    a_high = handcrafted_alpha(2.0, 2.0, 0.0)
    a_low = handcrafted_alpha(0.0, 0.0, 5.0)
    assert a_high > 0.8 and a_low < 0.2

def test_ema_smooth():
    a = np.full(50, 0.9)
    cur = 0.5
    for i in range(50):
        cur = ema_smooth(a[i], cur, step_s=1.0)
    # τ=30s、步长 1s：50 步后 0.9 - 0.4×(1-e^{-1/30})^50 ≈ 0.824
    assert 0.81 < cur < 0.84

def test_fuse_weighted_average():
    assert abs(fuse(0.8, 0.4, 0.25) - 0.5) < 1e-9     # 0.25*0.8 + 0.75*0.4

def test_viterbi_recovers_long_run():
    rng = np.random.RandomState(0)
    p = np.concatenate([rng.uniform(0, 0.2, 300),
                        rng.uniform(0.7, 0.95, 200),
                        rng.uniform(0, 0.2, 300)])
    states = viterbi(p)
    assert states[300:500].mean() > 0.9               # 高概率进食段被完整恢复
    assert states[:250].mean() < 0.05 and states[-250:].mean() < 0.05

def test_viterbi_removes_flickers():
    """1 帧闪烁被转移平滑消除。"""
    p = np.full(100, 0.9)
    p[50] = 0.01
    states = viterbi(p)
    assert states.sum() > 95                          # 闪烁不打断长段

def test_decode_events():
    t0 = np.arange(300) * 1000
    t1 = t0 + 5000
    states = np.zeros(300, dtype=np.int8)
    states[100:160] = 1
    evs = decode_events(states, t0, t1)
    assert len(evs) == 1
    assert abs(evs[0][0] - (100000 - 6000)) < 1000      # 窗 100 起：94000
    assert abs(evs[0][1] - (164000 + 6000)) < 1000      # 窗 159 止：t1=164000 → 170000

def test_calibrate_bounds():
    p = np.random.RandomState(0).uniform(0.1, 0.9, 1000)
    c = calibrate(p)
    assert c.min() >= 0 and c.max() <= 1.0
