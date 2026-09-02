# -*- coding: utf-8 -*-
"""惯用手/非惯用手双分支路由。

竞赛题面要求两种佩戴场景分别设计：惯用手（IMU 动作特征明显）与非惯用手
（动作弱，需上下文/生理信号辅助）。本项目实测约束：目标数据集 PPG 有效采样
仅约 2Hz，无法做逐拍 HR/HRV 估计——因此非惯用手一侧以"弱 IMU + 全天上下文"
实现（外部双腕数据预训练隐含学得弱信号判别能力）。

本模块提供两种路由策略（均不假定元数据提供佩戴侧标记，数据驱动）：
- AdaptiveRouter：以会话级 IMU 4-16Hz 能量分布连续插值分支权重 α——
  手-口动作峰值显著（Bite 特征丰富）时 α→1（惯用手深度分支主导），
  动作平缓时 α→α_min（弱信号上下文主导）；
- HardRouter：以会话前 10 分钟 4-16Hz 能量方差做硬切换（w∈{0,1}）。

两种策略在单侧佩戴的验证集上均无增益（深度分为主导信号时降权有害），
代码保留用于含非惯用手佩戴的测试场景。
"""
import numpy as np


class AdaptiveRouter:
    """会话级自适应路由权重计算。

    gh_env: 会话 1s 网格 gyro 4-16Hz 带通能量序列（需在原始采样率上计算）。
    统计量：
      p50/p90：绝对能量水平（bite 动作的强度）
      peak_ratio = p90/(p50+eps)：峰值突出度（间歇性手-口动作 vs 持续活动）
    映射：α = clip(α_min + (1-α_min) * g(score), α_min, 1)
      score = log1p(p90) 经 min-max 归一（参考值来自 FD-I 观测：p90 ~ 1e2-1e3 数量级）
    """

    def __init__(self, alpha_min=0.3, p90_ref_lo=50.0, p90_ref_hi=800.0):
        self.alpha_min = alpha_min
        self.lo, self.hi = p90_ref_lo, p90_ref_hi

    def score(self, gh_env):
        p50 = float(np.percentile(gh_env, 50))
        p90 = float(np.percentile(gh_env, 90))
        return p90, p50, p90 / (p50 + 1e-6)

    def session_weight(self, gh_env):
        """→ (alpha, 统计) alpha ∈ [0.3, 1]。"""
        p90, p50, pr = self.score(gh_env)
        # p90 对数尺度 min-max（FD 观测范围校准）
        t = (np.log1p(p90) - np.log1p(self.lo)) / (np.log1p(self.hi) - np.log1p(self.lo) + 1e-9)
        t = float(np.clip(t, 0.0, 1.0))
        alpha = self.alpha_min + (1.0 - self.alpha_min) * t
        return alpha, {"p90": p90, "p50": p50, "peak_ratio": pr, "t": t}


class HardRouter:
    """硬路由：以会话前 10 分钟 IMU 4-16Hz 能量方差做硬切换。

    方差 ≥ 阈值 → 判定惯用手/动作丰富 → w=1（深度分支全权重）
    方差 <  阈值 → 判定非惯用手/动作平缓 → w=0（弱信号上下文分支：
    LGBM 活动分 + 会话门控 + 时间先验）
    """

    def __init__(self, var_thresh=1.0):
        self.var_thresh = var_thresh

    def session_route(self, gh_env, lookback_s=600):
        """gh_env: 1s 网格 4-16Hz gyro 能量序列；返回 (w, var)。
        前 10 分钟（若会话短于 lookback 用全部前段）。"""
        seg = gh_env[:lookback_s]
        if len(seg) < 30:
            return 1.0, float(np.var(seg)) if len(seg) else 0.0
        var = float(np.var(seg))
        return (1.0 if var >= self.var_thresh else 0.0), var

    def fit_thresh(self, gh_envs, sids_with_meals, pct=40.0):
        """数据驱动阈值：用有餐会话（惯用手类）的方差分布低分位估计。"""
        vars_ = [np.var(g[:600]) for g in gh_envs if len(g) > 30]
        if not vars_:
            return self.var_thresh
        self.var_thresh = float(np.percentile(vars_, pct))
        return self.var_thresh


def gyro_high_band_env(gyro, fs, band=(4.0, 16.0), win_s=5.0, step_s=1.0):
    """原始 gyro (3, N) → 4-16Hz 带通能量包络（1s 网格，与 validate_baselines env 同构）。

    必须在原始采样率上滤波（10Hz 网格会丢失 4-16Hz 信息）。"""
    import scipy.signal
    if fs <= 2 * band[1]:          # 采样率不足（低采样率会话）→ 无高频信息
        return np.zeros(0, np.float32)
    win, st = int(win_s * fs), int(step_s * fs)
    n = gyro.shape[1]
    n_w = (n - win) // st + 1
    if n_w <= 0:
        return np.zeros(0, np.float32)
    sos = scipy.signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    out = np.empty(n_w, np.float32)
    for b0 in range(0, n_w, 4000):
        b1 = min(b0 + 4000, n_w)
        m = b1 - b0
        idx = b0 * st + np.arange(m)[:, None] * st + np.arange(win)[None, :]
        seg = gyro[:, idx]
        la = seg - np.median(seg, axis=2, keepdims=True)
        lam = np.linalg.norm(la, axis=0)
        e = scipy.signal.sosfiltfilt(sos, lam, axis=1)
        out[b0:b1] = np.abs(e).mean(axis=1)
    return out
