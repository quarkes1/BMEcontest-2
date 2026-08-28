# -*- coding: utf-8 -*-
"""LightGBM 基线特征 v1：窗口统计量（37 维）。"""
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import skew, kurtosis

def _bandpass(signal, low, high, fs):
    if fs <= 4.0:
        return np.zeros_like(signal)   # 极低行率（长空档会话）：带通无意义，降级为零
    high = min(high, fs / 2 * 0.95)    # 保证 Wn < fs/2
    if high <= low:
        return np.zeros_like(signal)
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, signal)

def window_features(session, start, end) -> np.ndarray:
    fs = session.meta.get("row_rate", 105.0)
    a = session.acc[:, start:end]
    g = session.gyro[:, start:end]
    am = np.linalg.norm(a, axis=0)                 # 合加速度
    gm = np.linalg.norm(g, axis=0)
    am_bp = _bandpass(am, 0.5, 2.0, fs)            # 0.5-2Hz 咀嚼带
    grav = np.median(a, axis=1, keepdims=True)     # 重力估计（窗口均值近似）
    la = a - grav                                   # 线加速度
    lam = np.linalg.norm(la, axis=0)
    grav_norm = np.linalg.norm(grav[:, 0]) + 1e-9
    tilt = np.degrees(np.arccos(np.clip(grav[2, 0] / grav_norm, -1, 1)))   # 腕部倾角（标量）

    feats = [
        np.mean(am), np.std(am), np.percentile(am, 90), np.max(am),
        np.mean(gm), np.std(gm), np.percentile(gm, 90),
        np.mean(np.abs(am_bp)), np.std(am_bp),
        np.mean(lam), np.std(lam), np.percentile(lam, 90), np.max(lam),
        tilt, np.std(a[2, :]) / grav_norm,                                # 倾角 + Z 轴波动
        np.mean(a, axis=1)[0], np.mean(a, axis=1)[1], np.mean(a, axis=1)[2],
        np.std(a, axis=1)[0], np.std(a, axis=1)[1], np.std(a, axis=1)[2],
        np.mean(g, axis=1)[0], np.mean(g, axis=1)[1], np.mean(g, axis=1)[2],
        np.std(g, axis=1)[0], np.std(g, axis=1)[1], np.std(g, axis=1)[2],
        np.mean(am_bp**2), np.max(np.abs(am_bp)),
        np.count_nonzero(np.diff(np.signbit(am_bp))),      # 过零率
        float(skew(am)), float(kurtosis(am)),              # 幅度分布形状
        float(session.imu_valid[start:end].mean()),        # IMU 有效率
        float(session.ppg_valid[start:end].mean()),        # PPG 有效率
        np.mean(session.ppg[:, start:end], axis=1)[:10].mean(),   # 前10通道均值
        np.std(session.ppg[:, start:end], axis=1)[:10].mean(),
        np.max(np.abs(session.ppg[:, start:end]), axis=1)[:10].mean(),
    ]
    return np.array(feats, dtype=np.float32)
