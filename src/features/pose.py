# -*- coding: utf-8 -*-
"""姿态解算（W2，spec §6）：
- 低通重力分离（fc 0.8Hz）+ 互补滤波（Mahony 型，kp 0.3 / ki 0.01，静止段零偏校正）
- Pitch/Roll 绝对（重力方向）、Yaw 相对（无磁力计；首个静止段归零）
- 输出 10Hz 姿态角曲线（web 与 L2 特征共用基础件）

约定（文档化，便于前端一致）：
- 世界系 Z 轴向上；传感器系 X=腕侧向、Y=前臂轴向近端、Z=表盘法向
- pitch = 传感器 X 轴相对水平面倾角（前臂旋转轴），roll = Y 轴倾角（抬腕）
- 倾角 tilt = 重力方向与腕 Z 轴夹角（L2 特征口径，与 baseline_features 一致）
"""
import numpy as np
from scipy.signal import butter, sosfilt

GRAV = 9.81


# ---------------------------------------------------------------- 静止检测
def static_mask(am, fs, win_sec=2.0, mag_thr=0.25, var_thr=0.012):
    """静止段掩码：合加速度接近 1g 且 2s 滑动方差小。am: (N,) 合加速度，fs: 行率。"""
    win = max(8, int(fs * win_sec))
    csum = np.cumsum(np.insert(am.astype(np.float64), 0, 0.0))
    csum2 = np.cumsum(np.insert(am.astype(np.float64) ** 2, 0, 0.0))
    n = np.arange(1, len(am) + 1)
    n0 = np.clip(n - win, 1, None)
    mean = (csum[1:] - csum[n0 - 1]) / (n - n0 + 1)
    var = np.clip((csum2[1:] - csum2[n0 - 1]) / (n - n0 + 1) - mean ** 2, 0, None)
    return (np.abs(am - GRAV) < mag_thr) & (var < var_thr)


# ---------------------------------------------------------------- 行级倾角（向量化，L2 训练特征用）
def per_row_tilt(acc, fs, smooth_sec=1.0):
    """每行倾角（度）：1s 均值重力方向与腕 Z 轴夹角。全量会话可用（纯向量化）。"""
    win = max(4, int(fs * smooth_sec))
    csum = np.cumsum(np.insert(acc.astype(np.float64), 0, 0.0, axis=1), axis=1)
    n = np.arange(1, acc.shape[1] + 1)
    n0 = np.clip(n - win, 1, None)
    g = (csum[:, 1:] - csum[:, n0 - 1]) / (n - n0 + 1)      # 3×N 平滑重力
    gn = np.linalg.norm(g, axis=0) + 1e-9
    return np.degrees(np.arccos(np.clip(g[2] / gn, -1, 1)))


# ---------------------------------------------------------------- 四元数小工具
def _q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _q_norm(q):
    return q / (np.linalg.norm(q) + 1e-12)


def _q_rot(q, v):
    """v 在世界系，返回 R(q)ᵀ 作用后的传感器系向量（姿态四元数 q: 传感器→世界）。"""
    vq = np.array([0.0, v[0], v[1], v[2]])
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    return _q_mul(_q_mul(qc, vq), q)[1:]


def _q_from_two(u, v):
    """最短弧四元数：把单位向量 u 旋转到单位向量 v。"""
    u, v = u / (np.linalg.norm(u) + 1e-12), v / (np.linalg.norm(v) + 1e-12)
    c = np.dot(u, v)
    if c < -0.999999:
        axis = np.cross(np.array([1.0, 0, 0]), u)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(np.array([0.0, 1.0, 0]), u)
        return _q_norm(np.array([0.0, *axis]))
    axis = np.cross(u, v)
    q = np.array([1.0 + c, *axis])
    return _q_norm(q)


# ---------------------------------------------------------------- 主解算
def _to_timestamp_grid(t_acc, vals):
    """按时间戳聚合（同戳多行取均值），返回 (t_ms, 3×M)。"""
    order = np.argsort(t_acc, kind="stable")
    t_sorted = t_acc[order]
    vals_sorted = vals[:, order]
    bounds = np.flatnonzero(np.r_[True, np.diff(t_sorted) != 0, True])
    t_out = t_sorted[bounds[:-1]]
    agg = np.stack([vals_sorted[:, bounds[i]:bounds[i + 1]].mean(axis=1)
                    for i in range(len(bounds) - 1)], axis=1)
    return t_out, agg


def estimate_attitude(session, out_hz=10.0, fc=0.8, kp=0.3, ki=0.01):
    """全会话姿态解算。返回 dict：t_sec/静态掩码 static/pitch/roll/yaw（度）。
    pitch/roll 绝对，yaw 相对（首个静止段归零）。"""
    fs = session.meta.get("row_rate", 105.0)
    valid = session.t_acc > 0
    t_ms, acc = _to_timestamp_grid(session.t_acc[valid], session.acc[:, valid])
    _, gyro = _to_timestamp_grid(session.t_acc[valid], session.gyro[:, valid])

    am = np.linalg.norm(acc, axis=0)
    gm = np.linalg.norm(gyro, axis=0)
    static = static_mask(am, fs) & (gm < 0.15)        # 加速度稳定 且 角速度小（绕重力轴纯旋转不算静止）
    if not static.any():
        static[:min(64, len(static))] = True          # 兜底：无静止段时用开头做初始对准

    # 重力分离：fc 低通
    sos = butter(2, min(fc, fs / 2 * 0.9), btype="low", fs=fs, output="sos")
    acc_lp = sosfilt(sos, acc, axis=1)
    # 静止段陀螺零偏（分段均值 + 线性插值）
    bias = np.zeros(3)
    seg = np.diff(np.r_[0, static.astype(int), 0])
    starts, ends = np.where(seg == 1)[0], np.where(seg == -1)[0]
    bias_pts_t, bias_pts = [], []
    for s, e in zip(starts, ends):
        if e - s < 8:
            continue
        bias_pts_t.append((t_ms[s] + t_ms[e - 1]) / 2)
        bias_pts.append(gyro[:, s:e].mean(axis=1))
    if not bias_pts_t:
        bias_pts_t, bias_pts = [t_ms[0], t_ms[-1]], [np.zeros(3), np.zeros(3)]
    if bias_pts_t[0] > t_ms[0]:
        bias_pts_t.insert(0, t_ms[0]); bias_pts.insert(0, bias_pts[0])
    if bias_pts_t[-1] < t_ms[-1]:
        bias_pts_t.append(t_ms[-1]); bias_pts.append(bias_pts[-1])
    bias_t = np.array(bias_pts_t); bias_v = np.array(bias_pts)
    bias_full = np.stack([np.interp(t_ms, bias_t, bias_v[:, i]) for i in range(3)], axis=1)

    # 初始四元数：首个静止段的传感器"上"（低通加速度方向）对准世界 Z 轴
    i0 = int(np.flatnonzero(static)[0])
    a0 = acc_lp[:, i0] / (np.linalg.norm(acc_lp[:, i0]) + 1e-12)
    q = _q_from_two(a0, np.array([0.0, 0.0, 1.0]))
    e_int = np.zeros(3)

    n = len(t_ms)
    qs = np.zeros((n, 4)); qs[0] = q
    for i in range(1, n):
        dt = (t_ms[i] - t_ms[i - 1]) / 1000.0
        if dt <= 0 or dt > 2.0:                       # 时间戳跳跃：重置积分
            q = _q_norm(q); qs[i] = q; e_int[:] = 0
            continue
        w = gyro[:, i] - bias_full[i]
        a_n = acc_lp[:, i] / (np.linalg.norm(acc_lp[:, i]) + 1e-12)
        v_hat = _q_rot(q, np.array([0.0, 0.0, 1.0]))   # 预测的传感器系重力方向
        e = np.cross(a_n, v_hat)
        e_int = np.clip(e_int + e * dt, -5.0, 5.0)
        w_corr = w + kp * e + ki * e_int
        wq = np.array([0.0, w_corr[0], w_corr[1], w_corr[2]])
        q = _q_norm(q + 0.5 * dt * _q_mul(q, wq))
        qs[i] = q

    # 四元数 → 欧拉角（传感器系重力方向反推）
    g_s = np.stack([_q_rot(qs[i], np.array([0.0, 0.0, 1.0])) for i in range(n)])
    pitch = np.degrees(np.arctan2(g_s[:, 0], np.sqrt(g_s[:, 1] ** 2 + g_s[:, 2] ** 2)))
    roll = np.degrees(np.arctan2(-g_s[:, 1], g_s[:, 2]))
    # ZYX 欧拉 yaw（纯 Z 旋转下 atan2(qz,qw) 只会给半角，须用完整公式）
    yaw = np.degrees(np.unwrap(np.arctan2(2 * (qs[:, 0] * qs[:, 3] + qs[:, 1] * qs[:, 2]),
                                          1 - 2 * (qs[:, 2] ** 2 + qs[:, 3] ** 2))))
    yaw -= np.median(yaw[static])                     # 相对 Yaw：静止段归零

    # 降采样到 out_hz
    t_sec = t_ms / 1000.0
    if out_hz is not None and len(t_sec) > 2:
        t_out = np.arange(t_sec[0], t_sec[-1], 1.0 / out_hz)
        pitch = np.interp(t_out, t_sec, pitch)
        roll = np.interp(t_out, t_sec, roll)
        yaw = np.interp(t_out, t_sec, yaw)
        static = np.interp(t_out, t_sec, static.astype(float)) > 0.5
        t_sec = t_out
    return {"t_sec": t_sec, "static": static, "pitch": pitch, "roll": roll, "yaw": yaw}


def pose_curve(session, out_hz=10.0, **kw):
    """alias：给 web/缓存用的 10Hz 姿态曲线。"""
    return estimate_attitude(session, out_hz=out_hz, **kw)
