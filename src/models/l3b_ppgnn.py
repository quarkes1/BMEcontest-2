# -*- coding: utf-8 -*-
"""L3b 生理专家网络（非惯用手分支，spec §3）：
- 输入：30s 窗 PPG 44 通道（720 行）+ 弱 IMU 姿态/HRV 特征
- 多波长噪声抵消：通道级共模回归（运动伪影分离，华为 EP4186416 思路的轻量实现）
- 波形嵌入：3 层 Conv（44→32→64→64，stride 3/3/2）→ GAP → 64 维
- 时序：GRU(72→96) 消费连续窗口序列（6 窗 = 2 分钟上下文）
- 输出：每窗口进食概率；~105K 参数（预算 ~150K）
- 物理锚点：交感激活/迷走撤离（RR 间期缩短、HRV 变化）
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

PPG_WINDOW_ROWS = 720          # 24 行/s × 30s
N_PPG_CHANNELS = 44
HRV_DIMS = 8
SEQ_LEN = 6                    # GRU 上下文窗口数（10s 步长 → 2 分钟）
EMBED_DIM = 64


def denoise_ppg(ppg: np.ndarray) -> np.ndarray:
    """通道级共模回归：每个通道减去其与通道均值（伪影代理）的线性投影。
    运动伪影常同步出现在多通道，共模分量以伪影为主。ppg: (44, T)。"""
    cm = ppg.mean(axis=0, keepdims=True)
    cmc = cm - cm.mean()
    denom = (cmc @ cmc.T).item() + 1e-9
    beta = (ppg @ cmc.T) / denom                       # (44, 1)
    return (ppg - beta * cmc).astype(np.float32)


def hrv_features(ppg: np.ndarray, gyro_std: float = 0.0, acc_var: float = 0.0) -> np.ndarray:
    """每 30s 窗的 HRV/生理特征（8 维）：mean_rr, rmssd, sdnn, 灌注指数 AC/DC,
    SNR, 峰率, 陀螺活动度, 加速度方差。无有效峰时安全退化为 0。"""
    s = ppg.mean(axis=0)                               # 通道均值做峰检测
    s = s - s.mean()
    peaks, _ = find_peaks(s, distance=8, prominence=np.std(s) * 0.3)
    feats = np.zeros(8, dtype=np.float32)
    if len(peaks) >= 3:
        rr = np.diff(peaks) / 24.0                     # 24 行/s 近似
        feats[0] = np.mean(rr)
        feats[1] = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else 0.0
        feats[2] = np.std(rr)
        feats[5] = len(peaks) / 30.0
    ac = np.std(s) + 1e-9
    dc = np.abs(np.mean(ppg)) + 1e-9
    feats[3] = ac / dc                                 # 灌注指数近似
    # SNR：信号方差 / 平滑残差方差
    from scipy.ndimage import uniform_filter1d
    residual = s - uniform_filter1d(s, size=9)
    feats[4] = float(np.var(s) / (np.var(residual) + 1e-9))
    feats[6] = gyro_std
    feats[7] = acc_var
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def build_ppg_window(session, start, end):
    """30s 窗（start/end = 行号，跨度 ~3150 行 @105 行/s）→ (X (44,720) fp16, hrv (8,))。
    PPG 78% 行全零为采样调度：压缩掉无效行得 ~720 有效行，不足补零。"""
    ppg = session.ppg[:, start:end]
    active = session.ppg_valid[start:end]
    X = ppg[:, active] if active.any() else ppg[:, :0]
    if X.shape[1] > PPG_WINDOW_ROWS:
        X = X[:, :PPG_WINDOW_ROWS]
    elif X.shape[1] < PPG_WINDOW_ROWS:
        X = np.pad(X, ((0, 0), (0, PPG_WINDOW_ROWS - X.shape[1])))
    X = denoise_ppg(X)
    g = session.gyro[:, start:end]
    a = session.acc[:, start:end]
    hrv = hrv_features(X, gyro_std=float(np.linalg.norm(g, axis=0).std()),
                       acc_var=float(np.linalg.norm(a, axis=0).var()))
    return X.astype(np.float16), hrv


class L3bPPGNN(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(N_PPG_CHANNELS, 32, 9, stride=3, padding=4, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 9, stride=3, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, embed_dim, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(embed_dim + HRV_DIMS, 96, batch_first=True)
        self.head = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1))

    def forward(self, x, feat):
        """x: (B, seq, 44, 720) fp32，feat: (B, seq, 8)。返回 (B, seq) 每窗口 logit。"""
        B, S = x.shape[:2]
        emb = self.embed(x.reshape(B * S, N_PPG_CHANNELS, PPG_WINDOW_ROWS))
        emb = emb.mean(dim=2).view(B, S, -1)           # GAP → (B, seq, 64)
        out, _ = self.gru(torch.cat([emb, feat], dim=-1))
        return self.head(out).squeeze(-1)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
