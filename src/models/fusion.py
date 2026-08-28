# -*- coding: utf-8 -*-
"""动态可靠性加权融合（spec §3）：
- alpha = sigmoid(MLP(PPG_SNR, 灌注指数, IMU 活动度))，30s EMA 平滑
- 融合分 = alpha·P_ppg + (1-alpha)·P_imu
- 对照：手工 alpha 公式（逻辑回归等效的可解释权重）
"""
import numpy as np
import torch
import torch.nn as nn

N_FEAT = 3          # PPG SNR / 灌注指数 / IMU 活动度
EMA_TAU_S = 30.0

class FusionMLP(nn.Module):
    """可学习 alpha：3 特征 → 2 层 MLP → sigmoid。"""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def handcrafted_alpha(ppg_snr, pi, imu_act):
    """手工对照（逻辑回归形态）：PPG 质量好且腕部安静时信任 PPG，腕部活跃时信任 IMU。"""
    z = 1.5 * np.clip(ppg_snr, 0, 2) + 2.0 * np.clip(pi, 0, 2) - 0.8 * np.clip(imu_act, 0, 5)
    return 1.0 / (1.0 + np.exp(-z))


def ema_smooth(alpha, alpha_prev, step_s=1.0):
    """逐点 EMA（部署流式；alpha_prev 为上一时刻平滑值）。"""
    k = 1.0 - np.exp(-step_s / EMA_TAU_S)
    return alpha_prev + k * (alpha - alpha_prev)


def fuse(p_ppg, p_imu, alpha):
    """融合分：alpha·P_ppg + (1-alpha)·P_imu。"""
    return alpha * p_ppg + (1 - alpha) * p_imu
