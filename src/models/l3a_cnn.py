# -*- coding: utf-8 -*-
"""L3a 动作专家网络（惯用手分支，spec §3）：
- 输入：5s×105Hz 窗口，11 通道 = 重力分离线加速度(3) + 陀螺(3)
       + 0.5-2Hz 带通显式通道(3) + 线加速度合值(1) + 带通能量包络(1)
- 结构：1×1 stem → 3× 深度可分离卷积块（32→64→128，stride 1/2/2）→ GAP → FC64
       → 双头：进食概率（BCE）+ 餐具多类辅助头（推理丢弃）
- 约束：≤300K 参数、≤50M MACs/窗；预处理不做平滑（文献结论）、重力分离、标准化
"""
import numpy as np
import torch
import torch.nn as nn
from src.features.baseline_features import _bandpass

N_CHANNELS = 11
WINDOW_LEN = 525          # 105Hz × 5s（行数，与 config.WINDOW_ROWS 一致）
STEM_OUT = 32


def build_raw_channels(session, start, end) -> np.ndarray:
    """原始窗口 → 11 通道 (11, W)。训练缓存与推理共用（保证分布一致）。
    不做平滑；重力 = 窗口加速度中位数。"""
    fs = session.meta.get("row_rate", 105.0)
    a = session.acc[:, start:end]
    g = session.gyro[:, start:end]
    la = a - np.median(a, axis=1, keepdims=True)          # 重力分离线加速度
    la_bp = np.stack([_bandpass(la[i], 0.5, 2.0, fs) for i in range(3)])
    lam = np.linalg.norm(la, axis=0)                      # 合值
    env = _bandpass(lam, 0.5, 2.0, fs)
    win = max(4, int(fs * 0.5))
    csum = np.cumsum(np.insert(np.abs(env).astype(np.float64), 0, 0.0))
    n = np.arange(1, len(env) + 1)
    n0 = np.clip(n - win, 1, None)
    env_smooth = (csum[1:] - csum[n0 - 1]) / (n - n0 + 1)  # 带通能量包络（0.5s 平滑）
    return np.vstack([la, g, la_bp, lam[None], env_smooth[None]]).astype(np.float32)


def mirror_channels(X: np.ndarray) -> np.ndarray:
    """左右镜像（spec §4 增强）：线加速度 X 与陀螺 X 取反，其余通道不变。
    合值与包络是标量不变式，无需处理。"""
    Y = X.copy()
    Y[0] = -Y[0]        # la_x
    Y[3] = -Y[3]        # gyro_x
    Y[6] = -Y[6]        # la_x 带通
    return Y


class _SepBlock(nn.Module):
    def __init__(self, c_in, c_out, stride, kernel=5):
        super().__init__()
        self.dw = nn.Conv1d(c_in, c_in, kernel, stride=stride, padding=kernel // 2,
                            groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm1d(c_in)
        self.pw = nn.Conv1d(c_in, c_out, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class L3aCNN(nn.Module):
    def __init__(self, num_tableware=5, stem_out=STEM_OUT):
        super().__init__()
        self.stem = nn.Conv1d(N_CHANNELS, stem_out, 1, bias=False)
        self.bn0 = nn.BatchNorm1d(stem_out)
        self.blocks = nn.Sequential(
            _SepBlock(stem_out, 64, stride=2),     # spec：逐块 stride 2（525→263→132→66）
            _SepBlock(64, 128, stride=2),
            _SepBlock(128, 128, stride=2),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.eat_head = nn.Linear(64, 1)
        self.tableware_head = nn.Linear(64, num_tableware)
        self.num_tableware = num_tableware

    def forward(self, x):
        x = nn.functional.relu(self.bn0(self.stem(x)))
        x = self.blocks(x)
        x = x.mean(dim=2)                     # GAP
        h = self.head(x)
        return self.eat_head(h).squeeze(-1), self.tableware_head(h)


class L3aCNNLarge(nn.Module):
    """L3a 满配版（IT1，准确率优先）：4 块 64→128→256→256，~218K 参数 / ~16M MACs。
    头部带 dropout 正则。"""
    def __init__(self, num_tableware=5, drop=0.2):
        super().__init__()
        self.stem = nn.Conv1d(N_CHANNELS, 64, 1, bias=False)
        self.bn0 = nn.BatchNorm1d(64)
        self.blocks = nn.Sequential(
            _SepBlock(64, 128, stride=2),
            _SepBlock(128, 256, stride=2),
            _SepBlock(256, 256, stride=2),
            _SepBlock(256, 256, stride=2),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU())
        self.eat_head = nn.Linear(64, 1)
        self.tableware_head = nn.Linear(64, num_tableware)
        self.num_tableware = num_tableware

    def forward(self, x):
        x = nn.functional.relu(self.bn0(self.stem(x)))
        x = self.blocks(x)
        x = x.mean(dim=2)
        h = self.head(x)
        return self.eat_head(h).squeeze(-1), self.tableware_head(h)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model, length=WINDOW_LEN) -> int:
    """逐层 MACs 估算（卷积=输出元素×核×输入通道；全连接=输出×输入）。"""
    total = 0
    L = length
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            out_len = (L + 2 * m.padding[0] - m.kernel_size[0]) // m.stride[0] + 1
            per_out = m.kernel_size[0] * (m.in_channels // m.groups)
            total += out_len * m.out_channels * per_out
            L = out_len
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
    return total
