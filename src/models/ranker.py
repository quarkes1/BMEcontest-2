# -*- coding: utf-8 -*-
"""MM-Ranker：多模态分层候选排序头（L2 细粒度二次校验分类器）。

架构（方向 1 分层融合 + 方向 2 TCN）：
- IMU 分支（高频手势）：6ch @10Hz → Conv1 → 膨胀 TCN 残差栈 → h_imu (T=2400, d)
  → 5s 块平均池 → (48, d)  —— 粗粒度动作节奏特征
- PPG 分支（低频生理）：22ch×3 块统计 (48, 66) → MLP → h_ppg (48, d)
  —— 心率/灌注调制；调度采样天然由块统计吸收
- MA 置信度掩码（方向 1）：acc 能量块统计 (48, 2) → MLP → sigmoid → gate (48, 1)
  —— 运动伪影严重时 PPG 特征降权（可学习，等效软掩码）
- 融合：concat(h_imu, h_ppg × gate) → (48, 2d) → Bi-GRU → 全局池(mean+max) → MLP → logit

训练配套（train_ranker.py）：Focal Loss（γ=2）+ 硬负样本挖掘（每 epoch 前向取
top-k 高置信度误判负样本 → 下 epoch 提权）+ 强正则（dropout/weight decay/早停）。
输入来自 build_candidate_windows.py 缓存（imu/ppg/ma/meta）。
"""
import math

import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """膨胀因果卷积残差块（核 5，BN + ReLU 两叠）。"""

    def __init__(self, d, dilation, dropout=0.3):
        super().__init__()
        pad = (5 - 1) * dilation
        self.conv1 = nn.Conv1d(d, d, 5, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(d)
        self.conv2 = nn.Conv1d(d, d, 5, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(dropout)
        self.dilation = dilation

    def forward(self, x):
        """x: (B, d, T) → (B, d, T)。因果：卷积后裁掉未来 pad。"""
        out = self.conv1(x)
        if self.dilation > 1:
            out = out[:, :, :-self.dilation]
        out = torch.relu(self.bn1(out))
        out = self.drop(out)
        out = self.conv2(out)
        if self.dilation > 1:
            out = out[:, :, :-self.dilation]
        out = torch.relu(self.bn2(out))
        return x + out


class MMRanker(nn.Module):
    def __init__(self, d_model=64, n_layers=6, dropout=0.3, n_ppg=66, n_blocks=48):
        super().__init__()
        # ---- IMU 分支 ----
        self.imu_in = nn.Sequential(
            nn.Conv1d(6, d_model, 7, padding=3), nn.BatchNorm1d(d_model), nn.ReLU())
        self.tcn = nn.ModuleList(
            [TCNBlock(d_model, dilation=2 ** i, dropout=dropout) for i in range(n_layers)])
        # ---- PPG 分支 ----
        self.ppg_in = nn.Sequential(
            nn.Linear(n_ppg, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.ppg_bn = nn.BatchNorm1d(d_model)
        # ---- MA 置信度掩码（acc 能量 → sigmoid 门控）----
        self.ma_gate = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        # ---- 融合 ----
        self.fuse = nn.GRU(2 * d_model, d_model, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(4 * d_model, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1))
        self.n_blocks = n_blocks

    def forward(self, imu, ppg, ma):
        """imu: (B, 2400, 6) fp16→float；ppg: (B, 48, 66)；ma: (B, 48, 2)。"""
        x = imu.transpose(1, 2).float()                 # (B, 6, 2400)
        x = self.imu_in(x)
        for blk in self.tcn:
            x = blk(x)
        # 5s 块平均池：2400 → 48
        B = x.size(0)
        h_imu = x.view(B, -1, self.n_blocks, x.size(2) // self.n_blocks).mean(3)  # (B, d, 48)
        h_imu = h_imu.transpose(1, 2)                   # (B, 48, d)

        h_ppg = self.ppg_in(ppg)                        # (B, 48, d)
        h_ppg = self.ppg_bn(h_ppg.transpose(1, 2)).transpose(1, 2)

        gate = self.ma_gate(ma)                         # (B, 48, 1) 运动伪影低→门控大
        h_ppg = h_ppg * gate                            # 掩码降权

        z = torch.cat([h_imu, h_ppg], dim=2)            # (B, 48, 2d)
        z, _ = self.fuse(z)                             # (B, 48, 2d)
        pooled = torch.cat([z.mean(1), z.max(1).values], dim=1)   # (B, 4d)
        return self.head(pooled).squeeze(1)             # (B,)


def focal_loss(logits, y, gamma=2.0, alpha=0.25, weights=None):
    """Focal Loss（方向 4）：类别不平衡直接优化。y: {0,1} float。"""
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    w = alpha * y + (1 - alpha) * (1 - y)              # α 平衡
    loss = w * (1 - pt) ** gamma * ce
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
