# -*- coding: utf-8 -*-
"""MM-Ranker：候选窗口深度排序头。

架构：
- IMU 分支（动作特征）：6 通道 @10Hz → 卷积 → 膨胀 TCN 残差栈 →
  5s 块平均池化（2400 样本 → 48 块）——粗粒度动作节奏特征
- PPG 分支（生理特征，默认关闭）：22 通道 × 3 项块统计 (48, 66) → MLP
  ——低采样率调度采样由块统计吸收
- MA 置信度掩码：加速度能量块统计 → sigmoid 门控——运动伪影严重时对
  PPG 特征降权（可学习软掩码）
- 融合：concat(IMU 特征, PPG 特征 × 掩码) → 双向 GRU → 全局池化(均值+最大)
  ⊕ 元特征（候选时长/时刻先验/会话门控概率）→ MLP → 置信度 logit

训练配套（train_ranker.py）：Focal Loss（γ=2）+ 硬负样本挖掘（每轮对高置信
误判负样本提权）+ 早停与正则化。输入来自 build_candidate_windows.py 缓存。
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
        """x: (B, d, T) → (B, d, T)。因果：Conv1d 双边 padding 后裁掉右侧 (k-1)·d。"""
        out = self.conv1(x)
        out = torch.relu(self.bn1(out))
        out = self.drop(out)
        out = self.conv2(out)
        out = torch.relu(self.bn2(out))
        T = x.size(2)
        if out.size(2) > T:
            out = out[:, :, :T]
        return x + out


class MMRanker(nn.Module):
    def __init__(self, d_model=64, n_layers=6, dropout=0.3, n_ppg=66, n_blocks=48, use_ppg=True,
                 n_imu=6):
        super().__init__()
        self.use_ppg = use_ppg
        # ---- IMU 分支 ----
        self.imu_in = nn.Sequential(
            nn.Conv1d(n_imu, d_model, 7, padding=3), nn.BatchNorm1d(d_model), nn.ReLU())
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
        # 元特征（候选时长/时刻先验/会话门控概率）——长候选误报的关键判别信号。
        # 会话门控概率是判别力最强的信号之一（会话级 AUC≈0.88）：低门控会话的
        # 候选应被排序头压制。注意训练/推理两侧须使用同一套门控打分（对全体
        # 会话统一打分），否则分布不一致会损害排序头。
        self.meta_in = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(4 * d_model + 32, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1))
        self.n_blocks = n_blocks

    def forward(self, imu, ppg, ma, meta):
        """imu: (B, 2400, 6) fp16→float；ppg: (B, 48, 66)；ma: (B, 48, 2)；meta: (B, 3)。"""
        x = imu.transpose(1, 2).float()                 # (B, 6, 2400)
        x = self.imu_in(x)
        for blk in self.tcn:
            x = blk(x)
        # 5s 块平均池：2400 → 48
        B = x.size(0)
        h_imu = x.view(B, -1, self.n_blocks, x.size(2) // self.n_blocks).mean(3)  # (B, d, 48)
        h_imu = h_imu.transpose(1, 2)                   # (B, 48, d)

        if self.use_ppg:
            h_ppg = self.ppg_in(ppg)                    # (B, 48, d)
            h_ppg = self.ppg_bn(h_ppg.transpose(1, 2)).transpose(1, 2)
            gate = self.ma_gate(ma)                     # (B, 48, 1) 运动伪影低→门控大
            h_ppg = h_ppg * gate                        # 掩码降权
        else:
            h_ppg = torch.zeros_like(h_imu)             # PPG 消融（结构不变）

        z = torch.cat([h_imu, h_ppg], dim=2)            # (B, 48, 2d)
        z, _ = self.fuse(z)                             # (B, 48, 2d)
        hm = self.meta_in(meta)                         # (B, 32)
        pooled = torch.cat([z.mean(1), z.max(1).values, hm], dim=1)   # (B, 4d+32)
        return self.head(pooled).squeeze(1)             # (B,)


def focal_loss(logits, y, gamma=2.0, alpha=0.25, weights=None):
    """Focal Loss：类别不平衡下的加权损失。y: {0,1} float。"""
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    w = alpha * y + (1 - alpha) * (1 - y)              # α 平衡
    loss = w * (1 - pt) ** gamma * ce
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def asymmetric_loss(logits, y, gamma_pos=1.0, gamma_neg=3.0, alpha=0.75, weights=None):
    """Asymmetric Loss（Ridnik et al. 2021）：正/负样本不同 γ 降权。

    γ_pos 小 → 正样本（真餐候选）几乎不降权、梯度大 → 强制 TP 输出更高置信度
    （解决真餐深度分偏低）；γ_neg 大 → 负样本重降权（难负样本挖掘自然完成）。
    y: {0,1} float。"""
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    gamma = gamma_pos * y + gamma_neg * (1 - y)        # 逐样本 γ（正 1.0 / 负 3.0）
    w = alpha * y + (1 - alpha) * (1 - y)              # α 平衡（正样本提权）
    loss = w * (1 - pt) ** gamma * ce
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
