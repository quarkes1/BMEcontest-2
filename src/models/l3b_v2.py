# -*- coding: utf-8 -*-
"""L3b v2（端-云协同扩容版）：PPG 波形嵌入（44→64→96→128）+ 双向 GRU + 自注意力。
输入/输出接口与 L3bPPGNN 一致：(B, seq, 44, 720) + (B, seq, 8) → (B, seq) logits。
HRV 特征需按会话 z-score（缓存构建时完成）。~0.9M 参数。"""
import torch
import torch.nn as nn
from src.models.l3b_ppgnn import (N_PPG_CHANNELS, PPG_WINDOW_ROWS, HRV_DIMS)

EMBED_DIM = 128
GRU_DIM = 128


class L3bBiGRU(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, gru_dim=GRU_DIM):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(N_PPG_CHANNELS, 64, 9, stride=3, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 96, 9, stride=3, padding=4, bias=False),
            nn.BatchNorm1d(96), nn.ReLU(),
            nn.Conv1d(96, embed_dim, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(embed_dim + HRV_DIMS, gru_dim, batch_first=True,
                          bidirectional=True)
        self.attn = nn.Linear(2 * gru_dim, 1)          # 加性注意力打分
        self.head = nn.Sequential(
            nn.Linear(2 * gru_dim, 96), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(96, 1))

    def forward(self, x, feat):
        """x: (B, seq, 44, 720) fp32，feat: (B, seq, 8)。返回 (B, seq) logits。"""
        B, S = x.shape[:2]
        emb = self.embed(x.reshape(B * S, N_PPG_CHANNELS, PPG_WINDOW_ROWS))
        emb = emb.mean(dim=2).view(B, S, -1)            # GAP → (B, seq, 128)
        out, _ = self.gru(torch.cat([emb, feat], dim=-1))       # (B, seq, 256)
        w = torch.softmax(self.attn(out), dim=1)                # 时间维注意力
        ctx = (w * out).sum(dim=1, keepdim=True)                # (B, 1, 256)
        h = out + ctx                                         # 残差融合上下文
        return self.head(h).squeeze(-1)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
