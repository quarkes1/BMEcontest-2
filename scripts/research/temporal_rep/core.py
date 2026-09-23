from __future__ import annotations

import numpy as np
import torch
from torch import nn


def robust_channel_stats(tokens: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tokens)[np.asarray(mask, bool)]
    center = np.median(values, axis=(0, 2))
    scale = np.median(np.abs(values - center[None, :, None]), axis=(0, 2)) * 1.4826
    return center.astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.ReLU(), nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.ReLU())

    def forward(self, x):
        return x + self.net(x)


class SmallTokenTCN(nn.Module):
    def __init__(self, channels: int = 24, embedding_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(6, channels, 5, padding=2), nn.ReLU(), ResidualBlock(channels, 1), ResidualBlock(channels, 2), ResidualBlock(channels, 4))
        self.project = nn.Linear(channels, embedding_dim)
        self.score = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        embedding = self.project(self.encoder(x).mean(-1))
        return self.score(torch.relu(embedding)).squeeze(-1), embedding


def topk_mean(scores: torch.Tensor, mask: torch.Tensor, fraction: float = .2) -> torch.Tensor:
    output=[]
    for row, valid in zip(scores, mask):
        values=row[valid]; k=max(1, int(np.ceil(len(values)*fraction)))
        output.append(torch.topk(values, k).values.mean())
    return torch.stack(output)
