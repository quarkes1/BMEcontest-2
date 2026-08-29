# -*- coding: utf-8 -*-
"""L3a ResNet-1D（端-云协同架构下扩容版，~1M+ 参数）：
stem(7,stride2) → 5×残差块 [64,128,256,256,256]，stride [2,2,2,1,1]
→ GAP → FC(256→128, dropout) → 双头（进食 BCE + 餐具辅助）。
输入 (B, 11, 525) 标准化后窗口；接口与 L3aCNN 一致（可直接替换训练脚本）。"""
import torch
import torch.nn as nn
from src.models.l3a_cnn import N_CHANNELS, WINDOW_LEN


class _ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.c1 = nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.c2 = nn.Conv1d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.shortcut = None
        if c_in != c_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out))
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        s = x if self.shortcut is None else self.shortcut(x)
        return self.act(h + s)


class L3aResNet(nn.Module):
    def __init__(self, num_tableware=5, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU())
        self.blocks = nn.Sequential(
            _ResBlock(64, 128, 2),
            _ResBlock(128, 256, 2),
            _ResBlock(256, 256, 2),
            _ResBlock(256, 256, 1),
            _ResBlock(256, 256, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(drop))
        self.eat_head = nn.Linear(128, 1)
        self.tableware_head = nn.Linear(128, num_tableware)
        self.num_tableware = num_tableware

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=2)                       # GAP
        h = self.head(x)
        return self.eat_head(h).squeeze(-1), self.tableware_head(h)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
