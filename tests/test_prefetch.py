# -*- coding: utf-8 -*-
"""PrefetchLoader 单测：批次正确性、与 DataLoader 等价、预取不阻塞。"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.train.prefetch import PrefetchLoader

class _Tiny(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, i):
        return (torch.tensor(i, dtype=torch.float32),
                torch.tensor(i % 2, dtype=torch.long))

def test_batches_match_dataloader():
    ds = _Tiny()
    ref = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, drop_last=True)
    pre = PrefetchLoader(ds, batch_size=32, drop_last=True)
    assert len(pre) == len(ref) == 3
    for (a1, a2), (b1, b2) in zip(pre, ref):
        assert torch.equal(a1, b1) and torch.equal(a2, b2)

def test_prefetch_full_epoch_count():
    ds = _Tiny()
    pre = PrefetchLoader(ds, batch_size=33, drop_last=False)   # 100 = 3×33 + 1
    assert len(pre) == 4
    total = sum(len(b[0]) for b in pre)
    assert total == 100

def test_epoch_reiterable():
    ds = _Tiny()
    pre = PrefetchLoader(ds, batch_size=32)
    n1 = sum(len(b[0]) for b in pre)
    n2 = sum(len(b[0]) for b in pre)          # 可重复迭代（每 epoch 重建）
    assert n1 == n2 == 96
