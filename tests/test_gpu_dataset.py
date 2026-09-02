# -*- coding: utf-8 -*-
"""GPU 驻留数据集单测（有 CUDA 时运行；无则跳过）。"""
import json
import numpy as np
import pytest
import torch
from scripts.train_l3a import RawDataset
from src.train.prefetch import PrefetchLoader

torch_installed_cuda = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not torch_installed_cuda, reason="no CUDA")

def _tiny_cache(tmp_path):
    out = tmp_path / "tiny"
    out.mkdir()
    rng = np.random.RandomState(0)
    for i in range(2):
        n = 64
        np.savez(out / f"s{i}.npz",
                 X=rng.randn(n, 11, 525).astype(np.float32),
                 y=(rng.rand(n) < 0.3).astype(np.int8),
                 tw=np.full(n, -1, dtype=np.int8),
                 t0=np.arange(n) * 1000, t1=np.arange(n) * 1000 + 5000)
    stats = {"mean": [0.0] * 11, "std": [1.0] * 11,
             "pos": 19, "neg": 109, "n_windows": 128}
    (out / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return out, stats

@requires_cuda
def test_gpu_dataset_build_and_getitem(tmp_path):
    out, stats = _tiny_cache(tmp_path)
    ds = RawDataset(out, stats, on_gpu=True)
    assert hasattr(ds, "X_gpu") and not hasattr(ds, "X")
    X, y, tw = ds[0]
    assert X.device.type == "cuda" and X.dtype == torch.float16
    assert X.shape == (11, 525)
    assert y.device.type == "cuda"

@requires_cuda
def test_augment_batch_shapes_and_finite(tmp_path):
    out, stats = _tiny_cache(tmp_path)
    ds = RawDataset(out, stats, on_gpu=True, mirror=True, jitter=True,
                    stretch=True, ch_drop=0.1)
    X = torch.randn(8, 11, 525, device="cuda")
    Y = ds.augment_batch(X)
    assert Y.shape == X.shape
    assert torch.isfinite(Y).all()

@requires_cuda
def test_prefetch_with_gpu_dataset(tmp_path):
    out, stats = _tiny_cache(tmp_path)
    ds = RawDataset(out, stats, on_gpu=True)
    ds.reshuffle(42)
    dl = PrefetchLoader(ds, batch_size=16, drop_last=True)
    total = 0
    for xb, yb, twb in dl:
        assert xb.device.type == "cuda"
        total += len(xb)
    assert total == len(ds) // 16 * 16

@requires_cuda
def test_gpu_memory_freed_after_delete(tmp_path):
    out, stats = _tiny_cache(tmp_path)
    ds = RawDataset(out, stats, on_gpu=True)
    assert not hasattr(ds, "X")                     # numpy 版已释放，数据只在显存
