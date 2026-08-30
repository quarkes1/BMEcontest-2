# -*- coding: utf-8 -*-
"""PPG 深度审计 v2：有效通道分布、调度采样率、与 IMU 对齐、MA 相关性。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config
from src.data import manifests, loader

idx = manifests.load_sensor_index()
sids = idx["session_id"].tolist()
rng = np.random.RandomState(0)
sample = [sids[i] for i in rng.choice(len(sids), 12, replace=False)]
nz_all = []; ok_all = []; ppg_ratio_all = []; fs_ppg_all = []; n22_all = []
for sid in sample:
    try:
        s = loader.load_session(sid)
    except Exception:
        continue
    ppg = s.ppg
    N = ppg.shape[1]
    nz = (ppg != 0).sum(1) / N
    nz_all.append(nz)
    var = ppg.var(1)
    ok = (nz > 0.9) & (var > 1e-6)
    ok_all.append(ok)
    n22_all.append((nz[:22] > 0.9).sum())
    t = s.t_ppg
    tv = t[t > 0]
    span_s = (tv.max() - tv.min()) / 1000 if len(tv) else 0
    fs = len(tv) / max(span_s, 1)
    fs_ppg_all.append(fs)
    ppg_ratio_all.append(s.ppg_valid.mean())
    print(f"{sid[-8:]}: 有效ch={ok.sum():2d}/44 前22ch有效={(nz[:22]>0.9).sum():2d} "
          f"ppg行有效={s.ppg_valid.mean():.2f} PPG_fs={fs:.0f}Hz", flush=True)
nz = np.array(nz_all)
print("\n通道非零率矩阵 (12会话 × 44ch) 概要:")
print("  前22ch: min", nz[:, :22].min(1).round(2), " med", np.median(nz[:, :22], 1).round(2))
print("  后22ch: max", nz[:, 22:].max(1).round(2))
print(f"  前22ch每通道非零率中位数: {np.median(nz[:, :22], 0).round(2)}")
print(f"  有效通道数分布(>0.9): {np.array([o.sum() for o in ok_all])}")
print(f"  PPG行有效比: {np.array(ppg_ratio_all).round(2)}")
print(f"  PPG采样率: {np.array(fs_ppg_all).round(0)}")
