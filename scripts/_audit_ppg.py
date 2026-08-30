# -*- coding: utf-8 -*-
"""PPG 44 通道审计：有效通道数、方差、时间分辨率、与 IMU 对齐情况。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config
from src.data import manifests, loader

idx = manifests.load_sensor_index()
sids = idx["session_id"].tolist()[:5]
for sid in sids:
    try:
        s = loader.load_session(sid)
    except Exception as e:
        print(sid, "ERR", e); continue
    ppg = s.ppg  # (44, N)
    N = ppg.shape[1]
    nz = (ppg != 0).sum(1) / N
    var = ppg.var(1)
    ok = (nz > 0.9) & (var > 1e-6)
    corr = np.corrcoef(ppg[:8]) if ok.sum() >= 8 else np.zeros((8, 8))
    t = s.t_ppg
    tv = t[t > 0]
    span_s = (tv.max() - tv.min()) / 1000 if len(tv) else 0
    fs = len(tv) / max(span_s, 1)
    ta = s.t_acc
    tva = ta[ta > 0]
    imu_fs = N / max((tva.max() - tva.min()) / 1000, 1) if len(tva) else 0
    # 时间戳对齐：PPG 有效行中 IMU 也有效的比例（同行数结构）
    both = ((s.imu_valid) & (s.ppg_valid)).mean()
    print(f"{sid}: N={N} 有效通道 {ok.sum()}/44 非零率中位 {np.median(nz):.2f} "
          f"方差中位 {np.median(var):.2e} PPG_fs≈{fs:.0f}Hz IMU_fs≈{imu_fs:.0f}Hz "
          f"同行双有效 {both:.2f}")
    print(f"  前8通道平均|corr|: {np.abs(corr[np.triu_indices(8,1)]).mean():.3f}")
