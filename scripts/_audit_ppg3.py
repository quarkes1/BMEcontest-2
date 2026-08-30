# -*- coding: utf-8 -*-
"""PPG v3：确认前22通道在有效行内的质量（值域/方差/通道相关）。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config
from src.data import manifests, loader

idx = manifests.load_sensor_index()
sids = idx["session_id"].tolist()
rng = np.random.RandomState(1)
sample = [sids[i] for i in rng.choice(len(sids), 8, replace=False)]
for sid in sample:
    try:
        s = loader.load_session(sid)
    except Exception:
        continue
    ppg = s.ppg[:22]  # 前22通道
    m = ppg[:, s.ppg_valid]   # 只取有效行
    print(f"{sid[-8:]}: 有效行 {m.shape[1]} | 值域 [{m.min():.0f},{m.max():.0f}] "
          f"均值 {m.mean():.0f} 通道std中位 {np.median(m.std(1)):.0f}", flush=True)
    # 行级：有效行中前22ch是否全非零
    full = (m != 0).all(0).mean()
    print(f"  有效行中 22ch全非零 占比 {full:.2f} | 前4通道相关矩阵均|r|="
          f"{np.abs(np.corrcoef(m[:4])[np.triu_indices(4,1)]).mean():.3f}", flush=True)
