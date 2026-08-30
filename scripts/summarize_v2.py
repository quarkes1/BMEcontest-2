# -*- coding: utf-8 -*-
"""5 折汇总：v2（MM-Ranker 分层解码）vs v1（LGBM 基线）——受试者级 F1。

- v2：每折 best（validation 网格最优）均值
- v1：每折 best + 固定配置 g0.7_k2_p0.3（诚实口径）
- v2 固定配置：全部 5 折共用 w1.0_t0.3_g0.3_p0.3_d60（fold0 最优）的诚实口径

运行：source activate bme && python scripts/summarize_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config

OUT = config.OUTPUT_DIR

V1_FIXED = "g0.7_k2_p0.3"
V2_FIXED = "w1.0_t0.3_g0.3_p0.3_d60"   # fold0 验证网格最优配置（跨折固定验证诚实口径）


def load(k, tag):
    p = OUT / f"rank_events{tag}_fold{k}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def row_f1(rows, name):
    for r in rows:
        if r["name"] == name:
            return r["f1"]
    return None


print(f"{'fold':>5} | {'v1 best':>8} {'v1 fixed':>9} | {'v2 best':>8} {'v2 fixed':>9} | Δbest")
v1_b, v1_f, v2_b, v2_f = [], [], [], []
for k in range(5):
    r1, r2 = load(k, ""), load(k, "_v2")
    if r1 is None or r2 is None:
        print(f"{k:>5} | 缺失 r1={r1 is not None} r2={r2 is not None}")
        continue
    b1, b2 = r1["best"]["f1"], r2["best"]["f1"]
    f1_, f2_ = row_f1(r1["rows"], V1_FIXED), row_f1(r2["rows"], V2_FIXED)
    v1_b.append(b1); v1_f.append(f1_); v2_b.append(b2); v2_f.append(f2_)
    print(f"{k:>5} | {b1:8.3f} {f1_:9.3f} | {b2:8.3f} {f2_:9.3f} | {b2 - b1:+.3f}")
print("-" * 52)
print(f"{'mean':>5} | {np.mean(v1_b):8.3f} {np.mean(v1_f):9.3f} | {np.mean(v2_b):8.3f} "
      f"{np.mean(v2_f):9.3f} | {np.mean(v2_b) - np.mean(v1_b):+.3f}")
print(f"\nv2 best 配置：", [load(k, '_v2')["best"]["name"] for k in range(5)])
print(f"目标 F1 ≥ 0.50；当前 v2 均值（每折最优）={np.mean(v2_b):.3f}")
