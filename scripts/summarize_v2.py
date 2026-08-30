# -*- coding: utf-8 -*-
"""5 折汇总：v2（MM-Ranker 分层解码）vs v1（LGBM 基线）——受试者级 F1。

- v2 best：每折 validation 网格最优
- v2 fixed：全部 5 折共用 fold0 best 配置（诚实口径，动态从 fold0 json 取）
- v1：rank_events.py 基线（best + 固定 g0.7_k2_p0.3）
- 双网格对比：`python scripts/summarize_v2.py [15m|60m ...]`，默认扫描盘上已有网格

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


def load(k, grid):
    p = OUT / f"rank_events_v2_fold{k}_{grid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_v1(k):
    p = OUT / f"rank_events_fold{k}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def row_f1(rows, name):
    for r in rows:
        if r["name"] == name:
            return r["f1"]
    return None


def summarize(grid):
    r0 = load(0, grid)
    v2_fixed = r0["best"]["name"] if r0 else None  # fold0 best（诚实口径，跨折固定）
    print(f"\n===== 网格 {grid}（v2 fixed = fold0 best「{v2_fixed}」）=====")
    if v2_fixed is None:
        print("  fold0 缺失，无法定 fixed 配置")
        return None
    print(f"{'fold':>5} | {'v1 best':>8} {'v1 fixed':>9} | {'v2 best':>8} {'v2 fixed':>9} | Δbest")
    v1_b, v1_f, v2_b, v2_f = [], [], [], []
    for k in range(5):
        r2, r1 = load(k, grid), load_v1(k)
        if r2 is None or r1 is None:
            print(f"{k:>5} | 缺失 v2={r2 is not None} v1={r1 is not None}")
            continue
        b2 = r2["best"]["f1"]
        f2_ = row_f1(r2["rows"], v2_fixed)
        b1 = r1["best"]["f1"]
        f1_ = row_f1(r1["rows"], V1_FIXED)
        v2_b.append(b2)
        v2_f.append(f2_ if f2_ is not None else np.nan)
        v1_b.append(b1)
        v1_f.append(f1_ if f1_ is not None else np.nan)
        print(f"{k:>5} | {b1:8.3f} {f1_:9.3f} | {b2:8.3f} {f2_:9.3f} | {b2 - b1:+.3f}")
    if not v2_b:
        return None
    print("-" * 52)
    print(f"{'mean':>5} | {np.nanmean(v1_b):8.3f} {np.nanmean(v1_f):9.3f} | {np.nanmean(v2_b):8.3f} "
          f"{np.nanmean(v2_f):9.3f} | {np.nanmean(v2_b) - np.nanmean(v1_b):+.3f}")
    return {"grid": grid, "v2_best_mean": float(np.nanmean(v2_b)),
            "v2_fixed_mean": float(np.nanmean(v2_f)), "config": v2_fixed}


def main():
    grids = (sys.argv[1:]
             or sorted(p.name[len("rank_events_v2_fold0_"):-len(".json")]
                       for p in OUT.glob("rank_events_v2_fold0_*.json")))
    results = [r for r in (summarize(g) for g in grids) if r]
    if len(results) > 1:
        print("\n===== 网格对比（v2 best 均值 / fixed 均值）=====")
        for r in results:
            print(f"  {r['grid']:>5}: best {r['v2_best_mean']:.3f}  fixed {r['v2_fixed_mean']:.3f}  ({r['config']})")
        best = max(results, key=lambda r: r["v2_best_mean"])
        print(f"\n★ 推荐网格: {best['grid']}（v2 best 均值 {best['v2_best_mean']:.3f}）")
    print("\n目标 F1 ≥ 0.50")


if __name__ == "__main__":
    main()
