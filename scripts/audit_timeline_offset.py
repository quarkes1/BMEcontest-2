# -*- coding: utf-8 -*-
"""时间轴偏移审计：每会话 TSV 首行 ACC_TIME vs manifest startTime（分钟）。

背景：validate_baselines 的 env 网格用 startTime + 行索引×1s，未用 TSV 真实时间戳。
实测部分会话偏移 60min/3min → env 与餐标签错位 → 覆盖统计与评估系统性失真。
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

import src.config as config
from src.data import manifests, loader


def _first_ts(sid):
    try:
        f = loader._find_collect_data(str(config.SENSOR_DIR / sid))
        with open(f, encoding="utf-8", errors="replace") as fh:
            fh.readline()
            ts = []
            for line in fh:
                parts = line.split("\t")
                if len(parts) < 53:
                    continue
                try:
                    v = int(parts[0])
                except ValueError:
                    continue
                if v > 0:
                    ts.append(v)
                if len(ts) >= 3:
                    break
        return ts[0] if ts else None
    except Exception:
        return None


def main():
    idx = manifests.load_sensor_index()
    starts = {r["session_id"]: int(r["timeStamp.startTime"]) for _, r in idx.iterrows()}
    sids = idx["session_id"].tolist()
    print(f"读取 {len(sids)} 个会话首行时间戳...", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        fts = dict(zip(sids, ex.map(_first_ts, sids, chunksize=16)))

    off, bad = [], 0
    for sid in sids:
        v = fts.get(sid)
        if v is None:
            bad += 1
            continue
        off.append((starts[sid] - v) / 60000.0)
    off = np.array(off)
    print(f"会话 {len(sids)}（读取失败 {bad}）", flush=True)
    print(f"偏移分钟: 中位 {np.median(off):.1f} | p25 {np.percentile(off, 25):.1f} | "
          f"p75 {np.percentile(off, 75):.1f} | max|.| {np.abs(off).max():.1f}", flush=True)
    print(f"|偏移|>2min: {(np.abs(off) > 2).sum()} ({(np.abs(off) > 2).mean() * 100:.0f}%) "
          f"| >10min: {(np.abs(off) > 10).sum()} | >30min: {(np.abs(off) > 30).sum()}", flush=True)
    hist, edges = np.histogram(np.clip(off, -70, 70), bins=14, range=(-70, 70))
    for c, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"  [{e0:5.0f},{e1:5.0f}]min: {'#' * c} {c}", flush=True)
    np.save(str(config.OUTPUT_DIR / "_ts_offset_min.npy"), off)


if __name__ == "__main__":
    main()
