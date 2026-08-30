# -*- coding: utf-8 -*-
"""餐标签 vs 原始数据实际时间轴对齐审计（2026-08-30 发现的第三次口径问题）。

现象：会话标称 endTime 可达 20+ 小时，但原始 TSV 数据实际只有 ~45 分钟；
餐标注（before/after 落在标称会话窗内）可能完全落在原始数据之外 →
env 网格永远覆盖不到 → 覆盖统计/评估里的系统性假 FN。

产物：outputs/meal_alignment_audit.json + stdout 汇总。
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits, loader

OUT_AUDIT = config.OUTPUT_DIR / "meal_alignment_audit.json"


def _real_dur(sid):
    """会话原始数据实际时长（秒）；失败返回 -1。"""
    try:
        s = loader.load_session(sid)
        fs = s.meta.get("row_rate", 105.0)
        return s.acc.shape[1] / fs
    except Exception:
        return -1.0


def main():
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    starts = {r["session_id"]: int(r["timeStamp.startTime"]) for _, r in idx.iterrows()}
    ends = {r["session_id"]: int(r["timeStamp.endTime"]) for _, r in idx.iterrows()}
    folds = splits.load_folds()
    sids = sorted({s for f in folds for s in f["val_sessions"]})

    print(f"读取 {len(sids)} 个会话实际时长（8 并行）...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        durs = dict(zip(sids, ex.map(_real_dur, sids, chunksize=8)))
    print(f"完成 {time.time()-t0:.0f}s；异常会话 {(np.array(list(durs.values())) < 0).sum()} 个", flush=True)

    per_fold, detail = {}, []
    for k, f in enumerate(folds):
        n_m = n_i = n_p = n_o = 0
        for sid in f["val_sessions"]:
            dur = durs.get(sid, -1.0)
            ext = dict(zip(idx["session_id"], idx["externalid"]))[sid]
            for m in meal_meta.get(ext, []):
                if not (m["before"] >= starts[sid] and m["after"] <= ends[sid]):
                    continue
                n_m += 1
                if dur <= 0:
                    n_o += 1
                    continue
                real_end = starts[sid] + dur * 1000
                m0, m1 = m["before"], m["after"]
                if m1 <= real_end:
                    n_i += 1
                elif m0 < real_end:
                    n_p += 1
                else:
                    n_o += 1
        per_fold[k] = {"meals": n_m, "inside": n_i, "partial": n_p, "outside": n_o}
        print(f"fold{k}: 餐 {n_m} | 完全在数据内 {n_i} | 部分在 {n_p} | 完全在外 {n_o}", flush=True)

    tot = sum(v["meals"] for v in per_fold.values())
    tot_i = sum(v["inside"] for v in per_fold.values())
    tot_p = sum(v["partial"] for v in per_fold.values())
    print(f"\n合计 餐 {tot} | 数据内 {tot_i} ({tot_i/max(tot,1)*100:.0f}%) | 部分 {tot_p} | 外 {tot-tot_i-tot_p}", flush=True)

    # 每折数据内餐数（后续评估的 n_true 口径依据）
    for k, v in per_fold.items():
        print(f"  fold{k} 数据内餐（评估口径）: {v['inside']} (+部分 {v['partial']})", flush=True)

    (config.OUTPUT_DIR / "meal_alignment_audit.json").write_text(
        json.dumps({"per_fold": per_fold, "total_inside": tot_i}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"→ outputs/meal_alignment_audit.json", flush=True)


if __name__ == "__main__":
    main()
