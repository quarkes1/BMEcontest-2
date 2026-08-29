# -*- coding: utf-8 -*-
"""构建 L3a 原始窗口缓存（按折）：cache/l3a_raw/fold{k}/{session_id}.npz
- 正样本：dominant 场景餐的窗口（步长 1s，全保留）
- 负样本：无餐窗口（步长 5s，比例自然 ≈3:1）
- X: n×11×525 float32（build_raw_channels），y/scene/tw/t0/t1；每折 stats.json 存通道均值/方差
运行：conda activate bme && python scripts/build_raw_cache.py --folds 0,1（8 进程）"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.data.loader import load_session, detect_binary, _find_collect_data
from src.data.windows import iter_window_labels
from src.models.l3a_cnn import build_raw_channels, WINDOW_LEN

BASE = config.CACHE_DIR / "l3a_raw"
NEG_STRIDE_ROWS = 5 * config.STRIDE_ROWS      # 负样本 5s 步长

def _match_meal(t0, t1, meals):
    dur = t1 - t0
    for m in meals:
        ov = min(t1, m["after"]) - max(t0, m["before"])
        if ov > 0 and ov / dur >= config.IOU_POS:
            return m
    return None

def _build(args):
    session_id, meal_list, out_dir = args
    out = out_dir / f"{session_id}.npz"
    if out.exists():
        return ("skip", session_id, 0, 0)
    try:
        d = config.SENSOR_DIR / session_id
        txt = _find_collect_data(str(d))
        if detect_binary(txt):
            return ("binary", session_id, 0, 0)
        s = load_session(session_id)
        pairs = [(m["before"], m["after"]) for m in meal_list]
        dom_pairs = [(m["before"], m["after"]) for m in meal_list if m["scene"] == "dominant"]
        Xs, ys, scs, tws, t0s, t1s = [], [], [], [], [], []
        # 正样本：dominant 餐窗口，步长 1s
        for w in iter_window_labels(s, pairs):
            if w["label"] != 1:
                continue
            m = _match_meal(w["t0_ms"], w["t1_ms"], meal_list)
            if m is None or m["scene"] != "dominant":
                continue
            Xs.append(build_raw_channels(s, w["start_row"], w["end_row"]))
            ys.append(1); scs.append(0); tws.append(m["tableware"])
            t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
        # 负样本：无餐窗口，步长 5s
        for w in iter_window_labels(s, dom_pairs, stride_rows=NEG_STRIDE_ROWS):
            if w["label"] != 0:
                continue
            Xs.append(build_raw_channels(s, w["start_row"], w["end_row"]))
            ys.append(0); scs.append(-1); tws.append(-1)
            t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
        if Xs:
            np.savez(out, X=np.stack(Xs),
                                y=np.array(ys, dtype=np.int8),
                                scene=np.array(scs, dtype=np.int8),
                                tw=np.array(tws, dtype=np.int8),
                                t0=np.array(t0s, dtype=np.int64), t1=np.array(t1s, dtype=np.int64))
        return ("ok", session_id, sum(ys), len(ys) - sum(ys))
    except Exception as e:
        return ("error", f"{session_id}: {type(e).__name__}: {e}", 0, 0)

def build_fold(fold, workers=8):
    f = splits.load_folds()[fold]
    out_dir = BASE / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index()
    idx = index.set_index("session_id")
    tasks = []
    for sid in f["train_sessions"]:
        if sid not in idx.index:
            continue
        tasks.append((sid, meal_meta.get(idx.loc[sid, "externalid"], []), out_dir))
    print(f"fold {fold}: {len(tasks)} 训练会话", flush=True)
    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "binary": 0, "error": [], "pos": 0, "neg": 0}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (status, info, npos, nneg) in enumerate(ex.map(_build, tasks, chunksize=4)):
            if status == "error":
                stats["error"].append(info)
            else:
                stats[status] = stats[status] + 1
                stats["pos"] += npos; stats["neg"] += nneg
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} 用时 {time.time()-t0:.0f}s", flush=True)
    # 通道均值/方差（鲁棒：逐窗统计 + 中位数池化。
    # 教训：E[x²]-E[x]² 在大均值通道上灾难性消减产生 std=0 → 标准化爆 fp16 inf（2026-08-30 实锤））
    wm_all, wv_all = [], []
    for f in out_dir.glob("*.npz"):
        X = np.load(f)["X"].astype(np.float32)          # (n, 11, 525)
        wm_all.append(X.mean(axis=2))
        wv_all.append(X.var(axis=2))
    Wm = np.concatenate(wm_all); Wv = np.concatenate(wv_all)
    mean = np.median(Wm, axis=0)
    var = np.median(Wv, axis=0) + np.median((Wm - mean) ** 2, axis=0)
    std = np.sqrt(np.clip(var, 0, None)).astype(np.float32)
    (out_dir / "stats.json").write_text(json.dumps(
        {"mean": mean.tolist(), "std": std.tolist(), "n_windows": n, **stats},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"fold {fold} 完成: ok={stats['ok']} pos={stats['pos']} neg={stats['neg']} "
          f"error={len(stats['error'])} 用时 {time.time()-t0:.0f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for k in [int(x) for x in args.folds.split(",")]:
        build_fold(k, args.workers)

if __name__ == "__main__":
    main()
