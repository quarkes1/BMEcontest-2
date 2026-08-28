# -*- coding: utf-8 -*-
"""构建 L3b PPG 窗口缓存（按折）：cache/l3b_raw/fold{k}/{session_id}.npz
- 正样本：nondominant 场景餐窗口（30s 窗 / 10s 步长）
- 负样本：无餐窗口（30s 窗 / 60s 步长）
- X: n×44×720 fp16（去噪后），hrv: n×8，y/scene/tw/t0/t1；不压缩（W2 教训）
运行：conda activate bme && python scripts/build_ppg_cache.py --folds 0,1"""
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
from src.models.l3b_ppgnn import build_ppg_window

BASE = config.CACHE_DIR / "l3b_raw"
WIN_ROWS = 3150          # 30s × 105 行/s
POS_STRIDE = 1050        # 10s
NEG_STRIDE = 6300        # 60s

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
        Xs, hs, ys, scs, tws, t0s, t1s = [], [], [], [], [], [], []
        for w in iter_window_labels(s, pairs, window_rows=WIN_ROWS, stride_rows=POS_STRIDE):
            if w["label"] != 1:
                continue
            m = _match_meal(w["t0_ms"], w["t1_ms"], meal_list)
            if m is None or m["scene"] != "nondominant":
                continue
            X, h = build_ppg_window(s, w["start_row"], w["end_row"])
            Xs.append(X); hs.append(h)
            ys.append(1); scs.append(1); tws.append(m["tableware"])
            t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
        for w in iter_window_labels(s, pairs, window_rows=WIN_ROWS, stride_rows=NEG_STRIDE):
            if w["label"] != 0:
                continue
            X, h = build_ppg_window(s, w["start_row"], w["end_row"])
            Xs.append(X); hs.append(h)
            ys.append(0); scs.append(-1); tws.append(-1)
            t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
        if Xs:
            np.savez(out, X=np.stack(Xs), hrv=np.stack(hs),
                     y=np.array(ys, dtype=np.int8), scene=np.array(scs, dtype=np.int8),
                     tw=np.array(tws, dtype=np.int8),
                     t0=np.array(t0s, dtype=np.int64), t1=np.array(t1s, dtype=np.int64))
        return ("ok", session_id, sum(ys), len(ys) - sum(ys))
    except Exception as e:
        return ("error", f"{session_id}: {type(e).__name__}: {e}", 0, 0)

def build_fold(fold):
    f = splits.load_folds()[fold]
    out_dir = BASE / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index().set_index("session_id")
    tasks = [(sid, meal_meta.get(index.loc[sid, "externalid"], []), out_dir)
             for sid in f["train_sessions"] if sid in index.index]
    print(f"fold {fold}: {len(tasks)} 训练会话", flush=True)
    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "binary": 0, "error": [], "pos": 0, "neg": 0}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (status, info, npos, nneg) in enumerate(ex.map(_build, tasks, chunksize=4)):
            if status == "error":
                stats["error"].append(info)
            else:
                stats[status] = stats[status] + 1
                stats["pos"] += npos; stats["neg"] += nneg
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} 用时 {time.time()-t0:.0f}s", flush=True)
    (out_dir / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"fold {fold} 完成: ok={stats['ok']} skip={stats['skip']} pos={stats['pos']} "
          f"neg={stats['neg']} error={len(stats['error'])} 用时 {time.time()-t0:.0f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1")
    args = ap.parse_args()
    for k in [int(x) for x in args.folds.split(",")]:
        build_fold(k)

if __name__ == "__main__":
    main()
