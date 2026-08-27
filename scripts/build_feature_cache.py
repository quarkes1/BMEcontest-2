# -*- coding: utf-8 -*-
"""并行构建基线窗口特征缓存：cache/baseline_features/{session_id}.npz
（X: n×37, y: n, t0_ms/t1_ms: n；label=-1 表示灰区，训练时过滤）
运行：conda activate bme && python scripts/build_feature_cache.py（8 进程，预计 20-40 分钟）"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import src.config as config
from src.data import manifests
from src.data.loader import load_session, detect_binary, _find_collect_data
from src.data.windows import iter_window_labels
from src.features.baseline_features import window_features

CACHE_DIR = config.CACHE_DIR / "baseline_features"

def build_one(session_id, meal_list):
    out = CACHE_DIR / f"{session_id}.npz"
    if out.exists():
        return ("skip", session_id)
    try:
        d = config.SENSOR_DIR / session_id
        txt = _find_collect_data(str(d))
        if detect_binary(txt):
            return ("binary", session_id)
        s = load_session(session_id)
        feats, labels, t0s, t1s = [], [], [], []
        for w in iter_window_labels(s, meal_list):
            feats.append(window_features(s, w["start_row"], w["end_row"]))
            labels.append(w["label"]); t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
        if feats:
            np.savez_compressed(out, X=np.vstack(feats), y=np.array(labels, dtype=np.int8),
                                t0=np.array(t0s, dtype=np.int64), t1=np.array(t1s, dtype=np.int64))
        return ("ok", session_id)
    except Exception as e:
        return ("error", f"{session_id}: {type(e).__name__}: {e}")

def _worker(args):
    return build_one(*args)

def main():
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = manifests.load_sensor_index()
    meals = manifests.load_meals()
    # session -> 该受试者用餐区间列表
    ext_meals = {ext: g[["before_ms", "after_ms"]].to_numpy().tolist()
                 for ext, g in meals.groupby("externalid")}
    tasks = [(sid, ext_meals.get(ext, [])) for sid, ext in
             index[["session_id", "externalid"]].itertuples(index=False)]
    print(f"构建特征缓存: {len(tasks)} 会话, 8 进程")
    stats = {"ok": 0, "skip": 0, "binary": 0, "error": []}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (status, info) in enumerate(ex.map(_worker, tasks, chunksize=4)):
            stats[status] = stats[status] + 1 if status in ("ok", "skip", "binary") else stats[status]
            if status == "error":
                stats["error"].append(info)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} 用时 {time.time()-t0:.0f}s", flush=True)
    (CACHE_DIR / "build_stats.json").write_text(
        json.dumps({"ok": stats["ok"], "skip": stats["skip"], "binary": stats["binary"],
                    "errors": stats["error"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成: ok={stats['ok']} skip={stats['skip']} binary={stats['binary']} "
          f"error={len(stats['error'])} 用时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
