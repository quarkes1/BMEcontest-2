# -*- coding: utf-8 -*-
"""重算已训折的标准化统计（原始缓存已删时的补救）：
- L3a: 每折训练会话窗口 build_raw_channels → 通道 sums/sumsq → models/l3a_resnet_stats_fold{k}.json
- L3b: 每折训练会话 30s 窗 hrv → mean/std → models/l3b_v2_stats_fold{k}.json
运行：conda activate bme && python scripts/recompute_stats.py --folds 0,1,2,3,4（8 进程）"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.data.loader import load_session, detect_binary, _find_collect_data
from src.data.windows import iter_window_labels
from src.models.l3a_cnn import build_raw_channels, N_CHANNELS
from src.models.l3b_ppgnn import build_ppg_window

def _l3a_one(args):
    sid, meal_list = args
    try:
        if detect_binary(_find_collect_data(str(config.SENSOR_DIR / sid))):
            return None
        s = load_session(sid)
        sums = np.zeros(N_CHANNELS, dtype=np.float64)
        sumsq = np.zeros(N_CHANNELS, dtype=np.float64)
        n = 0
        for w in iter_window_labels(s, [(m["before"], m["after"]) for m in meal_list]):
            X = build_raw_channels(s, w["start_row"], w["end_row"])
            sums += X.sum(axis=1); sumsq += (X.astype(np.float64) ** 2).sum(axis=1)
            n += 1
        return sums, sumsq, n
    except Exception:
        return None

def _l3b_one(args):
    sid, meal_list = args
    try:
        if detect_binary(_find_collect_data(str(config.SENSOR_DIR / sid))):
            return None
        s = load_session(sid)
        hs = []
        for w in iter_window_labels(s, [(m["before"], m["after"]) for m in meal_list],
                                    window_rows=3150, stride_rows=1050):
            X, h = build_ppg_window(s, w["start_row"], w["end_row"])
            hs.append(h)
        if not hs:
            return None
        H = np.stack(hs)
        return H.mean(axis=0), H.std(axis=0), len(H)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    folds = splits.load_folds()
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index().set_index("session_id")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for k in [int(x) for x in args.folds.split(",")]:
        f = folds[k]
        print(f"===== fold {k} =====", flush=True)
        # L3a
        tasks = [(sid, meal_meta.get(index.loc[sid, "externalid"], []))
                 for sid in f["train_sessions"] if sid in index.index]
        sums = np.zeros(N_CHANNELS, dtype=np.float64)
        sumsq = np.zeros(N_CHANNELS, dtype=np.float64)
        n = 0
        with ProcessPoolExecutor(max_workers=8) as ex:
            for r in ex.map(_l3a_one, tasks, chunksize=4):
                if r:
                    sums += r[0]; sumsq += r[1]; n += r[2]
        mean = sums / max(1, n)
        var = np.clip(sumsq / max(1, n) - mean ** 2, 0, None)
        std = np.sqrt(var).astype(np.float32)
        out = {"mean": mean.tolist(), "std": std.tolist(), "n_windows": n}
        (config.MODEL_DIR / f"l3a_resnet_stats_fold{k}.json").write_text(
            json.dumps(out), encoding="utf-8")
        print(f"  l3a: {n} 窗口 → models/l3a_resnet_stats_fold{k}.json", flush=True)
        # L3b（单遍收集 (mean,std,n)，加权合并）
        parts = []
        with ProcessPoolExecutor(max_workers=8) as ex:
            for r in ex.map(_l3b_one, tasks, chunksize=4):
                if r:
                    parts.append(r)
        if parts:
            Hm = np.average([p[0] for p in parts], axis=0, weights=[p[2] for p in parts])
            var = np.average([p[1] ** 2 + (p[0] - Hm) ** 2 for p in parts], axis=0,
                             weights=[p[2] for p in parts])
            std_h = np.sqrt(np.clip(var, 0, None))
            (config.MODEL_DIR / f"l3b_v2_stats_fold{k}.json").write_text(
                json.dumps({"mean_h": Hm.tolist(), "std_h": std_h.tolist()}), encoding="utf-8")
            print(f"  l3b: {sum(p[2] for p in parts)} 窗口 → models/l3b_v2_stats_fold{k}.json", flush=True)

if __name__ == "__main__":
    main()
