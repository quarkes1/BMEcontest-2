# -*- coding: utf-8 -*-
"""LightGBM 基线 5 折主流程（依赖特征缓存，见 build_feature_cache.py）。
运行：conda activate bme && python scripts/run_baseline.py"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 项目根入 sys.path

import src.config as config
from src.data import manifests, splits
from src.models.baseline_lgbm import train_one_fold, predict_session
from src.eval.metrics import compute_metrics

FEAT_CACHE = config.CACHE_DIR / "baseline_features"
N_WORKERS = 8

def _load_train_one(sid):
    """并行加载单会话窗口特征（训练用）。返回 None 表示无缓存。"""
    f = FEAT_CACHE / f"{sid}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    keep = d["y"] >= 0
    return (d["X"][keep], d["y"][keep], d["t0"][keep], d["t1"][keep])

def load_fold_features(session_ids, neg_ratio=3.0, seed=config.RANDOM_SEED):
    print(f"  加载 {len(session_ids)} 个会话特征缓存...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        parts = [p for p in ex.map(_load_train_one, session_ids) if p is not None]
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    t0s = np.concatenate([p[2] for p in parts]); t1s = np.concatenate([p[3] for p in parts])
    print(f"  共 {len(y)} 窗口（正 {int((y == 1).sum())}），加载用时 {time.time()-t0:.0f}s", flush=True)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), int(len(pos_idx) * neg_ratio)), replace=False)
    idx = np.sort(np.concatenate([pos_idx, keep_neg]))
    return X[idx], y[idx], t0s[idx], t1s[idx]

def _load_val_session(sid):
    """并行加载单会话验证数据：窗口列表 + 该会话真值事件（y==1 并集区间）。"""
    f = FEAT_CACHE / f"{sid}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    mask = d["y"] >= 0
    windows = [{"feat": d["X"][i], "t0_ms": int(d["t0"][i]), "t1_ms": int(d["t1"][i])}
               for i in np.where(mask)[0]]
    pos = np.where(d["y"] == 1)[0]
    true_events = []
    if len(pos):
        order = np.argsort(d["t0"][pos])
        cur_s, cur_e = int(d["t0"][pos[order[0]]]), int(d["t1"][pos[order[0]]])
        for i in order[1:]:
            if int(d["t0"][pos[i]]) <= cur_e:
                cur_e = max(cur_e, int(d["t1"][pos[i]]))
            else:
                true_events.append((cur_s, cur_e))
                cur_s, cur_e = int(d["t0"][pos[i]]), int(d["t1"][pos[i]])
        true_events.append((cur_s, cur_e))
    return (windows, true_events)

def load_val_data(session_ids):
    print(f"  加载 {len(session_ids)} 个验证会话...", flush=True)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        parts = [p for p in ex.map(_load_val_session, session_ids) if p is not None]
    windows = [w for p in parts for w in p[0]]
    true_events = [e for p in parts for e in p[1]]
    return windows, true_events

def eval_threshold(model, windows, true_events, threshold):
    evs, _ = predict_session(model, windows, threshold)
    return compute_metrics(evs, true_events)

def main():
    t0 = time.time()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = splits.load_folds()
    report = {"folds": [], "thresholds": {}}
    for f in folds:
        print(f"=== fold {f['fold']}: {f['train_meals']} train / {f['val_meals']} val meals ===", flush=True)
        Xtr, ytr, _, _ = load_fold_features(f["train_sessions"])
        model = train_one_fold(Xtr, ytr)
        val_windows, val_true = load_val_data(f["val_sessions"])
        # 阈值扫描：0.3~0.7 取验证 F1 最优（验证数据只加载一次）
        best = None
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            m = eval_threshold(model, val_windows, val_true, thr)
            print(f"    thr={thr}: F1={m['f1']:.3f} sens={m['sensitivity']:.3f} ppv={m['ppv']:.3f}", flush=True)
            if best is None or m["f1"] > best[1]["f1"]:
                best = (thr, m)
        thr, m = best
        report["thresholds"][str(f["fold"])] = thr
        print(f"  best thr={thr} F1={m['f1']:.3f} MAE_start={m['mae_start_s']}s "
              f"n_tp={m['n_tp']}/{m['n_true']}", flush=True)
        report["folds"].append({"fold": f["fold"], "threshold": thr, "metrics": m})
    f1s = [f["metrics"]["f1"] for f in report["folds"]]
    report["mean_f1"] = float(np.mean(f1s))
    print(f"=== 5 折平均 F1 = {np.mean(f1s):.3f}（总用时 {time.time()-t0:.0f}s）", flush=True)
    (config.OUTPUT_DIR / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("report -> outputs/baseline_report.json")

if __name__ == "__main__":
    main()
