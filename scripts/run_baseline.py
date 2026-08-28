# -*- coding: utf-8 -*-
"""LightGBM 基线 5 折主流程（依赖特征缓存，见 build_feature_cache.py）。
运行：conda activate bme && python scripts/run_baseline.py
内存策略：mmap 读缓存 + 按会话增量抽样，峰值内存 ~2GB（16GB 机器安全）。"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 项目根入 sys.path

import src.config as config
from src.data import splits
from src.models.baseline_lgbm import train_one_fold, predict_session
from src.eval.metrics import compute_metrics

FEAT_CACHE = config.CACHE_DIR / "baseline_features"
N_WORKERS = 4    # 保守并发：避免占满 IO/内存导致系统卡顿

def _count_one(sid):
    """统计单会话正样本与有效窗口总数（mmap，不拷贝数据）。"""
    f = FEAT_CACHE / f"{sid}.npz"
    if not f.exists():
        return (0, 0)
    d = np.load(f, mmap_mode="r")
    y = d["y"]
    valid = y >= 0
    return (int((y[valid] == 1).sum()), int(valid.sum()))

def load_fold_features(session_ids, neg_ratio=3.0, seed=config.RANDOM_SEED):
    """按会话增量加载并抽样：返回 (X, y, t0s, t1s)。峰值内存 = 抽样后数据量。"""
    print(f"  统计 {len(session_ids)} 个会话...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        counts = list(ex.map(_count_one, session_ids))
    pos_total = sum(c[0] for c in counts)
    neg_total = sum(c[1] - c[0] for c in counts)
    r_neg = min(1.0, neg_ratio * pos_total / max(1, neg_total))
    print(f"  正样本 {pos_total} / 负样本 {neg_total}，负采样率 {r_neg:.3f}", flush=True)

    Xs, ys, t0s, t1s = [], [], [], []
    for i, sid in enumerate(session_ids):
        f = FEAT_CACHE / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f, mmap_mode="r")
        keep = d["y"] >= 0
        y_k = np.asarray(d["y"][keep])
        pos_m = y_k == 1
        neg_m = ~pos_m
        rng = np.random.RandomState(seed + i)
        sel_neg = neg_m & (rng.random(int(neg_m.sum())) < r_neg)
        full_mask = np.zeros(len(keep), dtype=bool)
        full_mask[keep] = pos_m | sel_neg          # 直接在 mmap 上按掩码取数，避免全量拷贝
        Xs.append(np.asarray(d["X"][full_mask]))
        ys.append(np.asarray(d["y"][full_mask]))
        t0s.append(np.asarray(d["t0"][full_mask]))
        t1s.append(np.asarray(d["t1"][full_mask]))
    X = np.vstack(Xs); del Xs
    y = np.concatenate(ys); del ys
    t0 = np.concatenate(t0s); t1 = np.concatenate(t1s); del t0s, t1s
    print(f"  训练窗口 {len(y)}（加载+抽样用时 {time.time()-t0:.0f}s）", flush=True)
    return X, y, t0, t1

def _load_val_session(sid):
    """加载单会话验证窗口（数组）+ 该会话真值事件（y==1 并集区间）。"""
    f = FEAT_CACHE / f"{sid}.npz"
    if not f.exists():
        return None
    d = np.load(f, mmap_mode="r")
    mask = d["y"] >= 0
    Xv = np.asarray(d["X"][mask])
    t0v = np.asarray(d["t0"][mask])
    t1v = np.asarray(d["t1"][mask])
    pos = np.where(np.asarray(d["y"]) == 1)[0]
    true_events = []
    if len(pos):
        order = np.argsort(np.asarray(d["t0"])[pos])
        cur_s, cur_e = int(d["t0"][pos[order[0]]]), int(d["t1"][pos[order[0]]])
        for i in order[1:]:
            if int(d["t0"][pos[i]]) <= cur_e:
                cur_e = max(cur_e, int(d["t1"][pos[i]]))
            else:
                true_events.append((cur_s, cur_e))
                cur_s, cur_e = int(d["t0"][pos[i]]), int(d["t1"][pos[i]])
        true_events.append((cur_s, cur_e))
    return (Xv, t0v, t1v, true_events)

def load_val_data(session_ids):
    print(f"  加载 {len(session_ids)} 个验证会话...", flush=True)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        parts = [p for p in ex.map(_load_val_session, session_ids) if p is not None]
    Xv = np.vstack([p[0] for p in parts])
    t0v = np.concatenate([p[1] for p in parts])
    t1v = np.concatenate([p[2] for p in parts])
    true_events = [e for p in parts for e in p[3]]
    return Xv, t0v, t1v, true_events

def eval_threshold(model, Xv, t0v, t1v, true_events, threshold):
    evs, _ = predict_session(model, Xv, t0v, t1v, threshold)
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
        del Xtr, ytr
        Xv, t0v, t1v, val_true = load_val_data(f["val_sessions"])
        # 阈值扫描：0.3~0.7 取验证 F1 最优（验证数据只加载一次）
        best = None
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            m = eval_threshold(model, Xv, t0v, t1v, val_true, thr)
            print(f"    thr={thr}: F1={m['f1']:.3f} sens={m['sensitivity']:.3f} ppv={m['ppv']:.3f}", flush=True)
            if best is None or m["f1"] > best[1]["f1"]:
                best = (thr, m)
        thr, m = best
        report["thresholds"][str(f["fold"])] = thr
        print(f"  best thr={thr} F1={m['f1']:.3f} MAE_start={m['mae_start_s']}s "
              f"n_tp={m['n_tp']}/{m['n_true']}", flush=True)
        report["folds"].append({"fold": f["fold"], "threshold": thr, "metrics": m})
        del Xv, t0v, t1v, val_true, model
    f1s = [f["metrics"]["f1"] for f in report["folds"]]
    report["mean_f1"] = float(np.mean(f1s))
    print(f"=== 5 折平均 F1 = {np.mean(f1s):.3f}（总用时 {time.time()-t0:.0f}s）", flush=True)
    (config.OUTPUT_DIR / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("report -> outputs/baseline_report.json")

if __name__ == "__main__":
    main()
