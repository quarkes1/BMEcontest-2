# -*- coding: utf-8 -*-
"""LightGBM 基线 5 折主流程（依赖特征缓存，见 build_feature_cache.py）。
运行：conda activate bme && python scripts/run_baseline.py"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 项目根入 sys.path

import src.config as config
from src.data import manifests, splits
from src.models.baseline_lgbm import train_one_fold, predict_session
from src.eval.metrics import compute_metrics

FEAT_CACHE = config.CACHE_DIR / "baseline_features"

def load_fold_features(session_ids, neg_ratio=3.0, seed=config.RANDOM_SEED):
    """从缓存加载窗口特征；负样本 3:1 采样。返回 X, y, windows 元信息。"""
    Xs, ys, metas = [], [], []
    for sid in session_ids:
        f = FEAT_CACHE / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        keep = d["y"] >= 0
        Xs.append(d["X"][keep]); ys.append(d["y"][keep])
        metas.append({"t0": d["t0"][keep], "t1": d["t1"][keep]})
    if not Xs:
        raise RuntimeError("特征缓存为空，请先运行 scripts/build_feature_cache.py")
    X = np.vstack(Xs); y = np.concatenate(ys)
    t0 = np.concatenate([m["t0"] for m in metas]); t1 = np.concatenate([m["t1"] for m in metas])
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), int(len(pos_idx) * neg_ratio)), replace=False)
    idx = np.sort(np.concatenate([pos_idx, keep_neg]))
    return X[idx], y[idx], t0[idx], t1[idx]

def eval_fold(model, session_ids, threshold):
    """验证集滑窗推理 -> 事件级指标（按会话分开展开真值）。"""
    pred_events, true_events = [], []
    for sid in session_ids:
        f = FEAT_CACHE / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        mask = d["y"] >= 0
        windows = [{"feat": d["X"][i], "t0_ms": int(d["t0"][i]), "t1_ms": int(d["t1"][i])}
                   for i in np.where(mask)[0]]
        evs, _ = predict_session(model, windows, threshold)
        pred_events.extend(evs)
        # 该会话真值 = y==1 窗口的并集区间（从缓存重建，与训练标签一致）
        pos = np.where(d["y"] == 1)[0]
        if len(pos):
            t0s = d["t0"][pos]; t1s = d["t1"][pos]
            order = np.argsort(t0s)
            cur_s, cur_e = int(t0s[order[0]]), int(t1s[order[0]])
            for i in order[1:]:
                if int(t0s[i]) <= cur_e:
                    cur_e = max(cur_e, int(t1s[i]))
                else:
                    true_events.append((cur_s, cur_e)); cur_s, cur_e = int(t0s[i]), int(t1s[i])
            true_events.append((cur_s, cur_e))
    return compute_metrics(pred_events, true_events)

def main():
    t0 = time.time()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = splits.load_folds()
    report = {"folds": [], "thresholds": {}}
    for f in folds:
        print(f"=== fold {f['fold']}: {f['train_meals']} train / {f['val_meals']} val meals ===", flush=True)
        Xtr, ytr, _, _ = load_fold_features(f["train_sessions"])
        model = train_one_fold(Xtr, ytr)
        # 阈值扫描：0.3~0.7 取验证 F1 最优
        best = None
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            m = eval_fold(model, f["val_sessions"], thr)
            if best is None or m["f1"] > best[1]["f1"]:
                best = (thr, m)
        thr, m = best
        report["thresholds"][str(f["fold"])] = thr
        print(f"  best thr={thr} F1={m['f1']:.3f} sens={m['sensitivity']:.3f} ppv={m['ppv']:.3f} "
              f"MAE_start={m['mae_start_s']}s n_tp={m['n_tp']}/{m['n_true']}", flush=True)
        report["folds"].append({"fold": f["fold"], "threshold": thr, "metrics": m})
    f1s = [f["metrics"]["f1"] for f in report["folds"]]
    report["mean_f1"] = float(np.mean(f1s))
    print(f"=== 5 折平均 F1 = {np.mean(f1s):.3f} (用时 {time.time()-t0:.0f}s)", flush=True)
    (config.OUTPUT_DIR / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("report -> outputs/baseline_report.json")

if __name__ == "__main__":
    main()
