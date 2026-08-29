# -*- coding: utf-8 -*-
"""IT2: 事件后处理参数网格搜索（fold 0 验证集）：
合并间隔 {15,30,60,90} × 最小时长 {20,30,45,60} × 阈值 {0.1..0.6}，膨胀固定 ±6s。
复用 l3a_val_raw 缓存与已训练模型。产出 outputs/postprocess_tune.json。"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.eval.metrics import compute_metrics
from src.infer.events import windows_to_events
from src.models.l3a_cnn import L3aCNN, L3aCNNLarge, N_CHANNELS
from src.models.l3a_resnet import L3aResNet

A_VAL = config.CACHE_DIR / "l3a_val_raw"
MERGE_GAPS = (15, 30, 60, 90)
MIN_DURS = (20, 30, 45, 60)
THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)

def load_scores(fold, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = {"small": L3aCNN(5), "large": L3aCNNLarge(5),
             "resnet": L3aResNet(5)}[model_name].to(device).eval()
    model.load_state_dict(torch.load(
        config.MODEL_DIR / f"l3a_cnn_{model_name}_fold{fold}.pt", weights_only=True))
    p_stats = config.CACHE_DIR / "l3a_raw" / f"fold{fold}" / "stats.json"
    if not p_stats.exists():
        p_stats = config.MODEL_DIR / f"l3a_{model_name}_stats_fold{fold}.json"
    stats = json.loads(p_stats.read_text(encoding="utf-8"))
    mean = np.array(stats["mean"], dtype=np.float32).reshape(1, N_CHANNELS, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(1, N_CHANNELS, 1) + 1e-6
    probs, t0s, t1s = [], [], []
    with torch.no_grad():
        for f in sorted(A_VAL.glob("*.npz")):
            d = np.load(f)
            X = (d["X"].astype(np.float32) - mean) / std
            sp = []
            for b in range(0, len(X), 512):
                xb = torch.from_numpy(X[b:b + 512]).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logit, _ = model(xb)
                sp.append(torch.sigmoid(logit).float().cpu().numpy())
            probs.append(np.concatenate(sp))
            t0s.append(d["t0"]); t1s.append(d["t1"])
    return np.concatenate(probs), np.concatenate(t0s), np.concatenate(t1s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="small", choices=["small", "large", "resnet"])
    ap.add_argument("--build-val", action="store_true", help="先重建该折验证缓存")
    args = ap.parse_args()
    f = folds = splits.load_folds()[args.fold]
    if args.build_val:
        import shutil
        from scripts.run_full_pipeline import build_val_caches
        shutil.rmtree(A_VAL, ignore_errors=True)
        build_val_caches(f["val_sessions"])
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index().set_index("session_id")
    ext_set = {index.loc[s, "externalid"] for s in f["val_sessions"] if s in index.index}
    true_all = [(m["before"], m["after"]) for e in ext_set for m in meal_meta.get(e, [])]
    true_dom = [(m["before"], m["after"]) for e in ext_set for m in meal_meta.get(e, [])
                if m["scene"] == "dominant"]

    print("加载验证分数...", flush=True)
    p, t0, t1 = load_scores(args.fold, args.model)
    print(f"{len(p)} 窗口, {len(true_all)} 真值事件", flush=True)

    t_start = time.time()
    results = []
    for gap in MERGE_GAPS:
        for dur in MIN_DURS:
            for thr in THRESHOLDS:
                evs = windows_to_events(p, t0, t1, thr, merge_gap_s=gap, min_dur_s=dur)
                m = compute_metrics(evs, true_all)
                results.append({"gap": gap, "min_dur": dur, "thr": thr,
                                "f1": m["f1"], "sens": m["sensitivity"],
                                "ppv": m["ppv"], "n_pred": m["n_pred"]})
    results.sort(key=lambda r: -r["f1"])
    print(f"网格搜索完成（{len(results)} 组合, {time.time()-t_start:.0f}s）", flush=True)
    for r in results[:8]:
        print(f"  gap={r['gap']}s dur={r['min_dur']}s thr={r['thr']}: "
              f"F1={r['f1']:.3f} sens={r['sens']:.2f} ppv={r['ppv']:.3f} n_pred={r['n_pred']}", flush=True)
    best = results[0]
    best_evs = windows_to_events(p, t0, t1, best["thr"],
                                 merge_gap_s=best["gap"], min_dur_s=best["min_dur"])
    out = {"best": best, "top8": results[:8],
           "dominant_at_best": compute_metrics(best_evs, true_dom)}
    (config.OUTPUT_DIR / "postprocess_tune.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"dominant 场景在最优参数下 F1={out['dominant_at_best']['f1']:.3f} → outputs/postprocess_tune.json")

if __name__ == "__main__":
    main()
