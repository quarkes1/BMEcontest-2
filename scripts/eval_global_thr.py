# -*- coding: utf-8 -*-
"""全局 OOF 阈值评估（统一部署协议——消除每折独立选阈值的小样本乐观偏差）。

流程：5 折 bag 窗模型概率 → 各折 val 密度候选 → 全数据复核器打分
→ pooled（5 折合并）候选上选一个全局阈值（max 事件 F1）→ 各折用该阈值评估。
用法：D:/Anaconda3/envs/bme/python.exe scripts/eval_global_thr.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
import slide_verifier as sv
import official_iou_eval as oe
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
import joblib


def main():
    ver = joblib.load("dist/slide_models/verifier.joblib")
    pooled = []   # (fold, meta, vs, y)
    fold_meta = {}
    for k in range(5):
        # bag 窗模型概率
        models = []
        for kk in range(5):
            dd = joblib.load(f"dist/slide_models/wmodel_fold{kk}.joblib")
            models.append((dd["imp"], dd["model"]))
        d = np.load(config.CACHE_DIR / "slide" / f"fold{k}_val.npz", allow_pickle=True)
        wids = [json.loads(w) for w in d["wid"]]
        pc = np.zeros((len(wids), 1), np.float32)
        for j, w in enumerate(wids):
            hh = int((w[1] / 3.6e6) % 24)
            pc[j, 0] = sv.GLOBAL_PRIOR[hh]
        Xv = np.concatenate([d["feat"], pc], 1)
        prob = np.mean([m2.predict_proba(i2.transform(Xv))[:, 1] for i2, m2 in models], 0)
        sw = defaultdict(list)
        for wid, p in zip(wids, prob):
            sw[wid[0]].append((wid[1], wid[2], float(p)))
        cands = sv.density_candidates(sw, 0.28838)
        true_va = sv.eligible_meals(set(sw.keys()), sw)
        Xf, meta = sv.verifier_features(cands, sw, None)
        vs = ver.predict_proba(Xf)[:, 1]
        y = sv.match_labels([(m[0], m[1], m[2]) for m in meta], true_va)
        pooled.append((k, meta, vs, y, true_va))
        fold_meta[k] = len(cands)
        print(f"fold{k}: 候选 {len(cands)}（正 {int(y.sum())}）", flush=True)

    # pooled 全局阈值（max 事件 F1——与部署一致：一个阈值用全部数据选）
    all_vs = np.concatenate([p[2] for p in pooled])
    all_y = np.concatenate([p[3] for p in pooled])
    best_t, best_f1 = None, None
    tgrid = np.unique(np.concatenate([all_vs, np.linspace(0.2, 0.95, 151)]))
    for t in tgrid:
        acc = all_vs >= t
        tp = int((acc & (all_y == 1)).sum())
        fp = int((acc & (all_y == 0)).sum())
        fn = int(all_y.sum()) - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        if best_f1 is None or f1 > best_f1:
            best_f1, best_t = f1, float(t)
    print(f"全局 OOF 阈值 {best_t:.3f} → pooled 事件 F1 {best_f1:.3f}", flush=True)

    # 各折用全局阈值
    tot_tp = tot_elig = tot_pred = 0
    lines = []
    for k, meta, vs, y, true_va in pooled:
        preds = [(m[0], (m[1], m[2])) for m, a in zip(meta, vs >= best_t) if a]
        m = oe.official_metrics(preds, true_va)
        lines.append(f"fold{k}: F1={m['f1']:.3f} sens={m['sensitivity']:.3f} ppv={m['ppv']:.3f} "
                     f"({m['n_tp']}/{m['n_true']}, pred={m['n_pred']})")
        tot_tp += m["n_tp"]; tot_elig += m["n_true"]; tot_pred += m["n_pred"]
    sens = tot_tp / tot_elig; ppv = tot_tp / tot_pred
    lines.append(f"全局聚合: TP {tot_tp}/{tot_elig} pred {tot_pred} → "
                 f"sens {sens:.3f} ppv {ppv:.3f} F1 {2 * sens * ppv / (sens + ppv):.3f}")
    open("outputs/global_thr_result.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
