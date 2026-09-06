# -*- coding: utf-8 -*-
"""部署模型固化 v2（滑窗管线，纯 CPU 推理版）：
阶段 1：5 折窗模型（train 采样窗 + 时刻列）
阶段 2：候选生成统一用 5 模型 bag 概率（与部署推理一致——复核特征分布对齐）
阶段 3：复核器全数据训练（5 折 meal + no_meal 候选）
产物：dist/slide_models/
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

OUT = Path("dist/slide_models")
OUT.mkdir(parents=True, exist_ok=True)


def prior_col_for(npz_path):
    d0 = np.load(npz_path, allow_pickle=True)
    out = np.zeros((len(d0["wid"]), 1), np.float32)
    for j, w in enumerate([json.loads(x) for x in d0["wid"]]):
        hh = int((w[1] / 3.6e6) % 24)
        out[j, 0] = sv.GLOBAL_PRIOR[hh]
    return out


def main():
    # ---- 阶段 1：5 折窗模型（全数据 bag 组件） ----
    models = []
    for k in range(5):
        tr = np.load(config.CACHE_DIR / "slide" / f"fold{k}_train.npz", allow_pickle=True)
        keep = tr["label"] >= 0
        Xk = np.concatenate([tr["feat"], prior_col_for(config.CACHE_DIR / "slide" / f"fold{k}_train.npz")], 1)
        imp = SimpleImputer(strategy="median").fit(Xk[keep])
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
            min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
            random_state=20260901 + k)
        clf.fit(imp.transform(Xk[keep]), tr["label"][keep].astype(int))
        models.append((imp, clf))
        import joblib
        joblib.dump({"imp": imp, "model": clf}, OUT / f"wmodel_fold{k}.joblib")
        print(f"窗模型 fold{k} 固化（{keep.sum()} 窗）", flush=True)

    def bag_prob(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        X = np.concatenate([d["feat"], prior_col_for(npz_path)], 1)
        return np.mean([m2.predict_proba(i2.transform(X))[:, 1] for i2, m2 in models], 0)

    def sw_from(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        wids = [json.loads(w) for w in d["wid"]]
        prob = bag_prob(npz_path)
        sw = defaultdict(list)
        for wid, p in zip(wids, prob):
            sw[wid[0]].append((wid[1], wid[2], float(p)))
        return sw

    # ---- 阶段 2+3：复核器（全数据候选，bag 概率） ----
    X_all, y_all = [], []
    for k in range(5):
        sw_tr = sw_from(config.CACHE_DIR / "slide" / f"fold{k}_meal_train.npz")
        cand_tr = sv.density_candidates(sw_tr, 0.28838)
        true_tr = sv.eligible_meals(set(sw_tr.keys()), sw_tr)
        X1, meta1 = sv.verifier_features(cand_tr, sw_tr, None)
        y1 = sv.match_labels([(m[0], m[1], m[2]) for m in meta1], true_tr)
        sw_nm = sw_from(config.CACHE_DIR / "slide" / f"fold{k}_no_meal_train.npz")
        cand_nm = sv.density_candidates(sw_nm, 0.28838)
        X2, meta2 = sv.verifier_features(cand_nm, sw_nm, None)
        X_all.append(np.concatenate([X1, X2]))
        y_all.append(np.concatenate([y1, np.zeros(len(meta2), np.int8)]))
        print(f"fold{k} 复核候选：正 {int(y1.sum())} + 无餐 {len(meta2)}", flush=True)
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    print(f"复核器训练：{len(y_all)} 候选（正 {int(y_all.sum())}）", flush=True)
    ver = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                    ("lr", LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000,
                                              random_state=20260904))])
    ver.fit(X_all, y_all)
    import joblib
    joblib.dump(ver, OUT / "verifier.joblib")

    # 阈值：全数据复核器在 5 折 val 候选上的 max-F1（近似部署选择）
    thrs, f1s = [], []
    for k in range(5):
        sw_va = sw_from(config.CACHE_DIR / "slide" / f"fold{k}_val.npz")
        cand_va = sv.density_candidates(sw_va, 0.28838)
        true_va = sv.eligible_meals(set(sw_va.keys()), sw_va)
        Xv, meta_va = sv.verifier_features(cand_va, sw_va, None)
        vs = ver.predict_proba(Xv)[:, 1]
        best_t, best_m = None, None
        for t in sorted(set(np.concatenate([vs, [0.3, 0.5, 0.7, 0.9]]))):
            preds = [(m[0], (m[1], m[2])) for m, a in zip(meta_va, vs >= t) if a]
            m = oe.official_metrics(preds, true_va)
            if best_m is None or m["f1"] > best_m["f1"]:
                best_t, best_m = t, m
        thrs.append(float(best_t)); f1s.append(float(best_m["f1"]))
        print(f"fold{k} 部署复核阈值 {best_t:.3f} → F1 {best_m['f1']:.3f}", flush=True)
    cfg = {"thr_verifier_median": float(np.median(thrs)),
           "thr_verifier_folds": thrs, "fold_f1": f1s,
           "window_thr": 0.28838, "n_wmodel": 5, "no_tcn": True}
    (OUT / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT}/（中位阈值 {cfg['thr_verifier_median']:.3f}；部署复核 F1 均值 {np.mean(f1s):.3f}）", flush=True)


if __name__ == "__main__":
    main()
