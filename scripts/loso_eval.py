# -*- coding: utf-8 -*-
"""LOSO（留一受试者）评估——泛化诊断。

对每个受试者（externalid）：其全部会话作测试；其余受试者的滑窗特征作训练
（各折 train/meal/no_meal/val npz 拼窗，每会话 ≤3× 负采样）→ HGB 窗模型
（63 列含时刻）→ 打分 → 密度 → 全数据复核器（dist verifier）→ 阈值 → 评估。

用法：D:/Anaconda3/envs/bme/python.exe scripts/loso_eval.py [--limit N]
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
import slide_features as sf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
import joblib


def prior_col(wids):
    out = np.zeros((len(wids), 1), np.float32)
    for j, w in enumerate(wids):
        hh = int((w[1] / 3.6e6) % 24)
        out[j, 0] = sv.GLOBAL_PRIOR[hh]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="受试者数上限（调试）")
    ap.add_argument("--thr", type=float, default=0.717, help="复核阈值（dist config 中位）")
    args = ap.parse_args()

    # 受试者 → 会话
    from src.data import manifests, splits
    idx = manifests.load_sensor_index()
    ext_to_sids = defaultdict(list)
    for _, r in idx.iterrows():
        ext_to_sids[r["externalid"]].append(r["session_id"])
    # eligible 餐（按受试者）
    meal_meta, _ = manifests.load_meal_meta()

    def sid_meals_of(sid):
        row = idx[idx["session_id"] == sid]
        out = []
        for _, r in row.iterrows():
            for m in meal_meta.get(r["externalid"], []):
                if m["before"] >= int(r["timeStamp.startTime"]) and m["after"] <= int(r["timeStamp.endTime"]):
                    out.append(m)
        return out

    # 全量窗缓存索引：sid → npz 位置（fold, split）
    sid_loc = {}
    for k in range(5):
        for split in ("train", "meal_train", "no_meal_train", "val"):
            p = config.CACHE_DIR / "slide" / f"fold{k}_{split}.npz"
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            for w in d["wid"]:
                sid_loc.setdefault(json.loads(w)[0], []).append((k, split))
            d.close()

    ver = joblib.load("dist/slide_models/verifier.joblib")
    results = []
    t0 = __import__("time").time()
    subs = sorted(ext_to_sids.keys())
    done = 0
    for ext in subs:
        test_sids = ext_to_sids[ext]
        # 该受试者 eligible 餐
        test_meals = []
        for sid in test_sids:
            for m in sid_meals_of(sid):
                test_meals.append((sid, (m["before"], m["after"])))
        if not test_meals:
            continue
        # 训练窗：其他受试者的 0/1 窗（各折采样）
        from src.data import splits as _sp
        train_w = []
        for k in range(5):
            tr = np.load(config.CACHE_DIR / "slide" / f"fold{k}_train.npz", allow_pickle=True)
            for w, lab in zip(tr["wid"], tr["label"]):
                sid = json.loads(w)[0]
                if sid not in test_sids and lab >= 0:
                    train_w.append((k, w, lab))
            tr.close()
        # 负采样（每会话 3× 正）
        rng = np.random.default_rng(20260901)
        by_sid = defaultdict(list)
        for i, (k, w, lab) in enumerate(train_w):
            by_sid[json.loads(w)[0]].append(i)
        keep = np.zeros(len(train_w), bool)
        for sid, ids in by_sid.items():
            n_pos = sum(1 for i in ids if train_w[i][2] == 1)
            n_allow = max(3 * n_pos, 1)
            neg = [i for i in ids if train_w[i][2] == 0]
            if len(neg) > n_allow:
                neg = list(rng.choice(neg, n_allow, replace=False))
            for i in ids:
                keep[i] = train_w[i][2] == 1 or i in neg
        sel = [train_w[i] for i in range(len(train_w)) if keep[i]]
        if len(sel) < 500:
            continue
        # 组装特征（62 + 时刻）
        Xs, ys = [], []
        for k, wj, lab in sel:
            d = np.load(config.CACHE_DIR / "slide" / f"fold{k}_train.npz", allow_pickle=True)
            # 需要找 wj 在 npz 的索引——train npz 与 wid 顺序——用匹配
            d.close()
        # 上面方式低效——直接用 fold train npz 的采样（现有 train npz 就是每折 3×采样——直接拼 5 折 train 的 0/1 窗即可）
        Xs, ys = [], []
        for k in range(5):
            tr = np.load(config.CACHE_DIR / "slide" / f"fold{k}_train.npz", allow_pickle=True)
            wids = [json.loads(w) for w in tr["wid"]]
            keepk = (tr["label"] >= 0) & np.array([w[0] not in test_sids for w in wids])
            if not keepk.any():
                tr.close()
                continue
            pc = prior_col([wids[i] for i in range(len(wids)) if keepk[i]])
            Xs.append(np.concatenate([tr["feat"][keepk], pc], 1))
            ys.append(tr["label"][keepk])
            tr.close()
        if not Xs:
            continue
        X_all = np.concatenate(Xs); y_all = np.concatenate(ys).astype(int)
        imp = SimpleImputer(strategy="median").fit(X_all)
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
            min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
            random_state=42)
        clf.fit(imp.transform(X_all), y_all)
        # 测试：该受试者全部 val 窗（分布在各折 val npz——受试者的会话在其折 val）
        preds_all = []
        for k in range(5):
            d = np.load(config.CACHE_DIR / "slide" / f"fold{k}_val.npz", allow_pickle=True)
            wids = [json.loads(w) for w in d["wid"]]
            msk = np.array([w[0] in test_sids for w in wids])
            if not msk.any():
                d.close()
                continue
            pc = prior_col([wids[i] for i in range(len(wids)) if msk[i]])
            Xv = np.concatenate([d["feat"][msk], pc], 1)
            prob = clf.predict_proba(imp.transform(Xv))[:, 1]
            sw = defaultdict(list)
            for w, p in zip([wids[i] for i in range(len(wids)) if msk[i]], prob):
                sw[w[0]].append((w[1], w[2], float(p)))
            d.close()
            cands = sv.density_candidates(sw, 0.28838)
            if cands:
                Xf, meta = sv.verifier_features(cands, sw, None)
                vs = ver.predict_proba(Xf)[:, 1]
                for m, a in zip(meta, vs >= args.thr):
                    if a:
                        preds_all.append((m[0], (m[1], m[2])))
        # eligible 过滤
        elig = []
        for sid in test_sids:
            p = config.CACHE_DIR / "sessions" / f"{sid}.npz"
            if not p.exists():
                continue
            with np.load(p) as z:
                tv = z["t_acc"][z["imu_valid"]]
            for m in sid_meals_of(sid):
                lo = np.searchsorted(tv, m["before"]); hi = np.searchsorted(tv, m["after"])
                if hi > lo and (tv[min(hi, len(tv) - 1)] - tv[max(lo, 0)]) >= 0.5 * (m["after"] - m["before"]):
                    elig.append((sid, (m["before"], m["after"])))
        mm = oe.official_metrics(preds_all, elig)
        results.append({"subject": ext, "f1": mm["f1"], "n_tp": mm["n_tp"],
                        "n_true": mm["n_true"], "n_pred": mm["n_pred"]})
        done += 1
        print(f"[{done}] {ext}: {mm['n_tp']}/{mm['n_true']} F1={mm['f1']:.3f} "
              f"({__import__('time').time() - t0:.0f}s)", flush=True)
        if args.limit and done >= args.limit:
            break
    # 汇总
    tot_tp = sum(r["n_tp"] for r in results); tot_e = sum(r["n_true"] for r in results)
    tot_p = sum(r["n_pred"] for r in results)
    mean_f1 = np.mean([r["f1"] for r in results]) if results else 0
    sens = tot_tp / tot_e if tot_e else 0
    ppv = tot_tp / tot_p if tot_p else 0
    f1_agg = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0
    out = {"n_subjects": len(results), "mean_subject_f1": float(mean_f1),
           "aggregate": {"tp": tot_tp, "eligible": tot_e, "pred": tot_p,
                         "sens": sens, "ppv": ppv, "f1": f1_agg}}
    (config.OUTPUT_DIR / "loso_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nLOSO: {len(results)} 受试者 | 均值 F1 {mean_f1:.3f} | 聚合 TP {tot_tp}/{tot_e} "
          f"pred {tot_p} → sens {sens:.3f} ppv {ppv:.3f} F1 {f1_agg:.3f}", flush=True)


if __name__ == "__main__":
    main()
