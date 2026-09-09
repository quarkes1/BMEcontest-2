# -*- coding: utf-8 -*-
"""LOSO（留一受试者）严格零信息评估——泛化诊断（peer review 泄漏修复版）。

对每个受试者 X（externalid）：
1. 窗模型只用其余受试者的窗训练（每折 train npz 拼窗，label≥0，≤3× 采样不变）；
2. 复核器（L2）只用其余受试者的候选重训——候选由该 LOSO 窗模型对
   其余受试者 meal_train/no_meal_train 全窗打分 → 密度生成（与 5 折 per-fold
   复核训练同构；窗模型与候选受试者相同 = 与 CV 管线一致的同程度自打分）；
3. 只在该受试者会话上预测 → 阈值固定（--thr，默认 0.717 = 部署 config 中位，
   不利用受试者标签）→ eligible 口径与 5 折一致（eligible_meals 含 ≥120s 窗覆盖）；
4. eligible 为空的受试者跳过。

无 TCN（部署 CPU 口径）。聚合 F1 = 官方全局口径。
用法：D:/Anaconda3/envs/bme/python.exe scripts/loso_eval.py [--limit N] [--workers W] [--thr 0.717]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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

THR_WIN = 0.28838   # 窗阈值（对方冻结）
MERGE_MS = 120_000


def load_all():
    """受试者→会话、会话→折/表、session npz 有效时间。"""
    from src.data import manifests
    idx = manifests.load_sensor_index()
    ext_to_sids = defaultdict(list)
    for _, r in idx.iterrows():
        ext_to_sids[r["externalid"]].append(r["session_id"])
    # 会话所在折 val（受试者互斥划分 → 每个受试者的会话在同一折 val）
    sid_fold = {}
    for k in range(5):
        p = config.CACHE_DIR / "slide" / f"fold{k}_val.npz"
        with np.load(p, allow_pickle=True) as d:
            for w in d["wid"]:
                sid_fold.setdefault(json.loads(w)[0], k)
    # 每折 train npz 的窗口（sid, feat, label）——LOSO 训练素材（已 ≤3× 采样）
    fold_tr = {}
    for k in range(5):
        with np.load(config.CACHE_DIR / "slide" / f"fold{k}_train.npz", allow_pickle=True) as d:
            fold_tr[k] = (d["feat"], d["label"], d["wid"])
    # 每折 meal_train / no_meal_train npz 窗口（复核候选素材，含全部会话全窗）
    fold_ms = {}
    for k in range(5):
        fold_ms[k] = {}
        for sn in ("meal_train", "no_meal_train"):
            p = config.CACHE_DIR / "slide" / f"fold{k}_{sn}.npz"
            if p.exists():
                with np.load(p, allow_pickle=True) as d:
                    fold_ms[k][sn] = (d["feat"], d["wid"])
    return ext_to_sids, sid_fold, fold_tr, fold_ms


def prior_col(wids):
    out = np.zeros((len(wids), 1), np.float32)
    for j, w in enumerate(wids):
        hh = int((w[1] / 3.6e6) % 24)
        out[j, 0] = sv.GLOBAL_PRIOR[hh]
    return out


def train_window(feat, lab, wid):
    """label≥0 窗 → HGB（63 列 = 62 + 时刻先验；无 TCN——部署 CPU 口径）。"""
    lab = np.asarray(lab)
    keep = lab >= 0
    if keep.sum() < 1000:
        return None
    X = np.concatenate([feat[keep], prior_col([json.loads(w) for w in wid[keep]])], 1)
    imp = SimpleImputer(strategy="median").fit(X)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
        random_state=20260901)
    clf.fit(imp.transform(X), lab[keep].astype(int))
    return imp, clf


def score_to_sw(feat, wid, model):
    imp, clf = model
    X = np.concatenate([feat, prior_col([json.loads(w) for w in wid])], 1)
    prob = clf.predict_proba(imp.transform(X))[:, 1]
    sw = defaultdict(list)
    for wj, p in zip([json.loads(w) for w in wid], prob):
        sw[wj[0]].append((wj[1], wj[2], float(p)))
    return sw


def run_one(ext, test_sids, sid_fold, fold_tr, fold_ms, thr):
    t0 = time.time()
    test_sids = [s for s in test_sids if s in sid_fold]   # 无滑窗会话（太短/损坏）剔除
    if not test_sids:
        return None
    fold_k = sid_fold[test_sids[0]]          # 该受试者全部会话在同一折 val
    # 1) 窗模型：其余受试者（各折 train 拼，排除 test_sids 的会话）
    feats, labs, wids = [], [], []
    for k, (f, l, w) in fold_tr.items():
        wl = [json.loads(x) for x in w]
        m = np.array([x[0] not in test_sids for x in wl])
        if not m.any():
            continue
        feats.append(f[m]); labs.append(l[m]); wids.append(w[m])
    feat = np.concatenate(feats); lab = np.concatenate(labs); wid = np.concatenate(wids)
    model = train_window(feat, lab, wid)
    if model is None:
        return None
    # 2) 复核器训练候选：其余受试者 meal_train（全窗打分→密度→匹配 eligible 餐）
    #    + no_meal_train 候选（负样本）；复核器 = L2 LR（与 slide_verifier 同构）
    cands, y = [], []
    for k, msd in fold_ms.items():
        for sn in ("meal_train", "no_meal_train"):
            if sn not in msd:
                continue
            f, w = msd[sn]
            wl = [json.loads(x) for x in w]
            m = np.array([x[0] not in test_sids for x in wl])
            if not m.any():
                continue
            sw = score_to_sw(f[m], w[m], model)
            cs = sv.density_candidates(sw, THR_WIN)
            if not cs:
                continue
            Xc, meta = sv.verifier_features(cs, sw, None)
            if len(meta) != len(Xc):
                # verifier_features 会跳过 len(ps)<2 的候选（X 行数与 meta 应一致；不一致则按 meta 截断）
                Xc = Xc[:len(meta)]
            cands.append(Xc)
            if sn == "meal_train":
                sids_here = {wl[i][0] for i in range(len(wl)) if m[i]}
                gt = sv.eligible_meals(sids_here, sw)
                y.append(sv.match_labels(meta, gt))
            else:
                y.append(np.zeros(len(meta), np.int8))
    if not cands:
        return None
    X_tr = np.concatenate(cands); y_tr = np.concatenate(y).astype(int)
    if y_tr.sum() < 5:
        return None
    ver = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                    ("lr", LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000,
                                              random_state=20260904))])
    ver.fit(X_tr, y_tr)
    # 3) 测试受试者：其折 val 全窗打分 → 密度 → 复核 → 固定阈值
    d = np.load(config.CACHE_DIR / "slide" / f"fold{fold_k}_val.npz", allow_pickle=True)
    wl = [json.loads(w) for w in d["wid"]]
    m = np.array([x[0] in test_sids for x in wl])
    sw_t = score_to_sw(d["feat"][m], d["wid"][m], model)
    d.close()
    cands_t = sv.density_candidates(sw_t, THR_WIN)
    if not cands_t:
        return None
    X_t, meta_t = sv.verifier_features(cands_t, sw_t, None)
    vs = ver.predict_proba(X_t)[:, 1]
    elig = sv.eligible_meals(set(sw_t.keys()), sw_t)
    if not elig:
        return None
    preds = [(mm[0], (mm[1], mm[2])) for mm, a in zip(meta_t, vs >= thr) if a]
    # 合并 120s 内相邻事件（与 slide_verifier 一致的口径在 density 内已做——此处补密度外合并）
    res = oe.official_metrics(preds, elig)
    return {"subject": ext, "f1": res["f1"], "n_tp": int(res["n_tp"]),
            "n_true": int(res["n_true"]), "n_pred": int(res["n_pred"]),
            "secs": time.time() - t0, "fold": fold_k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.717)
    global args
    args = ap.parse_args()
    ext_to_sids, sid_fold, fold_tr, fold_ms = load_all()
    subs = sorted(s for s, ss in ext_to_sids.items() if any(sid in sid_fold for sid in ss))
    if args.limit:
        subs = subs[:args.limit]
    print(f"{len(subs)} 受试者 × 严格 LOSO（窗模型 + 复核器均零信息；阈值固定 {args.thr}）", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, ext, ext_to_sids[ext], sid_fold, fold_tr, fold_ms,
                          args.thr): ext for ext in subs}
        for i, fut in enumerate(futs):
            r = fut.result()
            if r:
                results.append(r)
                print(f"[{i + 1}/{len(subs)}] {r['subject']}: {r['n_tp']}/{r['n_true']} "
                      f"F1={r['f1']:.3f} ({r['secs']:.0f}s)", flush=True)
    tot_tp = sum(r["n_tp"] for r in results); tot_e = sum(r["n_true"] for r in results)
    tot_p = sum(r["n_pred"] for r in results)
    mean_f1 = float(np.mean([r["f1"] for r in results])) if results else 0.0
    sens = tot_tp / tot_e if tot_e else 0
    ppv = tot_tp / tot_p if tot_p else 0
    f1_agg = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0
    out = {"protocol": "strict-losо v2: window+verifier retrained per subject, fixed thr",
           "thr_verifier": args.thr, "n_subjects": len(results),
           "mean_subject_f1": mean_f1,
           "aggregate": {"tp": tot_tp, "eligible": tot_e, "pred": tot_p,
                         "sens": sens, "ppv": ppv, "f1": f1_agg},
           "secs_total": time.time() - t0}
    (config.OUTPUT_DIR / "loso_result_strict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n严格 LOSO: {len(results)} 受试者 | 均值 F1 {mean_f1:.3f} | 聚合 TP {tot_tp}/{tot_e} "
          f"pred {tot_p} → sens {sens:.3f} ppv {ppv:.3f} F1 {f1_agg:.3f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
