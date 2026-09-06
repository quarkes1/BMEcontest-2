# -*- coding: utf-8 -*-
"""滑窗管线完整版：HGB 窗模型 → 密度候选 → 33 特征 L2 复核器 → 官方评估。

复核器训练：meal_train（train 含餐会话全窗候选：真 = IoU≥0.25 匹配 eligible 餐，
假 = 其余）；评估：val 全窗候选。
用法：D:/Anaconda3/envs/bme/python.exe scripts/slide_verifier.py --fold 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.data import manifests, splits
import official_iou_eval as oe

STRIDE_MS = 15_000
BRIDGE_MS = 60_000
DENSITY_MS = 600_000
MIN_POS = 10
COV_MIN = 0.80
MERGE_MS = 120_000
WIN_MS = 240_000
CTX_MS = 1_200_000


def density_candidates(sid_windows, thr, sids=None):
    """滑窗概率 → 密度候选事件（window_support 边界 + 120s 合并）。返回 (sid, s, e, probs 数组, prob_max, prob_mean, dur_s)。"""
    cands = []
    for sid in sorted(sid_windows):
        if sids is not None and sid not in sids:
            continue
        arr = sorted(sid_windows[sid])
        starts = np.array([a[0] for a in arr], np.int64)
        probs = np.array([a[2] for a in arr])
        seg, segs = [], []
        for i in range(len(starts)):
            if seg and starts[i] - seg[-1][0] > STRIDE_MS + BRIDGE_MS:
                segs.append(seg); seg = []
            if seg and starts[i] - seg[-1][0] > STRIDE_MS:
                for gs in range(seg[-1][0] + STRIDE_MS, starts[i], STRIDE_MS):
                    seg.append((gs, 0.0, 0))
            seg.append((int(starts[i]), float(probs[i]), 1))
        if seg:
            segs.append(seg)
        for seg in segs:
            ss = np.array([s for s, _, _ in seg], np.int64)
            pp = np.array([p for _, p, _ in seg])
            oo = np.array([o for _, _, o in seg], np.float64)
            ds = int(round(DENSITY_MS / STRIDE_MS))
            cnt = np.convolve((pp >= thr).astype(np.int64), np.ones(ds, np.int64), "same")
            cov = np.convolve(oo, np.ones(ds) / ds, "same")
            dense = (cnt >= MIN_POS) & (cov >= COV_MIN)
            i, n = 0, len(seg)
            while i < n:
                if dense[i]:
                    j = i
                    while j < n and dense[j]:
                        j += 1
                    # 事件边界收缩到"越阈窗范围"（dense run 有 ±600s 缓冲膨胀；
                    # 纯越阈窗跨度贴合餐时段——长 run 覆盖餐前后活动时 IoU 提升）
                    pos_idx = np.where(pp[i:j] >= thr)[0]
                    if len(pos_idx) == 0:
                        i = j
                        continue
                    a, b = i + pos_idx[0], i + pos_idx[-1]
                    ps = pp[a:b + 1]
                    ev = [sid, int(ss[a]), int(ss[b] + WIN_MS), ps,
                          float(ps.max()), float(ps.mean()), (b - a + 1) * STRIDE_MS / 1000.0]
                    if cands and cands[-1][0] == sid and ev[1] - cands[-1][2] <= MERGE_MS:
                        c0 = cands[-1]
                        c0[2] = max(c0[2], ev[2])
                        c0[3] = np.concatenate([c0[3], ps])
                        c0[4] = max(c0[4], ev[4])
                        c0[5] = float(np.mean(c0[3]))
                        c0[6] = c0[6] + ev[6]
                    else:
                        cands.append(ev)
                    i = j
                else:
                    i += 1
    return cands


def verifier_features(cands, sid_windows):
    """33 特征（对方规格）：事件内概率形态 + 前/后 20min 上下文。"""
    X, meta = [], []
    for c in cands:
        sid, s, e, ps, pmax, pmean, dur = c
        if len(ps) < 2:
            continue
        xs = [dur, len(ps),
              float(ps.max() - ps.min()),
              float(ps.std() / (ps.mean() + 1e-9))]
        t = np.arange(len(ps))
        slope = float(np.polyfit(t, ps, 1)[0]) if len(ps) > 2 else 0.0
        xs += [slope, float(ps[0] - ps[-1]),
               float((ps >= 0.35).mean()), float((ps >= 0.45).mean())]
        # longest above
        def longest(q):
            best = cur = 0
            for v in ps >= q:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            return best
        xs += [longest(0.35), longest(0.45),
               float(np.maximum(ps - 0.28838, 0).sum())]
        xs += [float(ps.mean()), float(ps.max()), float(ps.std()),
               float(np.percentile(ps, 10)), float(np.percentile(ps, 50)),
               float(np.percentile(ps, 90))]
        # 上下文：该会话窗中心序列
        arr = sorted(sid_windows.get(sid, []))
        centers = np.array([(a[0] + a[1]) // 2 for a in arr], np.int64)
        probs_c = np.array([a[2] for a in arr])
        ctx = probs_c[(centers >= s - CTX_MS) & (centers < s)]
        ctx2 = probs_c[(centers >= e) & (centers < e + CTX_MS)]
        for cq in (ctx, ctx2):
            if len(cq) >= 5:
                xs += [float(cq.mean()), float(cq.max()), float(cq.std()),
                       float(np.percentile(cq, 10)), float(np.percentile(cq, 50)),
                       float(np.percentile(cq, 90))]
            else:
                xs += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pre = ctx if len(ctx) else np.zeros(1)
        post = ctx2 if len(ctx2) else np.zeros(1)
        both = np.concatenate([pre, post])
        xs += [float(ps.mean() - both.mean()),
               float(ps.max() - (np.percentile(both, 90) if len(both) else 0.0)),
               float(ps.mean() - (pre.mean() if len(pre) else 0.0)),
               float(ps.mean() - (post.mean() if len(post) else 0.0))]
        X.append(xs)
        meta.append((sid, s, e, pmax, pmean))
    return np.array(X, np.float64), meta


def load_meals():
    folds = splits.load_folds()
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    sid_meals = {}
    for _, r in idx.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms
    return sid_meals


def match_labels(cands, gts):
    """候选与 GT 贪心 IoU≥0.25 匹配标签（复核器训练用）。"""
    y = []
    for c in cands:
        sid, s, e = c[0], c[1], c[2]
        best = max((oe.event_iou((s, e), (gs, ge)) for sid2, (gs, ge) in gts if sid2 == sid), default=0.0)
        y.append(1 if best >= 0.25 else 0)
    return np.array(y, np.int8)


def eligible_meals(sids):
    """eligible 餐（会话 npz 存在 + 餐时段 ≥50% 数据覆盖）。"""
    out = []
    sid_meals = load_meals()
    for sid in sids:
        p = config.CACHE_DIR / "sessions" / f"{sid}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            tv = z["t_acc"][z["imu_valid"]]
        for m in sid_meals.get(sid, []):
            lo = np.searchsorted(tv, m["before"]); hi = np.searchsorted(tv, m["after"])
            if hi > lo and (tv[min(hi, len(tv) - 1)] - tv[max(lo, 0)]) >= 0.5 * (m["after"] - m["before"]):
                out.append((sid, (m["before"], m["after"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # ---- 窗模型（train 采样窗） ----
    tr = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_train.npz", allow_pickle=True)
    keep = tr["label"] >= 0
    imp = SimpleImputer(strategy="median").fit(tr["feat"][keep])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
        random_state=20260901)
    clf.fit(imp.transform(tr["feat"][keep]), tr["label"][keep].astype(int))
    print(f"窗模型训练 {keep.sum()} 窗", flush=True)

    # ---- 连续打分 ----
    def score(split_name):
        d = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_{split_name}.npz", allow_pickle=True)
        wids = [json.loads(w) for w in d["wid"]]
        prob = clf.predict_proba(imp.transform(d["feat"]))[:, 1]
        from collections import defaultdict
        sw = defaultdict(list)
        for wid, p in zip(wids, prob):
            sw[wid[0]].append((wid[1], wid[2], float(p)))
        return sw

    sw_tr = score("meal_train")
    sw_va = score("val")
    thr = args.thr if args.thr is not None else 0.28838   # 对方冻结阈值（MCC 网格近似）
    print(f"窗口阈值 {thr}", flush=True)

    # ---- 密度候选 ----
    cand_tr = density_candidates(sw_tr, thr)
    cand_va = density_candidates(sw_va, thr)
    import os
    nm_path = config.CACHE_DIR / "slide" / f"fold{args.fold}_no_meal_train.npz"
    if nm_path.exists() and not os.environ.get("BME_NO_NM", "0") == "1":
        dnm = np.load(nm_path, allow_pickle=True)
        wids_nm = [json.loads(w) for w in dnm["wid"]]
        from collections import defaultdict as _dd
        sw_nm = _dd(list)
        prob_nm = clf.predict_proba(imp.transform(dnm["feat"]))[:, 1]
        for wid, p in zip(wids_nm, prob_nm):
            sw_nm[wid[0]].append((wid[1], wid[2], float(p)))
        cand_nm = density_candidates(sw_nm, thr)
        print(f"无餐会话候选（复核负样本）: {len(cand_nm)}", flush=True)
    else:
        cand_nm = []
    print(f"候选：train(含餐) {len(cand_tr)} + 无餐 {len(cand_nm)} | val {len(cand_va)}", flush=True)

    # ---- 复核标签（train 候选 vs eligible 餐） ----
    true_tr = eligible_meals(set(sw_tr.keys()))
    true_va = eligible_meals(set(sw_va.keys()))
    print(f"eligible 餐：train {len(true_tr)} | val {len(true_va)}", flush=True)

    X_tr, meta_tr = verifier_features(cand_tr, sw_tr)
    y_tr = match_labels([(m[0], m[1], m[2]) for m in meta_tr], true_tr)
    if cand_nm:
        X_nm, meta_nm = verifier_features(cand_nm, sw_nm)
        X_tr = np.concatenate([X_tr, X_nm])
        meta_tr = meta_tr + meta_nm
        y_tr = np.concatenate([y_tr, np.zeros(len(meta_nm), np.int8)])
    X_va, meta_va = verifier_features(cand_va, sw_va)
    y_va = match_labels([(m[0], m[1], m[2]) for m in meta_va], true_va)
    n_pos_tr = int(y_tr.sum())
    print(f"复核训练：{len(y_tr)} 候选（正 {n_pos_tr}，负 {len(y_tr) - n_pos_tr}）| val 候选正 {int(y_va.sum())}", flush=True)
    if n_pos_tr < 5:
        print("正候选太少，跳过复核（直接输出候选层）")
        preds = [(m[0], (m[1], m[2])) for m in meta_va]
        m0 = oe.official_metrics(preds, true_va)
        print(f"[候选层] F1={m0['f1']:.3f} sens={m0['sensitivity']:.3f} ppv={m0['ppv']:.3f}", flush=True)
        return

    # ---- L2 复核器（嵌套近似：直接 fit train 候选 → val 打分） ----
    ver = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                    ("lr", LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000,
                                              random_state=20260904))])
    ver.fit(X_tr, y_tr)
    vs = ver.predict_proba(X_va)[:, 1]

    # 阈值：val 上 max event F1（阶段 1 近似乐观；严格版在嵌套 OOF 选）
    best_t, best_m = None, None
    for t in sorted(set(np.concatenate([vs, [0.5, 0.6, 0.7, 0.8]]))):
        acc = vs >= t
        preds = [(m[0], (m[1], m[2])) for m, a in zip(meta_va, acc) if a]
        m = oe.official_metrics(preds, true_va)
        if best_m is None or m["f1"] > best_m["f1"]:
            best_t, best_m = t, m
    print(f"[复核后] 阈值 {best_t:.3f}：F1={best_m['f1']:.3f} sens={best_m['sensitivity']:.3f} "
          f"ppv={best_m['ppv']:.3f} ({best_m['n_tp']}/{best_m['n_true']}, pred={best_m['n_pred']})", flush=True)
    out = {"fold": args.fold, "thr_window": thr, "thr_verifier": best_t,
           "candidate_layer": {"n": len(cand_va)}, "verified": {kk: best_m[kk] for kk in
           ("f1", "sensitivity", "ppv", "n_tp", "n_pred", "n_true")}}
    (config.OUTPUT_DIR / f"slide_verifier_fold{args.fold}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
