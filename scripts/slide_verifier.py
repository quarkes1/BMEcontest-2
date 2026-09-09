# -*- coding: utf-8 -*-
"""滑窗管线完整版：HGB 窗模型 → 密度候选 → 33 特征 L2 复核器 → 官方评估。

复核器训练：meal_train（train 含餐会话全窗候选：真 = IoU≥0.25 匹配 eligible 餐，
假 = 其余）；评估：val 全窗候选。
用法：D:/Anaconda3/envs/bme/python.exe scripts/slide_verifier.py --fold 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.data import manifests, splits
from src.pipeline.event_stack import (
    DensityConfig,
    EventRef,
    density_candidates as SHARED_DENSITY_CANDIDATES,
    select_event_threshold,
)
import official_iou_eval as oe

STRIDE_MS = 15_000
BRIDGE_MS = 60_000
DENSITY_MS = 600_000
MIN_POS = 10
COV_MIN = 0.80
MERGE_MS = 120_000
WIN_MS = 240_000
CTX_MS = 1_200_000
GLOBAL_PRIOR = np.array(   # 训练数据餐时刻先验（predict.py 同款）
    [0.174, 0.278, 0.546, 0.92, 0.889, 0.496, 0.187, 0.141, 0.408, 0.863,
     1.0, 0.681, 0.368, 0.216, 0.127, 0.073, 0.037, 0.012, 0.002, 0.0,
     0.0, 0.0, 0.0, 0.0], np.float32)


def density_candidates(sid_windows, thr, sids=None, min_pos=None, dens_ms=None,
                       bridge_ms=None):
    """滑窗概率 → 密度候选事件（window_support 边界 + 120s 合并）。返回 (sid, s, e, probs 数组, prob_max, prob_mean, dur_s)。
    min_pos/dens_ms/bridge_ms 可覆盖全局（参数扫描用）。"""
    selected = (
        sid_windows
        if sids is None
        else {sid: rows for sid, rows in sid_windows.items() if sid in sids}
    )
    shared = SHARED_DENSITY_CANDIDATES(
        selected,
        DensityConfig(
            stride_ms=STRIDE_MS,
            window_ms=WIN_MS,
            bridge_ms=BRIDGE_MS if bridge_ms is None else bridge_ms,
            density_ms=DENSITY_MS if dens_ms is None else dens_ms,
            min_positive=MIN_POS if min_pos is None else min_pos,
            coverage_min=COV_MIN,
            merge_ms=MERGE_MS,
            window_threshold=thr,
            context_ms=CTX_MS,
            coverage_fix=os.environ.get("BME_DENS_COVFIX", "0") == "1",
        ),
    )
    output = []
    for candidate in shared:
        probabilities = np.asarray(candidate.probabilities, dtype=np.float64)
        output.append(
            [
                candidate.event.sid,
                candidate.event.start_ms,
                candidate.event.end_ms,
                probabilities,
                float(probabilities.max()),
                float(probabilities.mean()),
                len(probabilities) * STRIDE_MS / 1000.0,
            ]
        )
    return output


def verifier_features(cands, sid_windows, tcn_scores=None):
    """33 特征（对方规格）+ 可选 TCN 深度分（事件内 max/mean，特征 34-35）。"""
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
        if os.environ.get("BME_VER_CTXN", "0") == "1":   # 实验：显式上下文窗计数（缺口邻域无上下文的"状态"特征）
            xs += [float(len(ctx)), float(len(ctx2))]
        pre = ctx if len(ctx) else np.zeros(1)
        post = ctx2 if len(ctx2) else np.zeros(1)
        both = np.concatenate([pre, post])
        xs += [float(ps.mean() - both.mean()),
               float(ps.max() - (np.percentile(both, 90) if len(both) else 0.0)),
               float(ps.mean() - (pre.mean() if len(pre) else 0.0)),
               float(ps.mean() - (post.mean() if len(post) else 0.0))]
        # TCN 深度分（特征 34-35）：事件覆盖窗的深度分 max/mean（无 TCN 数据→0 列对齐）
        tw_in = []
        if tcn_scores is not None:
            tw = tcn_scores.get(sid, [])
            tw_in = [v for ts, v in tw if ts >= s - 1000 and ts < e]
        xs += [float(max(tw_in)) if tw_in else 0.0,
               float(np.mean(tw_in)) if tw_in else 0.0]
        # 时刻先验（特征 36-37）：事件开始小时 + 全局时刻先验值（进食时刻分布强信号）
        hh = (s / 3.6e6) % 24
        xs += [float(hh), float(GLOBAL_PRIOR[int(hh) % 24])]
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


def eligible_meals(sids, sw=None):
    """eligible 餐（会话 npz 存在 + 餐时段数据覆盖 ≥50%；sw 给定时加"餐时段有滑窗覆盖 ≥120s"
    检查——与官方 IMU 质量合格口径对齐，数据碎片/缺口餐不罚检测器）。"""
    out = []
    sid_meals = load_meals()
    for sid in sids:
        p = config.CACHE_DIR / "sessions" / f"{sid}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            tv = z["t_acc"][z["imu_valid"]]
        win_arr = sorted(sw.get(sid, [])) if sw else []
        for m in sid_meals.get(sid, []):
            lo = np.searchsorted(tv, m["before"]); hi = np.searchsorted(tv, m["after"])
            if not (hi > lo and (tv[min(hi, len(tv) - 1)] - tv[max(lo, 0)]) >= 0.5 * (m["after"] - m["before"])):
                continue
            if sw is not None:
                ov = max((min(w[1], m["after"]) - max(w[0], m["before"])) for w in win_arr) if win_arr else 0
                if ov < 120_000:   # 餐时段无 ≥2min 窗覆盖 → 数据碎片不可达
                    continue
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

    # ---- TCN 深度分加载（窗模型融合特征 63-64；与 wid 对齐；BME_NO_TCN=1 禁用——
    #     dist 纯 CPU 推理版：TCN 分缺失时窗模型/复核保持一致性） ----
    no_tcn = os.environ.get("BME_NO_TCN", "0") == "1"

    def load_tcn_vec(split_name):
        if no_tcn:   # 纯 CPU 推理版：TCN 列不可用，保留时刻先验列（推理可算）
            d0 = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_{split_name}.npz", allow_pickle=True)
            out = np.zeros((len(d0["wid"]), 1), np.float32)
            for j, w in enumerate([json.loads(x) for x in d0["wid"]]):
                hh = int((w[1] / 3.6e6) % 24)
                out[j, 0] = GLOBAL_PRIOR[hh]
            return out
        p = config.CACHE_DIR / "slide" / f"fold{args.fold}_{split_name}_tcn.npz"
        if not p.exists():
            return None
        d0 = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_{split_name}.npz", allow_pickle=True)
        t = np.load(p)
        out = np.zeros((len(d0["wid"]), 2), np.float32)
        for j, sc in enumerate(t["score"]):
            if not np.isnan(sc):
                out[j, 0] = sc
        # 列 2：窗起点小时先验（62 特征外的补充）
        for j, w in enumerate([json.loads(x) for x in d0["wid"]]):
            hh = int((w[1] / 3.6e6) % 24)
            out[j, 1] = GLOBAL_PRIOR[hh]
        return out

    # ---- 窗模型（train 采样窗 + TCN 融合；BME_WBAG=1 时 5 折模型平均——弱折（fold2 型）
    #     对折内受试者迁移差，其他折模型平均候选 recall +0.12）
    #     注意：WBAG 的 5 折模型对折 k 的 val 有跨折受试者可见性（fold m≠k 模型训练过 S_k 会话）
    #     ——不是严格受试者互斥 bag（peer review 泄漏指控）。BME_SEEDS=n 修复：只在本折
    #     train 上训练 n 个模型，全部仅见过本折 train 受试者 → 严格互斥 bag。
    #     sklearn HGB 确定性（early_stopping=False 时 random_state 无效）→ 同数据同模型；
    #     BME_SEEDS_RESEED=1 时每种子从 meal/no_meal 全窗池按"≤3× 正/会话"重采样负样本
    #     → 真实多样 bag（与 train npz 构建同构）。 ----
    wbag = os.environ.get("BME_WBAG", "0") == "1"
    in_seeds = int(os.environ.get("BME_SEEDS", "0"))
    reseed = os.environ.get("BME_SEEDS_RESEED", "0") == "1"
    if in_seeds:
        models = []
        if reseed and in_seeds > 1:
            # 全窗池（本折 train 受试者：meal_train 含餐会话全窗 + no_meal_train 无餐会话全窗）
            from collections import defaultdict
            grp = defaultdict(list)   # sid -> 行号（先后接 meal_train / no_meal_train）
            X_pool_l, y_pool_l = [], []
            for sn in ("meal_train", "no_meal_train"):
                d = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_{sn}.npz", allow_pickle=True)
                wl = [json.loads(w) for w in d["wid"]]
                tv_l = load_tcn_vec(sn)
                Xs = d["feat"] if tv_l is None else np.concatenate([d["feat"], tv_l], 1)
                start = sum(len(x) for x in X_pool_l)   # 本 split 起始行号（拼接序）
                X_pool_l.append(Xs)
                y_pool_l.append(d["label"])
                for i, wj in enumerate(wl):
                    grp[wj[0]].append(start + i)
            X_pool = np.concatenate(X_pool_l)
            y_pool = np.concatenate(y_pool_l)
            for s in range(in_seeds):
                rng = np.random.default_rng(20260901 + s * 7919)
                rows = []
                for sid, ids in grp.items():
                    ids = np.array(ids)
                    labs = y_pool[ids]
                    pos = ids[labs == 1]
                    negs = ids[labs == 0]
                    if len(pos):
                        n_allow = max(3 * len(pos), 1)
                        if len(negs) > n_allow:
                            negs = rng.choice(negs, n_allow, replace=False)
                        rows.append(np.concatenate([pos, negs]))
                    elif len(negs):
                        rows.append(negs[rng.integers(len(negs)):rng.integers(len(negs)) + 1])
                sel = np.concatenate(rows)
                X_s, y_s = X_pool[sel], y_pool[sel].astype(int)
                imp_s = SimpleImputer(strategy="median").fit(X_s)
                clf_s = HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
                    min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
                    random_state=20260901)
                clf_s.fit(imp_s.transform(X_s), y_s)
                models.append((imp_s, clf_s))
            print(f"干净多样 bag：{in_seeds} 种子 × 负样本重采样（受试者互斥）", flush=True)
        else:
            tr = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_train.npz", allow_pickle=True)
            keep = tr["label"] >= 0
            tcn_tr_v = load_tcn_vec("train")
            X_tr_w = tr["feat"] if tcn_tr_v is None else np.concatenate([tr["feat"], tcn_tr_v], 1)
            for s in range(in_seeds):
                imp_s = SimpleImputer(strategy="median").fit(X_tr_w[keep])
                clf_s = HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
                    min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
                    random_state=20260901 + s)
                clf_s.fit(imp_s.transform(X_tr_w[keep]), tr["label"][keep].astype(int))
                models.append((imp_s, clf_s))
            print(f"干净 bag：本折 train 训练 {in_seeds} 种子（受试者互斥）", flush=True)
        wbag = True   # 下游走平均分支（模型均为本折 train 训练，val 受试者完全未见）
    elif wbag:
        models = []
        for kk in range(5):
            trk = np.load(config.CACHE_DIR / "slide" / f"fold{kk}_train.npz", allow_pickle=True)
            keepk = trk["label"] >= 0
            def ltv_k(sn, kk2=kk):
                if no_tcn:   # 纯 CPU 版：仅时刻先验列（训练/推理一致）
                    d0 = np.load(config.CACHE_DIR / "slide" / f"fold{kk2}_{sn}.npz", allow_pickle=True)
                    o = np.zeros((len(d0["wid"]), 1), np.float32)
                    for j, w in enumerate([json.loads(x) for x in d0["wid"]]):
                        hh = int((w[1] / 3.6e6) % 24)
                        o[j, 0] = GLOBAL_PRIOR[hh]
                    return o
                p = config.CACHE_DIR / "slide" / f"fold{kk2}_{sn}_tcn.npz"
                if not p.exists():
                    return None
                d0 = np.load(config.CACHE_DIR / "slide" / f"fold{kk2}_{sn}.npz", allow_pickle=True)
                t = np.load(p)
                o = np.zeros((len(d0["wid"]), 2), np.float32)
                for j, s in enumerate(t["score"]):
                    if not np.isnan(s):
                        o[j, 0] = s
                for j, w in enumerate([json.loads(x) for x in d0["wid"]]):
                    hh = int((w[1] / 3.6e6) % 24)
                    o[j, 1] = GLOBAL_PRIOR[hh]
                return o
            tv = ltv_k("train")
            Xk = trk["feat"] if tv is None else np.concatenate([trk["feat"], tv], 1)
            impk = SimpleImputer(strategy="median").fit(Xk[keepk])
            clfk = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
                min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
                random_state=20260901 + kk)
            clfk.fit(impk.transform(Xk[keepk]), trk["label"][keepk].astype(int))
            models.append((impk, clfk))
        print("窗模型 bag：5 折平均", flush=True)
    else:
        tr = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_train.npz", allow_pickle=True)
        keep = tr["label"] >= 0
        tcn_tr_v = load_tcn_vec("train")
        X_tr_w = tr["feat"] if tcn_tr_v is None else np.concatenate([tr["feat"], tcn_tr_v], 1)
        imp = SimpleImputer(strategy="median").fit(X_tr_w[keep])
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
            min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
            random_state=20260901)
        clf.fit(imp.transform(X_tr_w[keep]), tr["label"][keep].astype(int))
        print(f"窗模型训练 {keep.sum()} 窗（{'62+2 TCN 融合' if tcn_tr_v is not None else '62 特征'}）", flush=True)
        models = [(imp, clf)]

    # ---- 连续打分 ----
    def score(split_name):
        d = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_{split_name}.npz", allow_pickle=True)
        wids = [json.loads(w) for w in d["wid"]]
        tv_l = load_tcn_vec(split_name)
        Xw = d["feat"] if tv_l is None else np.concatenate([d["feat"], tv_l], 1)
        if wbag:
            ps = [clf2.predict_proba(imp2.transform(Xw))[:, 1] for imp2, clf2 in models]
            prob = np.mean(ps, 0)
        else:
            prob = clf.predict_proba(imp.transform(Xw))[:, 1]
        from collections import defaultdict
        sw = defaultdict(list)
        for wid, p in zip(wids, prob):
            sw[wid[0]].append((wid[1], wid[2], float(p)))
        return sw

    sw_tr = score("meal_train")
    sw_va = score("val")
    thr = args.thr if args.thr is not None else 0.28838   # 对方冻结阈值（MCC 网格近似）
    print(f"窗口阈值 {thr}", flush=True)

    # TCN 深度分（可选融合特征）：cache/slide/fold{k}_{split}_tcn.npz（与 wid 对齐）
    def load_tcn(split_name):
        if no_tcn:
            return None
        p = config.CACHE_DIR / "slide" / f"fold{args.fold}_{split_name}_tcn.npz"
        if not p.exists():
            return None
        d0 = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_{split_name}.npz", allow_pickle=True)
        t = np.load(p)
        wids0 = [json.loads(w) for w in d0["wid"]]
        from collections import defaultdict as _dd
        out = _dd(list)
        for w, sc in zip(wids0, t["score"]):
            if not np.isnan(sc):
                out[w[0]].append((w[1], float(sc)))
        return dict(out)

    tcn_tr = load_tcn("meal_train")
    tcn_va = load_tcn("val")

    # ---- 密度候选（BME_DENS_MP/MS 覆盖——低阈值/密网格救碎片餐） ----
    d_mp = int(os.environ.get("BME_DENS_MP", str(MIN_POS)))
    d_ms = int(os.environ.get("BME_DENS_MS", str(DENSITY_MS)))
    cand_tr = density_candidates(sw_tr, thr, min_pos=d_mp, dens_ms=d_ms)
    cand_va = density_candidates(sw_va, thr, min_pos=d_mp, dens_ms=d_ms)
    nm_path = config.CACHE_DIR / "slide" / f"fold{args.fold}_no_meal_train.npz"
    if nm_path.exists() and not os.environ.get("BME_NO_NM", "0") == "1":
        dnm = np.load(nm_path, allow_pickle=True)
        wids_nm = [json.loads(w) for w in dnm["wid"]]
        from collections import defaultdict as _dd
        sw_nm = _dd(list)
        tv_nm = load_tcn_vec("no_meal_train")
        Xnm_w = dnm["feat"] if tv_nm is None else np.concatenate([dnm["feat"], tv_nm], 1)
        if wbag:
            prob_nm = np.mean([clf2.predict_proba(imp2.transform(Xnm_w))[:, 1] for imp2, clf2 in models], 0)
        else:
            prob_nm = clf.predict_proba(imp.transform(Xnm_w))[:, 1]
        for wid, p in zip(wids_nm, prob_nm):
            sw_nm[wid[0]].append((wid[1], wid[2], float(p)))
        cand_nm = density_candidates(sw_nm, thr, min_pos=d_mp, dens_ms=d_ms)
        print(f"无餐会话候选（复核负样本）: {len(cand_nm)}", flush=True)
    else:
        cand_nm = []
    print(f"候选：train(含餐) {len(cand_tr)} + 无餐 {len(cand_nm)} | val {len(cand_va)}", flush=True)

    # ---- 复核标签（train 候选 vs eligible 餐） ----
    true_tr = eligible_meals(set(sw_tr.keys()), sw_tr)
    true_va = eligible_meals(set(sw_va.keys()), sw_va)
    print(f"eligible 餐：train {len(true_tr)} | val {len(true_va)}", flush=True)

    X_tr, meta_tr = verifier_features(cand_tr, sw_tr, tcn_tr)
    y_tr = match_labels([(m[0], m[1], m[2]) for m in meta_tr], true_tr)
    if cand_nm:
        X_nm, meta_nm = verifier_features(cand_nm, sw_nm)
        X_tr = np.concatenate([X_tr, X_nm])
        meta_tr = meta_tr + meta_nm
        y_tr = np.concatenate([y_tr, np.zeros(len(meta_nm), np.int8)])
    X_va, meta_va = verifier_features(cand_va, sw_va, tcn_va)
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
    if os.environ.get("BME_VER_HGB", "0") == "1":   # 实验：HGB 复核器（非线性）
        ver = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("hgb", HistGradientBoostingClassifier(
                            learning_rate=0.05, max_iter=200, max_leaf_nodes=8,
                            min_samples_leaf=20, l2_regularization=1.0,
                            early_stopping=False, random_state=20260904))])
    else:
        ver = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                        ("lr", LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000,
                                                  random_state=20260904))])
    ver.fit(X_tr, y_tr)
    vs = ver.predict_proba(X_va)[:, 1]

    # 仅作旧脚本诊断：阈值仍来自 val，不能作为 locked 指标或部署阈值。
    selection = select_event_threshold(
        [EventRef(m[0], m[1], m[2]) for m in meta_va],
        vs,
        [EventRef(sid, start, end) for sid, (start, end) in true_va],
    )
    best_t = selection.threshold
    best_m = {
        "f1": selection.metrics.f1,
        "sensitivity": selection.metrics.sensitivity,
        "ppv": selection.metrics.ppv,
        "n_tp": selection.metrics.n_tp,
        "n_pred": selection.metrics.n_pred,
        "n_true": selection.metrics.n_true,
    }
    print(f"[复核后] 阈值 {best_t:.3f}：F1={best_m['f1']:.3f} sens={best_m['sensitivity']:.3f} "
          f"ppv={best_m['ppv']:.3f} ({best_m['n_tp']}/{best_m['n_true']}, pred={best_m['n_pred']})", flush=True)
    out = {"fold": args.fold, "thr_window": thr, "thr_verifier": best_t,
           "threshold_role": "diagnostic_per_fold_optimum",
           "candidate_layer": {"n": len(cand_va)}, "verified": {kk: best_m[kk] for kk in
           ("f1", "sensitivity", "ppv", "n_tp", "n_pred", "n_true")}}
    (config.OUTPUT_DIR / f"slide_verifier_fold{args.fold}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 误差诊断（BME_DUMP=1）：每 GT 的分类（候选层漏/复核漏/阈值漏）+ FP 结构 ----
    if os.environ.get("BME_DUMP", "0") == "1" and len(meta_va):
        preds = [(m[0], (m[1], m[2])) for m, a in zip(meta_va, vs >= best_t) if a]
        ev_list = [(m[0], m[1], m[2], float(v)) for m, v in zip(meta_va, vs)]
        # 事件在 ev_list 的原始索引（pair 匹配用 preds 索引 → 映射回 ev_list）
        pass_idx = [k for k, (m, a) in enumerate(zip(meta_va, vs >= best_t)) if a]
        pairs = []
        for pi, i in enumerate(pass_idx):
            sid_p, (ps, pe) = preds[pi][0], preds[pi][1]
            for j, (sid_g, (gs, ge)) in enumerate(true_va):
                if sid_p == sid_g:
                    iou = oe.event_iou((ps, pe), (gs, ge))
                    if iou >= 0.25:
                        pairs.append((iou, pi, j))
        pairs.sort(key=lambda t: -t[0])
        used_p, used_g = set(), set()
        match = {}
        for iou, pi, j in pairs:
            if pi in used_p or j in used_g:
                continue
            used_p.add(pi); used_g.add(j)
            match[j] = (pi, iou)
        # 每 GT：覆盖最佳候选（meta_va 全候选含 <thr）
        gt_diag = []
        for j, (gsid, (gs, ge)) in enumerate(true_va):
            best_iou, best_v = 0.0, -1.0
            for (sid_c, cs, ce, vv) in ev_list:
                if sid_c != gsid:
                    continue
                io2 = oe.event_iou((cs, ce), (gs, ge))
                if io2 > best_iou:
                    best_iou, best_v = io2, vv
            if j in match:
                pi, iou = match[j]
                kind = "TP"
            elif best_iou >= 0.25:
                kind = "THR_MISS"      # 候选 IoU 达标但复核分 < 阈值
            elif best_iou > 0:
                kind = "VER_MISS"      # 候选层有覆盖但 IoU<0.25（框太偏/太窄）
            else:
                kind = "CAND_MISS"     # 候选层完全无覆盖
            gt_diag.append({"sid": gsid, "s": gs, "e": ge, "kind": kind,
                            "best_cand_iou": round(best_iou, 3), "best_cand_v": round(best_v, 3)})
        # FP 结构（未匹配事件）
        fp_diag = []
        for pi, (sid_p, (ps, pe)) in enumerate(preds):
            if pi in used_p:
                continue
            near = max((oe.event_iou((ps, pe), (gs, ge)) for sid_g, (gs, ge) in true_va if sid_g == sid_p), default=0.0)
            fp_diag.append({"sid": sid_p, "s": ps, "e": pe, "v": round(ev_list[pass_idx[pi]][3], 3),
                            "near_iou": round(near, 3), "dur_s": round((pe - ps) / 1000)})
        (config.OUTPUT_DIR / f"slide_diag_fold{args.fold}.json").write_text(
            json.dumps({"fold": args.fold, "thr": best_t, "gt": gt_diag, "fp": fp_diag},
                       ensure_ascii=False), encoding="utf-8")
        import collections
        kinds = collections.Counter(d["kind"] for d in gt_diag)
        print(f"[诊断] GT {len(gt_diag)}: {dict(kinds)} | FP {len(fp_diag)}", flush=True)
        # 候选级全量 dump（离线判别分析）：特征 + 复核分 + 与 GT 最佳 IoU
        lab_c = np.zeros(len(meta_va), np.int8)   # 1 = 匹配 GT / -1 标签由 IoU 定（0.25 阈值）
        iou_c = np.zeros(len(meta_va))
        for j, (gsid, (gs, ge)) in enumerate(true_va):
            for i2, (sid_c, cs, ce, _p1, _p2) in enumerate(meta_va):
                if sid_c != gsid:
                    continue
                io2 = oe.event_iou((cs, ce), (gs, ge))
                if io2 > iou_c[i2]:
                    iou_c[i2] = io2
        lab_c = (iou_c >= 0.25).astype(np.int8)
        np.savez_compressed(config.OUTPUT_DIR / f"slide_cand_fold{args.fold}.npz",
                            X=X_va.astype(np.float32), v=vs.astype(np.float32),
                            iou=iou_c.astype(np.float32), lab=lab_c,
                            thr=np.float32(best_t),
                            meta=np.array([json.dumps(m[:3]) for m in meta_va]))


if __name__ == "__main__":
    main()
