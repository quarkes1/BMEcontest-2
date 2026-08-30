# -*- coding: utf-8 -*-
"""阶段2：事件提案 + 排序头（检测即排序，不依赖密度模型）。

架构（S0-1 两池分治，2026-08-30）：
- 活动池：env×先验多参数连通域（复用 windows_to_events）→ 活动候选（窄、尖峰）
- 先验池：会话窗内整点 ±20min 时间窗候选（宽 40min，攻稀疏尖峰餐——餐内活动常
  <25% 餐长，IoU≥0.25 数学上无法由活动连通覆盖；45-62% 餐的 before 落在整点 ±15-20min）
- 两池独立训练 LightGBM 排序器（避免标签转移污染：贪心匹配把餐标给 IoU 更高的
  先验候选会把活动池正标签偷走，单池混合时 val 活动池排序崩溃 0.043 vs 0.018）
- 解码：活动池 top-k ∪ 先验池 top-1（先验分数 ≥pri_thr 且与活动已选 IoU<0.25 才追加），
  会话门控（AUC≈0.88）过滤无餐会话
- 评估：受试者级事件匹配 F1（IoU≥0.25 贪心，compute_metrics_by_subject）

运行：conda activate bme && python scripts/rank_events.py --fold 0
产物：outputs/rank_events_fold0.json"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.eval.metrics import compute_metrics_by_subject, event_iou
from src.infer.events import windows_to_events

OUT_CACHE = config.CACHE_DIR / "validate_baselines"
WINDOW_MS = 525_000          # 每窗时间跨度（5s 窗 @105Hz → 与 validate_baselines 一致）
SMOOTH = 31
PROP_PCT = (75, 82, 88, 92, 95)   # 提案追求召回（V2-B 最佳诚实配置 pct75 覆盖 19/38）
PROP_GAP = (30, 60)
PROP_DUR = (5, 10)
LOOSE_PCT = (50, 55, 62, 70, 75)  # S0-1 门控分流：高门控会话用更松阈值（攻弱信号/短餐）
LOOSE_GAP = (15, 30, 60)
LOOSE_DUR = (3, 5)
GATE_SPLIT = 0.5                  # 门控分流：gate_prob ≥ 此值 → loose 提案
PRIOR_HALF_W_S = 1200.0           # 先验候选半宽（20min→40min 窗；≤40min 才能与 15min 餐 IoU≥0.25）
PRIOR_MIN_W_S = 300.0             # 先验候选裁剪后最小宽（<5min 的边角碎片丢弃）
CTX_S = 1800.0             # 上下文对比窗（±30min）
IOU_LABEL = 0.25


def _prior(meal_before_hours, sigma_h=1.0):
    hist, _ = np.histogram(meal_before_hours, bins=24, range=(0, 24))
    prior = np.zeros(24, dtype=np.float32)
    for i, c in enumerate(hist):
        if c:
            prior += c * np.exp(-((np.arange(24) - i) ** 2) / (2 * sigma_h ** 2))
    return prior / prior.max()


def prior_candidates(sess_s, sess_e):
    """会话窗内全部整点 ±半宽 → 先验时间窗候选（与会话交集，≥PRIOR_MIN_W_S 保留）。

    餐的 before 45-62% 落在整点 ±15-20min（采集协议整点开餐），而餐内 IMU 活动常
    <25% 餐长（IoU≥0.25 数学上无法由活动连通覆盖）→ 整点铺先验窗。
    宽度必须 ≤40min：90min 半宽的教训——宽 180min 候选与 15min 餐最大 IoU=0.083。
    """
    cands = []
    base = sess_s - (sess_s % 3_600_000)        # 会话起点所在整点（UTC）
    n_h = int(np.ceil((sess_e - sess_s) / 3.6e6)) + 2
    h0 = int((sess_s / 3.6e6) % 24)
    for off in range(n_h):
        t_whole = base + off * 3.6e6            # 该小时的整点时刻
        s, e = int(t_whole - PRIOR_HALF_W_S * 1000), int(t_whole + PRIOR_HALF_W_S * 1000)
        s, e = max(s, sess_s), min(e, sess_e)
        if e - s >= PRIOR_MIN_W_S * 1000:
            cands.append((s, e, 1))
    return cands


def _merge(cands):
    """按起始排序，IoU>0.6 的相邻候选合并为更长者（先验标记 OR）。"""
    if not cands:
        return []
    cands = sorted(cands)
    merged = [list(cands[0])]
    for c in cands[1:]:
        if c[0] <= merged[-1][1] and event_iou((merged[-1][0], merged[-1][1]), (c[0], c[1])) > 0.6:
            merged[-1][1] = max(merged[-1][1], c[1])
            merged[-1][2] |= c[2]
        else:
            merged.append(list(c))
    return [(s, e, p) for s, e, p in merged]


def make_proposals(env, t0, prior, start_epoch, loose=False, no_prior=False, dilate_ms=0):
    """返回 (act_cands, pri_cands)：活动池（连通域）与先验池（整点窗）各自合并，两池不互并。
    loose=True 用更松阈值集（低 pct/短事件）；dilate_ms>0 时活动候选边界膨胀（短餐 IoU 修复：
    5-7min 短餐与候选边界偏移 1-2min 即掉出 IoU 0.25 匹配线，±60s 膨胀可救回）。"""
    h = (t0 / 3.6e6) % 24
    score = env * prior[np.clip(h.astype(int), 0, 23)]
    pcts, gaps, durs = (LOOSE_PCT, LOOSE_GAP, LOOSE_DUR) if loose else (PROP_PCT, PROP_GAP, PROP_DUR)
    act = []
    for pct in pcts:
        for gap in gaps:
            for dur in durs:
                evs = windows_to_events(score, t0, t0 + WINDOW_MS, float(np.percentile(score, pct)),
                                        merge_gap_s=gap, min_dur_s=dur, smooth_win=SMOOTH)
                act.extend((s, e, 0) for s, e in evs)
    act = _merge(act)
    if dilate_ms > 0:
        t_beg, t_end = int(t0.min()), int(t0[0]) + len(env) * 1000
        act = [(max(t_beg, int(s - dilate_ms)), min(t_end, int(e + dilate_ms)), ip)
               for s, e, ip in act]
    pri = [] if no_prior else prior_candidates(start_epoch, start_epoch + (len(env) - 1) * 1000)
    return act, _merge(pri)


def candidate_features(cands, env, t0, prior, sess_feats, gate_prob=0.5):
    """候选事件 → 特征矩阵（n_cand, 14）+ 标签无关（is_prior 为第 14 列）。"""
    X = []
    for s, e, is_prior in cands:
        dur = (e - s) / 1000.0
        seg = (t0 >= s) & (t0 < e)
        if seg.sum() < 2:
            continue
        sc = env[seg] * prior[np.clip(((t0[seg] / 3.6e6) % 24).astype(int), 0, 23)]
        peak = float(sc.max()); mean = float(sc.mean()); std = float(sc.std())
        p90 = float(np.percentile(sc, 90))
        ctx = (t0 >= s - CTX_S * 1000) & (t0 < e + CTX_S * 1000) & ~seg
        ctx_mean = float(env[ctx].mean()) if ctx.sum() > 10 else mean
        hh = (s / 3.6e6) % 24
        X.append([
            dur, peak, mean, std, p90,
            peak / (mean + 1e-6),                      # 形状：尖峰性
            float(prior[int(hh) % 24]),                # 时刻先验
            mean / (ctx_mean + 1e-6),                  # 上下文对比
            sess_feats["dur_h"], sess_feats["env_p95"], sess_feats["p95_ratio"],
            (s - sess_feats["start"]) / (sess_feats["dur_h"] * 3.6e6 + 1e-6),  # 会话内相对位置
            gate_prob,                                # 会话级"有无餐"门控概率（V1, AUC≈0.88）
            float(is_prior),                          # 先验窗来源标记
        ])
    return np.array(X, dtype=np.float32) if X else np.zeros((0, 14), dtype=np.float32)


def match_labels(cands, meals):
    """候选与真值贪心 IoU 匹配（与竞赛口径一致的匹配器）。"""
    y = np.zeros(len(cands), dtype=np.int8)
    pairs = [(event_iou((c[0], c[1]), (m["before"], m["after"])), i, j)
             for i, c in enumerate(cands)
             for j, m in enumerate(meals)
             if event_iou((c[0], c[1]), (m["before"], m["after"])) >= IOU_LABEL]
    pairs.sort(key=lambda x: -x[0])
    used_c, used_m = set(), set()
    for _, i, j in pairs:
        if i in used_c or j in used_m:
            continue
        used_c.add(i); used_m.add(j); y[i] = 1
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--no-prior", action="store_true", help="禁用先验池（对照）")
    ap.add_argument("--strict", action="store_true", help="禁用 loose 门控分流（全 PROP 提案，对照）")
    ap.add_argument("--loose", action="store_true",
                    help="全 loose 模式：所有会话用更松提案阈值（对照组，默认门控分流）")
    args = ap.parse_args()
    k = args.fold
    folds = splits.load_folds()
    f = folds[k]
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    starts = {r["session_id"]: int(r["timeStamp.startTime"]) for _, r in idx.iterrows()}
    subject_of = dict(zip(idx["session_id"], idx["externalid"]))

    sid_meals = {}
    for _, r in idx.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms

    prior = _prior(np.array([m["before"] / 3.6e6 % 24
                             for s in f["train_sessions"] if s in sid_meals for m in sid_meals[s]]))
    print(f"fold{k} 先验峰值小时:", int(np.argmax(prior)), flush=True)

    import lightgbm as lgb
    GATE_FEATS = ["dur_h", "env_mean", "env_p50", "env_p95", "env_std", "p95_ratio", "start_h_utc"]

    # ---- 会话级"有无餐"门控（V1，AUC≈0.88，train→val 不泄漏） ----
    def _load_feats(sid):
        p = OUT_CACHE / f"{sid}.npz"
        if not p.exists():
            return None
        return json.loads(np.load(p)["feats"].item())

    gate_feats = {}   # sid → [7 个统计特征]
    for split, sessions in (("tr", f["train_sessions"]), ("va", f["val_sessions"])):
        for sid in sessions:
            ft = _load_feats(sid)
            if ft is not None:
                gate_feats[sid] = [ft[kk] for kk in GATE_FEATS]
    Xg_tr = np.array([gate_feats[s] for s in f["train_sessions"] if s in gate_feats])
    yg_tr = np.array([1 if s in sid_meals else 0 for s in f["train_sessions"] if s in gate_feats])
    Xg_va = np.array([gate_feats[s] for s in f["val_sessions"] if s in gate_feats])
    gate = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                              min_child_samples=20, verbosity=-1)
    gate.fit(Xg_tr, yg_tr)
    # 只对 val 会话打分（train 特征恒 0.5，无需门控分）；zip 截断防缺 feats 会话错位
    gate_prob = {}
    for sid, x in zip(f["val_sessions"], Xg_va):
        gate_prob[sid] = float(gate.predict_proba(x[None])[0, 1])
    print(f"  会话门控: train {len(Xg_tr)} + val {len(Xg_va)} 会话打分完成（AUC≈0.88 见 validate_baselines）", flush=True)

    # ---- 提案生成 + 特征（train / val）：活动池与先验池分开 ----
    Xa_tr, ya_tr, Xp_tr, yp_tr = [], [], [], []
    Xa_va, Xp_va = [], []
    sess_meta = {}    # sid → (env, t0, sess_feats)
    prop_stats = {"tr_prop": 0, "tr_meals": 0, "tr_covered": 0,
                  "va_prop": 0, "va_meals": 0, "va_covered": 0}
    cov_detail = []   # val 每餐覆盖明细（时刻/信号/门控/来源）
    for split, sessions in (("tr", f["train_sessions"]), ("va", f["val_sessions"])):
        for sid in sessions:
            p = OUT_CACHE / f"{sid}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            env = d["env"].astype(np.float32)
            t0 = d["t0"].astype(np.int64)
            ft = json.loads(d["feats"].item())
            start = starts.get(sid, 0)
            sess_feats = {"dur_h": float(ft["dur_h"]), "env_p95": float(ft["env_p95"]),
                          "p95_ratio": float(ft["p95_ratio"]), "start": start}
            sess_meta[sid] = (env, t0, sess_feats)
            gp = gate_prob.get(sid, 0.5)
            act, pri = make_proposals(env, t0, prior, start, loose=args.loose, no_prior=args.no_prior)
            meals = sid_meals.get(sid, [])
            cands_all = act + pri
            n_covered = sum(1 for m in meals
                            if any(event_iou((c[0], c[1]), (m["before"], m["after"])) >= IOU_LABEL for c in cands_all))
            key = f"{split}_"
            prop_stats[key + "prop"] += len(cands_all)
            prop_stats[key + "meals"] += len(meals)
            prop_stats[key + "covered"] += n_covered
            if split == "va":
                for m in meals:
                    hh = (m["before"] / 3.6e6) % 24
                    seg = (t0 >= m["before"]) & (t0 < m["after"])
                    ratio = float(env[seg].max() / (np.percentile(env, 95) + 1e-6)) if seg.sum() > 5 else 0.0
                    ca = any(event_iou((c[0], c[1]), (m["before"], m["after"])) >= IOU_LABEL for c in act)
                    cp = any(event_iou((c[0], c[1]), (m["before"], m["after"])) >= IOU_LABEL for c in pri)
                    cov_detail.append({"sid": sid, "hour": round(float(hh), 1),
                                       "prior": float(prior[int(hh) % 24]),
                                       "dur_min": round((m["after"] - m["before"]) / 60000, 1),
                                       "ratio": round(ratio, 2), "gate_prob": round(float(gp), 2),
                                       "covered": int(ca or cp), "by_act": int(ca), "by_pri": int(cp)})
            if act:
                Xa = candidate_features(act, env, t0, prior, sess_feats, 0.5)
                if split == "tr":
                    Xa_tr.append(Xa); ya_tr.append(match_labels(act, meals))
                else:
                    Xa_va.append(Xa)
            if pri:
                Xp = candidate_features(pri, env, t0, prior, sess_feats, 0.5)
                if split == "tr":
                    Xp_tr.append(Xp); yp_tr.append(match_labels(pri, meals))
                else:
                    Xp_va.append(Xp)
    Xa_tr = np.concatenate(Xa_tr); ya_tr = np.concatenate(ya_tr)
    Xp_tr = np.concatenate(Xp_tr) if Xp_tr else np.zeros((0, 14), np.float32)
    yp_tr = np.concatenate(yp_tr) if yp_tr else np.zeros(0, np.int8)
    Xa_va = np.concatenate(Xa_va)
    Xp_va = np.concatenate(Xp_va) if Xp_va else np.zeros((0, 14), np.float32)
    print(f"  活动池: train {len(Xa_tr)}（正 {ya_tr.sum()}）| val {len(Xa_va)}", flush=True)
    print(f"  先验池: train {len(Xp_tr)}（正 {yp_tr.sum()}）| val {len(Xp_va)}", flush=True)
    print(f"  提案覆盖餐: train {prop_stats['tr_covered']}/{prop_stats['tr_meals']} "
          f"val {prop_stats['va_covered']}/{prop_stats['va_meals']}", flush=True)

    # ---- 两池 LightGBM 排序器 ----
    def _fit(X, y):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                 min_child_samples=10,
                                 scale_pos_weight=max(1.0, (len(y) - y.sum()) / max(y.sum(), 1)),
                                 verbosity=-1)
        clf.fit(X, y)
        return clf

    clf_act = _fit(Xa_tr, ya_tr)
    clf_pri = _fit(Xp_tr, yp_tr) if len(Xp_tr) else None
    pa = clf_act.predict_proba(Xa_va)[:, 1]
    pp = clf_pri.predict_proba(Xp_va)[:, 1] if clf_pri is not None else np.zeros(len(Xp_va))
    print("  活动池特征重要性:", [f"f{i}" for i in np.argsort(clf_act.feature_importances_)[::-1]][:6], flush=True)
    if clf_pri is not None:
        print("  先验池特征重要性:", [f"f{i}" for i in np.argsort(clf_pri.feature_importances_)[::-1]][:6], flush=True)

    # ---- 解码：会话门控 × 活动 top-k × 先验 top-1 网格（受试者级匹配） ----
    true_all = [(s, m["before"], m["after"]) for s in f["val_sessions"] if s in sid_meals for m in sid_meals[s]]
    true_sid = [(s, (b, e)) for s, b, e in true_all]
    # 每个 val 会话的候选与分数（按 sid 顺序切片）
    val_rows = []   # (sid, act_cands, pa_slice, pri_cands, pp_slice)
    ia = ip = 0
    for sid in f["val_sessions"]:
        if sid not in sess_meta:
            continue
        env, t0, sess_feats = sess_meta[sid]
        act, pri = make_proposals(env, t0, prior, starts.get(sid, 0),
                                  loose=args.loose,
                                  no_prior=args.no_prior)
        n_a, n_p = len(act), len(pri)
        val_rows.append((sid, act, pa[ia:ia + n_a], pri, pp[ip:ip + n_p]))
        ia += n_a; ip += n_p

    rows, best_row = [], None
    for thr_g in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7):
        for topk in (1, 2, 3, 4, 5):
            for thr_p in (0.0, 0.2, 0.3, 0.5):
                pred_sid = []
                for sid, act, sa, pri, sp in val_rows:
                    preds = []
                    if gate_prob.get(sid, 1.0) >= thr_g:
                        sel = np.argsort(sa)[-topk:] if len(sa) else []
                        preds = [(sid, (act[j][0], act[j][1])) for j in sel]
                        # 先验池 top-1：分数 ≥thr_p 且与活动已选 IoU<0.25 才追加（避免双报同一餐）
                        if clf_pri is not None and len(sp) and sp.max() >= thr_p:
                            jp = int(np.argmax(sp))
                            pc = (pri[jp][0], pri[jp][1])
                            if not any(event_iou(pc, (s0, e0)) >= IOU_LABEL for _, (s0, e0) in preds):
                                preds.append((sid, pc))
                    # 先验池高分旁路：pri_score≥0.85 时绕过会话门控直接输出
                    # （先验池负样本 p90≈0.03，≥0.85 几乎必为真餐——稀疏尖峰餐所在会话
                    #  门控概率低被过滤，先验窗是唯一可达覆盖）
                    if clf_pri is not None and len(sp) and sp.max() >= 0.85:
                        jp = int(np.argmax(sp))
                        pc = (pri[jp][0], pri[jp][1])
                        if not any(event_iou(pc, (s0, e0)) >= IOU_LABEL for _, (s0, e0) in preds):
                            preds.append((sid, pc))
                    pred_sid.extend(preds)
                m = compute_metrics_by_subject(pred_sid, true_sid, lambda s: subject_of[s])
                row = {"name": f"g{thr_g}_k{topk}_p{thr_p}", **{kk: m[kk] for kk in ("f1", "sensitivity", "ppv", "n_tp", "n_pred", "n_true")}}
                rows.append(row)
                if best_row is None or m["f1"] > best_row[1]["f1"]:
                    best_row = (row["name"], m)
    for row in rows:
        print(f"  {row['name']}: F1={row['f1']:.3f} sens={row['sensitivity']:.3f} ppv={row['ppv']:.3f} "
              f"({row['n_tp']}/{row['n_true']}, pred={row['n_pred']})", flush=True)
    print(f"  ★ 最佳: {best_row[0]} F1={best_row[1]['f1']:.3f} sens={best_row[1]['sensitivity']:.3f} "
          f"ppv={best_row[1]['ppv']:.3f} ({best_row[1]['n_tp']}/{best_row[1]['n_true']}, pred={best_row[1]['n_pred']})", flush=True)

    out = {"fold": k, "mode": "no_prior" if args.no_prior else ("loose" if args.loose else "gate_split"),
           "rows": rows, "best": {"name": best_row[0], **{kk: best_row[1][kk] for kk in ("f1", "sensitivity", "ppv", "n_tp", "n_pred", "n_true")}},
           "proposal_coverage": prop_stats, "cov_detail": cov_detail,
           "baselines": {"v2b_heuristic_subj": 0.085, "densitynet_subj": 0.020},
           "n_train_act": len(Xa_tr), "n_train_pri": len(Xp_tr)}
    (config.OUTPUT_DIR / f"rank_events_fold{k}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"→ outputs/rank_events_fold{k}.json", flush=True)


if __name__ == "__main__":
    main()
