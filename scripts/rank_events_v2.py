# -*- coding: utf-8 -*-
"""rank_events v2：多模态分层融合解码（方向 1 落地 + 方向 3/4）。

对比 v1（rank_events.py）的变更：
1. 活动池排序 = LightGBM 14 特征分数 ⊕ MM-Ranker 深度分数（加权融合，w 网格搜索）
   ——深度模型是"IMU 手势提案 → PPG+IMU 联合二次校验"分层中的细粒度校验器
2. 解码：自动阈值搜索（方向 4）——融合分数 τ × 权重 w × 会话门控 g × 先验阈值 thr_p
   × 膨胀 dilation 网格，validation 按受试者级 F1 选最优，替代手工 g×topk×p
3. 事件级后处理（方向 3）：活动事件合并（gap<120s）→ 过滤 <2min 孤立段 → 边界膨胀
4. 先验池通道不变（时间先验，无信号窗口 → 不做深度校验）；w=0 网格可复现 v1 的
   top-k 解码（τ 代替 top-k 语义）

运行：source activate bme && python scripts/rank_events_v2.py --fold 0 [--prior-grid 15m|60m]
产物：outputs/rank_events_v2_fold{k}_{grid}.json
"""
import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.data import manifests, splits
from src.eval.metrics import compute_metrics_by_subject, event_iou
import rank_events as v1

OUT_CACHE = config.CACHE_DIR / os.environ.get("BME_ENV_CACHE", "validate_baselines")
IOU_LABEL = 0.25
POST_MERGE_GAP_S = 120.0    # 事件合并 gap（<2min 断开视为同一餐）
POST_MIN_DUR_S = 120.0      # 过滤 <2min 孤立段（方向 3）
PRIOR_GRID_S = 900          # 先验网格步长 15min
PRIOR_HALF_W_S = 450        # 先验窗半宽 15min（31% 餐 <10min，40min 窗 IoU=D/40<0.25 数学不可达
                            # → 15min 窗覆盖 88% vs 49%；覆盖分析 scripts/analysis_prior_grid.py）
DL_W = (0.0, 0.3, 0.5, 0.7, 1.0)      # 深度分数融合权重（0=纯 LightGBM 对照）
DL_TAU = tuple(np.arange(0.15, 0.425, 0.025))  # 活动融合分数阈值（细化：覆盖保守分数区 0.15-0.40）
GATE_GS = (0.0, 0.15, 0.3, 0.45)  # 排序头已含 gate 元特征 → 解码门控可放宽，细化网格找新平衡
PRI_THRS = (0.3, 0.5, 0.7)  # 先验池 top-1 阈值（96 窗/天：0.3 可能过松，加 0.7 档）
DILATIONS = (60.0, 120.0)
TOPK = (0, 1, 2, 3, 4)                # 会话级最多事件数（0=不限；一天 3-5 餐的现实约束，砍 FP）


def load_dl_scores(k, norm="none"):
    """读 MM-Ranker val 候选分数 → {(sid, s, e): score}。

    norm='minmax'：用本折全部 val 候选的 5-95 分位做 min-max 归一（config 选择
    本就在 val 上进行，无新增泄漏）。目的：消除逐折深度分尺度漂移——fold1 负样本
    中位 0.151 vs fold0 0.050，固定 w 融合在逐折间含义不一致；归一化使 w 可比。
    """
    p = config.OUTPUT_DIR / f"mm_ranker_fold{k}_val.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    score = z["score"].astype(np.float64)
    meta = [json.loads(m.decode()) for m in z["meta"]]
    if norm == "minmax":
        lo, hi = float(np.percentile(score, 5)), float(np.percentile(score, 95))
        score = np.clip((score - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return {(m["sid"], int(m["s"]), int(m["e"])): float(sc)
            for m, sc in zip(meta, score)}


def postprocess_events_prov(evs, idxs=None, merge_gap_s=POST_MERGE_GAP_S,
                            min_dur_s=POST_MIN_DUR_S, dilation_s=60.0):
    """事件级形态学后处理（方向 3）：按起始排序合并相邻 → 过滤孤立短段 → 边界膨胀。
    返回 (事件, 每事件对应的原始索引列表)；idxs=None 时事件原样编号。"""
    if not evs:
        return [], []
    if idxs is None:
        idxs = list(range(len(evs)))
    merged, merged_i = [], []
    for (s, e), i in sorted(zip(evs, idxs), key=lambda t: t[0][0]):
        if merged and s - merged[-1][1] <= merge_gap_s * 1000:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            merged_i[-1].append(i)
        else:
            merged.append((s, e))
            merged_i.append([i])
    out, out_i = [], []
    for (s, e), ii in zip(merged, merged_i):
        if (e - s) / 1000.0 >= min_dur_s:
            out.append((max(0, int(s - dilation_s * 1000)), int(e + dilation_s * 1000)))
            out_i.append(ii)
    return out, out_i


def postprocess_events(evs, merge_gap_s=POST_MERGE_GAP_S, min_dur_s=POST_MIN_DUR_S,
                       dilation_s=60.0):
    out, _ = postprocess_events_prov(evs, None, merge_gap_s, min_dur_s, dilation_s)
    return out


def decode_session(row, gate_prob, cfg, clf_pri):
    """单会话解码 → (预测 [(sid,(s,e))...], 明细 [(sid,s,e,src,score)]).

    src: 'act'（活动池，score=活动融合分）| 'pri'（先验池，score=先验分）。
    先验池 top-2 非重叠窗（多餐会话的第 2 餐靠它）；修复：旧 0.85 旁路复用
    argmax(sp) → 与 top-1 同窗，IoU 去重恒跳过（死代码）。"""
    sid, act, sa, va_scores, pri, sp = row
    w, tau, thr_g, thr_p, dil, K = cfg
    act_out, pri_out = [], []
    fuse = None
    if gate_prob.get(sid, 1.0) >= thr_g and len(sa):
        if w > 0:
            fuse = w * va_scores + (1 - w) * sa
            fuse = np.where(np.isnan(va_scores), sa, fuse)  # 缺深度分→纯LGBM
        else:
            fuse = sa
        sel = np.where(fuse >= tau)[0]
        if K > 0 and len(sel) > K:  # 会话级 top-K（非重叠贪心：与已选窗 IoU≥0.5 的窗不计名额）
            order = np.argsort(fuse[sel])[::-1]
            picked = []
            for j in order:
                c = act[sel[j]][:2]
                if any(event_iou(c, act[sel[p]][:2]) >= 0.5 for p in picked):
                    continue
                picked.append(j)
                if len(picked) >= K:
                    break
            sel = sel[np.array(picked)]
        act_out = [(sid, (act[j][0], act[j][1])) for j in sel]
    # 先验通道：门控与活动池同权（低门控会话不发先验事件，避免无餐会话 FP）
    # 逐窗阈值：每个被选先验窗都须 sp ≥ thr_p（旧旁路只查 sp.max()，第 2 窗可低至 0.006）
    if gate_prob.get(sid, 1.0) >= thr_g and clf_pri is not None and len(sp) and sp.max() >= thr_p:
        act_evs = [(s0, e0) for _, (s0, e0) in act_out]
        pri_evs = [e for _, e in pri_out]  # pri_out 是 (sid,(s,e)) 元组
        for jp in np.argsort(sp)[::-1]:
            if sp[jp] < thr_p:
                break  # 降序扫描，低于阈值即止
            pc = (pri[jp][0], pri[jp][1])
            if any(event_iou(pc, e) >= IOU_LABEL for e in act_evs + pri_evs):
                continue
            pri_out.append((sid, pc))
            if len(pri_out) >= 2:
                break
    # 事件级后处理只作用于活动事件（先验窄窗不参与合并）
    keep, prov = postprocess_events_prov([e for _, e in act_out], list(range(len(act_out))),
                                         dilation_s=dil)
    pred = [(sid, e) for e in keep] + pri_out
    detail = []
    if fuse is not None:
        for e, ii in zip(keep, prov):
            if len(ii):
                detail.append((sid, e[0], e[1], "act", float(np.max(fuse[sel[ii]]))))
    for (sid2, (s0, e0)), jp in zip(pri_out, np.argsort(sp)[::-1][:len(pri_out)]):
        detail.append((sid2, s0, e0, "pri", float(sp[jp])))
    return pred, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--prior-grid", choices=("15m", "60m"), default="15m",
                    help="先验网格：15m=15min/15min 窗（覆盖短餐）；60m=整点/40min 窗（v1 基线）")
    ap.add_argument("--detail", action="store_true",
                    help="输出最佳配置逐事件明细（src/score/tp，FP 来源分析）")
    ap.add_argument("--norm-score", choices=("none", "minmax"), default="none",
                    help="深度分逐折归一化（minmax=5-95 分位，消除折间尺度漂移）")
    args = ap.parse_args()
    k = args.fold
    # 局部变量而非直接改模块常量：函数内任何赋值都会把名字变成局部，
    # 15m 分支不赋值时读取会 UnboundLocalError（2026-08-30 教训）
    grid_s, half_w_s = PRIOR_GRID_S, PRIOR_HALF_W_S
    if args.prior_grid == "60m":
        grid_s, half_w_s = 3600, 1200
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
    prior = v1._prior(np.array([m["before"] / 3.6e6 % 24
                                for s in f["train_sessions"] if s in sid_meals for m in sid_meals[s]]))

    # ---- 会话门控 + 两池 LightGBM（与 v1 相同） ----
    import lightgbm as lgb
    GATE_FEATS = ["dur_h", "env_mean", "env_p50", "env_p95", "env_std", "p95_ratio", "start_h_utc"]
    gate_feats = {}
    for sid in f["train_sessions"] + f["val_sessions"]:
        p = OUT_CACHE / f"{sid}.npz"
        if p.exists():
            ft = json.loads(np.load(p)["feats"].item())
            gate_feats[sid] = [ft[kk] for kk in GATE_FEATS]
    Xg_tr = np.array([gate_feats[s] for s in f["train_sessions"] if s in gate_feats])
    yg_tr = np.array([1 if s in sid_meals else 0 for s in f["train_sessions"] if s in gate_feats])
    gate = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                              min_child_samples=20, verbosity=-1)
    gate.fit(Xg_tr, yg_tr)
    gate_prob = {}
    for sid in f["val_sessions"]:
        if sid in gate_feats:
            gate_prob[sid] = float(gate.predict_proba(np.array(gate_feats[sid])[None])[0, 1])

    Xa_tr, ya_tr, Xp_tr, yp_tr = [], [], [], []
    Xa_va, Xp_va = [], []
    sess_meta = {}
    # z 完整性预检查：env_z 缺失/长度不符 → 全链路退化 14 维（避免 14/17 混 concat）
    z_ok = True
    for sid in f["train_sessions"] + f["val_sessions"]:
        p = OUT_CACHE / f"{sid}.npz"
        if p.exists():
            dz = np.load(p)
            if "env_z" not in dz.files or len(dz["env_z"]) != len(dz["env"]):
                z_ok = False
                break
    if not z_ok:
        print("  [warn] env_z 缺失/不一致 → 特征退化为 14 维（无 z 通道）", flush=True)
    for split, sessions in (("tr", f["train_sessions"]), ("va", f["val_sessions"])):
        for sid in sessions:
            p = OUT_CACHE / f"{sid}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            env, t0 = d["env"].astype(np.float32), d["t0"].astype(np.int64)
            zseq = d["env_z"].astype(np.float32) if z_ok else None
            ft = json.loads(d["feats"].item())
            sess_feats = {"dur_h": float(ft["dur_h"]), "env_p95": float(ft["env_p95"]),
                          "p95_ratio": float(ft["p95_ratio"]), "start": starts.get(sid, 0)}
            sess_meta[sid] = (env, zseq, t0, sess_feats)
            act, pri = v1.make_proposals(env, t0, prior, starts.get(sid, 0), dilate_ms=60000,
                                         prior_grid_s=grid_s, prior_half_w_s=half_w_s)
            meals = sid_meals.get(sid, [])
            if act:
                Xa = v1.candidate_features(act, env, t0, prior, sess_feats, 0.5)
                if split == "tr":
                    Xa_tr.append(Xa); ya_tr.append(v1.match_labels(act, meals))
                else:
                    Xa_va.append(Xa)
            if pri:
                Xp = v1.candidate_features(pri, env, t0, prior, sess_feats, 0.5)
                if split == "tr":
                    Xp_tr.append(Xp); yp_tr.append(v1.match_labels(pri, meals))
                else:
                    Xp_va.append(Xp)
    n_feat = 17 if any(z is not None for _, z, _, _ in sess_meta.values()) else 14
    Xa_tr, ya_tr = np.concatenate(Xa_tr), np.concatenate(ya_tr)
    Xp_tr = np.concatenate(Xp_tr) if Xp_tr else np.zeros((0, n_feat), np.float32)
    yp_tr = np.concatenate(yp_tr) if yp_tr else np.zeros(0, np.int8)
    Xa_va = np.concatenate(Xa_va)
    Xp_va = np.concatenate(Xp_va) if Xp_va else np.zeros((0, n_feat), np.float32)

    def _fit(X, y):
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=10,
                                  scale_pos_weight=max(1.0, (len(y) - y.sum()) / max(y.sum(), 1)),
                                  verbosity=-1).fit(X, y)

    clf_act = _fit(Xa_tr, ya_tr)
    clf_pri = _fit(Xp_tr, yp_tr) if len(Xp_tr) else None
    pa = clf_act.predict_proba(Xa_va)[:, 1]
    pp = clf_pri.predict_proba(Xp_va)[:, 1] if clf_pri is not None else np.zeros(len(Xp_va))

    # ---- 深度分数对齐（按 (sid, s, e) 精确匹配；缺失候选退化为纯 LGBM） ----
    dl = load_dl_scores(k, args.norm_score)
    val_rows = []
    ia = ip = 0
    n_hit = n_tot = 0
    for sid in f["val_sessions"]:
        if sid not in sess_meta:
            continue
        env, _zseq, t0, sess_feats = sess_meta[sid]
        act, pri = v1.make_proposals(env, t0, prior, starts.get(sid, 0), dilate_ms=60000,
                                     prior_grid_s=grid_s, prior_half_w_s=half_w_s)
        n_a, n_p = len(act), len(pri)
        va_scores = np.full(n_a, np.nan, np.float32)
        for j, c in enumerate(act):
            n_tot += 1
            if dl is not None and (sid, c[0], c[1]) in dl:
                va_scores[j] = dl[(sid, c[0], c[1])]
                n_hit += 1
        val_rows.append((sid, act, pa[ia:ia + n_a], va_scores, pri, pp[ip:ip + n_p]))
        ia += n_a; ip += n_p
    print(f"  深度分数对齐: {n_hit}/{n_tot} ({n_hit / max(n_tot, 1) * 100:.0f}%)", flush=True)

    true_all = [(s, m["before"], m["after"]) for s in f["val_sessions"] if s in sid_meals for m in sid_meals[s]]
    true_sid = [(s, (b, e)) for s, b, e in true_all]

    rows, best_row = [], None
    t0 = time.time()
    for w in DL_W:
        for tau in DL_TAU:
            for thr_g in GATE_GS:
                for thr_p in PRI_THRS:
                    for dil in DILATIONS:
                        for K in TOPK:
                            pred_sid = []
                            for row in val_rows:
                                pred_sid.extend(decode_session(
                                    row, gate_prob, (w, tau, thr_g, thr_p, dil, K), clf_pri)[0])
                            m = compute_metrics_by_subject(pred_sid, true_sid, lambda s: subject_of[s])
                            row = {"name": f"w{w}_t{round(tau, 3)}_g{thr_g}_p{thr_p}_d{dil:.0f}_k{K}",
                                   **{kk: m[kk] for kk in ("f1", "sensitivity", "ppv", "n_tp", "n_pred", "n_true")}}
                            rows.append(row)
                            if best_row is None or m["f1"] > best_row[1]["f1"]:
                                best_row = (row["name"], m)
    print(f"解码网格 {len(rows)} 组完成（{time.time()-t0:.0f}s）", flush=True)
    for row in sorted(rows, key=lambda r: -r["f1"])[:12]:
        print(f"  {row['name']}: F1={row['f1']:.3f} sens={row['sensitivity']:.3f} ppv={row['ppv']:.3f} "
              f"({row['n_tp']}/{row['n_true']}, pred={row['n_pred']})", flush=True)
    print(f"  ★ 最佳: {best_row[0]} F1={best_row[1]['f1']:.3f} sens={best_row[1]['sensitivity']:.3f} "
          f"ppv={best_row[1]['ppv']:.3f} ({best_row[1]['n_tp']}/{best_row[1]['n_true']}, pred={best_row[1]['n_pred']})", flush=True)

    # ---- 最佳配置逐事件明细（FP 来源分析，--detail） ----
    out = {"fold": k, "rows": rows, "best": {"name": best_row[0],
           **{kk: best_row[1][kk] for kk in ("f1", "sensitivity", "ppv", "n_tp", "n_pred", "n_true")}},
           "dl_alignment": round(n_hit / max(n_tot, 1), 3)}
    p1 = config.OUTPUT_DIR / f"rank_events_fold{k}.json"
    if p1.exists():
        r1 = json.loads(p1.read_text(encoding="utf-8"))
        out["baselines"] = {"v1_best": r1["best"]["f1"]}
    if args.detail and best_row is not None:
        import re
        mm = re.match(r"w([\d.]+)_t([\d.]+)_g([\d.]+)_p([\d.]+)_d(\d+)_k(\d+)", best_row[0])
        cfg = (float(mm[1]), float(mm[2]), float(mm[3]), float(mm[4]), int(mm[5]), int(mm[6]))
        detail, pred_sid = [], []
        for row in val_rows:
            pr, dt = decode_session(row, gate_prob, cfg, clf_pri)
            pred_sid.extend(pr)
            detail.extend(dt)
        true_by_sid = {}
        for s, (b, e) in true_sid:
            true_by_sid.setdefault(s, []).append((b, e))
        out["detail"] = [{"sid": sid, "s": s, "e": e, "src": src, "score": sc,
                          "tp": any(event_iou((s, e), t) >= IOU_LABEL
                                    for t in true_by_sid.get(sid, []))}
                         for sid, s, e, src, sc in detail]
        n_pri = sum(1 for d in out["detail"] if d["src"] == "pri")
        n_fp = sum(1 for d in out["detail"] if not d["tp"])
        n_fp_pri = sum(1 for d in out["detail"] if not d["tp"] and d["src"] == "pri")
        print(f"  明细: {len(out['detail'])} 预测（act {len(out['detail']) - n_pri} / pri {n_pri}），"
              f"FP {n_fp}（其中 pri 来源 {n_fp_pri}）", flush=True)
    norm_tag = f"_norm{args.norm_score}" if args.norm_score != "none" else ""
    (config.OUTPUT_DIR / f"rank_events_v2_fold{k}_{args.prior_grid}{norm_tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"→ outputs/rank_events_v2_fold{k}_{args.prior_grid}{norm_tag}.json", flush=True)


if __name__ == "__main__":
    main()
