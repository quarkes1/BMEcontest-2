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

运行：source activate bme && python scripts/rank_events_v2.py --fold 0
产物：outputs/rank_events_v2_fold0.json
"""
import argparse
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

OUT_CACHE = config.CACHE_DIR / "validate_baselines"
IOU_LABEL = 0.25
POST_MERGE_GAP_S = 120.0    # 事件合并 gap（<2min 断开视为同一餐）
POST_MIN_DUR_S = 120.0      # 过滤 <2min 孤立段（方向 3）
DL_W = (0.0, 0.3, 0.5, 0.7, 1.0)      # 深度分数融合权重（0=纯 LightGBM 对照）
DL_TAU = tuple(np.arange(0.15, 0.425, 0.025))  # 活动融合分数阈值（细化：覆盖保守分数区 0.15-0.40）
GATE_GS = (0.0, 0.3)
PRI_THRS = (0.3, 0.5)
DILATIONS = (60.0, 120.0)


def load_dl_scores(k):
    """读 MM-Ranker val 候选分数 → {(sid, s, e): score}。"""
    p = config.OUTPUT_DIR / f"mm_ranker_fold{k}_val.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    meta = [json.loads(m.decode()) for m in z["meta"]]
    return {(m["sid"], int(m["s"]), int(m["e"])): float(sc)
            for m, sc in zip(meta, z["score"])}


def postprocess_events(evs, merge_gap_s=POST_MERGE_GAP_S, min_dur_s=POST_MIN_DUR_S,
                       dilation_s=60.0):
    """事件级形态学后处理（方向 3）：按起始排序合并相邻 → 过滤孤立短段 → 边界膨胀。"""
    if not evs:
        return []
    merged = []
    for s, e in sorted(evs):
        if merged and s - merged[-1][1] <= merge_gap_s * 1000:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        if (e - s) / 1000.0 >= min_dur_s:
            out.append((max(0, int(s - dilation_s * 1000)), int(e + dilation_s * 1000)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
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
    for split, sessions in (("tr", f["train_sessions"]), ("va", f["val_sessions"])):
        for sid in sessions:
            p = OUT_CACHE / f"{sid}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            env, t0 = d["env"].astype(np.float32), d["t0"].astype(np.int64)
            ft = json.loads(d["feats"].item())
            sess_feats = {"dur_h": float(ft["dur_h"]), "env_p95": float(ft["env_p95"]),
                          "p95_ratio": float(ft["p95_ratio"]), "start": starts.get(sid, 0)}
            sess_meta[sid] = (env, t0, sess_feats)
            act, pri = v1.make_proposals(env, t0, prior, starts.get(sid, 0), dilate_ms=60000)
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
    Xa_tr, ya_tr = np.concatenate(Xa_tr), np.concatenate(ya_tr)
    Xp_tr = np.concatenate(Xp_tr) if Xp_tr else np.zeros((0, 14), np.float32)
    yp_tr = np.concatenate(yp_tr) if yp_tr else np.zeros(0, np.int8)
    Xa_va = np.concatenate(Xa_va)
    Xp_va = np.concatenate(Xp_va) if Xp_va else np.zeros((0, 14), np.float32)

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
    dl = load_dl_scores(k)
    val_rows = []
    ia = ip = 0
    n_hit = n_tot = 0
    for sid in f["val_sessions"]:
        if sid not in sess_meta:
            continue
        env, t0, sess_feats = sess_meta[sid]
        act, pri = v1.make_proposals(env, t0, prior, starts.get(sid, 0), dilate_ms=60000)
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
                        pred_sid = []
                        for sid, act, sa, va_scores, pri, sp in val_rows:
                            act_out, pri_out = [], []
                            if gate_prob.get(sid, 1.0) >= thr_g and len(sa):
                                if w > 0:
                                    fuse = w * va_scores + (1 - w) * sa
                                    fuse = np.where(np.isnan(va_scores), sa, fuse)  # 缺深度分→纯LGBM
                                    sel = np.where(fuse >= tau)[0]
                                else:
                                    sel = np.where(sa >= tau)[0]
                                act_out = [(sid, (act[j][0], act[j][1])) for j in sel]
                            # 先验池 top-1（时间先验通道，v1 逻辑）+ 高分旁路
                            if clf_pri is not None and len(sp) and sp.max() >= thr_p:
                                jp = int(np.argmax(sp))
                                pc = (pri[jp][0], pri[jp][1])
                                if not any(event_iou(pc, (s0, e0)) >= IOU_LABEL for _, (s0, e0) in act_out):
                                    pri_out.append((sid, pc))
                            if clf_pri is not None and len(sp) and sp.max() >= 0.85:
                                jp = int(np.argmax(sp))
                                pc = (pri[jp][0], pri[jp][1])
                                if not any(event_iou(pc, (s0, e0)) >= IOU_LABEL for _, (s0, e0) in act_out + pri_out):
                                    pri_out.append((sid, pc))
                            # 事件级后处理只作用于活动事件（先验 40min 宽窗不参与合并）
                            keep = postprocess_events([e for _, e in act_out], dilation_s=dil)
                            pred_sid.extend((sid, e) for e in keep)
                            pred_sid.extend(pri_out)
                        m = compute_metrics_by_subject(pred_sid, true_sid, lambda s: subject_of[s])
                        row = {"name": f"w{w}_t{round(tau, 3)}_g{thr_g}_p{thr_p}_d{dil:.0f}",
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

    out = {"fold": k, "rows": rows, "best": {"name": best_row[0],
           **{kk: best_row[1][kk] for kk in ("f1", "sensitivity", "ppv", "n_tp", "n_pred", "n_true")}},
           "dl_alignment": round(n_hit / max(n_tot, 1), 3),
           "baselines": {"v1_best": 0.242, "v1_fixed_g0.7k2p0.3": 0.176}}
    (config.OUTPUT_DIR / f"rank_events_v2_fold{k}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"→ outputs/rank_events_v2_fold{k}.json", flush=True)


if __name__ == "__main__":
    main()
