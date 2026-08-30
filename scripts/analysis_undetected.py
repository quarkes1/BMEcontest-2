# -*- coding: utf-8 -*-
"""逐餐诊断：fold{k} val 每餐的覆盖来源 + 检出失败层定位。

对每餐计算：
  act_cov: 活动池是否有候选 IoU≥0.25（dilate 60s 后）
  pri_cov: 15min 先验网格是否有窗 IoU≥0.25
  覆盖候选的最高分数：act LGBM 分 / 深度分 / w0.7 融合分
  是否被 best 配置检出（TP/FP/miss）

用途：区分"覆盖但评分低"vs"覆盖但解码漏"vs"无覆盖（数学不可达）"。
运行：source activate bme && python scripts/analysis_undetected.py --fold 0
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
from src.eval.metrics import event_iou
import rank_events as re
import rank_events_v2 as v2

IOU = 0.25
W = 0.7
TAU = 0.25


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
    prior = re._prior(np.array([m["before"] / 3.6e6 % 24
                                for s in f["train_sessions"] if s in sid_meals for m in sid_meals[s]]))

    dl = v2.load_dl_scores(k)
    out_cache = config.CACHE_DIR / "validate_baselines"

    # ---- 复刻 v2 的池打分 ----
    import lightgbm as lgb
    Xa_tr, ya_tr, Xp_tr, yp_tr = [], [], [], []
    sess_meta = {}
    gate_prob = {}
    GATE_FEATS = ["dur_h", "env_mean", "env_p50", "env_p95", "env_std", "p95_ratio", "start_h_utc"]
    gate_feats = {}
    for sid in f["train_sessions"] + f["val_sessions"]:
        p = out_cache / f"{sid}.npz"
        if p.exists():
            ft = json.loads(np.load(p)["feats"].item())
            gate_feats[sid] = [ft[kk] for kk in GATE_FEATS]
    Xg_tr = np.array([gate_feats[s] for s in f["train_sessions"] if s in gate_feats])
    yg_tr = np.array([1 if s in sid_meals else 0 for s in f["train_sessions"] if s in gate_feats])
    gate = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                              min_child_samples=20, verbosity=-1).fit(Xg_tr, yg_tr)
    for sid in f["val_sessions"]:
        if sid in gate_feats:
            gate_prob[sid] = float(gate.predict_proba(np.array(gate_feats[sid])[None])[0, 1])

    for split, sessions in (("tr", f["train_sessions"]), ("va", f["val_sessions"])):
        for sid in sessions:
            p = out_cache / f"{sid}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            env, t0 = d["env"].astype(np.float32), d["t0"].astype(np.int64)
            ft = json.loads(d["feats"].item())
            sess_feats = {"dur_h": float(ft["dur_h"]), "env_p95": float(ft["env_p95"]),
                          "p95_ratio": float(ft["p95_ratio"]), "start": starts.get(sid, 0)}
            sess_meta[sid] = (env, t0, sess_feats)
            act, pri = re.make_proposals(env, t0, prior, starts.get(sid, 0), dilate_ms=60000,
                                         prior_grid_s=v2.PRIOR_GRID_S, prior_half_w_s=v2.PRIOR_HALF_W_S)
            meals = sid_meals.get(sid, [])
            if act:
                Xa_tr.append(re.candidate_features(act, env, t0, prior, sess_feats, 0.5))
                ya_tr.append(re.match_labels(act, meals))
            if pri:
                Xp_tr.append(re.candidate_features(pri, env, t0, prior, sess_feats, 0.5))
                yp_tr.append(re.match_labels(pri, meals))
    Xa_tr = np.concatenate(Xa_tr); ya_tr = np.concatenate(ya_tr)
    Xp_tr = np.concatenate(Xp_tr); yp_tr = np.concatenate(yp_tr)

    def _fit(X, y):
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=10,
                                  scale_pos_weight=max(1.0, (len(y) - y.sum()) / max(y.sum(), 1)),
                                  verbosity=-1).fit(X, y)

    clf_act = _fit(Xa_tr, ya_tr)
    clf_pri = _fit(Xp_tr, yp_tr)

    # ---- 逐餐诊断 ----
    print(f"{'sid':>10} {'meal[h:m]':>10} {'dur':>5} | {'act_cov':>7} {'pri_cov':>7} | "
          f"{'act_max':>8} {'dl_max':>8} {'fuse_max':>9} {'pri_max':>8} | detect")
    stats = {"detected": 0, "act_cov_miss": 0, "pri_only": 0, "uncovered": 0}
    for sid in f["val_sessions"]:
        if sid not in sess_meta:
            continue
        env, t0, sess_feats = sess_meta[sid]
        act, pri = re.make_proposals(env, t0, prior, starts.get(sid, 0), dilate_ms=60000,
                                     prior_grid_s=v2.PRIOR_GRID_S, prior_half_w_s=v2.PRIOR_HALF_W_S)
        Xa = re.candidate_features(act, env, t0, prior, sess_feats, 0.5)
        sa = clf_act.predict_proba(Xa)[:, 1] if len(Xa) else np.zeros(0)
        va_scores = np.full(len(act), np.nan)
        for j, c in enumerate(act):
            if (sid, c[0], c[1]) in dl:
                va_scores[j] = dl[(sid, c[0], c[1])]
        fuse = np.where(np.isnan(va_scores), sa, W * va_scores + (1 - W) * sa)
        Xp = re.candidate_features(pri, env, t0, prior, sess_feats, 0.5)
        sp = clf_pri.predict_proba(Xp)[:, 1] if len(Xp) else np.zeros(0)
        for m in sid_meals.get(sid, []):
            b, e = m["before"], m["after"]
            iou_a = [event_iou((b, e), (c[0], c[1])) for c in act]
            iou_p = [event_iou((b, e), (c[0], c[1])) for c in pri]
            ioa = np.array(iou_a); iop = np.array(iou_p)
            act_cov = bool((ioa >= IOU).any())
            pri_cov = bool((iop >= IOU).any())
            act_max = float(sa[ioa >= IOU].max()) if act_cov else 0.0
            fuse_max = float(fuse[ioa >= IOU].max()) if act_cov else 0.0
            dl_max = float(va_scores[ioa >= IOU].max()) if act_cov else 0.0
            pri_max = float(sp[iop >= IOU].max()) if pri_cov else 0.0
            detected = bool((ioa >= IOU).any()) and act_max >= TAU and act_cov
            # 简化：单看 act 通道 τ 判定（pri 通道 top-1 判定复杂，略）
            if detected:
                stats["detected"] += 1
            elif act_cov:
                stats["act_cov_miss"] += 1
            elif pri_cov:
                stats["pri_only"] += 1
            else:
                stats["uncovered"] += 1
            print(f"{sid[-8:]:>10} {b/3.6e6%24:5.1f}h {(e-b)/60000:5.1f}min | "
                  f"{act_cov!s:>7} {pri_cov!s:>7} | {act_max:8.3f} {dl_max:8.3f} "
                  f"{fuse_max:9.3f} {pri_max:8.3f} | {'TP' if detected else 'miss'}")
    print(f"\n合计: 检出 {stats['detected']} | 活动覆盖但漏检 {stats['act_cov_miss']} "
          f"| 仅先验覆盖 {stats['pri_only']} | 无覆盖 {stats['uncovered']}")


if __name__ == "__main__":
    main()
