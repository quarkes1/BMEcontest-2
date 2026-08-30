# -*- coding: utf-8 -*-
"""逐餐漏检归因：固定配置下"可达但未检出"的每餐失败层定位。

对每餐（val）判定：检出 or 漏检；漏检按优先级归因：
  gated        — 会话门控未过（gate_prob < thr_g），两通道都不打分
  act_low      — 活动覆盖但融合分 < τ（w=1.0 时即深度分 < τ）
  act_cap      — 融合分 ≥ τ 但被会话 top-K 挤掉（本会话已有 K 个更高分事件）
  pri_low      — 仅先验覆盖但 sp < thr_p
  pri_cap      — sp ≥ thr_p 但被 top-2/去重挤掉
  uncovered    — 两池均无 IoU≥0.25 候选（数学不可达）

运行：source activate bme && python scripts/analysis_miss_reasons.py --fold 0 [--cfg ...] [--grid 15m|60m]
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


def parse_cfg(name):
    import re as _re
    mm = _re.match(r"w([\d.]+)_t([\d.]+)_g([\d.]+)_p([\d.]+)_d(\d+)_k(\d+)", name)
    return (float(mm[1]), float(mm[2]), float(mm[3]), float(mm[4]), int(mm[5]), int(mm[6]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--grid", choices=("15m", "60m"), default="15m")
    ap.add_argument("--cfg", default="w1.0_t0.3_g0.15_p0.7_d120_k1")
    args = ap.parse_args()
    k = args.fold
    grid_s, half_w_s = (900, 450) if args.grid == "15m" else (3600, 1200)
    cfg = parse_cfg(args.cfg)
    w, tau, thr_g, thr_p, dil, K = cfg

    folds = splits.load_folds()
    f = folds[k]
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    starts = {r["session_id"]: int(r["timeStamp.startTime"]) for _, r in idx.iterrows()}
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
    gate_prob = {sid: float(gate.predict_proba(np.array(gate_feats[sid])[None])[0, 1])
                 for sid in f["val_sessions"] if sid in gate_feats}

    # 注意：分类器只允许用 train 会话训练（val 仅打分）——否则泄漏产生假高分
    Xa_tr, ya_tr, Xp_tr, yp_tr = [], [], [], []
    sess_meta = {}
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
                                         prior_grid_s=grid_s, prior_half_w_s=half_w_s)
            meals = sid_meals.get(sid, [])
            if act:
                Xa = re.candidate_features(act, env, t0, prior, sess_feats, 0.5)
                if split == "tr":
                    Xa_tr.append(Xa)
                    ya_tr.append(re.match_labels(act, meals))
            if pri:
                Xp = re.candidate_features(pri, env, t0, prior, sess_feats, 0.5)
                if split == "tr":
                    Xp_tr.append(Xp)
                    yp_tr.append(re.match_labels(pri, meals))
    Xa_tr, ya_tr = np.concatenate(Xa_tr), np.concatenate(ya_tr)
    Xp_tr, yp_tr = np.concatenate(Xp_tr), np.concatenate(yp_tr)

    def _fit(X, y):
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=10,
                                  scale_pos_weight=max(1.0, (len(y) - y.sum()) / max(y.sum(), 1)),
                                  verbosity=-1).fit(X, y)

    clf_act = _fit(Xa_tr, ya_tr)
    clf_pri = _fit(Xp_tr, yp_tr) if len(Xp_tr) else None

    # ---- 逐餐归因 ----
    stats = dict.fromkeys(("detected", "gated", "act_low", "act_cap", "pri_low", "pri_cap", "uncovered"), 0)
    print(f"配置 {args.cfg}（{args.grid} 网格）| fold{k}")
    print(f"{'sid':>10} {'meal':>6} {'dur':>5} {'gate':>5} | {'act':>5} {'dl':>6} {'fuse':>6} "
          f"{'pri':>6} | 归因")
    for sid in f["val_sessions"]:
        if sid not in sess_meta:
            continue
        env, t0, sess_feats = sess_meta[sid]
        act, pri = re.make_proposals(env, t0, prior, starts.get(sid, 0), dilate_ms=60000,
                                     prior_grid_s=grid_s, prior_half_w_s=half_w_s)
        Xa = re.candidate_features(act, env, t0, prior, sess_feats, 0.5)
        sa = clf_act.predict_proba(Xa)[:, 1] if len(Xa) else np.zeros(0)
        va_scores = np.full(len(act), np.nan)
        for j, c in enumerate(act):
            if (sid, c[0], c[1]) in dl:
                va_scores[j] = dl[(sid, c[0], c[1])]
        fuse = np.where(np.isnan(va_scores), sa, w * va_scores + (1 - w) * sa) if len(sa) else np.zeros(0)
        Xp = re.candidate_features(pri, env, t0, prior, sess_feats, 0.5)
        sp = clf_pri.predict_proba(Xp)[:, 1] if len(Xp) else np.zeros(0)
        gp = gate_prob.get(sid, 1.0)
        preds, _ = v2.decode_session((sid, act, sa, va_scores, pri, sp), gate_prob, cfg, clf_pri)
        matched = [(e[0], e[1]) for (s2, e) in preds]

        # 无 K 上限模拟（活动通道）
        sel_all = np.where(fuse >= tau)[0] if (gp >= thr_g and len(sa)) else np.zeros(0, int)
        # 先验通道：sp ≥ thr_p 的窗（top-2 之前的集合）
        pri_ok = np.where(sp >= thr_p)[0] if clf_pri is not None and gp >= thr_g else np.zeros(0, int)

        for m in sid_meals.get(sid, []):
            b, e = m["before"], m["after"]
            iou_a = [event_iou((b, e), (c[0], c[1])) for c in act]
            iou_p = [event_iou((b, e), (c[0], c[1])) for c in pri]
            ioa, iop = np.array(iou_a), np.array(iou_p)
            act_cov = bool((ioa >= IOU).any())
            pri_cov = bool((iop >= IOU).any())
            act_max = float(sa[ioa >= IOU].max()) if act_cov and len(sa) else 0.0
            dl_max = float(np.nanmax(va_scores[ioa >= IOU])) if act_cov else 0.0
            fuse_max = float(fuse[ioa >= IOU].max()) if act_cov else 0.0
            pri_max = float(sp[iop >= IOU].max()) if pri_cov and len(sp) else 0.0
            if any(event_iou((b, e), m2) >= IOU for m2 in matched):
                reason = "TP"
                stats["detected"] += 1
            elif gp < thr_g:
                reason = "gated"
                stats["gated"] += 1
            elif act_cov and fuse_max >= tau:
                reason = "act_cap(K挤掉/去重)"  # 分够但没发出 → 会话内竞争
                stats["act_cap"] += 1
            elif act_cov:
                reason = f"act_low(fuse {fuse_max:.2f})"
                stats["act_low"] += 1
            elif pri_cov and pri_max >= thr_p:
                reason = "pri_cap(top-2/去重)"
                stats["pri_cap"] += 1
            elif pri_cov:
                reason = f"pri_low({pri_max:.2f})"
                stats["pri_low"] += 1
            else:
                reason = "uncovered"
                stats["uncovered"] += 1
            print(f"{sid[-8:]:>10} {b/3.6e6%24:5.1f}h {(e-b)/60000:5.1f}min {gp:5.2f} | "
                  f"{act_max:5.2f} {dl_max:6.2f} {fuse_max:6.2f} {pri_max:6.2f} | {reason}")
    print(f"\n合计: 检出 {stats['detected']} | gated {stats['gated']} | act_low {stats['act_low']} "
          f"| act_cap {stats['act_cap']} | pri_low {stats['pri_low']} | pri_cap {stats['pri_cap']} "
          f"| uncovered {stats['uncovered']}")


if __name__ == "__main__":
    main()
