# -*- coding: utf-8 -*-
"""阶段2：事件提案 + 排序头（检测即排序，不依赖密度模型）。
提案：env×先验多参数连通域（复用 windows_to_events）→ 候选事件
特征：时长/峰值/均值/形状/时刻先验/上下文对比/会话特征/受试者基线
标签：候选与真值 IoU≥0.25 贪心匹配
模型：LightGBM 二分类（train 折训练，val 折评估，受试者不泄漏）
评估：每会话 top-1/top-2 事件 F1（对比 V2-B 启发式 0.119、DensityNet 0.020）
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
CTX_S = 1800.0             # 上下文对比窗（±30min）
IOU_LABEL = 0.25


def _prior(meal_before_hours, sigma_h=1.0):
    hist, _ = np.histogram(meal_before_hours, bins=24, range=(0, 24))
    prior = np.zeros(24, dtype=np.float32)
    for i, c in enumerate(hist):
        if c:
            prior += c * np.exp(-((np.arange(24) - i) ** 2) / (2 * sigma_h ** 2))
    return prior / prior.max()


def make_proposals(env, t0, prior, start_epoch):
    """多参数连通域 → 候选事件（去重合并）。返回 [(s_ms, e_ms), ...]。"""
    h = (t0 / 3.6e6) % 24
    score = env * prior[np.clip(h.astype(int), 0, 23)]
    cands = []
    for pct in PROP_PCT:
        for gap in PROP_GAP:
            for dur in PROP_DUR:
                evs = windows_to_events(score, t0, t0 + WINDOW_MS, float(np.percentile(score, pct)),
                                        merge_gap_s=gap, min_dur_s=dur, smooth_win=SMOOTH)
                cands.extend(evs)
    if not cands:
        return []
    # 按起始排序，IoU>0.6 的相邻候选合并为更长者
    cands.sort()
    merged = [list(cands[0])]
    for c in cands[1:]:
        if c[0] <= merged[-1][1] and event_iou(tuple(merged[-1]), c) > 0.6:
            merged[-1][1] = max(merged[-1][1], c[1])
        else:
            merged.append(list(c))
    return [(s, e) for s, e in merged]


def candidate_features(cands, env, t0, prior, sess_feats, gate_prob=0.5):
    """候选事件 → 特征矩阵（n_cand, n_feat）+ 标签（贪心 IoU 匹配会话内餐）。"""
    X, y = [], []
    for s, e in cands:
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
        ])
    return np.array(X, dtype=np.float32) if X else np.zeros((0, 12), dtype=np.float32)


def match_labels(cands, meals):
    """候选与真值贪心 IoU 匹配（与竞赛口径一致的匹配器）。"""
    y = np.zeros(len(cands), dtype=np.int8)
    used = set()
    # 按 IoU 降序
    pairs = [(event_iou(c, (m["before"], m["after"])), i, j)
             for i, c in enumerate(cands)
             for j, m in enumerate(meals)
             if event_iou(c, (m["before"], m["after"])) >= IOU_LABEL]
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
    ap.add_argument("--topk", type=int, default=1, help="每会话保留候选数（1 或 2）")
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
    gate_prob = {}
    for sid, x in zip(f["val_sessions"], Xg_va):
        gate_prob[sid] = float(gate.predict_proba(x[None])[0, 1])
    print(f"  会话门控: val {len(Xg_va)} 会话打分完成（AUC≈0.88 见 validate_baselines）", flush=True)

    # ---- 提案生成 + 特征（train / val） ----
    Xtr, ytr, Xva, yva = [], [], [], []
    sess_meta = {}    # sid → (env, t0, prior 乘积后的 score 相关特征用到的 sess_feats)
    prop_stats = {"tr_prop": 0, "tr_meals": 0, "tr_covered": 0,
                  "va_prop": 0, "va_meals": 0, "va_covered": 0}
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
            cands = make_proposals(env, t0, prior, start)
            meals = sid_meals.get(sid, [])
            n_covered = sum(1 for m in meals
                            if any(event_iou(c, (m["before"], m["after"])) >= IOU_LABEL for c in cands))
            key = f"{split}_"
            prop_stats[key + "prop"] += len(cands)
            prop_stats[key + "meals"] += len(meals)
            prop_stats[key + "covered"] += n_covered
            if not cands:
                continue
            gp = gate_prob.get(sid, 0.5)
            X = candidate_features(cands, env, t0, prior, sess_feats, gp)
            y = match_labels(cands, meals)
            if split == "tr":
                Xtr.append(X); ytr.append(y)
            else:
                Xva.append(X); yva.append(y)
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xva = np.concatenate(Xva); yva = np.concatenate(yva)
    print(f"  提案: train {len(Xtr)}（正 {ytr.sum()}）| val {len(Xva)}（正 {yva.sum()}）", flush=True)
    print(f"  提案覆盖餐: train {prop_stats['tr_covered']}/{prop_stats['tr_meals']} "
          f"val {prop_stats['va_covered']}/{prop_stats['va_meals']}", flush=True)

    # ---- LightGBM 排序 ----
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                             min_child_samples=10, scale_pos_weight=max(1.0, (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)),
                             verbosity=-1)
    clf.fit(Xtr, ytr)
    pva = clf.predict_proba(Xva)[:, 1]
    print("  特征重要性:", [f"f{i}" for i in np.argsort(clf.feature_importances_)[::-1]][:6], flush=True)

    # ---- 解码：会话门控 × top-k 网格（受试者级匹配） ----
    true_all = [(s, m["before"], m["after"]) for s in f["val_sessions"] if s in sid_meals for m in sid_meals[s]]
    true_sid = [(s, (b, e)) for s, b, e in true_all]
    rows, best_row = [], None
    for thr_g in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7):
        for topk in (1, 2, 3):
            pred_sid = []
            i = 0
            for sid in f["val_sessions"]:
                if sid not in sess_meta:
                    continue
                env, t0, sess_feats = sess_meta[sid]
                cands = make_proposals(env, t0, prior, starts.get(sid, 0))
                n_c = len(cands)
                if n_c == 0:
                    continue
                scores = pva[i:i + n_c]
                i += n_c
                if gate_prob.get(sid, 1.0) < thr_g:
                    continue
                sel = np.argsort(scores)[-topk:]
                pred_sid.extend((sid, cands[j]) for j in sel)
            m = compute_metrics_by_subject(pred_sid, true_sid, lambda s: subject_of[s])
            row = {"name": f"g{thr_g}_k{topk}", **{kk: m[kk] for kk in ("f1", "sensitivity", "ppv", "n_tp", "n_pred", "n_true")}}
            rows.append(row)
            if best_row is None or m["f1"] > best_row[1]["f1"]:
                best_row = (row["name"], m)
    for row in rows:
        print(f"  {row['name']}: F1={row['f1']:.3f} sens={row['sensitivity']:.3f} ppv={row['ppv']:.3f} "
              f"({row['n_tp']}/{row['n_true']}, pred={row['n_pred']})", flush=True)
    print(f"  ★ 最佳: {best_row[0]} F1={best_row[1]['f1']:.3f} sens={best_row[1]['sensitivity']:.3f} "
          f"ppv={best_row[1]['ppv']:.3f} ({best_row[1]['n_tp']}/{best_row[1]['n_true']}, pred={best_row[1]['n_pred']})", flush=True)

    out = {"fold": k, "rows": rows, "best": {"name": best_row[0], **{kk: best_row[1][kk] for kk in ("f1", "sensitivity", "ppv", "n_tp", "n_pred", "n_true")}},
           "proposal_coverage": prop_stats,
           "baselines": {"v2b_heuristic_subj": 0.085, "densitynet_subj": 0.020},
           "n_train": len(Xtr), "n_val": len(Xva)}
    (config.OUTPUT_DIR / f"rank_events_fold{k}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"→ outputs/rank_events_fold{k}.json", flush=True)


if __name__ == "__main__":
    main()
