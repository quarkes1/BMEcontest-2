# -*- coding: utf-8 -*-
"""架构验证 1+2（只读，不训练模型，不占 GPU）：
V1 会话级可判别性：LightGBM 无餐/有餐会话分类（fold k train→val，受试者不泄漏）→ AUC
V2 启发式密度检测器下限：1s 活动包络 × 时刻先验 → 事件 → 竞赛口径 F1（fold k val）
   变体 A: env only / B: env×prior_abs / C: env×prior_rel（先验只从 train 折估计）
产物: outputs/validate_baselines.json（含 per-fold 汇总）
运行: conda activate bme && python scripts/validate_baselines.py --folds 0[,1..]
"""
import argparse
import os
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.data.loader import load_session, detect_binary, _find_collect_data
from src.eval.metrics import compute_metrics
from src.infer.events import windows_to_events

OUT_CACHE = config.CACHE_DIR / os.environ.get("BME_ENV_CACHE", "validate_baselines")
FS = 105.0
WIN_ROWS = int(5 * FS)        # 5s 窗
STRIDE = int(FS)              # 1s 步长
SMOOTH = 31
MERGE_GAPS = (15, 30, 60)
MIN_DURS = (10, 20, 30)
THR_PCT = (75, 85, 90, 92, 95, 97, 98)

import scipy.signal


def _density_one(args):
    sid, start_epoch = args
    out = OUT_CACHE / f"{sid}.npz"
    if out.exists():
        return ("skip", sid)
    try:
        d = config.SENSOR_DIR / sid
        if detect_binary(_find_collect_data(str(d))):
            return ("binary", sid)
        s = load_session(sid)
        fs = s.meta.get("row_rate", FS)
        valid = s.imu_valid
        t_v = s.t_acc[valid].astype(np.int64)
        a_v = s.acc[:, valid]
        if len(t_v) < int(5 * fs) or t_v[-1] <= t_v[0]:
            return ("short", sid)
        # 窗网格：真实时间轴（修复：原均匀行号轴漂移可达 5min/会话）。
        # 数据为包级时间戳（每戳 10 行，94-95ms 包间），窗定位精度 ~0.1s。
        win_ms, st_ms = 5000, 1000
        t0_real = int(t_v[0])
        n_w = max(0, (int(t_v[-1]) - t0_real - win_ms) // st_ms + 1)
        if n_w < 50:
            return ("short", sid)
        ws = t0_real + np.arange(n_w, dtype=np.int64) * st_ms
        lo = np.searchsorted(t_v, ws)
        hi = np.searchsorted(t_v, ws + win_ms)          # 窗 [ws, ws+5s) 内行
        envs = np.empty(n_w, dtype=np.float32)
        zcrs = np.empty(n_w, dtype=np.float32)
        sos = scipy.signal.butter(4, [0.5, 2.0], btype="bandpass", fs=fs, output="sos")
        sos_z = scipy.signal.butter(4, [0.1, 0.5], btype="bandpass", fs=fs, output="sos")
        min_rows = int(win_ms / 1000 * fs) // 2          # 窗内 <2.5s 数据 → 视为无数据
        for b0 in range(0, n_w, 1000):
            b1 = min(b0 + 1000, n_w)
            cnts = hi[b0:b1] - lo[b0:b1]
            ok = cnts >= min_rows
            maxc = int(cnts[ok].max()) if ok.any() else 0
            m = b1 - b0
            if maxc == 0:
                envs[b0:b1] = 0.0
                zcrs[b0:b1] = 0.0
                continue
            segs = np.zeros((m, maxc, 3), np.float32)
            for j, i in enumerate(range(b0, b1)):
                if cnts[j] >= min_rows:
                    n_c = int(cnts[j])
                    seg = a_v[:, lo[i]:hi[i]].T           # (n_c, 3)
                    segs[j, :n_c] = seg
                    if n_c < maxc:                        # 缺口窗 edge pad
                        segs[j, n_c:] = seg[-1]
                # 无数据窗保持 0 → lam 0 → env 0（提案无）
            la = segs - np.median(segs, axis=1, keepdims=True)
            lam = np.linalg.norm(la, axis=2)              # (m, maxc)
            with np.errstate(invalid="ignore"):
                e = scipy.signal.sosfiltfilt(sos, lam, axis=1)
                envs[b0:b1] = np.abs(e).mean(axis=1)
                ez = scipy.signal.sosfiltfilt(sos_z, lam, axis=1)
                zcrs[b0:b1] = ((ez[:, 1:] * ez[:, :-1]) < 0).mean(axis=1)
            envs[b0:b1][cnts < min_rows] = 0.0
            zcrs[b0:b1][cnts < min_rows] = 0.0
        t0_ms = ws                                 # 窗起点 = 真实 ACC 时间戳
        feats = {
            "dur_h": round(len(t_v) / fs / 3600, 4),
            "env_mean": float(envs.mean()), "env_p50": float(np.median(envs)),
            "env_p95": float(np.percentile(envs, 95)), "env_std": float(envs.std()),
            "p95_ratio": float(np.percentile(envs, 95) / (envs.mean() + 1e-6)),
            "z_p50": float(np.median(zcrs)), "z_p95": float(np.percentile(zcrs, 95)),
            "start_h_utc": round((start_epoch / 3.6e6) % 24, 3),
        }
        np.savez_compressed(out, env=envs, env_z=zcrs, t0=t0_ms, feats=json.dumps(feats))
        return ("ok", sid)
    except Exception as e:
        return ("error", f"{sid}: {type(e).__name__}: {e}")


def build_cache(fold, sessions):
    """并行构建 1s 密度曲线缓存（幂等，跳过已存在）。"""
    starts = {r["session_id"]: int(r["timeStamp.startTime"]) for _, r in _SESS_DF.iterrows()}
    tasks = [(sid, starts.get(sid, 0)) for sid in sessions]
    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "binary": 0, "short": 0, "error": 0}
    errs = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (st, info) in enumerate(ex.map(_density_one, tasks, chunksize=4)):
            if st == "error":
                stats["error"] += 1
                errs.append(info)
            else:
                stats[st] += 1
            if (i + 1) % 200 == 0:
                print(f"  fold{fold} 密度缓存 {i+1}/{len(tasks)} 用时 {time.time()-t0:.0f}s", flush=True)
    if errs:
        print("  错误示例:", errs[:3], flush=True)
    print(f"fold{fold} 密度缓存完成: {stats}", flush=True)


def _prior(meal_before_hours, sigma_h=1.0):
    """24h 高斯平滑直方图 → 归一化先验 P(h)。"""
    hist, _ = np.histogram(meal_before_hours, bins=24, range=(0, 24))
    h = np.arange(24, dtype=float)
    prior = np.zeros(24)
    for i, c in enumerate(hist):
        if c:
            prior += c * np.exp(-((h - i) ** 2) / (2 * sigma_h ** 2))
    prior = prior / prior.max()
    return prior.astype(np.float32)


def score_variant(env, t0_ms, prior, rel):
    """env × 先验；rel=True 用会话相对小时（t0-epoch 由调用方转换）。"""
    if prior is None:
        return env
    h = (t0_ms / 3.6e6) % 24
    w = prior[np.clip(h.astype(int), 0, 23)]
    return env * w


def grid_fold(fold, sessions, sids_with_meals, true_events):
    results = {}
    # 先验：只从 train 折估计（会话相对小时 + 绝对小时）
    rel_hours, abs_hours = [], []
    for sid, meals in sids_with_meals:
        row = _SESS_DF.loc[_SESS_DF["session_id"] == sid]
        start = int(row["timeStamp.startTime"].iloc[0]) if len(row) else 0
        for m in meals:
            abs_hours.append((m["before"] / 3.6e6) % 24)
            rel_hours.append(((m["before"] - start) / 3.6e6) % 24)
    prior_abs = _prior(np.array(abs_hours)) if abs_hours else None
    prior_rel = _prior(np.array(rel_hours)) if rel_hours else None
    priors = {"A": None, "B": prior_abs, "C": prior_rel}

    for name, prior in priors.items():
        best = None
        for pct in THR_PCT:
            for gap in MERGE_GAPS:
                for dur in MIN_DURS:
                    evs_all = []
                    for sid in sessions:
                        f = OUT_CACHE / f"{sid}.npz"
                        if not f.exists():
                            continue
                        d = np.load(f)
                        sc = score_variant(d["env"], d["t0"], prior, name == "C")
                        evs_all.extend(windows_to_events(sc, d["t0"], d["t0"] + WIN_ROWS * 1000,
                                                         float(np.percentile(sc, pct)),
                                                         merge_gap_s=gap, min_dur_s=dur, smooth_win=SMOOTH))
                    m = compute_metrics(evs_all, true_events)
                    if best is None or m["f1"] > best[1]["f1"]:
                        best = ((pct, gap, dur), m)
        (pct, gap, dur), m = best
        results[name] = {"best": {"pct": pct, "merge_gap_s": gap, "min_dur_s": dur},
                         "f1": m["f1"], "sens": m["sensitivity"], "ppv": m["ppv"],
                         "n_true": m["n_true"], "n_pred": m["n_pred"], "n_tp": m["n_tp"]}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    args = ap.parse_args()
    OUT_CACHE.mkdir(parents=True, exist_ok=True)

    global _SESS_DF
    _SESS_DF = manifests.load_sensor_index()
    meal_meta, _ = manifests.load_meal_meta()
    folds = splits.load_folds()

    # 会话 → 餐（epoch 落在 [start, end] 内）
    sid_meals = {}
    for _, r in _SESS_DF.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms

    report = {"folds": {}}
    for k in [int(x) for x in args.folds.split(",")]:
        f = folds[k]
        print(f"===== fold {k} =====", flush=True)
        build_cache(k, f["train_sessions"] + f["val_sessions"])
        tr_meals = [(s, sid_meals[s]) for s in f["train_sessions"] if s in sid_meals]
        val_meals = [(s, sid_meals[s]) for s in f["val_sessions"] if s in sid_meals]
        val_events = [(m["before"], m["after"]) for _, ms in val_meals for m in ms]
        print(f"  train 会话 {len(f['train_sessions'])}（含餐 {len(tr_meals)}）| val 会话 {len(f['val_sessions'])}（含餐 {len(val_meals)}，事件 {len(val_events)}）", flush=True)

        # ---- V1 会话级 LightGBM ----
        lgb_metrics = None
        try:
            import lightgbm as lgb
            feats_keys = ["dur_h", "env_mean", "env_p50", "env_p95", "env_std", "p95_ratio", "start_h_utc"]
            Xtr, ytr, Xva, yva = [], [], [], []
            for sid in f["train_sessions"]:
                p = OUT_CACHE / f"{sid}.npz"
                if p.exists():
                    ft = json.loads(np.load(p)["feats"].item())
                    Xtr.append([ft[kk] for kk in feats_keys]); ytr.append(1 if sid in sid_meals else 0)
            for sid in f["val_sessions"]:
                p = OUT_CACHE / f"{sid}.npz"
                if p.exists():
                    ft = json.loads(np.load(p)["feats"].item())
                    Xva.append([ft[kk] for kk in feats_keys]); yva.append(1 if sid in sid_meals else 0)
            clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                     min_child_samples=20, verbosity=-1)
            clf.fit(np.array(Xtr), np.array(ytr))
            proba = clf.predict_proba(np.array(Xva))[:, 1]
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(np.array(yva), proba)
            # 按概率阈值 0.5 的混淆
            pred = proba >= 0.5
            tp = ((pred == 1) & (np.array(yva) == 1)).sum(); tn = ((pred == 0) & (np.array(yva) == 0)).sum()
            fp = ((pred == 1) & (np.array(yva) == 0)).sum(); fn = ((pred == 0) & (np.array(yva) == 1)).sum()
            lgb_metrics = {"auc": float(auc), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                           "n_val": int(len(yva)), "n_meal_val": int(sum(yva)),
                           "imp": [feats_keys[i] for i in np.argsort(clf.feature_importances_)[::-1]]}
            print(f"  V1 会话级 AUC={auc:.3f} (val n={len(yva)}, 含餐 {sum(yva)})", flush=True)
            print(f"     混淆: TP={tp} TN={tn} FP={fp} FN={fn}", flush=True)
        except Exception as e:
            print(f"  V1 LightGBM 失败: {e}", flush=True)

        # ---- V2 启发式密度检测器 ----
        v2 = grid_fold(k, f["val_sessions"], tr_meals, val_events)
        for name, r in v2.items():
            print(f"  V2[{name}] F1={r['f1']:.3f} sens={r['sens']:.3f} ppv={r['ppv']:.3f} "
                  f"({r['n_tp']}/{r['n_true']} 匹配, pred={r['n_pred']}) 最佳: {r['best']}", flush=True)
        report["folds"][str(k)] = {"lgb": lgb_metrics, "v2": v2}

    (config.OUTPUT_DIR / "validate_baselines.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("→ outputs/validate_baselines.json", flush=True)


if __name__ == "__main__":
    main()
