# -*- coding: utf-8 -*-
"""多频带信号存在性检验（任务 #49 S0-2 前置）。

问题：现有 env 是单频带（acc 0.5-2.0Hz 包络）。fold1/fold3 漏检餐的深度分
普遍偏低——若它们的活动特征不在 0.5-2Hz 频带（如高频振动、腕部转动），
多频带包络才有救。先于缓存重建/重训检验信号，避免为不存在的信号投入。

频带设计（腕部进食运动学）：
  b0: acc 0.5-2.0 Hz  —— 现有 env（进食摆动主带）
  b1: acc 2.0-4.0 Hz  —— 高频振动（餐具接触/精细动作）
  b2: acc 0.1-0.5 Hz  —— 低频姿态漂移（慢速大动作）
  b3: gyro 0.5-2.0 Hz —— 角速度摆动（腕部转动，进食强信号）

检验内容：
  1. 逐频带 AUROC：真餐窗 vs 非餐窗（全局 + 逐受试者中位）
  2. 检出 vs 未检出餐的频带剖面（会话内 p95 归一）：未检出餐在哪个频带突出
  3. 融合增益估计：max/加权组合的 AUROC vs 单频带

运行：python scripts/analyze_bands.py --fold 1
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.data.loader import load_session

FS = 10.0  # 占位：每会话 row_rate（~105-113Hz）实际传入
WIN_S, ST_S = 5.0, 1.0
# 频带按真实意图（fs=row_rate ~110Hz；排序头 10Hz 缓存 Nyquist=5Hz）：
# b0/b1/b2 在排序头可见域内；b3/b4 高频（>5Hz）排序头不可见——若判别力存在，
# 需重建缓存带高频通道重训
BANDS = [("b0_acc_0.5-2", "acc", (0.5, 2.0)),
         ("b1_acc_2-4", "acc", (2.0, 4.0)),
         ("b2_acc_4-8", "acc", (4.0, 8.0)),
         ("b3_acc_8-16", "acc", (8.0, 16.0)),
         ("b4_acc_16-32", "acc", (16.0, 32.0)),
         ("b5_gyro_0.5-4", "gyro", (0.5, 4.0)),
         ("b6_gyro_4-16", "gyro", (4.0, 16.0))]
MEAL_PAD_S = 60        # 餐段膨胀（与 make_proposals dilate_ms=60000 一致）
NONMEAL_GAP_S = 1800   # 非餐窗：距所有餐段膨胀后 ≥30min


def band_feats(sig, band, fs):
    """与 validate_baselines._density_one 同构的包络计算（5s 窗/1s 步）+ 过零率。

    返回 2 组逐窗特征：
      E: |e| 均值（能量）
      Z: 过零率（bandpass 信号符号翻转比例——频带内规律性度量）
    """
    win, st = int(WIN_S * fs), int(ST_S * fs)
    n = sig.shape[1]
    n_w = (n - win) // st + 1
    sos = scipy.signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    E = np.empty(n_w, np.float32)
    Z = np.empty(n_w, np.float32)
    for b0 in range(0, n_w, 4000):
        b1 = min(b0 + 4000, n_w)
        m = b1 - b0
        idx = b0 * st + np.arange(m)[:, None] * st + np.arange(win)[None, :]
        seg = sig[:, idx]
        la = seg - np.median(seg, axis=2, keepdims=True)
        lam = np.linalg.norm(la, axis=0)                 # (m, win)
        e = scipy.signal.sosfiltfilt(sos, lam, axis=1)
        ae = np.abs(e)
        E[b0:b1] = ae.mean(1)
        Z[b0:b1] = ((e[:, 1:] * e[:, :-1]) < 0).mean(1)
    return E, Z


def session_bands(args):
    sid, start_epoch, meals = args
    try:
        s = load_session(sid)
        fs = s.meta.get("row_rate", 105.0)
        if s.acc.shape[1] < int(WIN_S * fs):
            return None
        out = {}
        for name, src, band in BANDS:
            out[name] = band_feats(getattr(s, src), band, fs)
        n_w = len(out["b0_acc_0.5-2"][0])
        t0 = start_epoch + np.arange(n_w, dtype=np.int64) * 1000   # 与 _density_one 同构
        return (sid, out, t0, meals)
    except Exception as e:
        print(f"  {sid}: {type(e).__name__}: {e}", flush=True)
        return None


def auc_binary(pos, neg):
    """AUROC：升序 rank 法（pos 高于 neg 比例），与 train_ranker 同口径。"""
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), np.float64)
    ranks[order] = np.arange(len(allv))
    s = ranks[:n_p].sum()
    return float((s - n_p * (n_p - 1) / 2) / (n_p * n_n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    k = args.fold
    folds = splits.load_folds()
    f = folds[k]

    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    subject_of = dict(zip(idx["session_id"], idx["externalid"]))
    sid_meals = {}
    for _, r in idx.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms

    starts = {r["session_id"]: int(r["timeStamp.startTime"]) for _, r in idx.iterrows()}
    tasks = [(sid, starts.get(sid, 0), sid_meals.get(sid, [])) for sid in f["val_sessions"]]
    # 深度分数（canonical = ens3）加载，判定检出/未检出
    dl = {}
    if (config.OUTPUT_DIR / f"mm_ranker_fold{k}_val.npz").exists():
        z = np.load(config.OUTPUT_DIR / f"mm_ranker_fold{k}_val.npz", allow_pickle=True)
        for m, sc in zip(z["meta"], z["score"]):
            mm = json.loads(m.decode())
            dl[(mm["sid"], int(mm["s"]), int(mm["e"]))] = float(sc)

    t0_glob = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(session_bands, tasks, chunksize=4)):
            if r:
                rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} 用时 {time.time()-t0_glob:.0f}s", flush=True)
    print(f"加载完成：{len(rows)}/{len(tasks)} 会话", flush=True)

    # ---- 窗级标签：餐窗（±60s）/ 非餐窗（距餐 ≥30min） ----
    names = [b[0] for b in BANDS]
    FEATS = ["E", "Z"]                    # 能量 / 过零率
    per_band = {n: {ft: {"pos": [], "neg": []} for ft in FEATS} for n in names}
    meal_prof = []                        # ({band: {feat: 峰值}}, 检出标志)
    subj_auc = {n: {ft: [] for ft in FEATS} for n in names}
    for sid, envs, t0, meals in rows:
        n_w = len(envs[names[0]][0])
        centers = t0 + int(WIN_S / 2 * 1000)
        is_meal = np.zeros(n_w, bool)
        for m in meals:
            lo, hi = m["before"] - MEAL_PAD_S * 1000, m["after"] + MEAL_PAD_S * 1000
            is_meal |= (centers >= lo) & (centers <= hi)
        # 非餐窗：距所有餐段（膨胀后）≥30min
        far = np.ones(n_w, bool)
        for m in meals:
            lo, hi = m["before"] - MEAL_PAD_S * 1000, m["after"] + MEAL_PAD_S * 1000
            far &= ~((centers >= lo - NONMEAL_GAP_S * 1000) & (centers <= hi + NONMEAL_GAP_S * 1000))
        if not is_meal.any():
            continue
        for n in names:
            E, Z = envs[n]
            if not far.any():
                continue
            p95 = np.percentile(E, 95) + 1e-6
            per_band[n]["E"]["pos"].extend(E[is_meal] / p95)
            per_band[n]["E"]["neg"].extend(E[far] / p95)
            per_band[n]["Z"]["pos"].extend(Z[is_meal])     # 比例特征，跨会话可比，不归一
            per_band[n]["Z"]["neg"].extend(Z[far])
            subj_auc[n]["E"].append(auc_binary(E[is_meal] / p95, E[far] / p95))
            subj_auc[n]["Z"].append(auc_binary(Z[is_meal], Z[far]))
        # 餐级剖面：每餐在窗集合内的各频带峰值；深度分匹配判定检出
        for m in meals:
            lo, hi = m["before"] - MEAL_PAD_S * 1000, m["after"] + MEAL_PAD_S * 1000
            sel = (centers >= lo) & (centers <= hi)
            if not sel.any():
                continue
            prof = {}
            for n in names:
                E, Z = envs[n]
                p95 = np.percentile(E, 95) + 1e-6
                prof[n] = {"E": float(E[sel].max() / p95), "Z": float(Z[sel].max())}
            det = 0
            if dl:
                for (sid2, s0, e0), sc in dl.items():
                    if sid2 != sid:
                        continue
                    inter = min(e0, m["after"]) - max(s0, m["before"])
                    union = max(e0, m["after"]) - min(s0, m["before"])
                    if inter > 0 and inter / union >= 0.25:   # 事件级 IoU>0.25 贪心匹配口径
                        det = 1
                        break
            meal_prof.append((prof, det))

    # ---- 输出 ----
    FEAT_TAG = {"E": "能量", "S": "方差", "Z": "过零率", "A": "自相关峰"}
    print("\n=== fold{} 多频带×时域形态判别力（餐窗 vs 非餐窗，AUROC）===".format(k))
    print(f"{'特征':<20}{'b0_acc05-2':>11}{'b1_acc2-4':>11}{'b2_acc01-5':>11}{'b3_gyro':>11}")
    for ft in FEATS:
        g = [auc_binary(np.array(per_band[n][ft]["pos"]), np.array(per_band[n][ft]["neg"])) for n in names]
        print(f"{ft+' '+FEAT_TAG[ft]:<20}" + "".join(f"{x:>11.3f}" for x in g))
    print("\n逐受试者 AUROC 中位（E/Z 两特征）：")
    for ft in ("E", "Z"):
        meds = [float(np.median([x for x in subj_auc[n][ft] if x == x])) for n in names]
        print(f"{ft+' '+FEAT_TAG[ft]:<20}" + "".join(f"{x:>11.3f}" for x in meds))

    # ---- 检出 vs 未检出餐剖面 ----
    if meal_prof:
        dets = [p for p, d in meal_prof if d]
        und = [p for p, d in meal_prof if not d]
        print(f"\n餐剖面（窗内峰值）：检出 {len(dets)} / 未检出 {len(und)}")
        for ft in ("E", "Z"):
            print(f"{'特征':<20}{'检出均值':>10}{'未检出均值':>12}{'差异':>8}")
            for n in names:
                dv = np.array([p[n][ft] for p in dets]) if dets else np.array([0.0])
                uv = np.array([p[n][ft] for p in und]) if und else np.array([0.0])
                print(f"{n:<20}{dv.mean():>10.3f}{uv.mean():>12.3f}{dv.mean()-uv.mean():>+8.3f}")
            print()


if __name__ == "__main__":
    main()
