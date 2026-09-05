# -*- coding: utf-8 -*-
"""decode 实验：act/pri 通道独立门控 + 官方口径快速扫描（一次性分析，不改 v2 架构）。

动机（fold0 分层分解）：40 餐中 12 餐被 gate(<0.45) 杀死（gate AUC 仅 0.729 且含餐
会话 median 0.21）、19 餐无 act 覆盖、act 通道高 tau 几乎无 FP、pri 通道（LGBM）FP 泛滥。
假设：act 通道放宽 gate + pri 通道独立高门控 → 救回被杀餐且压住 pri FP。

运行：D:/Anaconda3/envs/bme/python.exe scripts/diag_decode_split.py --fold 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rank_events_v2 as v2
import official_iou_eval as oe
import src.eval.metrics as mt

IOU = 0.25


def act_decode(row, gate_prob, tau, thr_g, dil):
    sid, act, sa, va, pri, sp = row
    if gate_prob.get(sid, 1.0) < thr_g or not len(sa):
        return []
    fuse = np.where(np.isnan(va), sa, va)   # 纯深度分（NaN→LGBM）
    sel = np.where(fuse >= tau)[0]
    if not len(sel):
        return []
    # 连续轴官方后处理（act 窗分数铺 1Hz）
    t0 = int(act[0][0]); t_end = int(act[-1][1])
    grid, series = oe.map_to_1hz_timeseries(act, fuse, t0, t_end)
    series = oe.smooth_series(series)
    evs = oe.threshold_episodes(grid, series, tau)
    evs = oe.min_dur_filter(oe.fuse_events(evs))
    if dil:
        evs = [(max(0, int(s - dil * 1000)), int(e + dil * 1000)) for s, e in evs]
    return [(sid, e) for e in evs]


def pri_decode(row, gate_prob, thr_p, thr_g):
    sid, act, sa, va, pri, sp = row
    if gate_prob.get(sid, 1.0) < thr_g or len(sp) == 0:
        return []
    out = []
    for jp in np.argsort(sp)[::-1]:
        if sp[jp] < thr_p:
            break
        pc = (pri[jp][0], pri[jp][1])
        if any(mt.event_iou(pc, e) >= IOU for e in out):
            continue
        out.append(pc)
        if len(out) >= 2:
            break
    return [(sid, e) for e in out]


def topk_decode(row, gate_prob, K, tau_min, thr_g, dil):
    """会话内 top-K 相对解码：取会话内深度分最高且 ≥tau_min 的 K 个非重叠窗，
    事件框 = 候选窗边界 ±dil（窗本身即活动段，相对阈值规避全局分数重叠）。"""
    sid, act, sa, va, pri, sp = row
    if gate_prob.get(sid, 1.0) < thr_g or not len(sa):
        return []
    fuse = np.where(np.isnan(va), sa, va)
    if float(np.nanmax(fuse)) < tau_min:
        return []
    sel = np.where(fuse >= tau_min)[0]
    if not len(sel):
        return []
    order = np.argsort(fuse[sel])[::-1]
    picked = []
    for j in order:
        c = tuple(act[sel[j]][:2])
        if any(mt.event_iou(c, pc) >= 0.5 for pc in picked):
            continue
        picked.append(c)
        if len(picked) >= K:
            break
    out = []
    for (s, e) in picked:
        if dil:
            out.append((max(0, int(s - dil * 1000)), int(e + dil * 1000)))
        else:
            out.append((s, e))
    return [(sid, ev) for ev in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    val_rows, gate_prob, clf_pri, true_sid, _, _ = v2.prepare_fold(args.fold)
    print(f"val 会话 {len(val_rows)} | 真值 {len(true_sid)}", flush=True)

    # ---- 1) act-only：gate 扫描 × tau × dilation ----
    best = None
    for thr_g in (0.0, 0.1, 0.2, 0.3, 0.45, 0.6):
        for tau in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            for dil in (60.0, 120.0):
                preds = sum((act_decode(r, gate_prob, tau, thr_g, dil) for r in val_rows), [])
                m = oe.official_metrics(preds, true_sid)
                tag = f"g{thr_g}_t{tau}_d{dil:.0f}"
                if best is None or m["f1"] > best[1]["f1"]:
                    best = (tag, m)
    print(f"[act-only] 最佳: {best[0]} F1={best[1]['f1']:.3f} sens={best[1]['sensitivity']:.3f} "
          f"ppv={best[1]['ppv']:.3f} ({best[1]['n_tp']}/{best[1]['n_true']}, pred={best[1]['n_pred']})", flush=True)
    # 全网格 top10
    rows = []
    for thr_g in (0.0, 0.1, 0.2, 0.3, 0.45):
        for tau in (0.10, 0.15, 0.20, 0.25, 0.30):
            for dil in (60.0, 120.0):
                preds = sum((act_decode(r, gate_prob, tau, thr_g, dil) for r in val_rows), [])
                m = oe.official_metrics(preds, true_sid)
                rows.append((m["f1"], f"g{thr_g}_t{tau}_d{dil:.0f}", m))
    for f1, tag, m in sorted(rows, key=lambda x: -x[0])[:6]:
        print(f"  {tag}: F1={f1:.3f} sens={m['sensitivity']:.3f} ppv={m['ppv']:.3f} ({m['n_tp']}/{m['n_true']}, pred={m['n_pred']})", flush=True)

    # ---- 1.5) 会话内 top-K 相对解码 ----
    bestK = None
    for thr_g in (0.0, 0.1, 0.2, 0.3, 0.45):
        for K in (1, 2, 3):
            for tau_min in (0.03, 0.05, 0.08):
                for dil in (0.0, 60.0, 120.0):
                    preds = sum((topk_decode(r, gate_prob, K, tau_min, thr_g, dil)
                                 for r in val_rows), [])
                    m = oe.official_metrics(preds, true_sid)
                    tag = f"top{K}_g{thr_g}_tm{tau_min}_d{dil:.0f}"
                    if bestK is None or m["f1"] > bestK[1]["f1"]:
                        bestK = (tag, m)
    print(f"[top-K] 最佳: {bestK[0]} F1={bestK[1]['f1']:.3f} sens={bestK[1]['sensitivity']:.3f} "
          f"ppv={bestK[1]['ppv']:.3f} ({bestK[1]['n_tp']}/{bestK[1]['n_true']}, pred={bestK[1]['n_pred']})", flush=True)
    rowsK = []
    for thr_g in (0.0, 0.2, 0.3):
        for K in (2, 3):
            for tau_min in (0.05, 0.08):
                for dil in (0.0, 120.0):
                    preds = sum((topk_decode(r, gate_prob, K, tau_min, thr_g, dil)
                                 for r in val_rows), [])
                    m = oe.official_metrics(preds, true_sid)
                    rowsK.append((m["f1"], f"top{K}_g{thr_g}_tm{tau_min}_d{dil:.0f}", m))
    for f1, tag, m in sorted(rowsK, key=lambda x: -x[0])[:6]:
        print(f"  {tag}: F1={f1:.3f} sens={m['sensitivity']:.3f} ppv={m['ppv']:.3f} ({m['n_tp']}/{m['n_true']}, pred={m['n_pred']})", flush=True)

    # ---- 2) act(低门控) + pri(独立高门控) ----
    best2 = None
    for thr_g in (0.0, 0.15, 0.3):
        for tau in (0.10, 0.15, 0.20, 0.25):
            for thr_p in (0.60, 0.75, 0.90):
                for thr_gp in (0.45, 0.65, 0.85):
                    preds = sum((act_decode(r, gate_prob, tau, thr_g, 120.0) +
                                 pri_decode(r, gate_prob, thr_p, thr_gp) for r in val_rows), [])
                    m = oe.official_metrics(preds, true_sid)
                    tag = f"act_g{thr_g}_t{tau} + pri_p{thr_p}_gp{thr_gp}"
                    if best2 is None or m["f1"] > best2[1]["f1"]:
                        best2 = (tag, m)
    print(f"[act+pri独立门控] 最佳: {best2[0]} F1={best2[1]['f1']:.3f} sens={best2[1]['sensitivity']:.3f} "
          f"ppv={best2[1]['ppv']:.3f} ({best2[1]['n_tp']}/{best2[1]['n_true']}, pred={best2[1]['n_pred']})", flush=True)
    # act-only 最佳之上叠加 pri 小网格
    bt, m0 = best
    print(f"  act-only best pred={m0['n_pred']} 下叠加 pri...", flush=True)
    best3 = None
    for thr_p in (0.5, 0.6, 0.7, 0.8):
        for thr_gp in (0.3, 0.45, 0.6, 0.8):
            preds = sum((act_decode(r, gate_prob, float(bt.split("_t")[1].split("_")[0]),
                                    float(bt.split("_g")[1].split("_")[0][:3]), 120.0) +
                         pri_decode(r, gate_prob, thr_p, thr_gp) for r in val_rows), [])
            m = oe.official_metrics(preds, true_sid)
            if best3 is None or m["f1"] > best3[1]["f1"]:
                best3 = (f"pri_p{thr_p}_gp{thr_gp}", m)
    print(f"[叠加 pri] 最佳: {best3[0]} F1={best3[1]['f1']:.3f} sens={best3[1]['sensitivity']:.3f} "
          f"ppv={best3[1]['ppv']:.3f} ({best3[1]['n_tp']}/{best3[1]['n_true']}, pred={best3[1]['n_pred']})", flush=True)


if __name__ == "__main__":
    main()
