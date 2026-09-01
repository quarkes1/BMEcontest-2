# -*- coding: utf-8 -*-
"""官方事件级（Episode-level）评估与 IoU 后处理模块（阶段一，Resources/试题.txt 口径）。

官方指标（全局聚合，非按受试者）：
  IoU = Intersection / Union ≥ 0.25 → TP（贪心一对一匹配）
  灵敏度 = TP / 真实事件数；PPV = TP / 预测事件数；F1 = 2·Sens·PPV/(Sens+PPV)
  次要指标：MAE（正确匹配事件的起止时间误差均值）

IoU 专属后处理流水线（官方要求）：
  1) 平滑：预测概率序列 60s 滑动平均
  2) Episode Fusion：时间间隔 < 180s 的相邻预测框合并
  3) Min Duration Filtering：总时长 < 120s 的孤立框过滤

运行（复用 rank_events_v2.prepare_fold 打分，无复刻泄漏）：
  python scripts/official_iou_eval.py --fold 0 --cfg w0.7_t0.325_g0.3_p0.5_d120_k2
  python scripts/official_iou_eval.py --all        # 5 折 best 配置全跑 + 汇总对比表
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.eval.metrics import event_iou
import rank_events_v2 as v2

IOU_THR = 0.25          # 官方 IoU 阈值（试题：IoU>0.25）
SMOOTH_S = 60.0         # 官方：概率序列 60s 滑动平均
FUSE_GAP_S = 180.0      # 官方：Episode Fusion 间隔 <3min 合并
MIN_DUR_S = 120.0       # 官方：过滤 <2min 孤立框


def official_metrics(preds, gts):
    """全局事件级评估：preds/gts 为 [(sid, (s, e))] 或 [(s, e)]（同列表内键须一致）。
    返回 dict(sens, ppv, f1, n_tp, n_pred, n_true, mae_s)。"""
    if not preds:
        return {"f1": 0.0, "sensitivity": 0.0, "ppv": 0.0,
                "n_tp": 0, "n_pred": 0, "n_true": len(gts), "mae_s": None}
    # 同 sid 组内贪心匹配（与竞赛一致：一对一，按 IoU 降序）
    pairs = []
    for i, (sid_p, (ps, pe)) in enumerate(preds):
        for j, (sid_g, (gs, ge)) in enumerate(gts):
            if sid_p != sid_g:
                continue
            iou = event_iou((ps, pe), (gs, ge))
            if iou >= IOU_THR:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda t: -t[0])
    used_p, used_g = set(), set()
    tp_pairs = []
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        tp_pairs.append((iou, i, j))
    n_tp = len(tp_pairs)
    n_pred, n_true = len(preds), len(gts)
    sens = n_tp / n_true if n_true else 0.0
    ppv = n_tp / n_pred if n_pred else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv > 0 else 0.0
    # MAE：正确匹配事件的起止时间绝对误差均值（次要指标）
    mae = None
    if tp_pairs:
        errs = []
        for _, i, j in tp_pairs:
            (ps, pe) = preds[i][1]; (gs, ge) = gts[j][1]
            errs.extend([abs(ps - gs), abs(pe - ge)])
        mae = float(np.mean(errs) / 1000.0)   # ms → s
    return {"f1": f1, "sensitivity": sens, "ppv": ppv,
            "n_tp": n_tp, "n_pred": n_pred, "n_true": n_true, "mae_s": mae}


def fuse_events(evs, gap_s=FUSE_GAP_S):
    """Episode Fusion：按起始排序，间隔 < gap_s 的相邻框合并（断点连通）。"""
    if not evs:
        return []
    merged = []
    for s, e in sorted(evs):
        if merged and s - merged[-1][1] <= gap_s * 1000:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def min_dur_filter(evs, min_s=MIN_DUR_S):
    """Min Duration Filtering：总时长 < min_s 的孤立框过滤。"""
    return [(s, e) for s, e in evs if (e - s) / 1000.0 >= min_s]


def map_to_1hz_timeseries(cands, fuse, t0, t_end):
    """候选概率 → 1Hz 全天连续时间轴（v1.1 设计锁定）。

    连续轴上每 1s 网格取覆盖它的候选窗概率最大值（重叠窗取 max——冲突消解）。
    返回 (grid, series)：grid 为秒级时间戳（ms），series 为对应概率。"""
    n = max(1, (t_end - t0) // 1000)
    grid = t0 + np.arange(n, dtype=np.int64) * 1000
    series = np.zeros(n, np.float64)
    for c, sc in zip(cands, fuse):
        cs, ce = c[0], c[1]                      # 候选三元组 (s, e, is_prior)
        lo = max(0, (cs - t0) // 1000)
        hi = min(n, (ce - t0) // 1000 + 1)
        if hi > lo:
            np.maximum(series[lo:hi], sc, out=series[lo:hi])
    return grid, series


def smooth_series(series, win_s=SMOOTH_S):
    """连续概率序列 60s 高斯/移动平均平滑（v1.1：作用于 1Hz 时间轴）。"""
    half = max(1, int(win_s))
    k = np.exp(-0.5 * ((np.arange(-half, half + 1) / (win_s / 2.0)) ** 2))
    k /= k.sum()
    padded = np.pad(series, half, mode="edge")
    return np.convolve(padded, k, mode="valid")


def threshold_episodes(grid, series, tau):
    """阈值截断生成 Episode：连续 ≥tau 的 1s 段 → (s, e) 框（s/e 为 ms）。"""
    on = series >= tau
    if not on.any():
        return []
    evs = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            evs.append((int(grid[i]), int(grid[j - 1]) + 1000))   # 闭区间右端 +1s
            i = j
        else:
            i += 1
    return evs


def decode_official(row, gate_prob, cfg, clf_pri, post=True, dilate_s=None):
    """官方后处理解码（v1.1 Pipeline 锁定）：
    候选概率 → 1Hz 连续时间轴 → 60s 平滑 → 阈值截断 → 180s 融合 → 120s 过滤
    →（可选）边界膨胀。post=False 仅原始选窗（跳过平滑，直接阈值选窗）。"""
    sid, act, sa, va_scores, pri, sp = row
    w, tau, thr_g, thr_p, dil, K = cfg
    out = []
    if gate_prob.get(sid, 1.0) >= thr_g and len(sa):
        fuse = w * va_scores + (1 - w) * sa
        fuse = np.where(np.isnan(va_scores), sa, fuse) if w > 0 else sa
        if post:
            t0 = int(act[0][0]) if act else 0
            t_end = int(act[-1][1]) if act else 0
            grid, series = map_to_1hz_timeseries(act, fuse, t0, t_end)
            series = smooth_series(series)
            evs = threshold_episodes(grid, series, tau)
            evs = min_dur_filter(fuse_events(evs))
            if dilate_s:
                evs = [(max(0, int(s - dilate_s * 1000)), int(e + dilate_s * 1000)) for s, e in evs]
            out = evs
        else:
            sel = np.where(fuse >= tau)[0]
            if K > 0 and len(sel) > K:
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
            out = [act[j][:2] for j in sel]
    return [(sid, e) for e in out]


def run_fold(k, cfg_name, official_post=True, legacy_post=True, dilate_s=None):
    """对 fold k 指定配置：同时输出官方后处理与现有管线（legacy）的事件级 F1。"""
    val_rows, gate_prob, clf_pri, true_sid, _, _ = v2.prepare_fold(k)
    mm = re.match(r"w([\d.]+)_t([\d.]+)_g([\d.]+)_p([\d.]+)_d(\d+)_k(\d+)", cfg_name)
    cfg = (float(mm[1]), float(mm[2]), float(mm[3]), float(mm[4]), int(mm[5]), int(mm[6]))

    pred_off = []
    for row in val_rows:
        pred_off.extend(decode_official(row, gate_prob, cfg, clf_pri, post=official_post,
                                        dilate_s=dilate_s))
    m_off = official_metrics(pred_off, true_sid)

    # 现有管线（decode_session 内 postprocess_events_prov：merge 120s / min 120s / dilation）
    pred_leg = []
    for row in val_rows:
        pred_leg.extend(v2.decode_session(row, gate_prob, cfg, clf_pri)[0])
    m_leg = official_metrics(pred_leg, true_sid)
    return m_off, m_leg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--cfg", default=None, help="配置名 w.._t.._g.._p.._d.._k..（默认各折 best）")
    ap.add_argument("--all", action="store_true", help="5 折 best 配置全跑 + 汇总对比表")
    args = ap.parse_args()

    if not args.all and args.fold is None:
        print("需 --fold k [--cfg name] 或 --all"); return

    folds_to_run = range(5) if args.all else [args.fold]
    header = (f"{'fold':>5} | {'原始选窗':>8} {'官方v1.1':>9} {'官方+60s膨':>10} "
              f"{'现有管线':>8} | cfg")
    print(header)
    print("-" * len(header))
    tot = {"raw": [], "off": [], "dil": [], "leg": []}
    for k in folds_to_run:
        cfg_name = args.cfg
        if not cfg_name:
            j = json.loads((config.OUTPUT_DIR / f"rank_events_v2_fold{k}_15m.json").read_text(encoding="utf-8"))
            cfg_name = j["best"]["name"]
        m_off, m_leg = run_fold(k, cfg_name)
        val_rows, gate_prob, clf_pri, true_sid, _, _ = v2.prepare_fold(k)
        mm = re.match(r"w([\d.]+)_t([\d.]+)_g([\d.]+)_p([\d.]+)_d(\d+)_k(\d+)", cfg_name)
        cfg = (float(mm[1]), float(mm[2]), float(mm[3]), float(mm[4]), int(mm[5]), int(mm[6]))
        def dec(post, dilate_s=None):
            return official_metrics(
                sum((decode_official(r, gate_prob, cfg, clf_pri, post=post, dilate_s=dilate_s)
                     for r in val_rows), []), true_sid)
        m_raw = dec(False)
        m_dil = dec(True, 60.0)
        for tag, m in (("raw", m_raw), ("off", m_off), ("dil", m_dil), ("leg", m_leg)):
            tot[tag].append(m)
        print(f"{k:>5} | {m_raw['f1']:>8.3f} {m_off['f1']:>9.3f} {m_dil['f1']:>10.3f} "
              f"{m_leg['f1']:>8.3f} | {cfg_name}")
    if args.all:
        print("-" * len(header))
        for tag, name in (("raw", "原始选窗"), ("off", "官方 v1.1 后处理"),
                          ("dil", "官方+60s 膨胀"), ("leg", "现有管线")):
            arr = tot[tag]
            print(f"{name} 均值: F1={np.mean([m['f1'] for m in arr]):.3f} "
                  f"Sens={np.mean([m['sensitivity'] for m in arr]):.3f} "
                  f"PPV={np.mean([m['ppv'] for m in arr]):.3f} "
                  f"MAE={np.mean([m['mae_s'] for m in arr if m['mae_s'] is not None]):.0f}s")


if __name__ == "__main__":
    main()
