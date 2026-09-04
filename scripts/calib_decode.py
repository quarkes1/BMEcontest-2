# -*- coding: utf-8 -*-
"""校准修复快速原型：per-session 分位数动态截断（替代绝对 τ）。

背景：B10 排序头排序能力强（AUROC 0.874）但分数校准差（正窗 sigmoid 0.142，
绝对阈值无法分离）。分位数截断按会话分数分布取动态阈值——免疫绝对尺度漂移。

用法：python scripts/calib_decode.py --npz outputs/b10_fold0.npz --fold 0
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
import rank_events_v2 as v2
import official_iou_eval as oe


def decode_quantile(row, gate_prob, clf_pri, q, thr_g, thr_p, dil, K):
    """per-session 分位数截断：该会话候选深度分 > 会话内 q 分位数才选。"""
    sid, act, sa, va_scores, pri, sp = row
    out = []
    if gate_prob.get(sid, 1.0) >= thr_g and len(sa):
        fuse = np.where(np.isnan(va_scores), sa, va_scores)
        finite = fuse[~np.isnan(fuse)]
        if len(finite) == 0:
            return []
        tau_dyn = float(np.percentile(finite, q))
        sel = np.where(fuse >= tau_dyn)[0]
        if K > 0 and len(sel) > K:
            order = np.argsort(fuse[sel])[::-1]
            picked = []
            for j in order:
                c = act[sel[j]][:2]
                if any(oe.event_iou(c, act[sel[p]][:2]) >= 0.5 for p in picked):
                    continue
                picked.append(j)
                if len(picked) >= K:
                    break
            sel = sel[np.array(picked)]
        evs = [act[j][:2] for j in sel]
        out = oe.min_dur_filter(oe.fuse_events(evs, gap_s=120.0), min_s=120.0)
        out = [(max(0, int(s - dil * 1000)), int(e + dil * 1000)) for s, e in out]
    return [(sid, e) for e in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    k = args.fold
    z = np.load(args.npz, allow_pickle=True)
    dl = {}
    for x, sc in zip(z["meta"], z["score"]):
        m = json.loads(x.decode())
        dl[(m["sid"], int(m["s"]), int(m["e"]))] = float(sc)
    val_rows, gate_prob, clf_pri, true_sid, _, _ = v2.prepare_fold(k)
    rows2 = []
    for r in val_rows:
        sid, act, sa, _, pri, sp = r
        vs = np.full(len(act), np.nan, np.float32)
        for jj, c in enumerate(act):
            if (sid, c[0], c[1]) in dl:
                vs[jj] = dl[(sid, c[0], c[1])]
        rows2.append((sid, act, sa, vs, pri, sp))
    best = None
    for q in (80, 85, 88, 90, 92, 95):
        for thr_g in (0.0, 0.3):
            for K in (0, 1, 2, 3):
                for dil in (60.0, 120.0):
                    pred = []
                    for r in rows2:
                        pred.extend(decode_quantile(r, gate_prob, clf_pri, q, thr_g, 0.7, dil, K))
                    m = oe.official_metrics(pred, true_sid)
                    tag = f"q{q}_g{thr_g}_k{K}_d{dil:.0f}"
                    print(f"  {tag}: F1={m['f1']:.3f} sens={m['sensitivity']:.3f} ppv={m['ppv']:.3f} "
                          f"TP={m['n_tp']}/{m['n_true']} pred={m['n_pred']}", flush=True)
                    if best is None or m["f1"] > best[1]["f1"]:
                        best = (tag, m)
    print(f"\n★ 最优: {best[0]} F1={best[1]['f1']:.3f}")


if __name__ == "__main__":
    main()
