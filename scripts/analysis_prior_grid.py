# -*- coding: utf-8 -*-
"""先验网格密度分析：不同 (网格步长, 窗宽) 设计对 val 餐的 IoU 覆盖上限。

对每个 val 餐计算"与任一先验候选窗的最大 IoU"（会话裁剪 + 最小宽过滤与
prior_candidates 一致）→ 覆盖率 = maxIoU ≥ 0.25 的比例。

设计：
  A: 60min 网格 / 40min 窗（当前线上）
  B: 30min 网格 / 40min 窗
  C: 15min 网格 / 15min 窗
  D: B ∪ C 双通道
另附餐时长分布与当前未覆盖餐画像（时长/距整点偏移）。

运行：source activate bme && python scripts/analysis_prior_grid.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import manifests, splits
import rank_events as re

IOU = 0.25
MIN_W_S = 300.0      # 与 PRIOR_MIN_W_S 一致


def prior_windows(sess_s, sess_e, step_s, half_w_s):
    """网格铺窗（复刻 prior_candidates 逻辑，参数化步长/半宽）。"""
    out = []
    base = sess_s - (sess_s % (step_s * 1000))
    n = int(np.ceil((sess_e - sess_s) / (step_s * 1000))) + 2
    for off in range(n):
        g = base + off * step_s * 1000
        s, e = int(g - half_w_s * 1000), int(g + half_w_s * 1000)
        s, e = max(s, sess_s), min(e, sess_e)
        if e - s >= MIN_W_S * 1000:
            out.append((s, e))
    return out


DESIGNS = {
    "A 60m/40m(now)": (3600, 1200),
    "B 30m/40m":      (1800, 1200),
    "C 15m/15m":      (900, 450),
}


def main():
    folds = splits.load_folds()
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    starts = {r["session_id"]: int(r["timeStamp.startTime"]) for _, r in idx.iterrows()}
    ends = {r["session_id"]: int(r["timeStamp.endTime"]) for _, r in idx.iterrows()}
    sid_meals = {}
    for _, r in idx.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms

    print(f"{'fold':>4} | {'meals':>5} {'<10m':>4} | " +
          " | ".join(f"{d:>16}" for d in DESIGNS) + " | unlockedB+C(meals not in A)")
    all_rows = []
    for k in range(5):
        f = folds[k]
        rows = []
        for sid in f["val_sessions"]:
            for m in sid_meals.get(sid, []):
                dur = (m["after"] - m["before"]) / 60_000.0
                c = (m["before"] + m["after"]) / 2
                off_h = (c / 3.6e6) % 1.0                    # 距整点的小时偏移 [0,1)
                off_min = min(off_h, 1 - off_h) * 60
                ss, se = starts.get(sid, 0), ends.get(sid, 0)
                best = {}
                for d, (step, hw) in DESIGNS.items():
                    iou = [re.event_iou((m["before"], m["after"]), (s, e))
                           for s, e in prior_windows(ss, se, step, hw)]
                    best[d] = max(iou) if iou else 0.0
                rows.append({"sid": sid, "dur": dur, "off_min": off_min, **best})
        n_meal = len(rows)
        n_short = sum(r["dur"] < 10 for r in rows)
        cov = {d: sum(r[d] >= IOU for r in rows) / max(n_meal, 1) for d in DESIGNS}
        unA = [r for r in rows if r["A 60m/40m(now)"] < IOU]
        unlocked = sum(1 for r in unA if max(r["B 30m/40m"], r["C 15m/15m"]) >= IOU)
        print(f"{k:>4} | {n_meal:>5} {n_short:>4} | " +
              " | ".join(f"{cov[d]*100:5.1f}% " for d in DESIGNS) +
              f" | {unlocked}/{len(unA)}")
        all_rows.extend(rows)

    print("\n时长分布（全部 val 餐）:")
    for lo in (0, 5, 10, 15, 20, 30):
        hi = lo + 5 if lo < 30 else 1e9
        n = sum(lo <= r["dur"] < hi for r in all_rows)
        if n:
            print(f"  {lo:>3}-{hi:<5.0f}min: {n} 餐 ({n/len(all_rows)*100:.0f}%)")
    print("\nA 未覆盖餐画像（距整点偏移 / 时长）:")
    unA = [r for r in all_rows if r["A 60m/40m(now)"] < IOU]
    print(f"  共 {len(unA)} 餐: 时长中位 {np.median([r['dur'] for r in unA]):.1f}min, "
          f"偏移中位 {np.median([r['off_min'] for r in unA]):.1f}min")
    for d in DESIGNS:
        n_cov = sum(r[d] >= IOU for r in unA)
        print(f"  {d}: 解锁 {n_cov}/{len(unA)} ({n_cov/max(len(unA),1)*100:.0f}%)")


if __name__ == "__main__":
    main()
