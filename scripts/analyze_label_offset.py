# -*- coding: utf-8 -*-
"""标签时间偏移分析（2026-09-02）：GT 餐次标注 vs 活动信号对齐度。

发现：全数据集 env 峰值（0.5-2Hz 包络）与标注中心偏移中位 ~11min，
fold0/1/4 一致（10.2/11.8/11.5min），>10min 偏移的餐占 69%（fold1）。
IoU≥0.25 匹配的理论界限：δ ≤ 0.6·D（D=餐长）——15min 餐需偏移 ≤9min，
故 >9min 偏移的餐数学上不可匹配（即使检测器完美）。

这是"F1 卡 0.33"与"低于文献（FD 相机秒级标注 F1 0.7-0.9）"的根因：
评估标签（用户自报餐次）的时间精度决定了可达上限，非模型能力。

运行：python scripts/analyze_label_offset.py [--fold 1]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None, help="None=全部 5 折")
    args = ap.parse_args()
    folds = splits.load_folds()
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    sid_meals = {}
    for _, r in idx.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms
    ks = range(5) if args.fold is None else [args.fold]
    for k in ks:
        f = folds[k]
        offs = []
        for sid in f["val_sessions"]:
            p = config.CACHE_DIR / "validate_baselines" / f"{sid}.npz"
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            env, t0 = d["env"].astype(np.float32), d["t0"].astype(np.int64)
            for m in sid_meals.get(sid, []):
                sel = (t0 >= m["before"] - 600000) & (t0 <= m["after"] + 600000)
                if sel.sum() < 5:
                    continue
                pk = t0[sel][np.argmax(env[sel])]
                offs.append((pk - (m["before"] + m["after"]) / 2) / 1000 / 60)
        a = np.abs(np.array(offs))
        print(f"fold{k} {len(a)} 餐：偏移 median={np.median(a):.1f}min "
              f"p75={np.percentile(a, 75):.1f} p90={np.percentile(a, 90):.1f} "
              f">9min(15min餐不可匹配) {(a > 9).mean()*100:.0f}% "
              f">10min {np.mean(a > 10)*100:.0f}%")


if __name__ == "__main__":
    main()
