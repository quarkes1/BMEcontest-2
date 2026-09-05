# -*- coding: utf-8 -*-
"""临时诊断：真实时间轴缓存下 fold0 val 的提案结构。
对比对象：b10 旧缓存 32650 候选（train+val）/ 103 正（fold0）。
目标：量化 1) act 提案碎片化（时长分布）2) 每餐正窗覆盖（IoU≥0.25 匹配）。
运行：D:/Anaconda3/envs/bme/python.exe scripts/diag_proposals.py
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.data import manifests, splits
import rank_events as re

# ---- 与 build_candidate_windows.py 相同的输入装配 ----
folds = splits.load_folds()
f = folds[0]
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

DILATE = 60000
tot_act = tot_cand = 0
dur_bins = {"<30s": 0, "30-60s": 0, "60-120s": 0, "120-300s": 0, ">300s": 0}
meal_cov = {"meals": 0, "cov_ok": 0, "cov_fail": [], "best_iou": []}
fail_examples = []
n_val_sess = 0
for sid in f["val_sessions"]:
    p = config.CACHE_DIR / "validate_baselines" / f"{sid}.npz"
    if not p.exists():
        continue
    d = np.load(p)
    env = d["env"].astype(np.float32)
    t0 = d["t0"].astype(np.int64)
    meals = sid_meals.get(sid, [])
    act, pri = re.make_proposals(env, t0, prior, starts.get(sid, 0), dilate_ms=DILATE)
    n_val_sess += 1
    tot_act += len(act)
    for s, e, _ in act:
        dur = (e - s) / 1000
        if dur < 30: dur_bins["<30s"] += 1
        elif dur < 60: dur_bins["30-60s"] += 1
        elif dur < 120: dur_bins["60-120s"] += 1
        elif dur < 300: dur_bins["120-300s"] += 1
        else: dur_bins[">300s"] += 1
    for m in meals:
        meal_cov["meals"] += 1
        ious = [re.event_iou((c[0], c[1]), (m["before"], m["after"])) for c in act]
        if ious:
            bi = max(ious)
            meal_cov["best_iou"].append(bi)
            if bi >= re.IOU_LABEL:
                meal_cov["cov_ok"] += 1
            else:
                meal_cov["cov_fail"].append((sid[-8:], bi))
                if len(fail_examples) < 5:
                    fail_examples.append((sid, m["before"], m["after"], bi,
                                          [(c[0], c[1]) for c in act[:5]]))
        else:
            meal_cov["cov_fail"].append((sid[-8:], 0.0))
            if len(fail_examples) < 5:
                fail_examples.append((sid, m["before"], m["after"], 0.0, []))
    # 每会话正窗数（match_labels）
    if meals:
        y = re.match_labels(act, meals)
        pos = int(y.sum())
        if pos:
            meal_cov.setdefault("pos_per_sess", []).append(pos)
        else:
            meal_cov.setdefault("pos_per_sess", []).append(0)
print(f"val 会话数（有 env 缓存）: {n_val_sess}")
print(f"act 提案总数: {tot_act}（均值 {tot_act/max(n_val_sess,1):.0f}/会话）")
print("act 时长分布:", json.dumps(dur_bins))
print(f"餐覆盖: {meal_cov['cov_ok']}/{meal_cov['meals']} 餐 IoU≥{re.IOU_LABEL}"
      f"（{meal_cov['cov_ok']/max(meal_cov['meals'],1)*100:.0f}%）")
cov_iou = meal_cov["best_iou"]
if cov_iou:
    print(f"每餐最大 IoU: 中位 {np.median(cov_iou):.2f} p25 {np.percentile(cov_iou,25):.2f}"
          f" p75 {np.percentile(cov_iou,75):.2f} 最小 {min(cov_iou):.2f}")
print("失败餐示例（sid尾, before, after, best_iou, 前5个act）:")
for sid, b, a, bi, acts in fail_examples:
    print(f"  {sid} meal[{b//1000//60}:{a//1000//60}] best_iou={bi:.2f} act={acts[:3]}")
pp = meal_cov.get("pos_per_sess", [])
print(f"含餐会话正窗数: {pp}")
