# -*- coding: utf-8 -*-
"""漏检归因（v4.2 wbag）：每折 eligible 餐分类为 候选可达（复核前 TP）/ 候选缺失。
运行：D:/Anaconda3/envs/bme/python.exe scripts/diag_miss_attribution.py
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent) if False else ".")
sys.path.insert(0, "scripts")

import slide_verifier as sv
import src.config as config
import official_iou_eval as oe
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

out_lines = []
tot_elig = tot_tp = tot_miss = tot_nocand = 0
for k in range(5):
    tr = np.load(config.CACHE_DIR / "slide" / f"fold{k}_train.npz", allow_pickle=True)
    keep = tr["label"] >= 0

    def ltv(sn, kk):
        p = config.CACHE_DIR / "slide" / f"fold{kk}_{sn}_tcn.npz"
        if not p.exists():
            return None
        d0 = np.load(config.CACHE_DIR / "slide" / f"fold{kk}_{sn}.npz", allow_pickle=True)
        t = np.load(p)
        o = np.zeros((len(d0["wid"]), 2), np.float32)
        for j, s in enumerate(t["score"]):
            if not np.isnan(s):
                o[j, 0] = s
        for j, w in enumerate([json.loads(x) for x in d0["wid"]]):
            hh = int((w[1] / 3.6e6) % 24)
            o[j, 1] = sv.GLOBAL_PRIOR[hh]
        return o

    models = []
    for kk in range(5):
        trk = np.load(config.CACHE_DIR / "slide" / f"fold{kk}_train.npz", allow_pickle=True)
        keepk = trk["label"] >= 0
        tv = ltv("train", kk)
        Xk = trk["feat"] if tv is None else np.concatenate([trk["feat"], tv], 1)
        impk = SimpleImputer(strategy="median").fit(Xk[keepk])
        clfk = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
            min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
            random_state=20260901 + kk)
        clfk.fit(impk.transform(Xk[keepk]), trk["label"][keepk].astype(int))
        models.append((impk, clfk))
    d = np.load(config.CACHE_DIR / "slide" / f"fold{k}_val.npz", allow_pickle=True)
    wids = [json.loads(w) for w in d["wid"]]
    Xv = np.concatenate([d["feat"], ltv("val", k)], 1)
    prob = np.mean([m2.predict_proba(i2.transform(Xv))[:, 1] for i2, m2 in models], 0)
    sw = defaultdict(list)
    for wid, p in zip(wids, prob):
        sw[wid[0]].append((wid[1], wid[2], float(p)))
    true_va = sv.eligible_meals(set(sw.keys()), sw)
    cands = sv.density_candidates(sw, 0.28838)
    # 每餐分类
    n_nocand = n_tp = 0
    miss_ex = []
    for sid, (gs, ge) in true_va:
        ious = [oe.event_iou((c[1], c[2]), (gs, ge)) for c in cands if c[0] == sid]
        best = max(ious) if ious else 0.0
        if best >= 0.25:
            n_tp += 1
        else:
            n_nocand += 1
            if len(miss_ex) < 3:
                miss_ex.append((sid[-6:], int((ge - gs) / 60000), round(best, 2)))
    out_lines.append(f"fold{k}: eligible {len(true_va)} | 候选可达 {n_tp} | 候选缺失 {n_nocand}")
    for ex in miss_ex:
        out_lines.append(f"    缺失例: {ex}")
    tot_elig += len(true_va); tot_tp += n_tp; tot_nocand += n_nocand
out_lines.append(f"合计: eligible {tot_elig} | 候选可达 {tot_tp} ({tot_tp/tot_elig:.3f}) | 候选缺失 {tot_nocand}")
open("outputs/miss_attr.txt", "w").write("\n".join(out_lines))
print("done")
