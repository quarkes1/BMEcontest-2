# -*- coding: utf-8 -*-
"""FD 数据集滑窗联合训练验证（fold 指定）。

FD 缓存（cache/fd_windows/fd_pretrain7.npz）：15440 窗 @10Hz（2400×7ch），正 1218
（相机秒级标注 Episode 标签——高质量正样本，正密度 7.9%）。
流程：FD 窗 → 62 特征（fs=10 同构）→ 与目标域窗模型训练数据合并（特征级白化对齐）
→ HGB → val 窗概率 → 密度 → 复核（目标域）→ 评估。

用法：D:/Anaconda3/envs/bme/python.exe scripts/fd_slide_exp.py --fold 2
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
import slide_verifier as sv
import official_iou_eval as oe
import slide_features as sf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

OUT_FD_FEAT = config.CACHE_DIR / "slide" / "fd_feat.npz"


def build_fd_features():
    if OUT_FD_FEAT.exists():
        d = np.load(OUT_FD_FEAT, allow_pickle=True)
        return d["feat"], d["label"]
    z = np.load(config.CACHE_DIR / "fd_windows" / "fd_pretrain7.npz", allow_pickle=True)
    imu = z["imu"].astype(np.float32)          # (15440, 2400, 7) float32 m/s²
    lab = z["label"]
    fs = 10.0                                   # 缓存为 10Hz 网格
    feats = []
    for i in range(len(imu)):
        seg = imu[i, :, :3].T                   # (3, 2400) acc 三轴
        f = sf.extract_62(seg, fs)
        if len(f) == 62 and not np.isnan(f).all():
            feats.append(f)
        if (i + 1) % 4000 == 0:
            print(f"  FD 特征 {i + 1}/{len(imu)}", flush=True)
    X = np.array(feats, np.float32)
    y = lab[:len(feats)].astype(np.int8)
    np.savez_compressed(OUT_FD_FEAT, feat=X, label=y)
    print(f"FD 特征表: {X.shape}（正 {int(y.sum())}）→ {OUT_FD_FEAT}", flush=True)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--fd-neg", type=int, default=3000, help="FD 负窗采样数")
    args = ap.parse_args()
    k = args.fold

    X_fd, y_fd = build_fd_features()
    rng = np.random.default_rng(20260907)
    fd_pos = np.where(y_fd == 1)[0]
    use_neg = os.environ.get("BME_FD_NEG", "1") == "1"
    fd_sel = np.concatenate([fd_pos, rng.choice(np.where(y_fd == 0)[0],
                             min(args.fd_neg, int((y_fd == 0).sum())), replace=False)]) if use_neg else fd_pos

    # 目标域训练（fold k train 采样窗 + 时刻列）
    tr = np.load(config.CACHE_DIR / "slide" / f"fold{k}_train.npz", allow_pickle=True)
    keep = tr["label"] >= 0
    d0 = np.load(config.CACHE_DIR / "slide" / f"fold{k}_train.npz", allow_pickle=True)
    prior_col = np.zeros((len(d0["wid"]), 1), np.float32)
    for j, w in enumerate([json.loads(x) for x in d0["wid"]]):
        hh = int((w[1] / 3.6e6) % 24)
        prior_col[j, 0] = sv.GLOBAL_PRIOR[hh]
    X_t = np.concatenate([tr["feat"][keep], prior_col[keep]], 1)
    y_t = tr["label"][keep].astype(int)

    # 特征级对齐：目标域特征标准化（减中位除 MAD——62 特征各自）→ FD 同处理
    def whiten(X):
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0) + 1e-6
        return np.nan_to_num((X - med) / mad, nan=0.0)
    X_t_w = whiten(X_t)
    X_fd_w = whiten(X_fd[fd_sel])
    # 合并（FD 补时刻列 0.5 中性——无绝对时刻；负窗取样）
    X_fd_full = np.concatenate([X_fd_w, np.full((len(fd_sel), 1), 0.5, np.float32)], 1)
    X_all = np.concatenate([X_t_w, X_fd_full])
    y_all = np.concatenate([y_t, y_fd[fd_sel]])
    print(f"联合训练：目标 {len(y_t)}（正 {int(y_t.sum())}）+ FD {len(fd_sel)}（正 {len(fd_pos)}）", flush=True)

    imp = SimpleImputer(strategy="median").fit(X_all)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, max_depth=4,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
        random_state=20260901 + k)
    w_all = np.ones(len(y_all), np.float32)
    w_all[len(y_t):] = float(os.environ.get("BME_FD_W", "0.5"))   # FD 窗降权（域外样本）
    clf.fit(imp.transform(X_all), y_all, sample_weight=w_all)

    # val 打分（白化用目标域参数——train 的中位/MAD）
    d = np.load(config.CACHE_DIR / "slide" / f"fold{k}_val.npz", allow_pickle=True)
    wids = [json.loads(w) for w in d["wid"]]
    pc = np.zeros((len(wids), 1), np.float32)
    for j, w in enumerate(wids):
        hh = int((w[1] / 3.6e6) % 24)
        pc[j, 0] = sv.GLOBAL_PRIOR[hh]
    med_t = np.nanmedian(X_t, axis=0); mad_t = np.nanmedian(np.abs(X_t - med_t), axis=0) + 1e-6
    X_v = np.concatenate([d["feat"], pc], 1)
    X_v_w = np.nan_to_num((X_v - med_t) / mad_t, nan=0.0)
    prob = clf.predict_proba(imp.transform(X_v_w))[:, 1]
    sw = defaultdict(list)
    for wid, p in zip(wids, prob):
        sw[wid[0]].append((wid[1], wid[2], float(p)))

    # 复核（目标域 v4.2 语义：no_meal 150 + bag？此处简化：单折复核——用于对比窗模型差异）
    # 候选层对比为主（窗模型改进的直接体现）
    true_va = sv.eligible_meals(set(sw.keys()), sw)
    cands = sv.density_candidates(sw, 0.28838)
    m = oe.official_metrics([(c[0], (c[1], c[2])) for c in cands], true_va)
    print(f"fold{k} FD联合 候选层: 候选 {len(cands)} F1={m['f1']:.3f} sens={m['sensitivity']:.3f} "
          f"ppv={m['ppv']:.3f} ({m['n_tp']}/{m['n_true']})", flush=True)
    out = {"fold": k, "fd_neg": args.fd_neg, "candidate_layer": {kk: m[kk] for kk in
           ("f1", "sensitivity", "ppv", "n_tp", "n_true", "n_pred")}}
    (config.OUTPUT_DIR / f"fd_slide_fold{k}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
