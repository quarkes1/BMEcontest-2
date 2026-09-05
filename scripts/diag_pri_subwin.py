# -*- coding: utf-8 -*-
"""验证：先验网格（15min/15min）内 240s 子窗的深度分数是否可判别"餐内"。

动机：act 提案只覆盖 42-47% val 餐（活动连通域太短 → IoU<0.25），pri 通道 LGBM
FP 泛滥。若 pri 网格内 4 个错位子窗（中心偏移 -360/-120/+120/+360s）经 MM-Ranker
打分后，餐内子窗分数显著高于非餐子窗 → 深度模型可直接撑起 pri 通道（v3 解码）。

运行：D:/Anaconda3/envs/bme/python.exe scripts/diag_pri_subwin.py --fold 0 --n 20
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.data import manifests, splits, loader
import rank_events as re

os.environ["BME_NO_GYRO"] = "1"          # 6ch（与训练缓存一致）
SUB_OFF = (-360, -120, 120, 360)         # 子窗中心相对 pri 网格中心的偏移（s）


def extract_window(s, c_s, c_e):
    """6ch imu 240s@10Hz（同 build_candidate_windows._extract 的简化路径）。"""
    c_mid = (c_s + c_e) // 2
    ws = c_mid - 120_000
    we = c_mid + 120_000
    grid = ws + np.arange(2400) * 100          # 10Hz
    t_v = s.t_acc[s.imu_valid].astype(np.float64)
    imu = np.zeros((2400, 6), np.float32)
    for i, ch in enumerate(np.concatenate([s.acc, s.gyro])[:, s.imu_valid]):
        imu[:, i] = np.interp(grid, t_v, ch.astype(np.float64))
    cov = ((grid >= t_v[0]) & (grid <= t_v[-1])).mean()
    if cov < 0.05:
        return None
    return imu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=20, help="分析会话数上限")
    args = ap.parse_args()

    import torch
    from src.models.ranker import MMRanker
    dev = torch.device("cpu")
    model = MMRanker(n_imu=6, use_ppg=False).to(dev).eval()
    ck = torch.load("checkpoints/fd_pretrained_s1.pt", map_location="cpu", weights_only=False)
    sd = ck["encoder"]
    model.load_state_dict(sd, strict=False)
    mu = np.asarray(ck["norm_mean"], np.float32)[:6]
    sdv = np.asarray(ck["norm_std"], np.float32)[:6] + 1e-6

    folds = splits.load_folds()
    f = folds[args.fold]
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    starts = {r["session_id"]: int(r["timeStamp.startTime"]) for _, r in idx.iterrows()}
    sid_meals = {}
    for _, r in idx.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms

    # 模型打分（无 FD 微调——验证用 fold0 权重更贴近实际；先试预训练权重感知）
    w = torch.load("models/mm_ranker_fold0.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(w, strict=False)

    z66 = torch.zeros(1, 48, 66)
    z2 = torch.zeros(1, 48, 2)
    z3 = torch.zeros(1, 3)

    def score_win(imu):
        x = torch.from_numpy(imu)[None]
        with torch.no_grad():
            lg = model(x, z66, z2, z3)
        return float(torch.sigmoid(lg).item())

    pos_s, neg_s, sids_done = [], [], 0
    for sid in f["val_sessions"]:
        meals = sid_meals.get(sid, [])
        if not meals or sids_done >= args.n:
            continue
        p = config.CACHE_DIR / "validate_baselines" / f"{sid}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        t0 = d["t0"].astype(np.int64)
        sess_s, sess_e = int(t0.min()), int(t0.max())
        s = loader.load_session(sid)
        # 会话内 pri 网格（900s 步长/900s 窗半宽 450）
        pri = re.prior_candidates(sess_s, sess_e, 900, 450)
        if not pri:
            continue
        sids_done += 1
        # 每 pri 窗 → 4 子窗
        scored = []
        for (ps_, pe_, _) in pri:
            pc = (ps_ + pe_) // 2
            for off in SUB_OFF:
                c = pc + off * 1000
                imu = extract_window(s, c - 120_000, c + 120_000)
                if imu is None:
                    continue
                imu = (imu - mu) / sdv
                scored.append((c, score_win(imu)))
        for m in meals:
            gs, ge = m["before"], m["after"]
            gc = (gs + ge) // 2
            for c, sc in scored:
                ov = max(0, min(c + 120_000, ge) - max(c - 120_000, gs))
                in_meal = ov >= 0.4 * 240_000 and abs(c - gc) <= 240_000
                (pos_s if in_meal else neg_s).append(sc)
        if sids_done >= args.n:
            break
    pos_s, neg_s = np.array(pos_s), np.array(neg_s)
    print(f"会话 {sids_done} | 餐内子窗 {len(pos_s)} 分 median={np.median(pos_s):.3f} "
          f"p25={np.percentile(pos_s, 25):.3f} p75={np.percentile(pos_s, 75):.3f}")
    print(f"非餐子窗 {len(neg_s)} 分 median={np.median(neg_s):.3f} "
          f"p95={np.percentile(neg_s, 95):.3f} p99={np.percentile(neg_s, 99):.3f} top={neg_s.max():.3f}")
    # AUC
    if len(pos_s) and len(neg_s):
        y = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
        sc = np.concatenate([pos_s, neg_s])
        order = np.argsort(sc)[::-1]
        ranks = np.empty(len(sc)); ranks[order] = np.arange(len(sc))
        n_pos = len(pos_s)
        auc = 1 - (ranks[:n_pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * len(neg_s))
        print(f"AUC(餐内 vs 非餐子窗) = {auc:.3f}")
    # top-k 命中
    k = max(1, len(pos_s))
    topk = np.argsort(np.concatenate([pos_s, neg_s]))[::-1][:k]
    hit = (topk < len(pos_s)).sum()
    print(f"top{len(pos_s)} 内命中餐内子窗 {hit}/{len(pos_s)}")


if __name__ == "__main__":
    main()
