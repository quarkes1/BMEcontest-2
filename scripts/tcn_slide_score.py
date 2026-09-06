# -*- coding: utf-8 -*-
"""TCN 深度模型对滑窗窗打分（混合方案：深度分作复核特征/密度概率）。

对 cache/slide/fold{k}_{split}.npz 的每窗从 sessions npz 提取 240s@10Hz 6ch 段
（与训练缓存同构：100ms 网格 interp）→ FD z-score → 本折模型 GPU 推理。
用法：D:/Anaconda3/envs/bme/python.exe scripts/tcn_slide_score.py --fold 0 --split val
产物：cache/slide/fold{k}_{split}_tcn.npz（score 与 wid 对齐）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.models.ranker import MMRanker


def extract_seg(acc, gyro, t_v, ws, we):
    """240s@10Hz 网格 interp（同 build_candidate_windows._extract 的 imu 部分）。"""
    grid = ws + np.arange(2400) * 100
    imu = np.zeros((2400, 6), np.float32)
    for i, ch in enumerate(np.concatenate([acc, gyro])):
        imu[:, i] = np.interp(grid, t_v, ch.astype(np.float64))
    return imu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=("val", "meal_train", "no_meal_train", "train"), default="val")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load("checkpoints/fd_pretrained_s1.pt", map_location="cpu", weights_only=False)
    mu = np.asarray(ck["norm_mean"], np.float32)[:6]
    sd = np.asarray(ck["norm_std"], np.float32)[:6] + 1e-6
    import os as _os
    bag = _os.environ.get("BME_TCN_BAG", "0") == "1"
    if bag:   # 5 模型 bagging（跨折模型 domain shift 在滑窗上重估——bag 降方差）
        model_list = []
        for kk in range(5):
            m = MMRanker(n_imu=6, use_ppg=False).to(dev).eval()
            m.load_state_dict(torch.load(f"models/mm_ranker_fold{kk}.pt",
                                         map_location="cpu", weights_only=False))
            model_list.append(m)
    else:
        model_list = [MMRanker(n_imu=6, use_ppg=False).to(dev).eval()]
        model_list[0].load_state_dict(torch.load(f"models/mm_ranker_fold{args.fold}.pt",
                                                 map_location="cpu", weights_only=False))
    z66 = torch.zeros(1, 48, 66, device=dev)
    z2 = torch.zeros(1, 48, 2, device=dev)

    d = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_{args.split}.npz", allow_pickle=True)
    wids = [json.loads(w) for w in d["wid"]]
    # 会话级 gate_prob（与训练一致的 LGBM 门控）
    import rank_events_v2 as v2
    _, gate_prob, _, _, _, _ = v2.prepare_fold(args.fold)
    # 时刻先验（全局 24h 直方图，predict.py 内置同款）
    GLOBAL_PRIOR = np.array(
        [0.174, 0.278, 0.546, 0.92, 0.889, 0.496, 0.187, 0.141, 0.408, 0.863,
         1.0, 0.681, 0.368, 0.216, 0.127, 0.073, 0.037, 0.012, 0.002, 0.0,
         0.0, 0.0, 0.0, 0.0], np.float32)
    # 按会话分组读 raw（一次读入复用）
    from collections import defaultdict
    by_sid = defaultdict(list)
    for i, w in enumerate(wids):
        by_sid[w[0]].append((i, w[1], w[2]))
    scores = np.zeros(len(wids), np.float32)
    BS = 1440
    import queue as _q
    import threading
    q = _q.Queue(maxsize=4)          # 生产者-消费者双缓冲：CPU 提取与 GPU 推理重叠

    def extract_batch(acc, gyro, t_v, wss):
        """会话级批量 interp：所有窗 grid 拼接一次 np.interp（~7× 快于逐窗）。"""
        n = len(wss)
        grid0 = np.arange(2400) * 100
        big = (wss[:, None] + grid0[None, :]).ravel()          # (n*2400,)
        imu = np.empty((n, 2400, 6), np.float32)
        for ch_i in range(3):
            imu[:, :, ch_i] = np.interp(big, t_v, acc[ch_i]).reshape(n, 2400)
            imu[:, :, ch_i + 3] = np.interp(big, t_v, gyro[ch_i]).reshape(n, 2400)
        return imu

    def producer():
        try:
            for sid, items in by_sid.items():
                p = config.CACHE_DIR / "sessions" / f"{sid}.npz"
                if not p.exists():
                    q.put(("nan", [i for i, _, _ in items], None, None))
                    continue
                with np.load(p) as z:
                    acc = z["acc"][:, z["imu_valid"]].astype(np.float32)
                    gyro = z["gyro"][:, z["imu_valid"]].astype(np.float32)
                    t_v = z["t_acc"][z["imu_valid"]].astype(np.float64)
                gp = float(gate_prob.get(sid, 0.5))
                # 按会话批量提取 → 分片入队
                idxs = [it[0] for it in items]
                wss = np.array([it[1] for it in items], np.int64)
                segs = extract_batch(acc, gyro, t_v, wss)
                segs = (segs - mu) / sd
                hh = (wss / 3.6e6).astype(np.int64) % 24
                metas = np.stack([np.full(len(wss), np.log1p(240.0)),
                                  GLOBAL_PRIOR[hh], np.full(len(wss), gp)], 1).astype(np.float32)
                for b0 in range(0, len(wss), BS):
                    b1 = min(b0 + BS, len(wss))
                    q.put(("seg", idxs[b0:b1],
                           [torch.from_numpy(segs[j]) for j in range(b0, b1)],
                           [torch.from_numpy(metas[j]) for j in range(b0, b1)]))
        finally:
            q.put((None, None, None, None))   # 结束哨兵

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    n_done = 0
    while True:
        kind, idxs, buf, mbs = q.get()
        if kind is None:
            break
        if kind == "nan":
            for i2 in idxs:
                scores[i2] = np.nan
            continue
        x = torch.stack(buf).to(dev)
        mb = torch.stack(mbs).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            lg = torch.stack([mm(x, z66, z2, mb) for mm in model_list]).cpu()
        p = torch.sigmoid(lg).mean(0).numpy().reshape(-1)   # bag：5 模型 sigmoid 均值
        for i2, pp in zip(idxs, p):
            scores[i2] = pp
        n_done += len(idxs)
        if n_done % 20000 == 0:
            print(f"  {n_done}/{len(wids)} 窗", flush=True)
    th.join()
    out = config.CACHE_DIR / "slide" / f"fold{args.fold}_{args.split}_tcn.npz"
    np.savez(out, score=scores)
    print(f"→ {out}: {len(scores)} 窗（nan {int(np.isnan(scores).sum())}）", flush=True)


if __name__ == "__main__":
    main()
