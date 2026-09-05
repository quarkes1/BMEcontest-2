# -*- coding: utf-8 -*-
"""5 折模型 bagging 全 val 评分（与 dist predict 部署口径一致）。

当前评估协议只用本折模型评本折 val（rank_events_v2.load_dl_scores 读
mm_ranker_fold{k}_val.npz）→ 低估 dist（5 模型平均）的真实水平。
本脚本对每折 val 候选窗用全部 5 折模型平均评分（跨折模型未见过该 val 会话，
无泄漏），输出 bag_fold{k}.npz（score/meta 与单折格式同构）。

运行：D:/Anaconda3/envs/bme/python.exe scripts/bag_eval.py --k 0,1,2,3,4
产物：outputs/bag_fold{k}.npz（score = 5 模型 sigmoid 均值）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.data import splits
from src.models.ranker import MMRanker

CKPT = "checkpoints/fd_pretrained_s1.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="0,1,2,3,4")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mu = np.asarray(ck["norm_mean"], np.float32)[:6]
    sd = np.asarray(ck["norm_std"], np.float32)[:6] + 1e-6

    # 加载 5 折模型
    models = []
    for k in range(5):
        m = MMRanker(n_imu=6, use_ppg=False).to(dev).eval()
        m.load_state_dict(torch.load(f"models/mm_ranker_fold{k}.pt",
                                     map_location="cpu", weights_only=False))
        models.append(m)
    print(f"5 折模型加载完成（{dev}）", flush=True)

    z66 = torch.zeros(1, 48, 66, device=dev)
    z2 = torch.zeros(1, 48, 2, device=dev)
    z3 = torch.zeros(1, 3, device=dev)

    folds = splits.load_folds()
    BS = 256
    for ks in [int(x) for x in args.k.split(",")]:
        va_set = set(folds[ks]["val_sessions"])
        d = config.CACHE_DIR / "cand_windows" / f"fold{ks}"
        scores_all, metas = [], []
        buf, buf_meta = [], []

        def flush():
            if not buf:
                return
            B = len(buf)
            x = torch.stack(buf).to(dev)
            meta_b = torch.stack(buf_meta).to(dev)   # 与训练协议一致的 meta（dur/prior/gate）
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                lgs = torch.stack([mm(x, z66, z2, meta_b) for mm in models]).cpu()
            scores_all.extend(torch.sigmoid(lgs).mean(0).tolist())
            buf.clear(); buf_meta.clear()

        for p in sorted(d.glob("*.npz")):
            sid = p.stem
            if sid not in va_set:
                continue
            with np.load(p, allow_pickle=True) as z:
                mjs = [json.loads(m.decode()) for m in z["meta"]]
                for j, mj in enumerate(mjs):
                    w = z[f"c{j}"].astype(np.float32)
                    w = (w - mu) / sd
                    buf.append(torch.from_numpy(w))
                    buf_meta.append(torch.tensor([np.log1p(mj["dur_s"]), mj["prior_h"],
                                                  mj.get("gate_prob", 0.5)], dtype=torch.float32))
                    if len(buf) >= BS:
                        flush()
                metas.extend(z["meta"])
        flush()
        sc = np.array(scores_all, np.float32)
        out = config.OUTPUT_DIR / f"bag_fold{ks}.npz"
        np.savez(out, score=sc, meta=np.array(metas))
        pos = np.where(np.array([json.loads(m.decode())["label"] for m in metas]) == 1)[0]
        print(f"fold{ks}: {len(sc)} val 窗（正 {len(pos)}）| 正窗 median={np.median(sc[pos]):.3f} "
              f"max={sc[pos].max():.3f} → {out}", flush=True)


if __name__ == "__main__":
    main()
