# -*- coding: utf-8 -*-
"""硬件基准：扫 batch/AMP/TF32/模型 组合，输出 steps/s 与单 epoch 预估。
用法：conda activate bme && python scripts/bench_hw.py [--batches 128,256,512]
      [--amp 0,1] [--tf32 0,1] [--model small,large] [--steps 40]
注意：会占用 ~10GB 内存（fold 0 数据集入内存）与 GPU；请勿与训练链并行跑。"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.models.l3a_cnn import L3aCNN, L3aCNNLarge
from scripts.train_l3a import RawDataset

FOLD = 0
WARMUP = 5

def build_dataset(data_mode="gpu"):
    out_dir = config.CACHE_DIR / "l3a_raw" / f"fold{FOLD}"
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    ds = RawDataset(out_dir, stats, mirror=False, stretch=False, jitter=False, ch_drop=0,
                    on_gpu=(data_mode == "gpu"))
    ds.reshuffle(config.RANDOM_SEED)
    return ds

def bench(batch, amp, tf32, model_name, steps, data_mode="gpu"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    ds = build_dataset(data_mode)
    from src.train.prefetch import PrefetchLoader
    dl = PrefetchLoader(ds, batch_size=batch, drop_last=True)
    model = (L3aCNNLarge(5) if model_name == "large" else L3aCNN(5)).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp))
    it = iter(dl)

    def step():
        xb, yb, twb = next(it)
        xb, yb = xb.cuda(), yb.cuda()
        if amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logit, tw_logit = model(xb)
                loss = bce(logit, yb * 0.9 + 0.05)
        else:
            logit, tw_logit = model(xb)
            loss = bce(logit, yb * 0.9 + 0.05)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    sps = steps / (time.time() - t0)
    n_steps_per_epoch = len(dl)
    return sps, n_steps_per_epoch / sps / 60.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="128,256,512")
    ap.add_argument("--amp", default="0,1")
    ap.add_argument("--tf32", default="0,1")
    ap.add_argument("--model", default="small,large")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--data", default="gpu", choices=["gpu", "cpu"], help="数据集驻留位置")
    args = ap.parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)}  显存: {torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GB")
    rows = []
    for model_name in args.model.split(","):
        for amp in [int(x) for x in args.amp.split(",")]:
            for tf32 in [int(x) for x in args.tf32.split(",")]:
                for batch in [int(x) for x in args.batches.split(",")]:
                    try:
                        sps, ep_min = bench(batch, amp, tf32, model_name, args.steps, args.data)
                        rows.append((sps, f"{model_name} batch={batch} amp={amp} tf32={tf32} data={args.data}"))
                        print(f"  {rows[-1][1]}: {sps:.1f} steps/s, epoch ≈ {ep_min:.1f} min", flush=True)
                    except Exception as e:
                        print(f"  {model_name} batch={batch} amp={amp} tf32={tf32}: FAIL {type(e).__name__}: {e}", flush=True)
                    finally:
                        torch.cuda.empty_cache()          # 释放上一轮的数据集显存
    rows.sort(key=lambda r: -r[0])
    print("\n===== 最快配置 =====")
    for sps, name in rows[:5]:
        print(f"  {name}: {sps:.1f} steps/s")

if __name__ == "__main__":
    main()
