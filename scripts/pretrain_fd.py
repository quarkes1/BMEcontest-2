# -*- coding: utf-8 -*-
"""FD-I/FD-II Episode 级预训练（阶段二 M2，设计文档 v1.1）。

数据：cache/fd_windows/fd_pretrain.npz（prep_fd.py --build 产出，z-score 归一化 240s 窗）
任务：Episode 级候选分类（餐窗 vs 非餐窗，与 MM-Ranker 任务头对齐——非 bite 级，
MO 教训：bite 级预训练不迁移到餐次级）
结构：MMRanker（n_imu=6，无 PPG——FD 无 PPG 通道），Focal + 硬负样本 + 早停
划分：subject-level（FD-I 双腕全部；按 subject 8:2 分 train/val）
产物：checkpoints/fd_pretrained_s1.pt = {encoder: imu_in+tcn state_dict,
        norm_mean, norm_std}——train_ranker --init-from 加载（含归一化统计量）

运行：source activate bme && python scripts/pretrain_fd.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.models.ranker import MMRanker, focal_loss

SEED = 42
EPOCHS = 40
BATCH = 64
LR = 1e-3
WD = 1e-4
PATIENCE = 10


class FDDataset(Dataset):
    def __init__(self, imu, y):
        self.imu = torch.from_numpy(imu)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.imu[i], self.y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    p = config.CACHE_DIR / "fd_windows" / "fd_pretrain.npz"
    z = np.load(p, allow_pickle=True)
    imu = z["imu"].astype(np.float32)
    y = z["label"].astype(np.float32)
    meta = [json.loads(x.decode()) for x in z["meta"]]
    print(f"FD 预训练集: {len(y)} 窗（正 {y.mean()*100:.1f}%）{imu.shape}", flush=True)

    # subject-level 划分（FD-I 的 subject 为键，双腕同 subject）
    subj_ids = sorted({m["which"] + str(m["subj"]) for m in meta})
    rng = np.random.RandomState(SEED)
    rng.shuffle(subj_ids)
    n_va = max(1, int(len(subj_ids) * 0.2))
    va_ids = set(subj_ids[:n_va])
    tr = np.array([m["which"] + str(m["subj"]) not in va_ids for m in meta])
    va = ~tr
    print(f"  train {tr.sum()} 窗（正 {y[tr].mean()*100:.1f}%）| val {va.sum()}（正 {y[va].mean()*100:.1f}%）",
          flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device(args.device)
    model = MMRanker(n_imu=6, use_ppg=False).to(dev)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3
    print(f"  MM-Ranker（无 PPG，n_imu=6）参数 {n_param:.0f}K", flush=True)

    tr_ds = FDDataset(imu[tr], y[tr])
    va_ds = FDDataset(imu[va], y[va])
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # meta 特征全 0（FD 无 prior/gate 上下文——窗口内动作是唯一信号）
    meta0 = torch.zeros(BATCH, 3)

    def evaluate(ds):
        model.eval()
        logs, ys = [], []
        with torch.no_grad():
            for ii, yy in va_loader:
                mm = torch.zeros(len(ii), 3)
                logs.append(model(ii.to(dev), torch.zeros(len(ii), 48, 66).to(dev),
                                  torch.zeros(len(ii), 48, 2).to(dev), mm.to(dev)).cpu())
                ys.append(yy)
        l = torch.cat(logs); yt = torch.cat(ys)
        p = torch.sigmoid(l)
        order = p.argsort(descending=True)
        ranks = torch.empty(len(p), dtype=torch.long); ranks[order] = torch.arange(len(p))
        n_pos = yt.sum(); n_neg = len(yt) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        s = float((yt * ranks).sum())
        return float(1.0 - (s - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))

    best_auc, best_sd, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0; nb = 0
        for ii, yy in tr_loader:
            mm = torch.zeros(len(ii), 3)
            lg = model(ii.to(dev), torch.zeros(len(ii), 48, 66).to(dev),
                       torch.zeros(len(ii), 48, 2).to(dev), mm.to(dev))
            loss = focal_loss(lg, yy.to(dev), gamma=2.0, alpha=0.35)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        auc = evaluate(va_ds)
        tag = ""
        if auc > best_auc:
            best_auc, best_sd, patience = auc, {kk: vv.clone() for kk, vv in model.state_dict().items()}, 0
            tag = " ★"
        else:
            patience += 1
        print(f"  ep{ep:02d} loss {tot/nb:.4f} valAUC {auc:.3f}{tag} [{time.time()-t0:.0f}s]", flush=True)
        if patience >= PATIENCE:
            print(f"  早停 @ep{ep}（best AUC {best_auc:.3f}）", flush=True)
            break

    # 保存 encoder（imu_in+tcn）+ 归一化统计量
    encoder = {}
    for k, v in best_sd.items():
        if k.startswith(("imu_in.", "tcn.")):
            encoder[k] = v
    out = config.CHECKPOINT_DIR
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": encoder,
                "norm_mean": z["norm_mean"], "norm_std": z["norm_std"]},
               out / "fd_pretrained_s1.pt")
    print(f"→ checkpoints/fd_pretrained_s1.pt（encoder {sum(v.numel() for v in encoder.values())/1e3:.0f}K，"
          f"valAUC {best_auc:.3f}）", flush=True)


if __name__ == "__main__":
    main()
