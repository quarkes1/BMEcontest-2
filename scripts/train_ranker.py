# -*- coding: utf-8 -*-
"""MM-Ranker 训练：fold{k} 候选窗口 → Focal Loss + 硬负样本挖掘 → 模型 + val 分数。

训练（方向 2/4）：
- Focal Loss（γ=2, α=0.25）替代 BCE，处理 ~1:12 候选类不平衡
- 硬负样本挖掘：每 epoch 末对训练集全量前向，取"非进食但被高置信度判正"的负样本
  （活动池负样本天然是刷牙/摸脸/托腮/游戏类手部活跃非餐）→ 下 epoch 提权 ×5
- 早停：val 候选 AUC（patience 12），保存最优 checkpoint
- 强正则：dropout 0.3 / weight decay 1e-4 / BN

运行：source activate bme && python scripts/train_ranker.py --fold 0
产物：models/mm_ranker_fold{k}.pt + outputs/mm_ranker_fold{k}_val.npz（val 候选分数，解码用）
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
from src.models.ranker import MMRanker, focal_loss, count_params

EPOCHS = 60
BATCH = 64
LR = 1e-3
WD = 1e-4
PATIENCE = 12
HARD_W = 5.0          # 硬负样本提权倍数
HARD_K = 3            # top-k = 3× 正样本数


class CandDS(Dataset):
    def __init__(self, imu, ppg, ma, y, weights=None):
        self.imu = torch.from_numpy(imu)
        self.ppg = torch.from_numpy(ppg)
        self.ma = torch.from_numpy(ma)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.weights = torch.ones(len(y), dtype=torch.float32) if weights is None \
            else torch.from_numpy(weights.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.imu[i], self.ppg[i], self.ma[i], self.y[i], self.weights[i]


def load_fold(k):
    """读 fold{k} 候选窗口缓存 → train/val 数组。"""
    return None  # 占位（未用）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    k = args.fold

    # ---- 加载缓存（按 fold 划分切 train/val） ----
    from src.data import splits
    folds = splits.load_folds()
    tr_set, va_set = set(folds[k]["train_sessions"]), set(folds[k]["val_sessions"])
    d = config.CACHE_DIR / "cand_windows" / f"fold{k}"
    X, Y, META = [], [], []
    for p in sorted(d.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        n = len(z["meta"])
        for j in range(n):
            imu = z[f"c{j}"].astype(np.float32)
            X.append((imu, z["ppg"][j], z["ma"][j]))
        m = [json.loads(x.decode()) for x in z["meta"]]
        META.extend(m)
        Y.extend(int(mm["label"]) for mm in m)
    imu = np.stack([x[0] for x in X]); ppg = np.stack([x[1] for x in X]); ma = np.stack([x[2] for x in X])
    y = np.array(Y, np.int8)
    split_idx = np.array([0 if m["sid"] in tr_set else 1 for m in META])
    print(f"fold{k}: {len(y)} 候选（正 {y.sum()}，{y.mean()*100:.1f}%）", flush=True)
    tr, va = split_idx == 0, split_idx == 1
    print(f"  train {(split_idx==0).sum()} 候选（正 {y[tr].sum()}）| val {(split_idx==1).sum()}（正 {y[va].sum()}）", flush=True)

    dev = torch.device(args.device)
    model = MMRanker().to(dev)
    print(f"  MM-Ranker 参数 {count_params(model)/1e3:.0f}K", flush=True)

    train_ds = CandDS(imu[tr], ppg[tr], ma[tr], y[tr])
    val_ds = CandDS(imu[va], ppg[va], ma[va], y[va])
    tr_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    all_tr = CandDS(imu[tr], ppg[tr], ma[tr], y[tr])   # 硬负样本挖掘全量

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def evaluate(ds):
        model.eval()
        logs, ys = [], []
        with torch.no_grad():
            for ii, pp, mm, yy, _ in DataLoader(ds, batch_size=256, shuffle=False):
                logs.append(model(ii.to(dev), pp.to(dev), mm.to(dev)).cpu())
                ys.append(yy)
        l = torch.cat(logs); yt = torch.cat(ys)
        p = torch.sigmoid(l)
        # AUC（0 基 rank：Σ正样本rank - n_pos(n_pos-1)/2）
        order = p.argsort(descending=True)
        ranks = torch.empty_like(order); ranks[order] = torch.arange(len(p))
        n_pos = yt.sum(); n_neg = len(yt) - n_pos
        if n_pos == 0 or n_neg == 0:
            auc = float("nan")
        else:
            auc = ((yt * ranks).sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
        return float(auc), p.numpy(), l.numpy()

    best_auc, best_state, patience, hard_w = -1.0, None, 0, None
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0; nb = 0
        for ii, pp, mm, yy, ww in tr_loader:
            lg = model(ii.to(dev), pp.to(dev), mm.to(dev))
            loss = focal_loss(lg, yy.to(dev), weights=ww.to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        auc, p_va, _ = evaluate(val_ds)
        tag = ""
        if auc > best_auc:
            best_auc, best_state, patience = auc, {kk: vv.clone() for kk, vv in model.state_dict().items()}, 0
            tag = " ★"
        else:
            patience += 1
        print(f"  ep{ep:02d} loss {tot/nb:.4f} valAUC {auc:.3f}{tag} [{time.time()-t0:.0f}s]", flush=True)
        if patience >= PATIENCE:
            print(f"  早停 @ep{ep}（best AUC {best_auc:.3f}）", flush=True)
            break
        # ---- 硬负样本挖掘：train 全量前向 → 误判负样本 top-k 提权 ----
        model.eval()
        with torch.no_grad():
            ltr = []
            for ii, pp, mm, _, _ in DataLoader(all_tr, batch_size=256, shuffle=False):
                ltr.append(model(ii.to(dev), pp.to(dev), mm.to(dev)).cpu())
            ltr = torch.cat(ltr)
        ptr = torch.sigmoid(ltr).numpy()
        fp_mask = (y[tr] == 0) & (ptr > 0.5)
        k = min(int(HARD_K * y[tr].sum()), int(fp_mask.sum()))
        if k > 0:
            idx = np.where(fp_mask)[0][np.argsort(ptr[fp_mask])[-k:]]
            hard_w = np.ones(len(all_tr), np.float32)
            hard_w[idx] = HARD_W
            train_ds.weights.copy_(torch.from_numpy(hard_w))
            print(f"    硬负样本: {k} 个（top sigmoid {ptr[idx].max():.3f}）", flush=True)
        else:
            hard_w = None

    model.load_state_dict(best_state)
    auc, p_va, _ = evaluate(val_ds)
    print(f"★ 最优 valAUC {best_auc:.3f}（重载后 {auc:.3f}）", flush=True)

    # ---- 保存：模型 + val 候选分数（解码用） ----
    torch.save(best_state, config.MODEL_DIR / f"mm_ranker_fold{k}.pt")
    order = [i for i, m in enumerate(META) if m["sid"] in va_set]
    np.savez(config.OUTPUT_DIR / f"mm_ranker_fold{k}_val.npz",
             score=p_va, label=np.array([y[i] for i in order], np.int8),
             meta=np.array([json.dumps(META[i]).encode() for i in order]))
    print(f"→ models/mm_ranker_fold{k}.pt + outputs/mm_ranker_fold{k}_val.npz（{len(order)} val 候选）", flush=True)


if __name__ == "__main__":
    main()
