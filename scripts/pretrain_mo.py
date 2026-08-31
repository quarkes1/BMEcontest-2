# -*- coding: utf-8 -*-
"""MO bite 预训练（S1）：KU Leuven Eating-Speed MO 数据集 → bite 检测 TCN。

迁移策略：bite 分类器的特征提取器（imu_in Conv1d(6,64,7)+BN + 6 层膨胀 TCN 残差栈）
与 MM-Ranker 的 IMU 分支**逐层同构**（src/models/ranker.py）——预训练后在
MM-Ranker 微调时按 state_dict 前缀拷贝，bite 头丢弃。学习目标：真实 bite 级
进食动作的 IMU 判别特征（本数据只有餐次标签，无 bite 标注）。

数据：refs/eating_speed/MO/{X_L,X_R,Y_L,Y_R}.pkl——46 人 × 双腕 (N,6) @100Hz，
acc(g)3 + gyro(°/s)3；Y 逐样本 bite 0/1。subj0 正样本率 46%（异常）剔除。
预处理：均值池化降采样 100→10Hz；30s 窗（300 样本）步长 15s；窗内 bite 占比
>3% 记正（bite 瞬态，30s 内 1-2 次 = 5-10%）。

训练：subject-level 8:2 划分（不泄漏），Focal Loss + 正样本过采样，早停 valAUC。
产物：models/mo_bite.pt（encoder state_dict = imu_in+tcn，MM-Ranker 可直接加载）

运行：source activate bme && python scripts/pretrain_mo.py [--epochs 30]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.models.ranker import TCNBlock, focal_loss

SEED = 42
SRC = 100.0
DST = 10.0
WIN_S, ST_S = 30.0, 15.0
BITE_POS = 0.03          # 窗内 bite 占比 >3% → 正
DS_STEP = int(SRC / DST)  # 10


class BiteTCN(torch.nn.Module):
    """bite 检测 TCN：特征提取器与 MM-Ranker IMU 分支逐层同构（可整体迁移）。"""

    def __init__(self, d_model=64, n_layers=6, dropout=0.3):
        super().__init__()
        self.imu_in = torch.nn.Sequential(
            torch.nn.Conv1d(6, d_model, 7, padding=3), torch.nn.BatchNorm1d(d_model),
            torch.nn.ReLU())
        self.tcn = torch.nn.ModuleList(
            [TCNBlock(d_model, dilation=2 ** i, dropout=dropout) for i in range(n_layers)])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * d_model, 128), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 1))

    def forward(self, x):
        """x: (B, T, 6) → (B,)。"""
        h = x.transpose(1, 2).float()                 # (B, 6, T)
        h = self.imu_in(h)
        for blk in self.tcn:
            h = blk(h)
        return self.head(torch.cat([h.mean(2), h.max(2).values], dim=1)).squeeze(1)

    def encoder_state(self):
        """特征提取器 state_dict（imu_in+tcn，供 MM-Ranker 加载）。"""
        sd = {}
        for k, v in self.state_dict().items():
            if k.startswith(("imu_in.", "tcn.")):
                sd[k] = v
        return sd


def load_mo():
    base = config.ROOT_DIR / "refs" / "eating_speed" / "MO"
    out_x, out_y = [], []
    for side in ("L", "R"):
        with open(base / f"X_{side}.pkl", "rb") as f:
            X = pickle.load(f)
        with open(base / f"Y_{side}.pkl", "rb") as f:
            Y = pickle.load(f)
        for i in range(len(X)):
            if i == 0:      # subj0 异常（正样本率 46%）
                continue
            x, y = X[i], Y[i]
            n = len(x) // DS_STEP * DS_STEP
            x = x[:n].reshape(n // DS_STEP, DS_STEP, 6).mean(1)   # (N/10, 6)
            y = y[:n].reshape(n // DS_STEP, DS_STEP).mean(1)      # 降采样标签（占比）
            out_x.append(x.astype(np.float32))
            out_y.append((y > 0).astype(np.float32))
    return out_x, out_y


def make_windows(xs, ys):
    win, st = int(WIN_S * DST), int(ST_S * DST)
    Wx, Wy = [], []
    for x, y in zip(xs, ys):
        n = x.shape[0]
        for b0 in range(0, n - win + 1, st):
            Wx.append(x[b0:b0 + win])
            Wy.append(1.0 if y[b0:b0 + win].mean() > BITE_POS else 0.0)
    return np.stack(Wx), np.array(Wy, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    xs, ys = load_mo()
    n_subj = len(xs)
    n_va = max(1, int(n_subj * args.val_split))
    print(f"MO: {n_subj} 腕样本（剔除 subj0），val 受试者 {n_va}", flush=True)
    tr_x, tr_y = [], []
    for i in range(n_subj - n_va):
        wx, wy = make_windows([xs[i]], [ys[i]])
        tr_x.append(wx); tr_y.append(wy)
    va_x, va_y = [], []
    for i in range(n_subj - n_va, n_subj):
        wx, wy = make_windows([xs[i]], [ys[i]])
        va_x.append(wx); va_y.append(wy)
    Xtr, Ytr = np.concatenate(tr_x), np.concatenate(tr_y)
    Xva, Yva = np.concatenate(va_x), np.concatenate(va_y)
    print(f"  train {len(Xtr)} 窗（正 {Ytr.mean()*100:.1f}%）| val {len(Xva)}（正 {Yva.mean()*100:.1f}%）", flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device(args.device)
    model = BiteTCN().to(dev)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3
    print(f"  BiteTCN 参数 {n_param:.0f}K（特征提取器与 MM-Ranker IMU 分支同构）", flush=True)

    pos_idx = np.where(Ytr == 1)[0]
    neg_idx = np.where(Ytr == 0)[0]
    n_pos = min(len(pos_idx), len(neg_idx) // 2)     # 正:负 ≈ 1:2
    sel = np.concatenate([pos_idx[:n_pos], neg_idx[np.random.choice(len(neg_idx), n_pos, False)]])
    Xtr, Ytr = Xtr[sel], Ytr[sel]
    print(f"  过采样平衡后 train {len(Xtr)}（正 {Ytr.mean()*100:.0f}%）", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    xt = torch.from_numpy(Xtr); yt = torch.from_numpy(Ytr)
    xv = torch.from_numpy(Xva); yv = torch.from_numpy(Yva)

    def evaluate(ds_x, ds_y):
        model.eval()
        with torch.no_grad():
            lg = torch.cat([model(b.to(dev)) for b in
                            DataLoader(ds_x, batch_size=256, shuffle=False)]).cpu()
        p = torch.sigmoid(lg)
        order = p.argsort(descending=True)
        ranks = torch.empty(len(p), dtype=torch.long); ranks[order] = torch.arange(len(p))
        n_p = ds_y.sum(); n_n = len(ds_y) - n_p
        if n_p == 0 or n_n == 0:
            return float("nan")
        s = float((ds_y * ranks).sum())
        return float(1.0 - (s - n_p * (n_p - 1) / 2) / (n_p * n_n))

    BATCH = 128
    n_tr = len(Xtr)
    best_auc, best_sd, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(n_tr)
        tot = 0.0; nb = 0
        for b0 in range(0, n_tr, BATCH):
            idx = perm[b0:b0 + BATCH]
            lg = model(xt[idx].to(dev))
            loss = focal_loss(lg, yt[idx].to(dev), gamma=2.0, alpha=0.35)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        auc = evaluate(xv, yv)
        tag = ""
        if auc > best_auc:
            best_auc, best_sd, patience = auc, {kk: vv.clone() for kk, vv in model.state_dict().items()}, 0
            tag = " ★"
        else:
            patience += 1
        print(f"  ep{ep:02d} loss {tot/nb:.4f} valAUC {auc:.3f}{tag} [{time.time()-t0:.0f}s]", flush=True)
        if patience >= 8:
            print(f"  早停 @ep{ep}", flush=True)
            break

    model.load_state_dict(best_sd)
    torch.save(model.encoder_state(), config.MODEL_DIR / "mo_bite.pt")
    print(f"→ models/mo_bite.pt（encoder={sum(v.numel() for v in best_sd.values())/1e3:.0f}K 参数，"
          f"valAUC {best_auc:.3f}）——train_ranker --init-from 加载", flush=True)


if __name__ == "__main__":
    main()
