# -*- coding: utf-8 -*-
"""MM-Ranker 训练：fold{k} 候选窗口 → Focal Loss + 硬负样本挖掘 → 模型 + val 分数。

训练目标：
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
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.models.ranker import MMRanker, focal_loss, asymmetric_loss, count_params

EPOCHS = 60
BATCH = 64
LR = 1e-3
WD = 1e-4
PATIENCE = 12
HARD_W = float(os.environ.get("BME_HARD_W", "5.0"))   # 硬负样本提权倍数（×10 验证：过激→保守化）
HARD_K = int(os.environ.get("BME_HARD_K", "5"))       # top-k = K× 正样本数
FOCAL_ALPHA = float(os.environ.get("BME_FOCAL_ALPHA", "0.35"))  # Focal α（0.25→0.35：更重视召回）
SEED = 42             # 固定随机种子（消融可复现）


class CandDS(Dataset):
    """候选窗口数据集。pos_rep>1 时训练集正样本复制（1:45 不平衡 → 过采样平衡）。
    meta 特征：dur_s/log、gate_prob、prior_h（归一化在构造时完成）。"""

    def __init__(self, imu, ppg, ma, meta, y, pos_rep=1):
        orig = np.arange(len(y))
        pos = np.where(y == 1)[0]
        if pos_rep > 1 and len(pos):
            orig = np.concatenate([orig, np.repeat(pos, pos_rep - 1)])
        self.orig = orig
        self.imu = torch.from_numpy(imu[orig])
        self.ppg = torch.from_numpy(ppg[orig])
        self.ma = torch.from_numpy(ma[orig])
        self.meta = torch.from_numpy(meta[orig].astype(np.float32))
        self.y = torch.from_numpy(y[orig].astype(np.float32))
        self.weights = torch.ones(len(orig), dtype=torch.float32)
        self.n_orig = len(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.imu[i], self.ppg[i], self.ma[i], self.meta[i], self.y[i], self.weights[i]


def load_fold(k):
    """读 fold{k} 候选窗口缓存 → train/val 数组。"""
    return None  # 占位（未用）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-ppg", action="store_true", help="PPG 分支消融（纯 IMU）")
    ap.add_argument("--all-train", action="store_true",
                    help="跨折训练扩充：训练集 = 全部会话 - 本折验证会话（受试者级划分无泄漏）")
    ap.add_argument("--init-from", default=None,
                    help="MO bite 预训练权重（models/mo_bite.pt）：imu_in+tcn 特征提取器逐层拷贝（S1 迁移）")
    ap.add_argument("--freeze-first", type=int, default=0,
                    help="冻结 TCN 前 N 层（保护预训练特征；N=3 冻结 imu_in+前 3 层）")
    ap.add_argument("--loss", choices=("focal", "asymmetric"), default="focal",
                    help="focal（α 可调）或 asymmetric（正 γ_pos=1 / 负 γ_neg=3）")
    ap.add_argument("--session-norm", action="store_true",
                    help="会话级 Instance Normalization——按会话统计量归一化 IMU"
                         "通道（抹平受试者动作幅度基线差异）")
    args = ap.parse_args()
    fold_idx = args.fold

    # ---- 加载缓存（按 fold 划分切 train/val） ----
    from src.data import splits
    folds = splits.load_folds()
    va_set = set(folds[fold_idx]["val_sessions"])
    if args.all_train:  # 跨折扩充：本折 val 之外的会话全部作训练（fold 划分按受试者，无泄漏）
        tr_set = None
        all_sid = set()
        for f in folds:
            all_sid |= set(f["train_sessions"]) | set(f["val_sessions"])
        tr_set = all_sid - va_set
    else:
        tr_set = set(folds[fold_idx]["train_sessions"])
    d = config.CACHE_DIR / "cand_windows" / f"fold{fold_idx}"
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
    meta = np.stack([[np.log1p(m["dur_s"]), m["prior_h"], m["gate_prob"]] for m in META]).astype(np.float32)  # gate_prob 全体会话真实分（build 脚本修复后一致）
    split_idx = np.array([0 if m["sid"] in tr_set else 1 for m in META])
    print(f"fold{fold_idx}: {len(y)} 候选（正 {y.sum()}，{y.mean()*100:.1f}%）", flush=True)
    tr, va = split_idx == 0, split_idx == 1
    print(f"  train {(split_idx==0).sum()} 候选（正 {y[tr].sum()}）| val {(split_idx==1).sum()}（正 {y[va].sum()}）", flush=True)

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dev = torch.device(args.device)
    model = MMRanker(d_model=int(os.environ.get("BME_D_MODEL", "64")),
                     n_layers=int(os.environ.get("BME_N_LAYERS", "6")),
                     use_ppg=not args.no_ppg,
                     n_imu=int(imu.shape[2])).to(dev)   # 缓存通道数（6 或 7：gyro 高频通道）
    norm_mean = norm_std = None
    if args.init_from:  # 预训练迁移：支持 FD 格式（{encoder, norm_mean, norm_std}）与平铺格式
        ckpt = torch.load(args.init_from, map_location=dev, weights_only=False)
        sd = ckpt["encoder"] if isinstance(ckpt, dict) and "encoder" in ckpt else ckpt
        loaded = 0
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in sd and sd[k].shape == v.shape:
                    v.copy_(sd[k]); loaded += 1
        if isinstance(ckpt, dict) and "norm_mean" in ckpt:
            norm_mean = np.asarray(ckpt["norm_mean"], np.float32)
            norm_std = np.asarray(ckpt["norm_std"], np.float32)
            print(f"  FD 预训练迁移：{loaded} 层权重 + z-score 归一化"
                  f"（mean {norm_mean.round(2)} / std {norm_std.round(2)}）", flush=True)
        else:
            print(f"  预训练迁移：{loaded} 层权重拷贝自 {args.init_from}", flush=True)
    if norm_mean is not None:  # FD 归一化统计量 → 本项目输入投影到 FD 尺度空间
        n_ch = min(imu.shape[2], len(norm_mean))
        imu[..., :n_ch] = (imu[..., :n_ch] - norm_mean[:n_ch]) / (norm_std[:n_ch] + 1e-6)
        print(f"  输入 z-score 归一化（前 {n_ch} 通道）", flush=True)
    if args.session_norm:  # 会话级 Instance Normalization（按会话分组，会话内通道统计）
        sids = np.array([m["sid"] for m in META])
        for sid in np.unique(sids):
            mask = sids == sid
            mu = imu[mask].mean((0, 1))
            sd = imu[mask].std((0, 1)) + 1e-6
            imu[mask] = (imu[mask] - mu) / sd
        print(f"  会话级 Instance Normalization（{len(np.unique(sids))} 会话，逐会话通道统计）",
              flush=True)
    if args.freeze_first > 0:   # 冻结 TCN 前半（imu_in + 前 N 层），保护预训练特征
        frozen = []
        for name, p in model.named_parameters():
            if name.startswith("imu_in.") or re.match(rf"tcn\.([0-{args.freeze_first - 1}])\.", name):
                p.requires_grad = False
                frozen.append(name)
        print(f"  冻结 {len(frozen)} 个参数组（imu_in + tcn 前 {args.freeze_first} 层）", flush=True)
    print(f"  MM-Ranker 参数 {count_params(model)/1e3:.0f}K（use_ppg={not args.no_ppg}）", flush=True)

    POS_REP = max(2, int((y[tr] == 0).sum() / max(y[tr].sum(), 1) / 4))  # 正:负 ≈ 1:4（1:8→1:4：提升正样本学习强度）
    train_ds = CandDS(imu[tr], ppg[tr], ma[tr], meta[tr], y[tr], pos_rep=POS_REP)
    val_ds = CandDS(imu[va], ppg[va], ma[va], meta[va], y[va])
    print(f"  正样本过采样 ×{POS_REP} → train {len(train_ds)}", flush=True)
    hm_ds = CandDS(imu[tr], ppg[tr], ma[tr], meta[tr], y[tr])   # 硬负样本挖掘全量（原始索引）
    tr_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def evaluate(ds):
        model.eval()
        logs, ys = [], []
        with torch.no_grad():
            for ii, pp, mm, mt, yy, _ in DataLoader(ds, batch_size=256, shuffle=False):
                logs.append(model(ii.to(dev), pp.to(dev), mm.to(dev), mt.to(dev)).cpu())
                ys.append(yy)
        l = torch.cat(logs); yt = torch.cat(ys)
        p = torch.sigmoid(l)
        # AUC：降序 0 基 rank 和 S_desc → AUC = 1 - (S_desc - n_pos(n_pos-1)/2)/(n_pos·n_neg)
        order = p.argsort(descending=True)
        ranks = torch.empty(len(p), dtype=torch.long); ranks[order] = torch.arange(len(p))
        n_pos = yt.sum(); n_neg = len(yt) - n_pos
        if n_pos == 0 or n_neg == 0:
            auc = float("nan")
        else:
            s_desc = float((yt * ranks).sum())
            auc = 1.0 - (s_desc - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
        return float(auc), p.numpy(), l.numpy()

    best_auc, best_state, patience, hard_w = -1.0, None, 0, None
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0; nb = 0
        for ii, pp, mm, mt, yy, ww in tr_loader:
            lg = model(ii.to(dev), pp.to(dev), mm.to(dev), mt.to(dev))
            if args.loss == "asymmetric":   # 非对称损失：正样本几乎不降权（提升真餐置信度）
                loss = asymmetric_loss(lg, yy.to(dev), gamma_pos=1.0, gamma_neg=3.0,
                                       alpha=FOCAL_ALPHA, weights=ww.to(dev))
            else:
                loss = focal_loss(lg, yy.to(dev), alpha=FOCAL_ALPHA, weights=ww.to(dev))
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
            for ii, pp, mm, mt, _, _ in DataLoader(hm_ds, batch_size=256, shuffle=False):
                ltr.append(model(ii.to(dev), pp.to(dev), mm.to(dev), mt.to(dev)).cpu())
            ltr = torch.cat(ltr)
        ptr = torch.sigmoid(ltr).numpy()
        fp_mask = (y[tr] == 0) & (ptr > 0.5)
        n_hard = min(int(HARD_K * y[tr].sum()), int(fp_mask.sum()))
        if n_hard > 0:
            idx = np.where(fp_mask)[0][np.argsort(ptr[fp_mask])[-n_hard:]]
            hard_w = np.ones(train_ds.n_orig, np.float32)
            hard_w[idx] = HARD_W
            # 映射：dataset 索引 → 原始索引 → 权重
            train_ds.weights = torch.from_numpy(hard_w[train_ds.orig]).float()
            print(f"    硬负样本: {n_hard} 个（top sigmoid {ptr[idx].max():.3f}）", flush=True)
        else:
            hard_w = None

    model.load_state_dict(best_state)
    auc, p_va, _ = evaluate(val_ds)
    print(f"★ 最优 valAUC {best_auc:.3f}（重载后 {auc:.3f}）", flush=True)

    # ---- 保存：模型 + val 候选分数（解码用） ----
    torch.save(best_state, config.MODEL_DIR / f"mm_ranker_fold{fold_idx}.pt")
    order = [i for i, m in enumerate(META) if m["sid"] in va_set]
    np.savez(config.OUTPUT_DIR / f"mm_ranker_fold{fold_idx}_val.npz",
             score=p_va, label=np.array([y[i] for i in order], np.int8),
             meta=np.array([json.dumps(META[i]).encode() for i in order]))
    print(f"→ models/mm_ranker_fold{fold_idx}.pt + outputs/mm_ranker_fold{fold_idx}_val.npz（{len(order)} val 候选）", flush=True)


if __name__ == "__main__":
    main()
