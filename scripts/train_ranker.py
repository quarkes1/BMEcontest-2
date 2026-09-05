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
BATCH = int(os.environ.get("BME_BATCH", "64"))   # 训练 batch（GPU 8GB：256 仍留余量；增速 ~4×）
EVAL_BATCH = int(os.environ.get("BME_EVAL_BATCH", "256"))   # val/硬负挖掘前向 batch
LR = 1e-3
WD = 1e-4
PATIENCE = 12
HARD_W = float(os.environ.get("BME_HARD_W", "5.0"))   # 硬负样本提权倍数（×10 验证：过激→保守化）
HARD_K = int(os.environ.get("BME_HARD_K", "5"))       # top-k = K× 正样本数
FOCAL_ALPHA = float(os.environ.get("BME_FOCAL_ALPHA", "0.35"))  # Focal α（0.25→0.35：更重视召回）
SEED = int(os.environ.get("BME_SEED", "42"))   # 随机种子（BME_SEED 覆盖——多套训练选优）
NEG_RATIO = int(os.environ.get("BME_NEG_RATIO", "0"))   # 负样本子采样比（>0：负:正 ≤ NEG_RATIO，抗 0.4% 稀释）
NO_GATE = os.environ.get("BME_NO_GATE", "0") == "1"   # 归零 gate_prob 元特征（gate AUC 0.729 弱 → 会话级捷径 → 窗级判别学不到）
AUG_POS = int(os.environ.get("BME_AUG_POS", "1"))   # 正样本增强倍数（>1：时间抖动±5s+噪声，正样本稀缺）


ZERO_PPG = torch.zeros(48, 66)   # --no-ppg 占位行（模型 use_ppg=False 不读；DataLoader stack 时逐行复制）
ZERO_MA = torch.zeros(48, 2)


class CandDS(Dataset):
    """候选窗口数据集。pos_rep>1 时训练集正样本复制（1:45 不平衡 → 过采样平衡）。
    meta 特征：dur_s/log、gate_prob、prior_h（归一化在构造时完成）。
    惰性索引：__getitem__ 按 orig 取行——torch.from_numpy 与 imu 共享内存，
    正样本过采样与 train/hm 双 dataset 不再各复制一份全量候选（历史 OOM 根因）。"""

    def __init__(self, imu, ppg, ma, meta, y, pos_rep=1):
        orig = np.arange(len(y))
        pos = np.where(y == 1)[0]
        if pos_rep > 1 and len(pos):
            orig = np.concatenate([orig, np.repeat(pos, pos_rep - 1)])
        self.orig = orig
        self.imu = torch.from_numpy(imu)          # 共享不复制（行索引在 __getitem__ 内）
        self.ppg = torch.from_numpy(ppg) if ppg is not None else None
        self.ma = torch.from_numpy(ma) if ma is not None else None
        self.meta = torch.from_numpy(meta)
        self.y = torch.from_numpy(y).float()   # focal loss 需 float（numpy 侧为 int8）
        self.weights = torch.ones(len(orig), dtype=torch.float32)
        self.n_orig = len(y)

    def __len__(self):
        return len(self.orig)

    def __getitem__(self, i):
        o = self.orig[i]
        if self.ppg is None:   # --no-ppg：共享零行占位（模型不读）
            return self.imu[o], ZERO_PPG, ZERO_MA, self.meta[o], self.y[o], self.weights[i]
        return self.imu[o], self.ppg[o], self.ma[o], self.meta[o], self.y[o], self.weights[i]


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
    ap.add_argument("--mix-fd", default=None,
                    help="FD 窗缓存（cache/fd_windows/*.npz）与目标候选联合训练"
                         "——FD 正密度 7.9% 提供强判别信号（针对排序头 valAUC 瓶颈）")
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
    need_ppg = not args.no_ppg
    X, Xp, Xm, Y, META = [], [], [], [], []
    for p in sorted(d.glob("*.npz")):
        with np.load(p, allow_pickle=True) as z:   # with: 释放 mmap 句柄（1108 文件累积致 OOM）
            n = len(z["meta"])
            for j in range(n):
                X.append(z[f"c{j}"])               # float16（缓存原生）
                if need_ppg:                        # --no-ppg 不读 ppg/ma（模型 use_ppg=False 忽略；省 ~2GB×N 份）
                    Xp.append(z["ppg"][j]); Xm.append(z["ma"][j])
            m = [json.loads(x.decode()) for x in z["meta"]]
            META.extend(m)
            Y.extend(int(mm["label"]) for mm in m)
    imu = np.stack(X); del X
    ppg = ma = None
    if need_ppg:
        ppg = np.stack(Xp); ma = np.stack(Xm)
        del Xp, Xm
    y = np.array(Y, np.int8)
    meta = np.stack([[np.log1p(m["dur_s"]), m["prior_h"],
                      0.0 if NO_GATE else m["gate_prob"]] for m in META]).astype(np.float32)  # gate_prob 全体会话真实分；BME_NO_GATE=1 时归零（窗级判别实验）
    split_idx = np.array([0 if m["sid"] in tr_set else 1 for m in META])
    print(f"fold{fold_idx}: {len(y)} 候选（正 {y.sum()}，{y.mean()*100:.1f}%）", flush=True)
    tr, va = split_idx == 0, split_idx == 1
    if NEG_RATIO > 0 and (y[tr] == 0).any():   # 负样本子采样（方案 A：抗正样本稀释）
        pos_n = int(y[tr].sum())
        neg_idx = np.where((split_idx == 0) & (y == 0))[0]
        keep_n = min(len(neg_idx), max(pos_n * NEG_RATIO, pos_n))
        rng = np.random.RandomState(SEED)
        hard_frac = float(os.environ.get("BME_NEG_HARD_FRAC", "0"))   # 分层采样：优先含餐会话的负窗
        if hard_frac > 0:
            # 含餐会话 = label==1 窗所在会话 ∪（train 含餐会话，经 meal_meta 判断）
            meal_sids = {m["sid"] for m in META if m["label"] == 1 and m["sid"] in tr_set}
            try:
                from src.data import manifests
                meal_meta, _ = manifests.load_meal_meta()
                idx = manifests.load_sensor_index()
                for _, r in idx.iterrows():
                    ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
                    if sid in tr_set and any(m["before"] >= st and m["after"] <= en
                                             for m in meal_meta.get(ext, [])):
                        meal_sids.add(sid)
            except Exception:
                pass
            hard_pool = np.array([i for i in neg_idx if META[i]["sid"] in meal_sids])
            easy_pool = np.array([i for i in neg_idx if META[i]["sid"] not in meal_sids])
            n_hard = int(keep_n * hard_frac)
            n_hard = min(n_hard, len(hard_pool))
            keep_neg = np.concatenate([
                rng.choice(hard_pool, n_hard, replace=False),
                rng.choice(easy_pool, keep_n - n_hard, replace=False)])
            print(f"  分层负采样：{keep_n}（含餐会话负窗 {n_hard} + 其余 {keep_n - n_hard}）", flush=True)
        else:
            keep_neg = rng.choice(neg_idx, keep_n, replace=False)
        tr = (split_idx == 0) & (y == 1)
        tr[keep_neg] = True
        print(f"  负样本子采样：{len(neg_idx)} → {keep_n}（正:负 1:{NEG_RATIO}）", flush=True)
    print(f"  train {tr.sum()} 候选（正 {y[tr].sum()}）| val {va.sum()}（正 {y[va].sum()}）", flush=True)
    if args.mix_fd:   # FD 联合训练：FD 窗（正密度高）混合进 train（val 保持目标域）
        assert need_ppg, "--mix-fd 需配合 PPG 分支（--no-ppg 时模型无 ppg 输入）"
        zp = config.CACHE_DIR / "fd_windows" / f"{args.mix_fd}.npz"
        assert zp.exists(), f"缺 FD 窗缓存 {zp}"
        zf = np.load(zp, allow_pickle=True)
        f_imu = zf["imu"].astype(np.float32)  # FD 窗 float32→下方转
        f_y = zf["label"].astype(np.float32)
        # FD 采样：正全量 + 负（混合后 FD 约占 train 半量，不淹没域校准）
        f_pos = np.where(f_y == 1)[0]
        f_neg = np.where(f_y == 0)[0]
        rng = np.random.RandomState(SEED)
        n_neg = min(len(f_neg), len(f_pos) * 10)
        f_sel = np.concatenate([f_pos, rng.choice(f_neg, n_neg, replace=False)])
        rng.shuffle(f_sel)
        n_fd = len(f_sel)
        n_tgt_tr = int(tr.sum())
        # 目标候选采样至与 FD 相当（1:1 混合）
        tgt_i = np.where(tr)[0]
        n_tgt = min(n_tgt_tr, n_fd)
        tgt_sel = rng.choice(tgt_i, n_tgt, replace=False)
        # 组装（FD 窗已 z-score 归一化，与 FD-init 后的目标域同分布）
        va_i = np.where(va)[0]                    # 原 val 全部保留
        imu = np.concatenate([imu[tgt_sel], f_imu[f_sel], imu[va_i]])
        ppg = np.concatenate([ppg[tgt_sel], np.zeros((n_fd,) + ppg.shape[1:], np.float32),
                              ppg[va_i]])
        ma = np.concatenate([ma[tgt_sel], np.zeros((n_fd,) + ma.shape[1:], np.float32),
                             ma[va_i]])
        y = np.concatenate([y[tgt_sel], f_y[f_sel], y[va_i]])
        meta = np.concatenate([meta[tgt_sel],
                               np.tile(np.array([[np.log1p(1800), 0.0, 0.5]], np.float32),
                                       (n_fd, 1)),
                               meta[va_i]])
        META = [META[i] for i in np.concatenate([tgt_sel, va_i])] + [None] * n_fd
        tr = np.concatenate([np.ones(n_tgt + n_fd, bool), np.zeros(len(va_i), bool)])
        va = ~tr
        n_tr_pos = int((y[:n_tgt] == 1).sum()) + int((f_y[f_sel] == 1).sum())
        print(f"  FD 联合：目标 {n_tgt}（正 {int((y[:n_tgt]==1).sum())}）+ FD {n_fd}"
              f"（正 {int((f_y[f_sel]==1).sum())}）+ val {len(va_i)}（正 {int((y[va_i]==1).sum())}）",
              flush=True)
    if AUG_POS > 1 and (y[tr] == 1).any():   # 正样本增强（时间抖动 + 幅值噪声）
        pos_i = np.where(tr & (y == 1))[0]
        rng = np.random.RandomState(SEED)
        # float32 域算 std：raw ADC float16 累加溢出（sum 超 float16 上限 → inf）
        std_ch = imu[pos_i].astype(np.float32).std((0, 1)) + 1e-6
        aug_i = np.concatenate([pos_i] + [
            pos_i for _ in range(AUG_POS - 1)])
        n_aug = len(pos_i) * (AUG_POS - 1)
        imu = np.concatenate([imu, np.zeros((n_aug, imu.shape[1], imu.shape[2]), np.float16)])
        if need_ppg:   # --no-ppg：无 ppg/ma 可扩（模型不读）
            ppg = np.concatenate([ppg, np.zeros((n_aug,) + ppg.shape[1:], np.float32)])
            ma = np.concatenate([ma, np.zeros((n_aug,) + ma.shape[1:], np.float32)])
        meta = np.concatenate([meta, np.zeros((n_aug, meta.shape[1]), np.float32)])
        y = np.concatenate([y, np.ones(n_aug, np.int8)])
        tr = np.concatenate([tr, np.ones(n_aug, bool)])
        va = np.concatenate([va, np.zeros(n_aug, bool)])
        for j, src in enumerate(pos_i):
            for rep in range(AUG_POS - 1):
                tgt = len(pos_i) + j * (AUG_POS - 1) + rep
                shift = int(rng.randint(-50, 51))       # ±5s 时间抖动
                imu[tgt] = np.roll(imu[src], shift, axis=0)
                imu[tgt] += rng.randn(*imu[tgt].shape).astype(np.float32) * (0.03 * std_ch)
        META = META + [None] * n_aug   # 仅占位（meta 数组已扩展；META 仅用于 sid 输出筛选——aug 不入 val）
        print(f"  正样本增强：{len(pos_i)} → ×{AUG_POS}（+{n_aug} 抖动窗）", flush=True)

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        if os.environ.get("BME_CUDNN_BENCH", "0") == "1":
            torch.backends.cudnn.benchmark = True   # 默认关：显存 ~96% 时 benchmark workspace 探索风险

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
        mu, sd = norm_mean[:n_ch].astype(np.float32), norm_std[:n_ch].astype(np.float32) + 1e-6
        blk = 4000   # 分块转 float32 归一化（原 float16-=float32 广播整量临时 ~7GB → OOM）
        for i0 in range(0, imu.shape[0], blk):
            b = imu[i0:i0 + blk].astype(np.float32)
            b[..., :n_ch] = (b[..., :n_ch] - mu) / sd
            imu[i0:i0 + blk] = b.astype(np.float16)
        print(f"  输入 z-score 归一化（前 {n_ch} 通道，分块）", flush=True)
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
    imu_tr, imu_va = imu[tr], imu[va]
    del imu   # 分区后释放全量（此后只持有 train/val 两份，CandDS 共享不复制）
    ppg_tr = ppg[tr] if ppg is not None else None
    ppg_va = ppg[va] if ppg is not None else None
    ma_tr = ma[tr] if ma is not None else None
    ma_va = ma[va] if ma is not None else None
    meta_tr, meta_va = meta[tr], meta[va]
    y_tr, y_va = y[tr], y[va]
    train_ds = CandDS(imu_tr, ppg_tr, ma_tr, meta_tr, y_tr, pos_rep=POS_REP)
    val_ds = CandDS(imu_va, ppg_va, ma_va, meta_va, y_va)
    print(f"  正样本过采样 ×{POS_REP} → train {len(train_ds)}", flush=True)
    hm_ds = CandDS(imu_tr, ppg_tr, ma_tr, meta_tr, y_tr)   # 硬负样本挖掘（共享 imu_tr——from_numpy 零复制）
    tr_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va_loader = DataLoader(val_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def evaluate(ds):
        model.eval()
        logs, ys = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for ii, pp, mm, mt, yy, _ in DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False):
                logs.append(model(ii.to(dev), pp.to(dev), mm.to(dev), mt.to(dev)).float().cpu())
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
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(ii.to(dev), pp.to(dev), mm.to(dev), mt.to(dev))
            lg = lg.float()   # loss 在 fp32 域（autocast 外），精度稳定
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
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ltr = []
            for ii, pp, mm, mt, _, _ in DataLoader(hm_ds, batch_size=EVAL_BATCH, shuffle=False):
                ltr.append(model(ii.to(dev), pp.to(dev), mm.to(dev), mt.to(dev)).float().cpu())
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
    order = [i for i, m in enumerate(META) if m is not None and m["sid"] in va_set]
    np.savez(config.OUTPUT_DIR / f"mm_ranker_fold{fold_idx}_val.npz",
             score=p_va, label=np.array([y[i] for i in order], np.int8),
             meta=np.array([json.dumps(META[i]).encode() for i in order]))
    print(f"→ models/mm_ranker_fold{fold_idx}.pt + outputs/mm_ranker_fold{fold_idx}_val.npz（{len(order)} val 候选）", flush=True)


if __name__ == "__main__":
    main()
