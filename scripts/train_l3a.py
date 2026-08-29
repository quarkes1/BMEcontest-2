# -*- coding: utf-8 -*-
"""L3a 动作专家网络两折训练（W2 交付：fold 0/1 F1）。
流程（每折）：确认原始缓存（缺则构建）→ 构建验证窗口缓存（步长 1s 全窗口）
→ 训练（AMP fp16 / WeightedRandomSampler / 镜像增强）→ 验证阈值扫描 → 报告。
运行：conda activate bme && python scripts/train_l3a.py --folds 0,1 --epochs 25
产物：models/l3a_cnn_fold{k}.pt + outputs/l3a_report.json"""
import argparse
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.train.prefetch import PrefetchLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.data.loader import load_session, detect_binary, _find_collect_data
from src.data.windows import iter_window_labels
from src.eval.metrics import compute_metrics
from src.infer.events import windows_to_events
from src.models.l3a_cnn import (L3aCNN, L3aCNNLarge, build_raw_channels,
                                mirror_channels, N_CHANNELS, WINDOW_LEN)
from src.models.l3a_resnet import L3aResNet

BASE = config.CACHE_DIR / "l3a_raw"
VAL_DIR = config.CACHE_DIR / "l3a_val_raw"
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
AUX_WEIGHT = 0.3
VAL_EVERY = 5


# ------------------------------------------------------------------ 验证窗口缓存
def _val_session(args):
    session_id, out_dir = args
    out = out_dir / f"{session_id}.npz"
    if out.exists():
        return ("skip", session_id)
    try:
        d = config.SENSOR_DIR / session_id
        txt = _find_collect_data(str(d))
        if detect_binary(txt):
            return ("binary", session_id)
        s = load_session(session_id)
        Xs, t0s, t1s = [], [], []
        for w in iter_window_labels(s, []):          # 无餐 → 全部窗口 label=0
            Xs.append(build_raw_channels(s, w["start_row"], w["end_row"]))
            t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
        if Xs:
            np.savez(out, X=np.stack(Xs),
                     t0=np.array(t0s, dtype=np.int64), t1=np.array(t1s, dtype=np.int64))
        return ("ok", session_id)
    except Exception as e:
        return ("error", f"{session_id}: {type(e).__name__}: {e}")

def build_val_cache(val_sessions):
    out_dir = VAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(sid, out_dir) for sid in val_sessions]
    stats = {"ok": 0, "skip": 0, "error": []}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (status, info) in enumerate(ex.map(_val_session, tasks, chunksize=4)):
            if status == "error":
                stats["error"].append(info)
            else:
                stats[status] = stats[status] + 1
            if (i + 1) % 50 == 0:
                print(f"  验证缓存 {i+1}/{len(tasks)} 用时 {time.time()-t0:.0f}s", flush=True)
    return stats


# ------------------------------------------------------------------ 数据集
# 教训（2026-08-29）：① 压缩 npz 随机访问 = 每 batch 解压上百文件（200h/epoch）
# ② 多进程 spawn 数据管线在 Windows 上不可靠（worker 满载、主进程死等、GPU 空转）。
# 最终方案：负样本 3:1 抽样后全量载入内存（~10GB），epoch 内纯内存洗牌，num_workers=0。
class RawDataset(Dataset):
    def __init__(self, out_dir, stats, neg_ratio=3.0, mirror=True,
                 stretch=True, jitter=True, ch_drop=0.1, on_gpu=False):
        self.mean = np.array(stats["mean"], dtype=np.float32).reshape(N_CHANNELS, 1)
        self.std = np.array(stats["std"], dtype=np.float32).reshape(N_CHANNELS, 1) + 1e-6
        self.mirror = mirror
        self.stretch = stretch
        self.jitter = jitter
        self.ch_drop = ch_drop
        self.on_gpu = on_gpu
        files = [f for f in sorted(out_dir.glob("*.npz")) if not f.name.endswith(".tmp.npz")]
        neg_prob = min(1.0, neg_ratio * stats["pos"] / max(1, stats["neg"]))
        rng = np.random.RandomState(config.RANDOM_SEED)
        # 预分配（先统计再填充，避免 concat 双倍内存峰值；掩码只取一次随机数）
        total = 0
        masks = []
        for f in files:
            d = np.load(f)
            m = (d["y"] == 1) | ((d["y"] == 0) & (rng.random(len(d["y"])) < neg_prob))
            masks.append(m)
            total += int(m.sum())
        self.X = np.empty((total, N_CHANNELS, WINDOW_LEN), dtype=np.float32)
        self.y = np.empty(total, dtype=np.float32)
        self.tw = np.empty(total, dtype=np.int64)
        off = 0
        for f, m in zip(files, masks):
            d = np.load(f)
            n = int(m.sum())
            self.X[off:off + n] = d["X"][m]
            self.y[off:off + n] = d["y"][m]
            self.tw[off:off + n] = d["tw"][m]
            off += n
        self.idx = np.arange(total)
        # 预标准化：一次完成，省去每次取数的逐样本归一化（原地操作避免峰值翻倍）
        self.X -= self.mean
        self.X /= self.std
        if on_gpu:
            self._to_gpu()
        print(f"  内存数据集: {total} 窗口（正 {int((self.y == 1).sum())}），"
              f"{'GPU fp16 驻留' if on_gpu else '已预标准化'}", flush=True)

    def _to_gpu(self):
        """分块搬到显存（fp16 ~5.1GB），避免整块 fp32 临时量爆显存。"""
        n = len(self.X)
        free, _ = torch.cuda.mem_get_info()
        need = n * N_CHANNELS * WINDOW_LEN * 2
        if need > free * 0.85:
            raise MemoryError(f"显存不足: 需 {need/2**30:.1f}GB, 可用 {free/2**30:.1f}GB")
        self.X_gpu = torch.empty((n, N_CHANNELS, WINDOW_LEN), dtype=torch.float16, device="cuda")
        CH = 32768
        for s in range(0, n, CH):
            self.X_gpu[s:s + CH] = torch.from_numpy(self.X[s:s + CH]).cuda().half()
        del self.X
        self.y_t = torch.from_numpy(self.y).cuda()
        self.tw_t = torch.from_numpy(self.tw).cuda()

    def reshuffle(self, seed):
        self.idx = np.random.RandomState(seed).permutation(len(self.idx))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        i = self.idx[idx]
        if self.on_gpu:
            return self._getitem_gpu(i)
        X = self.X[i].copy()
        rng = np.random
        if self.mirror and rng.random() < 0.5:
            X = mirror_channels(X)
        if self.jitter and rng.random() < 0.5:
            X *= rng.uniform(0.9, 1.1)                   # 幅度抖动 ±10%
        if self.stretch and rng.random() < 0.5:
            scale = rng.uniform(0.95, 1.05)              # 时间伸缩 ±5%
            new_len = int(round(X.shape[1] * scale))
            xs = np.linspace(0, X.shape[1] - 1, new_len)
            X = np.stack([np.interp(xs, np.arange(X.shape[1]), X[c]) for c in range(X.shape[0])])
            if new_len < WINDOW_LEN:
                X = np.pad(X, ((0, 0), (0, WINDOW_LEN - new_len)))
            else:
                X = X[:, :WINDOW_LEN]
        if self.ch_drop:
            X = X.copy()
            for c in range(6):                            # 6 个原始信号通道（la/gyro）
                if rng.random() < self.ch_drop:
                    X[c] = 0.0                            # 标准化空间置零 = 通道均值填补
        return (torch.from_numpy(X), torch.tensor(self.y[i], dtype=torch.float32),
                torch.tensor(self.tw[i], dtype=torch.long))

    def _getitem_gpu(self, i):
        """GPU 路径：只做 gather（增强由 augment_batch 在批级向量化完成）。"""
        return self.X_gpu[i], self.y_t[i], self.tw_t[i]

    def augment_batch(self, X):
        """批量级 GPU 增强：每 batch ~10 个 kernel（而非每样本），避免与训练抢 GPU。
        X: (B, 11, 525) fp32 cuda，返回增强后的 X。随机决策在 CPU 生成（无同步）。"""
        B = X.shape[0]
        rng = np.random
        if self.mirror:
            flips = torch.from_numpy(rng.random(B) < 0.5).cuda()
            X[flips][:, [0, 3, 6]] = -X[flips][:, [0, 3, 6]]
        if self.jitter:
            scales = torch.from_numpy(0.9 + 0.2 * rng.random(B)).cuda().float()
            X *= scales.view(-1, 1, 1)
        if self.ch_drop:
            for c in range(6):
                drop = torch.from_numpy(rng.random(B) < self.ch_drop).cuda()
                X[drop, c] = 0.0
        if self.stretch:
            # 时间伸缩的等价近似：每样本随机平移 ±0.5s（窗口内容不变）
            shifts = torch.from_numpy(rng.randint(-52, 53, size=B)).cuda()
            idx = (torch.arange(WINDOW_LEN, device="cuda")[None, :] - shifts[:, None]) % WINDOW_LEN
            X = X.gather(2, idx[:, None, :].expand(-1, N_CHANNELS, -1))
        return X


# ------------------------------------------------------------------ 验证打分
def score_val_cache(model, out_dir, mean, std, device):
    files = [f for f in sorted(out_dir.glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    probs, t0s, t1s = [], [], []
    mean_np = mean.cpu().numpy()
    std_np = std.cpu().numpy()
    model.eval()
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            X = ((d["X"].astype(np.float32) - mean_np) / std_np)
            for b in range(0, len(X), BATCH * 2):
                xb = torch.from_numpy(X[b:b + BATCH * 2]).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logit, _ = model(xb)
                probs.append(torch.sigmoid(logit).float().cpu().numpy())
            t0s.append(d["t0"]); t1s.append(d["t1"])
    return np.concatenate(probs), np.concatenate(t0s), np.concatenate(t1s)


def val_true_events(session_ids, scene="dominant"):
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index().set_index("session_id")
    ext_set = {index.loc[s, "externalid"] for s in session_ids if s in index.index}
    events = []
    for ext in ext_set:
        for m in meal_meta.get(ext, []):
            if scene is None or m["scene"] == scene:
                events.append((m["before"], m["after"]))
    return events


# ------------------------------------------------------------------ 训练主流程
def train_fold(fold, epochs, model_name, data_mode="auto", use_compile=False):
    f = splits.load_folds()[fold]
    out_dir = BASE / f"fold{fold}"
    if not (out_dir / "stats.json").exists():
        from scripts.build_raw_cache import build_fold
        build_fold(fold)
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    print(f"fold {fold}: 训练窗口 {stats['n_windows']}（正 {stats['pos']} / 负 {stats['neg']}）", flush=True)

    print("构建验证窗口缓存...", flush=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)      # 每折清空，防止混入他折残留文件
    build_val_cache(f["val_sessions"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True       # TF32 卷积加速 ~20-30%
    print(f"device={device}", flush=True)

    model = {"small": L3aCNN(5), "large": L3aCNNLarge(5),
             "resnet": L3aResNet(5)}[model_name].to(device)
    print(f"model: {model_name} ({sum(p.numel() for p in model.parameters())} params)", flush=True)
    if use_compile and device == "cuda":
        try:
            model = torch.compile(model)
            print("  torch.compile ON", flush=True)
        except Exception as e:
            print(f"  torch.compile 失败（{type(e).__name__}），回退 eager", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = (3 * stats["pos"] + int(min(stats["neg"], 3 * stats["pos"]))) // BATCH
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)

    mean = torch.from_numpy(np.array(stats["mean"], dtype=np.float32).reshape(1, N_CHANNELS, 1)).to(device)
    std = torch.from_numpy(np.array(stats["std"], dtype=np.float32).reshape(1, N_CHANNELS, 1)).to(device) + 1e-6

    best = {"f1_dominant": -1.0}
    use_gpu = data_mode in ("gpu", "auto")
    try:
        ds = RawDataset(out_dir, stats, on_gpu=use_gpu)
    except MemoryError as e:
        if data_mode == "gpu":
            raise
        print(f"  GPU 驻留失败（{e}），回退 CPU 内存数据集", flush=True)
        ds = RawDataset(out_dir, stats, on_gpu=False)
    for ep in range(1, epochs + 1):
        ds.reshuffle(config.RANDOM_SEED + ep)  # epoch 级纯内存洗牌
        dl = PrefetchLoader(ds, batch_size=BATCH)   # 单线程预取：CPU 增强与 GPU 计算重叠
        model.train()
        t0 = time.time(); loss_sum = 0.0; n_b = 0; pos_acc = 0.0
        for xb, yb, twb in dl:
            xb, yb, twb = xb.to(device), yb.to(device), twb.to(device)
            twb = twb.where(yb > 0.5, torch.full_like(twb, -1))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logit, tw_logit = model(xb)
                loss = bce(logit, yb * 0.9 + 0.05) + AUX_WEIGHT * ce(tw_logit, twb)   # label smoothing 0.1
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step()
            loss_sum += loss.item(); n_b += 1
            with torch.no_grad():
                pos_acc += float(((torch.sigmoid(logit) > 0.5) == (yb > 0.5))[yb > 0.5].float().mean())
        print(f"  ep {ep}/{epochs} loss={loss_sum/n_b:.4f} pos_acc={pos_acc/max(1,n_b):.3f} "
              f"用时 {time.time()-t0:.0f}s", flush=True)
        del dl

        if ep % VAL_EVERY == 0 or ep == epochs:
            probs, tv0, tv1 = score_val_cache(model, VAL_DIR, mean, std, device)
            true_dom = val_true_events(f["val_sessions"], "dominant")
            true_all = val_true_events(f["val_sessions"], None)
            best_thr, f1_dom = None, -1.0
            for thr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6):
                evs = windows_to_events(probs, tv0, tv1, thr)
                m = compute_metrics(evs, true_dom)
                if m["f1"] > f1_dom:
                    f1_dom, best_thr = m["f1"], thr
            m_all = compute_metrics(windows_to_events(probs, tv0, tv1, best_thr), true_all)
            print(f"  val: thr={best_thr} F1(dominant)={f1_dom:.3f} "
                  f"F1(overall)={m_all['f1']:.3f} n_true_dom={len(true_dom)}", flush=True)
            if f1_dom > best["f1_dominant"]:
                best = {"epoch": ep, "threshold": best_thr, "f1_dominant": f1_dom,
                        "f1_overall": m_all["f1"], "metrics_dominant": compute_metrics(
                            windows_to_events(probs, tv0, tv1, best_thr), true_dom)}
                config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), config.MODEL_DIR / f"l3a_cnn_{model_name}_fold{fold}.pt")
                print(f"  >>> 保存 best (ep {ep})", flush=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)      # 验证缓存 ~14GB/折，用完即删
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--model", default="small", choices=["small", "large", "resnet"])
    ap.add_argument("--compile", action="store_true", help="torch.compile 加速（cuda）")
    ap.add_argument("--data", default="auto", choices=["auto", "gpu", "cpu"],
                    help="数据集驻留位置：auto=优先显存，OOM 回退内存")
    args = ap.parse_args()
    t0 = time.time()
    report = {"folds": {}}
    for k in [int(x) for x in args.folds.split(",")]:
        ckpt = config.MODEL_DIR / f"l3a_cnn_{args.model}_fold{k}.pt"
        if ckpt.exists():
            print(f"===== L3a {args.model} fold {k} 已存在模型，跳过 =====", flush=True)
            continue
        print(f"===== L3a {args.model} fold {k} =====", flush=True)
        report["folds"][str(k)] = train_fold(k, args.epochs, args.model, args.data,
                                            use_compile=args.compile)
    f1s = [v["f1_dominant"] for v in report["folds"].values()]
    report["mean_f1_dominant"] = float(np.mean(f1s))
    report["total_seconds"] = round(time.time() - t0, 1)
    (config.OUTPUT_DIR / "l3a_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"===== L3a 两折完成：F1(dominant)={f1s} 报告 outputs/l3a_report.json "
          f"（总用时 {time.time()-t0:.0f}s）", flush=True)

if __name__ == "__main__":
    main()
