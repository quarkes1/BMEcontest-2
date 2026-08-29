# -*- coding: utf-8 -*-
"""L3b 生理专家网络两折训练（nondominant 场景，PPG 分支）。
沿用 L3a 的工程经验：不压缩缓存、负样本抽样后全量入内存、num_workers=0、
阈值扫描覆盖低概率区、事件后处理带平滑（10s 步长 → smooth_win=3）。
运行：conda activate bme && python scripts/train_l3b.py --folds 0,1 --epochs 25
产物：models/l3b_ppgnn_fold{k}.pt + outputs/l3b_report.json"""
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
from src.models.l3b_ppgnn import (L3bPPGNN, build_ppg_window, denoise_ppg, hrv_features,
                                  N_PPG_CHANNELS, PPG_WINDOW_ROWS, HRV_DIMS, SEQ_LEN)

BASE = config.CACHE_DIR / "l3b_raw"
VAL_DIR = config.CACHE_DIR / "l3b_val_raw"
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_EVERY = 5
SMOOTH_WIN = 3              # 10s 步长 → 3 窗 = 30s 平滑
WIN_ROWS = 3150
POS_STRIDE = 1050


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
        Xs, hs, t0s, t1s = [], [], [], []
        for w in iter_window_labels(s, [], window_rows=WIN_ROWS, stride_rows=POS_STRIDE):
            X, h = build_ppg_window(s, w["start_row"], w["end_row"])
            Xs.append(X); hs.append(h)
            t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
        if Xs:
            np.savez(out, X=np.stack(Xs), hrv=np.stack(hs),
                     t0=np.array(t0s, dtype=np.int64), t1=np.array(t1s, dtype=np.int64))
        return ("ok", session_id)
    except Exception as e:
        return ("error", f"{session_id}: {type(e).__name__}: {e}")

def build_val_cache(val_sessions):
    out_dir = VAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "skip": 0, "error": []}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (status, info) in enumerate(ex.map(_val_session, [(s, out_dir) for s in val_sessions], chunksize=4)):
            if status == "error":
                stats["error"].append(info)
            else:
                stats[status] = stats[status] + 1
            if (i + 1) % 50 == 0:
                print(f"  验证缓存 {i+1}/{len(val_sessions)} 用时 {time.time()-t0:.0f}s", flush=True)
    return stats


# ------------------------------------------------------------------ 数据集（全量入内存）
class PPGSeqDataset(Dataset):
    """窗口全量入内存（或显存 fp16）；每样本 = SEQ_LEN 连续窗口（跨会话截断，不足左侧补零）。
    epoch 级洗牌：正样本序列（含任一正窗口）权重 3。"""
    def __init__(self, out_dir, neg_ratio=3.0, seq_step=3, on_gpu=False):
        self.seq_step = seq_step
        files = [f for f in sorted(out_dir.glob("*.npz")) if not f.name.endswith(".tmp.npz")]
        Xs, hs, ys, tws = [], [], [], []
        self.seq_starts = []
        self.seq_session = []
        rng = np.random.RandomState(config.RANDOM_SEED)
        for f in files:
            d = np.load(f)
            n = len(d["y"])
            if n == 0:
                continue
            Xs.append(d["X"]); hs.append(d["hrv"])
            ys.append(d["y"]); tws.append(d["tw"])
            base = sum(len(x) for x in Xs[:-1])
            for s0 in range(0, n, seq_step):
                self.seq_starts.append(base + s0)
                self.seq_session.append(len(self.seq_starts) - 1)
        self.X = np.concatenate(Xs)
        self.hrv = np.concatenate(hs).astype(np.float32)
        self.y = np.concatenate(ys).astype(np.float32)
        self.tw = np.concatenate(tws).astype(np.int64)
        # 会话边界（用于截断序列）
        self.bounds = np.cumsum([0] + [len(x) for x in Xs])
        # 正样本序列权重 3
        self.base_weights = np.array([
            3.0 if self.y[s0:s0 + SEQ_LEN].sum() > 0 else 1.0
            for s0 in self.seq_starts], dtype=np.float32)
        self.epoch_idx = np.arange(len(self.seq_starts))
        self.on_gpu = on_gpu
        if on_gpu:
            n = len(self.X)
            self.X_gpu = torch.empty((n, N_PPG_CHANNELS, PPG_WINDOW_ROWS),
                                     dtype=torch.float16, device="cuda")
            CH = 8192
            for s in range(0, n, CH):
                self.X_gpu[s:s + CH] = torch.from_numpy(self.X[s:s + CH]).cuda().half()
            del self.X
        print(f"  数据集: {len(self.y)} 窗口, {len(self.seq_starts)} 序列"
              f"{'（GPU fp16 驻留）' if on_gpu else ''}", flush=True)

    def reshuffle(self, seed):
        rng = np.random.RandomState(seed)
        self.epoch_idx = rng.choice(len(self.seq_starts), size=len(self.seq_starts),
                                    replace=True, p=self.base_weights / self.base_weights.sum())

    def __len__(self):
        return len(self.epoch_idx)

    def __getitem__(self, idx):
        k = self.epoch_idx[idx]
        s0 = self.seq_starts[k]
        # 会话边界截断
        b = np.searchsorted(self.bounds, s0, side="right") - 1
        lo, hi = self.bounds[b], self.bounds[b + 1]
        s1 = min(s0 + SEQ_LEN, hi)
        n_keep = s1 - s0
        h = np.zeros((SEQ_LEN, HRV_DIMS), dtype=np.float32)
        y = np.zeros(SEQ_LEN, dtype=np.float32)
        tw = np.full(SEQ_LEN, -1, dtype=np.int64)
        off = SEQ_LEN - n_keep                    # 左侧补零（保持时间顺序）
        h[off:] = self.hrv[s0:s1]
        y[off:] = self.y[s0:s1]
        tw[off:] = self.tw[s0:s1]
        if self.on_gpu:
            X = torch.zeros((SEQ_LEN, N_PPG_CHANNELS, PPG_WINDOW_ROWS),
                            dtype=torch.float16, device="cuda")
            X[off:] = self.X_gpu[s0:s1]
        else:
            X = np.zeros((SEQ_LEN, N_PPG_CHANNELS, PPG_WINDOW_ROWS), dtype=np.float32)
            X[off:] = self.X[s0:s1].astype(np.float32)
            X = torch.from_numpy(X)
        return (X, torch.from_numpy(h), torch.from_numpy(y), torch.from_numpy(tw))


# ------------------------------------------------------------------ 验证打分（流式）
def score_val(model, mean_h, std_h, device):
    files = [f for f in sorted(VAL_DIR.glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    probs, t0s, t1s = [], [], []
    model.eval()
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            n = len(d["t0"])
            X = d["X"].astype(np.float32)
            h = d["hrv"].astype(np.float32)
            h = (h - mean_h) / std_h
            seq_probs = np.zeros(n, dtype=np.float32)
            for s0 in range(0, n, 256):
                s1 = min(s0 + 256, n)
                xs = np.zeros((s1 - s0, SEQ_LEN, N_PPG_CHANNELS, PPG_WINDOW_ROWS), dtype=np.float32)
                hs = np.zeros((s1 - s0, SEQ_LEN, HRV_DIMS), dtype=np.float32)
                for j, i in enumerate(range(s0, s1)):
                    lo = max(0, i - SEQ_LEN + 1)
                    off = SEQ_LEN - (i - lo + 1)
                    xs[j, off:] = X[lo:i + 1]
                    hs[j, off:] = h[lo:i + 1]
                xb = torch.from_numpy(xs).to(device)
                hb = torch.from_numpy(hs).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(xb, hb)
                seq_probs[s0:s1] = torch.sigmoid(logits[:, -1]).float().cpu().numpy()
            probs.append(seq_probs)
            t0s.append(d["t0"]); t1s.append(d["t1"])
    return np.concatenate(probs), np.concatenate(t0s), np.concatenate(t1s)


def val_true_events(session_ids, scene="nondominant"):
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index().set_index("session_id")
    ext_set = {index.loc[s, "externalid"] for s in session_ids if s in index.index}
    events = []
    for ext in ext_set:
        for m in meal_meta.get(ext, []):
            if scene is None or m["scene"] == scene:
                events.append((m["before"], m["after"]))
    return events


# ------------------------------------------------------------------ 主流程
def train_fold(fold, epochs, data_mode="auto"):
    f = splits.load_folds()[fold]
    out_dir = BASE / f"fold{fold}"
    if not (out_dir / "build_stats.json").exists():
        from scripts.build_ppg_cache import build_fold
        build_fold(fold)
    bstats = json.loads((out_dir / "build_stats.json").read_text(encoding="utf-8"))
    print(f"fold {fold}: 正 {bstats['pos']} / 负 {bstats['neg']}", flush=True)
    print("构建验证窗口缓存...", flush=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)
    build_val_cache(f["val_sessions"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device}", flush=True)
    use_gpu = data_mode in ("gpu", "auto")
    try:
        ds = PPGSeqDataset(out_dir, on_gpu=use_gpu)
    except MemoryError as e:
        if data_mode == "gpu":
            raise
        print(f"  GPU 驻留失败（{e}），回退 CPU 内存数据集", flush=True)
        ds = PPGSeqDataset(out_dir, on_gpu=False)
    # hrv 标准化统计
    h_all = ds.hrv
    mean_h = h_all.mean(axis=0).astype(np.float32)
    std_h = h_all.std(axis=0).astype(np.float32) + 1e-6

    model = L3bPPGNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(ds) // BATCH
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    bce = torch.nn.BCEWithLogitsLoss()

    best = {"f1_nondominant": -1.0}
    for ep in range(1, epochs + 1):
        ds.reshuffle(config.RANDOM_SEED + ep)
        dl = PrefetchLoader(ds, batch_size=BATCH, drop_last=True)
        model.train()
        t0 = time.time(); loss_sum = 0.0; n_b = 0; pos_acc = 0.0
        for xb, hb, yb, twb in dl:
            xb, hb, yb, twb = xb.to(device), hb.to(device), yb.to(device), twb.to(device)
            hb = (hb - torch.from_numpy(mean_h).to(device)) / torch.from_numpy(std_h).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb, hb)
                loss = bce(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step()
            loss_sum += loss.item(); n_b += 1
            with torch.no_grad():
                pos_acc += float(((torch.sigmoid(logits) > 0.5) == (yb > 0.5))[yb > 0.5].float().mean())
        print(f"  ep {ep}/{epochs} loss={loss_sum/n_b:.4f} pos_acc={pos_acc/max(1,n_b):.3f} "
              f"用时 {time.time()-t0:.0f}s", flush=True)
        del dl

        if ep % VAL_EVERY == 0 or ep == epochs:
            probs, tv0, tv1 = score_val(model, mean_h, std_h, device)
            true_non = val_true_events(f["val_sessions"], "nondominant")
            true_all = val_true_events(f["val_sessions"], None)
            best_thr, f1_non = None, -1.0
            for thr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6):
                evs = windows_to_events(probs, tv0, tv1, thr, smooth_win=SMOOTH_WIN)
                m = compute_metrics(evs, true_non)
                if m["f1"] > f1_non:
                    f1_non, best_thr = m["f1"], thr
            m_all = compute_metrics(windows_to_events(probs, tv0, tv1, best_thr, smooth_win=SMOOTH_WIN), true_all)
            print(f"  val: thr={best_thr} F1(nondominant)={f1_non:.3f} "
                  f"F1(overall)={m_all['f1']:.3f} n_true={len(true_non)}", flush=True)
            if f1_non > best["f1_nondominant"]:
                best = {"epoch": ep, "threshold": best_thr, "f1_nondominant": f1_non,
                        "f1_overall": m_all["f1"],
                        "metrics": compute_metrics(
                            windows_to_events(probs, tv0, tv1, best_thr, smooth_win=SMOOTH_WIN), true_non)}
                config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), config.MODEL_DIR / f"l3b_ppgnn_fold{fold}.pt")
                print(f"  >>> 保存 best (ep {ep})", flush=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--data", default="auto", choices=["auto", "gpu", "cpu"],
                    help="数据集驻留位置：auto=优先显存，OOM 回退内存")
    args = ap.parse_args()
    t0 = time.time()
    report = {"folds": {}}
    for k in [int(x) for x in args.folds.split(",")]:
        if (config.MODEL_DIR / f"l3b_ppgnn_fold{k}.pt").exists():
            print(f"===== L3b fold {k} 已存在模型，跳过 =====", flush=True)
            continue
        print(f"===== L3b fold {k} =====", flush=True)
        report["folds"][str(k)] = train_fold(k, args.epochs, args.data)
    f1s = [v["f1_nondominant"] for v in report["folds"].values()]
    report["mean_f1_nondominant"] = float(np.mean(f1s))
    report["total_seconds"] = round(time.time() - t0, 1)
    (config.OUTPUT_DIR / "l3b_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"===== L3b 两折完成：F1(nondominant)={f1s}（用时 {time.time()-t0:.0f}s）", flush=True)

if __name__ == "__main__":
    main()
