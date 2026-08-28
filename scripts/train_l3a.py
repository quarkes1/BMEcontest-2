# -*- coding: utf-8 -*-
"""L3a 动作专家网络两折训练（W2 交付：fold 0/1 F1）。
流程（每折）：确认原始缓存（缺则构建）→ 构建验证窗口缓存（步长 1s 全窗口）
→ 训练（AMP fp16 / WeightedRandomSampler / 镜像增强）→ 验证阈值扫描 → 报告。
运行：conda activate bme && python scripts/train_l3a.py --folds 0,1 --epochs 25
产物：models/l3a_cnn_fold{k}.pt + outputs/l3a_report.json"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.data.loader import load_session, detect_binary, _find_collect_data
from src.data.windows import iter_window_labels
from src.eval.metrics import compute_metrics
from src.infer.events import windows_to_events
from src.models.l3a_cnn import L3aCNN, build_raw_channels, mirror_channels, N_CHANNELS, WINDOW_LEN

BASE = config.CACHE_DIR / "l3a_raw"
VAL_DIR = config.CACHE_DIR / "l3a_val_raw"
BATCH = 128
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
            np.savez_compressed(out, X=np.stack(Xs),
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
class RawDataset(Dataset):
    def __init__(self, out_dir, stats, mirror=True):
        self.files = sorted(out_dir.glob("*.npz"))
        self.mean = np.array(stats["mean"], dtype=np.float32).reshape(1, N_CHANNELS, 1)
        self.std = np.array(stats["std"], dtype=np.float32).reshape(1, N_CHANNELS, 1) + 1e-6
        self.mirror = mirror
        self.entries = []          # (file_idx, local_idx, label, tw)
        self.file_labels = []
        self._cache = {}
        self._cache_order = []
        for fi, f in enumerate(self.files):
            d = np.load(f)
            n = len(d["y"])
            self.file_labels.append((d["y"], d["tw"]))
            for i in range(n):
                w = 3.0 if d["y"][i] == 1 else 1.0
                self.entries.append((fi, i, w))
        self.weights = np.array([e[2] for e in self.entries], dtype=np.float32)
        self.rng = np.random.RandomState(config.RANDOM_SEED)

    def __len__(self):
        return len(self.entries)

    def _get_file(self, fi):
        if fi not in self._cache:
            self._cache[fi] = np.load(self.files[fi])
            self._cache_order.append(fi)
            if len(self._cache) > 2:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
        return self._cache[fi]

    def __getitem__(self, idx):
        fi, i, _ = self.entries[idx]
        d = self._get_file(fi)
        X = d["X"][i].astype(np.float32)
        if self.mirror and self.rng.random() < 0.5:
            X = mirror_channels(X)
        X = (X - self.mean) / self.std
        y, tw = self.file_labels[fi]
        return (torch.from_numpy(X), torch.tensor(y[i], dtype=torch.float32),
                torch.tensor(tw[i], dtype=torch.long))


# ------------------------------------------------------------------ 验证打分
def score_val_cache(model, out_dir, mean, std, device):
    files = sorted(out_dir.glob("*.npz"))
    probs, t0s, t1s = [], [], []
    model.eval()
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            X = ((d["X"].astype(np.float32) - mean) / std)
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
def train_fold(fold, epochs):
    f = splits.load_folds()[fold]
    out_dir = BASE / f"fold{fold}"
    if not (out_dir / "stats.json").exists():
        from scripts.build_raw_cache import build_fold
        build_fold(fold)
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    print(f"fold {fold}: 训练窗口 {stats['n_windows']}（正 {stats['pos']} / 负 {stats['neg']}）", flush=True)

    print("构建验证窗口缓存...", flush=True)
    build_val_cache(f["val_sessions"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    ds = RawDataset(out_dir, stats, mirror=True)
    sampler = WeightedRandomSampler(ds.weights, num_samples=len(ds), replacement=True)
    dl = DataLoader(ds, batch_size=BATCH, sampler=sampler, num_workers=4,
                    pin_memory=True, persistent_workers=True, drop_last=True)

    model = L3aCNN(num_tableware=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(dl)
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)

    mean = torch.from_numpy(np.array(stats["mean"], dtype=np.float32).reshape(1, N_CHANNELS, 1)).to(device)
    std = torch.from_numpy(np.array(stats["std"], dtype=np.float32).reshape(1, N_CHANNELS, 1)).to(device) + 1e-6

    best = {"f1_dominant": -1.0}
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time(); loss_sum = 0.0; n_b = 0; pos_acc = 0.0
        for xb, yb, twb in dl:
            xb, yb, twb = xb.to(device), yb.to(device), twb.to(device)
            twb = twb.where(yb > 0.5, torch.full_like(twb, -1))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logit, tw_logit = model(xb)
                loss = bce(logit, yb) + AUX_WEIGHT * ce(tw_logit, twb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step()
            loss_sum += float(loss); n_b += 1
            with torch.no_grad():
                pos_acc += float(((torch.sigmoid(logit) > 0.5) == (yb > 0.5))[yb > 0.5].float().mean())
        print(f"  ep {ep}/{epochs} loss={loss_sum/n_b:.4f} pos_acc={pos_acc/max(1,n_b):.3f} "
              f"用时 {time.time()-t0:.0f}s", flush=True)

        if ep % VAL_EVERY == 0 or ep == epochs:
            probs, tv0, tv1 = score_val_cache(model, VAL_DIR, mean, std, device)
            true_dom = val_true_events(f["val_sessions"], "dominant")
            true_all = val_true_events(f["val_sessions"], None)
            best_thr, f1_dom = None, -1.0
            for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
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
                torch.save(model.state_dict(), config.MODEL_DIR / f"l3a_cnn_fold{fold}.pt")
                print(f"  >>> 保存 best (ep {ep})", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    t0 = time.time()
    report = {"folds": {}}
    for k in [int(x) for x in args.folds.split(",")]:
        print(f"===== L3a fold {k} =====", flush=True)
        report["folds"][str(k)] = train_fold(k, args.epochs)
    f1s = [v["f1_dominant"] for v in report["folds"].values()]
    report["mean_f1_dominant"] = float(np.mean(f1s))
    report["total_seconds"] = round(time.time() - t0, 1)
    (config.OUTPUT_DIR / "l3a_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"===== L3a 两折完成：F1(dominant)={f1s} 报告 outputs/l3a_report.json "
          f"（总用时 {time.time()-t0:.0f}s）", flush=True)

if __name__ == "__main__":
    main()
