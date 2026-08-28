# -*- coding: utf-8 -*-
"""拟合 L1 守门员 + L2 场景门控并评估（按受试者折划分：folds 0-3 训练，fold 4 评估）。
依赖 build_gate_cache.py 的缓存。产出 models/l1_gate.pkl、models/l2_scene_gate.pkl、
outputs/l1_l2_report.json。"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import splits
from src.models.l1_gate import L1Gate
from src.models.l2_scene_gate import L2SceneGate

CACHE_DIR = config.CACHE_DIR / "gate_features"

def _load_session_files(session_ids):
    files = [CACHE_DIR / f"{sid}.npz" for sid in session_ids if (CACHE_DIR / f"{sid}.npz").exists()]
    G8, G7, y, scene = [], [], [], []
    for f in files:
        d = np.load(f)
        G8.append(d["G8"]); G7.append(d["G7"])
        y.append(d["y"]); scene.append(d["scene"])
    return (np.vstack(G8), np.vstack(G7), np.concatenate(y), np.concatenate(scene))

def main():
    t0 = time.time()
    folds = splits.load_folds()
    tr_sessions = sorted({s for f in folds[:4] for s in f["train_sessions"]})
    va_sessions = sorted({s for f in folds[:4] for s in f["val_sessions"]})
    print(f"加载缓存: {len(tr_sessions)} 训练会话 / {len(va_sessions)} 验证会话...", flush=True)
    X8, X7, y, scene = _load_session_files(tr_sessions)
    X8v, X7v, yv, scenv = _load_session_files(va_sessions)
    print(f"训练窗口 {len(y)}（正 {int((y==1).sum())}），验证窗口 {len(yv)}", flush=True)

    # ---- L1：全部窗口，负样本 3:1 抽样 ----
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.RandomState(config.RANDOM_SEED)
    neg_sel = rng.choice(neg, size=min(len(neg), 3 * len(pos)), replace=False)
    sel = np.sort(np.r_[pos, neg_sel])
    l1 = L1Gate().fit(X8[sel], y[sel])
    wake = l1.wakeup(X8v)
    sens = wake[yv == 1].mean()
    spec = 1 - wake[yv == 0].mean()
    wake_pos_rate = (yv[wake] == 1).mean()
    print(f"L1 验证: 灵敏度={sens:.3f} 特异度={spec:.3f} 唤醒后正占比={wake_pos_rate:.3f} "
          f"体积={l1.size_bytes()}B", flush=True)

    # ---- L2：仅正样本（scene 真值） ----
    pos_tr = (y == 1) & (scene >= 0)
    pos_va = (yv == 1) & (scenv >= 0)
    l2 = L2SceneGate().fit(X7[pos_tr], scene[pos_tr])
    p = l2.predict_proba(X7v[pos_va])
    pred = (p >= 0.5).astype(int)
    acc = (pred == scenv[pos_va]).mean()
    conf_rate = (np.abs(p - 0.5) >= 0.1).mean()
    print(f"L2 验证: 正样本 {int(pos_va.sum())} 场景准确率={acc:.3f} 高置信占比={conf_rate:.3f}", flush=True)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    l1.save(config.MODEL_DIR / "l1_gate.pkl")
    l2.save(config.MODEL_DIR / "l2_scene_gate.pkl")
    report = {
        "l1": {"sensitivity": float(sens), "specificity": float(spec),
               "wake_positive_rate": float(wake_pos_rate),
               "size_bytes": l1.size_bytes(), "train_split": "folds 0-3 train", "eval_split": "folds 0-3 val"},
        "l2": {"scene_accuracy": float(acc), "confident_rate": float(conf_rate),
               "n_pos_val": int(pos_va.sum()), "train_split": "folds 0-3 train", "eval_split": "folds 0-3 val"},
    }
    (config.OUTPUT_DIR / "l1_l2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"models/l1_gate.pkl + models/l2_scene_gate.pkl + outputs/l1_l2_report.json "
          f"（用时 {time.time()-t0:.0f}s）")

if __name__ == "__main__":
    main()
