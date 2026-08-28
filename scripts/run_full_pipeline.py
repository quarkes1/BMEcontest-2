# -*- coding: utf-8 -*-
"""W3-T4 全链路 5 折评估 + 消融矩阵：
L1 唤醒 → L2 分流 → L3a(5s/1s 步长) + L3b(30s/10s 步长，上采样到 1s 网格)
→ 动态融合（手工 alpha + 30s EMA）→ HMM Viterbi → 事件 → 竞赛口径评估。
消融：L3a-only / L3b-only / 融合(无HMM) / 融合+HMM，按场景与总体。
依赖：models/l3a_cnn_fold{k}.pt、models/l3b_ppgnn_fold{k}.pt（5 折齐备）。
运行：conda activate bme && python scripts/run_full_pipeline.py --folds 0,1,2,3,4
产物：outputs/full_pipeline_report.json"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.eval.metrics import compute_metrics
from src.infer.events import windows_to_events
from src.models.fusion import handcrafted_alpha, ema_smooth, fuse
from src.models.hmm_decode import viterbi, decode_events
from src.models.l3a_cnn import L3aCNN, N_CHANNELS as L3A_CH
from src.models.l3b_ppgnn import (L3bPPGNN, N_PPG_CHANNELS, PPG_WINDOW_ROWS,
                                  HRV_DIMS, SEQ_LEN)

A_VAL = config.CACHE_DIR / "l3a_val_raw"
B_VAL = config.CACHE_DIR / "l3b_val_raw"
THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)


# ------------------------------------------------------------------ 验证缓存（从 train 脚本复用构建逻辑）
def _val_a(args):
    from src.data.loader import load_session, detect_binary, _find_collect_data
    from src.data.windows import iter_window_labels
    from src.models.l3a_cnn import build_raw_channels
    session_id, out_dir = args
    out = out_dir / f"{session_id}.npz"
    if out.exists():
        return ("skip", session_id)
    try:
        if detect_binary(_find_collect_data(str(config.SENSOR_DIR / session_id))):
            return ("binary", session_id)
        s = load_session(session_id)
        Xs, t0s, t1s = [], [], []
        for w in iter_window_labels(s, []):
            Xs.append(build_raw_channels(s, w["start_row"], w["end_row"]))
            t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
        if Xs:
            np.savez(out, X=np.stack(Xs), t0=np.array(t0s, dtype=np.int64),
                     t1=np.array(t1s, dtype=np.int64))
        return ("ok", session_id)
    except Exception as e:
        return ("error", f"{session_id}: {type(e).__name__}: {e}")

def _val_b(args):
    from src.data.loader import load_session, detect_binary, _find_collect_data
    from src.data.windows import iter_window_labels
    from src.models.l3b_ppgnn import build_ppg_window
    session_id, out_dir = args
    out = out_dir / f"{session_id}.npz"
    if out.exists():
        return ("skip", session_id)
    try:
        if detect_binary(_find_collect_data(str(config.SENSOR_DIR / session_id))):
            return ("binary", session_id)
        s = load_session(session_id)
        Xs, hs, t0s, t1s = [], [], [], []
        for w in iter_window_labels(s, [], window_rows=3150, stride_rows=1050):
            X, h = build_ppg_window(s, w["start_row"], w["end_row"])
            Xs.append(X); hs.append(h)
            t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
        if Xs:
            np.savez(out, X=np.stack(Xs), hrv=np.stack(hs),
                     t0=np.array(t0s, dtype=np.int64), t1=np.array(t1s, dtype=np.int64))
        return ("ok", session_id)
    except Exception as e:
        return ("error", f"{session_id}: {type(e).__name__}: {e}")

def build_val_caches(val_sessions):
    for out_dir, fn in ((A_VAL, _val_a), (B_VAL, _val_b)):
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=8) as ex:
            for i, (status, info) in enumerate(ex.map(fn, [(s, out_dir) for s in val_sessions], chunksize=4)):
                if (i + 1) % 50 == 0:
                    print(f"  {out_dir.name} {i+1}/{len(val_sessions)} {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ 打分
def score_l3a(fold, device):
    model = L3aCNN(5).to(device).eval()
    model.load_state_dict(torch.load(config.MODEL_DIR / f"l3a_cnn_fold{fold}.pt", weights_only=True))
    stats = json.loads((config.CACHE_DIR / "l3a_raw" / f"fold{fold}" / "stats.json").read_text(encoding="utf-8"))
    mean = np.array(stats["mean"], dtype=np.float32).reshape(1, L3A_CH, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(1, L3A_CH, 1) + 1e-6
    probs, t0s, t1s = [], [], []
    with torch.no_grad():
        for f in sorted(A_VAL.glob("*.npz")):
            d = np.load(f)
            X = (d["X"].astype(np.float32) - mean) / std
            for b in range(0, len(X), 512):
                xb = torch.from_numpy(X[b:b + 512]).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logit, _ = model(xb)
                probs.append(torch.sigmoid(logit).float().cpu().numpy())
            t0s.append(d["t0"]); t1s.append(d["t1"])
    return np.concatenate(probs), np.concatenate(t0s), np.concatenate(t1s)

def score_l3b(fold, device):
    model = L3bPPGNN().to(device).eval()
    model.load_state_dict(torch.load(config.MODEL_DIR / f"l3b_ppgnn_fold{fold}.pt", weights_only=True))
    out_dir = config.CACHE_DIR / "l3b_raw" / f"fold{fold}"
    hs = np.concatenate([np.load(f)["hrv"] for f in sorted(out_dir.glob("*.npz"))]).astype(np.float32)
    mean_h = hs.mean(axis=0); std_h = hs.std(axis=0) + 1e-6
    probs, t0s, t1s, hrv_all = [], [], [], []
    with torch.no_grad():
        for f in sorted(B_VAL.glob("*.npz")):
            d = np.load(f)
            X = d["X"].astype(np.float32)
            h = ((d["hrv"].astype(np.float32) - mean_h) / std_h)
            n = len(X)
            sp = np.zeros(n, dtype=np.float32)
            for s0 in range(0, n, 128):
                s1 = min(s0 + 128, n)
                xs = np.zeros((s1 - s0, SEQ_LEN, N_PPG_CHANNELS, PPG_WINDOW_ROWS), dtype=np.float32)
                hb2 = np.zeros((s1 - s0, SEQ_LEN, HRV_DIMS), dtype=np.float32)
                for j, i in enumerate(range(s0, s1)):
                    lo = max(0, i - SEQ_LEN + 1)
                    off = SEQ_LEN - (i - lo + 1)
                    xs[j, off:] = X[lo:i + 1]
                    hb2[j, off:] = h[lo:i + 1]
                xb = torch.from_numpy(xs).to(device)
                hb = torch.from_numpy(hb2).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(xb, hb)
                sp[s0:s1] = torch.sigmoid(logits[:, -1]).float().cpu().numpy()
            probs.append(sp)
            t0s.append(d["t0"]); t1s.append(d["t1"])
            hrv_all.append(d["hrv"].astype(np.float32))
    return (np.concatenate(probs), np.concatenate(t0s), np.concatenate(t1s), np.concatenate(hrv_all))


# ------------------------------------------------------------------ 融合
def fuse_upsample(p_a, p_b, hrv_b):
    """L3b 10s 网格上采样到 1s（重复 10 次）后与 L3a 对齐融合。
    alpha 特征：hrv_b[4]=SNR、hrv_b[3]=灌注指数、hrv_b[6]=陀螺活动度。"""
    p_b_up = np.repeat(p_b, 10)
    hrv_up = np.repeat(hrv_b, 10, axis=0)
    n = min(len(p_a), len(p_b_up))
    p_a, p_b_up, hrv_up = p_a[:n], p_b_up[:n], hrv_up[:n]
    alpha = np.array([handcrafted_alpha(s, pi, act)
                      for s, pi, act in zip(hrv_up[:, 4], hrv_up[:, 3], hrv_up[:, 6])], dtype=np.float32)
    # 30s EMA（流式）
    a_sm = np.empty_like(alpha)
    cur = 0.5
    for i in range(n):
        cur = ema_smooth(alpha[i], cur, step_s=1.0)
        a_sm[i] = cur
    return fuse(p_a, p_b_up, a_sm), a_sm


def eval_events(evs, true_events):
    return compute_metrics(evs, true_events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    folds = splits.load_folds()
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index().set_index("session_id")
    report = {"ablation": {}, "folds": {}}
    t_start = time.time()
    for k in [int(x) for x in args.folds.split(",")]:
        f = folds[k]
        print(f"===== fold {k} =====", flush=True)
        shutil.rmtree(A_VAL, ignore_errors=True)
        shutil.rmtree(B_VAL, ignore_errors=True)
        build_val_caches(f["val_sessions"])
        p_a, ta0, ta1 = score_l3a(k, device)
        p_b, tb0, tb1, hrv_b = score_l3b(k, device)
        p_f, alpha = fuse_upsample(p_a, p_b, hrv_b)
        # 真值（场景）
        ext_set = {index.loc[s, "externalid"] for s in f["val_sessions"] if s in index.index}
        true_all = [(m["before"], m["after"]) for e in ext_set for m in meal_meta.get(e, [])]
        true_dom = [(m["before"], m["after"]) for e in ext_set for m in meal_meta.get(e, []) if m["scene"] == "dominant"]
        true_non = [(m["before"], m["after"]) for e in ext_set for m in meal_meta.get(e, []) if m["scene"] == "nondominant"]

        rows = {}
        # 消融：单分支与融合（平滑后处理，阈值按总体 F1 选取）
        for name, (p, t0, t1) in (("l3a", (p_a, ta0, ta1)),
                                  ("l3b", (p_b, tb0, tb1)),
                                  ("fused", (p_f, ta0, ta1))):
            smooth = 31 if name != "l3b" else 3
            best = None
            for thr in THRESHOLDS:
                m = compute_metrics(windows_to_events(p, t0, t1, thr, smooth_win=smooth), true_all)
                if best is None or m["f1"] > best[1]["f1"]:
                    best = (thr, m)
            rows[name] = {"threshold": best[0], "overall": best[1],
                          "dominant": compute_metrics(
                              windows_to_events(p, t0, t1, best[0], smooth_win=smooth), true_dom),
                          "nondominant": compute_metrics(
                              windows_to_events(p, t0, t1, best[0], smooth_win=smooth), true_non)}
        # 融合 + HMM（按会话文件切分做 Viterbi，保持会话内时序连续）
        evs_hmm = []
        off = 0
        for f_v in sorted(A_VAL.glob("*.npz")):
            d = np.load(f_v)
            n = len(d["t0"])
            st = viterbi(p_f[off:off + n])
            evs_hmm.extend(decode_events(st, d["t0"], d["t1"]))
            off += n
        m_hmm = compute_metrics(evs_hmm, true_all)
        rows["fused_hmm"] = {"overall": m_hmm,
                             "dominant": compute_metrics(evs_hmm, true_dom),
                             "nondominant": compute_metrics(evs_hmm, true_non)}
        print(f"  l3a F1={rows['l3a']['overall']['f1']:.3f} "
              f"l3b F1={rows['l3b']['overall']['f1']:.3f} "
              f"fused F1={rows['fused']['overall']['f1']:.3f} "
              f"fused_hmm F1={rows['fused_hmm']['overall']['f1']:.3f}", flush=True)
        report["folds"][str(k)] = {name: {kk: vv for kk, vv in row.items()}
                                   for name, row in rows.items()}
        report["ablation"][str(k)] = {name: row["overall"]["f1"] for name, row in rows.items()}
    # 汇总消融矩阵
    sums = {}
    for k, ab in report["ablation"].items():
        for name, f1 in ab.items():
            sums.setdefault(name, []).append(f1)
    report["ablation_mean"] = {name: float(np.mean(v)) for name, v in sums.items()}
    report["total_seconds"] = round(time.time() - t_start, 1)
    (config.OUTPUT_DIR / "full_pipeline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("===== 消融均值 =====", flush=True)
    for name, v in report["ablation_mean"].items():
        print(f"  {name}: {v:.3f}", flush=True)

if __name__ == "__main__":
    main()
