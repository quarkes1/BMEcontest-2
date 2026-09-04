# -*- coding: utf-8 -*-
"""进食事件推理入口（竞赛可执行文件）。

对给定的智能手表传感器会话目录（HUAWEI Research 格式，含 collect_data*.txt）
输出检测到的进食事件列表。本文件为自包含推理管线——不依赖训练期拟合的
LightGBM 排序器/门控，也不依赖 FD 数据集（预训练权重随包发布）：

    提案（规则：活动包络 × 内置时刻先验的多阈值连通域）
      → 候选窗（240s，10Hz）
      → 深度排序（5 折 MM-Ranker 权重 bagging 平均；z-score 归一化统计量随权重内置）
      → 阈值解码 + 形态学后处理 → 事件

用法：
  python predict.py --input <会话目录或含目录列表的 txt> --output out.json
                   [--tau 0.3] [--merge-gap 120] [--min-dur 120] [--dilation 60]
                   [--device cpu|cuda] [--weights-dir ../models] [--norm-ckpt ../checkpoints/fd_pretrained_s1.pt]

输出 JSON：{ "<会话目录名>": [ [start_ms, end_ms], ... ], ... }
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal

# ---- 依赖注入：允许从任意工作目录运行（dist 布局为 ./predict.py + ./src/ + ./models/ + ./checkpoints/） ----
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if (_p / "src").is_dir():
        sys.path.insert(0, str(_p))
        break

# 内置全局 24h 时刻先验（训练数据全部餐次 before 时刻的高斯平滑直方图，0-23h）
GLOBAL_PRIOR = np.array(
    [0.174, 0.278, 0.546, 0.92, 0.889, 0.496, 0.187, 0.141, 0.408, 0.863,
     1.0, 0.681, 0.368, 0.216, 0.127, 0.073, 0.037, 0.012, 0.002, 0.0,
     0.0, 0.0, 0.0, 0.0], np.float32)

FS_RAW = 105.0        # 原始 IMU 行率（估计）
FS10 = 10.0           # 重采样
WIN_S, ST_S = 5.0, 1.0     # 包络窗
CTX_S = 120.0         # 候选窗半宽
GRID_MS = 100
WINDOW_MS = 525_000
SMOOTH = 31
PROP_PCT = (75, 82, 88, 92, 95)
PROP_GAP = (30, 60)
PROP_DUR = (5, 10)
IOU_MERGE = 0.6


def load_session_tsv(txt_path):
    """最小 TSV 解析：acc/gyro 6 通道 + 有效时间戳（与 src/data/loader 同构）。"""
    acc_x, acc_y, acc_z, gx, gy, gz = [], [], [], [], [], []
    t_acc, imu_valid = [], []
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        assert "ACC_TIME" in header, f"bad header: {txt_path}"
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 53:
                continue
            try:
                at, pt, gt = int(parts[0]), int(parts[1]), int(parts[2])
                vals = list(map(float, parts[3:53]))
            except ValueError:
                continue
            a = vals[44:47]   # N_PPG=44：PPG 44 + acc 3 + gyro 3（与 src/data/loader 一致）
            g = vals[47:50]
            imu_ok = (at > 0 or gt > 0) and not all(v == 0 for v in a)
            acc_x.append(a[0]); acc_y.append(a[1]); acc_z.append(a[2])
            gx.append(g[0]); gy.append(g[1]); gz.append(g[2])
            t_acc.append(at if imu_ok else -1)
            imu_valid.append(imu_ok)
    acc = np.array([acc_x, acc_y, acc_z], np.float32)
    gyro = np.array([gx, gy, gz], np.float32)
    t = np.array(t_acc, np.int64)
    valid = np.array(imu_valid, bool)
    valid_ts = t[valid]
    fs = FS_RAW
    if len(valid_ts) > 10:
        span = (valid_ts.max() - valid_ts.min()) / 1000.0
        if span > 60:
            fs = len(valid_ts) / span
    return acc, gyro, t, valid, fs


def band_env(sig, fs, band=(0.5, 2.0)):
    """1s 网格带通包络（5s 窗/1s 步，同 validate_baselines）。"""
    win, st = int(WIN_S * fs), int(ST_S * fs)
    n = sig.shape[1]
    n_w = (n - win) // st + 1
    if n_w <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.int64)
    sos = scipy.signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    out = np.empty(n_w, np.float32)
    for b0 in range(0, n_w, 4000):
        b1 = min(b0 + 4000, n_w)
        m = b1 - b0
        idx = b0 * st + np.arange(m)[:, None] * st + np.arange(win)[None, :]
        seg = sig[:, idx]
        la = seg - np.median(seg, axis=2, keepdims=True)
        lam = np.linalg.norm(la, axis=0)
        e = scipy.signal.sosfiltfilt(sos, lam, axis=1)
        out[b0:b1] = np.abs(e).mean(axis=1)
    return out


def windows_to_events(score, t0, t0_end, thr, merge_gap_s=30, min_dur_s=5, smooth_win=31):
    """阈值连通域 → 事件（src/infer/events 同构实现）。"""
    if smooth_win > 1 and len(score) >= smooth_win:
        k = np.ones(smooth_win) / smooth_win
        s = np.convolve(score, k, mode="same")
    else:
        s = score
    on = s >= thr
    evs = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            evs.append((int(t0[i]), int(t0[j - 1]) + 1000))
            i = j
        else:
            i += 1
    return evs


def make_proposals(env, t0, prior):
    """活动提案（多阈值连通域并集）。"""
    h = (t0 / 3.6e6) % 24
    score = env * prior[np.clip(h.astype(int), 0, 23)]
    act = []
    for pct in PROP_PCT:
        for gap in PROP_GAP:
            for dur in PROP_DUR:
                evs = windows_to_events(score, t0, t0 + WINDOW_MS,
                                        float(np.percentile(score, pct)),
                                        merge_gap_s=gap, min_dur_s=dur, smooth_win=SMOOTH)
                act.extend(evs)
    # 合并 IoU>0.6
    if not act:
        return []
    act = sorted(act)
    merged = [list(act[0])]
    for s, e in act[1:]:
        a = merged[-1]
        inter = min(a[1], e) - max(a[0], s)
        union = max(a[1], e) - min(a[0], s)
        if inter > 0 and inter / union > IOU_MERGE:
            a[1] = max(a[1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def extract_window(acc, gyro, t, valid, fs, c_s, c_e, norm_mean, norm_std):
    """候选窗 240s@10Hz 6 通道 + z-score 归一化（与训练侧同构）。"""
    c_mid = (c_s + c_e) // 2
    ws = c_mid - int(CTX_S * 1000)
    we = c_mid + int(CTX_S * 1000)
    n_step = int((we - ws) / GRID_MS)
    grid = ws + np.arange(n_step) * GRID_MS
    t_v = t[valid]
    imu = np.zeros((n_step, 6), np.float32)
    for i, ch in enumerate(np.concatenate([acc, gyro])[:, valid]):
        imu[:, i] = np.interp(grid, t_v, ch.astype(np.float64))
    cov = ((grid >= t_v[0]) & (grid <= t_v[-1])).mean()
    if cov < 0.05:
        return None
    return (imu - norm_mean) / (norm_std + 1e-6)


def postprocess(evs, merge_gap_s, min_dur_s, dilation_s):
    """形态学后处理：合并相邻 → 过滤孤立短段 → 边界膨胀。"""
    if not evs:
        return []
    merged = []
    for s, e in sorted(evs):
        if merged and s - merged[-1][1] <= merge_gap_s * 1000:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if (e - s) / 1000.0 >= min_dur_s:
            out.append([max(0, int(s - dilation_s * 1000)), int(e + dilation_s * 1000)])
    return out


def predict_session(sid_dir, models, device, tau, merge_gap, min_dur, dilation):
    """单会话推理 → 事件列表。models: [{"model": MMRanker, "norm_mean": arr, "norm_std": arr}]"""
    import torch
    import os
    from src.models.ranker import MMRanker
    names = [n for n in os.listdir(str(sid_dir)) if n.startswith("collect_data")]
    if not names:
        return []
    txt = sid_dir / sorted(names)[0]
    acc, gyro, t, valid, fs = load_session_tsv(str(txt))
    if not valid.any() or acc.shape[1] < int(WIN_S * fs):
        return []
    env = band_env(acc, fs)
    if len(env) < 60:
        return []
    t_v = t[valid]
    start_epoch = int(t_v[0])
    t0 = start_epoch + np.arange(len(env), dtype=np.int64) * 1000
    act = make_proposals(env, t0, GLOBAL_PRIOR)
    if not act:
        return []
    # 归一化统计量（与训练侧一致，来自 FD 预训练权重）
    norm_mean = models[0]["norm_mean"]
    norm_std = models[0]["norm_std"]
    metas = []
    for c_s, c_e in act:
        win = extract_window(acc, gyro, t, valid, fs, c_s, c_e, norm_mean, norm_std)
        metas.append(win)
    scores = np.full(len(act), np.nan, np.float32)
    z66 = torch.zeros(1, 48, 66, device=device)
    z2 = torch.zeros(1, 48, 2, device=device)
    z3 = torch.zeros(1, 3, device=device)
    for win_i, win in enumerate(metas):
        if win is None:
            continue
        x = torch.from_numpy(win)[None].to(device)
        logits = []
        with torch.no_grad():
            for m in models:
                lg = m["model"](x, z66, z2, z3)
                logits.append(lg.cpu())
        scores[win_i] = float(torch.sigmoid(torch.stack(logits)).mean())
    sel = np.where(~np.isnan(scores) & (scores >= tau))[0]
    evs = [(act[i][0], act[i][1]) for i in sel]
    return postprocess(evs, merge_gap, min_dur, dilation)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="会话目录（含 collect_data*.txt），或含多个目录路径的 txt/换行列表")
    ap.add_argument("--output", default="predictions.json")
    ap.add_argument("--tau", type=float, default=0.30, help="深度分阈值（默认 0.30）")
    ap.add_argument("--merge-gap", type=float, default=120.0, help="事件合并间隔秒（默认 120）")
    ap.add_argument("--min-dur", type=float, default=120.0, help="最短事件时长秒（默认 120）")
    ap.add_argument("--dilation", type=float, default=60.0, help="边界膨胀秒（默认 60）")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--weights-dir", default=str(_HERE / "models"),
                    help="5 折微调权重目录（mm_ranker_fold{k}.pt）")
    ap.add_argument("--norm-ckpt", default=str(_HERE / "checkpoints" / "fd_pretrained_s1.pt"),
                    help="预训练权重（提供 z-score 归一化统计量）")
    args = ap.parse_args()

    import torch
    from src.models.ranker import MMRanker
    dev = torch.device(args.device)

    # 加载归一化统计量
    norm_mean = np.zeros(6, np.float32)
    norm_std = np.ones(6, np.float32)
    np_path = Path(args.norm_ckpt)
    if np_path.exists():
        ck = torch.load(np_path, map_location="cpu", weights_only=False)
        if "norm_mean" in ck:
            norm_mean = np.asarray(ck["norm_mean"], np.float32)[:6]
            norm_std = np.asarray(ck["norm_std"], np.float32)[:6]

    # 加载 5 折模型
    models = []
    for k in range(5):
        p = Path(args.weights_dir) / f"mm_ranker_fold{k}.pt"
        if not p.exists():
            print(f"[warn] 缺权重 {p}，跳过", flush=True)
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)
        m = MMRanker(n_imu=6, use_ppg=False)
        m.load_state_dict(sd)
        m.to(dev).eval()
        models.append({"model": m, "norm_mean": norm_mean, "norm_std": norm_std})
    if not models:
        raise SystemExit("没有可用模型权重")
    print(f"模型 {len(models)} 个（5 折 bagging）| 设备 {args.device} | τ={args.tau}", flush=True)

    # 会话列表
    inp = Path(args.input)
    if inp.is_dir():
        sids = [inp]
    else:
        sids = [Path(x.strip()) for x in inp.read_text(encoding="utf-8").splitlines() if x.strip()]
    results = {}
    t_start = time.time()
    for i, sd in enumerate(sids):
        evs = predict_session(sd, models, dev, args.tau, args.merge_gap, args.min_dur,
                              args.dilation)
        results[sd.name] = evs
        print(f"  [{i+1}/{len(sids)}] {sd.name}: {len(evs)} 事件 ({time.time()-t_start:.0f}s)",
              flush=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    print(f"→ {args.output}（{len(results)} 会话）", flush=True)


if __name__ == "__main__":
    main()
