# -*- coding: utf-8 -*-
"""FD-I/FD-II 数据预处理与通道映射校验（阶段二，设计文档 v1.1）。

FD 数据（Shimmer3 @64Hz，6ch [ax,ay,az,gx,gy,gz] 双腕）：
  FD-I: 34 人全天（≥6h），bite 级标签 0/1/2（others/eating/drinking）
  FD-II: 27 人全天，仅餐时段 bite 标签 + meal boundaries
  （ReadMe 2.5：github.com/Pituohai/Eating-Speed-Dataset）

流程（设计文档锁定）：
  1) 通道映射校验：提取 1-2 个 session → 合幅值 + PSD → 确认重采样 10Hz 后
     Mean/Std/Mag 与目标数据集同数量级（通过后才全量）
  2) Episode 推导：bite=1（eating）连续簇（gap<180s 合并）→ meal Episode（+膨胀）
  3) 重采样 64→10Hz + 240s 窗切分 → 候选窗缓存（与 MM-Ranker 输入对齐）

运行：python scripts/prep_fd.py --check   # 通道校验（1-2 session）
      python scripts/prep_fd.py --build   # 全量缓存构建
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config

FD_ROOT = config.ROOT_DIR / "FDdatasets" / "doi-10.48804-CN8VBB" / "v1.0"
FD_SRC = 64.0          # FD 采样率
DST = 10.0             # 目标采样率
CTX_S = 120.0          # 候选窗半宽（±120s，与 build_candidate_windows 一致）
GRID_MS = 100          # 10Hz 网格步长
EPISODE_GAP_S = 180.0  # bite 簇合并间隔（与官方 Episode 定义一致）
EPISODE_PAD_S = 60.0   # Episode 边界膨胀（与 dilate_ms=60000 一致）
BITE_POS_RATIO = 0.02  # 窗内 eating 手势占比 >2% → 正样本候选


def load_fd(which="FD-I"):
    """加载 FD 双腕 pkl → (X_L, X_R, Y_L, Y_R)，每个是 list[np.ndarray]。"""
    base = FD_ROOT / which
    out = []
    for name in ("X_L", "X_R", "Y_L", "Y_R"):
        p = base / f"{name}.pkl"
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out


def _resample_10hz(x, y=None):
    """64Hz → 10Hz：interp 到 100ms 网格（与 build_candidate_windows 的 np.interp 同法）。"""
    n = len(x)
    t = np.arange(n) / FD_SRC * 1000.0                      # ms
    grid = np.arange(0, t[-1], GRID_MS)
    xr = np.stack([np.interp(grid, t, x[:, c]) for c in range(6)], axis=1)
    if y is None:
        return xr, grid
    yr = np.round(np.interp(grid, t, y.astype(np.float64))).astype(np.int32)
    return xr, yr, grid


def derive_episodes(y, gap_s=EPISODE_GAP_S, pad_s=EPISODE_PAD_S):
    """bite 标签（0/1/2）→ meal Episode [(s_idx, e_idx)]（样本索引，@64Hz）。
    逻辑：eating(1) 连续簇，簇间 gap < gap_s 合并，边界 ±pad_s 膨胀。"""
    eating = y == 1
    n = len(y)
    starts, ends = [], []
    i = 0
    while i < n:
        if eating[i]:
            j = i
            while j < n and eating[j]:
                j += 1
            starts.append(i); ends.append(j)
            i = j
        else:
            i += 1
    if not starts:
        return []
    # 合并：簇间 gap（秒）
    eps = []
    cs, ce = starts[0], ends[0]
    gap_n = int(gap_s * FD_SRC)
    pad_n = int(pad_s * FD_SRC)
    for s, e in zip(starts[1:], ends[1:]):
        if s - ce <= gap_n:
            ce = e
        else:
            eps.append((max(0, cs - pad_n), min(n, ce + pad_n)))
            cs, ce = s, e
    eps.append((max(0, cs - pad_n), min(n, ce + pad_n)))
    return eps


def channel_check(which="FD-I", sidx=0):
    """通道映射校验（设计文档 v1.1 修正②）：合幅值 + PSD + 重采样后特征分布对比。"""
    print(f"===== 通道校验 {which} subject {sidx} =====", flush=True)
    XL, XR, YL, YR = load_fd(which)
    x, y = XL[sidx], YL[sidx]
    print(f"  X shape {x.shape} dtype {x.dtype} | Y shape {y.shape} 标签分布 "
          f"{{0:{np.mean(y==0)*100:.0f}%, 1:{np.mean(y==1)*100:.0f}%, 2:{np.mean(y==2)*100:.0f}%}}",
          flush=True)
    print(f"  时长 {x.shape[0]/FD_SRC/3600:.1f}h @{FD_SRC}Hz", flush=True)
    # 通道统计（原始）
    for c, name in enumerate(("ax", "ay", "az", "gx", "gy", "gz")):
        v = x[:, c]
        print(f"  {name}: mean={v.mean():+.3f} std={v.std():.3f} mag={np.abs(v).mean():.3f}", flush=True)
    # 合幅值
    acc_mag = np.sqrt((x[:, :3] ** 2).sum(1))
    gyro_mag = np.sqrt((x[:, 3:] ** 2).sum(1))
    print(f"  acc 合幅值: mean={acc_mag.mean():.3f} (g) p95={np.percentile(acc_mag, 95):.3f} "
          f"| gyro 合幅值: mean={gyro_mag.mean():.3f} p95={np.percentile(gyro_mag, 95):.3f} (deg/s)",
          flush=True)
    # PSD（0.5-4Hz 与 4-16Hz 频带能量）
    for sig, name in ((acc_mag, "acc"), (gyro_mag, "gyro")):
        f, P = scipy.signal.welch(sig, fs=FD_SRC, nperseg=1024)
        for lo, hi in ((0.5, 4.0), (4.0, 16.0)):
            band = (f >= lo) & (f <= hi)
            print(f"  {name} PSD[{lo}-{hi}Hz]: {P[band].mean():.4g}", flush=True)
    # 重采样后分布（与目标数据集对比：本项目 env ~0.5-2Hz acc 包络）
    xr, yr, grid = _resample_10hz(x, y)
    print(f"  重采样 10Hz: {xr.shape} 标签正占比 {np.mean(yr == 1):.2%}", flush=True)
    eps = derive_episodes(y)
    print(f"  Episode 推导: {len(eps)} 个 meal episode", flush=True)
    if eps:
        durs = [(e - s) / FD_SRC / 60 for s, e in eps]
        print(f"    时长范围 {min(durs):.0f}-{max(durs):.0f} min（共 {sum(durs):.0f} min）", flush=True)
    return eps


def build_windows(which, norm_stats, use_neg=True):
    """构建 240s 候选窗（10Hz）：
    - 正样本：窗与 meal Episode 重叠 ≥50%（FD-I + FD-II）
    - 负样本：仅 FD-I（0=确认非餐），窗与所有 Episode 无重叠
    返回 (X, labels, metas)。"""
    XL, XR, YL, YR = load_fd(which)
    n_subj = len(XL)
    win_n = int(2 * CTX_S * DST)          # 2400
    step_n = int(120 * DST)               # 1200（120s 步）
    X, Y, M = [], [], []
    for i in range(n_subj):
        for hand, (x, y) in enumerate((("L", (XL[i], YL[i])), ("R", (XR[i], YR[i])))):
            x, y = y
            n = len(x)
            if n < win_n * (FD_SRC / DST):
                continue
            eps = derive_episodes(y)       # 64Hz 样本索引
            if not eps and not use_neg:
                continue
            xr, yr, _ = _resample_10hz(x, y)
            n_w = (len(xr) - win_n) // step_n + 1
            for b in range(n_w):
                w0 = b * step_n
                w1 = w0 + win_n
                win = xr[w0:w1]
                # 窗与 Episode 的重叠（10Hz 索引尺度换算回 64Hz 比例）
                is_pos = False
                for s, e in eps:
                    s10, e10 = int(s / (FD_SRC / DST)), int(e / (FD_SRC / DST))
                    ov = min(w1, e10) - max(w0, s10)
                    if ov > 0 and ov / win_n >= 0.5:
                        is_pos = True
                        break
                if is_pos:
                    Y.append(1.0)
                elif use_neg and not any(w1 > s10 and w0 < e10 for s, e in
                                         [(int(s / (FD_SRC / DST)), int(e / (FD_SRC / DST))) for s, e in eps]):
                    Y.append(0.0)
                else:
                    continue
                # z-score 归一化（FD 统计量）
                xn = (win - norm_stats["mean"]) / (norm_stats["std"] + 1e-6)
                X.append(xn.astype(np.float32))
                M.append({"which": which, "subj": i, "hand": hand, "w0": int(w0)})
        if (i + 1) % 8 == 0:
            print(f"  {which} {i+1}/{n_subj} subject，累计 {len(X)} 窗（正 {np.mean(Y) if Y else 0:.1%}）",
                  flush=True)
    return np.stack(X), np.array(Y, np.float32), M


def compute_norm_stats():
    """归一化统计量：FD-I 全量（双腕）6 通道 10Hz 信号的 mean/std。"""
    XL, XR, _, _ = load_fd("FD-I")
    sums, sumsq, n = np.zeros(6), np.zeros(6), 0
    for x in list(XL) + list(XR):
        xr, _ = _resample_10hz(x)
        sums += xr.sum(0)
        sumsq += (xr ** 2).sum(0)
        n += len(xr)
    mean = sums / n
    std = np.sqrt(sumsq / n - mean ** 2)
    return {"mean": mean, "std": std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="通道映射校验（1-2 session）")
    ap.add_argument("--build", action="store_true", help="全量候选窗缓存构建")
    args = ap.parse_args()
    if args.check:
        channel_check("FD-I", 0)
        channel_check("FD-I", 5)
        channel_check("FD-II", 0)
    elif args.build:
        t0 = time.time()
        stats = compute_norm_stats()
        print(f"归一化统计量: mean={np.round(stats['mean'], 3)} std={np.round(stats['std'], 3)}",
              flush=True)
        out_dir = config.CACHE_DIR / "fd_windows"
        out_dir.mkdir(parents=True, exist_ok=True)
        # FD-I：正+负；FD-II：仅正
        Xi, Yi, Mi = build_windows("FD-I", stats, use_neg=True)
        Xp, Yp, Mp = build_windows("FD-II", stats, use_neg=False)
        X = np.concatenate([Xi, Xp]); Y = np.concatenate([Yi, Yp]); M = Mi + Mp
        import json as _json
        np.savez(out_dir / "fd_pretrain.npz",
                 imu=X, label=Y,
                 meta=np.array([_json.dumps(m).encode() for m in M]),
                 norm_mean=stats["mean"], norm_std=stats["std"])
        print(f"→ cache/fd_windows/fd_pretrain.npz: {len(X)} 窗（正 {Y.mean()*100:.1f}%）"
              f" 用时 {time.time()-t0:.0f}s", flush=True)
    else:
        print("用法：--check 校验 | --build 全量构建", flush=True)


if __name__ == "__main__":
    main()
