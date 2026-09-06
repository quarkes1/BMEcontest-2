# -*- coding: utf-8 -*-
"""滑窗管线推理（部署版，纯 CPU，自包含）。

会话 TSV → 240s/15s 全覆盖滑窗 → 62+1 特征（ACC 统计 + 1s 包络时间特征 + 时刻先验）
→ 5 折 HGB bag 概率 → 密度聚合候选（600s ≥10 越阈 ≥0.8 覆盖）→ 37 特征复核 → 事件。

用法：python predict_slide.py --input <会话目录或列表txt> --output out.json
产物模型：slide_models/（wmodel_fold{k}.joblib + verifier.joblib + config.json）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal

_HERE = Path(__file__).resolve().parent

WIN_MS = 240_000
STRIDE_MS = 15_000
NEG_BUFFER_S = 300
COV_MIN = 0.80
BRIDGE_MS = 60_000
DENSITY_MS = 600_000
MIN_POS = 10
MERGE_MS = 120_000
CTX_MS = 1_200_000
WIN_THR = 0.28838
GLOBAL_PRIOR = np.array(
    [0.174, 0.278, 0.546, 0.92, 0.889, 0.496, 0.187, 0.141, 0.408, 0.863,
     1.0, 0.681, 0.368, 0.216, 0.127, 0.073, 0.037, 0.012, 0.002, 0.0,
     0.0, 0.0, 0.0, 0.0], np.float32)


# ---------------- 62 特征（与训练侧 slide_features.extract_62 同构） ----------------
def _stats(x, names):
    x = x.astype(np.float64)
    p = np.percentile(x, [10, 25, 75, 90])
    med = float(np.median(x))
    full = {"mean": float(x.mean()), "std": float(x.std()), "median": med,
            "p10": p[0], "p25": p[1], "p75": p[2], "p90": p[3],
            "p90_p10": p[3] - p[0], "iqr": p[2] - p[1],
            "rms": float(np.sqrt((x ** 2).mean())),
            "mad": float(np.median(np.abs(x - med)))}
    return [full[n] for n in names]


STATS = ("mean", "std", "median", "p10", "p25", "p75", "p90", "p90_p10", "iqr", "rms", "mad")
POSE_STATS = ("mean", "median", "p10", "p25", "p75", "p90", "rms")


def extract_62(seg, fs):
    """seg: (3, n) raw 行 → 62 特征 + 内部时间特征（63 列含时刻由调用方补）。"""
    mag = np.linalg.norm(seg, axis=0)
    x, y, z = seg[0], seg[1], seg[2]
    jerk = np.linalg.norm(np.diff(seg, axis=1), axis=0) * fs
    out = []
    for ch in (y, z, mag):
        out.extend(_stats(ch, STATS))
    out.extend(_stats(x, POSE_STATS))
    out.extend(_stats(jerk, STATS))
    out.extend(_temporal(seg, fs))
    return out


def _temporal(seg, fs):
    n_samp = int(fs)
    usable = int(seg.shape[1] // n_samp) * n_samp
    if usable < 30 * n_samp:
        return np.full(11, np.nan)
    blk = seg[:, :usable].reshape(3, -1, n_samp)
    valid = (~np.isnan(blk)).all(axis=(0, 2))
    ar = np.percentile(blk, 90, axis=2) - np.percentile(blk, 10, axis=2)
    env = np.linalg.norm(ar, axis=0)
    env[~valid] = np.nan
    if (~np.isnan(env)).sum() < 0.8 * len(env):
        return np.full(11, np.nan)
    idx = np.arange(len(env))[~np.isnan(env)]
    env = np.interp(np.arange(len(env)), idx, env[~np.isnan(env)])
    n = len(env)
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)))
    scale = max(mad, 1e-6)
    norm = (env - med) / scale
    sm = np.convolve(norm, np.ones(3) / 3, mode="same")
    active = float((env > med + 2.0 * scale).mean())
    peaks, _ = scipy.signal.find_peaks(sm, prominence=1.0, distance=2)
    rate = len(peaks) / (n / 60.0)
    if len(peaks) >= 2:
        gaps = np.diff(peaks)
        gmed, gcv = float(np.median(gaps)), float(gaps.std() / (gaps.mean() + 1e-9))
    else:
        gmed = float(n) if len(peaks) == 1 else np.nan
        gcv = np.nan
    c = env - env.mean()
    pw = np.abs(np.fft.rfft(c * np.hanning(n))) ** 2
    if pw.size > 1:
        pw_nz = pw[1:]
        ent = -float((pw_nz / pw_nz.sum() * np.log(pw_nz / pw_nz.sum() + 1e-12)).sum()) / np.log(pw_nz.size)
        dom = float(np.argmax(pw_nz) / n)
        tot = pw_nz.sum() + 1e-12
        def bp(a, b):
            idx2 = np.arange(1, n // 2 + 1) / n
            return float(pw_nz[(idx2 >= a) & (idx2 < b)].sum() / tot)
        slow, mid, fast = bp(0.02, 0.08), bp(0.08, 0.20), bp(0.20, 0.45)
    else:
        ent = dom = slow = mid = fast = np.nan
    return [float(np.median(env)), float(np.percentile(env, 90)), active, rate,
            gmed, gcv, ent, dom, slow, mid, fast]


# ---------------- TSV 读取 ----------------
def load_session_tsv(txt_path):
    acc = [[], [], []]; gyro = [[], [], []]
    t_acc = []; valid = []
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        assert "ACC_TIME" in header, f"bad header: {txt_path}"
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 53:
                continue
            try:
                at = int(parts[0]); gt = int(parts[2])
                vals = list(map(float, parts[3:53]))
            except ValueError:
                continue
            a = vals[44:47]; g = vals[47:50]
            ok = (at > 0 or gt > 0) and not all(v == 0 for v in a)
            for c in range(3):
                acc[c].append(a[c]); gyro[c].append(g[c])
            t_acc.append(at if ok else -1)
            valid.append(ok)
    acc = np.array(acc, np.float32); gyro = np.array(gyro, np.float32)
    t = np.array(t_acc, np.int64); valid = np.array(valid, bool)
    return acc, gyro, t, valid


def session_features(sid_dir, models):
    """会话 → (窗行: (ws, we, feat63)) 列表（feat = 62 + 时刻）。"""
    names = [n for n in sorted(sid_dir.iterdir()) if n.name.startswith("collect_data")]
    if not names:
        return []
    txt = sid_dir / names[0]
    try:
        acc, gyro, t, valid = load_session_tsv(str(txt))
    except Exception:
        return []
    if not valid.any():
        return []
    t_v = t[valid].astype(np.int64)
    t_start, t_end = int(t_v[0]), int(t_v[-1])
    if t_end - t_start < WIN_MS:
        return []
    fs = 105.0
    if len(t_v) > 10:
        span = (t_v[-1] - t_v[0]) / 1000.0
        if span > 60:
            fs = len(t_v) / span
    if fs < 10 or fs > 500:   # 稀疏/异常时间戳保护（extract_62 分块需要 fs≥1）
        fs = 105.0
    acc_v = acc[:, valid].astype(np.float32)
    gyro_v = gyro[:, valid].astype(np.float32)
    ws_all = t_start + np.arange(0, t_end - t_start - WIN_MS + 1, STRIDE_MS, dtype=np.int64)
    n_full = WIN_MS / 1000.0 * fs
    rows = []
    for ws in ws_all:
        we = ws + WIN_MS
        lo = int(np.searchsorted(t_v, ws)); hi = int(np.searchsorted(t_v, we))
        if hi - lo < COV_MIN * n_full:
            continue
        seg = np.concatenate([acc_v, gyro_v])[:, lo:hi]   # 6ch（acc+gyro 同构时间轴）
        f = extract_62(seg[:3], fs)                        # 62 特征用 acc 三轴
        if len(f) != 62:
            continue
        hh = int((ws / 3.6e6) % 24)
        rows.append((int(ws), int(we), np.array(f + [float(GLOBAL_PRIOR[hh])], np.float32)))
    return rows


def predict_session(rows, models, verifier, thr_ver):
    """窗特征行 → 事件列表。"""
    if len(rows) < 10:
        return []
    X = np.array([r[2] for r in rows], np.float32)
    probs = np.mean([m["model"].predict_proba(m["imp"].transform(X))[:, 1] for m in models], 0)
    # 密度聚合：以连续窗（gap ≤60s 桥接）分段后 600s 密度
    starts = np.array([r[0] for r in rows], np.int64)
    cands = []
    segs = []
    cur = []
    for i in range(len(rows)):
        if cur and starts[i] - starts[cur[-1]] > STRIDE_MS + BRIDGE_MS:
            segs.append(cur); cur = []
        cur.append(i)
    segs.append(cur)
    for seg_i in segs:
        if len(seg_i) < MIN_POS:
            continue
        ss = starts[seg_i]
        pp = probs[seg_i]
        oo = np.ones(len(seg_i))
        ds = int(round(DENSITY_MS / STRIDE_MS))
        cnt = np.convolve((pp >= WIN_THR).astype(np.int64), np.ones(ds, np.int64), "same")
        cov = np.convolve(oo, np.ones(ds) / ds, "same")
        dense = (cnt >= MIN_POS) & (cov >= COV_MIN)
        i = 0
        while i < len(seg_i):
            if dense[i]:
                j = i
                while j < len(seg_i) and dense[j]:
                    j += 1
                pos_idx = np.where(pp[i:j] >= WIN_THR)[0]
                if len(pos_idx) == 0:
                    i = j; continue
                a, b = i + pos_idx[0], i + pos_idx[-1]
                ev = [int(ss[a]), int(ss[b] + WIN_MS), pp[a:b + 1].copy(),
                      (b - a + 1) * STRIDE_MS / 1000.0]   # dur = 越阈窗跨度（与训练 density_candidates 一致）
                if cands and ev[0] - cands[-1][1] <= MERGE_MS:
                    cands[-1][1] = max(cands[-1][1], ev[1])
                    cands[-1][2] = np.concatenate([cands[-1][2], ev[2]])
                else:
                    cands.append(ev)
                i = j
            else:
                i += 1
    if not cands:
        return []
    # 复核特征（33 + TCN 0 列 + 时刻 2 = 37——与训练 no_tcn 版一致）
    def vfeat(ev):
        s, e, ps, dur = ev
        xs = [dur, len(ps), float(ps.max() - ps.min()),
              float(ps.std() / (ps.mean() + 1e-9))]
        tq = np.arange(len(ps))
        xs += [float(np.polyfit(tq, ps, 1)[0]) if len(ps) > 2 else 0.0,
               float(ps[0] - ps[-1]),
               float((ps >= 0.35).mean()), float((ps >= 0.45).mean())]
        def longest(q):
            best = cur2 = 0
            for v in ps >= q:
                cur2 = cur2 + 1 if v else 0
                best = max(best, cur2)
            return best
        xs += [longest(0.35), longest(0.45), float(np.maximum(ps - WIN_THR, 0).sum())]
        xs += [float(ps.mean()), float(ps.max()), float(ps.std()),
               float(np.percentile(ps, 10)), float(np.percentile(ps, 50)),
               float(np.percentile(ps, 90))]
        # 前后 20min 上下文（窗概率序列——用窗中心对齐，与训练 verifier_features 一致）
        pc = probs
        sc = starts + WIN_MS // 2   # 窗中心（训练侧 centers = (s+e)//2 = s+120s）
        inwin = (sc >= s - CTX_MS) & (sc < s)
        inwin2 = (sc >= e) & (sc < e + CTX_MS)
        for msk in (inwin, inwin2):
            cq = pc[msk]
            if len(cq) >= 5:
                xs += [float(cq.mean()), float(cq.max()), float(cq.std()),
                       float(np.percentile(cq, 10)), float(np.percentile(cq, 50)),
                       float(np.percentile(cq, 90))]
            else:
                xs += [0.0] * 6
        pre = pc[inwin] if inwin.any() else np.zeros(1)
        post = pc[inwin2] if inwin2.any() else np.zeros(1)
        both = np.concatenate([pre, post])
        xs += [float(ps.mean() - both.mean()),
               float(ps.max() - (np.percentile(both, 90) if len(both) else 0.0)),
               float(ps.mean() - (pre.mean() if len(pre) else 0.0)),
               float(ps.mean() - (post.mean() if len(post) else 0.0)),
               0.0, 0.0]   # TCN 列（no_tcn 版恒 0）
        hh = (s / 3.6e6) % 24
        xs += [float(hh), float(GLOBAL_PRIOR[int(hh) % 24])]
        return np.array(xs, np.float64)
    Xv = np.array([vfeat(ev) for ev in cands])
    vs = verifier.predict_proba(Xv)[:, 1]
    if os.environ.get("BME_DEBUG_VER", "0") == "1":
        for i in range(len(cands)):
            print(f"  cand {i}: [{cands[i][0]}, {cands[i][1]}] len={(cands[i][1]-cands[i][0])/1000:.0f}s "
                  f"ver={vs[i]:.3f} ({'KEEP' if vs[i] >= thr_ver else 'drop'})", flush=True)
    return [(cands[i][0], cands[i][1]) for i in range(len(cands)) if vs[i] >= thr_ver]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.json")
    ap.add_argument("--thr", type=float, default=None, help="复核阈值（默认 config.json 中位）")
    args = ap.parse_args()

    import joblib
    mdir = _HERE / "slide_models"
    models = []
    for k in range(5):
        d = joblib.load(mdir / f"wmodel_fold{k}.joblib")
        models.append({"imp": d["imp"], "model": d["model"]})
    ver = joblib.load(mdir / "verifier.joblib")
    cfg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
    thr_ver = args.thr if args.thr is not None else float(cfg["thr_verifier_median"])
    print(f"模型 {len(models)} 窗 bag + 复核器（阈值 {thr_ver:.3f}）", flush=True)

    inp = Path(args.input)
    sids = [inp] if inp.is_dir() else [Path(x.strip()) for x in inp.read_text(encoding="utf-8").splitlines() if x.strip()]
    results = {}
    t0 = time.time()
    for i, sd in enumerate(sids):
        rows = session_features(sd, models)
        evs = predict_session(rows, models, ver, thr_ver)
        results[sd.name] = evs
        print(f"  [{i + 1}/{len(sids)}] {sd.name}: {len(rows)} 窗 → {len(evs)} 事件 "
              f"({time.time() - t0:.0f}s)", flush=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {args.output}", flush=True)


if __name__ == "__main__":
    main()
