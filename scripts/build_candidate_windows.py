# -*- coding: utf-8 -*-
"""L1 候选窗口缓存：对每折 train+val 候选提取 ±120s 原始信号窗口（MM-Ranker 深度排序头输入）。

输入：rank_events 的两池候选（活动池 IMU 包络连通域 + 先验池整点窗），按 TSV 真实时间戳提取。
输出：cache/cand_windows/fold{k}/ 下每会话一个 npz：
  imu: (2400, 6) float16 — acc3+gyro3 @10Hz 网格（窗 [s-120s, e+120s]，300ms 网格）
  ppg: (48, 66) float32 — 22ch × [mean_norm, std_norm, nz_rate] 5s 块（会话级中位数归一）
  ma:  (48, 2)  float32 — acc 能量块均值/峰值（MA 置信度掩码参考信号）
  meta: json str — sid/s/e/is_prior/label/act_iou/pri_iou/gate_prob/session_span

关键事实（2026-08-30 审计）：PPG 仅前 22 通道有数据（调度采样 ~22% 行 ≈25Hz，raw ADC
0-7e7），后 22 通道恒 0 → 块统计是稳定利用方式；TSV 时间戳与 startTime 中位偏移 0，
但部分会话偏移 60min → 窗落在 TSV 之外时空窗标记（ma 全 0，模型学丢弃）。

运行：source activate bme && python scripts/build_candidate_windows.py --fold 0
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.data import manifests, splits, loader
import rank_events as re

CTX_S = 120.0                  # 窗口半宽（±120s）
FS10 = 10                      # IMU 重采样率
GRID_STEP_MS = 100             # 10Hz 网格步长
BLOCK_S = 5.0                  # PPG 块宽
N_BLOCK = 48                   # 240s / 5s
N_PPG = 22                     # 有效 PPG 通道数
MIN_EMPTY_RATIO = 0.05         # 窗有效 IMU 覆盖率 <5% → 标记空窗


def _session_ppg_stats(s):
    """会话级 PPG 中位数（每通道，非零值），用于块统计归一。"""
    m = s.ppg[:N_PPG, s.ppg_valid]
    med = np.zeros(N_PPG, np.float32)
    for c in range(N_PPG):
        nz = m[c] != 0
        med[c] = np.median(m[c, nz]) if nz.any() else 0.0
    return med


def _extract(s, med_c, c_s, c_e, valid_ts):
    """提取单候选窗口信号（以候选中心 ±CTX_S 定长窗口，统一 2400 步 @10Hz）。
    valid_ts: TSV 有效 IMU 时间戳（用于空窗判定）。返回 (imu, ppg, ma, ok) 或 None。"""
    c_mid = (c_s + c_e) // 2
    ws = c_mid - int(CTX_S * 1000)
    we = c_mid + int(CTX_S * 1000)
    n_step = int((we - ws) / GRID_STEP_MS) + 1
    grid = ws + np.arange(n_step) * GRID_STEP_MS          # 10Hz 网格（绝对 ms）

    t = s.t_acc.astype(np.float64)
    valid = s.imu_valid
    if not valid.any():
        return None
    t_v = t[valid]
    imu = np.zeros((n_step, 6), np.float32)
    for i, ch in enumerate(np.concatenate([s.acc, s.gyro])[:, valid]):
        imu[:, i] = np.interp(grid, t_v, ch.astype(np.float64))
    e_norm = np.sqrt((imu[:, :3] ** 2).sum(1))            # acc 能量
    cov = ((grid >= t_v[0]) & (grid <= t_v[-1])).mean()
    if cov < MIN_EMPTY_RATIO:
        return None

    # PPG 块统计
    tp = s.t_ppg.astype(np.float64)
    pv = s.ppg_valid
    ppg = np.zeros((N_BLOCK, N_PPG * 3), np.float32)
    ma = np.zeros((N_BLOCK, 2), np.float32)
    for b in range(N_BLOCK):
        b0 = ws + int(b * BLOCK_S * 1000)
        b1 = b0 + int(BLOCK_S * 1000)
        sel = (tp >= b0) & (tp < b1) & pv
        gm = (grid >= b0) & (grid < b1)
        rows = s.ppg[:N_PPG, sel]                          # (22, n_m)
        if rows.shape[1]:
            for c in range(N_PPG):
                nz = rows[c] != 0
                if nz.any():
                    v = rows[c, nz].astype(np.float64)
                    base = med_c[c] + 1e-3
                    ppg[b, c * 3] = (v.mean() - base) / base        # 相对漂移
                    ppg[b, c * 3 + 1] = v.std() / base              # 相对变异性
                    ppg[b, c * 3 + 2] = nz.mean()                   # 采样密度
                else:
                    ppg[b, c * 3 + 2] = 0.0
        if gm.any():
            ma[b, 0] = e_norm[gm].mean()
            ma[b, 1] = e_norm[gm].max()
    # 边角块（窗边界溢出）清零
    ppg[np.isnan(ppg)] = 0.0
    return imu, ppg, ma, True


def _process_sid(args):
    sid, fold, gate_probs, starts, prior, split, meals = args
    out_dir = config.CACHE_DIR / "cand_windows" / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"{sid}.npz"
    try:
        s = loader.load_session(sid)
    except Exception as e:
        return (sid, f"load_err: {e}")
    p = config.CACHE_DIR / "validate_baselines" / f"{sid}.npz"
    if not p.exists():
        return (sid, "no_env")
    d = np.load(p)
    env = d["env"].astype(np.float32)
    t0 = d["t0"].astype(np.int64)
    ft = json.loads(d["feats"].item())
    sess_feats = {"dur_h": float(ft["dur_h"]), "env_p95": float(ft["env_p95"]),
                  "p95_ratio": float(ft["p95_ratio"]), "start": starts.get(sid, 0)}
    act, pri = re.make_proposals(env, t0, prior, starts.get(sid, 0))
    meals = meals.get(sid, [])
    med_c = _session_ppg_stats(s)
    tv = s.t_acc[s.imu_valid]
    t_valid = (tv.min(), tv.max())

    # 深度模型只评活动池候选（信号级二次校验）：先验池是纯时间先验（40min 宽窗
    # 无统一信号片段），保留 LightGBM 通道，不建窗口缓存。
    rows = []
    for (s0, e0, _), y in zip(act, re.match_labels(act, meals)):
        r = _extract(s, med_c, s0, e0, t_valid)
        if r is None:
            continue
        imu, ppg, ma, _ = r
        rows.append({
            "imu": imu.astype(np.float16), "ppg": ppg, "ma": ma,
            "meta": json.dumps({"sid": sid, "s": s0, "e": e0, "is_prior": 0,
                                "label": int(y), "gate_prob": float(gate_probs.get(sid, 0.5)),
                                "span_s": float(s.acc.shape[1] / s.meta.get("row_rate", 105.0))}),
        })
    if not rows:
        return (sid, "no_cands")
    np.savez_compressed(out_p, **{f"c{i}": rows[i]["imu"] for i in range(len(rows))},
                        ppg=np.stack([r["ppg"] for r in rows]).astype(np.float32),
                        ma=np.stack([r["ma"] for r in rows]).astype(np.float32),
                        meta=[r["meta"].encode() for r in rows])
    return (sid, f"ok:{len(rows)}")


sid_meals = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    k = args.fold
    global sid_meals
    folds = splits.load_folds()
    f = folds[k]
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    starts = {r["session_id"]: int(r["timeStamp.startTime"]) for _, r in idx.iterrows()}
    for _, r in idx.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms
    prior = re._prior(np.array([m["before"] / 3.6e6 % 24
                                for s in f["train_sessions"] if s in sid_meals for m in sid_meals[s]]))
    # 会话门控（与 rank_events 相同：V1 7 特征 LightGBM，仅 val 打分）
    import lightgbm as lgb
    GATE_FEATS = ["dur_h", "env_mean", "env_p50", "env_p95", "env_std", "p95_ratio", "start_h_utc"]
    gate_feats = {}
    for split in ("train_sessions", "val_sessions"):
        for sid in f[split]:
            p = config.CACHE_DIR / "validate_baselines" / f"{sid}.npz"
            if not p.exists():
                continue
            ft = json.loads(np.load(p)["feats"].item())
            gate_feats[sid] = [ft[kk] for kk in GATE_FEATS]
    Xg_tr = np.array([gate_feats[s] for s in f["train_sessions"] if s in gate_feats])
    yg_tr = np.array([1 if s in sid_meals else 0 for s in f["train_sessions"] if s in gate_feats])
    gate = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                              min_child_samples=20, verbosity=-1)
    gate.fit(Xg_tr, yg_tr)
    gate_probs = {}
    for sid in f["val_sessions"]:
        if sid in gate_feats:
            gate_probs[sid] = float(gate.predict_proba(np.array(gate_feats[sid])[None])[0, 1])

    sids = f["train_sessions"] + f["val_sessions"]
    print(f"fold{k}: 提取 {len(sids)} 会话候选窗口（8 并行）...", flush=True)
    t0 = time.time()
    tasks = [(sid, k, gate_probs, starts, prior, "tr" if sid in f["train_sessions"] else "va", sid_meals)
             for sid in sids]
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(_process_sid, tasks, chunksize=4):
            if res is not None:
                sid, msg = res
                if not msg.startswith("ok"):
                    print(f"  {sid[-8:]} {msg}", flush=True)
    n_out = len(list((config.CACHE_DIR / "cand_windows" / f"fold{k}").glob("*.npz")))
    print(f"完成 {time.time()-t0:.0f}s → cache/cand_windows/fold{k}/ {n_out} 会话", flush=True)


if __name__ == "__main__":
    main()
