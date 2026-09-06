# -*- coding: utf-8 -*-
"""滑窗管线评估（阶段 1：验证密度候选层——对方候选层 recall 0.69 / PPV 0.18）。

流程：HGB(train 采样窗) → val 全窗概率 → 密度聚合（600s 中心 ≥10 越阈 ≥0.8 覆盖）
→ window_support 事件 + 120s 合并 → 官方评估。复核器阶段 2 加入。

用法：D:/Anaconda3/envs/bme/python.exe scripts/slide_eval.py --fold 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.data import manifests, splits
import official_iou_eval as oe

STRIDE_MS = 15_000
BRIDGE_MS = 60_000
DENSITY_MS = 600_000
MIN_POS = 10
COV_MIN = 0.80
MERGE_MS = 120_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--thr", type=float, default=None, help="窗口概率阈值（默认 None=OOF MCC 网格）")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer

    tr = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_train.npz", allow_pickle=True)
    Xtr, ytr = tr["feat"], tr["label"]
    keep_tr = ytr >= 0
    imp = SimpleImputer(strategy="median").fit(Xtr[keep_tr])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
        random_state=20260901)
    clf.fit(imp.transform(Xtr[keep_tr]), ytr[keep_tr].astype(int))
    print(f"HGB 训练：{keep_tr.sum()} 窗（正 {int((ytr[keep_tr] == 1).sum())}）", flush=True)

    va = np.load(config.CACHE_DIR / "slide" / f"fold{args.fold}_val.npz", allow_pickle=True)
    Xva = va["feat"]
    wids = [json.loads(w) for w in va["wid"]]
    prob = clf.predict_proba(imp.transform(Xva))[:, 1]
    print(f"val 全窗 {len(prob)}（正 {int((va['label'] == 1).sum())}）打分完成", flush=True)

    # OOF MCC 阈值（在 val 的标注窗上粗选——阶段 1 近似；严格版在 train OOF 上）
    from sklearn.metrics import matthews_corrcoef
    if args.thr is None:
        lab_known = va["label"] >= 0
        best_t, best_m = 0.5, -1
        for t in np.linspace(0.01, 0.80, 316):
            m = matthews_corrcoef(va["label"][lab_known].astype(int), prob[lab_known] >= t)
            if m > best_m:
                best_m, best_t = m, t
        thr = best_t
        print(f"阈值（val MCC 网格）: {thr:.4f}（MCC {best_m:.3f}）", flush=True)
    else:
        thr = args.thr

    # ---- 密度聚合（按会话） ----
    from collections import defaultdict
    sid_windows = defaultdict(list)
    for wid, p in zip(wids, prob):
        sid_windows[wid[0]].append((wid[1], wid[2], float(p)))
    val_rows_t = []
    for sid in sorted(sid_windows):
        arr = sorted(sid_windows[sid])
        starts = np.array([a[0] for a in arr], np.int64)
        probs = np.array([a[2] for a in arr])
        # 60s 桥接：缺口 <=60s 合成窗（概率 0，observed False）；>60s 断段
        seg = []
        for i in range(len(starts)):
            if seg and starts[i] - seg[-1][0] > STRIDE_MS + BRIDGE_MS:
                process_segment(seg, thr, val_rows_t, sid)
                seg = []
            if seg and starts[i] - seg[-1][0] > STRIDE_MS:   # 桥接小缺口
                for gs in range(seg[-1][0] + STRIDE_MS, starts[i], STRIDE_MS):
                    seg.append((gs, 0.0, 0))
            seg.append((int(starts[i]), float(probs[i]), 1))
        if seg:
            process_segment(seg, thr, val_rows_t, sid)
    print(f"密度候选事件：{len(val_rows_t)}", flush=True)

    # ---- 评估 ----
    gt_sids = set()
    folds = splits.load_folds()
    meal_meta, _ = manifests.load_meal_meta()
    idx = manifests.load_sensor_index()
    sid_meals = {}
    for _, r in idx.iterrows():
        ext, sid, st, en = r["externalid"], r["session_id"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"])
        ms = [m for m in meal_meta.get(ext, []) if m["before"] >= st and m["after"] <= en]
        if ms:
            sid_meals[sid] = ms
    true_sid = []
    for sid in sorted(sid_windows):
        for m in sid_meals.get(sid, []):
            true_sid.append((sid, (m["before"], m["after"])))
    m = oe.official_metrics(val_rows_t, true_sid)
    print(f"[候选层] F1={m['f1']:.3f} sens={m['sensitivity']:.3f} ppv={m['ppv']:.3f} "
          f"({m['n_tp']}/{m['n_true']}, pred={m['n_pred']})", flush=True)
    out = {"fold": args.fold, "thr": thr, "candidates": {k2: v for k2, v in m.items()}}
    (config.OUTPUT_DIR / f"slide_fold{args.fold}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def process_segment(seg, thr, out, sid):
    """seg: [(start_ms, prob, observed)] 连续段 → 密度 run → 事件。"""
    starts = np.array([s for s, _, _ in seg], np.int64)
    probs = np.array([p for _, p, _ in seg])
    obs = np.array([o for _, _, o in seg])
    n = len(seg)
    if n < MIN_POS:
        return
    step = STRIDE_MS
    density_steps = int(round(DENSITY_MS / step))
    pos = probs >= thr
    # rolling sum（中心窗 600s）
    cnt = np.convolve(pos.astype(np.int64), np.ones(density_steps, np.int64), mode="same")
    cov = np.convolve(obs.astype(np.float64), np.ones(density_steps) / density_steps, mode="same")
    dense = (cnt >= MIN_POS) & (cov >= COV_MIN)
    # dense run → 事件（window_support 边界）
    i = 0
    while i < n:
        if dense[i]:
            j = i
            while j < n and dense[j]:
                j += 1
            ev_s = int(starts[i])
            ev_e = int(starts[j - 1] + WIN_END_MS)
            # 120s 合并（与前事件）
            if out and out[-1][0] == sid and ev_s - out[-1][1][1] <= MERGE_MS:
                out[-1] = (sid, (out[-1][1][0], max(out[-1][1][1], ev_e)))
            else:
                out.append((sid, (ev_s, ev_e)))
            i = j
        else:
            i += 1


WIN_END_MS = 240_000


if __name__ == "__main__":
    main()
