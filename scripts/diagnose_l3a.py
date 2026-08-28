# -*- coding: utf-8 -*-
"""L3a 诊断：逐餐统计"餐内窗口概率通过率"，量化跨餐泛化（W2 关键结论）。
用法：conda activate bme && python scripts/diagnose_l3a.py [fold]
输出：每餐一行（受试者/餐时间/餐内窗口数/通过率），供 W2 报告引用。"""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests
from src.models.l3a_cnn import L3aCNN, N_CHANNELS

THRESHOLD = 0.4

def main(fold=0):
    stats = json.loads((Path(f"cache/l3a_raw/fold{fold}") / "stats.json").read_text(encoding="utf-8"))
    mean = np.array(stats["mean"], dtype=np.float32).reshape(1, N_CHANNELS, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(1, N_CHANNELS, 1) + 1e-6
    model = L3aCNN(5).cuda().eval()
    ckpt = config.MODEL_DIR / f"l3a_cnn_fold{fold}.pt"
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index().set_index("session_id")
    f = json.loads(Path(f"cache/splits/fold{fold}.json").read_text(encoding="utf-8"))
    sess_range = {s: (int(index.loc[s, "timeStamp.startTime"]), int(index.loc[s, "timeStamp.endTime"]))
                  for s in f["val_sessions"] if s in index.index}
    val_exts = {index.loc[s, "externalid"] for s in sess_range}

    rows = []
    for ext in sorted(val_exts):
        for m in meal_meta.get(ext, []):
            if m["scene"] != "dominant":
                continue
            sid = next((s for s in f["val_sessions"]
                        if s in sess_range and sess_range[s][0] <= m["before"]
                        and sess_range[s][1] >= m["after"]), None)
            fp = Path("cache/l3a_val_raw") / f"{sid}.npz" if sid else None
            if sid is None or not fp.exists():
                rows.append((ext, m["before"], 0, float("nan")))
                continue
            d = np.load(fp)
            X = (d["X"].astype(np.float32) - mean) / std
            mid = (d["t0"] + d["t1"]) / 2
            in_m = (mid > m["before"]) & (mid < m["after"])
            probs = []
            with torch.no_grad():
                for b in range(0, len(X), 512):
                    xb = torch.from_numpy(X[b:b + 512]).cuda()
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logit, _ = model(xb)
                    probs.append(torch.sigmoid(logit).float().cpu().numpy())
            p = np.concatenate(probs)
            rows.append((ext, m["before"], int(in_m.sum()), float((p[in_m] > THRESHOLD).mean())))

    rates = np.array([r[3] for r in rows if not np.isnan(r[3])])
    print(f"fold {fold}: {len(rates)} 个 dominant 餐可评分，通过率>{THRESHOLD} 的占比 "
          f"{float((rates > 0.5).mean()):.2f}，均值 {np.nanmean(rates):.2f}")
    for ext, before, n, rate in rows:
        print(f"{ext} {before}: n={n} pass={rate:.2f}" if not np.isnan(rate) else f"{ext} {before}: 无覆盖")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
