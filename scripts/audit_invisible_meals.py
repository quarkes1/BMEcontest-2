# -*- coding: utf-8 -*-
"""隐形餐审计：对比"模型不可见餐"与"可见餐"的原始信号特征，
判断是①信号真不存在（→投 PPG）②窗口/时间对齐问题（→修数据管线）
③信号存在但模型没学会（→模型问题）。产出 outputs/invisible_audit.json + 对比图。"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests, splits
from src.data.loader import load_session
from src.features.baseline_features import _bandpass

def meal_signal_profile(session, before, after, fs):
    """餐区间内的原始信号统计。"""
    s = session
    # 用时间戳定位行范围
    t = s.t_acc
    valid = t > 0
    rows = np.flatnonzero(valid & (t >= before) & (t <= after))
    if len(rows) == 0:
        return None
    a = s.acc[:, rows]
    g = s.gyro[:, rows]
    am = np.linalg.norm(a, axis=0)
    gm = np.linalg.norm(g, axis=0)
    am_bp = _bandpass(am, 0.5, 2.0, fs)
    ppg_valid = s.ppg_valid[rows]
    ppg = s.ppg[:, rows][:, ppg_valid] if ppg_valid.any() else s.ppg[:, :0]
    return {
        "n_rows": len(rows),
        "am_mean": float(am.mean()), "am_std": float(am.std()),
        "gm_std": float(gm.std()),
        "bp_energy": float(np.mean(np.abs(am_bp))),
        "ppg_valid_frac": float(ppg_valid.mean()),
        "ppg_ac": float(np.std(ppg - ppg.mean(axis=1, keepdims=True))) if ppg.size else 0.0,
    }

def main():
    folds = splits.load_folds()
    f0 = folds[0]
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index().set_index("session_id")
    sess_range = {s: (int(index.loc[s, "timeStamp.startTime"]), int(index.loc[s, "timeStamp.endTime"]))
                  for s in f0["val_sessions"] if s in index.index}

    # 从 W2 诊断结论选取：不可见（通过率<0.1）与可见（>0.8）各若干
    invisible = [("HNU21011", 1784714211000), ("HNU21011", 1784625644000),
                 ("HNU21011", 1784605011000), ("HNU21035", 1784973236000),
                 ("HNU21035", 1784955296000), ("HNU21028", 1785067901000),
                 ("HNU21009", 1784692184000), ("HNU21008", 1784714263000)]
    visible = [("HNU21031", 1785040326000), ("HNU21031", 1785026545000),
               ("hnu21033", 1785329191129), ("HNU21035", 1785123654000)]

    out = {"invisible": [], "visible": []}
    for tag, (ext, before) in list(zip(["invisible"] * len(invisible) + ["visible"] * len(visible),
                                       invisible + visible)):
        m = next((x for x in meal_meta.get(ext, []) if x["before"] == before), None)
        if m is None:
            continue
        sid = next((s for s in f0["val_sessions"]
                    if s in sess_range and sess_range[s][0] <= m["before"]
                    and sess_range[s][1] >= m["after"]), None)
        if sid is None:
            out[tag].append({"ext": ext, "before": before, "error": "无覆盖会话"})
            continue
        session = load_session(sid)
        fs = session.meta.get("row_rate", 105.0)
        prof = meal_signal_profile(session, m["before"], m["after"], fs)
        if prof is None:
            out[tag].append({"ext": ext, "before": before, "error": "餐区间无有效行"})
            continue
        out[tag].append({"ext": ext, "before": before, "session": sid[:40], **prof})

    (config.OUTPUT_DIR / "invisible_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for tag in ("invisible", "visible"):
        rows = [r for r in out[tag] if "am_std" in r]
        if rows:
            print(f"{tag}: 平均 am_std={np.mean([r['am_std'] for r in rows]):.3f} "
                  f"gm_std={np.mean([r['gm_std'] for r in rows]):.3f} "
                  f"bp_energy={np.mean([r['bp_energy'] for r in rows]):.4f} "
                  f"ppg_ac={np.mean([r['ppg_ac'] for r in rows]):.2f} "
                  f"ppg_valid={np.mean([r['ppg_valid_frac'] for r in rows]):.3f}")
    for r in out["invisible"] + out["visible"]:
        print(f"  {r.get('ext')} {r.get('before')}: am_std={r.get('am_std', r.get('error'))} "
              f"gm_std={r.get('gm_std','')} bp={r.get('bp_energy','')} ppg_ac={r.get('ppg_ac','')}")
    print("→ outputs/invisible_audit.json")

if __name__ == "__main__":
    main()
