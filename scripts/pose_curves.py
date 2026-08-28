# -*- coding: utf-8 -*-
"""姿态解算曲线产出（W2 交付，web 三级缓存的数据源）：
选 4-6 个高光会话（dominant 餐、时长 12-25min、正常行率），
输出 cache/pose/{sid}.npz（10Hz pitch/roll/yaw/static）+ outputs/pose_{sid}.png 校验图。
运行：conda activate bme && python scripts/pose_curves.py"""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests
from src.data.loader import load_session
from src.features.pose import estimate_attitude

OUT_DIR = config.CACHE_DIR / "pose"
MAX_SESSIONS = 6

def pick_highlight_sessions():
    meal_meta, _ = manifests.load_meal_meta()
    index = manifests.load_sensor_index()
    # session -> (externalid, start, end)
    by_sid = {}
    for _, r in index.iterrows():
        by_sid[r["session_id"]] = (r["externalid"], int(r["timeStamp.startTime"]), int(r["timeStamp.endTime"]))
    picks = []
    for ext, meals in meal_meta.items():
        for m in meals:
            if m["scene"] != "dominant":
                continue
            dur_min = (m["after"] - m["before"]) / 60000.0
            if not (12 <= dur_min <= 25):
                continue
            for sid, (e, s0, s1) in by_sid.items():
                if e == ext and s0 <= m["before"] and s1 >= m["after"]:
                    picks.append((sid, m))
                    break
    return picks[:MAX_SESSIONS]

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks = pick_highlight_sessions()
    print(f"高光会话 {len(picks)} 个")
    for sid, m in picks:
        s = load_session(sid)
        r = estimate_attitude(s, out_hz=10.0)
        np.savez_compressed(OUT_DIR / f"{sid}.npz",
                            t_sec=r["t_sec"].astype(np.float32),
                            pitch=r["pitch"].astype(np.float32),
                            roll=r["roll"].astype(np.float32),
                            yaw=r["yaw"].astype(np.float32),
                            static=r["static"].astype(bool))
        # 校验图（不带中文，避免字体问题）
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        for ax, name, series in zip(axes, ("pitch", "roll", "yaw"),
                                    (r["pitch"], r["roll"], r["yaw"])):
            ax.plot(r["t_sec"], series, lw=0.6)
            ax.set_ylabel(name + " (deg)")
            ax.grid(alpha=0.3)
            for st, en in _static_spans(r["t_sec"], r["static"]):
                ax.axvspan(st, en, color="green", alpha=0.12)
        axes[0].set_title(f"{sid[:40]} meal {m['before']}")
        axes[-1].set_xlabel("t (s)")
        fig.tight_layout()
        fig.savefig(config.OUTPUT_DIR / f"pose_{sid[:48]}.png", dpi=110)
        plt.close(fig)
        print(f"  {sid}: {len(r['t_sec'])} 点 -> cache/pose + outputs/pose_{sid[:48]}.png", flush=True)
    print(f"完成，用时 {time.time()-t0:.0f}s")

def _static_spans(t, static):
    spans = []
    cur = None
    for i in range(len(static)):
        if static[i] and cur is None:
            cur = t[i]
        elif not static[i] and cur is not None:
            spans.append((cur, t[i])); cur = None
    if cur is not None:
        spans.append((cur, t[-1]))
    return spans

if __name__ == "__main__":
    main()
