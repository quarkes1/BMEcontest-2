# -*- coding: utf-8 -*-
"""一次性 TSV 解析 → 二进制会话缓存 cache/sessions/{sid}.npz（约 5GB）。
此后 loader.load_session 自动优先读缓存（~50ms vs ~1.5s 解析），
所有特征/窗口缓存构建共享，避免每条链重复解析同一批 TSV。
运行：conda activate bme && python scripts/parse_sessions.py（8 进程，约 30-40 分钟）"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.data import manifests
from src.data.loader import load_session, detect_binary, _find_collect_data

OUT_DIR = config.CACHE_DIR / "sessions"

def parse_one(session_id):
    out = OUT_DIR / f"{session_id}.npz"
    if out.exists():
        return ("skip", session_id)
    try:
        d = config.SENSOR_DIR / session_id
        txt = _find_collect_data(str(d))
        if detect_binary(txt):
            return ("binary", session_id)
        s = load_session(session_id)
        np.savez_compressed(out,
                 acc=s.acc, gyro=s.gyro, ppg=s.ppg,
                 t_acc=s.t_acc, t_ppg=s.t_ppg,
                 imu_valid=s.imu_valid, ppg_valid=s.ppg_valid,
                 row_rate=np.float32(s.meta["row_rate"]), rows=np.int64(s.meta["rows"]))
        return ("ok", session_id)
    except Exception as e:
        return ("error", f"{session_id}: {type(e).__name__}: {e}")

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = manifests.load_sensor_index()
    tasks = list(index["session_id"])
    print(f"解析 {len(tasks)} 会话 → cache/sessions/（8 进程）")
    stats = {"ok": 0, "skip": 0, "binary": 0, "error": []}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (status, info) in enumerate(ex.map(parse_one, tasks, chunksize=4)):
            if status == "error":
                stats["error"].append(info)
            else:
                stats[status] = stats[status] + 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} 用时 {time.time()-t0:.0f}s", flush=True)
    (OUT_DIR / "parse_stats.json").write_text(
        json.dumps({**stats, "elapsed_s": round(time.time() - t0)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"完成: ok={stats['ok']} skip={stats['skip']} binary={stats['binary']} "
          f"error={len(stats['error'])} 用时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
