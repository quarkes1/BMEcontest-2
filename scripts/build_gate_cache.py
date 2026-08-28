# -*- coding: utf-8 -*-
"""构建 L1/L2 门控特征缓存：cache/gate_features/{session_id}.npz
（G8: n×8 L1 特征, G7: n×7 L2 特征, y: n, scene: n, tw: n, t0/t1: n；
scene/tw 仅正样本有效，其余 -1。灰区窗口已跳过。）
运行：conda activate bme && python scripts/build_gate_cache.py（8 进程，预计 30-40 分钟）"""
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
from src.data.windows import iter_window_labels
from src.features.pose import per_row_tilt
from src.models.l1_gate import gate_features
from src.models.l2_scene_gate import scene_features

CACHE_DIR = config.CACHE_DIR / "gate_features"

def _match_meal(t0, t1, meals):
    """找与窗口重叠比例 >= 0.5 的餐（与 windows.py 判定口径一致）。"""
    dur = t1 - t0
    for m in meals:
        ov = min(t1, m["after"]) - max(t0, m["before"])
        if ov > 0 and ov / dur >= config.IOU_POS:
            return m
    return None

def build_one(session_id, meal_list):
    out = CACHE_DIR / f"{session_id}.npz"
    if out.exists():
        return ("skip", session_id)
    try:
        d = config.SENSOR_DIR / session_id
        txt = _find_collect_data(str(d))
        if detect_binary(txt):
            return ("binary", session_id)
        s = load_session(session_id)
        tilts = per_row_tilt(s.acc, s.meta.get("row_rate", 105.0))
        g8s, g7s, ys, scs, tws, t0s, t1s = [], [], [], [], [], [], []
        for w in iter_window_labels(s, [(m["before"], m["after"]) for m in meal_list]):
            g8s.append(gate_features(s, w["start_row"], w["end_row"]))
            g7s.append(scene_features(s, w["start_row"], w["end_row"], tilt_rows=tilts))
            ys.append(w["label"]); t0s.append(w["t0_ms"]); t1s.append(w["t1_ms"])
            m = _match_meal(w["t0_ms"], w["t1_ms"], meal_list) if w["label"] == 1 else None
            if m is not None:
                scs.append(0 if m["scene"] == "dominant" else 1)
                tws.append(m["tableware"])
            else:
                scs.append(-1); tws.append(-1)
        if g8s:
            np.savez_compressed(out,
                                G8=np.vstack(g8s), G7=np.vstack(g7s),
                                y=np.array(ys, dtype=np.int8),
                                scene=np.array(scs, dtype=np.int8),
                                tw=np.array(tws, dtype=np.int8),
                                t0=np.array(t0s, dtype=np.int64), t1=np.array(t1s, dtype=np.int64))
        return ("ok", session_id)
    except Exception as e:
        return ("error", f"{session_id}: {type(e).__name__}: {e}")

def _worker(args):
    return build_one(*args)

def main():
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = manifests.load_sensor_index()
    meal_meta, tw_classes = manifests.load_meal_meta()
    tasks = [(sid, meal_meta.get(ext, [])) for sid, ext in
             index[["session_id", "externalid"]].itertuples(index=False)]
    print(f"构建门控特征缓存: {len(tasks)} 会话, 8 进程")
    stats = {"ok": 0, "skip": 0, "binary": 0, "error": []}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (status, info) in enumerate(ex.map(_worker, tasks, chunksize=4)):
            if status == "error":
                stats["error"].append(info)
            else:
                stats[status] = stats[status] + 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} 用时 {time.time()-t0:.0f}s", flush=True)
    (CACHE_DIR / "build_stats.json").write_text(
        json.dumps({"ok": stats["ok"], "skip": stats["skip"], "binary": stats["binary"],
                    "tableware_classes": tw_classes, "errors": stats["error"]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成: ok={stats['ok']} skip={stats['skip']} binary={stats['binary']} "
          f"error={len(stats['error'])} 用时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
