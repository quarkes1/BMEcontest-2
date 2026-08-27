# -*- coding: utf-8 -*-
"""全量数据质量校验：并行扫描 1165 会话，产出质量报告与黑名单。
运行：conda activate bme && python scripts/validate_data.py（预计 1-2 小时）"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 项目根入 sys.path

import src.config as config
from src.data import manifests
from src.data.loader import detect_binary, load_session_tsv, _find_collect_data

def check_one(session_id: str) -> dict:
    d = config.SENSOR_DIR / session_id
    rec = {"session_id": session_id}
    try:
        txt = _find_collect_data(str(d))
    except FileNotFoundError:
        rec["error"] = "no_collect_data_file"
        return rec
    if detect_binary(txt):
        rec["binary"] = True
        return rec
    try:
        s = load_session_tsv(txt)
    except Exception as e:  # 单会话解析失败记录而不中断全量扫描
        rec["error"] = f"parse_failed: {type(e).__name__}: {e}"
        return rec
    N = s.acc.shape[1]
    dup = 0.0
    if N > 1:
        diff = np.abs(np.diff(s.acc, axis=1)).sum(axis=0)
        dup = float((diff == 0).mean())
    tail_zero = 0
    for k in range(min(50, N), 0, -1):
        if not (s.imu_valid[N - k] or s.ppg_valid[N - k]):
            tail_zero = k
        else:
            break
    rec.update({
        "binary": False, "rows": N,
        "dup_ratio": round(dup, 4),
        "tail_zero_rows": tail_zero,
        "imu_valid_ratio": round(float(s.imu_valid.mean()), 4),
        "ppg_valid_ratio": round(float(s.ppg_valid.mean()), 4),
        "row_rate": s.meta.get("row_rate"),
    })
    return rec

def main():
    t0 = time.time()
    index = manifests.load_sensor_index()
    ids = sorted(index["session_id"].tolist())
    print(f"扫描 {len(ids)} 个会话...")
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for rec in ex.map(check_one, ids, chunksize=8):
            results.append(rec)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "data_quality.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    blacklist = [r["session_id"] for r in results if r.get("binary") or r.get("error")]
    (config.OUTPUT_DIR / "data_quality_blacklist.txt").write_text(
        "\n".join(blacklist) + "\n", encoding="utf-8")
    bad_tail = [r["session_id"] for r in results if r.get("tail_zero_rows", 0) >= 50]
    errs = [r for r in results if r.get("error")]
    print(f"完成, 用时 {time.time()-t0:.0f}s")
    print(f"二进制损坏: {sum(1 for r in results if r.get('binary'))}")
    print(f"解析失败/缺文件: {len(errs)} {errs[:3]}")
    print(f"尾随置零>=50行: {len(bad_tail)}")
    valid = [r for r in results if "dup_ratio" in r]
    if valid:
        print(f"重复行率: 中位 {np.median([r['dup_ratio'] for r in valid]):.3f}, "
              f"均值 {np.mean([r['dup_ratio'] for r in valid]):.3f}")
        print(f"row_rate 中位: {np.median([r['row_rate'] for r in valid if r['row_rate']]):.1f}")

if __name__ == "__main__":
    main()
