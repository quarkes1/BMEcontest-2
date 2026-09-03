# -*- coding: utf-8 -*-
"""会话缓存压缩（写新目录版）：cache/sessions/*.npz → cache/sessions_c/*.npz 压缩版。

规避 in-place 替换的间歇失败与 tmp 残留问题：输出独立目录，逐文件独立处理，
全部完成后由调用方切换目录（原目录保留待确认后删除）。
"""
import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SRC = Path("cache/sessions")
DST = Path("cache/sessions_c")


def convert_one(name: str):
    src = SRC / name
    dst = DST / name
    try:
        d = np.load(src)
        np.savez_compressed(dst,
                            acc=d["acc"], gyro=d["gyro"], ppg=d["ppg"],
                            t_acc=d["t_acc"], t_ppg=d["t_ppg"],
                            imu_valid=d["imu_valid"], ppg_valid=d["ppg_valid"],
                            row_rate=d["row_rate"], rows=d["rows"])
        d.close()
        return dst.stat().st_size
    except Exception as e:
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        return f"{name}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in SRC.glob("*.npz") if "tmp" not in p.name)
    print(f"{len(names)} 个会话待压缩 → {DST}", flush=True)
    t0 = time.time()
    ok, fails = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(convert_one, names, chunksize=4)):
            if isinstance(r, int):
                ok += 1
            else:
                fails.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(names)} 用时 {time.time()-t0:.0f}s", flush=True)
    print(f"完成：{ok}/{len(names)} 成功，{len(fails)} 失败（{time.time()-t0:.0f}s）", flush=True)
    for f in fails[:10]:
        print(f"  失败: {f}", flush=True)
    if not fails:
        print("→ 切换：删除 cache/sessions 并重命名 sessions_c（需确认后执行）", flush=True)


if __name__ == "__main__":
    main()
