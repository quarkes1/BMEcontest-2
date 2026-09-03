# -*- coding: utf-8 -*-
"""批量压缩 cache/sessions/*.npz（np.savez → savez_compressed）。

背景：loader.py 曾以无压缩 np.savez 写会话缓存（PPG 44ch raw ADC float32），
1130 会话膨胀至 ~116G；压缩后 ~11MB/会话（总量 ~13G，释放 ~100G）。
本脚本就地转换：读 npz → 压缩写 .tmp → 原子替换（先写后删，中断安全）。

运行：python scripts/compress_sessions.py [--workers 4]
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def convert_one(p: Path):
    try:
        d = np.load(p)
        tmp = p.with_suffix(".npz.tmp")
        np.savez_compressed(tmp,
                            acc=d["acc"], gyro=d["gyro"], ppg=d["ppg"],
                            t_acc=d["t_acc"], t_ppg=d["t_ppg"],
                            imu_valid=d["imu_valid"], ppg_valid=d["ppg_valid"],
                            row_rate=d["row_rate"], rows=d["rows"])
        d.close()
        os.replace(tmp, p)          # 原子替换（中断不产生半文件）
        return p.stat().st_size
    except Exception as e:
        return f"{p.name}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    files = sorted(Path("cache/sessions").glob("*.npz"))
    print(f"{len(files)} 个会话缓存待压缩", flush=True)
    t0 = time.time()
    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(convert_one, files, chunksize=4)):
            if isinstance(r, int):
                ok += 1
            else:
                print(f"  失败: {r}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(files)} 用时 {time.time()-t0:.0f}s", flush=True)
    print(f"完成：{ok}/{len(files)} 压缩（{time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
