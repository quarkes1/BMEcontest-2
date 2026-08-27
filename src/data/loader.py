# -*- coding: utf-8 -*-
"""TSV 流式解析：行空间即时间网格，掩码标记占空比采样。"""
import re
from dataclasses import dataclass, field
import os
import numpy as np
import src.config as config

N_PPG = 44

@dataclass
class SessionData:
    acc: np.ndarray          # (3, N)
    gyro: np.ndarray         # (3, N)
    ppg: np.ndarray          # (44, N)
    t_acc: np.ndarray        # (N,) 毫秒；无效行 -1
    t_ppg: np.ndarray        # (N,) 毫秒；无效行 -1
    imu_valid: np.ndarray    # (N,) bool
    ppg_valid: np.ndarray    # (N,) bool
    meta: dict = field(default_factory=dict)

def detect_binary(path, head=512):
    with open(path, "rb") as f:
        chunk = f.read(head)
    if not chunk:
        return False
    text_ratio = sum(1 for b in chunk if b in b"\t\n\r" or 32 <= b < 127) / len(chunk)
    return text_ratio < 0.9

def _find_collect_data(path):
    """返回目录内 collect_data*.txt（每会话恰 1 个）。"""
    names = [n for n in os.listdir(path) if re.match(r"collect_data\d+_\d+_\d+\.txt$", n)]
    if not names:
        raise FileNotFoundError(f"no collect_data txt in {path}")
    return os.path.join(path, sorted(names)[0])

def load_session_tsv(txt_path) -> SessionData:
    acc_x, acc_y, acc_z, gx, gy, gz = [], [], [], [], [], []
    t_acc, t_ppg, imu_valid, ppg_valid = [], [], [], []
    ppg = [[] for _ in range(N_PPG)]
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        assert "ACC_TIME" in header, f"bad header: {txt_path}"
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 53:
                continue
            try:
                at, pt, gt = int(parts[0]), int(parts[1]), int(parts[2])
                vals = list(map(float, parts[3:53]))
            except ValueError:
                continue
            ppg_vals = vals[:N_PPG]
            a = vals[N_PPG:N_PPG + 3]; g = vals[N_PPG + 3:N_PPG + 6]
            imu_ok = (at > 0 or gt > 0) and not all(v == 0 for v in a)
            ppg_ok = pt > 0 and not all(v == 0 for v in ppg_vals)
            acc_x.append(a[0]); acc_y.append(a[1]); acc_z.append(a[2])
            gx.append(g[0]); gy.append(g[1]); gz.append(g[2])
            for j in range(N_PPG):
                ppg[j].append(ppg_vals[j])
            t_acc.append(at if imu_ok else -1)
            t_ppg.append(pt if ppg_ok else -1)
            imu_valid.append(imu_ok); ppg_valid.append(ppg_ok)
    N = len(t_acc)
    row_rate = config.IMU_ROW_RATE
    if N and any(imu_valid):
        valid_ts = [t for t in t_acc if t > 0]
        span = (max(valid_ts) - min(valid_ts)) / 1000.0
        if span > 60:
            row_rate = N / span
    return SessionData(
        acc=np.array([acc_x, acc_y, acc_z], dtype=np.float32),
        gyro=np.array([gx, gy, gz], dtype=np.float32),
        ppg=np.array(ppg, dtype=np.float32),
        t_acc=np.array(t_acc, dtype=np.int64),
        t_ppg=np.array(t_ppg, dtype=np.int64),
        imu_valid=np.array(imu_valid, dtype=bool),
        ppg_valid=np.array(ppg_valid, dtype=bool),
        meta={"path": txt_path, "rows": N, "row_rate": round(row_rate, 1)})

def load_session(session_id: str) -> SessionData:
    d = config.SENSOR_DIR / session_id
    return load_session_tsv(_find_collect_data(str(d)))
