# -*- coding: utf-8 -*-
"""基于统一原始数据解析器的历史会话加载器，支持缓存。"""

import os
import re

import numpy as np

import src.config as config
from src.pipeline.io.raw_session import SessionData, _parse_collect_data_tsv

N_PPG = 44


def detect_binary(path, head=512):
    with open(path, "rb") as handle:
        chunk = handle.read(head)
    if not chunk:
        return False
    text_ratio = sum(1 for byte in chunk if byte in b"\t\n\r" or 32 <= byte < 127) / len(chunk)
    return text_ratio < 0.9


def _find_collect_data(path):
    """返回会话目录内第一个符合历史命名约定的原始数据文件。"""
    names = [name for name in os.listdir(path) if re.match(r"collect_data\d+_\d+_\d+\.txt$", name)]
    if not names:
        raise FileNotFoundError(f"no collect_data txt in {path}")
    return os.path.join(path, sorted(names)[0])


def load_session_tsv(txt_path) -> SessionData:
    """对无副作用的统一解析器提供历史接口兼容。"""
    return _parse_collect_data_tsv(txt_path)


def load_session(session_id: str) -> SessionData:
    """优先读取已有缓存；此历史接口允许写入缓存文件。"""
    bin_path = config.CACHE_DIR / "sessions" / f"{session_id}.npz"
    if bin_path.exists():
        data = np.load(bin_path)
        return SessionData(
            acc=data["acc"], gyro=data["gyro"], ppg=data["ppg"],
            t_acc=data["t_acc"], t_ppg=data["t_ppg"],
            imu_valid=data["imu_valid"], ppg_valid=data["ppg_valid"],
            meta={"path": str(bin_path), "rows": int(data["rows"]), "row_rate": float(data["row_rate"])},
        )
    session = load_session_tsv(_find_collect_data(str(config.SENSOR_DIR / session_id)))
    try:
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bin_path, acc=session.acc, gyro=session.gyro, ppg=session.ppg,
                            t_acc=session.t_acc, t_ppg=session.t_ppg,
                            imu_valid=session.imu_valid, ppg_valid=session.ppg_valid,
                            row_rate=np.float32(session.meta["row_rate"]), rows=np.int64(session.meta["rows"]))
    except OSError:
        pass
    return session
