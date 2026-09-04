# -*- coding: utf-8 -*-
"""全局配置：路径与常量。所有模块从本文件取数。"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent          # D:\BMEtest
DATA_DIR = ROOT_DIR / "Data"
SENSOR_DIR = DATA_DIR / "t_zsstnnrj_sensororiginaldata_system附件0826_1857"
CACHE_DIR = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"    # FD 预训练权重（阶段二）

RANDOM_SEED = 42
IMU_ROW_RATE = 105        # IMU 有效行率（行空间即时间网格, 行/s）
WINDOW_ROWS = 525         # 5s 窗口
STRIDE_ROWS = 105         # 1s 步长
PPG_WINDOW_ROWS = 24 * 30 # L3b 30s 窗口
IOU_POS = 0.5             # 窗口正样本：与事件重叠占窗口时长比例下限（非事件级 IoU）
IOU_EVENT = 0.25          # 事件匹配 IoU 阈值（组委会口径）
EVENT_MERGE_GAP_SEC = 30
EVENT_MIN_DUR_SEC = 45
BOUNDARY_DILATION_SEC = 6

SCENES = ("dominant", "nondominant")
