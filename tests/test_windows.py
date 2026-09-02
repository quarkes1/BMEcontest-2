# -*- coding: utf-8 -*-
import numpy as np
from src.data.windows import time_iou, iter_window_labels
from src.data.loader import SessionData

def test_time_iou():
    assert time_iou(0, 10, 5, 15) == 0.5 / 1.5          # 交 5 / 并 15
    assert time_iou(0, 10, 20, 30) == 0.0
    assert time_iou(0, 10, 0, 10) == 1.0

def test_iter_window_labels_pos_neg_gray():
    N = 525 * 3 + 1
    t_acc = np.arange(N) * 10   # 每行 10ms -> 105 行 = 1.05s
    s = SessionData(
        acc=np.zeros((3, N), dtype=np.float32), gyro=np.zeros((3, N), dtype=np.float32),
        ppg=np.zeros((44, N), dtype=np.float32),
        t_acc=t_acc.astype(np.int64), t_ppg=np.full(N, -1, dtype=np.int64),
        imu_valid=np.ones(N, dtype=bool), ppg_valid=np.zeros(N, dtype=bool),
        meta={"row_rate": 100.0})
    meals = [(0, 3000)]   # 0~3s 用餐
    out = list(iter_window_labels(s, meals, window_rows=525, stride_rows=105))
    labels = [w["label"] for w in out]
    # 窗口0: 0~5.24s，与 0~3s IoU=3000/5240=0.57>=0.5 -> 正
    # 窗口1/2: 与餐部分重叠 -> 灰区跳过；窗口3: 3.15~8.39s 不重叠 -> 负
    assert labels[0] == 1
    assert labels[1] == 0
    assert len(out) >= 2
