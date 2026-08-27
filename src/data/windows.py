# -*- coding: utf-8 -*-
"""窗口切分与 IoU 标签构造（行空间，行率 ~105Hz）。"""
from typing import Iterator, List, Tuple
import src.config as config
from src.data.loader import SessionData

def time_iou(t0, t1, before, after) -> float:
    inter = max(0.0, min(t1, after) - max(t0, before))
    union = max(t1, after) - min(t0, before)
    return inter / union if union > 0 else 0.0

def _window_times(t_acc, start, end):
    seg = t_acc[start:end]
    seg = seg[seg > 0]
    if len(seg) == 0:
        return None
    return int(seg[0]), int(seg[-1])

def iter_window_labels(session: SessionData, meals: List[Tuple[int, int]],
                       window_rows: int = None, stride_rows: int = None) -> Iterator[dict]:
    """窗口遍历：label=1（与任一餐 IoU>=0.5）、label=0（与所有餐 IoU==0）、灰区跳过。"""
    W = window_rows or config.WINDOW_ROWS
    S = stride_rows or config.STRIDE_ROWS
    N = session.acc.shape[1]
    for i in range(0, N - W + 1, S):
        times = _window_times(session.t_acc, i, i + W)
        if times is None:
            continue
        t0, t1 = times
        label = None
        for before, after in meals:
            iou = time_iou(t0, t1, before, after)
            if iou >= config.IOU_POS:
                label = 1
                break
            if iou > 0:
                label = None          # 灰区：丢弃，不参与训练
                break
        if label is None and any(time_iou(t0, t1, b, a) > 0 for b, a in meals):
            continue                   # 灰区跳过
        if label is None:
            label = 0
        yield {"start_row": i, "end_row": i + W, "t0_ms": t0, "t1_ms": t1, "label": label}
