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
    """窗口遍历：label=1（与餐的重叠 >= 窗口时长的 IOU_POS 比例）、
    label=0（与所有餐零重叠）、灰区（0 < 重叠比例 < IOU_POS）跳过。
    注意：5s 窗口 vs ~15min 事件，IoU 恒 <0.01，必须用"重叠占窗口比例"规则。"""
    W = window_rows or config.WINDOW_ROWS
    S = stride_rows or config.STRIDE_ROWS
    N = session.acc.shape[1]
    for i in range(0, N - W + 1, S):
        times = _window_times(session.t_acc, i, i + W)
        if times is None:
            continue
        t0, t1 = times
        win_dur = t1 - t0
        if win_dur <= 0:
            continue
        label = None
        for before, after in meals:
            overlap = min(t1, after) - max(t0, before)
            if overlap <= 0:
                continue
            frac = overlap / win_dur          # 重叠占窗口时长比例
            if frac >= config.IOU_POS:
                label = 1
                break
            label = None                       # 灰区：丢弃，不参与训练
            break
        if label is None and any(min(t1, a) - max(t0, b) > 0 for b, a in meals):
            continue                            # 灰区跳过
        if label is None:
            label = 0
        yield {"start_row": i, "end_row": i + W, "t0_ms": t0, "t1_ms": t1, "label": label}
