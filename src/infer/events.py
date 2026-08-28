# -*- coding: utf-8 -*-
"""窗口打分 → 事件列表（spec §3 L4 后处理，训练评估与推理共用）：
平滑（可选中值滤波）→ 阈值 → 合并间隔 <merge_gap_s → 过滤持续 <min_dur_s → 边界膨胀 ±dilation_s。"""
import numpy as np
from scipy.ndimage import median_filter
import src.config as config

def windows_to_events(probs, t0_ms, t1_ms, threshold,
                      merge_gap_s=config.EVENT_MERGE_GAP_SEC,
                      min_dur_s=config.EVENT_MIN_DUR_SEC,
                      dilation_s=config.BOUNDARY_DILATION_SEC,
                      smooth_win=31):
    """probs: (n,) 窗口分数；t0_ms/t1_ms: 窗口起止（毫秒，单调不减）。
    smooth_win: 概率序列中值平滑窗（窗口数，1s 步长 → 31 ≈ 31s；0/1 关闭）。
    返回 [(start_ms, end_ms), ...]。"""
    if smooth_win > 1 and len(probs) > smooth_win:
        probs = median_filter(probs, size=smooth_win)   # 拉平窗内震荡，利于形成长连续段
    hits = probs >= threshold
    events = []
    cur_s = cur_e = None
    for i in range(len(hits)):
        if not hits[i]:
            continue
        if cur_s is None:
            cur_s, cur_e = int(t0_ms[i]), int(t1_ms[i])
        elif t0_ms[i] - cur_e <= merge_gap_s * 1000:
            cur_e = max(cur_e, int(t1_ms[i]))
        else:
            events.append((cur_s, cur_e)); cur_s, cur_e = int(t0_ms[i]), int(t1_ms[i])
    if cur_s is not None:
        events.append((cur_s, cur_e))
    out = []
    for s, e in events:
        if (e - s) / 1000.0 < min_dur_s:
            continue
        s2 = max(0, s - dilation_s * 1000)
        e2 = e + dilation_s * 1000
        out.append((int(s2), int(e2)))
    return out
