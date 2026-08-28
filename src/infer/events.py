# -*- coding: utf-8 -*-
"""窗口打分 → 事件列表（spec §3 L4 后处理，训练评估与推理共用）：
合并间隔 <merge_gap_s → 过滤持续 <min_dur_s → 边界膨胀 ±dilation_s。"""
import src.config as config

def windows_to_events(probs, t0_ms, t1_ms, threshold,
                      merge_gap_s=config.EVENT_MERGE_GAP_SEC,
                      min_dur_s=config.EVENT_MIN_DUR_SEC,
                      dilation_s=config.BOUNDARY_DILATION_SEC):
    """probs: (n,) 窗口分数；t0_ms/t1_ms: 窗口起止（毫秒，单调不减）。
    返回 [(start_ms, end_ms), ...]。"""
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
