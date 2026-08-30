# -*- coding: utf-8 -*-
"""事件级评估：IoU>0.25 一对一贪心匹配 -> F1 + 起止 MAE（组委会口径）。

⚠️ 匹配作用域：按受试者（externalid）分组匹配——跨受试者事件不得匹配他人餐次
（实测多人同时段采集，全局匹配会把他人手腕活动误算为自己的 TP，2026-08-30 修正）。
"""
import src.config as config

def event_iou(pred, true) -> float:
    p0, p1 = pred; t0, t1 = true
    inter = max(0.0, min(p1, t1) - max(p0, t0))
    union = max(p1, t1) - min(p0, t0)
    return inter / union if union > 0 else 0.0

def _greedy_pairs(pred_events, true_events, iou_thr):
    """按 IoU 降序贪心一对一匹配，返回 (matched_pairs[(pi,ti)], used_p, used_t)。"""
    candidates = []
    for pi, p in enumerate(pred_events):
        for ti, t in enumerate(true_events):
            iou = event_iou(p, t)
            if iou >= iou_thr:
                candidates.append((iou, pi, ti))
    candidates.sort(key=lambda x: -x[0])
    used_p, used_t = set(), set()
    matched = []
    for iou, pi, ti in candidates:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi); used_t.add(ti)
        matched.append((pi, ti))
    return matched, used_p, used_t

def match_events(pred_events, true_events, iou_thr=None):
    iou_thr = iou_thr if iou_thr is not None else config.IOU_EVENT
    matched, used_p, used_t = _greedy_pairs(pred_events, true_events, iou_thr)
    matched_true = [ti for _, ti in matched]
    unmatched_pred = [i for i in range(len(pred_events)) if i not in used_p]
    unmatched_true = [i for i in range(len(true_events)) if i not in used_t]
    return matched_true, unmatched_pred, unmatched_true

def compute_metrics(pred_events, true_events, iou_thr=None) -> dict:
    iou_thr = iou_thr if iou_thr is not None else config.IOU_EVENT
    matched, used_p, used_t = _greedy_pairs(pred_events, true_events, iou_thr)
    n_true, n_pred, n_tp = len(true_events), len(pred_events), len(matched)
    sens = n_tp / n_true if n_true else 0.0
    ppv = n_tp / n_pred if n_pred else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else 0.0
    mae_s = mae_e = None
    if matched:
        mae_s = sum(abs(pred_events[pi][0] - true_events[ti][0]) for pi, ti in matched) / n_tp / 1000.0
        mae_e = sum(abs(pred_events[pi][1] - true_events[ti][1]) for pi, ti in matched) / n_tp / 1000.0
    return {"n_true": n_true, "n_pred": n_pred, "n_tp": n_tp,
            "n_fp": n_pred - n_tp, "n_fn": n_true - n_tp,
            "sensitivity": sens, "ppv": ppv, "f1": f1,
            "mae_start_s": mae_s, "mae_end_s": mae_e}


def compute_metrics_by_subject(pred_sid, true_sid, subject_of, iou_thr=None):
    """受试者级评估：pred_sid/true_sid = [(sid, (s_ms, e_ms)), ...]，subject_of: sid→externalid。
    按受试者分组，组内一对一贪心匹配（同一受试者可跨会话匹配），再汇总 TP/FP/FN。"""
    iou_thr = iou_thr if iou_thr is not None else config.IOU_EVENT
    groups = {}
    for sid, ev in pred_sid:
        groups.setdefault(subject_of(sid), {"pred": [], "true": []})["pred"].append(ev)
    for sid, ev in true_sid:
        groups.setdefault(subject_of(sid), {"pred": [], "true": []})["true"].append(ev)
    n_tp = n_pred = n_true = 0
    matched_pairs = []
    for ext, g in groups.items():
        matched, _, _ = _greedy_pairs(g["pred"], g["true"], iou_thr)
        n_tp += len(matched)
        n_pred += len(g["pred"])
        n_true += len(g["true"])
    sens = n_tp / n_true if n_true else 0.0
    ppv = n_tp / n_pred if n_pred else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else 0.0
    return {"n_true": n_true, "n_pred": n_pred, "n_tp": n_tp,
            "n_fp": n_pred - n_tp, "n_fn": n_true - n_tp,
            "sensitivity": sens, "ppv": ppv, "f1": f1}
