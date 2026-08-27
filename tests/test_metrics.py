# -*- coding: utf-8 -*-
import pytest
from src.eval.metrics import event_iou, match_events, compute_metrics

def test_event_iou():
    assert event_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert event_iou((0, 10), (100, 200)) == 0.0

def test_match_events_one_to_one():
    preds = [(0, 100), (0, 100)]     # 两个重复预测
    trues = [(0, 100), (500, 600)]
    matched, unmatched_p, unmatched_t = match_events(preds, trues, iou_thr=0.25)
    assert matched == [0]            # 贪心：只匹配第一个真值
    assert unmatched_p == [1]
    assert unmatched_t == [1]

def test_compute_metrics_perfect_and_missing():
    m = compute_metrics([(0, 100), (200, 300)], [(0, 100), (200, 300)])
    assert m["f1"] == 1.0 and m["mae_start_s"] == 0.0
    m2 = compute_metrics([], [(0, 100)])
    assert m2["sensitivity"] == 0.0 and m2["ppv"] == 0.0 and m2["f1"] == 0.0
