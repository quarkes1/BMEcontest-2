"""Probe metrics and registered Q1-vs-Q0 engineering gate."""

import numpy as np

from scripts.research.representation_v2.evaluate import probe_metrics, select_p2_branch


def test_single_class_ap_is_undefined_and_positive_baseline_is_recorded():
    result = probe_metrics(np.array([1, 1], dtype=np.int8), np.array([0.3, 0.7]))
    assert result["average_precision"] is None
    assert result["always_positive_f1"] == 1.0
    assert result["candidate_count"] == 2


def test_p2_gate_requires_four_wrapper_wins_and_log_loss_guard():
    wrappers = [
        {"q0": {"average_precision": .40, "log_loss": .50},
         "q1": {"average_precision": .44, "log_loss": .51}}
        for _ in range(4)
    ] + [{"q0": {"average_precision": .40, "log_loss": .50},
          "q1": {"average_precision": .39, "log_loss": .51}}]
    chosen = select_p2_branch(wrappers)
    assert chosen["status"] == "P2_SUPPORTED_BRANCH"
    assert chosen["selected_arm"] == "q1"
    wrappers[0]["q1"]["log_loss"] = .9
    assert select_p2_branch(wrappers)["status"] == "P2_NO_SUPPORTED_BRANCH"


def test_p2_near_equal_ap_uses_registered_arm_order():
    wrappers = [{"q0": {"average_precision": .40, "log_loss": .50},
                 "q1": {"average_precision": .45, "log_loss": .51},
                 "q2": {"average_precision": .4500005, "log_loss": .51}}
                for _ in range(5)]
    assert select_p2_branch(wrappers)["selected_arm"] == "q1"
