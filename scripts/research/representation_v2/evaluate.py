"""Candidate-probe metrics and the preregistered P2 engineering gate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def probe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not set(y.tolist()) <= {0, 1}:
        raise ValueError("probe requires aligned nonempty binary labels and probabilities")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probe probabilities must be finite and in [0, 1]")
    positive = int(y.sum())
    negative = int(len(y) - positive)
    chosen = p >= .5
    tp = int(np.count_nonzero(chosen & (y == 1)))
    fp = int(np.count_nonzero(chosen & (y == 0)))
    fn = positive - tp
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "candidate_count": len(y), "positive_count": positive, "negative_count": negative,
        "prevalence": positive / len(y),
        "always_positive_f1": 2 * positive / (len(y) + positive) if positive else 0.0,
        "always_negative_f1": 0.0,
        "average_precision": float(average_precision_score(y, p)) if positive and negative else None,
        "roc_auc": float(roc_auc_score(y, p)) if positive and negative else None,
        "log_loss": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))),
        "brier": float(np.mean((p - y) ** 2)),
        "precision_05": tp / (tp + fp) if tp + fp else 0.0,
        "recall_05": tp / positive if positive else 0.0,
        "f1_05": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp_05": tp, "fp_05": fp, "fn_05": fn,
    }


def select_p2_branch(wrappers: Sequence[dict[str, dict[str, float | int | None]]]) -> dict[str, object]:
    """Macro wrapper AP comparison; missing transfer arms are not invented."""
    if len(wrappers) != 5 or any("q0" not in row for row in wrappers):
        raise ValueError("P2 gate requires five complete Q0 wrapper summaries")
    arms = tuple(arm for arm in ("q1", "q2", "q3") if all(arm in row for row in wrappers))
    q0_ap = np.asarray([row["q0"]["average_precision"] for row in wrappers], dtype=float)
    q0_loss = np.asarray([row["q0"]["log_loss"] for row in wrappers], dtype=float)
    if not np.isfinite(q0_ap).all() or not np.isfinite(q0_loss).all():
        raise ValueError("undefined or non-finite wrapper control metric")
    diagnostics = {}
    eligible = []
    for arm in arms:
        ap = np.asarray([row[arm]["average_precision"] for row in wrappers], dtype=float)
        loss = np.asarray([row[arm]["log_loss"] for row in wrappers], dtype=float)
        if not np.isfinite(ap).all() or not np.isfinite(loss).all():
            raise ValueError(f"undefined or non-finite wrapper metric: {arm}")
        delta = float(np.mean(ap) - np.mean(q0_ap))
        wins = int(np.count_nonzero(ap > q0_ap))
        loss_delta = float(np.mean(loss) - np.mean(q0_loss))
        passed = delta >= .02 - 1e-12 and wins >= 4 and loss_delta <= .02 + 1e-12
        diagnostics[arm] = {"macro_ap": float(np.mean(ap)), "ap_delta_vs_q0": delta,
                            "positive_wrappers": wins, "log_loss_delta_vs_q0": loss_delta,
                            "eligible": passed}
        if passed:
            eligible.append(arm)
    # All registered branches share one inference architecture.  Within the
    # predeclared 1e-6 AP tie band, Q1/Q2/Q3 order is the deterministic tie.
    selected = None
    for arm in eligible:
        if selected is None or diagnostics[arm]["macro_ap"] > diagnostics[selected]["macro_ap"] + 1e-6:
            selected = arm
    return {"status": "P2_SUPPORTED_BRANCH" if selected else "P2_NO_SUPPORTED_BRANCH",
            "selected_arm": selected, "q0_macro_ap": float(np.mean(q0_ap)), "arms": diagnostics}
