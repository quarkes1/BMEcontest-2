import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from src.eval.metrics import event_iou
from src.pipeline import runner


def sample_weights(candidates, labels, groups, truths, owners):
    y, g = np.asarray(labels), np.asarray(groups)
    if y.shape != (len(candidates),) or g.shape != y.shape or set(y.tolist()) != {0, 1}:
        raise ValueError('binary aligned training ledger required')
    if any(t.sid not in owners for t in truths):
        raise ValueError('unknown training truth owner')
    assignment = []
    for c, label, group in zip(candidates, y, g):
        e = c.event
        if e.sid not in owners or owners[e.sid] != group:
            raise ValueError('candidate owner differs')
        overlaps = [(event_iou(e.interval, t.interval), j) for j, t in enumerate(truths) if e.sid == t.sid]
        best = max(overlaps, key=lambda pair: (pair[0], -pair[1])) if overlaps else (0, -1)
        if int(best[0] >= .25) != label:
            raise ValueError('training label differs from best-IoU assignment')
        assignment.append(best[1] if label else -1)
    assignment = np.asarray(assignment); w = np.zeros(len(y), float)
    for subject in sorted(set(g)):
        pos = (g == subject)&(y == 1); neg = (g == subject)&(y == 0)
        events = set(assignment[pos])
        for j in events:
            rows = pos&(assignment == j)
            w[rows] = 1/(len(events)*rows.sum())
        if neg.any():
            w[neg] = 1/neg.sum()
    for label in (0, 1):
        rows = y == label
        w[rows] *= rows.sum()/w[rows].sum()
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('invalid event-balanced weight')
    return w


def probabilities(x, y, validation, weights, *, seed):
    x, y, z, w = np.asarray(x), np.asarray(y), np.asarray(validation), np.asarray(weights)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != 116 or z.shape[1] != 116 or w.shape != y.shape or len(x) != len(y):
        raise ValueError('weighted verifier alignment differs')
    if not np.isfinite(w).all() or np.any(w <= 0) or set(y.tolist()) != {0, 1}:
        raise ValueError('invalid weighted verifier labels/weights')
    values = []
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for model in (runner._verifier_estimator(seed+20, .1), runner._verifier_lgbm_estimator(seed+21)):
            model.fit(x, y, model__sample_weight=w)
            values.append(runner._positive_probability(model, z))
    return .5*values[0]+.5*values[1]
