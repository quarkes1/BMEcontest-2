"""A2 frozen 4-segment representation and subject-weighted linear readout."""
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.research.representation_v2.models import sample_time_coverage


def subject_weights(subjects):
    groups = np.asarray(subjects)
    if groups.ndim != 1 or not len(groups):
        raise ValueError('nonempty subject vector required')
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return len(groups) / (len(counts) * counts[inverse])


def pool_ordered(embeddings, bag):
    z = np.asarray(embeddings)
    if z.shape != (len(bag.token_mask), 16) or not np.isfinite(z).all():
        raise ValueError('finite aligned token embeddings required')
    placement = np.clip((bag.token_center_ms - bag.start_ms) * 4 // (bag.end_ms-bag.start_ms), 0, 3)
    blocks = []
    for segment in range(4):
        rows = z[(placement == segment) & bag.token_mask]
        blocks.extend([rows.mean(0), rows.std(0)] if len(rows) else [np.zeros(16), np.zeros(16)])
    coverage = sample_time_coverage(torch.from_numpy(bag.sample_mask[None]),
                                   torch.tensor([[bag.start_ms, bag.end_ms]]), bins=4)[0].numpy()
    return np.concatenate([*blocks, coverage, [np.log1p((bag.end_ms-bag.start_ms)/1000)]])


def encode_candidates(encoder, bags, center, scale, *, device='cpu', chunk=512):
    encoder.to(device).eval()
    rows = []
    with torch.inference_mode():
        for bag in bags:
            z = np.zeros((len(bag.token_mask), 16), np.float32)
            indices = np.flatnonzero(bag.token_mask)
            for left in range(0, len(indices), chunk):
                idx = indices[left:left+chunk]
                mask = bag.sample_mask[idx]
                x = np.where(mask[:,None,:], (bag.values[idx]-center[None,:,None])/scale[None,:,None], 0.)
                z[idx] = encoder(torch.as_tensor(x, dtype=torch.float32, device=device),
                                 torch.as_tensor(mask, device=device)).cpu().numpy()
            rows.append(pool_ordered(z, bag))
    return np.asarray(rows).reshape(-1, 133)


def fit_readout(features, labels, subjects):
    x, y = np.asarray(features), np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or len(subjects) != len(y) or not np.isfinite(x).all():
        raise ValueError('readout rows must align and be finite')
    if set(y.tolist()) != {0, 1}:
        raise ValueError('readout requires both classes')
    weights = subject_weights(subjects)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = LogisticRegression(C=.1, solver='lbfgs', max_iter=2000, tol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(x), y, sample_weight=weights)
    return {'scaler': scaler, 'model': model, 'prior': float(np.average(y, weights=weights))}


def predict_readout(fitted, features):
    return fitted['model'].decision_function(fitted['scaler'].transform(features))


def readout_payload(fitted):
    return {'mean': fitted['scaler'].mean_.tolist(), 'scale': fitted['scaler'].scale_.tolist(),
            'coefficient': fitted['model'].coef_.tolist(), 'intercept': fitted['model'].intercept_.tolist(),
            'classes': fitted['model'].classes_.tolist(), 'prior': fitted['prior'],
            'iterations': fitted['model'].n_iter_.tolist()}
