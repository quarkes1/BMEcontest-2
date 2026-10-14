"""Convex offset-logistic model; base log-odds coefficient is fixed at one."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def logit_prior(base):
    base = np.asarray(base, float)
    if not np.isfinite(base).all() or np.any((base < 0) | (base > 1)):
        raise ValueError('invalid frozen prior probability')
    p = np.clip(base, 1e-6, 1-1e-6)
    return np.log(p)-np.log1p(-p)


def objective(beta, x, y, prior, weights):
    z = prior+x@beta
    loss = float(np.sum(weights*(np.logaddexp(0, z)-y*z)) + 8*np.dot(beta, beta))
    gradient = x.T@(weights*(expit(z)-y))+16*beta
    return loss, gradient


def design(features, center, scale):
    x = np.asarray(features, float)
    center, scale = np.asarray(center), np.asarray(scale)
    x = np.where(np.isfinite(x), x, center)
    return np.column_stack([np.ones(len(x)), np.clip((x-center)/scale, -3, 3)])


def fit(features, labels, base, groups):
    x, y, base, groups = np.asarray(features, float), np.asarray(labels), np.asarray(base, float), np.asarray(groups)
    if x.ndim != 2 or y.shape != base.shape or y.shape != groups.shape or len(x) != len(y):
        raise ValueError('residual training alignment differs')
    prior = logit_prior(base)
    if not np.isin(y, [0, 1]).all() or np.isinf(x).any():
        raise ValueError('invalid residual features/labels')
    counts = dict(positive=int((y == 1).sum()), negative=int((y == 0).sum()))
    model = dict(active=counts['positive'] >= 10 and counts['negative'] >= 30, support=counts, width=x.shape[1],
                 fit_subjects=sorted(map(str, set(groups))))
    if not model['active']:
        return model
    center, scale = [], []
    for col in x.T:
        finite = col[np.isfinite(col)]
        center.append(float(np.median(finite)) if len(finite) else 0.)
        scale.append(max(float(np.std(finite)), 1e-6) if len(finite) else 1.)
    matrix = design(x, center, scale)
    unique, inverse, n = np.unique(groups, return_inverse=True, return_counts=True)
    weights = len(y)/(len(unique)*n[inverse])
    result = minimize(objective, np.zeros(matrix.shape[1]), args=(matrix, y, prior, weights), jac=True,
                      method='L-BFGS-B', options=dict(maxiter=500, ftol=1e-12, gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('offset logistic optimization failed: '+str(result.message))
    model.update(center=center, scale=scale, beta=result.x.tolist(), iterations=int(result.nit),
                 objective=float(result.fun), ridge=16., group_weight='N/(G*N_group)')
    return model


def predict(model, features, base, valid):
    x, base, valid = np.asarray(features, float), np.asarray(base, float), np.asarray(valid, bool)
    if x.ndim != 2 or len(x) != len(base) or valid.shape != base.shape:
        raise ValueError('residual prediction alignment differs')
    prior = logit_prior(base)
    if not model['active']:
        return base.copy()
    correction = design(x, model['center'], model['scale'])@np.asarray(model['beta'])
    return np.where(valid & (correction != 0), expit(prior+correction), base)
