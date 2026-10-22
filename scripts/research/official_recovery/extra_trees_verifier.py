"""Fixed randomized nonlinear readout; all fitting uses producer-training rows."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.pipeline import runner


def estimator(seed):
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('model',ExtraTreesClassifier(n_estimators=512,max_depth=8,min_samples_leaf=3,
            max_features=.5,class_weight='balanced',bootstrap=False,n_jobs=8,random_state=seed))])


def fit_verifiers(x,y,z,config,regularization,blend):
    if not 0<=blend<=1 or set(np.asarray(y).tolist())!={0,1}:
        raise ValueError('binary verifier targets and valid blend required')
    components={}
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for name,model in [('lr',runner._verifier_estimator(config.seed+3,regularization)),
                           ('lgbm',estimator(config.seed+23))]:
            model.fit(x,y)
            components[name]=runner._positive_probability(model,z) if len(z) else np.empty(0)
    p=blend*components['lr']+(1-blend)*components['lgbm']
    if not np.isfinite(p).all(): raise ValueError('nonfinite verifier probability')
    return p,components
