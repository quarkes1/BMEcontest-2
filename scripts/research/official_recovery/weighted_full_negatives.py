"""Retain all old-label macro negatives with the sampled per-session mass."""
from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np

from .negative_clean import prepare_train
from scripts.research.representation_v2.crossfit import slice_outer_train


def full_negative_weights(labels,sessions):
    y,s=np.asarray(labels),np.asarray(sessions)
    if (y.ndim!=1 or s.ndim!=1 or len(y)!=len(s) or not set(np.unique(y)).issubset({-1,0,1})
        or any(not isinstance(v,str) or not v for v in s)):
        raise ValueError('invalid weighted full-window identity or labels')
    weights=np.ones(len(y),dtype=np.float64);records=[]
    for sid in sorted(set(s)):
        rows=s==sid;positive=int(np.count_nonzero(rows&(y==1)));negative=int(np.count_nonzero(rows&(y==0)))
        retained=min(negative,max(1,3*positive));weight=retained/negative if negative else None
        if negative: weights[rows&(y==0)]=weight
        records.append(dict(sid=sid,positive=positive,negative=negative,effective_negative=retained,negative_weight=weight))
    return weights,dict(fit_rows=int(np.count_nonzero(y>=0)),ignored_rows=int(np.count_nonzero(y<0)),
        effective_positive=float(weights[y==1].sum()),effective_negative=float(weights[y==0].sum()),
        weight_min=float(weights.min()) if len(weights) else None,weight_max=float(weights.max()) if len(weights) else None,
        weight_sha256=hashlib.sha256(weights.astype('<f8').tobytes()).hexdigest(),sessions=records)


def prepare_weighted_train(data,train,truths):
    prepared,sampling=prepare_train(data,train,truths,'A')
    full=slice_outer_train(data,train).candidate_train
    weights,stats=full_negative_weights(full.labels,np.asarray([w.sid for w in full.windows]))
    if (not np.isclose(stats['effective_positive'],sampling['macro']['sample']['positive'],rtol=0,atol=1e-8)
        or not np.isclose(stats['effective_positive']+stats['effective_negative'],sampling['macro']['sample']['rows'],rtol=0,atol=1e-8)):
        raise ValueError('weighted effective mass differs from canonical sample')
    return replace(prepared,window_train=full),sampling,weights,stats
