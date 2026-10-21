"""Train-role-only censoring of high-confidence contradictory negatives."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .consistent_refit import sample_batch
from .event_probe import full_candidate_labels,partition_truths
from .label_probe import corrected_labels
from scripts.research.representation_v2.crossfit import slice_outer_train


def _counts(labels,masked):
    return dict(rows=len(labels),positive=int(np.count_nonzero(labels==1)),
        negative=int(np.count_nonzero(labels==0)),ignored=int(np.count_nonzero(labels==-1)),
        censored_negative=int(np.count_nonzero(masked)))


def censor_window_batch(batch,truths,mapping,*,micro):
    old=np.asarray(batch.labels)
    if old.ndim!=1 or len(old)!=len(batch.windows) or not set(np.unique(old)).issubset({-1,0,1}):
        raise ValueError('invalid window label array')
    if any(w.sid not in mapping for w in batch.windows): raise ValueError('unknown window session')
    labels=old.copy();subjects=np.asarray([mapping[w.sid] for w in batch.windows])
    intervals=np.asarray([(w.start_ms,w.end_ms) for w in batch.windows],dtype=np.int64).reshape(-1,2)
    mask=np.zeros(len(old),dtype=bool)
    for subject in sorted(set(subjects)):
        rows=np.flatnonzero(subjects==subject)
        meals=[(start,end) for owner,start,end in truths if owner==subject]
        complete=corrected_labels(intervals[rows],meals,micro=micro)
        mask[rows]=(old[rows]==0)&(complete==1)
    labels[mask]=-1
    return replace(batch,labels=labels),_counts(labels,mask)


def censor_candidate_labels(candidates,labels,truths,mapping):
    old=np.asarray(labels)
    if old.ndim!=1 or len(old)!=len(candidates) or not set(np.unique(old)).issubset({0,1}):
        raise ValueError('invalid candidate label array')
    complete=full_candidate_labels(candidates,mapping,truths)
    mask=(old==0)&(complete==1);updated=old.copy();updated[mask]=-1
    return updated,_counts(updated,mask)


def prepare_train(universe,train,truths,arm):
    if arm not in ('A','W','E'): raise ValueError('unknown negative-clean arm')
    data=slice_outer_train(universe,train)
    targets=partition_truths(truths,set(train));batches={};stats={}
    for name,batch in [('macro',data.candidate_train),('micro',data.micro_candidate_train)]:
        if batch is None: raise ValueError('micro path required')
        if arm=='W': raw,masked=censor_window_batch(batch,targets,data.subject_by_session,micro=name=='micro')
        else: raw=batch;masked=_counts(np.asarray(batch.labels),np.zeros(len(batch.labels),dtype=bool))
        sampled,details=sample_batch(raw)
        batches[name]=sampled;stats[name]=dict(mask=masked,sample=details)
    return replace(data,window_train=batches['macro'],micro_window_train=batches['micro']),stats
