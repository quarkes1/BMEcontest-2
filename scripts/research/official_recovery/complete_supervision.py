"""Subject-complete labels and train-only micro policy for the recovery probe."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np

from .consistent_refit import sample_batch
from .event_probe import complete_metrics,partition_truths
from .label_probe import corrected_labels
from scripts.research.representation_v2.crossfit import slice_outer_train
from src.pipeline.event_stack import micro_candidates


def relabel_batch(batch,truths,mapping,*,micro):
    if any(w.sid not in mapping for w in batch.windows) or set(t[0] for t in truths)-set(mapping.values()):
        raise ValueError('window/truth owner outside training mapping')
    old=np.asarray(batch.labels)
    if old.ndim!=1 or len(old)!=len(batch.windows) or not set(np.unique(old)).issubset({-1,0,1}):
        raise ValueError('invalid original window labels')
    subjects=np.asarray([mapping[w.sid] for w in batch.windows]);new=np.empty(len(old),dtype=np.int8)
    intervals=np.asarray([(w.start_ms,w.end_ms) for w in batch.windows],dtype=np.int64).reshape(-1,2)
    for subject in sorted(set(subjects)):
        rows=np.flatnonzero(subjects==subject)
        new[rows]=corrected_labels(intervals[rows],[(a,b) for s,a,b in truths if s==subject],micro=micro)
    stats=dict(rows=len(new),positive=int(np.count_nonzero(new==1)),negative=int(np.count_nonzero(new==0)),
        ignored=int(np.count_nonzero(new<0)),old0_to_new1=int(np.count_nonzero((old==0)&(new==1))),
        old1_to_new0=int(np.count_nonzero((old==1)&(new==0))),
        label_sha256=hashlib.sha256(new.tobytes()).hexdigest())
    return replace(batch,labels=new),stats


def prepare_complete_train(data,train,truths):
    selected=slice_outer_train(data,train);targets=partition_truths(truths,set(train));batches={};stats={}
    for name,batch in (('macro',selected.candidate_train),('micro',selected.micro_candidate_train)):
        if batch is None: raise ValueError('complete supervision requires both scales')
        relabelled,changes=relabel_batch(batch,targets,selected.subject_by_session,micro=name=='micro')
        batches[name],sample=sample_batch(relabelled);stats[name]=dict(labels=changes,sample=sample)
    return replace(selected,window_train=batches['macro'],micro_window_train=batches['micro']),stats


def choose_micro(windows,mapping,truths,thresholds,config):
    options=tuple(float(t) for t in thresholds)
    if not options or len(set(options))!=len(options) or any(not np.isfinite(t) or not 0<=t<=1 for t in options):
        raise ValueError('invalid complete micro threshold grid')
    if any(sid not in mapping for sid in windows) or set(t[0] for t in truths)-set(mapping.values()):
        raise ValueError('micro source/truth owner mismatch')
    budget=3*max(len(truths),1);rows=[]
    for threshold in options:
        candidates=micro_candidates(windows,threshold,config)
        metrics=complete_metrics([c.event for c in candidates],mapping,truths)
        rows.append(SimpleNamespace(threshold=threshold,metrics=metrics,candidate_count=len(candidates),budget=budget))
    feasible=[r for r in rows if r.candidate_count<=budget]
    if feasible: return max(feasible,key=lambda r:(r.metrics['recall'],-r.candidate_count,r.threshold))
    return max(rows,key=lambda r:(r.metrics['f1'],r.metrics['ppv'],r.threshold))
