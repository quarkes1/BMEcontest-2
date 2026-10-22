"""Label-free within-session gyro summaries for a registered macro representation."""
from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np


def augment_batch(macro,micro):
    x,y=np.asarray(macro.features),np.asarray(micro.features)
    if x.shape!=(len(macro.windows),62) or y.shape!=(len(micro.windows),47) or not np.isfinite(y).all():
        raise ValueError('expected original 62D macro and finite 47D micro rows')
    keys=[(w.sid,w.start_ms,w.end_ms) for w in micro.windows]
    if len(set(keys))!=len(keys) or any(w.end_ms<=w.start_ms for w in (*macro.windows,*micro.windows)):
        raise ValueError('duplicate micro or invalid intervals')
    by_micro={};by_macro={}
    for i,w in enumerate(micro.windows): by_micro.setdefault(w.sid,[]).append(i)
    for i,w in enumerate(macro.windows): by_macro.setdefault(w.sid,[]).append(i)
    pooled=np.full((len(x),24),np.nan,dtype=np.float64);counts=np.zeros(len(x),dtype=np.int64)
    for sid,macro_rows in by_macro.items():
        rows=sorted(by_micro.get(sid,()),key=lambda i:(micro.windows[i].start_ms,micro.windows[i].end_ms))
        if not rows: continue
        starts=np.asarray([micro.windows[i].start_ms for i in rows],dtype=np.int64)
        ends=np.asarray([micro.windows[i].end_ms for i in rows],dtype=np.int64)
        if np.any(np.diff(ends)<0): raise ValueError('micro ends must be monotonic')
        values=np.asarray(y[rows,35:47],dtype=np.float64)
        left=np.searchsorted(starts,[macro.windows[i].start_ms for i in macro_rows],side='left')
        right=np.searchsorted(ends,[macro.windows[i].end_ms for i in macro_rows],side='right')
        n=np.maximum(right-left,0);counts[macro_rows]=n
        valid=n>=2;chosen=np.asarray(macro_rows)[valid];lo,hi=left[valid],right[valid]
        if not len(chosen): continue
        for target,a,b in zip(chosen,lo,hi):
            chunk=values[a:b]
            pooled[target,:12]=chunk.mean(axis=0)
            pooled[target,12:]=chunk.std(axis=0,ddof=0)
    return replace(macro,features=np.concatenate([x,pooled.astype(np.float32)],axis=1)),counts


def augment_data(data):
    if data.micro_candidate_train is None: raise ValueError('full micro source required')
    batches={};stats={}
    for name,batch in (('window_train',data.window_train),('candidate_train',data.candidate_train)):
        batches[name],counts=augment_batch(batch,data.micro_candidate_train)
        stats[name]=dict(rows=len(counts),paired=int(np.count_nonzero(counts>=2)),
            missing=int(np.count_nonzero(counts<2)),micro_count_min=int(counts.min()) if len(counts) else 0,
            micro_count_max=int(counts.max()) if len(counts) else 0,
            feature_sha256=hashlib.sha256(np.ascontiguousarray(batches[name].features).tobytes()).hexdigest())
    return replace(data,**batches),dict(raw_width=86,model_width=87,pooling='mean12,std12',batches=stats)
