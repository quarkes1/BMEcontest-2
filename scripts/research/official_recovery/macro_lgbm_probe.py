"""Paired macro-window classifier screen, never an event or official score."""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .label_probe import load_pools,score_metrics,validate_partitions
from .negative_clean_probe import _read,_write,protected_sources
from .geometry_probe import inputs,MANIFEST
from .scope_audit import _sha
from src.pipeline import runner
from src.pipeline.micro_cache import sample_training_rows,MICRO_SAMPLE_SEED

SPEC='docs/official_recovery_macro_lgbm_v1.md'


def macro_estimator(seed):
    model=runner._micro_window_estimator(seed)
    model.set_params(model__class_weight=None)
    return model


def split_rows(pool,train,score):
    known=set(pool['subjects']);train,score=set(train),set(score)
    if not train or not score or train&score or (train|score)-known: raise ValueError('invalid train/score roles')
    if not set(np.unique(pool['old'])).issubset({-1,0,1}) or not set(np.unique(pool['new'])).issubset({-1,0,1}):
        raise ValueError('invalid window labels')
    train_rows=np.flatnonzero(np.isin(pool['subjects'],sorted(train)))
    sampled=sample_training_rows(pool['old'][train_rows],pool['sessions'][train_rows],MICRO_SAMPLE_SEED,negative_ratio=3)
    selected=train_rows[sampled]
    testing=np.flatnonzero(np.isin(pool['subjects'],sorted(score))&(pool['new']>=0))
    if set(pool['old'][selected])!={0,1} or set(pool['new'][testing])!={0,1}: raise ValueError('binary training/scoring required')
    return selected,testing


def gate(rows):
    if len(rows)!=3 or {r['fold'] for r in rows}!=set(range(3)): raise ValueError('three unique complete window folds required')
    ap={a:float(np.mean([r['metrics'][a]['ap'] for r in rows])) for a in ('A','L')}
    wins=sum(r['metrics']['L']['ap']>r['metrics']['A']['ap'] for r in rows)
    return dict(passed=bool(ap['L']-ap['A']>=.02-1e-12 and wins>=2),mean_ap=ap,delta_ap=ap['L']-ap['A'],positive_folds=int(wins))


def execute(root,output):
    root,output=Path(root).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();completed=0
    def fingerprints():
        d=inputs(root,'t24');d[SPEC]=_sha(root/SPEC);return d
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(): raise RuntimeError('commit before window screen')
        hashes=fingerprints();protected=protected_sources(root)
        manifest=_read(root/MANIFEST);validate_partitions(manifest)
        _write(output/'plan.json',dict(id='MACRO-LGBM-V1-WINDOW',source_sha256=hashes,protected=protected,
            code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),seed=20260910,
            model_fits=6,window_labels='old',score_labels='complete',wrappers=manifest['wrappers']))
        def guard():
            if hashes!=fingerprints() or protected!=protected_sources(root): raise ValueError('screen inputs/protection changed')
            if time.perf_counter()-start>600 or shutil.disk_usage(root).free<10*1024**3: raise RuntimeError('screen wall/disk budget')
            if runner.peak_working_set_bytes()>8*1024**3: raise RuntimeError('screen memory budget')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>16*1024**2: raise RuntimeError('screen evidence budget')
        guard();pool=load_pools(root,manifest)['slide'];records=[]
        if pool['features'].shape[1]!=63: raise ValueError('macro input width differs')
        for w in manifest['wrappers']:
            guard();train=set(w['train']);health=set(w['health']);fit,score=split_rows(pool,train,health)
            row=dict(fold=w['id'],train=sorted(train),score=sorted(health),train_rows=len(fit),train_positive=int(np.count_nonzero(pool['old'][fit]==1)),
                train_identity_sha256=hashlib.sha256(fit.astype('<i8').tobytes()).hexdigest(),health_rows=len(score),metrics={},subjects=[])
            probabilities={}
            for arm,factory in (('A',runner._window_estimator),('L',macro_estimator)):
                before=time.perf_counter()
                with threadpool_limits(limits=8):
                    model=factory(20260910).fit(pool['features'][fit],pool['old'][fit])
                    p=runner._positive_probability(model,pool['features'][score])
                probabilities[arm]=p;row['metrics'][arm]=score_metrics(pool['new'][score],p)
                row['metrics'][arm]['elapsed_seconds']=time.perf_counter()-before
                row.setdefault('graphs',{})[arm]={'score':dict(fit_subjects=sorted(set(pool['subjects'][fit])),parents=[])}
                print(f"macro-screen fold{w['id']} {arm} AP={row['metrics'][arm]['ap']:.6f}",flush=True)
                del model;completed+=1;gc.collect()
            for subject in sorted(health):
                mask=pool['subjects'][score]==subject;y=pool['new'][score][mask]
                row['subjects'].append(dict(subject=subject,rows=int(mask.sum()),positive=int(np.count_nonzero(y==1)),
                    metrics={a:score_metrics(y,p[mask]) if set(y)=={0,1} else None for a,p in probabilities.items()}))
            path=output/f"fold{w['id']}_scores.npz"
            np.savez_compressed(path,**probabilities,master_rows=score,label=pool['new'][score])
            row['score_sha256']=_sha(path);records.append(row);_write(output/f"fold{w['id']}_report.json",row)
        guard()
        result=dict(state='COMPLETE',evidence_kind='WINDOW_CLASSIFIER_SCREEN_NOT_EVENT_F1',**gate(records),
            model_fits=completed,outer_calls=0,protected_unchanged=True,elapsed_seconds=time.perf_counter()-start,
            peak_working_set_bytes=runner.peak_working_set_bytes())
        _write(output/'summary.json',result);_write(output/'state.json',{'state':'COMPLETE'});return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_model_fits=completed));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(execute(Path(__file__).resolve().parents[3],args.output),flush=True)
