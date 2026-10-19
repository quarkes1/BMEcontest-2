"""Paired, subject-contained window probe for the cross-file meal-label contract."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from .scope_audit import _sha, normalized_id, protection


def corrected_labels(intervals,meals,*,micro):
    spans=np.asarray(intervals,dtype=np.int64).reshape(-1,2)
    truth=np.asarray(meals,dtype=np.int64).reshape(-1,2)
    if np.any(spans[:,1]<=spans[:,0]) or np.any(truth[:,1]<=truth[:,0]):
        raise ValueError('positive intervals required')
    if not len(truth):
        return np.zeros(len(spans),dtype=np.int8)
    start,end=spans[:,0,None],spans[:,1,None]
    overlap=np.maximum(0,np.minimum(end,truth[:,1])-np.maximum(start,truth[:,0])).max(axis=1)
    duration=spans[:,1]-spans[:,0]
    positive=overlap*2>=duration if micro else overlap*2>duration
    distance=np.where(end<=truth[:,0],truth[:,0]-end,
                      np.where(start>=truth[:,1],start-truth[:,1],0)).min(axis=1)
    labels=np.full(len(spans),-1,dtype=np.int8)
    labels[(overlap==0)&(distance>=300000)]=0
    labels[positive]=1
    return labels


def sample_rows(labels,sessions):
    rng=np.random.default_rng(20260901)
    keep=[]
    labels,sessions=np.asarray(labels),np.asarray(sessions)
    for sid in sorted(set(sessions)):
        indices=np.flatnonzero(sessions==sid)
        positive=indices[labels[indices]==1]
        negative=indices[labels[indices]==0]
        n=min(len(negative),max(1,3*len(positive)))
        selected=np.sort(rng.choice(negative,n,replace=False)) if n<len(negative) else negative
        keep.extend(positive.tolist());keep.extend(selected.tolist())
    return np.asarray(sorted(keep),dtype=np.int64)


def validate_partitions(manifest,*,expected_subjects=24):
    identities=manifest['identities']
    subjects={normalized_id(s) for s in identities['t24']}
    forbidden={normalized_id(s) for key in ('v7','outer') for s in identities[key]}
    if len(subjects)!=expected_subjects or subjects&forbidden:
        raise ValueError('invalid partition population')
    coverage=Counter()
    for wrapper in manifest['wrappers']:
        train={normalized_id(s) for s in wrapper['train']}
        health={normalized_id(s) for s in wrapper['health']}
        if train&health or train|health!=subjects or (train|health)&forbidden:
            raise ValueError('invalid partition train/health/forbidden identity')
        coverage.update(health)
    if set(coverage)!=subjects or any(value!=1 for value in coverage.values()):
        raise ValueError('partition health coverage must be exactly once')


def load_pools(root,manifest):
    from src.data.manifests import load_meals,load_sensor_index
    from src.pipeline.features.macro import add_time_prior
    from src.pipeline.event_stack import EventRef
    allowed={normalized_id(s) for s in manifest['identities']['t24']}
    meals=defaultdict(list)
    for row in load_meals().itertuples():
        s=normalized_id(row.externalid)
        if s in allowed:
            meals[s].append((int(row.before_ms),int(row.after_ms)))
    by_sid={str(r.session_id):normalized_id(r.externalid) for r in load_sensor_index().itertuples()}
    result={}
    for kind in ('slide','micro15'):
        feats,old,refs=[],[],[]
        for fold in range(5):
            with np.load(root/'cache'/kind/f'fold{fold}_val.npz',allow_pickle=False) as arrays:
                all_refs=[tuple(json.loads(str(w))) for w in arrays['wid']]
                select=np.array([by_sid[r[0]] in allowed for r in all_refs])
                feats.append(arrays['feat'][select])
                old.append(arrays['label'][select])
                refs.extend(r for r,k in zip(all_refs,select) if k)
        if len(set(refs))!=len(refs):
            raise ValueError('duplicate master window identity')
        features=np.concatenate(feats)
        old_labels=np.concatenate(old)
        sessions=np.asarray([r[0] for r in refs])
        subjects=np.asarray([by_sid[r[0]] for r in refs])
        if set(subjects)!=allowed:
            raise ValueError('window population lacks a registered T24 subject')
        new_labels=np.empty(len(refs),dtype=np.int8)
        intervals=np.asarray([(r[1],r[2]) for r in refs],dtype=np.int64)
        for s in sorted(allowed):
            indices=np.flatnonzero(subjects==s)
            new_labels[indices]=corrected_labels(intervals[indices],meals[s],micro=kind=='micro15')
        if kind=='slide':
            features=add_time_prior(features,[EventRef(*r) for r in refs])
        result[kind]=dict(features=features,old=old_labels,new=new_labels,
                          sessions=sessions,subjects=subjects,intervals=intervals)
    return result


def score_metrics(labels,probabilities):
    if len(set(labels))!=2:
        raise ValueError('health metrics require both classes')
    return dict(ap=float(average_precision_score(labels,probabilities)),
        bce=float(log_loss(labels,probabilities,labels=[0,1])),
        auc=float(roc_auc_score(labels,probabilities)),
        recall_at_fixed_half=float(np.mean(probabilities[labels==1]>=.5)))


def execute(root,output):
    from src.data.manifests import MEALS_CSV,INDEX_CSV
    from src.pipeline.runner import _window_estimator,_micro_window_estimator
    root,output=Path(root).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    def budget():
        if time.perf_counter()-start>1800 or shutil.disk_usage(root).free<10*1024**3:
            raise RuntimeError('label probe wall/disk budget exceeded')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>64*1024**2:
            raise RuntimeError('label probe evidence budget exceeded')
    protected=protection(root)
    manifest_path=root/'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json'
    spec=root/'docs/official_recovery_label_contract_v1.md'
    inputs=[manifest_path,spec,Path(__file__).resolve(),Path(MEALS_CSV),Path(INDEX_CSV)]
    inputs.extend(root/'cache'/kind/f'fold{fold}_val.npz' for kind in ('slide','micro15') for fold in range(5))
    input_sha={str(p.relative_to(root)):_sha(p) for p in inputs}
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_partitions(manifest)
    plan=dict(id='LABEL-COVERAGE-V1',source_sha256=input_sha,protected_sha256=protected,
              development_only=True,outer_calls=0,architecture_changes=False,wrappers=manifest['wrappers'])
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    try:
        budget()
        pools=load_pools(root,manifest)
        results=[]
        for wrapper in manifest['wrappers']:
            train={normalized_id(s) for s in wrapper['train']}
            health={normalized_id(s) for s in wrapper['health']}
            for kind,pool in pools.items():
                budget()
                training=np.flatnonzero(np.isin(pool['subjects'],list(train)))
                testing=np.flatnonzero(np.isin(pool['subjects'],list(health))&(pool['new']>=0))
                labels=pool['new'][testing]
                paired={}
                scores={}
                for arm in ('old','new'):
                    selected=training[sample_rows(pool[arm][training],pool['sessions'][training])]
                    if set(pool['subjects'][selected])&health:
                        raise ValueError('held-out subject reached model fitting')
                    estimator=(_window_estimator if kind=='slide' else _micro_window_estimator)(20260908)
                    before=time.perf_counter()
                    with threadpool_limits(limits=8):
                        estimator.fit(pool['features'][selected],pool[arm][selected])
                        probabilities=estimator.predict_proba(pool['features'][testing])[:,1]
                    if not np.isfinite(probabilities).all():
                        raise ValueError('nonfinite probe scores')
                    paired[arm]=dict(**score_metrics(labels,probabilities),
                        train_rows=len(selected),train_positive=int(sum(pool[arm][selected]==1)),
                        fit_subjects=sorted(set(pool['subjects'][selected])),
                        elapsed_seconds=time.perf_counter()-before)
                    scores[arm]=probabilities.astype(np.float32)
                    print(f"wrapper{wrapper['id']} {kind} {arm}: AP={paired[arm]['ap']:.6f}",flush=True)
                    del estimator
                    budget()
                score_path=output/f"wrapper{wrapper['id']}_{kind}_scores.npz"
                unique_sessions,inverse=np.unique(pool['sessions'][testing],return_inverse=True)
                np.savez_compressed(score_path,old=scores['old'],new=scores['new'],label=labels,
                    interval=pool['intervals'][testing],sessions=unique_sessions,session_code=inverse.astype(np.int32))
                result=dict(wrapper=wrapper['id'],kind=kind,health_subjects=sorted(health),
                    health_windows=len(testing),health_positive=int(sum(labels==1)),arms=paired,
                    ap_delta=paired['new']['ap']-paired['old']['ap'],score_sha256=_sha(score_path),
                    training_old0_to_new1=int(sum((pool['old'][training]==0)&(pool['new'][training]==1))))
                results.append(result)
                (output/f"wrapper{wrapper['id']}_{kind}.json").write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        aggregate={}
        for kind in pools:
            rows=[r for r in results if r['kind']==kind]
            old=float(np.mean([r['arms']['old']['ap'] for r in rows]))
            new=float(np.mean([r['arms']['new']['ap'] for r in rows]))
            positive=sum(r['ap_delta']>0 for r in rows)
            aggregate[kind]=dict(old_ap=old,new_ap=new,ap_delta=new-old,
                                 positive_folds=positive,supports_event_probe=new-old>=.02 and positive>=2)
        if protected!=protection(root) or input_sha!={str(p.relative_to(root)):_sha(p) for p in inputs}:
            raise ValueError('protected release or input changed')
        report=dict(state='COMPLETE',evidence_kind='T24_WINDOW_LABEL_PROBE_NOT_EVENT_F1',
            model_fits=12,outer_calls=0,aggregate=aggregate,folds=results,
            supports_nested_event_probe=any(r['supports_event_probe'] for r in aggregate.values()),
            protected_unchanged=True,elapsed_seconds=time.perf_counter()-start)
        (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        budget()
        return report
    except BaseException as exc:
        (output/'state.json').write_text(json.dumps(dict(state='INCOMPLETE',error=repr(exc)),indent=2)+'\n',encoding='utf-8')
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=execute(Path(__file__).resolve().parents[3],args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='folds'},ensure_ascii=False))


if __name__=='__main__':
    main()
