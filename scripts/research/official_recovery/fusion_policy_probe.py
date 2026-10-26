"""Train-OOF convex verifier calibration using frozen component probabilities."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import time

import numpy as np

from .cap8_probe import controlled_revision, protected_baseline, cap_gate, REFERENCES
from .event_probe import check_graph, complete_metrics, partition_truths
from .geometry_probe import inputs, load_context, T24_SOURCE, FIVE_SOURCE, MANIFEST
from .negative_clean_probe import _read, _write, validate_policy_population
from .relative_score_probe import choose_threshold, load_scores, decode_relative, gate as r3_gate
from .scope_audit import _sha
from src.pipeline.runner import peak_working_set_bytes

WEIGHTS=(0.,.25,.5,.75,1.)
SPEC='docs/official_recovery_fusion_policy_v5.md'
OUTPUTS={kind:f'outputs/official_recovery/fusion_policy_v5_{kind}' for kind in ('t24','five')}
ADMISSION=dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=6)


def blend_scores(lr,lgbm,weight):
    a,b=np.asarray(lr,dtype=float),np.asarray(lgbm,dtype=float)
    if (weight not in WEIGHTS or a.ndim!=1 or a.shape!=b.shape
        or any(not np.isfinite(v).all() or np.any((v<0)|(v>1)) for v in (a,b))):
        raise ValueError('invalid frozen fusion components or registered weight')
    return weight*a+(1-weight)*b


def choose_fusion(cs,lr,lgbm,groups,mapping,truths):
    policies={}
    for weight in WEIGHTS:
        policy=choose_threshold(cs,blend_scores(lr,lgbm,weight),groups,ADMISSION,mapping,truths,use_relative=False)
        policy['weight']=weight;policies[str(weight)]=policy
    best=max(policies.values(),key=lambda p:(p['selection_metrics']['f1'],p['selection_metrics']['ppv'],
        -abs(p['weight']-.25),-p['weight']))
    return dict(policy=best,by_weight=policies)


def deployment_policy(rows):
    if len(rows)!=5: raise ValueError('five complete fold selections required')
    selected=[r['policy']['weight'] for r in rows]
    if any(w not in WEIGHTS for w in selected): raise ValueError('unregistered selected weight')
    weight=max(WEIGHTS,key=lambda w:(selected.count(w),-abs(w-.25),-w))
    try: thresholds=[float(r['by_weight'][str(weight)]['threshold']) for r in rows]
    except (KeyError,TypeError) as exc: raise ValueError('same-weight thresholds missing') from exc
    if not np.isfinite(thresholds).all() or any(not 0<=v<=1 for v in thresholds):
        raise ValueError('invalid same-weight thresholds')
    return dict(weight=weight,threshold=float(np.median(thresholds)),thresholds=thresholds,fold_selected_weights=selected)


def fusion_gate(records,cohort):
    result=cap_gate(records,cohort)
    result['passed']=bool(result['delta_f1']>=.005-1e-12 and result['delta_tp']>=-2
        and result['delta_fp']<=2*max(result['delta_tp'],0)
        and result['nonworse_folds']>=(2 if cohort=='t24' else 3))
    return result


def components(path,mapping,train,score,excluded):
    cs,p,groups,graph=load_scores(path,mapping,train,score,excluded)
    rows=_read(path)['rows']
    lr=np.asarray([r[7] for r in rows]);lgbm=np.asarray([r[8] for r in rows])
    blend_scores(lr,lgbm,.25)
    return cs,p,groups,graph,lr,lgbm


def verify_parent(root):
    folder=root/OUTPUTS['t24'];summary=_read(folder/'summary.json')
    result=fusion_gate([_read(folder/f'fold{k}_report.json') for k in range(3)],'t24')
    if summary['state']!='COMPLETE' or not result['passed'] or any(summary[k]!=v for k,v in result.items()):
        raise ValueError('passed complete fusion T24 required')
    for name,h in _read(folder/'plan.json')['source_sha256'].items():
        if _sha(root/name)!=h: raise ValueError('T24 source drift')
    for name,h in _read(folder/'policy_seal.json')['sha256'].items():
        if _sha(folder/name)!=h: raise ValueError('T24 policy seal changed')


def execute(root,cohort='t24',phase='policy'):
    if cohort not in OUTPUTS or phase not in ('policy','score'): raise ValueError('unknown fusion phase/cohort')
    root=Path(root).resolve();folder=root/OUTPUTS[cohort];started=time.perf_counter();completed=0
    revision,edits=controlled_revision(root)
    if cohort=='five': verify_parent(root)
    def fingerprints():
        hashes=inputs(root,cohort,parent_directory=REFERENCES['t24'])
        hashes[SPEC]=_sha(root/SPEC)
        for kind in ('t24',cohort):
            hashes.update({p.relative_to(root).as_posix():_sha(p) for p in (root/REFERENCES[kind]).glob('*.json')})
        return hashes
    if phase=='policy':
        from .policy_trial_export import ensure_new
        ensure_new(root,folder);folder.mkdir(parents=True)
    try:
        if phase=='policy':
            hashes=fingerprints();protected=protected_baseline(root)
            _write(folder/'plan.json',dict(id='FUSION-POLICY-V5',cohort=cohort,source_sha256=hashes,
                protected=protected,user_edits=edits,code_sha=revision,actual_exe_cv=False,model_fits=0,weights=list(WEIGHTS)))
        else:
            plan=_read(folder/'plan.json');hashes=plan['source_sha256'];protected=plan['protected']
            if (plan['id']!='FUSION-POLICY-V5' or plan['cohort']!=cohort or plan['user_edits']!=edits
                or _read(folder/'state.json')['state']!='POLICY_FROZEN'):
                raise ValueError('matching frozen fusion policy required')
            if any(_sha(folder/n)!=h for n,h in _read(folder/'policy_seal.json')['sha256'].items()):
                raise ValueError('fusion policy seal drift')
        if hashes!=fingerprints() or protected!=protected_baseline(root): raise ValueError('fusion source/protected drift')
        truths,mapping,_,wrappers=load_context(root,cohort,parent_directory=REFERENCES['t24'],
            parent_gate=lambda rows,kind:r3_gate(rows,kind,'deploy_f1'))
        source=root/(T24_SOURCE if cohort=='t24' else FIVE_SOURCE)
        if cohort=='t24':
            manifest=_read(root/MANIFEST)
            parts={w['id']:[(s['train'],s['score']) for s in w['policy_splits']] for w in manifest['wrappers']}
            forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        else:
            parts={int(k):v for k,v in _read(source/'plan.json')['policy_partitions'].items()};forbidden=set()
        records=[]
        for w in wrappers:
            k=w['id'];train=set(w['train']);score=set(w['score'])
            if phase=='policy':
                cs=[];lr=[];lgbm=[];groups=[];seen=[];graph={}
                for i,(fit,valid) in enumerate(parts[k]):
                    fit,valid=set(fit),set(valid)
                    if fit&valid or fit|valid!=train: raise ValueError('fusion producer split differs')
                    sub,p,gs,parent,a,b=components(source/f'fold{k}_policy{i}_A_scores.json',mapping,fit,valid,score|forbidden)
                    cs.extend(sub);lr.extend(a);lgbm.extend(b);groups.extend(gs);seen.extend(valid)
                    for name,node in parent.items():
                        graph[f'p{i}_{name}']=dict(fit_subjects=node['fit_subjects'],parents=[f'p{i}_{p}' for p in node['parents']])
                validate_policy_population(seen,groups,train)
                graph['policy']=dict(fit_subjects=sorted(train),parents=[f'p{i}_decision_score' for i in range(3)])
                check_graph(graph,'policy',score|forbidden,train)
                selected=choose_fusion(cs,lr,lgbm,groups,mapping,partition_truths(truths,train))
                selected['graph']=graph;_write(folder/f'fold{k}_policy.json',selected)
                completed+=1;print(f'fusion {cohort} policy{k} frozen w='+str(selected['policy']['weight']),flush=True)
                continue
            selected=_read(folder/f'fold{k}_policy.json');check_graph(selected['graph'],'policy',score|forbidden,train)
            role='health' if cohort=='t24' else 'outer'
            cs,p,groups,graph,lr,lgbm=components(source/f'fold{k}_{role}_A_scores.json',mapping,train,score,forbidden)
            reference=_read(root/REFERENCES[cohort]/f'fold{k}_policy.json')['policy']
            if reference['admission']!=ADMISSION or selected['policy']['admission']!=ADMISSION:
                raise ValueError('fusion admission differs')
            es=dict(A=decode_relative(cs,p,groups,reference,use_relative=False),
                G=decode_relative(cs,blend_scores(lr,lgbm,selected['policy']['weight']),groups,selected['policy'],use_relative=False))
            target=partition_truths(truths,score);old=_read(root/REFERENCES[cohort]/f'fold{k}_report.json')
            if [asdict(e) for e in es['A']]!=old['predictions']['G'] or complete_metrics(es['A'],mapping,target)!=old['events']['G']:
                raise ValueError('official-best event replay differs')
            row=dict(fold=k,events={a:complete_metrics(e,mapping,target) for a,e in es.items()},
                predictions={a:[asdict(event) for event in ev] for a,ev in es.items()},policy=selected,score_graph=graph,
                subject_events=[dict(subject=s,events={a:complete_metrics([e for e in ev if mapping[e.sid]==s],mapping,
                    partition_truths(target,{s})) for a,ev in es.items()}) for s in sorted(score)])
            records.append(row);completed+=1;_write(folder/f'fold{k}_report.json',row)
            print(f'fusion {cohort} score{k}: '+str(row['events']),flush=True)
        if hashes!=fingerprints() or protected!=protected_baseline(root): raise ValueError('fusion source/protected drift')
        if peak_working_set_bytes() is None or peak_working_set_bytes()>12*1024**3 or time.perf_counter()-started>600 or shutil.disk_usage(root).free<10*1024**3:
            raise RuntimeError('fusion budget exceeded')
        if phase=='policy':
            _write(folder/'policy_seal.json',dict(sha256={p.name:_sha(p) for p in folder.glob('*.json')}))
            result=dict(state='POLICY_FROZEN',completed_folds=completed)
        else:
            result=dict(state='COMPLETE',cohort=cohort,**fusion_gate(records,cohort),
                evidence_kind='REPEATED_FROZEN_MODEL_FUSION_POLICY_DEVELOPMENT_ONLY',actual_exe_cv=False,
                model_fits=0,protected_unchanged=True,user_edits_unchanged=True,elapsed_seconds=time.perf_counter()-started)
            if cohort=='five': result['deployment_policy']=deployment_policy([_read(folder/f'fold{k}_policy.json') for k in range(5)])
            _write(folder/'summary.json',result)
        _write(folder/'state.json',dict(state=result['state']));return result
    except BaseException as exc:
        _write(folder/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_folds=completed));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',choices=('t24','five'),default='t24');p.add_argument('--phase',choices=('policy','score'),default='policy')
    a=p.parse_args();print(execute(Path(__file__).resolve().parents[3],a.cohort,a.phase))
