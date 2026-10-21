"""Fixed max fusion on frozen, subject-isolated verifier components."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .dense_mil_probe import _score_file
from .event_probe import complete_metrics,partition_truths,decode,check_graph
from .geometry_probe import load_context,inputs,T24_SOURCE,FIVE_SOURCE
from .consistent_validation import aggregate
from .negative_clean_probe import _read,_write,protected_sources
from .scope_audit import _sha
from src.pipeline.candidate_control import CandidateAdmissionConfig,admit_candidates

SPEC='docs/official_recovery_max_fusion_v1.md'
T24_OUTPUT='outputs/official_recovery/max_fusion_v1_t24'


def fusion_gate(records,cohort):
    if cohort not in ('t24','five'): raise ValueError('unknown fusion cohort')
    n,total,wins_required=(3,160,2) if cohort=='t24' else (5,254,3)
    if len(records)!=n or {r['fold'] for r in records}!=set(range(n)): raise ValueError('complete folds required')
    if any(r['events']['A']['truths']!=r['events']['G']['truths'] for r in records): raise ValueError('paired truths differ')
    totals={a:aggregate([r['events'][a] for r in records]) for a in ('A','G')}
    if totals['A']['truths']!=total: raise ValueError('complete truth denominator differs')
    delta=totals['G']['f1']-totals['A']['f1']
    wins=sum(aggregate([r['events']['G']])['f1']>aggregate([r['events']['A']])['f1'] for r in records)
    return dict(passed=bool(delta>=.02-1e-12 and wins>=wins_required and totals['G']['tp']>=totals['A']['tp']),
        aggregate=totals,delta_f1=delta,positive_folds=int(wins),delta_tp=totals['G']['tp']-totals['A']['tp'],
        delta_fp=totals['G']['fp']-totals['A']['fp'])


def fuse(lr,lgbm,baseline):
    lr,lgbm,baseline=(np.asarray(v,dtype=float) for v in (lr,lgbm,baseline))
    if lr.ndim!=1 or lr.shape!=lgbm.shape or lr.shape!=baseline.shape:
        raise ValueError('component shape differs')
    if any(not np.isfinite(v).all() or np.any((v<0)|(v>1)) for v in (lr,lgbm,baseline)):
        raise ValueError('invalid component probabilities')
    if not np.allclose(baseline,.25*lr+.75*lgbm,rtol=0,atol=1e-12):
        raise ValueError('original component identity differs')
    return np.maximum(lr,lgbm)


def load_scores(path,mapping,train,score,excluded):
    candidates,old,groups,graph=_score_file(path,mapping,train,score,excluded)
    rows=_read(path)['rows']
    return candidates,fuse([r[7] for r in rows],[r[8] for r in rows],old),groups,graph


def select_threshold(candidates,scores,groups,admission,mapping,truths):
    rows=np.asarray(admit_candidates(candidates,scores,groups,CandidateAdmissionConfig(**admission)),dtype=int)
    choices=[]
    for threshold in np.unique(np.r_[np.asarray(scores)[rows],0.,1.,np.nextafter(1.,2.)]):
        metrics=complete_metrics([candidates[i].event for i in rows if scores[i]>=threshold],mapping,truths)
        choices.append(((metrics['f1'],metrics['ppv'],float(threshold)),metrics))
    rank,metrics=max(choices,key=lambda r:r[0])
    return dict(admission=dict(admission),threshold=rank[2],selection_metrics=metrics)


def execute(root,output,phase='policy',cohort='t24'):
    root,output=Path(root).resolve(),Path(output).resolve();started=time.perf_counter();completed=0
    if phase=='policy': output.mkdir(parents=True,exist_ok=False)
    elif phase!='score': raise ValueError('unknown phase')
    def fingerprints():
        result=inputs(root,cohort,parent_directory=T24_OUTPUT);result[SPEC]=_sha(root/SPEC);return result
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
            raise RuntimeError('commit protocol and code before fusion selection')
        if phase=='policy':
            hashes=fingerprints();protected=protected_sources(root)
            _write(output/'plan.json',dict(id='MAX-FUSION-V1',cohort=cohort,source_sha256=hashes,protected=protected,
                code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),model_fits=0,
                formula='max(lr,lgbm)',component_score_recomputations=0))
        else:
            plan=_read(output/'plan.json');hashes=plan['source_sha256'];protected=plan['protected']
            if plan['cohort']!=cohort or _read(output/'state.json')['state']!='POLICY_FROZEN':
                raise ValueError('matching frozen policies required')
            if any(_sha(output/n)!=h for n,h in _read(output/'policy_seal.json')['sha256'].items()):
                raise ValueError('fusion policy seal differs')
        def guard():
            if hashes!=fingerprints() or protected!=protected_sources(root): raise ValueError('fusion inputs/protection changed')
            if time.perf_counter()-started>600 or shutil.disk_usage(root).free<10*1024**3: raise RuntimeError('wall/disk limit')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>16*1024**2: raise RuntimeError('evidence limit')
        guard();truths,mapping,_,wrappers=load_context(root,cohort,parent_directory=T24_OUTPUT,parent_gate=fusion_gate)
        source=root/(T24_SOURCE if cohort=='t24' else FIVE_SOURCE)
        if cohort=='t24':
            from .geometry_probe import MANIFEST
            manifest=_read(root/MANIFEST);parts={w['id']:[(s['train'],s['score']) for s in w['policy_splits']] for w in manifest['wrappers']}
            forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        else: parts={int(k):v for k,v in _read(source/'plan.json')['policy_partitions'].items()};forbidden=set()
        records=[]
        for w in wrappers:
            fold=w['id'];train=set(w['train']);score=set(w['score']);frozen=_read(source/f'fold{fold}_policies.json')
            if phase=='policy':
                candidates=[];scores=[];groups=[]
                for i,(fit,valid) in enumerate(parts[fold]):
                    cs,ps,gs,_=load_scores(source/f'fold{fold}_policy{i}_A_scores.json',mapping,set(fit),set(valid),score|forbidden)
                    candidates.extend(cs);scores.extend(ps);groups.extend(gs)
                policy=select_threshold(candidates,np.asarray(scores),groups,frozen['policies']['A']['admission'],mapping,partition_truths(truths,train))
                graph={k:dict(fit_subjects=v['fit_subjects'],parents=v['parents']) for k,v in w['policy_graph'].items()}
                graph['fusion_policy']=dict(fit_subjects=sorted(train),parents=['policy']);check_graph(graph,'fusion_policy',score|forbidden,train)
                _write(output/f'fold{fold}_policy.json',dict(policy=policy,graph=graph));completed+=1;continue
            selected=_read(output/f'fold{fold}_policy.json');check_graph(selected['graph'],'fusion_policy',score|forbidden,train)
            role='health' if cohort=='t24' else 'outer'
            cs,ps,gs,graph=load_scores(source/f'fold{fold}_{role}_A_scores.json',mapping,train,score,forbidden)
            events={'A':w['health_events'],'G':decode(cs,ps,gs,selected['policy'])};targets=partition_truths(truths,score)
            row=dict(fold=fold,formula='max(lr,lgbm)',events={a:complete_metrics(es,mapping,targets) for a,es in events.items()},
                policy=selected['policy'],truths=targets,source_score_graph=graph,fusion_graph=selected['graph'],
                predictions={a:[asdict(e) for e in es] for a,es in events.items()},
                subject_events=[dict(subject=s,events={a:complete_metrics([e for e in es if mapping[e.sid]==s],mapping,partition_truths(targets,{s}))
                    for a,es in events.items()}) for s in sorted(score)])
            records.append(row);completed+=1;_write(output/f'fold{fold}_report.json',row)
        guard()
        if phase=='policy':
            _write(output/'policy_seal.json',dict(sha256={p.name:_sha(p) for p in output.glob('*.json') if p.name not in ('state.json','policy_seal.json')}))
            result=dict(state='POLICY_FROZEN',completed_folds=completed)
        else:
            result=dict(state='COMPLETE',evidence_kind='FROZEN_COMPONENT_MAX_FUSION_DEVELOPMENT_ONLY',cohort=cohort,
                **fusion_gate(records,cohort),model_fits=0,component_score_recomputations=0,formula='max(lr,lgbm)',
                protected_unchanged=True,elapsed_seconds=time.perf_counter()-started)
            _write(output/'summary.json',result)
        _write(output/'state.json',{'state':result['state']});return result
    except BaseException as exc:
        if output.is_dir(): _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_folds=completed))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('policy','score'),default='policy');parser.add_argument('--cohort',choices=('t24','five'),default='t24')
    args=parser.parse_args();print(execute(Path(__file__).resolve().parents[3],args.output,args.phase,args.cohort),flush=True)
