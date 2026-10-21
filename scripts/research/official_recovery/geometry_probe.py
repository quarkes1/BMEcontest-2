"""Model-free, train-OOF-selected event geometry with frozen baseline scores."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .consistent_validation import aggregate
from .dense_mil_probe import frozen_baseline,_score_file
from .event_probe import complete_metrics,partition_truths,check_graph,decode
from .label_probe import validate_partitions
from .negative_clean_probe import _read,_write,protected_sources
from .scope_audit import _sha,validated_session_subjects
from src.pipeline.event_geometry import calibrate_events,observed_session_bounds,GEOMETRY_SCALES,GEOMETRY_SHIFTS
from src.pipeline.event_stack import EventRef

SPEC='docs/official_recovery_geometry_v1.md'
T24_SOURCE='outputs/official_recovery/negative_clean_v1'
FIVE_SOURCE='outputs/official_recovery/negative_clean_validation_v1'
T24_OUTPUT='outputs/official_recovery/geometry_v1_t24'
MANIFEST='outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json'


def choose_geometry(events,bounds,mapping,truths):
    choices=[]
    for scale in GEOMETRY_SCALES:
        for shift in GEOMETRY_SHIFTS:
            transformed=calibrate_events(events,bounds,scale,shift)
            metrics=complete_metrics(transformed,mapping,truths)
            choices.append(((metrics['f1'],-abs(scale[0]/scale[1]-1),-abs(shift),-scale[0]/scale[1],-shift),
                dict(scale=list(scale),shift_ms=shift)))
    return max(choices,key=lambda row:row[0])[1]


def geometry_gate(records,cohort):
    if cohort not in ('t24','five'): raise ValueError('unknown geometry cohort')
    n,truth_count,minimum_wins=(3,160,2) if cohort=='t24' else (5,254,3)
    if len(records)!=n or {r['fold'] for r in records}!=set(range(n)): raise ValueError('complete unique folds required')
    if any(r['events']['A']['truths']!=r['events']['G']['truths'] for r in records): raise ValueError('paired denominator differs')
    totals={a:aggregate([r['events'][a] for r in records]) for a in ('A','G')}
    if totals['A']['truths']!=truth_count: raise ValueError('complete denominator differs')
    if totals['A']['predictions']!=totals['G']['predictions']: raise ValueError('geometry changed event count')
    delta=totals['G']['f1']-totals['A']['f1']
    wins=sum(aggregate([r['events']['G']])['f1']>aggregate([r['events']['A']])['f1'] for r in records)
    return dict(passed=bool(delta>=.02-1e-12 and wins>=minimum_wins and totals['G']['tp']>=totals['A']['tp']),
        aggregate=totals,delta_f1=delta,positive_folds=int(wins),delta_tp=totals['G']['tp']-totals['A']['tp'],
        delta_fp=totals['G']['fp']-totals['A']['fp'])


def verify_parent_identity(root,parent,scope,metadata):
    if protected_sources(root)!=parent['protected']: raise ValueError('parent protected artifacts changed')
    if any(scope['source_sha256'].get(str(p.relative_to(root)))!=_sha(p) for p in metadata):
        raise ValueError('scope metadata provenance differs')


def load_context(root,cohort):
    from src.data.manifests import load_sensor_index,load_meals,MEALS_CSV,INDEX_CSV
    rows=load_sensor_index().to_dict('records');mapping=validated_session_subjects(rows)
    bounds={str(r['session_id']):(int(r['timeStamp.startTime']),int(r['timeStamp.endTime'])) for r in rows}
    for kind in ('slide','micro15'):
        for k in range(5):
            with np.load(root/'cache'/kind/f'fold{k}_val.npz',allow_pickle=False) as arrays:
                windows=(EventRef(str(s),int(a),int(b)) for s,a,b in
                    (json.loads(str(w)) for w in arrays['wid']))
                bounds=observed_session_bounds(bounds,windows)
    if cohort=='t24':
        manifest=_read(root/MANIFEST);validate_partitions(manifest)
        truths=partition_truths([(r.externalid,int(r.before_ms),int(r.after_ms)) for r in load_meals().itertuples()],set(manifest['identities']['t24']))
        if len(truths)!=160: raise ValueError('T24 truth count differs')
        if _read(root/T24_SOURCE/'state.json')['state']!='COMPLETE': raise ValueError('completed baseline required')
        base,_=frozen_baseline(root,mapping,truths,manifest)
        wrappers=[dict(id=w['id'],train=w['train'],score=w['health'],**base[w['id']]) for w in manifest['wrappers']]
    elif cohort=='five':
        gate_source=root/T24_OUTPUT
        if _read(gate_source/'state.json')['state']!='COMPLETE': raise ValueError('completed T24 geometry gate required')
        gate=geometry_gate([_read(gate_source/f'fold{k}_report.json') for k in range(3)],'t24')
        summary=_read(gate_source/'summary.json')
        if not gate['passed'] or any(summary[k]!=v for k,v in gate.items()): raise ValueError('T24 geometry gate did not pass')
        if any(_sha(root/p)!=h for p,h in _read(gate_source/'plan.json')['source_sha256'].items()):
            raise ValueError('T24 gate inputs changed before fivefold')
        source=root/FIVE_SOURCE
        if _read(source/'state.json')['state']!='COMPLETE': raise ValueError('completed fivefold baseline required')
        plan=_read(source/'plan.json');subject_fold=plan['subject_fold']
        scope=_read(root/'outputs/official_recovery/scope_audit_confirmed/summary.json')
        verify_parent_identity(root,_read(gate_source/'plan.json'),scope,[Path(MEALS_CSV),Path(INDEX_CSV)])
        if len(subject_fold)!=39 or subject_fold!={r['subject']:r['fold'] for r in scope['per_subject']}:
            raise ValueError('fivefold population differs')
        truths=[(r['subject'],int(r['start_ms']),int(r['end_ms'])) for r in scope['truth_rows']]
        if len(truths)!=254: raise ValueError('fivefold complete truth count differs')
        wrappers=[];counts=[]
        for fold in range(5):
            train={s for s,k in subject_fold.items() if k!=fold};score=set(subject_fold)-train
            frozen=_read(source/f'fold{fold}_policies.json');policy=frozen['policies']['A'];graph=frozen['graphs']['A']
            check_graph(graph,'policy',score,train)
            candidates=[];probabilities=[];groups=[]
            for i,(fit,valid) in enumerate(plan['policy_partitions'][str(fold)]):
                cs,ps,gs,_=_score_file(source/f'fold{fold}_policy{i}_A_scores.json',mapping,set(fit),set(valid),score)
                candidates.extend(cs);probabilities.extend(ps);groups.extend(gs)
            policy_events=decode(candidates,np.asarray(probabilities),groups,policy)
            cs,ps,gs,score_graph=_score_file(source/f'fold{fold}_outer_A_scores.json',mapping,train,score,())
            events=decode(cs,ps,gs,policy);report=_read(source/f'fold{fold}_report.json')
            if [asdict(e) for e in events]!=report['predictions']['A']: raise ValueError('fivefold baseline event replay differs')
            m=complete_metrics(events,mapping,partition_truths(truths,score))
            if m!=report['events']['A']: raise ValueError('fivefold baseline metric replay differs')
            counts.append(m);wrappers.append(dict(id=fold,train=sorted(train),score=sorted(score),policy_events=policy_events,
                health_events=events,policy_graph=graph,health_graph=score_graph))
        total=aggregate(counts)
        if (total['tp'],total['fp'],total['fn'])!=(138,80,116): raise ValueError('fivefold baseline counts differ')
    else: raise ValueError('unknown geometry cohort')
    for wrapper in wrappers:
        for key in ('policy_events','health_events'):
            if calibrate_events(wrapper[key],bounds)!=wrapper[key]: raise ValueError('identity geometry did not replay')
    return truths,mapping,bounds,wrappers


def inputs(root,cohort):
    from src.data.manifests import MEALS_CSV,INDEX_CSV
    paths={Path(MEALS_CSV),Path(INDEX_CSV),root/SPEC,root/MANIFEST}
    paths.update(root/'cache'/kind/f'fold{k}_val.npz' for kind in ('slide','micro15') for k in range(5))
    paths.update(p for tree in ('src','scripts') for p in (root/tree).rglob('*.py'))
    paths.update((root/T24_SOURCE).glob('*.json'))
    if cohort=='five':
        paths.update((root/FIVE_SOURCE).glob('*.json'));paths.update((root/T24_OUTPUT).glob('*.json'))
        paths.add(root/'outputs/official_recovery/scope_audit_confirmed/summary.json')
    return {str(p.relative_to(root)):_sha(p) for p in paths}


def execute(root,output,phase='policy',cohort='t24'):
    root,output=Path(root).resolve(),Path(output).resolve();completed=0;started=time.perf_counter()
    if phase=='policy': output.mkdir(parents=True,exist_ok=False)
    elif phase!='score': raise ValueError('unknown geometry phase')
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
            raise RuntimeError('commit code and prior evidence before geometry selection')
        if phase=='policy':
            fingerprints=inputs(root,cohort);protected=protected_sources(root)
            _write(output/'plan.json',dict(id='EVENT-GEOMETRY-V1',cohort=cohort,source_sha256=fingerprints,
                protected=protected,model_fits=0,score_recomputations=0,
                code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        else:
            if _read(output/'state.json')['state']!='POLICY_FROZEN': raise ValueError('frozen geometry policies required')
            if any(_sha(output/n)!=h for n,h in _read(output/'policy_seal.json')['sha256'].items()): raise ValueError('geometry policy seal changed')
            plan=_read(output/'plan.json');fingerprints=plan['source_sha256'];protected=plan['protected']
            if plan['cohort']!=cohort: raise ValueError('geometry cohort changed')
        def unchanged():
            if fingerprints!=inputs(root,cohort) or protected!=protected_sources(root): raise ValueError('geometry sources/protection changed')
        def guard():
            if time.perf_counter()-started>600 or shutil.disk_usage(root).free<10*1024**3: raise RuntimeError('geometry wall/disk limit exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>16*1024**2: raise RuntimeError('geometry evidence limit exceeded')
        unchanged();guard();truths,mapping,bounds,wrappers=load_context(root,cohort);records=[]
        for wrapper in wrappers:
            fold=wrapper['id'];train=set(wrapper['train']);score=set(wrapper['score'])
            if phase=='policy':
                policy=choose_geometry(wrapper['policy_events'],bounds,mapping,partition_truths(truths,train))
                graph={k:dict(fit_subjects=v['fit_subjects'],parents=v['parents']) for k,v in wrapper['policy_graph'].items()}
                graph['geometry']=dict(fit_subjects=sorted(train),parents=['policy'])
                check_graph(graph,'geometry',score,train)
                _write(output/f'fold{fold}_policy.json',dict(policy=policy,graph=graph));completed+=1
                continue
            frozen=_read(output/f'fold{fold}_policy.json');check_graph(frozen['graph'],'geometry',score,train)
            events={'A':wrapper['health_events'],'G':calibrate_events(wrapper['health_events'],bounds,**frozen['policy'])}
            targets=partition_truths(truths,score)
            metrics={a:complete_metrics(v,mapping,targets) for a,v in events.items()}
            # Both original score lineage and calibration-policy lineage exclude these scoring subjects.
            check_graph(wrapper['health_graph'],'score',score,train)
            row=dict(fold=fold,events=metrics,policy=frozen['policy'],truths=targets,
                source_score_graph=wrapper['health_graph'],geometry_graph=frozen['graph'],
                subject_events=[dict(subject=s,events={a:complete_metrics([e for e in es if mapping[e.sid]==s],mapping,partition_truths(targets,{s}))
                    for a,es in events.items()}) for s in sorted(score)],
                predictions={a:[asdict(e) for e in es] for a,es in events.items()})
            records.append(row);completed+=1;_write(output/f'fold{fold}_report.json',row)
        unchanged();guard()
        if phase=='policy':
            _write(output/'policy_seal.json',dict(sha256={p.name:_sha(p) for p in output.glob('*.json')
                if p.name not in ('state.json','policy_seal.json')},completed_policy_folds=completed))
            result=dict(state='POLICY_FROZEN',completed_folds=completed)
        else:
            result=dict(state='COMPLETE',evidence_kind='SCORE_FROZEN_EVENT_GEOMETRY_DEVELOPMENT_ONLY',cohort=cohort,
                **geometry_gate(records,cohort),protected_unchanged=True,model_fits=0,score_recomputations=0,
                elapsed_seconds=time.perf_counter()-started)
            _write(output/'summary.json',result)
        _write(output/'state.json',{'state':result['state']});return result
    except BaseException as exc:
        if output.is_dir(): _write(output/'state.json',dict(state='INCOMPLETE',phase=phase,error=repr(exc),completed_folds=completed))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('policy','score'),default='policy');parser.add_argument('--cohort',choices=('t24','five'),default='t24')
    args=parser.parse_args();print(execute(Path(__file__).resolve().parents[3],args.output,args.phase,args.cohort),flush=True)
