"""Nested, frozen-baseline full-timeline MIL rescue development experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .dense_mil import build_spans,fit_model,predict_model,dense_proposals,rescue
from .consistent_validation import aggregate
from .event_probe import check_graph,complete_metrics,decode,partition_truths
from .label_probe import validate_partitions
from .negative_clean_probe import _read,_write,protected_sources
from .scope_audit import _sha,validated_session_subjects
from src.pipeline.event_stack import EventRef,MultiScaleCandidate
from src.pipeline.runner import peak_working_set_bytes

SPEC='docs/official_recovery_dense_mil_v1.md'
BASELINE='outputs/official_recovery/negative_clean_v1'
MANIFEST='outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json'


def rescue_gate(records):
    if len(records)!=3 or {r['fold'] for r in records}!=set(range(3)):
        raise ValueError('three unique complete folds required')
    if any(r['events']['A']['truths']!=r['events']['R']['truths'] for r in records):
        raise ValueError('paired denominator differs')
    totals={a:aggregate([r['events'][a] for r in records]) for a in ('A','R')}
    if totals['A']['truths']!=160: raise ValueError('complete denominator must be 160')
    delta=totals['R']['f1']-totals['A']['f1']
    wins=sum(aggregate([r['events']['R']])['f1']>aggregate([r['events']['A']])['f1'] for r in records)
    return dict(passed=bool(delta>=.02-1e-12 and wins>=2 and totals['R']['tp']>=totals['A']['tp']),
        aggregate=totals,delta_f1=delta,positive_folds=int(wins),delta_tp=totals['R']['tp']-totals['A']['tp'],
        delta_fp=totals['R']['fp']-totals['A']['fp'])


def choose_policy(base,candidates,mapping,truths):
    choices=[]
    for threshold in (.5,.65,.8,.9,.95,.99):
        for cap in (0,1,2,3):
            events=rescue(base,candidates,mapping,threshold,cap);m=complete_metrics(events,mapping,truths)
            choices.append(((m['f1'],m['ppv'],-(len(events)-len(base)),threshold,-cap),dict(threshold=threshold,cap=cap)))
    return max(choices,key=lambda row:row[0])[1]


def _score_file(path,mapping,train,score,excluded):
    evidence=_read(path)
    if set(evidence['train_subjects'])!=set(train) or set(evidence['score_subjects'])!=set(score):
        raise ValueError('frozen baseline score roles differ')
    check_graph(evidence['graph'],'score',set(score)|set(excluded),set(train))
    expected=['row','sid','start_ms','end_ms','macro','micro','subject','lr','lgbm','score']
    if evidence['columns']!=expected: raise ValueError('baseline score schema differs')
    rows=evidence['rows'];keys=[(r[1],r[2],r[3]) for r in rows]
    # Stream details are absent from compact scores, but cannot affect sorting when geometry is unique.
    if len(set(keys))!=len(keys) or [r[0] for r in rows]!=list(range(len(rows))):
        raise ValueError('ambiguous frozen candidate identity')
    if any(r[1] not in mapping or mapping[r[1]]!=r[6] or r[6] not in score for r in rows):
        raise ValueError('baseline candidate owner differs')
    candidates=[MultiScaleCandidate(EventRef(r[1],int(r[2]),int(r[3])),None,None) for r in rows]
    probabilities=np.asarray([r[9] for r in rows],float)
    if not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('invalid frozen scores')
    return candidates,probabilities,[r[6] for r in rows],evidence['graph']


def frozen_baseline(root,mapping,truths,manifest):
    source=root/BASELINE;forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
    old=_read(source/'plan.json')
    for p,digest in old['source_sha256'].items():
        if p.replace('\\','/').startswith(('Data/','cache/')) and _sha(root/p)!=digest:
            raise ValueError('baseline input identity changed: '+p)
    result={};metrics=[]
    for wrapper in manifest['wrappers']:
        fold=wrapper['id'];train=set(wrapper['train']);health=set(wrapper['health'])
        policy=_read(source/f'fold{fold}_policies.json')
        check_graph(policy['graphs']['A'],'policy',health|forbidden,train)
        candidates=[];probabilities=[];groups=[]
        for i,split in enumerate(wrapper['policy_splits']):
            cs,ps,gs,_=_score_file(source/f'fold{fold}_policy{i}_A_scores.json',mapping,set(split['train']),set(split['score']),health|forbidden)
            candidates.extend(cs);probabilities.extend(ps);groups.extend(gs)
        base_policy=decode(candidates,np.asarray(probabilities),groups,policy['policies']['A'])
        cs,ps,gs,graph=_score_file(source/f'fold{fold}_health_A_scores.json',mapping,train,health,forbidden)
        base_health=decode(cs,ps,gs,policy['policies']['A'])
        report=_read(source/f'fold{fold}_report.json');expected=report['predictions']['A']
        if [asdict(e) for e in base_health]!=expected: raise ValueError('frozen health baseline does not replay exactly')
        m=complete_metrics(base_health,mapping,partition_truths(truths,health))
        if m!=report['events']['A']: raise ValueError('frozen health baseline counts differ')
        metrics.append(m);result[fold]=dict(policy_events=base_policy,health_events=base_health,
            policy_graph=policy['graphs']['A'],health_graph=graph)
    totals=aggregate(metrics)
    if (totals['tp'],totals['fp'],totals['fn'])!=(89,62,71): raise ValueError('baseline aggregate differs')
    return result,totals


def load_context(root):
    from src.data.manifests import load_sensor_index,load_meals
    manifest=_read(root/MANIFEST);validate_partitions(manifest)
    allowed=set(manifest['identities']['t24']);mapping=validated_session_subjects(load_sensor_index().to_dict('records'))
    truths=partition_truths([(r.externalid,int(r.before_ms),int(r.after_ms)) for r in load_meals().itertuples()],allowed)
    if len(truths)!=160 or set(truths)!=set(map(tuple,_read(root/BASELINE/'population.json')['complete_truths'])):
        raise ValueError('complete target population differs')
    windows=[];features=[]
    for k in range(5):
        with np.load(root/'cache/micro15'/f'fold{k}_val.npz',allow_pickle=False) as z:
            refs=[tuple(json.loads(str(w))) for w in z['wid']]
            if any(r[0] not in mapping for r in refs): raise ValueError('unknown window session')
            keep=np.asarray([mapping[r[0]] in allowed for r in refs])
            windows.extend(EventRef(str(s),int(a),int(b)) for (s,a,b),use in zip(refs,keep) if use)
            features.append(z['feat'][keep])
    spans=build_spans(np.concatenate(features),windows,mapping)
    if {s.subject for s in spans}!=allowed: raise ValueError('full timeline subject coverage differs')
    baseline,metrics=frozen_baseline(root,mapping,truths,manifest)
    return spans,truths,mapping,manifest,baseline,metrics


def inputs(root):
    from src.data.manifests import MEALS_CSV,INDEX_CSV
    paths={Path(MEALS_CSV),Path(INDEX_CSV),root/SPEC,root/MANIFEST}
    paths.update((root/BASELINE).glob('*.json'))
    paths.update(root/'cache/micro15'/f'fold{k}_val.npz' for k in range(5))
    paths.update(p for tree in ('src','scripts') for p in (root/tree).rglob('*.py'))
    return {str(p.relative_to(root)):_sha(p) for p in paths}


def _merge_graphs(parts):
    result={}
    for prefix,graph in parts:
        for name,node in graph.items():
            result[prefix+name]=dict(fit_subjects=node['fit_subjects'],parents=[prefix+p for p in node['parents']])
    return result


def _save_predictions(output,prefix,scored,fit_subjects,score_subjects):
    offsets=[0];probs=[];starts=[];spans=[]
    for span,p in scored:
        probs.extend(p.tolist());starts.extend(w.start_ms for w in span.windows);offsets.append(len(probs))
        spans.append(dict(sid=span.sid,subject=span.subject,start_ms=span.windows[0].start_ms,end_ms=span.windows[-1].end_ms))
    path=output/(prefix+'.npz')
    if path.exists(): raise FileExistsError('dense scores already exist')
    np.savez_compressed(path,probability=np.asarray(probs,np.float32),start_ms=np.asarray(starts,np.int64),offsets=np.asarray(offsets,np.int64))
    _write(output/(prefix+'.json'),dict(spans=spans,sha256=_sha(path),fit_subjects=sorted(fit_subjects),score_subjects=sorted(score_subjects)))


def execute(root,output,phase='policy',device='cuda'):
    import torch
    root,output=Path(root).resolve(),Path(output).resolve();completed=0;started=time.perf_counter()
    if phase=='policy': output.mkdir(parents=True,exist_ok=False)
    elif phase!='health': raise ValueError('unknown dense phase')
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
            raise RuntimeError('commit code and evidence before training')
        if device not in ('cuda','cpu') or (device=='cuda' and not torch.cuda.is_available()):
            raise ValueError('registered device unavailable')
        if phase=='policy':
            fingerprints=inputs(root);protected=protected_sources(root)
            _write(output/'plan.json',dict(id='DENSE-MIL-V1',source_sha256=fingerprints,protected=protected,
                started_epoch=time.time(),device=device,steps_per_fit=1000,expected_fits=12,
                code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),outer_calls=0))
        else:
            if _read(output/'state.json')['state']!='POLICY_FROZEN': raise ValueError('frozen dense policies required')
            if any(_sha(output/p)!=h for p,h in _read(output/'policy_seal.json')['sha256'].items()): raise ValueError('policy seal changed')
            old=_read(output/'plan.json');fingerprints=old['source_sha256'];protected=old['protected']
            if old['device']!=device: raise ValueError('registered device changed')
        plan=_read(output/'plan.json');peak_rss=0;ticks=0
        def guard(force=False):
            nonlocal peak_rss,ticks
            ticks+=1
            if not force and ticks%25: return
            measured=peak_working_set_bytes()
            if measured is None: raise RuntimeError('process memory measurement unavailable')
            peak_rss=max(peak_rss,measured)
            if time.time()-plan['started_epoch']>7200 or peak_rss>8*1024**3 or shutil.disk_usage(root).free<10*1024**3:
                raise RuntimeError('dense wall/RSS/disk budget exceeded')
            if device=='cuda' and torch.cuda.max_memory_allocated()>6*1024**3: raise RuntimeError('dense GPU budget exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>128*1024**2: raise RuntimeError('dense evidence budget exceeded')
        def unchanged():
            if fingerprints!=inputs(root) or protected!=protected_sources(root): raise ValueError('inputs or protected files changed')
        guard(True);unchanged();spans,truths,mapping,manifest,baseline,base_metrics=load_context(root)
        _write(output/f'{phase}_population.json',dict(subjects=manifest['identities']['t24'],truths=truths,
            sessions=len({s.sid for s in spans}),spans=len(spans),windows=sum(len(s.windows) for s in spans),baseline=base_metrics))
        forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer']);records=[]
        def fit_score(fit_ids,score_ids,seed,prefix):
            guard(True)
            if set(fit_ids)&(set(score_ids)|forbidden): raise ValueError('dense fit/score identity overlap')
            fitted,stats=fit_model(spans,truths,set(fit_ids),seed=seed,device=device,callback=guard)
            scored=predict_model(fitted,spans,set(score_ids));guard(True)
            proposals=[p for span,prob in scored for p in dense_proposals(span,prob)]
            graph=dict(normalizer=dict(fit_subjects=sorted(fit_ids),parents=[]),
                encoder=dict(fit_subjects=sorted(fit_ids),parents=['normalizer']),score=dict(fit_subjects=[],parents=['encoder']))
            check_graph(graph,'score',set(score_ids)|forbidden,set(fit_ids))
            _save_predictions(output,prefix,scored,fit_ids,score_ids)
            _write(output/(prefix+'_fit.json'),dict(statistics=stats,graph=graph,proposal_count=len(proposals)))
            del fitted,scored;gc.collect()
            if device=='cuda': torch.cuda.empty_cache()
            return proposals,graph
        for wrapper in manifest['wrappers']:
            fold=wrapper['id'];train=set(wrapper['train']);health=set(wrapper['health']);base=baseline[fold]
            if phase=='policy':
                proposals=[];parts=[('base_',base['policy_graph'])];parents=['base_policy'];seen=[]
                for inner,split in enumerate(wrapper['policy_splits']):
                    fit_ids=set(split['train']);score_ids=set(split['score'])
                    if fit_ids|score_ids!=train or fit_ids&score_ids: raise ValueError('policy partition differs')
                    print(f'dense-MIL policy {fold}/{inner}: train{len(fit_ids)} score{len(score_ids)}',flush=True)
                    p,g=fit_score(fit_ids,score_ids,20261021+100*fold+inner,f'fold{fold}_policy{inner}')
                    proposals.extend(p);parts.append((f'p{inner}_',g));parents.append(f'p{inner}_score');seen.extend(score_ids);completed+=1
                if sorted(seen)!=sorted(train): raise ValueError('OOF policy subject coverage differs')
                graph=_merge_graphs(parts);graph['policy']=dict(fit_subjects=sorted(train),parents=parents)
                check_graph(graph,'policy',health|forbidden,train)
                selected=choose_policy(base['policy_events'],proposals,mapping,partition_truths(truths,train))
                _write(output/f'fold{fold}_policy.json',dict(policy=selected,graph=graph))
                print(f'dense-MIL policy {fold} frozen',flush=True);continue
            frozen=_read(output/f'fold{fold}_policy.json');check_graph(frozen['graph'],'policy',health|forbidden,train)
            print(f'dense-MIL health {fold}: train{len(train)} score{len(health)}',flush=True)
            proposals,graph=fit_score(train,health,20261021+100*fold+3,f'fold{fold}_health')
            predicted=rescue(base['health_events'],proposals,mapping,**frozen['policy'])
            targets=partition_truths(truths,health);predictions={'A':base['health_events'],'R':predicted}
            events={a:complete_metrics(p,mapping,targets) for a,p in predictions.items()}
            combined=_merge_graphs([('new_',graph),('base_',base['health_graph']),('policy_',frozen['graph'])])
            combined['decision']=dict(fit_subjects=[],parents=['new_score','base_score','policy_policy'])
            check_graph(combined,'decision',health|forbidden,train)
            row=dict(fold=fold,events=events,truths=targets,graph=combined,added=len(predicted)-len(base['health_events']),
                subject_events=[dict(subject=s,events={a:complete_metrics([e for e in p if mapping[e.sid]==s],mapping,partition_truths(targets,{s}))
                    for a,p in predictions.items()}) for s in sorted(health)],
                predictions={a:[asdict(e) for e in p] for a,p in predictions.items()})
            records.append(row);completed+=1;_write(output/f'fold{fold}_report.json',row)
            print('dense-MIL paired counts '+str(events),flush=True)
        guard(True);unchanged()
        if phase=='policy':
            _write(output/'policy_seal.json',dict(sha256={p.name:_sha(p) for p in output.iterdir()
                if p.is_file() and p.name not in ('state.json','policy_seal.json')},completed_policy_fits=completed,
                elapsed_seconds=time.perf_counter()-started))
            result=dict(state='POLICY_FROZEN',completed_fits=completed)
        else:
            differences=[s['events']['R']['f1']-s['events']['A']['f1'] for r in records for s in r['subject_events']]
            result=dict(state='COMPLETE',evidence_kind='T24_DENSE_MIL_RESCUE_DEVELOPMENT_ONLY',**rescue_gate(records),
                subject_diagnostics=dict(improved=sum(d>0 for d in differences),worsened=sum(d<0 for d in differences),tied=sum(d==0 for d in differences)),
                protected_unchanged=True,outer_calls=0,fits=12,total_wall_seconds=time.time()-plan['started_epoch'],peak_rss_bytes=peak_rss)
            _write(output/'summary.json',result)
        _write(output/'state.json',{'state':result['state']});return result
    except BaseException as exc:
        if output.is_dir(): _write(output/'state.json',dict(state='INCOMPLETE',phase=phase,error=repr(exc),completed_fits=completed))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('policy','health'),default='policy');parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    with threadpool_limits(limits=8): print(execute(Path(__file__).resolve().parents[3],args.output,args.phase,args.device),flush=True)
