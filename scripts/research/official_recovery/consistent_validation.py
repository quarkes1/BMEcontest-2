"""Sealed-policy five-fold comparison of release-style and CV-consistent refits."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .eligibility_provenance import eligibility_inputs
from .event_probe import check_graph,complete_metrics,decode,partition_truths
from .full_pool_probe import expand_candidates,reachable_truths
from .label_probe import load_pools
from .refit_sampling_probe import fit_windows,candidate_view
from .scope_audit import RUN_KEY,_sha,fold_subjects,protection,read_fold_subjects,validated_session_subjects
from scripts.research.representation_v2.crossfit import build_candidate_ledgers,slice_outer_train
from src.pipeline import runner
from src.pipeline.artifacts import verify_promotion_attestation
from src.pipeline.challenger.e03 import registered_e03_config


def _read(path): return json.loads(path.read_text(encoding='utf-8'))
def _write(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def read_fold_policy(directory,fold,micro_threshold):
    directory=Path(directory);manifest=_read(directory/'manifest.json')
    if manifest.get('role')!='outer-fold-evidence' or manifest.get('run_key')!=f'outer-fold-{fold}':
        raise ValueError('sealed fold identity differs')
    for name in ('policy.json','run_config.json'):
        if _sha(directory/name)!=manifest['files'][name]: raise ValueError('sealed policy/config SHA mismatch')
    config=_read(directory/'run_config.json');raw=_read(directory/'policy.json')
    if config['outer_fold']!=fold: raise ValueError('sealed config fold differs')
    if raw['max_events_per_group'] is not None: raise ValueError('unsupported sealed final event cap')
    if len(config['verifier_c_grid'])!=1: raise ValueError('ambiguous sealed verifier C')
    if not 0<=raw['blend_weight']<=1 or not 0<=micro_threshold<=1: raise ValueError('invalid sealed score settings')
    return dict(policy=dict(admission=dict(nms_iou=raw['nms_iou'],threshold=raw['admission_threshold'],
        max_candidates_per_subject=raw['max_candidates_per_subject']),threshold=raw['threshold']),
        blend=raw['blend_weight'],micro_threshold=float(micro_threshold),verifier_c=config['verifier_c_grid'][0],config=config)


def assemble_full_data(old,pools,mapping,bounds):
    # This is a scoring universe, not a fit role. Every fitting call below receives its train-only slice.
    subjects=set(pools['slide']['subjects'])
    sids={str(s) for pool in pools.values() for s in pool['sessions']}|set(old.subject_by_session)
    base=replace(old,subject_by_session={s:mapping[s] for s in sids},session_bounds_by_sid={s:bounds[s] for s in sids})
    if set(base.subject_by_session.values())!=subjects: raise ValueError('full scoring universe differs from cohort')
    return expand_candidates(base,pools,mapping,bounds)


def fit_verifiers(x,y,z,config,regularization,blend):
    if not 0<=blend<=1: raise ValueError('invalid verifier blend')
    if set(np.asarray(y).tolist())!={0,1}: raise ValueError('binary verifier targets required')
    components={}
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for name,model in [('lr',runner._verifier_estimator(config.seed+3,regularization)),
                           ('lgbm',runner._verifier_lgbm_estimator(config.seed+23))]:
            model.fit(x,y)
            components[name]=runner._positive_probability(model,z) if len(z) else np.empty(0)
    probabilities=blend*components['lr']+(1-blend)*components['lgbm']
    if not np.isfinite(probabilities).all(): raise ValueError('nonfinite verifier probability')
    return probabilities,components


def aggregate(rows):
    tp=sum(r['tp'] for r in rows);pred=sum(r['predictions'] for r in rows);true=sum(r['truths'] for r in rows)
    return dict(tp=tp,fp=pred-tp,fn=true-tp,predictions=pred,truths=true,
        f1=2*tp/(pred+true) if pred+true else 0.,ppv=tp/pred if pred else 0.,recall=tp/true if true else 0.)


def validation_gate(records):
    if len(records)!=5 or {r['fold'] for r in records}!=set(range(5)):
        raise ValueError('five unique complete folds required')
    if any(r['release_style']['truths']!=r['consistent_style']['truths'] for r in records):
        raise ValueError('paired denominator differs')
    metrics={a:aggregate([r[a] for r in records]) for a in ('release_style','consistent_style')}
    differences=[aggregate([r['consistent_style']])['f1']-aggregate([r['release_style']])['f1'] for r in records]
    delta=metrics['consistent_style']['f1']-metrics['release_style']['f1']
    return dict(passed=bool(delta>=.03-1e-12 and sum(d>=0 for d in differences)>=3 and sum(d>0 for d in differences)>=3),
        aggregate=metrics,delta_f1=delta,positive_folds=sum(d>0 for d in differences),nonworse_folds=sum(d>=0 for d in differences))


def _save_scores(path,candidates,scores,components,groups,graph,train,score):
    # Compact evidence; no signal copies, feature cache, weights or duplicated probability histories.
    rows=[[i,c.event.sid,int(c.event.start_ms),int(c.event.end_ms),bool(c.macro),bool(c.micro),str(groups[i]),
           float(components['lr'][i]),float(components['lgbm'][i]),float(scores[i])] for i,c in enumerate(candidates)]
    path.write_text(json.dumps(dict(columns=['row','sid','start_ms','end_ms','macro','micro','subject','lr','lgbm','score'],
        rows=rows,graph=graph,train_subjects=sorted(train),score_subjects=sorted(score)),separators=(',',':'))+'\n',encoding='utf-8')


def execute(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();records=[]
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True,stderr=subprocess.STDOUT).strip():
            raise RuntimeError('commit code and protocol before validation')
        from src.data.manifests import MEALS_CSV,INDEX_CSV,load_sensor_index
        protected=protection(root);run_root=root/'models/event_stack'/RUN_KEY
        problems=verify_promotion_attestation(run_root,expected_run_key=RUN_KEY)
        if problems: raise ValueError('invalid release attestation: '+str(problems))
        summary_path=run_root/'promotion_summary.json';summary=_read(summary_path)
        fold_records,fold_paths=read_fold_subjects(run_root,summary);subject_fold=fold_subjects(fold_records)
        if len(subject_fold)!=39: raise ValueError('fixed cohort is not 39 evaluated subjects')
        scope_path=root/'outputs/official_recovery/scope_audit_confirmed/summary.json';scope=_read(scope_path)
        for path in (Path(MEALS_CSV),Path(INDEX_CSV)):
            if scope['source_sha256'].get(str(path.relative_to(root)))!=_sha(path):
                raise ValueError('metadata differs from the frozen scope audit')
        if {r['subject']:r['fold'] for r in scope['per_subject']}!=subject_fold:
            raise ValueError('scope population differs from attested folds')
        complete=[(r['subject'],int(r['start_ms']),int(r['end_ms'])) for r in scope['truth_rows']]
        eligible=[(r['subject'],int(r['start_ms']),int(r['end_ms'])) for r in scope['truth_rows'] if r['legacy_eligible']]
        if len(complete)!=254 or len(eligible)!=153: raise ValueError('fixed truth denominators differ')
        settings={r['outer_fold']:read_fold_policy(run_root/f"outer-fold-{r['outer_fold']}",r['outer_fold'],r['micro_threshold']) for r in summary['folds']}
        files=[summary_path,run_root/'promotion_attestation.json',scope_path,Path(MEALS_CSV),Path(INDEX_CSV),Path(__file__).resolve(),
               root/'docs/official_recovery_consistent_refit_validation_v1.md']+list(fold_paths)
        for fold in range(5):
            files.extend(run_root/f'outer-fold-{fold}'/name for name in ('policy.json','run_config.json'))
            files.append(root/'cache/splits'/f'fold{fold}.json')
            files.extend(root/'cache'/kind/f'fold{fold}_{split}.npz' for kind in ('slide','micro15')
                         for split in ('train','meal_train','no_meal_train','val'))
        files.extend(root/path for path in ('scripts/research/official_recovery/refit_sampling_probe.py',
            'scripts/research/official_recovery/full_pool_probe.py','scripts/research/official_recovery/label_probe.py',
            'scripts/research/official_recovery/event_probe.py','scripts/research/official_recovery/scope_audit.py',
            'scripts/research/official_recovery/eligibility_provenance.py','scripts/research/representation_v2/crossfit.py',
            'src/pipeline/runner.py','src/pipeline/event_stack.py','src/pipeline/candidate_control.py',
            'src/pipeline/context_features.py','src/pipeline/features/macro.py','src/data/manifests.py'))
        hashes={str(p.relative_to(root)):_sha(p) for p in set(files)}
        index=load_sensor_index().to_dict('records');mapping=validated_session_subjects(index)
        bounds={str(r['session_id']):(int(r['timeStamp.startTime']),int(r['timeStamp.endTime'])) for r in index}
        cohort=set(subject_fold)
        pools=load_pools(root,{'identities':{'t24':sorted(cohort)}})
        # Loader reads candidate-role session arrays when resolving old eligible labels.
        role_sids=set()
        for fold in range(5):
            for split in ('meal_train','no_meal_train'):
                with np.load(root/'cache/slide'/f'fold{fold}_{split}.npz',allow_pickle=False) as cache:
                    role_sids.update(json.loads(str(w))[0] for w in cache['wid'])
        eligibility_before=eligibility_inputs(root,role_sids)
        _write(output/'eligibility_inputs.json',eligibility_before)
        _write(output/'plan.json',dict(id='CONSISTENT-REFIT-FIVEFOLD-V1',source_sha256=hashes,settings=settings,
            subject_fold=subject_fold,protected=protected,complete_truth_count=254,eligible_truth_count=153,
            code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            expected_window_fits=60,expected_verifier_fits=20,policy_selections=0,development_only=True))
        def budget():
            if time.perf_counter()-started>7200 or shutil.disk_usage(root).free<10*1024**3:
                raise RuntimeError('validation wall/disk budget exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>128*1024**2:
                raise RuntimeError('validation evidence budget exceeded')
        for fold in range(5):
            budget();frozen=settings[fold];config=registered_e03_config(fold,workers=8)
            sealed=frozen['config'].copy();registered=json.loads(json.dumps(asdict(config)));sealed.pop('workers');registered.pop('workers')
            if sealed!=registered: raise ValueError('registered factory config differs from sealed fold')
            train={s for s,k in subject_fold.items() if k!=fold};score=cohort-train
            old=slice_outer_train(runner.FilesystemDataSource(root).load_outer_train_only(config),train)
            universe=assemble_full_data(old,pools,mapping,bounds)
            t=slice_outer_train(universe,train);v=slice_outer_train(universe,score)
            if set(t.subject_by_session.values())&score: raise ValueError('heldout identity in fit slice')
            targets=partition_truths(complete,score);eligible_targets=partition_truths(eligible,score)
            events={};eligible_events={};predictions={};diagnostics={}
            print(f'fivefold {fold} release-style: train{len(train)}/score{len(score)}',flush=True)
            macro,micro,stats=fit_windows(t,config,'full')
            tc,x,_=candidate_view(t,config,macro,micro,frozen['micro_threshold'])
            vc,z,quantiles=candidate_view(v,config,macro,micro,frozen['micro_threshold'])
            labels=runner._candidate_labels(tc,t.train_truths)
            p,components=fit_verifiers(x,labels,z,config,frozen['verifier_c'],frozen['blend'])
            groups=runner._groups_for([c.event for c in vc],v.subject_by_session)
            graph={'macro':{'fit_subjects':sorted(train),'parents':[]},'micro':{'fit_subjects':sorted(train),'parents':[]},
                   'verifier':{'fit_subjects':sorted(train),'parents':['macro','micro']},'score':{'fit_subjects':[],'parents':['verifier']}}
            check_graph(graph,'score',score,train)
            predictions['release_style']=decode(vc,p,groups,frozen['policy'])
            diagnostics['release_style']=dict(window_training=stats,train_candidates=len(tc),
                train_candidate_positive=int(sum(labels)),raw_reachable_truths=reachable_truths(vc,mapping,targets),health_window_quantiles=quantiles)
            _save_scores(output/f'fold{fold}_release_scores.json',vc,p,components,groups,graph,train,score)
            del macro,micro,tc,vc,x,z,p,components;gc.collect();budget()
            print(f'fivefold {fold} consistent-style: four-fold window OOF',flush=True)
            graph={}
            train_ledger,score_ledger=build_candidate_ledgers(universe,replace(config,seed=config.seed+2,
                micro_threshold_grid=(frozen['micro_threshold'],)),train,score,fit_audit=graph)
            p,components=fit_verifiers(train_ledger.features,train_ledger.labels,score_ledger.features,config,
                                       frozen['verifier_c'],frozen['blend'])
            graph['verifier']={'fit_subjects':sorted(train),'parents':['candidates']}
            graph['score']={'fit_subjects':[],'parents':['verifier']};check_graph(graph,'score',score,train)
            predictions['consistent_style']=decode(score_ledger.candidates,p,score_ledger.groups,frozen['policy'])
            diagnostics['consistent_style']=dict(window_training={name:dict(rows=int(sum(batch.labels>=0)),
                positive=int(sum(batch.labels==1)),positive_ratio=float(np.mean(batch.labels[batch.labels>=0]==1)))
                for name,batch in [('macro',t.window_train),('micro',t.micro_window_train)]},
                train_candidates=len(train_ledger.candidates),train_candidate_positive=int(sum(train_ledger.labels)),
                raw_reachable_truths=reachable_truths(score_ledger.candidates,mapping,targets))
            _save_scores(output/f'fold{fold}_consistent_scores.json',score_ledger.candidates,p,components,score_ledger.groups,graph,train,score)
            for arm,pred in predictions.items():
                events[arm]=complete_metrics(pred,mapping,targets);eligible_events[arm]=complete_metrics(pred,mapping,eligible_targets)
            subjects=[dict(subject=s,fold=fold,events={a:complete_metrics([p for p in pred if mapping[p.sid]==s],mapping,
                partition_truths(targets,{s})) for a,pred in predictions.items()}) for s in sorted(score)]
            _write(output/f'fold{fold}_report.json',dict(fold=fold,train=sorted(train),score=sorted(score),events=events,
                eligible_events=eligible_events,subject_events=subjects,diagnostics=diagnostics,truths=targets,
                predictions={a:[asdict(p) for p in pred] for a,pred in predictions.items()}))
            records.append(dict(fold=fold,**events))
            _write(output/'state.json',dict(state='IN_PROGRESS',completed_folds=len(records)))
            print('fivefold complete-meal counts '+json.dumps(events),flush=True)
            del universe,t,v,train_ledger,score_ledger,p,components,old;gc.collect()
        budget();gate=validation_gate(records)
        if any(m['truths']!=254 for m in gate['aggregate'].values()): raise ValueError('final truth coverage differs')
        if (protected!=protection(root) or hashes!={str(p.relative_to(root)):_sha(p) for p in set(files)}
                or eligibility_inputs(root,role_sids)!=eligibility_before): raise ValueError('input/protected artifacts changed')
        reports=[_read(output/f'fold{k}_report.json') for k in range(5)]
        subject_rows=[r for report in reports for r in report['subject_events']]
        improvements=[r['events']['consistent_style']['f1']-r['events']['release_style']['f1'] for r in subject_rows]
        result=dict(state='COMPLETE',evidence_kind='FIVEFOLD_DEVELOPMENT_REFIT_CONSISTENCY_NOT_OFFICIAL_SCORE',**gate,
            eligible_diagnostics={a:aggregate([r['eligible_events'][a] for r in reports]) for a in ('release_style','consistent_style')},
            subject_diagnostics=dict(improved=sum(d>0 for d in improvements),worsened=sum(d<0 for d in improvements),
                tied=sum(d==0 for d in improvements)),protected_unchanged=True,policy_selections=0,
            window_fits=60,verifier_fits=20,elapsed_seconds=time.perf_counter()-started)
        _write(output/'summary.json',result);_write(output/'state.json',{'state':'COMPLETE'})
        return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_folds=len(records)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=8):
        print(json.dumps(execute(Path(__file__).resolve().parents[3],args.output),ensure_ascii=False),flush=True)
