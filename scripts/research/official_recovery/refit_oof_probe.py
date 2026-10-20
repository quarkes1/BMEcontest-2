"""Same sampled window models and fixed policy; change only meta-training to OOF."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import shutil
import subprocess
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .day_budget_probe import restore_candidates
from .eligibility_provenance import eligibility_inputs
from .event_probe import assess_gate,check_graph,complete_metrics,decode,partition_truths
from .full_pool_probe import expand_candidates,verify_partial_candidates
from .label_probe import load_pools,validate_partitions
from .refit_sampling_probe import validate_fixed_source
from .scope_audit import _sha,protection,validated_session_subjects
from scripts.research.representation_v2.crossfit import build_candidate_ledgers,slice_outer_train
from src.pipeline import runner
from src.pipeline.challenger.e03 import registered_e03_config


def fit_meta(train,score,config):
    if set(train.groups)&set(score.score_subjects): raise ValueError('heldout subject in meta fit')
    if train.features.shape[1]!=116 or score.features.shape[1]!=116 or set(train.labels.tolist())!={0,1}:
        raise ValueError('binary 116D OOF meta features required')
    probabilities=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for model in (runner._verifier_estimator(config.seed+3,.1),runner._verifier_lgbm_estimator(config.seed+23)):
            model.fit(train.features,train.labels)
            probabilities.append(runner._positive_probability(model,score.features) if len(score.features) else np.empty(0))
    return .5*probabilities[0]+.5*probabilities[1]


def _read(path): return json.loads(path.read_text(encoding='utf-8'))
def _write(path,payload): path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def execute(root,source,confirmed,pool_source,output):
    from src.data.manifests import load_sensor_index,INDEX_CSV,MEALS_CSV
    root,source,confirmed,pool_source,output=map(lambda p:Path(p).resolve(),(root,source,confirmed,pool_source,output))
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit experiment and source evidence before fitting')
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();protected=protection(root);records=[]
    try:
        summary=_read(source/'summary.json');source_plan=_read(source/'plan.json')
        if summary.get('state')!='COMPLETE' or summary.get('evidence_kind')!='T24_FINAL_REFIT_SAMPLING_DEVELOPMENT_ONLY':
            raise ValueError('completed sampling experiment required')
        manifest_path=root/'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json'
        manifest=_read(manifest_path);validate_partitions(manifest)
        allowed=set(manifest['identities']['t24']);forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        pool_plan=_read(pool_source/'plan.json')
        validate_fixed_source(root,confirmed,pool_plan['source_sha256'],manifest)
        reference=_read(confirmed/'eligibility_inputs.json')
        if eligibility_inputs(root,reference)!=reference: raise ValueError('session eligibility source changed')
        files=[manifest_path,Path(INDEX_CSV),Path(MEALS_CSV),Path(__file__).resolve(),root/'docs/official_recovery_refit_oof_v1.md',pool_source/'plan.json']
        files.extend(p for p in source.rglob('*') if p.is_file())
        files.extend(root/path for path in pool_plan['source_sha256'])
        files.extend(root/path for path in ('scripts/research/official_recovery/refit_sampling_probe.py',
            'scripts/research/official_recovery/full_pool_probe.py','scripts/research/representation_v2/crossfit.py',
            'src/pipeline/runner.py','src/pipeline/event_stack.py'))
        for path,digest in pool_plan['source_sha256'].items():
            if path.replace('\\','/').startswith('cache/') and _sha(root/path)!=digest:
                raise ValueError('window cache differs from fixed source')
        inputs={str(p.relative_to(root)):_sha(p) for p in set(files)}
        for w in manifest['wrappers']:
            selected=_read(confirmed/f"fold{w['id']}_policies.json")['policies']['old']
            if source_plan['policies'][str(w['id'])]!=selected: raise ValueError('fixed policy lineage differs')
        _write(output/'plan.json',dict(id='REFIT-OOF-V1',source_sha256=inputs,policies=source_plan['policies'],
            micro_thresholds=source_plan['micro_thresholds'],protected=protected,outer_calls=0,
            expected_window_fits=30,expected_verifier_fits=6,code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        config=registered_e03_config(0,workers=4)
        old=slice_outer_train(runner.FilesystemDataSource(root).load_outer_train_only(config),allowed)
        population=_read(confirmed/'population.json')
        if [asdict(t) for t in old.train_truths]!=population['legacy_eligible_truths']: raise ValueError('eligible truth snapshot changed')
        index=load_sensor_index().to_dict('records');mapping=validated_session_subjects(index)
        bounds={str(r['session_id']):(int(r['timeStamp.startTime']),int(r['timeStamp.endTime'])) for r in index}
        data=expand_candidates(old,load_pools(root,manifest),mapping,bounds)
        truths=[tuple(t) for t in population['complete_truths']]
        def budget():
            if time.perf_counter()-started>1800 or shutil.disk_usage(root).free<10*1024**3:
                raise RuntimeError('OOF refit wall/disk limit exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>64*1024**2:
                raise RuntimeError('OOF refit evidence limit exceeded')
        for wrapper in manifest['wrappers']:
            budget();fold=wrapper['id'];train=wrapper['train'];health=wrapper['health']
            baseline=_read(source/f'fold{fold}_sampled_scores.json')
            check_graph(baseline['graph'],'score',set(health)|forbidden,set(train))
            threshold=source_plan['micro_thresholds'][str(fold)];policy=source_plan['policies'][str(fold)]
            graph={};print(f'OOF-refit health{fold}: train{len(train)}/health{len(health)}',flush=True)
            t,v=build_candidate_ledgers(data,replace(config,seed=config.seed+2,micro_threshold_grid=(threshold,)),train,health,fit_audit=graph)
            baseline_candidates=restore_candidates(baseline['candidates'])
            verify_partial_candidates(v.candidates,baseline_candidates,{c.event.sid for c in v.candidates}|{c.event.sid for c in baseline_candidates})
            scores=fit_meta(t,v,config)
            graph['verifier']={'fit_subjects':list(train),'parents':['candidates']}
            graph['score']={'fit_subjects':[],'parents':['verifier']}
            check_graph(graph,'score',set(health)|forbidden,set(train))
            predictions={'old':decode(baseline_candidates,np.asarray(baseline['scores']),baseline['groups'],policy),
                         'full':decode(v.candidates,scores,v.groups,policy)}
            targets=partition_truths(truths,set(health))
            events={a:complete_metrics(pred,data.subject_by_session,targets) for a,pred in predictions.items()}
            subjects=[dict(subject=s,events={a:complete_metrics([p for p in pred if data.subject_by_session[p.sid]==s],
                data.subject_by_session,partition_truths(targets,{s})) for a,pred in predictions.items()}) for s in health]
            _write(output/f'fold{fold}_scores.json',dict(graph=graph,train=train,health=health,
                candidates=[asdict(c) for c in v.candidates],scores=scores.tolist(),groups=v.groups.tolist(),
                train_oof_candidates=len(t.candidates),train_oof_positive=int(sum(t.labels)),health_candidates_replayed=True))
            _write(output/f'fold{fold}_report.json',dict(events=events,subject_events=subjects,truths=targets,
                predictions={a:[asdict(p) for p in pred] for a,pred in predictions.items()}))
            records.append(events);print('OOF-refit paired complete-meal counts '+json.dumps(events),flush=True)
        budget();gate=assess_gate(records)
        if gate['aggregate']['old']!=summary['aggregate']['full']: raise ValueError('sampled in-sample baseline did not replay')
        if (protected!=protection(root) or inputs!={str(p.relative_to(root)):_sha(p) for p in set(files)}
                or eligibility_inputs(root,reference)!=reference): raise ValueError('source/protected artifacts changed')
        result=dict(state='COMPLETE',evidence_kind='T24_FINAL_REFIT_OOF_DEVELOPMENT_ONLY',**gate,
            window_fits=30,verifier_fits=6,outer_calls=0,protected_unchanged=True,elapsed_seconds=time.perf_counter()-started)
        _write(output/'summary.json',result);_write(output/'state.json',{'state':'COMPLETE'})
        return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_health_folds=len(records)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('source','confirmed','pool-source','output'): parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        print(json.dumps(execute(Path(__file__).resolve().parents[3],args.source,args.confirmed,args.pool_source,args.output),ensure_ascii=False),flush=True)
