"""Train-only simulation of final-refit negative sampling, with fixed policies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from .event_probe import assess_gate,check_graph,complete_metrics,decode,partition_truths
from .full_pool_probe import expand_candidates
from .label_probe import load_pools,validate_partitions
from .scope_audit import _sha,protection,validated_session_subjects
from scripts.research.representation_v2.crossfit import slice_outer_train
from src.pipeline import runner
from src.pipeline.challenger.e03 import registered_e03_config


def fit_windows(data,config,role):
    if role not in ('full','sampled'): raise ValueError('unknown window training role')
    macro=data.candidate_train if role=='full' else data.window_train
    micro=data.micro_candidate_train if role=='full' else data.micro_window_train
    keep=np.asarray(macro.labels)>=0
    micro_keep=runner._micro_training_keep(micro,data.train_truths,config.micro_positive_middle_fraction)
    macro_model=runner._window_estimator(config.seed+2).fit(
        runner._with_time_prior(macro.features,macro.windows)[keep],macro.labels[keep])
    micro_model=runner._micro_window_estimator(config.seed+12).fit(micro.features[micro_keep],micro.labels[micro_keep])
    stats={kind:dict(rows=int(sum(mask)),positive=int(sum(batch.labels[mask]==1)),
        positive_ratio=float(np.mean(batch.labels[mask]==1))) for kind,batch,mask in
        [('macro',macro,keep),('micro',micro,micro_keep)]}
    return macro_model,micro_model,stats


def candidate_view(data,config,macro_model,micro_model,micro_threshold):
    macro_scores=runner._positive_probability(macro_model,runner._with_time_prior(data.candidate_train.features,data.candidate_train.windows))
    micro_scores=runner._positive_probability(micro_model,data.micro_candidate_train.features)
    macro=runner._windows_by_session(data.candidate_train.windows,macro_scores)
    micro=runner._windows_by_session(data.micro_candidate_train.windows,micro_scores)
    candidates=tuple(runner.union_candidates(runner.density_candidates(macro,config.density),
        runner.micro_candidates(micro,micro_threshold,config.micro_candidate)))
    features=runner.multiscale_verifier_features(candidates,macro,micro,context_features_version='v1',
        session_bounds_by_sid=runner._context_session_bounds(data,candidates,data.candidate_train.windows,data.micro_candidate_train.windows))
    quantiles={kind:np.quantile(values,[0.,.25,.5,.75,.9,.99,1.]).tolist()
               for kind,values in [('macro',macro_scores),('micro',micro_scores)]}
    return candidates,features,quantiles


def _read(path): return json.loads(path.read_text(encoding='utf-8'))
def _write(path,payload): path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def validate_fixed_source(root,source,anchored_hashes,manifest):
    root,source=Path(root).resolve(),Path(source).resolve()
    files=[source/name for name in ('summary.json','population.json','eligibility_inputs.json')]
    files.extend(source/f"fold{w['id']}_{suffix}.json" for w in manifest['wrappers']
                 for suffix in ('policies','health_scores'))
    actual={str(p.relative_to(root)):_sha(p) for p in files}
    if any(anchored_hashes.get(path)!=digest for path,digest in actual.items()):
        raise ValueError('fixed source missing anchor or source content changed')
    summary=_read(source/'summary.json')
    if (summary.get('state')!='COMPLETE' or summary.get('protected_unchanged') is not True or
            summary.get('evidence_kind')!='T24_NESTED_EVENT_COMPLETE_MEAL_DEVELOPMENT_ONLY'):
        raise ValueError('unexpected fixed source identity')
    forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
    for w in manifest['wrappers']:
        train,health=set(w['train']),set(w['health'])
        policy=_read(source/f"fold{w['id']}_policies.json")
        if set(policy['graph']['policy']['fit_subjects'])!=train:
            raise ValueError('fixed source policy fit identities differ')
        check_graph(policy['graph'],'policy',health|forbidden,train)
        row=_read(source/f"fold{w['id']}_health_scores.json")
        if set(row['train'])!=train or set(row['score'])!=health:
            raise ValueError('fixed source score identities differ')
        check_graph(row['graph'],'threshold',health|forbidden,train)
    return actual


def execute(root,source,pool_source,output):
    from src.data.manifests import load_sensor_index,INDEX_CSV,MEALS_CSV
    root,source,pool_source,output=map(lambda p:Path(p).resolve(),(root,source,pool_source,output))
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit code, spec and completed source evidence first')
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();protected=protection(root);records=[]
    try:
        manifest_path=root/'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json'
        manifest=_read(manifest_path);validate_partitions(manifest)
        allowed=set(manifest['identities']['t24']);forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        reference=_read(source/'eligibility_inputs.json')
        if eligibility_inputs(root,reference)!=reference: raise ValueError('eligible session source changed')
        pool_plan=_read(pool_source/'plan.json')
        if _read(pool_source/'summary.json').get('state')!='COMPLETE': raise ValueError('complete candidate pool source required')
        validate_fixed_source(root,source,pool_plan['source_sha256'],manifest)
        files=[manifest_path,Path(INDEX_CSV),Path(MEALS_CSV),Path(__file__).resolve(),root/'docs/official_recovery_refit_sampling_v1.md']
        files.append(pool_source/'plan.json')
        files.extend(root/path for path in pool_plan['source_sha256'])
        files.extend(root/path for path in ('scripts/research/official_recovery/full_pool_probe.py',
            'src/pipeline/runner.py','src/pipeline/event_stack.py','src/pipeline/context_features.py'))
        for path,digest in pool_plan['source_sha256'].items():
            if path.replace('\\','/').startswith('cache/') and _sha(root/path)!=digest:
                raise ValueError('candidate pool cache changed')
        inputs={str(p.relative_to(root)):_sha(p) for p in set(files)}
        policies={w['id']:_read(source/f"fold{w['id']}_policies.json")['policies']['old'] for w in manifest['wrappers']}
        thresholds={}
        for w in manifest['wrappers']:
            row=_read(source/f"fold{w['id']}_health_scores.json")
            if set(row['train'])!=set(w['train']) or set(row['score'])!=set(w['health']): raise ValueError('source identity mismatch')
            check_graph(row['graph'],'threshold',set(w['health'])|forbidden,set(w['train']))
            thresholds[w['id']]=row['micro_threshold']
        _write(output/'plan.json',dict(id='REFIT-SAMPLING-V1',source_sha256=inputs,policies=policies,
            micro_thresholds=thresholds,verifier_blend=.5,verifier_targets='legacy eligible',
            meta_feature_mode='in_sample_both_arms',expected_model_fits=24,protected=protected,outer_calls=0,
            code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        config=registered_e03_config(0,workers=4)
        old=slice_outer_train(runner.FilesystemDataSource(root).load_outer_train_only(config),allowed)
        population=_read(source/'population.json')
        if [asdict(t) for t in old.train_truths]!=population['legacy_eligible_truths']: raise ValueError('eligible truths changed')
        index=load_sensor_index().to_dict('records');mapping=validated_session_subjects(index)
        bounds={str(r['session_id']):(int(r['timeStamp.startTime']),int(r['timeStamp.endTime'])) for r in index}
        data=expand_candidates(old,load_pools(root,manifest),mapping,bounds)
        truths=[tuple(row) for row in population['complete_truths']]
        def budget():
            if time.perf_counter()-start>1800 or shutil.disk_usage(root).free<10*1024**3:
                raise RuntimeError('refit sampling wall/disk budget exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>64*1024**2:
                raise RuntimeError('refit sampling evidence limit exceeded')
        for wrapper in manifest['wrappers']:
            fold=wrapper['id'];train=wrapper['train'];health=wrapper['health']
            t=slice_outer_train(data,train);v=slice_outer_train(data,health)
            targets=partition_truths(truths,set(health));events={};predictions={};evidence={}
            for arm,role in [('old','full'),('full','sampled')]:
                budget();print(f'refit health{fold} {role}: train{len(train)}/health{len(health)}',flush=True)
                macro,micro,training_stats=fit_windows(t,config,role)
                tc,x,_=candidate_view(t,config,macro,micro,thresholds[fold])
                vc,z,score_quantiles=candidate_view(v,config,macro,micro,thresholds[fold])
                labels=runner._candidate_labels(tc,t.train_truths)
                if set(labels.tolist())!={0,1}: raise ValueError('binary refit verifier targets required')
                components=[]
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    for model in (runner._verifier_estimator(config.seed+3,.1),runner._verifier_lgbm_estimator(config.seed+23)):
                        model.fit(x,labels);components.append(runner._positive_probability(model,z) if len(vc) else np.empty(0))
                scores=.5*components[0]+.5*components[1]
                groups=runner._groups_for([c.event for c in vc],v.subject_by_session)
                predictions[arm]=decode(vc,scores,groups,policies[fold])
                events[arm]=complete_metrics(predictions[arm],data.subject_by_session,targets)
                graph={'macro':{'fit_subjects':list(train),'parents':[]},'micro':{'fit_subjects':list(train),'parents':[]},
                    'verifier':{'fit_subjects':list(train),'parents':['macro','micro']},
                    'score':{'fit_subjects':[],'parents':['verifier']}}
                check_graph(graph,'score',set(health)|forbidden,set(train))
                evidence[arm]=dict(role=role,training_stats=training_stats,health_window_quantiles=score_quantiles,
                    train_candidates=len(tc),train_candidate_positive=int(sum(labels)),graph=graph)
                _write(output/f'fold{fold}_{role}_scores.json',dict(candidates=[asdict(c) for c in vc],
                    scores=scores.tolist(),groups=groups.tolist(),**evidence[arm]))
                gc.collect()
            subjects=[dict(subject=s,events={a:complete_metrics([p for p in pred if data.subject_by_session[p.sid]==s],
                data.subject_by_session,partition_truths(targets,{s})) for a,pred in predictions.items()}) for s in health]
            _write(output/f'fold{fold}_report.json',dict(events=events,subject_events=subjects,evidence=evidence,
                truths=targets,predictions={a:[asdict(p) for p in pred] for a,pred in predictions.items()}))
            records.append(events);print('refit paired complete-meal counts '+json.dumps(events),flush=True)
        budget()
        if (protected!=protection(root) or inputs!={str(p.relative_to(root)):_sha(p) for p in set(files)}
                or eligibility_inputs(root,reference)!=reference): raise ValueError('source or protected artifacts changed')
        result=dict(state='COMPLETE',evidence_kind='T24_FINAL_REFIT_SAMPLING_DEVELOPMENT_ONLY',**assess_gate(records),
            model_fits=24,outer_calls=0,protected_unchanged=True,elapsed_seconds=time.perf_counter()-start)
        _write(output/'summary.json',result);_write(output/'state.json',{'state':'COMPLETE'})
        return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_health_folds=len(records)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('source','pool-source','output'): parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        print(json.dumps(execute(Path(__file__).resolve().parents[3],args.source,args.pool_source,args.output),ensure_ascii=False),flush=True)
