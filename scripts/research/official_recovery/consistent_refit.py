"""Independently versioned, sampled-window / OOF-meta deployment refit."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .consistent_validation import validation_gate,positive_count
from .eligibility_provenance import eligibility_inputs
from .event_probe import check_graph
from .scope_audit import _sha,protection,validated_session_subjects
from scripts.promote_event_stack import canonical_deployment_policy,_validated_feature_schema,_model_mapping,_config_from_summary
from scripts.research.representation_v2.crossfit import build_candidate_ledgers
from src.pipeline import runner
from src.pipeline.artifacts import EventStackBundle,write_event_stack_bundle,load_event_stack_bundle
from src.pipeline.candidate_control import CandidateAdmissionConfig
from src.pipeline.event_stack import EventMetrics,EventSelectionPolicy
from src.pipeline.micro_cache import sample_training_rows,MICRO_SAMPLE_SEED

VERSION='official_consistent_refit_v1'


def _read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def _write(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sample_batch(batch):
    keys=[(w.sid,w.start_ms,w.end_ms) for w in batch.windows]
    if len(set(keys))!=len(keys): raise ValueError('duplicate training window')
    if not set(np.unique(batch.labels)).issubset({-1,0,1}): raise ValueError('unsupported training label')
    selected=sample_training_rows(batch.labels,[w.sid for w in batch.windows],MICRO_SAMPLE_SEED,negative_ratio=3)
    sampled=runner.WindowBatch(batch.features[selected],batch.labels[selected],tuple(batch.windows[i] for i in selected))
    positives=positive_count(sampled.labels)
    identity=[[w.sid,int(w.start_ms),int(w.end_ms),int(label)] for w,label in zip(sampled.windows,sampled.labels)]
    return sampled,dict(seed=MICRO_SAMPLE_SEED,negative_ratio=3,rows=len(selected),positive=positives,
        positive_ratio=positives/len(selected) if len(selected) else 0.,
        sample_identity_sha256=hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest())


def aggregate_policy(settings):
    if set(map(int,settings))!=set(range(5)): raise ValueError('five sealed settings required')
    records=[]
    for frozen in settings.values():
        admission=frozen['policy']['admission']
        records.append(dict(micro_threshold=frozen['micro_threshold'],selected_blend_weight=frozen['blend'],
            selected_admission_nms_iou=admission['nms_iou'],selected_admission_threshold=admission['threshold'],
            selected_admission_subject_cap=admission['max_candidates_per_subject'],threshold=frozen['policy']['threshold'],
            max_events_per_subject=None,verifier_c=frozen['verifier_c']))
    return canonical_deployment_policy(records)


def validated_gate(root,source):
    root,source=Path(root).resolve(),Path(source).resolve();summary=_read(source/'summary.json')
    if (summary.get('state')!='COMPLETE' or summary.get('passed') is not True or
        summary.get('protected_unchanged') is not True or summary.get('policy_selections')!=0 or
        summary.get('evidence_kind')!='FIVEFOLD_DEVELOPMENT_REFIT_CONSISTENCY_NOT_OFFICIAL_SCORE'):
        raise ValueError('completed consistency gate required')
    plan=_read(source/'plan.json')
    if plan['id']!='CONSISTENT-REFIT-FIVEFOLD-V1' or len(plan['subject_fold'])!=39:
        raise ValueError('gate population differs')
    rows=[]
    for fold in range(5):
        report=_read(source/f'fold{fold}_report.json')
        score={s for s,k in plan['subject_fold'].items() if k==fold};train=set(plan['subject_fold'])-score
        if report['fold']!=fold or set(report['score'])!=score or set(report['train'])!=train:
            raise ValueError('gate fold identity differs')
        rows.append(dict(fold=fold,**report['events']))
        for arm in ('release','consistent'):
            evidence=_read(source/f'fold{fold}_{arm}_scores.json')
            if set(evidence['train_subjects'])!=train or set(evidence['score_subjects'])!=score:
                raise ValueError('gate score identity differs')
            check_graph(evidence['graph'],'score',score,train)
    gate=validation_gate(rows)
    if not gate['passed'] or any(summary[key]!=gate[key] for key in gate): raise ValueError('gate recomputation differs')
    if any(r['truths']!=254 for r in gate['aggregate'].values()): raise ValueError('gate truth scope differs')
    # These two historical files were edited only AFTER completion: ledger and int8 logging fix.
    # All actual model/input/factory files remain bound to their original evidence SHA.
    historical={'scripts/research/official_recovery/consistent_validation.py','docs/official_recovery_consistent_refit_validation_v1.md'}
    for path,digest in plan['source_sha256'].items():
        if path.replace('\\','/') not in historical and _sha(root/path)!=digest:
            raise ValueError('gate input changed: '+path)
    if protection(root)!=plan['protected']: raise ValueError('protected release changed')
    return plan,summary


def fit_consistent(data,config,policy):
    subjects=set(data.subject_by_session.values());graph={}
    ledger,_=build_candidate_ledgers(data,replace(config,seed=config.seed+2,
        micro_threshold_grid=(policy['micro_threshold'],)),subjects,(),fit_audit=graph)
    macro_keep=np.asarray(data.window_train.labels)>=0
    micro_keep=runner._micro_training_keep(data.micro_window_train,data.train_truths,config.micro_positive_middle_fraction)
    event_policy=EventSelectionPolicy(policy['threshold'],policy['max_events_per_group'],EventMetrics(0,0,0,0.,0.,0.))
    admission=CandidateAdmissionConfig(policy['nms_iou'],policy['admission_threshold'],policy['max_candidates_per_subject'])
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        fit=runner._fit_final_models(config,
            train_features=runner._with_time_prior(data.window_train.features,data.window_train.windows)[macro_keep],
            train_labels=data.window_train.labels[macro_keep],micro_train_features=data.micro_window_train.features[micro_keep],
            micro_train_labels=data.micro_window_train.labels[micro_keep],train_candidate_features=ledger.features,
            train_candidate_labels=ledger.labels,verifier_c=policy['verifier_c'],policy=event_policy,admission=admission,
            blend_weight=policy['blend_weight'])
    graph['verifier']={'fit_subjects':sorted(subjects),'parents':['candidates']}
    check_graph(graph,'verifier',set(),subjects)
    statistics=dict(candidate_count=len(ledger.candidates),candidate_positive=positive_count(ledger.labels),
        verifier_feature_width=int(ledger.features.shape[1]),fit_graph=graph,
        micro_fit_rows=int(np.count_nonzero(micro_keep)),micro_fit_positive=positive_count(data.micro_window_train.labels[micro_keep]))
    return fit,statistics


def install_new_directory(source,destination):
    # Windows rename has atomic fail-if-destination-exists semantics, including empty dirs.
    # Do not substitute Path.replace or the POSIX rename implementation here.
    if os.name!='nt': raise RuntimeError('atomic no-replace publication requires Windows')
    os.rename(source,destination)


def publish_bundle_new(destination,bundle,event_stack_root):
    root=Path(event_stack_root);destination=Path(destination)
    if destination!=root/VERSION/'deployment': raise ValueError('unexpected independent bundle destination')
    root.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.consistent-refit-',dir=root))
    try:
        write_event_stack_bundle(staging/'deployment',bundle,event_stack_root=root)
        install_new_directory(staging,destination.parent)
    finally:
        if staging.exists():
            if staging.parent!=root or not staging.name.startswith('.consistent-refit-'):
                raise RuntimeError('unsafe model staging cleanup')
            shutil.rmtree(staging)


def execute(root,source,output):
    from src.data.manifests import load_sensor_index
    root,source,output=map(lambda p:Path(p).resolve(),(root,source,output))
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
            raise RuntimeError('commit code and evidence before refitting')
        gate_plan,gate=validated_gate(root,source);protected=protection(root)
        destination=root/'models/event_stack'/VERSION/'deployment'
        if destination.parent.exists(): raise FileExistsError('independent model version already exists')
        mapping=validated_session_subjects(load_sensor_index().to_dict('records'))
        eligibility=eligibility_inputs(root,mapping)
        _write(output/'eligibility_inputs.json',eligibility)
        paths={root/p for p in gate_plan['source_sha256'] if p.replace('\\','/') not in
            {'scripts/research/official_recovery/consistent_validation.py','docs/official_recovery_consistent_refit_validation_v1.md'}}
        paths.update(source.glob('*.json'))
        paths.update({Path(__file__).resolve(),root/'scripts/research/official_recovery/consistent_validation.py',
            root/'src/pipeline/micro_cache.py',root/'scripts/promote_event_stack.py',root/'src/pipeline/artifacts.py',
            root/'docs/official_recovery_consistent_refit_validation_v1.md',output/'eligibility_inputs.json'})
        fingerprints=tuple(runner.FilesystemDataSource._deployment_file_state(p) for p in sorted(paths,key=str))
        policy=aggregate_policy(gate_plan['settings'])
        configs=tuple(_config_from_summary(
            gate_plan['settings'][str(k)]['config']) for k in range(5))
        config=replace(configs[0],workers=8)
        _write(output/'plan.json',dict(id='CONSISTENT-DEPLOYMENT-REFIT-V1',version=VERSION,policy=policy,
            sample_seed=MICRO_SAMPLE_SEED,negative_ratio=3,inner_splits=config.inner_splits,
            source_fingerprints=fingerprints,protected=protected,code_sha=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=root,text=True).strip(),no_training_f1=True,no_formal_promotion=True))
        def guard():
            if time.perf_counter()-start>7200 or shutil.disk_usage(root).free<10*1024**3:
                raise RuntimeError('refit wall/disk budget exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>128*1024**2:
                raise RuntimeError('refit evidence budget exceeded')
        guard();print('Loading unique full-target validation union',flush=True)
        full=runner.build_full_target_dataset(configs,runner.FilesystemDataSource(root),expected_truth_count=153)
        canonical={sid:mapping[sid] for sid in full.subject_by_session}
        if set(canonical.values())!=set(gate_plan['subject_fold']): raise ValueError('refit cohort differs from validated cohort')
        macro,macro_stats=sample_batch(full.window_train);micro,micro_stats=sample_batch(full.micro_window_train)
        data=runner.OuterTrainOnlyData(macro,full.candidate_train,full.train_truths,canonical,full.session_bounds_by_sid,
                                      micro,full.micro_candidate_train)
        _write(output/'sampling.json',dict(macro=macro_stats,micro=micro_stats,subjects=sorted(set(canonical.values())),
            truth_count=len(full.train_truths),candidate_macro_rows=len(full.candidate_train.windows),
            candidate_micro_rows=len(full.micro_candidate_train.windows)))
        del full;gc.collect();guard();print('Fitting four-fold subject OOF meta and final sampled windows',flush=True)
        fit,statistics=fit_consistent(data,config,policy);guard()
        if eligibility_inputs(root,mapping)!=eligibility: raise ValueError('eligibility arrays changed during refit')
        if fingerprints!=tuple(runner.FilesystemDataSource._deployment_file_state(p) for p in sorted(paths,key=str)):
            raise ValueError('refit source changed')
        if protection(root)!=protected: raise ValueError('protected release changed')
        bundle=EventStackBundle(models=_model_mapping(fit),policy={k:policy[k] for k in
            ('blend_weight','admission_threshold','nms_iou','max_candidates_per_subject','threshold','max_events_per_group')},
            run_config=dict(deployment_training='single sampled full-target union; subject-OOF verifier',
                policy_aggregation=policy['aggregation'],micro_threshold=policy['micro_threshold'],verifier_c=policy['verifier_c'],
                fold_configs=[asdict(c) for c in configs],trial_version=VERSION),
            feature_schema=_validated_feature_schema(fit,config),metrics=dict(evidence_kind='INDEPENDENT_OFFICIAL_TRIAL_NOT_PROMOTED',
                development_validation=gate,no_official_score=True,no_training_f1=True),source_fingerprints=fingerprints,role='deployment')
        publish_bundle_new(destination,bundle,event_stack_root=root/'models/event_stack')
        loaded=load_event_stack_bundle(destination,expected_role='deployment')
        if set(loaded.models)!=set(bundle.models): raise ValueError('serialized model set differs')
        result=dict(state='COMPLETE',version=VERSION,bundle=str(destination.relative_to(root)),
            bundle_sha256={p.name:_sha(p) for p in destination.iterdir() if p.is_file()},sampling=dict(macro=macro_stats,micro=micro_stats),
            statistics=statistics,protected_unchanged=protection(root)==protected,no_training_f1=True,
            no_formal_promotion=True,elapsed_seconds=time.perf_counter()-start)
        _write(output/'summary.json',result);_write(output/'state.json',{'state':'COMPLETE'});return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=8):
        print(json.dumps(execute(Path(__file__).resolve().parents[3],args.source,args.output)),flush=True)
