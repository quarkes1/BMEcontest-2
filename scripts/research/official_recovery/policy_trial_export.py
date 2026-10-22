"""Copy accepted model bytes into a separately versioned final-threshold trial."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .consistent_refit import install_new_directory
from .dense_mil_probe import _score_file
from .event_probe import check_graph,complete_metrics,partition_truths
from .geometry_probe import load_context,FIVE_SOURCE
from .negative_clean_probe import _read,_write,protected_sources
from .relative_score_probe import mode_contract,gate,verify_parent,deployment_reference,decode_relative
from .scope_audit import _sha
from .trial_exe import require_clean_tree,source_snapshot
from src.pipeline.artifacts import load_event_stack_bundle,_is_link_or_reparse_point


def trial_paths(root,mode):
    choices={'deploy_f1':'deployment_f1_policy_v3','deploy_f15':'deployment_f15_policy_v3'}
    if mode not in choices: raise ValueError('unregistered policy trial')
    stem=choices[mode];root=Path(root).absolute();version='official_'+stem
    return dict(version=version,source=root/'models/event_stack/official_consistent_refit_v1/deployment',
        bundle=root/'models/event_stack'/version/'deployment',exe_directory=root/'dist/official_trials'/stem,
        evidence=root/'outputs/official_recovery'/f'{stem}_five_attempt2',
        export_report=root/'outputs/official_recovery'/f'{stem}_export',
        build_report=root/'outputs/official_recovery'/f'{stem}_exe',
        acceptance_report=root/'outputs/official_recovery'/f'{stem}_acceptance')


def ensure_new(root,path):
    root,path=Path(root).absolute(),Path(path).absolute()
    if not path.is_relative_to(root) or path==root: raise ValueError('trial path outside workspace')
    current=root
    for part in ('',*path.relative_to(root).parts):
        if part: current=current/part
        if _is_link_or_reparse_point(current): raise ValueError('linked trial path forbidden')
    if path.exists(): raise FileExistsError('independent trial destination exists; never overwrite')


def cleanup_owned(path,parent,prefix):
    path,parent=Path(path),Path(parent)
    if path.parent.resolve()!=parent.resolve() or not path.name.startswith(prefix) or _is_link_or_reparse_point(path):
        raise ValueError('unsafe trial scratch cleanup')
    if path.exists(): shutil.rmtree(path)


def validated_diagnostic(root,mode):
    root=Path(root).resolve();paths=trial_paths(root,mode);folder=paths['evidence'];contract=mode_contract(mode)
    plan=_read(folder/'plan.json');summary=_read(folder/'summary.json')
    if (plan['id']!=contract['id'] or plan['mode']!=mode or plan['cohort']!='five'
        or _read(folder/'state.json')['state']!='COMPLETE' or summary.get('passed') is not True
        or summary.get('evidence_kind')!='FROZEN_MODEL_DEPLOYMENT_POLICY_DEVELOPMENT_TRIAL_ONLY'
        or summary.get('actual_exe_cv') is not False or summary.get('protected_unchanged') is not True):
        raise ValueError('completed matching policy diagnostic required')
    verify_parent(root/contract['t24'],mode)
    if protected_sources(root)!=plan['protected']: raise ValueError('protected first/formal trial changed')
    for name,digest in plan['source_sha256'].items():
        if _sha(root/name)!=digest: raise ValueError('policy source changed: '+name)
    for name,digest in _read(folder/'policy_seal.json')['sha256'].items():
        if _sha(folder/name)!=digest: raise ValueError('policy seal changed')
    reference,_=deployment_reference(root)
    if reference!=plan['deployment_policy']: raise ValueError('deployment reference changed')
    truths,mapping,_,wrappers=load_context(root,'five',parent_directory=contract['t24'],
        parent_gate=lambda rows,kind:gate(rows,kind,mode))
    reports=[];thresholds=[]
    for w in wrappers:
        k=w['id'];train=set(w['train']);score=set(w['score'])
        selected=_read(folder/f'fold{k}_policy.json');policy=selected['policy']
        if policy['admission']!=reference['admission']: raise ValueError('deployment admission differs')
        check_graph(selected['graph'],'policy',score,train)
        cs,p,groups,_=_score_file(root/FIVE_SOURCE/f'fold{k}_outer_A_scores.json',mapping,train,score,())
        report=_read(folder/f'fold{k}_report.json');target=partition_truths(truths,score)
        for arm,decision in (('A',reference),('G',policy)):
            events=decode_relative(cs,p,groups,decision,use_relative=False)
            if ([asdict(e) for e in events]!=report['predictions'][arm]
                or complete_metrics(events,mapping,target)!=report['events'][arm]): raise ValueError('policy event replay differs')
        thresholds.append(float(policy['threshold']));reports.append(report)
    result=gate(reports,'five',mode)
    if not result['passed'] or any(summary.get(k)!=v for k,v in result.items()): raise ValueError('policy gate recomputation differs')
    threshold=float(np.median(thresholds))
    if not 0<=threshold<reference['threshold']: raise ValueError('trial did not reduce the deployed threshold')
    return dict(threshold=threshold,thresholds=thresholds,diagnostic=summary,
        evidence=str(folder.relative_to(root)),evidence_sha256=_sha(folder/'summary.json'))


def copy_policy_bundle(root,mode,threshold,diagnostic):
    root=Path(root).absolute();paths=trial_paths(root,mode);destination=paths['bundle']
    reference,_=deployment_reference(root)
    if not np.isfinite(threshold) or not 0<=threshold<reference['threshold']: raise ValueError('invalid recall policy threshold')
    ensure_new(root,destination.parent)
    parent=destination.parent.parent;parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.policy-trial-',dir=parent))
    try:
        copied=staging/'deployment';shutil.copytree(paths['source'],copied)
        policy=_read(copied/'policy.json');policy['threshold']=float(threshold);_write(copied/'policy.json',policy)
        config=_read(copied/'run_config.json');config.update(trial_version=paths['version'],
            model_reused_from='official_consistent_refit_v1',policy_aggregation=dict(method='median of five training-only final thresholds',
                thresholds=diagnostic.get('thresholds'),fixed_admission=reference['admission']))
        _write(copied/'run_config.json',config)
        manifest=_read(copied/'manifest.json');manifest.update(metrics=dict(evidence_kind='INDEPENDENT_POLICY_TRIAL_NOT_PROMOTED',
            actual_exe_cv=False,no_official_score=True,policy_development_diagnostic=diagnostic),
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() if (root/'.git').exists() else None)
        manifest['files']={name:_sha(copied/name) for name in manifest['files']};_write(copied/'manifest.json',manifest)
        load_event_stack_bundle(copied,expected_role='deployment')
        for name in ('macro.joblib','micro.joblib','verifier_logistic.joblib','verifier_lgbm.joblib','feature_schema.json'):
            if _sha(copied/name)!=_sha(paths['source']/name): raise ValueError('model/schema bytes changed')
        install_new_directory(staging,destination.parent)
    finally:
        cleanup_owned(staging,parent,'.policy-trial-')
    return destination


def export(root,mode):
    root=Path(root).resolve();paths=trial_paths(root,mode);output=paths['export_report'];ensure_new(root,output)
    output.mkdir(parents=True)
    try:
        revision=require_clean_tree(root);protected=protected_sources(root);sources=source_snapshot(root)
        data=validated_diagnostic(root,mode);destination=copy_policy_bundle(root,mode,data['threshold'],data)
        if protected_sources(root)!=protected or source_snapshot(root)!=sources: raise ValueError('source/protected drift during policy copy')
        result=dict(state='COMPLETE',version=paths['version'],mode=mode,bundle=str(destination.relative_to(root)),
            bundle_sha256={p.name:_sha(p) for p in destination.iterdir() if p.is_file()},
            threshold=data['threshold'],thresholds=data['thresholds'],diagnostic=data['diagnostic'],
            model_bytes_unchanged=True,protected_unchanged=True,no_formal_promotion=True,no_official_score=True,
            actual_exe_cv=False,code_sha=revision,source_sha256=sources,protected_sha256=protected)
        _write(output/'summary.json',result);return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('deploy_f1','deploy_f15'),required=True)
    args=parser.parse_args();print(export(Path(__file__).resolve().parents[3],args.mode))
