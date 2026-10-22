"""Fixed subject-relative event decisions after unchanged raw-score admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.special import expit,logit

from .dense_mil_probe import _score_file
from .event_probe import check_graph,complete_metrics,partition_truths
from .geometry_probe import load_context,inputs,T24_SOURCE,FIVE_SOURCE,MANIFEST
from .max_fusion_probe import fusion_gate
from .negative_clean_probe import _read,_write,protected_sources,validate_policy_population
from .scope_audit import _sha
from src.pipeline.candidate_control import CandidateAdmissionConfig,admit_candidates
from src.pipeline import runner

SPEC='docs/official_recovery_relative_score_v1.md'
T24_OUTPUT='outputs/official_recovery/relative_score_v1_t24'
RECALL_SPEC='docs/official_recovery_aggressive_recall_v2.md'
DEPLOY_SPEC='docs/official_recovery_deployment_policy_v3.md'


def mode_contract(mode):
    options={
        'relative':dict(id='RELATIVE-SCORE-V1',t24=T24_OUTPUT,beta=1.,use_relative=True),
        'recall125':dict(id='RECALL125-POLICY-V2',t24='outputs/official_recovery/recall125_policy_v2_t24',beta=1.25,use_relative=False),
        'recall150':dict(id='RECALL150-POLICY-V2',t24='outputs/official_recovery/recall150_policy_v2_t24',beta=1.5,use_relative=False),
        'deploy_f1':dict(id='DEPLOY-F1-POLICY-V3',t24='outputs/official_recovery/deployment_f1_policy_v3_t24_attempt2',beta=1.,use_relative=False,deployment_reference=True),
        'deploy_f15':dict(id='DEPLOY-F15-POLICY-V3',t24='outputs/official_recovery/deployment_f15_policy_v3_t24_attempt2',beta=1.5,use_relative=False,deployment_reference=True)}
    if mode not in options: raise ValueError('unregistered decision mode')
    return options[mode]


def deployment_reference(root):
    from src.pipeline.artifacts import load_event_stack_bundle
    version='official_consistent_refit_v1'
    folder=root/'models/event_stack'/version/'deployment'
    report=root/'outputs/official_recovery/consistent_refit_v1/summary.json';summary=_read(report)
    if summary.get('state')!='COMPLETE' or summary.get('version')!=version or summary.get('protected_unchanged') is not True:
        raise ValueError('accepted first-trial model evidence required')
    actual={p.name:_sha(p) for p in folder.iterdir() if p.is_file()}
    if actual!=summary['bundle_sha256']: raise ValueError('accepted source bundle SHA differs')
    bundle=load_event_stack_bundle(folder,expected_role='deployment');p=bundle.policy
    if (bundle.run_config.get('trial_version')!=version or p['blend_weight']!=.25 or p['max_events_per_group'] is not None
        or not np.isfinite(p['threshold']) or not 0<=p['threshold']<=1): raise ValueError('unsupported deployment policy')
    admission=dict(nms_iou=p['nms_iou'],threshold=p['admission_threshold'],max_candidates_per_subject=p['max_candidates_per_subject'])
    CandidateAdmissionConfig(**admission)
    files={str((folder/name).relative_to(root)):_sha(folder/name) for name in actual}
    files[str(report.relative_to(root))]=_sha(report)
    return dict(admission=admission,threshold=p['threshold']),dict(source_sha256=files,
        trial_version=version,micro_threshold=bundle.run_config['micro_threshold'])


def relative_scores(probabilities,groups):
    p=np.asarray(probabilities,dtype=float);g=np.asarray(groups,dtype=str)
    if p.ndim!=1 or g.shape!=p.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('invalid relative-score inputs')
    result=np.empty_like(p)
    for subject in np.unique(g):
        rows=np.flatnonzero(g==subject)
        anchor=np.quantile(p[rows],.9,method='linear')
        result[rows]=expit(logit(np.clip(p[rows],1e-6,1-1e-6))-.5*logit(np.clip(anchor,1e-6,1-1e-6)))
    return result


def decision_scores(raw,groups,use_relative):
    if use_relative: return relative_scores(raw,groups)
    p=np.asarray(raw,dtype=float)
    if p.ndim!=1 or np.asarray(groups).shape!=p.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('invalid raw decision scores')
    return p


def decode_relative(candidates,raw,groups,policy,*,use_relative=True):
    relative=decision_scores(raw,groups,use_relative)
    rows=admit_candidates(candidates,raw,groups,CandidateAdmissionConfig(**policy['admission']))
    return [candidates[i].event for i in rows if relative[i]>=policy['threshold']]


def choose_threshold(candidates,raw,groups,admission,mapping,truths,*,beta=1.,use_relative=True):
    if beta not in (1.,1.25,1.5): raise ValueError('unregistered beta')
    relative=decision_scores(raw,groups,use_relative)
    rows=np.asarray(admit_candidates(candidates,raw,groups,CandidateAdmissionConfig(**admission)),dtype=int)
    choices=[]
    for threshold in np.unique(np.r_[relative[rows],0.,1.,np.nextafter(1.,2.)]):
        metrics=complete_metrics([candidates[i].event for i in rows if relative[i]>=threshold],mapping,truths)
        denom=(1+beta*beta)*metrics['tp']+beta*beta*metrics['fn']+metrics['fp']
        objective=(1+beta*beta)*metrics['tp']/denom if denom else 0.
        choices.append(((objective,metrics['f1'],metrics['ppv'],float(threshold)),metrics))
    rank,metrics=max(choices,key=lambda row:row[0])
    return dict(admission=dict(admission),threshold=rank[-1],selection_metrics=metrics)


def gate(records,cohort,mode='relative'):
    contract=mode_contract(mode)
    result=fusion_gate(records,cohort)
    if not contract['use_relative']:
        nonworse=sum(r['events']['G']['tp']>=r['events']['A']['tp'] for r in records)
        result.update(passed=bool(result['delta_tp']>=(3 if cohort=='t24' else 5)
            and result['delta_fp']<=3*result['delta_tp']+2 and result['delta_f1']>=-.02-1e-12
            and nonworse>=(2 if cohort=='t24' else 3)),nonworse_recall_folds=nonworse,
            observed_f1_improvement=result['delta_f1']>0,trial_only=True)
        return result
    result['passed']=bool(result['delta_f1']>=.01-1e-12 and result['positive_folds']>=(2 if cohort=='t24' else 3)
        and result['delta_tp']>=0 and (result['delta_fp']<=0 or result['delta_tp']>result['delta_fp']))
    return result


def verify_parent(directory,mode='relative'):
    plan=_read(directory/'plan.json')
    if (plan['id']!=mode_contract(mode)['id'] or plan.get('mode','relative')!=mode or plan['cohort']!='t24'
        or _read(directory/'state.json')['state']!='COMPLETE' or _read(directory/'summary.json').get('passed') is not True):
        raise ValueError('successful relative-score parent required')
    result=gate([_read(directory/f'fold{k}_report.json') for k in range(3)],'t24',mode)
    summary=_read(directory/'summary.json')
    if not result['passed'] or any(summary.get(k)!=v for k,v in result.items()):
        raise ValueError('parent gate recomputation differs')


def load_scores(path,mapping,train,score,excluded):
    candidates,p,groups,graph=_score_file(path,mapping,train,score,excluded)
    rows=_read(path)['rows'];lr=np.asarray([r[7] for r in rows],float);lgbm=np.asarray([r[8] for r in rows],float)
    if any(not np.isfinite(v).all() or np.any((v<0)|(v>1)) for v in (lr,lgbm)):
        raise ValueError('invalid original components')
    if not np.allclose(p,.25*lr+.75*lgbm,atol=1e-12,rtol=0): raise ValueError('original score blend differs')
    graph=dict(graph);graph['decision_score']=dict(fit_subjects=[],parents=['score'])
    check_graph(graph,'decision_score',set(score)|set(excluded),set(train))
    return candidates,p,groups,graph


def execute(root,output,phase='policy',cohort='t24',mode='relative'):
    contract=mode_contract(mode);parent_directory=contract['t24']
    root,output=Path(root).resolve(),Path(output).resolve();started=time.perf_counter();completed=0
    if phase=='policy': output.mkdir(parents=True,exist_ok=False)
    elif phase!='score': raise ValueError('unknown phase')
    def fingerprints():
        result=inputs(root,cohort,parent_directory=parent_directory);result[SPEC]=_sha(root/SPEC)
        result[RECALL_SPEC]=_sha(root/RECALL_SPEC)
        if contract.get('deployment_reference'):
            _,reference=deployment_reference(root);result.update(reference['source_sha256']);result[DEPLOY_SPEC]=_sha(root/DEPLOY_SPEC)
        return result
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
            raise RuntimeError('commit code and evidence before relative selection')
        if cohort=='five': verify_parent(root/parent_directory,mode)
        deployment,reference=deployment_reference(root) if contract.get('deployment_reference') else (None,None)
        if phase=='policy':
            hashes=fingerprints();protected=protected_sources(root)
            _write(output/'plan.json',dict(id=contract['id'],mode=mode,cohort=cohort,source_sha256=hashes,protected=protected,
                code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                model_fits=0,component_score_recomputations=0,beta=contract['beta'],use_relative=contract['use_relative'],
                quantile=.9 if contract['use_relative'] else None,alpha=.5 if contract['use_relative'] else None,
                admission_score='original',event_score=mode,eligibility='inherited frozen baseline evidence'))
            if deployment is not None:
                plan=_read(output/'plan.json');plan.update(deployment_policy=deployment,deployment_reference=reference,
                    reference_policy_scope='prior full-cohort aggregate; not outer-isolated',actual_exe_cv=False)
                _write(output/'plan.json',plan)
        else:
            plan=_read(output/'plan.json');hashes=plan['source_sha256'];protected=plan['protected']
            if plan['id']!=contract['id'] or plan.get('mode','relative')!=mode or plan['cohort']!=cohort or _read(output/'state.json')['state']!='POLICY_FROZEN':
                raise ValueError('matching sealed relative policies required')
            if any(_sha(output/n)!=h for n,h in _read(output/'policy_seal.json')['sha256'].items()):
                raise ValueError('relative seal changed')
        def guard():
            if hashes!=fingerprints() or protected!=protected_sources(root): raise ValueError('relative input/protected drift')
            peak=runner.peak_working_set_bytes()
            if peak is None or peak>12*1024**3: raise RuntimeError('relative RSS budget')
            if time.perf_counter()-started>600 or shutil.disk_usage(root).free<10*1024**3: raise RuntimeError('relative wall/disk budget')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>16*1024**2: raise RuntimeError('relative evidence budget')
        guard();truths,mapping,_,wrappers=load_context(root,cohort,parent_directory=parent_directory,
            parent_gate=lambda rows,kind:gate(rows,kind,mode))
        source=root/(T24_SOURCE if cohort=='t24' else FIVE_SOURCE)
        if deployment is not None:
            choices={p.name:_read(p)['A_meta']['micro_threshold'] for p in source.glob('*training.json')}
            if any(v!=reference['micro_threshold'] for name,v in choices.items() if '_outer_' in name):
                raise ValueError('outer micro policy differs from source deployment')
            if phase=='policy': _write(output/'source_micro_thresholds.json',choices)
            elif choices!=_read(output/'source_micro_thresholds.json'): raise ValueError('micro provenance changed')
        if cohort=='t24':
            manifest=_read(root/MANIFEST)
            parts={w['id']:[(s['train'],s['score']) for s in w['policy_splits']] for w in manifest['wrappers']}
            forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        else:
            parts={int(k):v for k,v in _read(source/'plan.json')['policy_partitions'].items()};forbidden=set()
        records=[]
        for w in wrappers:
            fold=w['id'];train=set(w['train']);score=set(w['score'])
            frozen=_read(source/f'fold{fold}_policies.json')['policies']['A']
            admission=deployment['admission'] if deployment is not None else frozen['admission']
            if phase=='policy':
                candidates=[];raw=[];groups=[];seen=[];graph={}
                for i,(fit,valid) in enumerate(parts[fold]):
                    fit,valid=set(fit),set(valid)
                    if fit&valid or fit|valid!=train: raise ValueError('relative producer partition differs')
                    cs,ps,gs,parent=load_scores(source/f'fold{fold}_policy{i}_A_scores.json',mapping,fit,valid,score|forbidden)
                    candidates.extend(cs);raw.extend(ps);groups.extend(gs);seen.extend(valid)
                    for name,node in parent.items():
                        graph[f'p{i}_{name}']=dict(fit_subjects=node['fit_subjects'],parents=[f'p{i}_{p}' for p in node['parents']])
                validate_policy_population(seen,groups,train)
                graph['policy']=dict(fit_subjects=sorted(train),parents=[f'p{i}_decision_score' for i in range(3)])
                check_graph(graph,'policy',score|forbidden,train)
                policy=choose_threshold(candidates,np.asarray(raw),groups,admission,mapping,partition_truths(truths,train),
                    beta=contract['beta'],use_relative=contract['use_relative'])
                _write(output/f'fold{fold}_policy.json',dict(policy=policy,graph=graph));completed+=1
                print(f'relative policy {fold} frozen',flush=True);continue
            selected=_read(output/f'fold{fold}_policy.json');check_graph(selected['graph'],'policy',score|forbidden,train)
            if selected['policy']['admission']!=admission: raise ValueError('relative admission changed')
            role='health' if cohort=='t24' else 'outer'
            cs,ps,gs,graph=load_scores(source/f'fold{fold}_{role}_A_scores.json',mapping,train,score,forbidden)
            events={'A':w['health_events'],'G':decode_relative(cs,ps,gs,selected['policy'],use_relative=contract['use_relative'])};targets=partition_truths(truths,score)
            if deployment is not None:
                events['source_A']=events['A'];events['A']=decode_relative(cs,ps,gs,deployment,use_relative=False)
            row=dict(fold=fold,events={a:complete_metrics(es,mapping,targets) for a,es in events.items()},truths=targets,
                predictions={a:[asdict(e) for e in es] for a,es in events.items()},policy=selected['policy'],
                score_graph=graph,policy_graph=selected['graph'],
                subject_events=[dict(subject=s,events={a:complete_metrics([e for e in es if mapping[e.sid]==s],mapping,partition_truths(targets,{s}))
                    for a,es in events.items()}) for s in sorted(score)])
            records.append(row);completed+=1;_write(output/f'fold{fold}_report.json',row)
            print(f'relative score {fold}: '+str(row['events']),flush=True)
        guard()
        if phase=='policy':
            _write(output/'policy_seal.json',dict(sha256={p.name:_sha(p) for p in output.glob('*.json') if p.name not in ('state.json','policy_seal.json')}))
            result=dict(state='POLICY_FROZEN',completed_folds=completed)
        else:
            kind=('FROZEN_MODEL_DEPLOYMENT_POLICY_DEVELOPMENT_TRIAL_ONLY' if deployment is not None else
                'FROZEN_MODEL_RELATIVE_DECISION_DEVELOPMENT_ONLY' if contract['use_relative'] else 'FROZEN_MODEL_RECALL_POLICY_TRIAL_ONLY')
            result=dict(state='COMPLETE',evidence_kind=kind,actual_exe_cv=False,cohort=cohort,
                **gate(records,cohort,mode),mode=mode,model_fits=0,component_score_recomputations=0,protected_unchanged=True,
                elapsed_seconds=time.perf_counter()-started)
            _write(output/'summary.json',result)
        _write(output/'state.json',dict(state=result['state']));return result
    except BaseException as exc:
        if output.is_dir(): _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_folds=completed))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('policy','score'),default='policy');parser.add_argument('--cohort',choices=('t24','five'),default='t24')
    parser.add_argument('--mode',choices=('relative','recall125','recall150','deploy_f1','deploy_f15'),default='relative')
    args=parser.parse_args();print(execute(Path(__file__).resolve().parents[3],args.output,args.phase,args.cohort,args.mode),flush=True)
