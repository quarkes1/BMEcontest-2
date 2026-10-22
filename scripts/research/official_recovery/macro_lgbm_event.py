"""Exploratory M1: nested macro-LGBM event validation against frozen A."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import gc
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .consistent_validation import fit_verifiers,_save_scores
from .eligibility_provenance import eligibility_inputs
from .event_probe import check_graph,choose_complete_policy,complete_metrics,decode,partition_truths
from .geometry_probe import load_context as baseline_context,T24_SOURCE,FIVE_SOURCE
from .macro_lgbm_probe import macro_estimator,SPEC
from .max_fusion_probe import fusion_gate
from .negative_clean import prepare_train
from .negative_clean_probe import _read,_write,protected_sources,load_context as train_context,_inputs,validate_policy_population
from .scope_audit import _sha
from scripts.research.representation_v2.crossfit import build_candidate_ledgers
from src.pipeline import runner

T24_OUTPUT='outputs/official_recovery/macro_lgbm_v1_t24'
WINDOW_SOURCE='outputs/official_recovery/macro_lgbm_v1_window'
WINDOW_PARENT_PLAN_SHA='2b6c1d8fc33ac0462c1e6a516163af79c57289bb8a12a2f79d76d1639a457e3a'
WINDOW_PARENT_CODE_SHA='24aa4667597483a9eea144bf14d5a2986a9c485a'
ENSEMBLE_SPEC='docs/official_recovery_macro_ensemble_v1.md'
ENSEMBLE_T24='outputs/official_recovery/macro_soft_ensemble_v1_t24'
M1_PARENT_PLAN_SHA='fe1eb7de1de58c99cf6c610f8ef62c970944666946e575af7bfcb66e1571f8e2'


def macro_ensemble_factory(seed):
    from sklearn.ensemble import VotingClassifier
    return VotingClassifier([('hgb',runner._window_estimator(seed)),('lgbm',macro_estimator(seed))],
        voting='soft',n_jobs=1)


def macro_contract(hypothesis):
    contracts={
        'lgbm':dict(id='MACRO-LGBM-EVENT-M1',t24=T24_OUTPUT,spec=SPEC,factory=macro_estimator,member_fits=2),
        'ensemble':dict(id='MACRO-SOFT-ENSEMBLE-V1',t24=ENSEMBLE_T24,spec=ENSEMBLE_SPEC,factory=macro_ensemble_factory,member_fits=3),
        'full_weight':dict(id='WEIGHTED-FULL-NEGATIVES-V1',t24='outputs/official_recovery/weighted_full_negative_v1_t24',
            spec='docs/official_recovery_weighted_full_negative_v1.md',factory=runner._window_estimator,member_fits=2),
        'full_labels':dict(id='COMPLETE-SUPERVISION-V1',t24='outputs/official_recovery/complete_supervision_v1_t24',
            spec='docs/official_recovery_complete_supervision_v1.md',factory=runner._window_estimator,member_fits=2),
    }
    if hypothesis not in contracts: raise ValueError('unknown macro hypothesis')
    return contracts[hypothesis]


def event_gate(records,cohort,hypothesis='lgbm'):
    macro_contract(hypothesis)
    result=fusion_gate(records,cohort)
    if hypothesis in ('ensemble','full_weight','full_labels'):
        result['passed']=bool(result['delta_f1']>=.01-1e-12 and result['positive_folds']>=(2 if cohort=='t24' else 3)
            and result['delta_tp']>=0 and (result['delta_fp']<=0 or result['delta_tp']>result['delta_fp']))
    return result


def verify_event_parent(root,hypothesis):
    contract=macro_contract(hypothesis);parent=root/contract['t24']
    plan=_read(parent/'plan.json');expected=contract['id']
    if (plan['id']!=expected or plan['cohort']!='t24' or plan.get('hypothesis','lgbm')!=hypothesis
        or _read(parent/'state.json')['state']!='COMPLETE' or _read(parent/'summary.json')['passed'] is not True):
        raise ValueError('successful matching macro parent required')


def verify_failed_m1(root,protected):
    parent=root/T24_OUTPUT;plan=_read(parent/'plan.json');summary=_read(parent/'summary.json')
    if (_sha(parent/'plan.json')!=M1_PARENT_PLAN_SHA or plan['id']!='MACRO-LGBM-EVENT-M1'
        or plan.get('hypothesis','lgbm')!='lgbm' or plan['cohort']!='t24' or plan['protected']!=protected
        or _read(parent/'state.json')['state']!='COMPLETE' or summary['state']!='COMPLETE'
        or summary['protected_unchanged'] is not True): raise ValueError('reviewed failed M1 identity differs')
    result=fusion_gate([_read(parent/f'fold{k}_report.json') for k in range(3)],'t24')
    if (result['passed'] or any(summary[k]!=v for k,v in result.items())
        or tuple(result['aggregate']['G'][k] for k in ('tp','fp','fn'))!=(87,60,73)):
        raise ValueError('reviewed failed M1 result differs')
    allowed={SPEC,'scripts/research/official_recovery/macro_lgbm_event.py'}
    for name,digest in plan['source_sha256'].items():
        if name.replace('\\','/') not in allowed and _sha(root/name)!=digest:
            raise ValueError('failed M1 source differs: '+name)


def verify_window_parent(directory,protected,root=None):
    root=Path(root) if root is not None else Path(directory)
    state=_read(directory/'state.json');summary=_read(directory/'summary.json');plan=_read(directory/'plan.json')
    if _sha(directory/'plan.json')!=WINDOW_PARENT_PLAN_SHA or plan.get('code_sha')!=WINDOW_PARENT_CODE_SHA:
        raise ValueError('reviewed window parent plan/code differs')
    allowed={SPEC,'scripts/research/representation_v2/crossfit.py'}
    for name,digest in plan['source_sha256'].items():
        if name.replace('\\','/') not in allowed and _sha(root/name)!=digest:
            raise ValueError('window parent source differs: '+name)
    if (state['state']!='COMPLETE' or summary['state']!='COMPLETE' or summary['passed'] is not False
        or abs(summary['delta_ap']-.018877746493557)>1e-12 or summary['positive_folds']!=2
        or summary['protected_unchanged'] is not True or plan['id']!='MACRO-LGBM-V1-WINDOW'
        or plan['protected']!=protected):
        raise ValueError('explicit M1 window parent identity differs')


def fingerprints(root,cohort,hypothesis='lgbm'):
    hashes=_inputs(root)
    files={root/SPEC};files.update((root/WINDOW_SOURCE).glob('*'))
    files.update((root/T24_SOURCE).glob('*.json'))
    contract=macro_contract(hypothesis);parent_directory=contract['t24'];files.add(root/contract['spec'])
    if hypothesis=='ensemble':
        files.add(root/ENSEMBLE_SPEC);files.update((root/T24_OUTPUT).glob('*.json'))
    if cohort=='five':
        from .negative_clean_validation import inputs as five_inputs
        hashes.update(five_inputs(root));files.update((root/FIVE_SOURCE).glob('*.json'));files.update((root/parent_directory).glob('*.json'))
    hashes.update({str(p.relative_to(root)):_sha(p) for p in files if p.is_file()})
    return hashes


def fit_score(data,config,train,score,excluded,truths,output,prefix,source,*,factory=macro_estimator,arm='L',weighted=False,complete=False):
    train,score,excluded=map(set,(train,score,excluded))
    if train&(score|excluded) or score&excluded: raise ValueError('producer roles overlap')
    weights=None;weight_stats=None
    selector=None;micro_details={}
    if complete:
        from .complete_supervision import prepare_complete_train,choose_micro
        if weighted or config.micro_positive_middle_fraction is not None: raise ValueError('incompatible complete supervision options')
        prepared,sampling=prepare_complete_train(data,train,truths)
        targets=partition_truths(truths,train)
        def selector(windows,unused_eligible,grid,micro_config):
            selected=choose_micro(windows,prepared.subject_by_session,targets,grid,micro_config)
            micro_details.update(threshold=selected.threshold,metrics=selected.metrics,candidate_count=selected.candidate_count,budget=selected.budget)
            return selected
    elif weighted:
        from .weighted_full_negatives import prepare_weighted_train
        prepared,sampling,weights,weight_stats=prepare_weighted_train(data,train,truths)
    else: prepared,sampling=prepare_train(data,train,truths,'A')
    scoring=replace(data,window_train=prepared.window_train,micro_window_train=prepared.micro_window_train)
    graph={};seed=config.seed+2
    t,v=build_candidate_ledgers(scoring,replace(config,seed=seed),train,score,fit_audit=graph,
        macro_estimator_factory=lambda:factory(seed),macro_fit_sample_weight=weights,micro_threshold_selector=selector)
    # Window sampling and micro-only learned threshold must remain precisely the baseline contract.
    if not complete:
        frozen=_read(source/f'{prefix}_training.json')
        if sampling!=frozen['A'] or t.micro_threshold!=frozen['A_meta']['micro_threshold']:
            raise ValueError('single-factor sampling or micro threshold differs')
    labels=t.labels
    if complete:
        from .event_probe import full_candidate_labels
        labels=full_candidate_labels(t.candidates,prepared.subject_by_session,targets)
    p,components=fit_verifiers(t.features,labels,v.features,config,.1,.25)
    graph['verifier']=dict(fit_subjects=sorted(train),parents=['candidates'])
    graph['score']=dict(fit_subjects=[],parents=['verifier']);check_graph(graph,'score',score|excluded,train)
    _save_scores(output/f'{prefix}_{arm}_scores.json',v.candidates,p,components,v.groups,graph,train,score)
    _write(output/f'{prefix}_training.json',dict(sample=sampling,micro_threshold=t.micro_threshold,
        meta_rows=len(labels),meta_positive=int(np.count_nonzero(labels==1)),weighted_macro=weight_stats,
        complete_micro_selection=micro_details,
        meta_label_sha256=__import__('hashlib').sha256(np.asarray(labels,dtype=np.int8).tobytes()).hexdigest()))
    gc.collect();return v,p,graph


def execute(root,output,phase='policy',cohort='t24',hypothesis='lgbm'):
    contract=macro_contract(hypothesis);factory=contract['factory'];parent_directory=contract['t24']
    arm={'lgbm':'L','ensemble':'E','full_weight':'W','full_labels':'C'}[hypothesis]
    window_fits=contract['member_fits']*(60 if cohort=='t24' else 100)
    root,output=Path(root).resolve(),Path(output).resolve();completed=0;started=time.perf_counter()
    if phase=='policy': output.mkdir(parents=True,exist_ok=False)
    elif phase!='score': raise ValueError('unknown event phase')
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
            raise RuntimeError('commit source/evidence before event training')
        protected=protected_sources(root)
        if hypothesis in ('lgbm','ensemble'): verify_window_parent(root/WINDOW_SOURCE,protected,root)
        if hypothesis=='ensemble': verify_failed_m1(root,protected)
        if phase=='policy':
            hashes=fingerprints(root,cohort,hypothesis)
            plan=dict(id=contract['id'],
                hypothesis=hypothesis,cohort=cohort,source_sha256=hashes,protected=protected,
                started_epoch=time.time(),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                window_gate_failed=hypothesis in ('lgbm','ensemble'),exploratory_amendment=hypothesis,expected_window_fits=window_fits,
                expected_meta_fits=24 if cohort=='t24' else 40)
            _write(output/'plan.json',plan)
        else:
            plan=_read(output/'plan.json');hashes=plan['source_sha256']
            if (plan['cohort']!=cohort or plan.get('hypothesis','lgbm')!=hypothesis or plan['protected']!=protected
                or _read(output/'state.json')['state']!='POLICY_FROZEN'):
                raise ValueError('matching frozen policy phase required')
            if any(_sha(output/n)!=h for n,h in _read(output/'policy_seal.json')['sha256'].items()): raise ValueError('event seal differs')
        def guard():
            if time.time()-plan['started_epoch']>7200 or shutil.disk_usage(root).free<10*1024**3: raise RuntimeError('event wall/disk limit')
            peak=runner.peak_working_set_bytes()
            if peak is None or peak>12*1024**3: raise RuntimeError('Windows memory measurement required within budget')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>128*1024**2: raise RuntimeError('event evidence limit')
        def unchanged():
            if hashes!=fingerprints(root,cohort,hypothesis) or protected!=protected_sources(root): raise ValueError('event input/protection changed')
        guard();unchanged()
        if cohort=='five': verify_event_parent(root,hypothesis)
        truths,mapping,_,wrappers=baseline_context(root,cohort,parent_directory=parent_directory,
            parent_gate=lambda rows,kind:event_gate(rows,kind,hypothesis))
        population={s for w in wrappers for s in (*w['train'],*w['score'])}
        before_eligibility=eligibility_inputs(root,{sid for sid,s in mapping.items() if s in population})
        if cohort=='t24':
            config,data,loaded_truths,manifest,forbidden=train_context(root)
            configs={w['id']:config for w in manifest['wrappers']}
            parts={w['id']:[(s['train'],s['score']) for s in w['policy_splits']] for w in manifest['wrappers']}
        elif cohort=='five':
            from .negative_clean_validation import load_context as five_context
            cs,data,loaded_truths,subject_fold,_=five_context(root)
            configs={i:c for i,c in enumerate(cs)};forbidden=set()
            parts={int(k):v for k,v in _read(root/FIVE_SOURCE/'plan.json')['policy_partitions'].items()}
            if subject_fold!={r['subject']:r['fold'] for r in _read(root/'outputs/official_recovery/scope_audit_confirmed/summary.json')['per_subject']}:
                raise ValueError('fivefold fit population differs')
        else: raise ValueError('unknown event cohort')
        if set(loaded_truths)!=set(truths) or any(mapping[sid]!=s for sid,s in data.subject_by_session.items()):
            raise ValueError('baseline/fitting population identity differs')
        eligibility=eligibility_inputs(root,data.subject_by_session)
        if eligibility!={sid:before_eligibility[sid] for sid in data.subject_by_session}:
            raise ValueError('eligibility changed during loading')
        if phase=='policy': _write(output/'eligibility_inputs.json',eligibility)
        elif eligibility!=_read(output/'eligibility_inputs.json'): raise ValueError('eligibility changed since selection')
        source=root/(T24_SOURCE if cohort=='t24' else FIVE_SOURCE);records=[]
        for w in wrappers:
            guard();fold=w['id'];train=set(w['train']);score=set(w['score']);config=configs[fold]
            if phase=='policy':
                candidates=[];probabilities=[];groups=[];graph={};seen=[]
                for i,(fit,valid) in enumerate(parts[fold]):
                    guard()
                    if set(fit)&set(valid) or set(fit)|set(valid)!=train: raise ValueError('policy partition differs')
                    print(f'macro-event {cohort} policy {fold}/{i}: train{len(fit)} score{len(valid)}',flush=True)
                    ledger,p,parent=fit_score(data,config,fit,valid,score|forbidden,truths,output,f'fold{fold}_policy{i}',source,
                        factory=factory,arm=arm,weighted=hypothesis=='full_weight',complete=hypothesis=='full_labels')
                    candidates.extend(ledger.candidates);probabilities.extend(p);groups.extend(ledger.groups);seen.extend(valid)
                    for name,node in parent.items():
                        graph[f'p{i}_{name}']=dict(fit_subjects=node['fit_subjects'],parents=[f'p{i}_{p}' for p in node['parents']])
                    completed+=1;del ledger,p,parent;gc.collect()
                validate_policy_population(seen,groups,train)
                graph['policy']=dict(fit_subjects=sorted(train),parents=[f'p{i}_score' for i in range(3)])
                check_graph(graph,'policy',score|forbidden,train)
                policy=choose_complete_policy(candidates,np.asarray(probabilities),groups,partition_truths(truths,train),mapping,config)
                _write(output/f'fold{fold}_policy.json',dict(policy=policy,graph=graph))
                print(f'macro-event {cohort} fold{fold} policy frozen',flush=True);continue
            frozen=_read(output/f'fold{fold}_policy.json');check_graph(frozen['graph'],'policy',score|forbidden,train)
            role='health' if cohort=='t24' else 'outer'
            print(f'macro-event {cohort} score {fold}: train{len(train)} score{len(score)}',flush=True)
            ledger,p,graph=fit_score(data,config,train,score,forbidden,truths,output,f'fold{fold}_{role}',source,
                factory=factory,arm=arm,weighted=hypothesis=='full_weight',complete=hypothesis=='full_labels')
            events={'A':w['health_events'],'G':decode(ledger.candidates,p,ledger.groups,frozen['policy'])};targets=partition_truths(truths,score)
            row=dict(fold=fold,events={a:complete_metrics(es,mapping,targets) for a,es in events.items()},truths=targets,
                predictions={a:[asdict(e) for e in es] for a,es in events.items()},score_graph=graph,policy_graph=frozen['graph'],
                subject_events=[dict(subject=s,events={a:complete_metrics([e for e in es if mapping[e.sid]==s],mapping,partition_truths(targets,{s}))
                    for a,es in events.items()}) for s in sorted(score)])
            records.append(row);completed+=1;_write(output/f'fold{fold}_report.json',row)
            print('macro-event counts '+str(row['events']),flush=True);del ledger,p,graph;gc.collect()
        guard();unchanged()
        if eligibility!=eligibility_inputs(root,data.subject_by_session): raise ValueError('eligibility changed during fitting')
        if phase=='policy':
            _write(output/'policy_seal.json',dict(sha256={p.name:_sha(p) for p in output.glob('*.json') if p.name not in ('state.json','policy_seal.json')}))
            result=dict(state='POLICY_FROZEN',completed_splits=completed)
        else:
            result=dict(state='COMPLETE',evidence_kind='EXPLORATORY_MACRO_CLASSIFIER_EVENT_DEVELOPMENT_ONLY',cohort=cohort,
                **event_gate(records,cohort,hypothesis),hypothesis=hypothesis,protected_unchanged=True,window_gate_failed=hypothesis in ('lgbm','ensemble'),
                exploratory_amendment=hypothesis,
                window_fits=window_fits,meta_fits=24 if cohort=='t24' else 40,
                elapsed_seconds=time.time()-plan['started_epoch'])
            _write(output/'summary.json',result)
        _write(output/'state.json',{'state':result['state']});return result
    except BaseException as exc:
        if output.is_dir(): _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_splits=completed))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('policy','score'),default='policy');parser.add_argument('--cohort',choices=('t24','five'),default='t24')
    parser.add_argument('--hypothesis',choices=('lgbm','ensemble','full_weight','full_labels'),default='lgbm')
    args=parser.parse_args()
    with threadpool_limits(limits=8): print(execute(Path(__file__).resolve().parents[3],args.output,args.phase,args.cohort,args.hypothesis),flush=True)
