"""Read-only policy-OOF score diagnostics; never select or evaluate health policies."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from scripts.research.representation_v2.explore import write_json
from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
from src.pipeline.event_stack import CandidateEvent,EventRef,MultiScaleCandidate,compute_event_metrics
from src.pipeline.candidate_control import CandidateAdmissionConfig,admit_candidates


FD_TRAIN={f'FD-R-{i}' for i in (0,1,2,4,5,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,24,26,29,30,31,32,33)}


def allowed_input(name):
    if not re.fullmatch(r'fold[0-2]_(policy[0-2]_scores|policies)\.json',name):
        raise ValueError('only registered policy OOF inputs allowed')
    return name


def profile(labels,scores):
    y=np.asarray(labels); p=np.asarray(scores,dtype=float)
    if y.ndim!=1 or y.shape!=p.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('binary aligned labels and finite probabilities required')
    if not len(y):return None
    bins=[]; ece=0.
    index=np.minimum((p*10).astype(int),9)
    for k in range(10):
        keep=index==k; n=int(keep.sum())
        mean=float(p[keep].mean()) if n else None; prevalence=float(y[keep].mean()) if n else None
        bins.append({'lower':k/10,'upper':(k+1)/10,'n':n,'mean_score':mean,'prevalence':prevalence})
        if n:ece+=n/len(y)*abs(mean-prevalence)
    return {'n':len(y),'positives':int(y.sum()),'prevalence':float(y.mean()),'mean_score':float(p.mean()),
            'q10_q50_q90':np.quantile(p,[.1,.5,.9]).tolist(),'brier':float(np.mean((p-y)**2)),
            'ece':float(ece),'bins':bins}


def validate_ledger(d):
    train,score,excluded=(set(d[k]) for k in ('train','score','excluded'))
    if train&(score|excluded) or score&excluded:raise ValueError('overlapping subject partitions')
    graph=d['graph']
    assert_no_heldout_ancestors('score',score|excluded,graph)
    if any(set(n['fit_subjects'])-(train|FD_TRAIN) for n in graph.values()):raise ValueError('unexpected fit ancestor')
    y=np.asarray(d['labels']); valid=np.asarray(d['motion_valid'],dtype=bool); groups=np.asarray(d['groups'])
    n=len(d['candidates'])
    if any(v.shape!=(n,) for v in (y,valid,groups)) or set(groups)-score:raise ValueError('candidate alignment invalid')
    for section in ('scores','component_probabilities'):
        for p in d[section].values():profile(y,p)
    base=np.asarray(d['component_probabilities']['base'])
    if not np.allclose(base,d['scores']['b'],atol=1e-12,rtol=0):raise ValueError('base identity changed')
    for arm in ('r','f'):
        p=np.asarray(d['component_probabilities'][arm])
        if not np.allclose(.75*base+.25*p,d['scores'][arm],atol=1e-12,rtol=0):raise ValueError('fusion identity changed')
        if not np.allclose(p[~valid],d['readouts'][arm]['prior'],atol=1e-12,rtol=0):raise ValueError('invalid fallback changed')


def restore_candidate(d):
    def part(v):
        if v is None:return None
        values=dict(v); values['event']=EventRef(**values['event'])
        return CandidateEvent(**values)
    return MultiScaleCandidate(EventRef(**d['event']),part(d['macro']),part(d['micro']))


def sensitivity(candidates,scores,groups,truths,policy):
    values=np.asarray(scores)
    admitted=admit_candidates(candidates,values,groups,CandidateAdmissionConfig(**policy['admission']))
    rows=[]
    for offset in (-.05,-.025,0.,.025,.05):
        threshold=policy['threshold']+offset
        predictions=[candidates[i].event for i in admitted if values[i]>=threshold]
        m=compute_event_metrics(predictions,truths)
        rows.append({'offset':offset,'threshold':threshold,'tp':m.n_tp,'fp':m.n_pred-m.n_tp,
                     'fn':m.n_true-m.n_tp,'truth_count':m.n_true,'admitted_count':len(admitted)})
    return rows


def describe(d,policies):
    y=np.asarray(d['labels']); valid=np.asarray(d['motion_valid'],dtype=bool); groups=np.asarray(d['groups'])
    channels={**d['scores'],**{'component_'+a:p for a,p in d['component_probabilities'].items()}}
    masks={'all':np.ones(len(y),bool),'valid':valid,'invalid':~valid,'positive':y==1,'negative':y==0,
           'invalid_positive':(~valid)&(y==1),'invalid_negative':(~valid)&(y==0)}
    result={'candidate_count':len(y),'valid_count':int(valid.sum()),'profiles':{},'subjects':[],
            'near_frozen_threshold':{},'invalid_score_shift':{}}
    for name,p in channels.items():
        result['profiles'][name]={key:profile(y[keep],np.asarray(p)[keep]) for key,keep in masks.items()}
    for subject in sorted(set(groups)):
        keep=groups==subject
        result['subjects'].append({'subject':str(subject),'valid_count':int(valid[keep].sum()),
            'scores':{a:profile(y[keep],np.asarray(d['scores'][a])[keep]) for a in ('b','r','f')}})
    for a in ('b','r','f'):
        close=np.abs(np.asarray(d['scores'][a])-policies[a]['threshold'])<=.025
        result['near_frozen_threshold'][a]={key:int((close&keep).sum()) for key,keep in masks.items()}
    for a in ('r','f'):
        delta=np.asarray(d['scores'][a])-np.asarray(d['scores']['b'])
        result['invalid_score_shift'][a]={key:{'n':int(keep.sum()),'mean':float(delta[keep].mean()) if keep.any() else None,
                  'q10_q50_q90':np.quantile(delta[keep],[.1,.5,.9]).tolist() if keep.any() else None}
                  for key,keep in masks.items() if key.startswith('invalid')}
    return result


def run(root):
    root=Path(root); source=root/'outputs/challenger_70/representation_v2/amendment_a3_events'
    output=root/'outputs/challenger_70/representation_v2/amendment_a3_stability'
    if output.exists():raise RuntimeError('audit exists; no overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise RuntimeError('commit audit before run')
    started=time.perf_counter(); inputs={}; files={}
    for k in range(3):
        for name in [f'fold{k}_policies.json',*[f'fold{k}_policy{j}_scores.json' for j in range(3)]]:
            path=source/allowed_input(name); raw=path.read_bytes()
            committed=subprocess.check_output(['git','show',f'HEAD:{path.relative_to(root).as_posix()}'],cwd=root)
            if raw!=committed:raise ValueError('input differs from committed evidence')
            inputs[name]=hashlib.sha256(raw).hexdigest(); files[name]=json.loads(raw)
    plan={'id':'RV2-A3-S3-001','diagnostic_only':True,'health_data_read':False,'outer_calls':0,'training_calls':0,
          'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'spec_sha256':hashlib.sha256((root/'docs/challenge_070_amendment_rv2_a3_stability.md').read_bytes()).hexdigest(),
          'input_sha256':inputs}
    write_json(output/'plan.json',plan)
    for k in range(3):
        policy=files[f'fold{k}_policies.json']; policies=policy['policies']; ledgers=[]; fold_profiles=[]
        for j in range(3):
            d=files[f'fold{k}_policy{j}_scores.json']; validate_ledger(d); ledgers.append(d)
            fold_profiles.append({'policy_fold':j,'train_subjects':d['train'],'score_subjects':d['score'],
                                  'prior':{a:d['readouts'][a]['prior'] for a in ('r','f')},**describe(d,policies)})
        combined={key:[v for d in ledgers for v in d[key]] for key in ('labels','groups','motion_valid','candidates')}
        if sum(len(set(d['score'])) for d in ledgers)!=len(set(s for d in ledgers for s in d['score'])):
            raise ValueError('policy validation partitions overlap')
        for key in ('scores','component_probabilities'):
            combined[key]={a:[v for d in ledgers for v in d[key][a]] for a in ledgers[0][key]}
        cs=[restore_candidate(c) for c in combined['candidates']]; truths=[EventRef(**t) for t in policy['policy_truths']]
        curves={a:sensitivity(cs,combined['scores'][a],combined['groups'],truths,policies[a]) for a in policies}
        for a,curve in curves.items():
            original=policies[a]['selection_metrics']; center=curve[2]
            if (center['tp'],center['tp']+center['fp'],center['truth_count'])!=(original['n_tp'],original['n_pred'],original['n_true']):
                raise ValueError('frozen policy selection count replay failed')
        write_json(output/f'fold{k}_audit.json',{'wrapper':k,'diagnostic_only':True,'policy_folds':fold_profiles,
                   'combined':describe(combined,policies),'frozen_policies':{a:{'admission':p['admission'],'threshold':p['threshold']} for a,p in policies.items()},
                   'policy_oof_sensitivity':curves})
    if any(hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest for name,digest in inputs.items()):
        raise RuntimeError('audit source evidence changed')
    size=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size>10*1024**2 or time.perf_counter()-started>900:raise RuntimeError('audit budget exceeded')
    summary={'status':'S3_DIAGNOSTIC_COMPLETE','health_data_read':False,'training_calls':0,'outer_calls':0,
             'input_files':len(inputs),'input_unchanged':True,'frozen_policy_count_replay':True,
             'wall_seconds':time.perf_counter()-started,'artifact_bytes_before_summary':size}
    write_json(output/'summary.json',summary); print(json.dumps(summary),flush=True)


if __name__=='__main__':run(Path(__file__).resolve().parents[3])
