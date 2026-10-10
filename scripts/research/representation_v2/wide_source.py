"""S13 source-only full-window encoder; no target-data dependencies."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.special import logit
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.representation_v2 import unitfree as u
from scripts.research.temporal_rep.core import ResidualBlock

SCHEMA='rv2-a3-wide-source-v1'
ARCHITECTURE={'channels':64,'dilations':[1,2,4,8,16],'embedding_dim':16,'samples':125}
REFERENCE_COMMIT='29a625e'
SOURCE='outputs/challenger_70/representation_v2/amendment_a3_source'


class WideTokenEncoder(nn.Module):
    receptive_field_samples=129

    def __init__(self):
        super().__init__()
        self.stem=nn.Conv1d(6,64,5,padding=2)
        self.blocks=nn.ModuleList(ResidualBlock(64,d) for d in (1,2,4,8,16))
        self.project=nn.Linear(64,16)

    def forward(self,values,mask):
        if (values.ndim!=3 or values.shape[1:]!=(6,125) or mask.shape!=(len(values),125)
                or mask.dtype!=torch.bool or not torch.isfinite(values.masked_select(mask[:,None,:])).all()):
            raise ValueError('finite observed N x6 x125 values and boolean masks required')
        weight=mask[:,None,:].to(values.dtype)
        x=torch.relu(self.stem(torch.where(mask[:,None,:],values,0.)))*weight
        for block in self.blocks:x=block(x)*weight
        return self.project(x.sum(-1)/weight.sum(-1).clamp_min(1))


def source_gate(metrics,baseline_bce,checks):
    values=[metrics.get(k) for k in ('average_precision','raw_logit_bce','prevalence')]+[baseline_bce]
    if any(v is None or not np.isfinite(v) for v in values):raise ValueError('defined finite source metrics required')
    ap,bce,prevalence,_=values
    if not (0<=ap<=1 and 0<=prevalence<=1 and bce>=0 and baseline_bce>=0):
        raise ValueError('source probability metrics or BCE out of range')
    passed=(ap>=.8440475046760654+.01-1e-12 and bce<=.25704393608246073+.02+1e-12
            and ap>=prevalence+.1-1e-12 and bce<=baseline_bce+1e-12
            and all(checks.get(k) is True for k in ('finite','isolation','optimizer','resource','protected')))
    return {'passed':bool(passed),'status':'S13_SUPPORTS_TARGET_PROBE' if passed else 'S13_NO_SUPPORTED_WIDE_SOURCE'}


def train_bank(bank,train,validation,*,device,steps=5000,batch_size=128,observe=None,budget=None):
    train,validation=tuple(train),tuple(validation)
    groups,y,values=(np.asarray(bank[k]) for k in ('groups','labels','values'))
    if (not train or not validation or len(set(train))!=len(train) or len(set(validation))!=len(validation)
            or set(train)&set(validation) or set(train)|set(validation)!=set(groups.tolist())
            or y.shape!=groups.shape or values.shape!=(len(y),6,125) or not np.isfinite(values).all()
            or not set(y.tolist())<= {0,1} or steps<1 or batch_size<1):
        raise ValueError('finite binary bank and complete disjoint subject partitions required')
    keep=np.isin(groups,train);held=np.isin(groups,validation)
    x,ytrain,g=values[keep],y[keep],groups[keep]
    if set(ytrain.tolist())!={0,1}:raise ValueError('training population requires both classes')
    choices={int(s):{int(c):np.flatnonzero((g==s)&(ytrain==c)) for c in np.unique(ytrain[g==s])} for s in train}
    names=sorted(choices)
    torch.manual_seed(20260926)
    encoder=WideTokenEncoder().to(device);head=nn.Linear(16,1).to(device)
    initial={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()}
    params=list(encoder.parameters())+list(head.parameters())
    optimizer=torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-4)
    rng=np.random.default_rng(20260926);history=[];recent=[];subject_draws=Counter();class_draws=Counter()
    started=time.perf_counter()
    for step in range(1,steps+1):
        if budget is not None:budget(time.perf_counter()-started)
        indices=[]
        for _ in range(batch_size):
            subject=int(rng.choice(names));label=int(rng.choice(list(choices[subject])))
            indices.append(int(rng.choice(choices[subject][label])))
            subject_draws[subject]+=1;class_draws[label]+=1
        batch=torch.as_tensor(x[indices],device=device)
        batch=u.rotate_triads(batch,u.rotation_matrices(len(batch),device))
        mask=torch.ones(len(batch),125,dtype=torch.bool,device=device)
        optimizer.zero_grad(set_to_none=True)
        logits=head(encoder(batch,mask)).squeeze(1)
        loss=F.binary_cross_entropy_with_logits(logits,torch.as_tensor(ytrain[indices],dtype=torch.float32,device=device))
        if not torch.isfinite(loss):raise RuntimeError('nonfinite source loss')
        loss.backward();norm=nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite source gradient')
        optimizer.step();recent.append([float(loss.detach()),float(norm)])
        if step%100==0 or step==steps:
            mean=np.mean(recent,axis=0);recent=[]
            row={'step':step,'loss':float(mean[0]),'gradient_norm':float(mean[1]),'elapsed_seconds':time.perf_counter()-started}
            history.append(row)
            if observe is not None:observe(history)
    if device=='cuda':torch.cuda.synchronize()
    training_seconds=time.perf_counter()-started
    if budget is not None:budget(training_seconds)
    encoder.eval();head.eval();scores=[]
    with torch.inference_mode():
        for left in range(0,int(held.sum()),512):
            batch=torch.as_tensor(values[held][left:left+512],device=device)
            scores.extend(head(encoder(batch,torch.ones(len(batch),125,dtype=torch.bool,device=device))).squeeze(1).cpu().tolist())
    payload={'schema':SCHEMA,'architecture':dict(ARCHITECTURE),
             'encoder':{k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()},
             'head':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},'random_encoder':initial,
             'provenance':{'source_train':list(train),'source_validation':list(validation),'target_subjects':[],
                           'steps':steps,'batch_size':batch_size,'normalization':'per-token-per-triad-demean-rms'}}
    return {'payload':payload,'logits':np.asarray(scores),'history':history,'training_seconds':training_seconds,
            'subject_draws':dict(subject_draws),'class_draws':dict(class_draws)}


def validate_payload(payload):
    try:
        p=payload['provenance'];train=p['source_train'];val=p['source_validation']
        if (payload['schema']!=SCHEMA or payload['architecture']!=ARCHITECTURE or p['target_subjects']!=[]
                or p['normalization']!='per-token-per-triad-demean-rms' or p['steps']<1 or p['batch_size']<1
                or len(set(train))!=len(train) or len(set(val))!=len(val) or not train or not val
                or set(train)&set(val) or set(train)|set(val)!=set(range(34)) or len(train)!=27 or len(val)!=7):
            raise ValueError('wide source contract invalid')
        with torch.random.fork_rng(devices=[]):
            modules={'encoder':WideTokenEncoder(),'random_encoder':WideTokenEncoder(),'head':nn.Linear(16,1)}
        for name,module in modules.items():
            state=payload[name];expected=module.state_dict()
            if (set(state)!=set(expected) or any(not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape
                    or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() for k,v in state.items())):
                raise ValueError('wide source weights invalid')
            module.load_state_dict(state,strict=True)
        return payload
    except (KeyError,TypeError,RuntimeError) as exc:raise ValueError('wide source artifact invalid') from exc


def save_artifact(path,payload):
    validate_payload(payload);path=Path(path)
    if path.exists():raise RuntimeError('checkpoint exists; no overwrite')
    temporary=path.with_suffix('.pt.tmp')
    try:
        torch.save(payload,temporary)
        validate_payload(torch.load(temporary,map_location='cpu',weights_only=True))
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)
    return u._file_sha(path)


def load_artifact(path,expected_sha):
    path=Path(path)
    if u._file_sha(path)!=expected_sha:raise ValueError('wide source SHA mismatch')
    return validate_payload(torch.load(path,map_location='cpu',weights_only=True))


def read_reference(root):
    reference={};hashes={}
    for name in ('plan','population','validation','summary'):
        path=f'{SOURCE}/{name}.json'
        raw=subprocess.check_output(['git','show',f'{REFERENCE_COMMIT}:{path}'],cwd=root)
        if (root/path).read_bytes()!=raw:raise RuntimeError('sealed S0 reference changed')
        reference[name]=json.loads(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    path='outputs/challenger_70/representation_v2/amendment_a3_gesture_morphology/state.json'
    raw=subprocess.check_output(['git','show',f'{REFERENCE_COMMIT}:{path}'],cwd=root)
    if (root/path).read_bytes()!=raw or json.loads(raw)!={'status':'S12_NO_SUPPORTED_GESTURE_MORPHOLOGY','outer_calls':0}:
        raise RuntimeError('sealed failed S12 closure required')
    if (reference['summary']['status']!='S0_SUPPORTS_TARGET_PROBE' or reference['summary']['passed'] is not True
            or reference['summary']['metrics']['average_precision']!=.8440475046760654
            or reference['summary']['metrics']['raw_logit_bce']!=.25704393608246073):
        raise RuntimeError('S0 baseline differs from registration')
    reference['hashes']=hashes
    reference['helper_sha256']={}
    for name in ('scripts/research/representation_v2/unitfree.py','scripts/research/representation_v2/fd_source.py'):
        raw=subprocess.check_output(['git','show',f'{REFERENCE_COMMIT}:{name}'],cwd=root)
        if (root/name).read_bytes()!=raw:raise RuntimeError('sealed source preprocessing code changed')
        reference['helper_sha256'][name]=hashlib.sha256(raw).hexdigest()
    return reference


def validate_reference_bank(bank,train,validation,reference):
    p=reference['plan'];population={'counts':bank['counts'],'valid_tokens':len(bank['labels']),'persistent_token_cache_bytes':0}
    if (list(train)!=p['source_train'] or list(validation)!=p['source_validation']
            or json.loads(json.dumps(population))!=json.loads(json.dumps(reference['population']))):
        raise ValueError('S0 split or population drift')
    held=np.isin(bank['groups'],validation)
    rows=[(int(g),int(s),int(y)) for g,s,y in zip(bank['groups'][held],bank['starts'][held],bank['labels'][held])]
    expected=[(r['participant'],r['start_sample'],r['label']) for r in reference['validation']]
    if rows!=expected:raise ValueError('S0 validation token identity drift')


def run(root):
    root=Path(root);out=root/'outputs/challenger_70/representation_v2/amendment_a3_wide_source'
    if out.exists():raise RuntimeError('S13 attempt exists; no overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit registration and code before source training')
    started=time.perf_counter();reference=read_reference(root)
    snapshot=u.protected_snapshot(root)
    adapter=u.FDIAdapter(root/'FDdatasets/doi-10.48804-CN8VBB/v1.0')
    train,validation=adapter.participant_split(seed=20260926)
    files={}
    for name,expected in reference['plan']['source_files'].items():
        path=root/name;actual={'bytes':path.stat().st_size,'sha256':u._file_sha(path)}
        if actual!=expected:raise RuntimeError('source file fingerprint drift')
        files[name]=actual
    if 2*sum(r['bytes'] for r in files.values())>.8*u.available_memory_bytes():raise RuntimeError('source decode RAM reserve insufficient')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    free_cuda=torch.cuda.mem_get_info()[0] if device=='cuda' else None
    if shutil.disk_usage(root).free<20*1024**3 or (free_cuda is not None and free_cuda<1280*1024**2):
        raise RuntimeError('resource reserve insufficient')
    torch.set_num_threads(4)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    plan={'id':'RV2-A3-S13-001','code_sha':sha,'steps':5000,'batch_size':128,'seed':20260926,'device':device,
          'architecture':dict(ARCHITECTURE),'encoder_parameters':126544,'head_parameters':17,'receptive_field_samples':129,
          'source_train':list(train),'source_validation':list(validation),'source_files':files,
          'reference_commit':REFERENCE_COMMIT,'reference_sha256':reference['hashes'],
          'source_helper_sha256':reference['helper_sha256'],
          'source_checkpoint_sha256':reference['summary']['checkpoint_sha256'],
          'spec_sha256':u._file_sha(root/'docs/challenge_070_amendment_rv2_a3_receptive_field.md'),
          'max_wall_seconds':14400,'max_training_seconds':7200,'outer_calls':0,'target_signal_read':False,
          'protected_digest':hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()}
    u.write_json(out/'plan.json',plan)
    u.write_json(out/'state.json',{'status':'S13_IN_PROGRESS','outer_calls':0})
    try:
        print('S13 loading FD-I right wrist only; target signal access disabled',flush=True)
        bank=u.load_source_bank(adapter)
        validate_reference_bank(bank,train,validation,reference)
        held=np.isin(bank['groups'],validation);keep=np.isin(bank['groups'],train)
        # Replay the old frozen model on identical source tokens before the new training.
        oldpath=root/SOURCE/'source_final.pt'
        if u._file_sha(oldpath)!=reference['summary']['checkpoint_sha256']:raise RuntimeError('S0 checkpoint drift')
        old=torch.load(oldpath,map_location='cpu',weights_only=True)
        oldencoder=u.MotionBranch().token_encoder.to(device).eval();oldencoder.load_state_dict(old['encoder'],strict=True)
        oldhead=nn.Linear(16,1).to(device).eval();oldhead.load_state_dict(old['head'],strict=True)
        old_logits=[];xv=bank['values'][held]
        with torch.inference_mode():
            for left in range(0,len(xv),512):
                x=torch.as_tensor(xv[left:left+512],device=device)
                old_logits.extend(oldhead(oldencoder(x,torch.ones(len(x),125,dtype=torch.bool,device=device))).squeeze(1).cpu().tolist())
        saved=np.asarray([r['logit'] for r in reference['validation']])
        np.testing.assert_allclose(old_logits,saved,atol=1e-5,rtol=1e-5)
        u.write_json(out/'reference_replay.json',{'source_checkpoint_sha256':reference['summary']['checkpoint_sha256'],
            'validation_tokens':len(saved),'max_absolute_logit_difference':float(np.max(np.abs(saved-old_logits))),
            'rtol':1e-5,'atol':1e-5,'population_and_identity_equal':True})
        del old,oldencoder,oldhead,xv
        population={'counts':bank['counts'],'valid_tokens':len(bank['labels']),'persistent_token_cache_bytes':0}
        u.write_json(out/'population.json',population)
        priors=u.training_priors(bank['counts'],train,bank['groups'][keep],bank['labels'][keep])

        def resource():
            size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            return (time.perf_counter()-started<=14400 and size<=16*1024**2 and shutil.disk_usage(root).free>=20*1024**3
                    and (free_cuda is None or torch.cuda.max_memory_allocated()<=.8*free_cuda))

        def budget(seconds):
            if seconds>7200 or time.perf_counter()-started>14400:raise TimeoutError('S13 training/wall budget exceeded')

        def observe(history):
            u.write_json(out/'training_history.json',history)
            if not resource():raise RuntimeError('S13 resource budget exceeded')
            if history[-1]['step']%1000==0:print(f"S13 step={history[-1]['step']} train_loss={history[-1]['loss']:.4f}",flush=True)

        if not resource():raise RuntimeError('S13 preparation resource budget exceeded')
        result=train_bank(bank,train,validation,device=device,observe=observe,budget=budget)
        y=bank['labels'][held];z=result['logits'];metric=u.logit_diagnostics(y,z)
        baseline=u.logit_diagnostics(y,np.full(len(y),logit(priors['sampler'])))
        rows=[{'participant':int(g),'start_sample':int(s),'label':int(c),'logit':float(p)}
              for g,s,c,p in zip(bank['groups'][held],bank['starts'][held],y,z)]
        subjects=[]
        for g in validation:
            mask=bank['groups'][held]==g
            subjects.append({'participant':int(g),'metrics':u.logit_diagnostics(y[mask],z[mask]),
                             's0_metrics':u.logit_diagnostics(y[mask],saved[mask])})
        u.write_json(out/'validation.json',rows);u.write_json(out/'subject_metrics.json',subjects)
        payload=result['payload'];payload['provenance'].update({'code_sha':sha,'plan_sha256':u._file_sha(out/'plan.json')})
        checkpoint_sha=save_artifact(out/'source_final.pt',payload)
        unchanged=u.protected_snapshot(root)==snapshot
        checks={'finite':bool(np.isfinite(z).all()),'isolation':not bool(set(train)&set(validation)),
                'optimizer':result['history'][-1]['step']==5000,
                'resource':bool(resource() and result['training_seconds']<=7200),'protected':unchanged}
        if not unchanged or not checks['resource']:raise RuntimeError('final resource/protected check failed')
        report={**source_gate(metric,baseline['raw_logit_bce'],checks),'metrics':metric,'sampler_baseline':baseline,
            'training_priors':priors,'checks':checks,'checkpoint_sha256':checkpoint_sha,'optimizer_steps':5000,
            'training_seconds':result['training_seconds'],'wall_seconds':time.perf_counter()-started,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'train_token_count':int(keep.sum()),'validation_token_count':int(held.sum()),
            'subject_draws':result['subject_draws'],'class_draws':result['class_draws'],
            'persistent_token_cache_bytes':0,'target_signal_read':False,'outer_calls':0,'protected_unchanged':True}
        u.write_json(out/'summary.json',report);u.write_json(out/'state.json',{'status':report['status'],'outer_calls':0})
        print(json.dumps(report,sort_keys=True),flush=True)
    except Exception as exc:
        u.write_json(out/'state.json',{'status':'S13_INCOMPLETE','error':str(exc),'outer_calls':0,
                                     'wall_seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':
    with u.threadpool_limits(limits=4):run(Path(__file__).resolve().parents[3])
