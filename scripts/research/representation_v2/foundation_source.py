"""S14 frozen LIMU-BERT-X source-only unit-free transfer research.

Encoder equations adapted from WANDS-HKUST/LIMU-BERT_Experience (MIT),
Copyright (c) 2025 WANDS HKUST. Full upstream notice retained with experiment.
No remote code is imported or executed. This is NOT physical-unit calibration.
"""
from pathlib import Path
import hashlib
import math
import json
import shutil
import subprocess
import time
import urllib.request

import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
from sklearn.decomposition import PCA

from scripts.research.representation_v2 import unitfree as u
from scripts.research.representation_v2.low_capacity import fit_readout,predict_readout,readout_payload
from scripts.research.representation_v2.wide_source import read_reference,validate_reference_bank,SOURCE

UPSTREAM_COMMIT='bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb'
WEIGHT_SHA='65697e7017020f36429a221382253f1fc370fdfd493cfb71667f95846c18516c'
WEIGHT_BYTES=226783
OUTPUT='outputs/challenger_70/representation_v2/amendment_a4_foundation_source'


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__();self.gamma=nn.Parameter(torch.ones(72));self.beta=nn.Parameter(torch.zeros(72))

    def forward(self,x):
        centered=x-x.mean(-1,keepdim=True)
        return self.gamma*centered/torch.sqrt(centered.square().mean(-1,keepdim=True)+1e-12)+self.beta


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__();self.lin=nn.Linear(6,72);self.pos_embed=nn.Embedding(20,72);self.norm=LayerNorm()

    def forward(self,x):
        return self.norm(self.norm(self.lin(x))+self.pos_embed(torch.arange(20,device=x.device)))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q=nn.Linear(72,72);self.proj_k=nn.Linear(72,72);self.proj_v=nn.Linear(72,72)

    def forward(self,x):
        q,k,v=[layer(x).reshape(len(x),20,4,18).transpose(1,2) for layer in
               (self.proj_q,self.proj_k,self.proj_v)]
        return ((q@k.transpose(-2,-1)/math.sqrt(18)).softmax(-1)@v).transpose(1,2).reshape(len(x),20,72)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__();self.fc1=nn.Linear(72,144);self.fc2=nn.Linear(144,72)

    def forward(self,x):
        h=self.fc1(x);return self.fc2(h*.5*(1+torch.erf(h/math.sqrt(2))))


class LimuEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.embed=Embeddings();self.attn=Attention();self.proj=nn.Linear(72,72)
        self.norm1=LayerNorm();self.pwff=FeedForward();self.norm2=LayerNorm()

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(20,6) or not torch.isfinite(x).all():
            raise ValueError('LIMU input requires finite N x20 x6')
        h=self.embed(x)
        for _ in range(4):
            h=self.attn(h)
            h=self.norm1(h+self.proj(h))
            h=self.norm2(h+self.pwff(h))
        return h


def head_template():
    """Known upstream reconstruction tensors; validated, never used for classification."""
    return {k:torch.zeros(shape) for k,shape in {
        'fc.weight':(72,72),'fc.bias':(72,),'linear.weight':(72,72),'linear.bias':(72,),
        'norm.gamma':(72,),'norm.beta':(72,),'decoder.weight':(6,72),'decoder.bias':(6,)}.items()}


def load_pretrained(path,*,expected_sha=WEIGHT_SHA):
    path=Path(path)
    if path.stat().st_size>1024**2 or u._file_sha(path)!=expected_sha:
        raise ValueError('external checkpoint size/SHA mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]):model=LimuEncoder()
    expected={'transformer.'+k:v for k,v in model.state_dict().items()};expected.update(head_template())
    if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError('external state keys mismatch')
    for key,template in expected.items():
        value=state[key]
        if (not isinstance(value,torch.Tensor) or value.shape!=template.shape or value.dtype!=torch.float32
                or not torch.isfinite(value).all()):raise ValueError('invalid external tensor')
    model.load_state_dict({k:state['transformer.'+k] for k in model.state_dict()},strict=True)
    return model.eval().requires_grad_(False)


def bridge_tokens(values):
    x=np.asarray(values)
    if x.ndim!=3 or x.shape[1:]!=(6,125) or not np.isfinite(x).all():
        raise ValueError('finite unit-free N x6 x125 required')
    return resample_poly(x,4,5,axis=2,padtype='line').transpose(0,2,1).reshape(len(x),5,20,6).astype(np.float32)


def foundation_gate(metrics,checks):
    passed=all(checks.get(k) is True for k in ('finite','isolation','frozen','resource','protected'))
    try:
        r,d,f=[metrics[k] for k in ('r','d','f')]
        for row in (r,d,f):
            ap,bce,prior=[row[k] for k in ('average_precision','raw_logit_bce','prevalence')]
            passed &= bool(np.isfinite([ap,bce,prior]).all() and 0<=ap<=1 and bce>=0 and 0<=prior<=1)
        passed &= (f['average_precision']>=max(r['average_precision']+.01,d['average_precision']+.01,
                      .8440475046760654+.01,f['prevalence']+.1)-1e-12
                   and f['raw_logit_bce']<=min(r['raw_logit_bce']+.02,d['raw_logit_bce']+.02,
                                             .25704393608246073+.02)+1e-12)
    except (KeyError,TypeError,ValueError):passed=False
    return {'passed':bool(passed),'status':'S14_SUPPORTS_TARGET_PROBE' if passed else
            'S14_NO_SUPPORTED_FOUNDATION_TRANSFER'}


def fit_source_probe(embeddings,labels,groups,train,validation,*,project):
    x,y,g=map(np.asarray,(embeddings,labels,groups));train,validation=tuple(train),tuple(validation)
    expected=(len(y),5,72) if project else (len(y),16)
    if (x.shape!=expected or y.ndim!=1 or g.shape!=y.shape or not np.isfinite(x).all()
            or not set(y.tolist())<= {0,1} or not train or not validation or set(train)&set(validation)
            or len(set(train))!=len(train) or len(set(validation))!=len(validation)
            or set(train)|set(validation)!=set(g.tolist())):
        raise ValueError('finite aligned bank and complete disjoint subject partitions required')
    keep=np.isin(g,train);held=np.isin(g,validation);projection=None
    if project:
        pca=PCA(n_components=32,svd_solver='full',whiten=False).fit(x[keep].reshape(-1,72))
        z=pca.transform(x.reshape(-1,72)).reshape(len(x),5,32)
        features=np.concatenate([z.mean(1),z.std(1),z.max(1)],axis=1)
        projection={'mean':pca.mean_.tolist(),'components':pca.components_.tolist(),
                    'explained_variance':pca.explained_variance_.tolist(),'fit_rows':int(keep.sum()*5)}
    else:features=x
    fitted=fit_readout(features[keep],y[keep],g[keep])
    return {'payload':{'schema':'rv2-a4-source-readout-v1','train_subjects':list(train),
                      'excluded_validation_subjects':list(validation),'feature_width':features.shape[1],
                      'pca':projection,'readout':readout_payload(fitted)},
            'logits':predict_readout(fitted,features[held])}


def fetch_public(relative,*,max_bytes=1024**2,expected_sha=None,opener=None):
    allowed={'LICENSE','README.md','models.py','config.py','config/limu_bert.json',
             'dataset/data_config.json','weights/limu_bert_x.pt'}
    if relative not in allowed:raise ValueError('unregistered upstream file')
    url='https://raw.githubusercontent.com/WANDS-HKUST/LIMU-BERT_Experience/'+UPSTREAM_COMMIT+'/'+relative
    with (opener or urllib.request.urlopen)(url,timeout=30) as response:data=response.read(max_bytes+1)
    if len(data)>max_bytes or (expected_sha is not None and hashlib.sha256(data).hexdigest()!=expected_sha):
        raise ValueError('upstream download limit/SHA mismatch')
    return data


def encode_limu(model,values,*,device,chunk=512,budget=None):
    model=model.to(device).eval().requires_grad_(False);blocks=bridge_tokens(values)
    flat=blocks.reshape(-1,20,6);rows=[]
    with torch.inference_mode():
        for left in range(0,len(flat),chunk):
            if budget:budget()
            z=model(torch.as_tensor(flat[left:left+chunk],device=device)).mean(1).cpu().numpy()
            rows.append(z)
    if budget:budget()
    return np.concatenate(rows).reshape(len(values),5,72)


def check_budget(output,*,wall_seconds,peak_cuda,free_cuda):
    if wall_seconds>7200:raise TimeoutError('S14 wall budget exceeded')
    if shutil.disk_usage(output).free<20*1024**3:raise RuntimeError('S14 disk reserve below20GiB')
    if sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())>32*1024**2:
        raise RuntimeError('S14 output budget exceeded')
    if free_cuda is not None and peak_cuda>.8*free_cuda:raise RuntimeError('S14 VRAM budget exceeded')


def check_preflight_resources(output,*,source_bytes,available_ram,free_cuda):
    try:
        if 2*source_bytes>.8*available_ram:raise RuntimeError('source decode RAM reserve insufficient')
        if free_cuda is not None and free_cuda<1280*1024**2:raise RuntimeError('S14 GPU reserve insufficient')
        if shutil.disk_usage(Path(output).parent).free<20*1024**3:
            raise RuntimeError('S14 disk reserve insufficient')
    except Exception as exc:
        u.write_json(Path(output)/'state.json',{'status':'S14_INCOMPLETE','phase':'preflight',
                                             'error':str(exc),'outer_calls':0})
        raise


def _state_digest(model):
    digest=hashlib.sha256()
    for k,v in model.state_dict().items():
        digest.update(k.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(root):
    root=Path(root);out=root/OUTPUT
    if out.exists():raise RuntimeError('S14 attempt exists; no overwrite/retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit S14 registration and implementation first')
    started=time.perf_counter();reference=read_reference(root)
    prior=root/'outputs/challenger_70/representation_v2/amendment_a3_wide_source/state.json'
    if (prior.read_bytes()!=subprocess.check_output(['git','show',
            'c1246fb:outputs/challenger_70/representation_v2/amendment_a3_wide_source/state.json'],cwd=root)
            or json.loads(prior.read_text())!={'outer_calls':0,'status':'S13_NO_SUPPORTED_WIDE_SOURCE'}):
        raise RuntimeError('sealed S13 closure required')
    snapshot=u.protected_snapshot(root)
    sourcefiles={}
    for name,expected in reference['plan']['source_files'].items():
        path=root/name;actual={'bytes':path.stat().st_size,'sha256':u._file_sha(path)}
        if actual!=expected:raise RuntimeError('source file fingerprint drift')
        sourcefiles[name]=actual
    device='cuda' if torch.cuda.is_available() else 'cpu'
    free_cuda=torch.cuda.mem_get_info()[0] if device=='cuda' else None
    check_preflight_resources(out,source_bytes=sum(v['bytes'] for v in sourcefiles.values()),
                              available_ram=u.available_memory_bytes(),free_cuda=free_cuda)
    torch.set_num_threads(4)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    plan={'id':'RV2-A4-S14-001','code_sha':code_sha,'seed':20260926,'device':device,
          'upstream_commit':UPSTREAM_COMMIT,'checkpoint_sha256':WEIGHT_SHA,'torch_version':str(torch.__version__),
          'source_files':sourcefiles,'reference_sha256':reference['hashes'],
          'source_helper_sha256':reference['helper_sha256'],'source_train':reference['plan']['source_train'],
          'source_validation':reference['plan']['source_validation'],
          'spec_sha256':u._file_sha(root/'docs/challenge_070_amendment_rv2_a4_foundation.md'),
          'normalization':'unit-free per-token per-triad demean RMS; not official physical input',
          'sample_hz':20,'subwindow_samples':20,'subwindows_per_token':5,'projection':32,'pooled_width':96,
          'readout_c':.1,'readout_fits':3,'neural_optimizer_steps':0,'target_signal_read':False,'outer_calls':0,
          'max_wall_seconds':7200,'max_output_bytes':32*1024**2,
          'protected_digest':hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()}
    u.write_json(out/'plan.json',plan);u.write_json(out/'state.json',{'status':'S14_IN_PROGRESS','outer_calls':0})
    def budget():
        check_budget(out,wall_seconds=time.perf_counter()-started,
                     peak_cuda=torch.cuda.max_memory_allocated() if device=='cuda' else 0,free_cuda=free_cuda)
    try:
        upstream={}
        for name in ('LICENSE','README.md','models.py','config.py','config/limu_bert.json',
                     'dataset/data_config.json','weights/limu_bert_x.pt'):
            data=fetch_public(name,expected_sha=WEIGHT_SHA if name.startswith('weights/') else None)
            if name.startswith('weights/') and len(data)!=WEIGHT_BYTES:raise ValueError('weight bytes changed')
            path=out/'upstream'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            upstream[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};budget()
        u.write_json(out/'upstream_manifest.json',upstream)
        config=json.loads((out/'upstream/config/limu_bert.json').read_text())['base_v4']
        if config!={'feature_num':6,'hidden':72,'hidden_ff':144,'n_layers':4,'n_heads':4,'seq_len':20}:
            raise RuntimeError('upstream architecture contract mismatch')
        pretrained=load_pretrained(out/'upstream/weights/limu_bert_x.pt')
        torch.manual_seed(20260926);random=LimuEncoder().eval().requires_grad_(False)
        initial={name:_state_digest(model) for name,model in [('r',random),('f',pretrained)]}
        adapter=u.FDIAdapter(root/'FDdatasets/doi-10.48804-CN8VBB/v1.0')
        train,val=adapter.participant_split(seed=20260926)
        print('S14 source-only FD-I right wrist; frozen random/pretrained LIMU and frozen FD readouts',flush=True)
        bank=u.load_source_bank(adapter);validate_reference_bank(bank,train,val,reference);budget()
        keep=np.isin(bank['groups'],train);held=np.isin(bank['groups'],val)
        u.write_json(out/'population.json',{'counts':bank['counts'],'train_tokens':int(keep.sum()),
                     'validation_tokens':int(held.sum()),'persistent_token_cache_bytes':0})
        oldpath=root/SOURCE/'source_final.pt'
        if u._file_sha(oldpath)!=reference['summary']['checkpoint_sha256']:raise RuntimeError('S0 checkpoint drift')
        old=torch.load(oldpath,map_location='cpu',weights_only=True)
        fd=u.MotionBranch().token_encoder.to(device).eval().requires_grad_(False)
        fd.load_state_dict(old['encoder'],strict=True);initial['d']=_state_digest(fd)
        head=nn.Linear(16,1).to(device);head.load_state_dict(old['head'],strict=True)
        fdrows=[]
        with torch.inference_mode():
            for left in range(0,len(bank['values']),512):
                budget();x=torch.as_tensor(bank['values'][left:left+512],device=device)
                fdrows.append(fd(x,torch.ones(len(x),125,dtype=torch.bool,device=device)).cpu().numpy())
        fdrows=np.concatenate(fdrows)
        with torch.inference_mode():replay=head(torch.as_tensor(fdrows[held],device=device)).squeeze(1).cpu().numpy()
        expected=np.asarray([r['logit'] for r in reference['validation']])
        np.testing.assert_allclose(replay,expected,atol=1e-5,rtol=1e-5)
        u.write_json(out/'reference_replay.json',{'max_absolute_logit_difference':float(np.max(np.abs(replay-expected))),
                     'population_identity_equal':True,'source_checkpoint_sha256':reference['summary']['checkpoint_sha256']})
        results={};timings={}
        for arm,encoder in [('d',fd),('r',random),('f',pretrained)]:
            begin=time.perf_counter()
            features=fdrows if arm=='d' else encode_limu(encoder,bank['values'],device=device,budget=budget)
            result=fit_source_probe(features,bank['labels'],bank['groups'],train,val,project=arm!='d')
            result['payload'].update({'encoder_state_sha256':initial[arm],'code_sha':code_sha,
                                      'plan_sha256':u._file_sha(out/'plan.json')})
            results[arm]=result;timings[arm]=time.perf_counter()-begin
            u.write_json(out/(arm+'_readout.json'),result['payload'])
            print(f'S14 arm {arm}: fixed encoding/readout complete; no validation-driven changes',flush=True)
            del features;budget()
        metrics={arm:u.logit_diagnostics(bank['labels'][held],r['logits']) for arm,r in results.items()}
        rows=[{'participant':int(g),'start_sample':int(s),'label':int(y),
               **{arm+'_logit':float(r['logits'][i]) for arm,r in results.items()}}
              for i,(g,s,y) in enumerate(zip(bank['groups'][held],bank['starts'][held],bank['labels'][held]))]
        u.write_json(out/'validation.json',rows)
        subject_metrics=[]
        for pid in val:
            local=bank['groups'][held]==pid
            subject_metrics.append({'participant':int(pid),'metrics':{arm:u.logit_diagnostics(
                bank['labels'][held][local],r['logits'][local]) for arm,r in results.items()}})
        u.write_json(out/'subject_metrics.json',subject_metrics)
        unchanged=u.protected_snapshot(root)==snapshot;budget()
        checks={'finite':all(np.isfinite(r['logits']).all() for r in results.values()),
                'isolation':not bool(set(train)&set(val)),
                'frozen':all(initial[a]==_state_digest(e) for a,e in [('d',fd),('r',random),('f',pretrained)]),
                'resource':True,'protected':unchanged}
        checks={k:bool(v) for k,v in checks.items()}
        if not all(checks.values()):raise RuntimeError('S14 final integrity checks failed')
        summary={**foundation_gate(metrics,checks),'metrics':metrics,'checks':checks,
                 'encoder_parameters':sum(p.numel() for p in pretrained.parameters()),'state_sha256':initial,
                 'timing_seconds':timings,'wall_seconds':time.perf_counter()-started,
                 'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
                 'readout_fits':3,'neural_optimizer_steps':0,'persistent_token_cache_bytes':0,
                 'target_signal_read':False,'outer_calls':0,'protected_unchanged':unchanged}
        u.write_json(out/'summary.json',summary);budget()
        u.write_json(out/'state.json',{'status':summary['status'],'outer_calls':0})
        print(json.dumps(summary,sort_keys=True),flush=True)
    except Exception as exc:
        u.write_json(out/'state.json',{'status':'S14_INCOMPLETE','error':str(exc),'outer_calls':0})
        raise


if __name__=='__main__':
    with u.threadpool_limits(limits=4):run(Path(__file__).resolve().parents[3])
