from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.research.representation_v2 import wide_source as w
from scripts.research.representation_v2.unitfree import rotation_matrices, rotate_triads


def bank():
    groups=np.repeat(np.arange(34),2)
    return {'values':np.random.default_rng(4).normal(size=(68,6,125)).astype('float32'),
            'labels':np.tile([0,1],34),'groups':groups,'starts':np.tile([0,320],34),
            'counts':[{'participant':p,'available':{0:1,1:1},'selected':{0:1,1:1},'valid':{0:1,1:1}} for p in range(34)]}


def test_full_window_encoder_has_real_long_range_gradient_and_mask_isolation():
    torch.set_num_threads(2)
    model=w.WideTokenEncoder()
    assert sum(p.numel() for p in model.parameters())==126544
    assert model.receptive_field_samples==129
    for p in model.parameters():torch.nn.init.constant_(p,.01)
    x=torch.ones(1,6,125,requires_grad=True);mask=torch.ones(1,125,dtype=torch.bool)
    captured=[]
    hook=model.blocks[-1].register_forward_hook(lambda m,i,o:captured.append(o))
    z=model(x,mask);hook.remove()
    captured[0][0,0,62].backward()
    assert x.grad[0,:,0].abs().sum()>0 and x.grad[0,:,124].abs().sum()>0
    assert z.shape==(1,16) and torch.isfinite(z).all()
    mask[:,20:40]=False;dirty=x.detach().clone();dirty[:,:,20:40]=float('nan')
    torch.testing.assert_close(model(x.detach(),mask),model(dirty,mask),atol=0,rtol=0)
    with pytest.raises(ValueError):model(dirty,torch.ones_like(mask))


def test_encoder_cpu_cuda_parity():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(2);cpu=w.WideTokenEncoder().eval();gpu=deepcopy(cpu).cuda()
    x=torch.randn(3,6,125);mask=torch.ones(3,125,dtype=torch.bool);mask[:,40:50]=False
    with torch.inference_mode():
        torch.testing.assert_close(cpu(x,mask),gpu(x.cuda(),mask.cuda()).cpu(),atol=1e-5,rtol=1e-4)


def test_rotation_keeps_both_triads_in_the_same_frame():
    r=rotation_matrices(3,'cpu');x=torch.randn(3,3,125)
    y=rotate_triads(torch.cat((x,2*x),1),r)
    torch.testing.assert_close(y[:,3:],2*y[:,:3])


def metrics():return {'average_precision':.8540475046760654,'raw_logit_bce':.27704393608246073,'prevalence':.2}
def checks():return {k:True for k in ('finite','isolation','optimizer','resource','protected')}


def test_gate_accepts_exact_registered_boundaries():
    assert w.source_gate(metrics(),.3,checks())['passed'] is True


@pytest.mark.parametrize('key,value',[('average_precision',.8540474),('raw_logit_bce',.277044),('prevalence',.8)])
def test_gate_rejects_each_metric_failure(key,value):
    m=metrics();m[key]=value
    assert w.source_gate(m,.3,checks())['passed'] is False


def test_gate_rejects_sampler_baseline_failure_and_missing_integrity():
    assert w.source_gate(metrics(),.27,checks())['passed'] is False
    c=checks();del c['optimizer']
    assert w.source_gate(metrics(),.3,c)['passed'] is False
    with pytest.raises(ValueError):w.source_gate({**metrics(),'average_precision':np.nan},.3,checks())


@pytest.mark.parametrize('key,value',[('average_precision',1.1),('raw_logit_bce',-.1),('prevalence',-.1)])
def test_gate_rejects_out_of_range_metrics(key,value):
    with pytest.raises(ValueError):w.source_gate({**metrics(),key:value},.3,checks())


def test_read_reference_binds_the_shared_source_preprocessing_code():
    from pathlib import Path
    ref=w.read_reference(Path(__file__).resolve().parents[2])
    assert set(ref['helper_sha256'])=={'scripts/research/representation_v2/unitfree.py',
                                    'scripts/research/representation_v2/fd_source.py'}


def test_real_training_excludes_validation_and_its_signal_cannot_change_weights(tmp_path):
    b=bank();train=tuple(range(27));valid=tuple(range(27,34))
    first=w.train_bank(b,train,valid,device='cpu',steps=2,batch_size=4)
    changed=deepcopy(b);changed['values'][54:]*=100
    second=w.train_bank(changed,train,valid,device='cpu',steps=2,batch_size=4)
    p=first['payload'];assert set(first['subject_draws']).issubset(set(train))
    assert sum(first['subject_draws'].values())==8
    assert first['history'][-1]['step']==2 and np.isfinite(first['logits']).all()
    assert any(not torch.equal(p['encoder'][k],p['random_encoder'][k]) for k in p['encoder'])
    for k in p['encoder']:torch.testing.assert_close(p['encoder'][k],second['payload']['encoder'][k],atol=0,rtol=0)
    path=tmp_path/'final.pt';sha=w.save_artifact(path,p)
    loaded=w.load_artifact(path,sha)
    assert loaded['provenance']['steps']==2 and not list(tmp_path.glob('*.tmp'))
    for k in p['encoder']:torch.testing.assert_close(p['encoder'][k],loaded['encoder'][k])
    with pytest.raises(ValueError):w.load_artifact(path,'0'*64)


@pytest.mark.parametrize('mutation',['schema','shape','ancestry','nonfinite'])
def test_checkpoint_rejects_wrong_contract(tmp_path,mutation):
    p=w.train_bank(bank(),tuple(range(27)),tuple(range(27,34)),device='cpu',steps=1,batch_size=2)['payload']
    if mutation=='schema':p['schema']='old'
    elif mutation=='shape':p['head']['weight']=torch.zeros(1,17)
    elif mutation=='ancestry':p['provenance']['target_subjects']=['target-person']
    else:p['encoder']['stem.weight'][0,0,0]=float('nan')
    with pytest.raises(ValueError):w.save_artifact(tmp_path/'bad.pt',p)
    assert not (tmp_path/'bad.pt').exists() and not list(tmp_path.glob('*.tmp'))


def test_overlap_and_missing_participants_fail_before_training():
    with pytest.raises(ValueError):w.train_bank(bank(),(0,1),(1,2),device='cpu',steps=1,batch_size=2)
    with pytest.raises(ValueError):w.train_bank(bank(),tuple(range(27)),(27,),device='cpu',steps=1,batch_size=2)


def test_reference_population_and_validation_identity_are_not_allowed_to_drift():
    b=bank();train=tuple(range(27));valid=tuple(range(27,34))
    reference={'plan':{'source_train':list(train),'source_validation':list(valid)},
               'population':{'counts':deepcopy(b['counts']),'valid_tokens':68,'persistent_token_cache_bytes':0},
               'validation':[{'participant':p,'start_sample':s,'label':c,'logit':0.} for p in valid for s,c in [(0,0),(320,1)]]}
    w.validate_reference_bank(b,train,valid,reference)
    changed=deepcopy(b);changed['starts'][-1]=321
    with pytest.raises(ValueError):w.validate_reference_bank(changed,train,valid,reference)
    changed=deepcopy(b);changed['counts'][0]['available'][0]=2
    with pytest.raises(ValueError):w.validate_reference_bank(changed,train,valid,reference)


def test_existing_attempt_fails_before_source_or_target_access(tmp_path):
    out=tmp_path/'outputs/challenger_70/representation_v2/amendment_a3_wide_source'
    out.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='attempt exists'):w.run(tmp_path)


def test_budget_aborts_training_before_updates():
    def exhausted(seconds):raise TimeoutError('budget exhausted')
    with pytest.raises(TimeoutError):w.train_bank(bank(),tuple(range(27)),tuple(range(27,34)),device='cpu',steps=2,batch_size=4,budget=exhausted)


def test_budget_rejects_overrun_by_the_last_optimizer_step(monkeypatch):
    clock={'elapsed':0.}
    monkeypatch.setattr(w.time,'perf_counter',lambda:clock['elapsed'])
    def advance_after_last_step(history):clock['elapsed']=10.
    def limit(seconds):
        if seconds>5:raise TimeoutError('last step exceeded budget')
    with pytest.raises(TimeoutError,match='last step exceeded'):
        w.train_bank(bank(),tuple(range(27)),tuple(range(27,34)),device='cpu',steps=1,batch_size=2,
                     observe=advance_after_last_step,budget=limit)
