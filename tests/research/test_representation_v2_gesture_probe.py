from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from scripts.research.representation_v2 import gesture_probe as gp
from scripts.research.representation_v2.dataset import TokenBag
from scripts.research.representation_v2.models import MotionBranch


def bag(n=4):
    rng=np.random.default_rng(19)
    return TokenBag(('s',0,max(n,1)*5000),rng.normal(size=(n,6,125)).astype(np.float32),
                    np.ones((n,125),bool),np.ones(n,bool),np.arange(n)*5000+2500,
                    0,max(n,1)*5000,np.ones(n))


def test_morphology_literal_schema_order_and_inclusive_thresholds():
    x=gp.morphology(np.array([.2,.8,.8,.2]),bag())
    expected=[.5,.3,.8,.2,.5,.8,.8,.8,1,.5,.5,1,1,1,.5,1,.5,-.5,-1,.4,1,1,np.log(5),np.log(21)]
    np.testing.assert_allclose(x,expected,atol=1e-12)


def test_invalid_tokens_and_true_time_gaps_break_runs_and_lag_chains():
    b=replace(bag(),token_mask=np.array([1,0,1,1],bool))
    x=gp.morphology(np.array([.8,np.nan,.8,.8]),b)
    np.testing.assert_allclose(x[11:17],[2,2/3,2,2/3,2,2/3])
    assert np.isnan(x[17:19]).all() and x[19]==0 and x[20]==.75
    gap=replace(bag(),token_center_ms=np.array([2500,7500,17500,22500]),end_ms=25000)
    z=gp.morphology(np.full(4,.8),gap)
    np.testing.assert_allclose(z[11:17],[2,.5,2,.5,2,.5])


def test_lag_two_requires_contiguous_valid_intervening_token():
    x=gp.morphology(np.array([.1,.2,.4,.8]),bag())
    np.testing.assert_allclose(x[17:19],[1,1])
    b=replace(bag(),token_mask=np.array([1,0,1,1],bool))
    assert np.isnan(gp.morphology(np.array([.1,999,.4,.8]),b)[18])


def test_empty_invalid_and_single_token_keep_missingness_distinct_from_zero():
    for b in (bag(0),replace(bag(),token_mask=np.zeros(4,bool))):
        x=gp.morphology(np.full(len(b.token_mask),np.nan),b)
        assert x.shape==(24,) and np.isnan(x[:20]).all()
        assert x[20]==0 and x[22]==0 and np.isfinite(x[20:]).all()
    x=gp.morphology(np.array([.2]),bag(1))
    assert x[1]==0 and np.isnan(x[17:20]).all()
    np.testing.assert_allclose(x[8:17],[1,0,0,1,1,0,0,0,0])


@pytest.mark.parametrize('change',[
    {'token_center_ms':np.array([2500,7500,7500,17500])},
    {'token_center_ms':np.array([7500,2500,12500,17500])},
    {'end_ms':0}, {'sample_mask':np.ones((4,124),bool)},
])
def test_malformed_geometry_fails_closed(change):
    with pytest.raises(ValueError):gp.morphology(np.full(4,.5),replace(bag(),**change))


@pytest.mark.parametrize('values',[[.1,.2,np.nan,.4],[.1,-.1,.3,.4],[.1,.2,1.1,.4]])
def test_nonfinite_or_out_of_range_valid_probabilities_rejected(values):
    with pytest.raises(ValueError):gp.morphology(np.array(values),bag())


def test_source_head_scores_only_valid_tokens_without_weight_or_gradient_changes():
    torch.manual_seed(4)
    encoder=MotionBranch().token_encoder.requires_grad_(False)
    head=torch.nn.Linear(16,1).requires_grad_(False)
    before_e=deepcopy(encoder.state_dict());before_h=deepcopy(head.state_dict())
    b=replace(bag(),token_mask=np.array([1,0,1,1],bool))
    z=gp.ap.encode_tokens(encoder,[b],'cpu')
    scores=gp.source_probabilities(z,[b],head,device='cpu')[0]
    with torch.inference_mode():expected=torch.sigmoid(head(torch.tensor(z[0][b.token_mask]))).squeeze(-1).numpy()
    np.testing.assert_allclose(scores[b.token_mask],expected,atol=1e-7)
    assert np.isnan(scores[1])
    for k,v in encoder.state_dict().items():torch.testing.assert_close(v,before_e[k],atol=0,rtol=0)
    for k,v in head.state_dict().items():torch.testing.assert_close(v,before_h[k],atol=0,rtol=0)
    assert all(p.grad is None for p in list(encoder.parameters())+list(head.parameters()))
    head.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):gp.source_probabilities(z,[b],head,device='cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_source_probability_cpu_cuda_parity():
    torch.manual_seed(4)
    e=MotionBranch().token_encoder.requires_grad_(False);h=torch.nn.Linear(16,1).requires_grad_(False)
    b=bag();cpu=gp.ap.encode_tokens(e,[b],'cpu');cuda=gp.ap.encode_tokens(e,[b],'cuda')
    np.testing.assert_allclose(cpu[0],cuda[0],rtol=1e-4,atol=1e-5)
    pc=gp.source_probabilities(cpu,[b],h,device='cpu')[0]
    pg=gp.source_probabilities(cuda,[b],h,device='cuda')[0]
    np.testing.assert_allclose(pc,pg,rtol=1e-4,atol=1e-5)


def test_real_140d_verifier_handles_all_missing_train_morphology_without_validation_fit():
    rng=np.random.default_rng(3);x=rng.normal(size=(80,140));y=(x[:,0]>0).astype(int)
    x[:,116:136]=np.nan;v=x[:4].copy();v[:,116:136]=.9
    p=gp.gesture_probabilities(x,y,v,seed=1)
    assert p.shape==(4,) and np.isfinite(p).all()
    v[:,116:136]=999
    np.testing.assert_allclose(gp.gesture_probabilities(x,y,v,seed=1),p,atol=1e-12)
    with pytest.raises(ValueError):gp.gesture_probabilities(x[:,:139],y,v,seed=1)
    base=x[:,:116].copy();motion=x[:,116:].copy()
    joined=gp.gesture_features(base,motion)
    assert joined.shape==(80,140)
    np.testing.assert_array_equal(joined[:,:116],base)


def test_real_split_has_source_ancestry_and_no_score_truth_in_fit(monkeypatch,tmp_path):
    from tests.research.test_representation_v2_crossfit import _data
    from src.pipeline.challenger.e03 import registered_e03_config
    from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
    data=_data();config=replace(registered_e03_config(0,workers=1),micro_positive_middle_fraction=None)
    # Tiny ten-second fixture needs an explicit test-only decoder geometry.
    config=replace(config,density=replace(config.density,stride_ms=10000,window_ms=10000,
        density_ms=10000,min_positive=1,coverage_min=0.,merge_ms=0,bridge_ms=0))
    torch.manual_seed(9);encoder=MotionBranch().token_encoder.requires_grad_(False)
    head=torch.nn.Linear(16,1)
    payload={'head':head.state_dict(),'provenance':{'source_train':[0],'source_validation':[1]}}
    def raw_bags(root,candidates,*,config):
        assert config.gravity_align is False
        return tuple(replace(bag(2),candidate_key=(c.event.sid,c.event.start_ms,c.event.end_ms),
                             start_ms=c.event.start_ms,end_ms=c.event.end_ms,
                             token_center_ms=np.array([c.event.start_ms+2500,c.event.start_ms+7500])) for c in candidates)
    monkeypatch.setattr(gp.ep,'load_candidate_bags',raw_bags)
    # Synthetic tiny data cannot replay the historical real-data ledger; only this I/O boundary is replaced.
    monkeypatch.setattr(gp.ap.jp,'baseline_replay',lambda *args:'synthetic-ledger-fixture')
    monkeypatch.setattr(gp,'frozen_reference_replay',lambda *args:'synthetic-reference-fixture')
    split=gp.prepare(tmp_path,payload,tmp_path,'cpu')
    train={f'person{i}' for i in range(6)};score={'person6','person7'}
    _,scores,graph,_=split(tmp_path,data,config,train,score,set(),{'f':encoder},
                         {'FD-R-0'},{'FD-R-1'},tmp_path,'left','cpu')
    assert graph['r_verifier']['parents']==['candidates','source']
    assert graph['f_verifier']['parents']==['candidates','source']
    assert_no_heldout_ancestors('score',score|{'FD-R-1'},graph)
    changed=replace(data,train_truths=tuple(t for t in data.train_truths if t.sid!='sid7'))
    _,right,_,_=split(tmp_path,changed,config,train,score,set(),{'f':encoder},
                     {'FD-R-0'},{'FD-R-1'},tmp_path,'right','cpu')
    for arm in ('b','r','f'):np.testing.assert_allclose(scores[arm],right[arm],atol=0,rtol=0)


def test_frozen_reference_replay_detects_changed_scores_not_just_same_model_name(monkeypatch):
    old={'train':['a'],'score':['b'],'excluded':['c'],'candidates':[],'groups':['b'],'labels':[0],
         'scores':{'r':[.7]}}
    monkeypatch.setattr(gp.subprocess,'check_output',lambda *args,**kwargs:json.dumps(old).encode())
    assert gp.frozen_reference_replay('.', 'fold0_policy0',old)
    changed=deepcopy(old);changed['scores']['r']=[.71]
    with pytest.raises(AssertionError):gp.frozen_reference_replay('.', 'fold0_policy0',changed)


def test_s11_precondition_rejects_incomplete_and_unsealed_evidence(monkeypatch,tmp_path):
    p=tmp_path/gp.S11_PATH;p.mkdir(parents=True)
    summary={'status':'S11_NO_SUPPORTED_OPTIMIZATION_EVENT_GAIN','passed':False,'completed_splits':12,
             'protected_unchanged':True,'outer_calls':0}
    state={'status':summary['status']}
    sealed={name:json.dumps(d).encode() for name,d in [('summary.json',summary),('state.json',state)]}
    for name,raw in sealed.items():(p/name).write_bytes(raw)
    monkeypatch.setattr(gp.subprocess,'check_output',lambda args,**kwargs:sealed[args[-1].split('/')[-1]])
    assert gp.assert_s11_closed(tmp_path)
    summary['completed_splits']=11;sealed['summary.json']=json.dumps(summary).encode()
    (p/'summary.json').write_bytes(sealed['summary.json'])
    with pytest.raises(RuntimeError,match='complete failed'):gp.assert_s11_closed(tmp_path)
    summary['completed_splits']=12;(p/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(RuntimeError,match='sealed'):gp.assert_s11_closed(tmp_path)
