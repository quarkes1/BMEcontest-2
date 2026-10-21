import numpy as np
import pytest

from src.pipeline.event_stack import EventRef


def spans_for(owners=('A',),n=600):
    from scripts.research.official_recovery.dense_mil import build_spans
    windows=[EventRef(s,i*7500,i*7500+15000) for s in owners for i in range(n)]
    x=np.vstack([np.full((n,47),1. if s=='A' else 1000.,np.float32) for s in owners])
    return build_spans(x,windows,{s:s for s in owners})


def test_spans_break_on_session_or_actual_grid_gap_and_reject_duplicates():
    from scripts.research.official_recovery.dense_mil import build_spans
    w=[EventRef('a',0,15000),EventRef('a',7500,22500),EventRef('a',22500,37500),EventRef('b',0,15000)]
    spans=build_spans(np.zeros((4,47)),w,{'a':'A','b':'A'})
    assert [len(s.windows) for s in spans]==[2,1,1]
    with pytest.raises(ValueError): build_spans(np.zeros((2,47)),w[:1]*2,{'a':'A'})
    with pytest.raises(ValueError): build_spans(np.zeros((4,47)),w,{'a':'A'})


def test_normalizer_uses_only_fit_subjects_and_handles_missing_channels():
    from scripts.research.official_recovery.dense_mil import fit_normalizer
    spans=spans_for(('A','B'),20);spans[0].features[:,0]=np.nan
    center,scale=fit_normalizer(spans,{'A'})
    assert center[0]==0 and scale[0]==1
    assert np.all(center[1:]==1) and np.all(scale>0)
    with pytest.raises(ValueError): fit_normalizer(spans,{'unknown'})


def test_normalizer_sampling_uses_physical_time_not_incoming_span_order():
    from scripts.research.official_recovery.dense_mil import build_spans,fit_normalizer
    windows=[EventRef(s,offset+i*7500,offset+i*7500+15000)
        for s,offset in (('a',0),('b',100000000)) for i in range(3000)]
    spans=build_spans(np.random.default_rng(1).normal(size=(6000,47)),windows,{'a':'A','b':'A'})
    a=fit_normalizer(spans,{'A'});b=fit_normalizer(list(reversed(spans)),{'A'})
    assert all(np.array_equal(x,y) for x,y in zip(a,b))


def test_normalizer_missing_gap_duration_does_not_repeat_edge_observations():
    from dataclasses import replace
    from scripts.research.official_recovery.dense_mil import build_spans,fit_normalizer
    windows=[EventRef(s,offset+i*7500,offset+i*7500+15000)
        for s,offset in (('a',0),('b',100000000)) for i in range(3000)]
    spans=build_spans(np.random.default_rng(3).normal(size=(6000,47)),windows,{'a':'A','b':'A'})
    later=replace(spans[1],windows=tuple(EventRef(w.sid,w.start_ms+1000000000,w.end_ms+1000000000) for w in spans[1].windows))
    before=fit_normalizer(spans,{'A'});after=fit_normalizer([spans[0],later],{'A'})
    assert all(np.array_equal(x,y) for x,y in zip(before,after))


def test_bags_use_all_fragments_without_cross_session_convolution():
    from scripts.research.official_recovery.dense_mil import build_spans,make_bags,pack_spans
    w=[EventRef(s,i*7500+offset,i*7500+offset+15000) for s,offset in (('a',0),('b',100000)) for i in range(10)]
    spans=build_spans(np.zeros((20,47)),w,{'a':'A','b':'A'})
    bags,stats=make_bags(spans,[('A',0,200000),('B',0,200000)],{'A'})
    positive=[b for b in bags if b.label==1]
    assert len(positive)==1 and set(positive[0].refs[:,0])=={0,1}
    assert stats['positive_truths']==1
    packed=pack_spans(spans,np.zeros(47),np.ones(47))
    assert len(packed['x'])==2 and packed['mask'].sum().item()==20


def test_negative_bags_are_wholly_buffered_and_durations_train_only():
    from scripts.research.official_recovery.dense_mil import make_bags
    spans=spans_for(('A',));truths=[('A',600000,660000),('B',0,4000000)]
    bags,stats=make_bags(spans,truths,{'A'})
    negative=[b for b in bags if b.label==0]
    assert negative and stats['positive_bags']==1
    assert all(b.interval[1]-b.interval[0]==60000 for b in negative)
    assert all(b.interval[1]<=300000 or b.interval[0]>=960000 for b in negative)
    assert all(b.subject=='A' for b in bags)


def test_top_fraction_loss_does_not_label_every_meal_token_positive():
    import torch
    from scripts.research.official_recovery.dense_mil import bag_logit
    logits=torch.tensor([-2.,-1.,0.,1.,2.],requires_grad=True)
    value=bag_logit(logits);value.backward()
    assert value.item()==2 and logits.grad.tolist()==[0.,0.,0.,0.,1.]
    with pytest.raises(ValueError): bag_logit(torch.empty(0))


def test_chunked_encoder_matches_full_span_without_gt_boundary_padding():
    import torch
    from scripts.research.official_recovery.dense_mil import TemporalMIL,pack_spans
    torch.set_num_threads(2);torch.manual_seed(2)
    spans=spans_for(('A',),600);spans[0].features[:]=np.random.default_rng(2).normal(size=(600,47))
    packed=pack_spans(spans,np.zeros(47),np.ones(47));model=TemporalMIL().eval()
    with torch.no_grad():
        direct=model(torch.from_numpy(spans[0].features[None]),torch.ones((1,600),dtype=torch.bool))[0]
        pieces=model(packed['x'],packed['mask'])
    actual=torch.cat([pieces[i,31:31+row['length']] for i,row in enumerate(packed['records'])])
    assert torch.allclose(actual,direct,atol=2e-5,rtol=2e-5)


def test_proposals_do_not_cross_gap_and_clip_expansion():
    from scripts.research.official_recovery.dense_mil import build_spans,dense_proposals
    w=[EventRef('a',t,t+15000) for t in (0,7500,300000,307500)]
    spans=build_spans(np.zeros((4,47)),w,{'a':'A'})
    proposals=[p for s in spans for p in dense_proposals(s,np.array([.8,.9]))]
    assert [(e.start_ms,e.end_ms) for e,_ in proposals]==[(0,22500),(300000,322500)]
    with pytest.raises(ValueError): dense_proposals(spans[0],np.array([np.nan,.9]))


def test_rescue_preserves_baseline_and_has_subject_caps_and_stable_ties():
    from scripts.research.official_recovery.dense_mil import rescue
    base=[EventRef('a',0,100)];mapping={'a':'A','b':'B'}
    candidates=[(EventRef('a',10,110),.99),(EventRef('a',1000,1100),.9),
                (EventRef('a',2000,2100),.9),(EventRef('b',0,100),.9)]
    assert rescue(base,candidates,mapping,.8,0)==base
    out=rescue(base,list(reversed(candidates)),mapping,.8,1)
    assert out==[base[0],EventRef('a',1000,1100),EventRef('b',0,100)]
    assert base==[EventRef('a',0,100)]


def test_fit_and_prediction_are_subject_isolated_and_budget_checked_before_steps():
    import torch
    from scripts.research.official_recovery.dense_mil import fit_model,predict_model
    before=torch.are_deterministic_algorithms_enabled()
    calls=[]
    try:
        spans=spans_for(('A','B'),600)
        fit,stats=fit_model(spans,[('A',600000,660000),('B',0,4000000)],{'A'},
            seed=4,device='cpu',steps=2,callback=lambda:calls.append(1))
        assert len(calls)==3
        assert fit['fit_subjects']=={'A'} and np.all(fit['center']==1)
        assert stats['steps']==2 and stats['positive_truths']==1
        scores=predict_model(fit,spans,{'B'})
        assert len(scores)==1 and np.isfinite(scores[0][1]).all() and len(scores[0][1])==600
        with pytest.raises(ValueError,match='fit'): predict_model(fit,spans,{'A'})
    finally:
        torch.use_deterministic_algorithms(before)


def test_cuda_synthetic_fit_and_cpu_cuda_probability_parity():
    import torch
    from scripts.research.official_recovery.dense_mil import fit_model,predict_model
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    before=torch.are_deterministic_algorithms_enabled()
    try:
        spans=spans_for(('A','B'),200)
        fit,stats=fit_model(spans,[('A',300000,330000)],{'A'},seed=5,device='cuda',steps=2)
        gpu=predict_model(fit,spans,{'B'})[0][1]
        fit['model']=fit['model'].cpu();fit['device']='cpu'
        cpu=predict_model(fit,spans,{'B'})[0][1]
        assert np.allclose(gpu,cpu,atol=1e-5,rtol=1e-4)
        assert stats['cuda_peak_bytes']>0
    finally:
        torch.use_deterministic_algorithms(before)


def test_rescue_policy_can_choose_exact_baseline_instead_of_adding_fp():
    from scripts.research.official_recovery.dense_mil_probe import choose_policy
    base=[EventRef('a',0,100)];candidates=[(EventRef('a',1000,1100),.999)]
    policy=choose_policy(base,candidates,{'a':'A'},[('A',0,100)])
    assert policy['cap']==0 and policy['threshold']==.99


def test_dense_gate_preserves_complete_denominator_and_tp_floor():
    from scripts.research.official_recovery.dense_mil_probe import rescue_gate
    rows=[dict(fold=i,events={a:dict(tp=20+(3 if a=='R' else 0),predictions=30+(3 if a=='R' else 0),truths=n)
        for a in ('A','R')}) for i,n in enumerate((50,55,55))]
    assert rescue_gate(rows)['passed']
    with pytest.raises(ValueError): rescue_gate(rows[:2])
    rows[0]['events']['R']['truths']+=1
    with pytest.raises(ValueError): rescue_gate(rows)
    rows[0]['events']['R']['truths']-=1
    for row in rows: row['events']['R'].update(tp=19,predictions=20)
    assert not rescue_gate(rows)['passed']


def test_dense_probe_failure_is_incomplete_and_old_output_not_overwritten(tmp_path):
    import json
    from scripts.research.official_recovery.dense_mil_probe import execute
    out=tmp_path/'out'
    with pytest.raises(Exception): execute(tmp_path,out)
    assert json.loads((out/'state.json').read_text())['state']=='INCOMPLETE'
    before=(out/'state.json').read_bytes()
    with pytest.raises(FileExistsError): execute(tmp_path,out)
    assert (out/'state.json').read_bytes()==before
