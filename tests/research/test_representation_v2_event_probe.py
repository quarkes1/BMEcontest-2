from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.research.representation_v2.event_probe import (
    fuse_scores, motion_probabilities, choose_policy, decode_policy,
    event_gate, verifier_probabilities, check_graph,
)
from src.pipeline.event_stack import EventRef, MultiScaleCandidate


def test_fixed_fusion_preserves_all_rows_and_rejects_misalignment():
    result=fuse_scores(np.array([.8,.2]),np.array([.4,.6]),np.array([.2,.8]))
    assert result['b'].tolist()==[.8,.2]
    np.testing.assert_allclose(result['r'],[.7,.3])
    np.testing.assert_allclose(result['f'],[.65,.35])
    with pytest.raises(ValueError): fuse_scores(np.array([.1]),np.array([.1,.2]),np.array([.1]))


def test_invalid_motion_rows_receive_training_prior_not_deleted():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(20,133)); y=np.array([0,1]*10); groups=np.array(['a']*10+['b']*10)
    p,_=motion_probabilities(x,y,groups,np.ones(20,bool),np.zeros((2,133)),np.array([True,False]))
    assert p.shape==(2,) and p[1]==.5 and np.isfinite(p).all()


def test_policy_roundtrip_keeps_one_match_and_rejects_empty_scoring():
    candidates=[MultiScaleCandidate(EventRef('s',0,100),None,None),
                MultiScaleCandidate(EventRef('s',1000,1100),None,None)]
    config=SimpleNamespace(admission_nms_iou_grid=(.5,),admission_threshold_grid=(.2,),
                           admission_subject_cap_grid=(5,),admission_minimum_recall=.8)
    p=choose_policy(candidates,np.array([.8,.3]),['p','p'],[EventRef('s',0,100)],config)
    assert decode_policy(candidates,np.array([.8,.3]),['p','p'],p)==[EventRef('s',0,100)]
    assert decode_policy([],np.array([]),[],p)==[]


def test_gate_rejects_tp_loss_even_if_f1_rises_and_requires_majority():
    metric=lambda tp,fp:{'n_tp':tp,'n_pred':tp+fp,'n_true':10}
    row={'events':{'b':metric(7,5),'r':metric(7,4),'f':metric(8,3)},'checks':{'integrity':True,'resource':True}}
    rows=[deepcopy(row) for _ in range(3)]
    assert event_gate(rows)['passed']
    for r in rows: r['events']['f']=metric(6,0)
    assert not event_gate(rows)['passed']
    rows=[deepcopy(row) for _ in range(3)]
    rows[0]['events']['f']=metric(7,6); rows[1]['events']['f']=metric(7,6)
    assert not event_gate(rows)['passed']


def test_real_verifier_supports_missing_features_and_finite_probabilities():
    rng=np.random.default_rng(8); x=rng.normal(size=(40,116)); x[:,0]=np.nan
    result=verifier_probabilities(x,np.array([0,1]*20),x[:3],seed=1)
    assert result.shape==(3,) and np.isfinite(result).all() and ((result>=0)&(result<=1)).all()


def test_graph_rejects_leaked_producer_ancestor():
    graph={'producer':{'fit_subjects':['heldout'],'parents':[]},
           'score':{'fit_subjects':['train'],'parents':['producer']}}
    with pytest.raises(ValueError): check_graph(graph,'score',{'heldout'},{'train'})
