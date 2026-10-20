"""Frozen-policy, full-cohort validation must preserve the heldout boundary."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.pipeline.event_stack import EventRef
from src.pipeline.runner import WindowBatch,OuterTrainOnlyData


def policy_fixture(tmp_path):
    policy=dict(admission_threshold=.2,blend_weight=.25,max_candidates_per_subject=6,
                max_events_per_group=None,nms_iou=.3,threshold=.6)
    values={'policy.json':policy,'run_config.json':{'outer_fold':0,'verifier_c_grid':[.1]}}
    for name,value in values.items(): (tmp_path/name).write_text(json.dumps(value))
    manifest={'role':'outer-fold-evidence','run_key':'outer-fold-0','files':{
        name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in values}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return policy


def test_sealed_policy_supplies_admission_blend_and_micro_threshold(tmp_path):
    from scripts.research.official_recovery.consistent_validation import read_fold_policy
    policy_fixture(tmp_path)
    frozen=read_fold_policy(tmp_path,0,.2)
    assert frozen['policy']['admission']==dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=6)
    assert frozen['blend']==.25 and frozen['micro_threshold']==.2 and frozen['verifier_c']==.1
    with pytest.raises(ValueError,match='fold'):
        read_fold_policy(tmp_path,1,.2)


def test_changed_sealed_policy_cannot_silently_change_control_values(tmp_path):
    from scripts.research.official_recovery.consistent_validation import read_fold_policy
    policy_fixture(tmp_path)
    (tmp_path/'policy.json').write_text('{}')
    with pytest.raises(ValueError,match='SHA'):
        read_fold_policy(tmp_path,0,.1)


def test_full_heldout_candidates_do_not_add_heldout_training_rows():
    from scripts.research.official_recovery.consistent_validation import assemble_full_data
    macro=WindowBatch(np.ones((1,62)),np.array([1]),(EventRef('a',0,100),))
    micro=WindowBatch(np.ones((1,47)),np.array([1]),(EventRef('a',0,100),))
    old=OuterTrainOnlyData(macro,macro,(),{'a':'A'},{'a':(0,100)},micro,micro)
    pools={kind:dict(features=np.ones((2,width)),old=np.array([1,0]),sessions=np.array(['a','b']),
        subjects=np.array(['A','B']),intervals=np.array([[0,100],[0,100]])) for kind,width in [('slide',63),('micro15',47)]}
    expanded=assemble_full_data(old,pools,{'a':'A','b':'B'},{'a':(0,100),'b':(0,100)})
    assert expanded.window_train is macro and expanded.micro_window_train is micro
    assert [w.sid for w in expanded.candidate_train.windows]==['a','b']
    assert expanded.subject_by_session['b']=='B'


def test_train_role_uses_validated_canonical_ids_before_subject_slice():
    from scripts.research.official_recovery.consistent_validation import canonical_train_role
    batch=WindowBatch(np.ones((2,62)),np.array([1,0]),(EventRef('a',0,100),EventRef('b',0,100)))
    old=OuterTrainOnlyData(batch,batch,(),{'a':'hnu 21013','b':'B'},
                           {'a':(0,100),'b':(0,100)},batch,batch)
    result=canonical_train_role(old,{'a':'HNU21013','b':'B','c':'C'},{'HNU21013'})
    assert result.subject_by_session=={'a':'HNU21013'}
    assert [w.sid for w in result.window_train.windows]==['a']
    np.testing.assert_array_equal(result.window_train.features,np.ones((1,62)))
    with pytest.raises(KeyError):
        canonical_train_role(old,{'b':'B'},{'HNU21013'})


def test_blend_uses_the_sealed_weight_and_rejects_invalid_weights():
    from scripts.research.official_recovery.consistent_validation import fit_verifiers
    x=np.zeros((8,116));x[:,0]=np.arange(8);y=np.array([0]*4+[1]*4)
    p,components=fit_verifiers(x,y,x,SimpleNamespace(seed=1),.1,.25)
    np.testing.assert_allclose(p,.25*components['lr']+.75*components['lgbm'])
    assert not np.allclose(components['lr'],components['lgbm'])
    with pytest.raises(ValueError,match='blend'):
        fit_verifiers(x,y,x,SimpleNamespace(seed=1),.1,1.5)


def metric(tp): return dict(tp=tp,predictions=10,truths=10)


def test_five_fold_gate_requires_practical_gain_and_majority_not_one_large_fold():
    from scripts.research.official_recovery.consistent_validation import validation_gate
    good=[dict(fold=i,release_style=metric(7),consistent_style=metric(8)) for i in range(5)]
    assert validation_gate(good)['passed'] is True
    one=[dict(fold=i,release_style=metric(1),consistent_style=metric(8 if i==0 else 1)) for i in range(5)]
    assert validation_gate(one)['passed'] is False
    tiny=[dict(fold=i,release_style=dict(tp=70,predictions=100,truths=100),
               consistent_style=dict(tp=71,predictions=100,truths=100)) for i in range(5)]
    assert validation_gate(tiny)['passed'] is False


def test_gate_refuses_missing_folds_or_different_denominators():
    from scripts.research.official_recovery.consistent_validation import validation_gate
    rows=[dict(fold=i,release_style=metric(7),consistent_style=metric(8)) for i in range(5)]
    with pytest.raises(ValueError,match='five'):
        validation_gate(rows[:4])
    rows[0]['consistent_style']['truths']=11
    with pytest.raises(ValueError,match='denominator'):
        validation_gate(rows)


def test_preflight_failure_writes_incomplete_and_existing_output_is_not_overwritten(tmp_path):
    from scripts.research.official_recovery.consistent_validation import execute
    output=tmp_path/'result'
    with pytest.raises(Exception):
        execute(tmp_path,output)
    assert json.loads((output/'state.json').read_text())['state']=='INCOMPLETE'
    before=(output/'state.json').read_bytes()
    with pytest.raises(FileExistsError):
        execute(tmp_path,output)
    assert (output/'state.json').read_bytes()==before
