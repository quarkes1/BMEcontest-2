"""Negative cleanup must never manufacture positives or change scoring coverage."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from src.pipeline.event_stack import EventRef
from src.pipeline.runner import WindowBatch,OuterTrainOnlyData


def batch(labels=(0,1,-1,0)):
    return WindowBatch(np.ones((4,2)),np.array(labels,dtype=np.int8),
        (EventRef('a',100,200),EventRef('a',300,400),EventRef('a',110,210),EventRef('b',100,200)))


def test_window_cleanup_uses_cross_file_truth_and_preserves_old_positive_gray_and_other_subject():
    from scripts.research.official_recovery.negative_clean import censor_window_batch
    raw=batch();clean,stats=censor_window_batch(raw,[('A',50,250)],{'a':'A','b':'B'},micro=False)
    assert clean.labels.tolist()==[-1,1,-1,0]
    assert raw.labels.tolist()==[0,1,-1,0] and clean.windows==raw.windows
    assert clean.features is raw.features and stats['censored_negative']==1


def test_macro_and_micro_keep_their_distinct_half_overlap_boundaries():
    from scripts.research.official_recovery.negative_clean import censor_window_batch
    raw=WindowBatch(np.ones((1,2)),np.array([0],dtype=np.int8),(EventRef('a',100,200),))
    macro,_=censor_window_batch(raw,[('A',50,150)],{'a':'A'},micro=False)
    micro,_=censor_window_batch(raw,[('A',50,150)],{'a':'A'},micro=True)
    assert macro.labels.tolist()==[0] and micro.labels.tolist()==[-1]


def test_window_cleanup_rejects_unknown_session_and_invalid_label():
    from scripts.research.official_recovery.negative_clean import censor_window_batch
    with pytest.raises(ValueError,match='session'): censor_window_batch(batch(),[],{'a':'A'},micro=False)
    with pytest.raises(ValueError,match='label'): censor_window_batch(batch((2,1,-1,0)),[],{'a':'A','b':'B'},micro=False)


def test_candidate_cleanup_is_strict_iou_and_does_not_add_positives():
    from scripts.research.official_recovery.negative_clean import censor_candidate_labels
    candidates=[SimpleNamespace(event=EventRef('a',0,end)) for end in (100,101,101)]
    old=np.array([0,0,1],dtype=np.int8)
    new,stats=censor_candidate_labels(candidates,old,[('A',0,400)],{'a':'A'})
    assert new.tolist()==[0,-1,1] and old.tolist()==[0,0,1]
    assert stats['censored_negative']==1 and stats['positive']==1


def test_prepare_train_samples_only_train_without_dropping_full_candidate_windows():
    from scripts.research.official_recovery.negative_clean import prepare_train
    macro=batch();micro=batch()
    universe=OuterTrainOnlyData(macro,macro,(),{'a':'A','b':'B'},
        {'a':(0,500),'b':(0,500)},micro,micro)
    data,stats=prepare_train(universe,{'A'},[('A',50,250),('B',50,250)],'W')
    assert set(data.subject_by_session.values())=={'A'}
    assert len(data.candidate_train.windows)==3
    assert [w.sid for w in data.window_train.windows]==['a'] and data.window_train.labels.tolist()==[1]
    assert stats['macro']['sample']['positive']==1
    assert universe.candidate_train.labels.tolist()==[0,1,-1,0]
    with pytest.raises(ValueError,match='arm'): prepare_train(universe,{'A'},[],'WE')


def _records():
    return [dict(fold=i,events={a:dict(tp=20+(2 if a=='W' else 0),predictions=25,truths=n)
        for a in ('A','W','E')}) for i,n in enumerate((50,55,55))]


def test_gate_requires_practical_gain_majority_and_no_tp_loss():
    from scripts.research.official_recovery.negative_clean_probe import assess_clean_gate
    rows=_records();gate=assess_clean_gate(rows)
    assert gate['winner']=='W' and gate['arms']['W']['passed'] and not gate['arms']['E']['passed']
    for row in rows: row['events']['W']=dict(tp=18,predictions=10,truths=row['events']['A']['truths'])
    assert assess_clean_gate(rows)['winner'] is None
    rows=_records()
    for i,row in enumerate(rows): row['events']['W']['tp']=30 if i==0 else 20
    assert assess_clean_gate(rows)['winner'] is None


def test_gate_rejects_missing_fold_different_truths_or_small_gain():
    from scripts.research.official_recovery.negative_clean_probe import assess_clean_gate
    rows=_records()
    with pytest.raises(ValueError,match='three'): assess_clean_gate(rows[:2])
    rows[0]['events']['E']['truths']+=1
    with pytest.raises(ValueError,match='denominator'): assess_clean_gate(rows)
    rows=_records()
    for i,row in enumerate(rows): row['events']['W']['tp']=21 if i<2 else 20
    assert assess_clean_gate(rows)['winner'] is None


def test_probe_preflight_failure_is_incomplete_without_overwriting_existing_output(tmp_path):
    from scripts.research.official_recovery.negative_clean_probe import execute
    output=tmp_path/'output'
    with pytest.raises(Exception): execute(tmp_path,output)
    import json
    assert json.loads((output/'state.json').read_text())['state']=='INCOMPLETE'
    before=(output/'state.json').read_bytes()
    with pytest.raises(FileExistsError): execute(tmp_path,output)
    assert (output/'state.json').read_bytes()==before


def test_policy_population_retains_subjects_with_zero_candidates():
    from scripts.research.official_recovery.negative_clean_probe import validate_policy_population
    validate_policy_population(['A','B'],['A','A'],{'A','B'})
    with pytest.raises(ValueError): validate_policy_population(['A','A'],['A'],{'A','B'})
    with pytest.raises(ValueError): validate_policy_population(['A','B'],['C'],{'A','B'})
