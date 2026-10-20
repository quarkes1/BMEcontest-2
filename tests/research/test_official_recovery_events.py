"""Catch label scope, matching, policy and isolation mistakes with small fixtures."""
from types import SimpleNamespace

import numpy as np
import pytest


def candidate(sid, start, end):
    from src.pipeline.event_stack import EventRef
    from src.pipeline.event_stack import MultiScaleCandidate
    return MultiScaleCandidate(EventRef(sid, start, end), None, None)


def test_cross_file_candidate_becomes_positive_without_shortening_whole_meal():
    from scripts.research.official_recovery.event_probe import full_candidate_labels
    rows=[candidate('a1',0,40),candidate('a2',40,80),candidate('b',0,40)]
    labels=full_candidate_labels(rows,{'a1':'A','a2':'A','b':'B'},[('A',0,100)])
    assert labels.tolist()==[1,1,0]


def test_candidate_label_strict_boundary_and_missing_session():
    from scripts.research.official_recovery.event_probe import full_candidate_labels
    assert full_candidate_labels([candidate('s',0,25)],{'s':'A'},[('A',0,100)]).tolist()==[0]
    with pytest.raises(ValueError,match='session'):
        full_candidate_labels([candidate('unknown',0,30)],{},[('A',0,100)])


def test_subject_prediction_matching_counts_one_cross_file_meal_only_once():
    from scripts.research.official_recovery.event_probe import complete_metrics
    events=[candidate('a1',0,50).event,candidate('a2',50,100).event]
    result=complete_metrics(events,{'a1':'A','a2':'A'},[('A',0,100),('A',200,300)])
    assert (result['tp'],result['fp'],result['fn'])==(1,1,1)
    assert result['f1']==.5


def test_full_truth_population_keeps_uncovered_and_rejects_duplicate_gt():
    from scripts.research.official_recovery.event_probe import partition_truths
    rows=[('A',0,100),('A',1000,1100),('B',0,100)]
    assert partition_truths(rows,{'A'})==rows[:2]
    with pytest.raises(ValueError,match='duplicate'):
        partition_truths([*rows,rows[0]],{'A'})


def test_full_policy_threshold_uses_training_truth_not_session_containment():
    from scripts.research.official_recovery.event_probe import choose_complete_policy, decode
    rows=[candidate('s1',0,80),candidate('s2',300,350)]
    config=SimpleNamespace(admission_nms_iou_grid=(.3,),admission_threshold_grid=(.2,),
        admission_subject_cap_grid=(6,),admission_minimum_recall=.8)
    policy=choose_complete_policy(rows,np.array([.8,.3]),['A','A'],[('A',0,100)],
        {'s1':'A','s2':'A'},config)
    assert policy['threshold']==.8
    assert policy['selection_metrics']['tp']==1
    assert len(decode(rows,np.array([.8,.3]),['A','A'],policy))==1


def test_complete_policy_preserves_unreachable_truths_and_empty_prediction_case():
    from scripts.research.official_recovery.event_probe import choose_complete_policy
    config=SimpleNamespace(admission_nms_iou_grid=(.3,),admission_threshold_grid=(.2,),
        admission_subject_cap_grid=(6,),admission_minimum_recall=.8)
    policy=choose_complete_policy([],np.array([]),[],[('A',0,100)],{},config)
    assert policy['selection_metrics']['fn']==1


def test_fit_graph_rejects_heldout_ancestor_and_unauthorized_fit_subject():
    from scripts.research.official_recovery.event_probe import check_graph
    graph={'upstream':{'fit_subjects':['A','HEALTH'],'parents':[]},
           'score':{'fit_subjects':[],'parents':['upstream']}}
    with pytest.raises((ValueError,RuntimeError)):
        check_graph(graph,'score',{'HEALTH'},{'A'})
    graph['upstream']['fit_subjects']=['UNEXPECTED']
    with pytest.raises((ValueError,RuntimeError)):
        check_graph(graph,'score',{'HEALTH'},{'A'})


def test_gate_rejects_one_fold_win_and_accepts_consistent_actual_gain():
    from scripts.research.official_recovery.event_probe import assess_gate
    bad=[{'old':{'tp':8,'predictions':10,'truths':20},'full':{'tp':x,'predictions':10,'truths':20}}
         for x in (12,8,8)]
    assert assess_gate(bad)['passed'] is False
    good=[{'old':{'tp':8,'predictions':10,'truths':20},'full':{'tp':9,'predictions':10,'truths':20}}]*3
    result=assess_gate(good)
    assert result['passed'] is True
    assert result['aggregate']['full']['tp']==27
