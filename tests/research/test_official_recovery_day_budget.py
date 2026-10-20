"""Catch changing scores/NMS when only the budget grouping should change."""
from dataclasses import asdict

import numpy as np
import pytest

from src.pipeline.event_stack import EventRef, MultiScaleCandidate, CandidateEvent


def test_day_grouping_uses_candidate_center_and_fixed_utc8_midnight():
    from scripts.research.official_recovery.day_budget_probe import budget_groups
    rows=[MultiScaleCandidate(EventRef('s',-28800002,-28800000),None,None),
          MultiScaleCandidate(EventRef('s',-28800000,-28799998),None,None)]
    assert budget_groups(rows,['A','A'],daily=False)==['A','A']
    assert budget_groups(rows,['A','A'],daily=True)==['A|-1','A|0']


def test_daily_cap_rescues_other_day_without_changing_score_threshold_or_nms():
    from scripts.research.official_recovery.day_budget_probe import budget_groups
    from scripts.research.official_recovery.event_probe import decode
    rows=[MultiScaleCandidate(EventRef('s',day*86400000,day*86400000+1000),None,None) for day in (0,1)]
    scores=np.array([.9,.8]); before=scores.copy()
    policy=dict(admission=dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=1),threshold=.5)
    assert len(decode(rows,scores,budget_groups(rows,['A','A'],daily=False),policy))==1
    assert len(decode(rows,scores,budget_groups(rows,['A','A'],daily=True),policy))==2
    np.testing.assert_array_equal(scores,before)


def test_reconstruction_preserves_stream_support_and_geometry():
    from scripts.research.official_recovery.day_budget_probe import restore_candidates
    event=EventRef('s',0,100)
    stream=CandidateEvent(event,(.2,.8),.9,0,0,2,3)
    row=MultiScaleCandidate(event,stream,None)
    assert restore_candidates([asdict(row)])==[row]


def test_score_ledger_rejects_incorrect_fit_identity_or_learned_health_ancestor():
    from scripts.research.official_recovery.day_budget_probe import validate_ledger
    row=dict(train=['A'],score=['B'],groups=['B'],excluded=['C'],
        graph={'score':{'fit_subjects':['B'],'parents':[]}})
    with pytest.raises(ValueError):
        validate_ledger(row,['A'],['B'],{'C'})
    row['graph']['score']['fit_subjects']=['A']
    validate_ledger(row,['A'],['B'],{'C'})
    with pytest.raises(ValueError):
        validate_ledger(row,['B'],['A'],{'C'})


def test_zero_candidate_subject_remains_a_valid_score_identity_and_its_truth_is_fn():
    from scripts.research.official_recovery.day_budget_probe import validate_ledger
    from scripts.research.official_recovery.event_probe import complete_metrics
    row=dict(train=['A'],score=['B','C'],groups=['B'],excluded=['D'],
        graph={'score':{'fit_subjects':['A'],'parents':[]}})
    validate_ledger(row,['A'],['B','C'],{'D'})
    metrics=complete_metrics([EventRef('b',0,100)],{'b':'B'},[('B',0,100),('C',0,100)])
    assert (metrics['tp'],metrics['fn'],metrics['truths'])==(1,1,2)
