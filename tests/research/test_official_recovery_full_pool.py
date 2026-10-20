"""Candidate expansion must not change the supervised window training pool."""
from dataclasses import replace

import numpy as np
import pytest

from src.pipeline.event_stack import EventRef,MultiScaleCandidate
from src.pipeline.runner import OuterTrainOnlyData,WindowBatch


def test_expansion_keeps_training_arrays_and_adds_missing_candidate_session():
    from scripts.research.official_recovery.full_pool_probe import expand_candidates
    window=EventRef('old',0,100)
    macro=WindowBatch(np.ones((1,62),np.float32),np.array([1]),(window,))
    micro=WindowBatch(np.ones((1,47),np.float32),np.array([1]),(window,))
    original=OuterTrainOnlyData(macro,macro,(),{'old':'A'},{'old':(0,100)},micro,micro)
    pools={}
    for kind,width in [('slide',63),('micro15',47)]:
        pools[kind]=dict(features=np.ones((2,width),np.float32),old=np.array([1,0]),
            sessions=np.array(['old','new']),subjects=np.array(['A','A']),intervals=np.array([[0,100],[200,300]]))
    expanded=expand_candidates(original,pools,{'old':'A','new':'A'},{'old':(0,100),'new':(200,300)})
    assert expanded.window_train is original.window_train
    assert expanded.micro_window_train is original.micro_window_train
    assert [w.sid for w in expanded.candidate_train.windows]==['old','new']
    assert expanded.subject_by_session=={'old':'A','new':'A'}
    assert expanded.candidate_train.features.shape==(2,62)


def test_expansion_refuses_incompatible_overlapping_features():
    from scripts.research.official_recovery.full_pool_probe import validate_subset
    old=WindowBatch(np.ones((1,2)),np.array([0]),(EventRef('s',0,100),))
    full=replace(old,features=np.zeros((1,2)))
    with pytest.raises(ValueError,match='features'):
        validate_subset(old,full)


def test_shared_producer_must_replay_every_partial_session_candidate():
    from scripts.research.official_recovery.full_pool_probe import verify_partial_candidates
    old=MultiScaleCandidate(EventRef('a',0,100),None,None)
    extra=MultiScaleCandidate(EventRef('b',0,100),None,None)
    verify_partial_candidates([old,extra],[old],{'a'})
    with pytest.raises(ValueError,match='candidate'):
        verify_partial_candidates([old,extra],[],{'a'})
