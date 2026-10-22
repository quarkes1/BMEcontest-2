from dataclasses import replace

import numpy as np
import pytest

from src.pipeline.event_stack import EventRef
from src.pipeline.runner import WindowBatch


def test_complete_window_labels_cross_sid_and_use_registered_overlap_boundaries():
    from scripts.research.official_recovery.complete_supervision import relabel_batch
    windows=(EventRef('left',0,100),EventRef('right',100,200),EventRef('right',400000,400100))
    batch=WindowBatch(np.zeros((3,62)),np.array([0,0,0]),windows)
    mapping={'left':'A','right':'A'};truths=[('A',50,150)]
    macro,stats=relabel_batch(batch,truths,mapping,micro=False)
    micro,_=relabel_batch(batch,truths,mapping,micro=True)
    np.testing.assert_array_equal(macro.labels,[-1,-1,0])
    np.testing.assert_array_equal(micro.labels,[1,1,0])
    np.testing.assert_array_equal(batch.labels,[0,0,0])
    assert stats['old0_to_new1']==0 and macro.features is batch.features
    with pytest.raises(ValueError): relabel_batch(batch,truths,{'left':'A'},micro=False)
    with pytest.raises(ValueError): relabel_batch(batch,[('B',0,100)],mapping,micro=False)


def test_complete_window_negative_buffer_is_exact():
    from scripts.research.official_recovery.complete_supervision import relabel_batch
    windows=(EventRef('s',300100,300200),EventRef('s',300099,300199))
    batch=WindowBatch(np.zeros((2,62)),np.zeros(2,dtype=int),windows)
    out,_=relabel_batch(batch,[('A',0,100)],{'s':'A'},micro=False)
    np.testing.assert_array_equal(out.labels,[0,-1])


def test_complete_micro_selection_counts_cross_sid_gt_once_and_strictly(monkeypatch):
    from scripts.research.official_recovery import complete_supervision as module
    from src.pipeline.event_stack import CandidateEvent
    # Two independently reachable fragments belong to the same full subject GT.
    def candidates(windows,threshold,config):
        return [CandidateEvent(EventRef(s,0,100),(),1.,0,0,0,0) for s in ('a','b')[:(2 if threshold==.1 else 1)]]
    monkeypatch.setattr(module,'micro_candidates',candidates)
    result=module.choose_micro({}, {'a':'A','b':'A'},[('A',0,100)],(.1,.2),None)
    assert result.threshold==.2 and result.metrics['truths']==1 and result.metrics['tp']==1
    boundary=module.choose_micro({}, {'a':'A','b':'A'},[('A',0,400)],(.1,.2),None)
    assert boundary.metrics['tp']==0


def test_complete_contract_and_gate_are_separate_from_prior_failed_variants():
    from scripts.research.official_recovery.macro_lgbm_event import macro_contract,event_gate
    from src.pipeline import runner
    c=macro_contract('full_labels')
    assert c['id']=='COMPLETE-SUPERVISION-V1' and c['factory'] is runner._window_estimator
    rows=[dict(fold=i,events={'A':dict(tp=20,predictions=30,truths=n),'G':dict(tp=20,predictions=28,truths=n)}) for i,n in enumerate((54,55,51))]
    assert event_gate(rows,'t24','full_labels')['passed']
    assert not event_gate(rows,'t24','lgbm')['passed']


def test_preparation_only_relabels_and_samples_fit_subjects():
    from scripts.research.official_recovery.complete_supervision import prepare_complete_train
    from src.pipeline.runner import OuterTrainOnlyData
    windows=tuple(EventRef(s,start,start+100) for s in ('a','b') for start in (0,400000,500000,600000))
    macro=WindowBatch(np.zeros((8,62)),np.zeros(8,dtype=np.int8),windows)
    micro=replace(macro,features=np.zeros((8,47)))
    data=OuterTrainOnlyData(macro,macro,(),{'a':'A','b':'B'},{'a':(0,700000),'b':(0,700000)},micro,micro)
    truths=[('A',0,100),('B',0,100)]
    left,stats=prepare_complete_train(data,{'A'},truths)
    right,stats2=prepare_complete_train(data,{'A'},[('A',0,100),('B',400000,600100)])
    assert stats==stats2 and set(left.subject_by_session.values())=={'A'}
    assert stats['macro']['sample']['positive']==1 and stats['macro']['sample']['rows']==4
    np.testing.assert_array_equal(left.window_train.labels,[1,0,0,0])
    np.testing.assert_array_equal(left.window_train.labels,right.window_train.labels)
    assert all(w.sid=='a' for w in left.window_train.windows)
    np.testing.assert_array_equal(data.window_train.labels,np.zeros(8))
