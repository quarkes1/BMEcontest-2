"""An inner score's candidate producers must exclude its subjects."""

import numpy as np
import pytest
from dataclasses import replace

from scripts.research.representation_v2.crossfit import build_candidate_ledgers, inner_subject_partitions, slice_outer_train
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.event_stack import EventRef
from src.pipeline.runner import OuterTrainOnlyData, WindowBatch


def _batch(sids):
    windows = tuple(EventRef(sid, i * 10_000, (i + 1) * 10_000) for i, sid in enumerate(sids))
    return WindowBatch(np.zeros((len(sids), 2)), np.asarray([i % 2 for i in range(len(sids))]), windows)


def _data():
    sids = tuple(f"sid{i}" for i in range(8))
    return OuterTrainOnlyData(
        window_train=_batch(sids), candidate_train=_batch(sids),
        train_truths=(EventRef("sid0", 0, 10_000), EventRef("sid7", 70_000, 80_000)),
        subject_by_session={sid: f"person{i}" for i, sid in enumerate(sids)},
        session_bounds_by_sid={sid: (0, 100_000) for sid in sids},
        micro_window_train=_batch(sids), micro_candidate_train=_batch(sids),
    )


def test_gyro_pool_actual_crossfit_uses_87_macro_columns_and_original_micro(monkeypatch):
    from scripts.research.official_recovery.gyro_pool_macro import augment_data
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    data=_data();base=data.candidate_train
    base=replace(base,features=np.zeros((len(base.windows),62)))
    refs=tuple(EventRef(w.sid,w.start_ms+j*5000,w.start_ms+(j+1)*5000) for w in base.windows for j in range(2))
    micro=WindowBatch(np.ones((len(refs),47)),np.tile([0,1],len(base.windows)),refs)
    data=replace(data,window_train=base,candidate_train=base,micro_window_train=micro,micro_candidate_train=micro)
    enhanced,stats=augment_data(data);seen=[];original=Pipeline.fit
    def tracked(self,x,y,**kwargs):
        seen.append((isinstance(self.named_steps['model'],HistGradientBoostingClassifier),np.asarray(x).shape[1]))
        return original(self,x,y,**kwargs)
    monkeypatch.setattr(Pipeline,'fit',tracked)
    config=replace(registered_e03_config(0,workers=1),micro_positive_middle_fraction=None)
    with threadpool_limits(limits=2):
        t,v=build_candidate_ledgers(enhanced,config,{f'person{i}' for i in range(6)},{'person6','person7'})
    assert stats['batches']['candidate_train']['paired']==8
    assert seen and all(width==(87 if macro else 47) for macro,width in seen)
    assert t.features.shape[1]==v.features.shape[1]==116


def test_slice_removes_held_out_subject_from_every_train_input():
    sliced = slice_outer_train(_data(), {"person0", "person1"})
    assert {x.sid for x in sliced.window_train.windows} == {"sid0", "sid1"}
    assert {x.sid for x in sliced.candidate_train.windows} == {"sid0", "sid1"}
    assert {x.sid for x in sliced.micro_window_train.windows} == {"sid0", "sid1"}
    assert {x.sid for x in sliced.micro_candidate_train.windows} == {"sid0", "sid1"}
    assert [x.sid for x in sliced.train_truths] == ["sid0"]
    assert set(sliced.subject_by_session) == {"sid0", "sid1"}


def test_inner_partitions_are_subject_disjoint_and_cover_every_subject():
    folds = inner_subject_partitions(_data(), n_splits=4)
    assert len(folds) == 4
    assert set.union(*(set(score) for _, score in folds)) == {f"person{i}" for i in range(8)}
    assert all(not set(train) & set(score) for train, score in folds)


def test_slice_rejects_unknown_subject():
    with pytest.raises(ValueError, match="unknown"):
        slice_outer_train(_data(), {"not-present"})


def test_candidate_producer_rejects_overlapping_subject_ancestry():
    config = replace(registered_e03_config(0, workers=1), micro_positive_middle_fraction=None)
    with pytest.raises(ValueError, match="overlap"):
        build_candidate_ledgers(_data(), config, {"person0", "person1"}, {"person1"})


def test_candidate_producer_never_uses_score_truth_for_features():
    data = _data()
    config = replace(registered_e03_config(0, workers=1), micro_positive_middle_fraction=None)
    train = {f"person{i}" for i in range(6)}
    score = {"person6", "person7"}
    audit = {}
    left_t, left_v = build_candidate_ledgers(data, config, train, score, fit_audit=audit)
    assert len([k for k in audit if k.endswith('refit')]) == 2
    assert set(audit['macro_refit']['fit_subjects']) == train
    assert set(audit['threshold']['fit_subjects']) == {'person0'}
    from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
    assert_no_heldout_ancestors('candidates', score, audit)
    changed = replace(data, train_truths=tuple(truth for truth in data.train_truths if truth.sid != "sid7"))
    right_t, right_v = build_candidate_ledgers(changed, config, train, score)
    assert left_t.fit_subjects == tuple(sorted(train))
    assert left_v.score_subjects == tuple(sorted(score))
    assert not set(left_v.fit_subjects) & set(left_v.score_subjects)
    assert left_v.candidates == right_v.candidates
    np.testing.assert_array_equal(left_v.features, right_v.features)
    assert left_v.micro_threshold == right_v.micro_threshold


def test_training_only_ledger_matches_oof_without_validation_refit():
    data = _data()
    config = replace(registered_e03_config(0, workers=1), micro_positive_middle_fraction=None)
    train = {f'person{i}' for i in range(6)}
    reference, _ = build_candidate_ledgers(data, config, train, {'person6', 'person7'})
    audit = {}
    actual, validation = build_candidate_ledgers(data, config, train, (), fit_audit=audit)
    assert validation is None
    assert not any(name.endswith('refit') for name in audit)
    assert actual.candidates == reference.candidates
    np.testing.assert_array_equal(actual.features, reference.features)
    np.testing.assert_array_equal(actual.labels, reference.labels)
    assert actual.micro_threshold == reference.micro_threshold
    assert all(set(node['fit_subjects']) <= train for node in audit.values())


def test_explicit_macro_factory_matches_default_without_touching_micro():
    from src.pipeline import runner
    from threadpoolctl import threadpool_limits
    data=_data();config=replace(registered_e03_config(0,workers=1),micro_positive_middle_fraction=None)
    train={f'person{i}' for i in range(6)};score={'person6','person7'};calls=[]
    def factory():
        calls.append(config.seed)
        return runner._window_estimator(config.seed)
    with threadpool_limits(limits=2):
        a,b=build_candidate_ledgers(data,config,train,score)
        x,y=build_candidate_ledgers(data,config,train,score,macro_estimator_factory=factory)
    assert len(calls)==config.inner_splits+1
    assert a.candidates==x.candidates and b.candidates==y.candidates
    np.testing.assert_array_equal(a.features,x.features)
    np.testing.assert_array_equal(b.features,y.features)
    assert a.micro_threshold==x.micro_threshold


def test_macro_weights_are_subject_sliced_then_fold_sliced_and_never_reach_micro(monkeypatch):
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    data=_data();batch=data.window_train
    data=replace(data,window_train=replace(batch,features=np.column_stack([np.arange(8),np.zeros(8)])))
    config=replace(registered_e03_config(0,workers=1),micro_positive_middle_fraction=None)
    seen=[];original=Pipeline.fit
    def tracked(self,x,y,**kwargs):
        macro=isinstance(self.named_steps['model'],HistGradientBoostingClassifier)
        seen.append((macro,np.asarray(x)[:,0].copy(),kwargs))
        return original(self,x,y,**kwargs)
    monkeypatch.setattr(Pipeline,'fit',tracked)
    with threadpool_limits(limits=2):
        build_candidate_ledgers(data,config,{f'person{i}' for i in range(6)},{'person6','person7'},
            macro_fit_sample_weight=np.arange(1,9,dtype=np.float64))
    assert sum(macro for macro,_,_ in seen)==config.inner_splits+1
    for macro,rows,kwargs in seen:
        if macro: np.testing.assert_array_equal(kwargs['model__sample_weight'],rows+1)
        else: assert not kwargs


def test_custom_micro_selector_uses_only_train_oof_and_none_is_legacy():
    from src.pipeline import runner
    from threadpoolctl import threadpool_limits
    data=_data();config=replace(registered_e03_config(0,workers=1),micro_positive_middle_fraction=None)
    train={f'person{i}' for i in range(6)};score={'person6','person7'};seen=[];audit={}
    def select(windows,truths,grid,micro_config):
        assert set(windows)<={'sid'+str(i) for i in range(6)}
        assert {t.sid for t in truths}=={'sid0'}
        seen.append(True)
        return runner.select_micro_candidate_threshold(windows,truths,grid,micro_config)
    with threadpool_limits(limits=2):
        a,b=build_candidate_ledgers(data,config,train,score)
        x,y=build_candidate_ledgers(data,config,train,score,micro_threshold_selector=select,fit_audit=audit)
    assert seen==[True] and a.candidates==x.candidates and b.candidates==y.candidates
    np.testing.assert_array_equal(a.features,x.features);np.testing.assert_array_equal(b.features,y.features)
    assert set(audit['threshold']['fit_subjects'])==train
