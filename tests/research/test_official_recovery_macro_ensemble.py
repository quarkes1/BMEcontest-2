import numpy as np
import pytest


def test_macro_soft_voter_is_equal_probability_mean_of_independent_members():
    from scripts.research.official_recovery.macro_lgbm_event import macro_ensemble_factory
    from threadpoolctl import threadpool_limits
    x=np.random.default_rng(2).normal(size=(320,63));y=(x[:,0]+x[:,1]>0).astype(int)
    with threadpool_limits(limits=2):
        model=macro_ensemble_factory(20260910).fit(x,y)
        expected=sum(m.predict_proba(x) for m in model.estimators_)/2
        np.testing.assert_allclose(model.predict_proba(x),expected,rtol=0,atol=0)
    assert model.voting=='soft' and model.weights is None and model.n_jobs==1
    assert len(model.estimators_)==2 and model.estimators_[0] is not model.estimators_[1]
    assert all(m.named_steps['model'].random_state==20260910 for m in model.estimators_)


def test_ensemble_gate_practical_gain_does_not_relax_legacy_m1_gate():
    from scripts.research.official_recovery.macro_lgbm_event import event_gate
    rows=[dict(fold=i,events={'A':dict(tp=20,predictions=30,truths=n),
        'G':dict(tp=20,predictions=28,truths=n)}) for i,n in enumerate((54,55,51))]
    assert event_gate(rows,'t24','ensemble')['passed']
    assert not event_gate(rows,'t24','lgbm')['passed']
    with pytest.raises(ValueError): event_gate(rows,'t24','unknown')


def test_ensemble_gate_rejects_cheap_recall_with_excess_fp():
    from scripts.research.official_recovery.macro_lgbm_event import event_gate
    rows=[dict(fold=i,events={'A':dict(tp=5,predictions=10,truths=n),
        'G':dict(tp=8,predictions=18,truths=n)}) for i,n in enumerate((54,55,51))]
    assert event_gate(rows,'t24','ensemble')['delta_f1']>.01
    assert not event_gate(rows,'t24','ensemble')['passed']


def test_ensemble_fivefold_parent_rejects_m1_or_failed_ensemble(tmp_path):
    from scripts.research.official_recovery import macro_lgbm_event as probe
    from scripts.research.official_recovery.negative_clean_probe import _write
    parent=tmp_path/probe.ENSEMBLE_T24;parent.mkdir(parents=True)
    _write(parent/'plan.json',{'id':'MACRO-LGBM-EVENT-M1','cohort':'t24','hypothesis':'lgbm'})
    _write(parent/'state.json',{'state':'COMPLETE'});_write(parent/'summary.json',{'passed':True})
    with pytest.raises(ValueError): probe.verify_event_parent(tmp_path,'ensemble')
    _write(parent/'plan.json',{'id':'MACRO-SOFT-ENSEMBLE-V1','cohort':'t24','hypothesis':'ensemble'})
    _write(parent/'summary.json',{'passed':False})
    with pytest.raises(ValueError): probe.verify_event_parent(tmp_path,'ensemble')
    _write(parent/'summary.json',{'passed':True})
    probe.verify_event_parent(tmp_path,'ensemble')


def test_ensemble_requires_the_exact_failed_m1_not_just_a_snapshot(tmp_path,monkeypatch):
    from scripts.research.official_recovery import macro_lgbm_event as probe
    from scripts.research.official_recovery.negative_clean_probe import _write
    parent=tmp_path/probe.T24_OUTPUT;parent.mkdir(parents=True)
    _write(parent/'plan.json',{'id':'MACRO-LGBM-EVENT-M1','cohort':'t24','protected':{},'source_sha256':{}})
    monkeypatch.setattr(probe,'M1_PARENT_PLAN_SHA',probe._sha(parent/'plan.json'),raising=False)
    rows=[]
    for i,(a,g,n) in enumerate(((30,33,54),(30,31,55),(29,23,51))):
        row={'fold':i,'events':{'A':{'tp':a,'predictions':(47,53,51)[i],'truths':n},
            'G':{'tp':g,'predictions':(53,53,41)[i],'truths':n}}}
        rows.append(row);_write(parent/f'fold{i}_report.json',row)
    _write(parent/'state.json',{'state':'COMPLETE'})
    _write(parent/'summary.json',{'state':'COMPLETE','protected_unchanged':True,**probe.fusion_gate(rows,'t24')})
    probe.verify_failed_m1(tmp_path,{})
    _write(parent/'state.json',{'state':'INCOMPLETE'})
    with pytest.raises(ValueError): probe.verify_failed_m1(tmp_path,{})
    _write(parent/'state.json',{'state':'COMPLETE'})
    with pytest.raises(ValueError): probe.verify_failed_m1(tmp_path,{'changed':'artifact'})
    rows[0]['events']['G']['tp']+=1;_write(parent/'fold0_report.json',rows[0])
    with pytest.raises(ValueError): probe.verify_failed_m1(tmp_path,{})
