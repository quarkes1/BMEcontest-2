import numpy as np
import pytest


def test_randomized_verifier_factory_is_fixed_and_only_imputes_training_rows():
    from scripts.research.official_recovery.extra_trees_verifier import estimator
    model=estimator(42)
    params=model.named_steps['model'].get_params()
    assert {k:params[k] for k in ('n_estimators','max_depth','min_samples_leaf','max_features',
        'class_weight','bootstrap','n_jobs','random_state')}==dict(n_estimators=512,max_depth=8,
        min_samples_leaf=3,max_features=.5,class_weight='balanced',bootstrap=False,n_jobs=8,random_state=42)
    x=np.array([[0.,np.nan],[1.,np.nan],[2.,np.nan],[3.,np.nan],[4.,np.nan],[5.,np.nan]])
    model.fit(x,[0,0,0,1,1,1])
    assert model.named_steps['imputer'].statistics_.tolist()==[2.5,0.]
    model.predict_proba([[1000000.,np.nan]])
    assert model.named_steps['imputer'].statistics_.tolist()==[2.5,0.]


def test_randomized_verifier_real_fit_blend_and_empty_score():
    from scripts.research.official_recovery.extra_trees_verifier import fit_verifiers
    from src.pipeline import runner
    from types import SimpleNamespace
    x=np.random.default_rng(1).normal(size=(80,6));y=(x[:,0]>0).astype(int)
    config=SimpleNamespace(seed=42)
    p,c=fit_verifiers(x,y,x[:9],config,.1,.25)
    np.testing.assert_allclose(p,.25*c['lr']+.75*c['lgbm'],rtol=0,atol=0)
    assert p.shape==(9,) and np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))
    lr=runner._verifier_estimator(45,.1).fit(x,y)
    np.testing.assert_allclose(c['lr'],runner._positive_probability(lr,x[:9]))
    empty,parts=fit_verifiers(x,y,x[:0],config,.1,.25)
    assert empty.shape==parts['lr'].shape==parts['lgbm'].shape==(0,)
    with pytest.raises(ValueError): fit_verifiers(x,np.zeros(len(x)),x[:1],config,.1,.25)
    with pytest.raises(ValueError): fit_verifiers(x,y,x[:1],config,.1,1.1)


def test_randomized_verifier_contract_retains_old_window_factory_and_other_modes():
    from scripts.research.official_recovery.macro_lgbm_event import macro_contract,event_gate
    from scripts.research.official_recovery.extra_trees_verifier import fit_verifiers
    from src.pipeline import runner
    contract=macro_contract('extra_trees')
    assert contract['id']=='EXTRA-TREES-VERIFIER-V1'
    assert contract['factory'] is runner._window_estimator and contract['member_fits']==2
    assert contract['verifier_fit'] is fit_verifiers
    for name in ('lgbm','ensemble','full_weight','full_labels'):
        assert macro_contract(name).get('verifier_fit') is None
    rows=[dict(fold=i,events={'A':dict(tp=20,predictions=30,truths=n),
        'G':dict(tp=20,predictions=28,truths=n)}) for i,n in enumerate((54,55,51))]
    assert event_gate(rows,'t24','extra_trees')['passed']
    assert not event_gate(rows,'t24','lgbm')['passed']
