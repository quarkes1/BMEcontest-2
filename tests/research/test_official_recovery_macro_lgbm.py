import numpy as np
import pytest


def test_macro_alternative_has_fixed_capacity_without_extra_class_reweighting():
    from scripts.research.official_recovery.macro_lgbm_probe import macro_estimator
    model=macro_estimator(20260910).named_steps['model']
    p=model.get_params()
    assert p['n_estimators']==300 and p['num_leaves']==31 and p['min_child_samples']==100
    assert p['class_weight'] is None and p['n_jobs']==1 and p['random_state']==20260910


def test_sampling_only_training_old_labels_and_complete_health_scoring():
    from scripts.research.official_recovery.macro_lgbm_probe import split_rows
    pool={'subjects':np.array(['A','A','A','B','B','B']),
          'sessions':np.array(['a','a','a','b','b','b']),
          'old':np.array([1,0,-1,0,0,0]),'new':np.array([1,0,1,1,0,-1])}
    fit,score=split_rows(pool,{'A'},{'B'})
    assert set(fit)=={0,1} and score.tolist()==[3,4]
    pool['new'][:3]=-1
    np.testing.assert_array_equal(split_rows(pool,{'A'},{'B'})[0],fit)
    with pytest.raises(ValueError): split_rows(pool,{'A'},{'A'})
    with pytest.raises(ValueError): split_rows(pool,{'X'},{'B'})
    pool['old'][0]=2
    with pytest.raises(ValueError): split_rows(pool,{'A'},{'B'})


def test_macro_ap_gate_needs_three_unique_pairs_and_two_improved():
    from scripts.research.official_recovery.macro_lgbm_probe import gate
    rows=[{'fold':i,'metrics':{'A':{'ap':.3},'L':{'ap':p}}} for i,p in enumerate((.34,.34,.29))]
    assert gate(rows)['passed']
    with pytest.raises(ValueError): gate(rows[:2])
    rows[0]['fold']=1
    with pytest.raises(ValueError): gate(rows)


def test_alternative_real_fit_handles_missing_input():
    from scripts.research.official_recovery.macro_lgbm_probe import macro_estimator
    from threadpoolctl import threadpool_limits
    x=np.random.default_rng(1).normal(size=(256,63));x[0,0]=np.nan
    with threadpool_limits(limits=2):
        model=macro_estimator(20260910).fit(x,np.tile([0,1],128))
        p=model.predict_proba(x)[:,1]
    assert p.shape==(256,) and np.isfinite(p).all()
