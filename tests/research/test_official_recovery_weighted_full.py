import numpy as np
import pytest


def test_full_negative_weights_preserve_canonical_sample_mass_per_session():
    from scripts.research.official_recovery.weighted_full_negatives import full_negative_weights
    from src.pipeline.micro_cache import sample_training_rows
    y=np.array([1,0,0,0,0,-1,0,0,0,1,1,-1]);s=np.array(['a']*6+['b']*3+['c']*3)
    weights,stats=full_negative_weights(y,s)
    selected=sample_training_rows(y,s,20260901,negative_ratio=3)
    assert weights.dtype==np.float64 and np.all(weights>0)
    for sid in set(s):
        for label in (0,1):
            assert weights[(s==sid)&(y==label)].sum()==pytest.approx(np.count_nonzero((s[selected]==sid)&(y[selected]==label)))
    assert weights[1]==.75 and weights[6]==pytest.approx(1/3)
    assert stats['effective_positive']==3 and stats['effective_negative']==4
    assert stats['fit_rows']==10 and stats['ignored_rows']==2
    perm=np.random.default_rng(1).permutation(len(y))
    reordered,_=full_negative_weights(y[perm],s[perm])
    np.testing.assert_array_equal(reordered,weights[perm])


def test_full_negative_weights_reject_invalid_labels_and_session_alignment():
    from scripts.research.official_recovery.weighted_full_negatives import full_negative_weights
    for y,s in (([2],['a']),([1],[]),([0],['']),([[0]],['a'])):
        with pytest.raises(ValueError): full_negative_weights(y,s)


def test_weighted_hgb_uses_effective_prior_not_raw_negative_count():
    from scripts.research.official_recovery.weighted_full_negatives import full_negative_weights
    from src.pipeline import runner
    from threadpoolctl import threadpool_limits
    y=np.r_[np.ones(2,dtype=int),np.zeros(98,dtype=int)]
    w,_=full_negative_weights(y,np.array(['s']*100))
    with threadpool_limits(limits=2):
        model=runner._window_estimator(1).set_params(model__max_iter=1).fit(np.ones((100,63)),y,model__sample_weight=w)
        p=model.predict_proba(np.ones((1,63)))[0,1]
    assert p==pytest.approx(2/8,abs=1e-7) and abs(p-.02)>.2


def test_full_weight_contract_has_its_own_parent_and_preserves_legacy_gate():
    from scripts.research.official_recovery.macro_lgbm_event import macro_contract,event_gate
    from src.pipeline import runner
    c=macro_contract('full_weight')
    assert c['id']=='WEIGHTED-FULL-NEGATIVES-V1' and c['factory'] is runner._window_estimator
    assert c['t24'].endswith('weighted_full_negative_v1_t24')
    assert macro_contract('lgbm')['id']=='MACRO-LGBM-EVENT-M1'
    rows=[dict(fold=i,events={'A':dict(tp=20,predictions=30,truths=n),'G':dict(tp=20,predictions=28,truths=n)}) for i,n in enumerate((54,55,51))]
    assert event_gate(rows,'t24','full_weight')['passed']
    assert not event_gate(rows,'t24','lgbm')['passed']
