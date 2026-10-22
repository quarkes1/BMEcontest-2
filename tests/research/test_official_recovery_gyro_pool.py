from dataclasses import replace

import numpy as np
import pytest

from src.pipeline.event_stack import EventRef
from src.pipeline.runner import WindowBatch


def test_pool_uses_only_contained_same_sid_windows_with_literal_mean_and_std():
    from scripts.research.official_recovery.gyro_pool_macro import augment_batch
    macro=WindowBatch(np.full((2,62),9.,dtype=np.float32),np.array([1,0]),
        (EventRef('a',0,100),EventRef('b',0,100)))
    windows=(EventRef('a',-1,9),EventRef('a',0,10),EventRef('a',90,100),
        EventRef('a',91,101),EventRef('b',0,10),EventRef('b',90,100))
    x=np.zeros((6,47),dtype=np.float32);x[:,35:]=np.array([100,2,6,100,10,14])[:,None]
    micro=WindowBatch(x,np.full(6,-1),windows)
    out,counts=augment_batch(macro,micro)
    assert out.features.shape==(2,86)
    np.testing.assert_array_equal(out.features[:,:62],macro.features)
    np.testing.assert_array_equal(out.features[:,62:74],np.array([[4.]*12,[12.]*12]))
    np.testing.assert_array_equal(out.features[:,74:],np.full((2,12),2.))
    np.testing.assert_array_equal(out.labels,macro.labels)
    assert counts.tolist()==[2,2] and out.windows==macro.windows
    # Different labels, input order, outside-window and other-sid values do not affect row a.
    altered=x.copy();altered[[0,3,4,5],35:]+=1000
    reverse=np.arange(5,-1,-1)
    other=replace(micro,features=altered[reverse],labels=np.zeros(6),windows=tuple(windows[i] for i in reverse))
    changed,_=augment_batch(macro,other)
    np.testing.assert_array_equal(changed.features[0],out.features[0])


def test_pool_missing_and_malformed_sources_fail_closed_without_mutation():
    from scripts.research.official_recovery.gyro_pool_macro import augment_batch
    macro=WindowBatch(np.zeros((2,62)),np.array([0,1]),(EventRef('a',0,100),EventRef('none',0,100)))
    micro=WindowBatch(np.zeros((1,47)),np.array([1]),(EventRef('a',0,100),))
    out,counts=augment_batch(macro,micro)
    assert np.isnan(out.features[:,62:]).all() and counts.tolist()==[1,0]
    assert macro.features.shape==(2,62)
    with pytest.raises(ValueError): augment_batch(out,micro)
    with pytest.raises(ValueError): augment_batch(macro,replace(micro,features=np.zeros((1,46))))
    with pytest.raises(ValueError): augment_batch(macro,replace(micro,features=np.full((1,47),np.inf)))
    with pytest.raises(ValueError): augment_batch(macro,replace(micro,features=np.zeros((2,47)),
        labels=np.ones(2),windows=micro.windows*2))


def test_pooled_matrix_uses_original_time_prior_and_contract():
    from scripts.research.official_recovery.gyro_pool_macro import augment_batch
    from scripts.research.official_recovery.macro_lgbm_event import macro_contract
    from src.pipeline import runner
    windows=(EventRef('a',0,100),)
    macro=WindowBatch(np.zeros((1,62),dtype=np.float32),np.array([1]),windows)
    micro=WindowBatch(np.ones((2,47)),np.array([-1,0]),(EventRef('a',0,40),EventRef('a',60,100)))
    out,_=augment_batch(macro,micro)
    before=runner._with_time_prior(macro.features,macro.windows)
    after=runner._with_time_prior(out.features,out.windows)
    assert after.shape==(1,87)
    np.testing.assert_array_equal(after[:,:62],before[:,:62])
    np.testing.assert_array_equal(after[:,-1],before[:,-1])
    c=macro_contract('gyro_pool')
    assert c['id']=='GYRO-POOL-MACRO-V1' and c['factory'] is runner._window_estimator
    assert c.get('verifier_fit') is None


def test_constant_high_energy_gyro_has_zero_std_even_late_in_long_session():
    from scripts.research.official_recovery.gyro_pool_macro import augment_batch
    micro=WindowBatch(np.full((1000,47),1e12,dtype=np.float32),np.zeros(1000),
        tuple(EventRef('a',i*10,(i+1)*10) for i in range(1000)))
    macro=WindowBatch(np.zeros((1,62)),np.zeros(1),(EventRef('a',9000,10000),))
    out,_=augment_batch(macro,micro)
    np.testing.assert_array_equal(out.features[:,74:],np.zeros((1,12)))
