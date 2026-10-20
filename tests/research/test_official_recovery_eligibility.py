"""Fingerprint the actual cached arrays used by the legacy eligible-meal loader."""
import numpy as np
import pytest


def test_used_time_or_validity_changes_are_detected_but_unread_vectors_are_not(tmp_path):
    from scripts.research.official_recovery.eligibility_provenance import eligibility_inputs
    folder=tmp_path/'cache/sessions';folder.mkdir(parents=True)
    path=folder/'s.npz'
    np.savez_compressed(path,t_acc=np.array([1,2,3]),imu_valid=np.array([True,True,False]),acc=np.zeros((3,3)))
    initial=eligibility_inputs(tmp_path,['s'])
    np.savez_compressed(path,t_acc=np.array([1,2,3]),imu_valid=np.array([True,True,False]),acc=np.ones((3,3)))
    assert eligibility_inputs(tmp_path,['s'])==initial
    np.savez_compressed(path,t_acc=np.array([1,2,4]),imu_valid=np.array([True,True,False]))
    assert eligibility_inputs(tmp_path,['s'])!=initial
    np.savez_compressed(path,t_acc=np.array([1,2,3]),imu_valid=np.array([True,False,False]))
    assert eligibility_inputs(tmp_path,['s'])!=initial


def test_missing_session_is_fingerprinted_and_parent_traversal_rejected(tmp_path):
    from scripts.research.official_recovery.eligibility_provenance import eligibility_inputs
    result=eligibility_inputs(tmp_path,['missing'])
    assert result['missing']=={'present':False}
    with pytest.raises(ValueError,match='session'):
        eligibility_inputs(tmp_path,['../elsewhere'])
