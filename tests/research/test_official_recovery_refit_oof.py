"""The meta learner must fit supplied OOF rows and never the heldout subject."""
import numpy as np
import pytest
from types import SimpleNamespace
from scripts.research.representation_v2.crossfit import CandidateLedger


def ledgers():
    train=CandidateLedger((),np.zeros((2,116)),np.array([0,1]),np.array(['A','A']),.1,('A',),('A',))
    score=CandidateLedger((),np.zeros((3,116)),np.array([0,0,0]),np.array(['B']*3),.1,('A',),('B',))
    return train,score


def test_oof_meta_uses_training_labels_not_scoring_labels():
    from scripts.research.official_recovery.refit_oof_probe import fit_meta
    train,score=ledgers()
    result=fit_meta(train,score,SimpleNamespace(seed=20260908))
    assert result.shape==(3,)
    np.testing.assert_allclose(result,[.5,.5,.5])


def test_oof_meta_refuses_heldout_identity_in_training_groups():
    from dataclasses import replace
    from scripts.research.official_recovery.refit_oof_probe import fit_meta
    train,score=ledgers()
    with pytest.raises(ValueError,match='heldout'):
        fit_meta(replace(train,groups=np.array(['A','B'])),score,SimpleNamespace(seed=1))
