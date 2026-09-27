import numpy as np
from copy import deepcopy
import hashlib
import json
import torch
import pytest

from scripts.research.representation_v2.dataset import TokenBag
from scripts.research.representation_v2.transfer_probe import unitfree_bags, transfer_gate, load_source_artifact


def test_target_unitfree_bags_do_not_mutate_original_or_accept_static_gyro():
    x=np.random.default_rng(1).normal(size=(2,6,125)).astype(np.float32)
    x[1,3:]=0
    bag=TokenBag(('s',0,10000),x,np.ones((2,125),bool),np.ones(2,bool),
                 np.array([2500,7500]),0,10000,np.ones(2))
    result=unitfree_bags([bag])[0]
    assert result.token_mask.tolist()==[True,False]
    assert bag.token_mask.tolist()==[True,True]
    assert np.all(result.values[1]==0)


def test_transfer_gate_requires_multiple_folds_and_all_integrity_checks():
    metric=lambda ap,bce:{'all':{'average_precision':ap,'raw_logit_bce':bce}}
    row={'metrics':{'m':metric(.2,.4),'r':metric(.3,.35),'f':metric(.35,.34),'prior':metric(.1,.36)},
         'checks':dict.fromkeys(['provenance','finite','optimizer','resource'],True)}
    rows=[deepcopy(row) for _ in range(3)]
    assert transfer_gate(rows)['passed']
    rows[0]['checks']['provenance']=False
    assert not transfer_gate(rows)['passed']
    rows=[deepcopy(row) for _ in range(3)]
    rows[0]['metrics']['f']=metric(.29,.34)
    rows[1]['metrics']['f']=metric(.29,.34)
    assert not transfer_gate(rows)['passed']


@pytest.mark.parametrize('defect',['valid','schema','participant','target_flag','checks','missing_weights'])
def test_source_loader_rejects_self_consistent_hashes_with_invalid_contract(tmp_path,defect):
    from scripts.research.representation_v2.models import MotionBranch
    from scripts.research.representation_v2.explore import write_json
    plan={'id':'RV2-A3-S0-001','code_sha':'a'*40,'steps':5000,'target_signal_read':False,
          'outer_calls':0,'source_train':list(range(27)),'source_validation':list(range(27,34))}
    if defect=='target_flag': plan['target_signal_read']=True
    write_json(tmp_path/'plan.json',plan)
    encoder=MotionBranch().token_encoder.state_dict()
    weights={'schema':'rv2-a3-unitfree-source-v1','encoder':encoder,'random_encoder':encoder,
             'head':torch.nn.Linear(16,1).state_dict(),'provenance':{
                 'source_train':plan['source_train'],'source_validation':plan['source_validation'],
                 'target_subjects':[],'code_sha':plan['code_sha'],'steps':5000,
                 'plan_sha256':hashlib.sha256((tmp_path/'plan.json').read_bytes()).hexdigest(),
                 'normalization':'per-token-per-triad-demean-rms'}}
    if defect=='schema': weights['schema']='wrong'
    if defect=='participant': weights['provenance']['source_train']=[99]
    if defect=='missing_weights': weights.pop('random_encoder')
    torch.save(weights,tmp_path/'source_final.pt')
    expected=hashlib.sha256((tmp_path/'source_final.pt').read_bytes()).hexdigest()
    report={'status':'S0_SUPPORTS_TARGET_PROBE','passed':True,'checkpoint_sha256':expected,
            'checks':dict.fromkeys(['finite','isolation','optimizer','resource'],True),
            'outer_calls':0,'target_signal_read':False}
    if defect=='checks': report['checks']['isolation']=False
    write_json(tmp_path/'summary.json',report)
    if defect=='valid':
        assert load_source_artifact(tmp_path,expected)['schema']==weights['schema']
    else:
        with pytest.raises(ValueError): load_source_artifact(tmp_path,expected)
