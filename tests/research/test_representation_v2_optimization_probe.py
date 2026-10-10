from copy import deepcopy
import hashlib
import json
import numpy as np
import pytest
import torch

from scripts.research.representation_v2 import optimization_probe as op


def metric(bce):
    return {'subject_weighted_bce':bce,'subject_bce':{str(i):bce for i in range(11)}}


def test_registered_preference_is_not_best_training_loss_search():
    initial=metric(.22)
    arms={'unclipped':{'metrics':metric(.20),'source_bce':.24},'larger_batch':{'metrics':metric(.10),'source_bce':.22}}
    assert op.select_optimizer_arm(initial,arms,.22)['chosen']=='unclipped'
    arms['unclipped']['metrics']=metric(.215)
    assert op.select_optimizer_arm(initial,arms,.22)['chosen']=='larger_batch'


def test_rejects_forgetting_or_nonmajority_even_if_train_average_better():
    arms={'unclipped':{'metrics':metric(.20),'source_bce':.26},'larger_batch':{'metrics':metric(.20),'source_bce':.22}}
    for i in range(6):arms['larger_batch']['metrics']['subject_bce'][str(i)]=.23
    assert op.select_optimizer_arm(metric(.22),arms,.22)['chosen'] is None
    arms=deepcopy(arms);arms['unclipped']['metrics']['subject_weighted_bce']=float('nan')
    with pytest.raises(ValueError):op.select_optimizer_arm(metric(.22),arms,.22)


def test_source_bce_weights_subject_and_class_not_raw_row_count():
    class FirstSample(torch.nn.Module):
        def forward(self,x,mask):return x[:,0,:1].expand(-1,16)
    head=torch.nn.Linear(16,1)
    with torch.no_grad():head.weight.zero_();head.weight[0,0]=1;head.bias.zero_()
    x=np.zeros((5,6,125),np.float32);x[:,0,0]=[0,0,2,1,1]
    bank={'values':x,'labels':np.array([0,0,1,0,1]),'groups':np.array([0,0,0,1,1])}
    assert op.source_train_bce(FirstSample(),head,bank,'cpu')==pytest.approx(.6116496416598409,abs=1e-7)


def test_event_loader_rejects_modified_pilot_metrics_or_incomplete_state(tmp_path):
    initial=metric(.22);arms={n:{'metrics':metric(.20),'source_bce':.22} for n in op.OPTIONS}
    comparison={'initial':initial,'initial_source_bce':.22,'arms':arms}
    plan={'id':'RV2-A3-S10-001','options':op.OPTIONS,'train_subjects':list(initial['subject_bce']),
          'steps_per_arm':1000,'source_weight':.25,'seed':20261011,'source_checkpoint_sha256':op.ep.REGISTERED_SOURCE_SHA}
    for name,d in [('plan.json',plan),('train_comparison.json',comparison)]:op.ep.write_json(tmp_path/name,d)
    model_hashes={}
    for n in op.OPTIONS:
        path=tmp_path/(n+'_adaptation.json');op.ep.write_json(path,{'fixture':'identity-only'})
        model_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    selection={**op.select_optimizer_arm(initial,arms,.22),'id':'RV2-A3-S10-001','health_files_read':0,'outer_calls':0,
        'plan_sha256':hashlib.sha256((tmp_path/'plan.json').read_bytes()).hexdigest(),
        'comparison_sha256':hashlib.sha256((tmp_path/'train_comparison.json').read_bytes()).hexdigest(),
        'model_file_sha256':model_hashes}
    op.ep.write_json(tmp_path/'selection.json',selection)
    state={'status':'S10_OPTIMIZER_SELECTED','chosen':'unclipped','protected_unchanged':True,'resource_ok':True,'health_files_read':0,'outer_calls':0}
    op.ep.write_json(tmp_path/'state.json',state)
    assert op.load_pilot_selection(tmp_path)['chosen']=='unclipped'
    state['status']='S10_INCOMPLETE';op.ep.write_json(tmp_path/'state.json',state)
    with pytest.raises(RuntimeError):op.load_pilot_selection(tmp_path)
    state['status']='S10_OPTIMIZER_SELECTED';op.ep.write_json(tmp_path/'state.json',state)
    comparison['arms']['unclipped']['source_bce']=.5;op.ep.write_json(tmp_path/'train_comparison.json',comparison)
    with pytest.raises(RuntimeError,match='comparison'):op.load_pilot_selection(tmp_path)
