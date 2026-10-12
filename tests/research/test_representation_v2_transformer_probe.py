"""Execution must be sealed, non-overwriting and fail closed across stages."""
import json
from pathlib import Path

import pytest


def test_early_health_is_blocked(tmp_path):
    from scripts.research.representation_v2.transformer.probe import run_health
    with pytest.raises(RuntimeError,match='policies not frozen'):
        run_health(tmp_path)


def test_freeze_requires_all_policies_and_detects_tampering(tmp_path):
    from scripts.research.representation_v2.transformer.probe import freeze_policies, read_frozen_policies
    from scripts.research.representation_v2.transformer.contract import ARMS
    with pytest.raises(RuntimeError): freeze_policies(tmp_path)
    for i in range(3):
        (tmp_path/f'wrapper{i}_policies.json').write_text(json.dumps({'wrapper':i,'policies':{a:{'threshold':.5} for a in ARMS}}))
    freeze_policies(tmp_path)
    assert len(read_frozen_policies(tmp_path))==3
    (tmp_path/'wrapper1_policies.json').write_text('{}')
    with pytest.raises(RuntimeError,match='changed'):
        read_frozen_policies(tmp_path)


def test_phase_exception_records_incomplete_and_cannot_resume(tmp_path):
    from scripts.research.representation_v2.transformer.probe import phase,write_json
    write_json(tmp_path/'state.json',{'status':'AUDIT_PASSED'})
    with pytest.raises(ValueError,match='boom'):
        with phase(tmp_path,'AUDIT_PASSED','POLICIES_FROZEN'):
            raise ValueError('boom')
    assert json.loads((tmp_path/'state.json').read_text())['status']=='INCOMPLETE'
    with pytest.raises(RuntimeError):
        with phase(tmp_path,'AUDIT_PASSED','POLICIES_FROZEN'): pass


def test_existing_attempt_is_never_overwritten(tmp_path):
    from scripts.research.representation_v2.transformer.probe import run_manifest
    from scripts.research.representation_v2.transformer.contract import OUTPUT
    (tmp_path/OUTPUT).mkdir(parents=True)
    with pytest.raises(RuntimeError,match='exists'):
        run_manifest(tmp_path)


def test_registered_code_checksum_is_checked(tmp_path):
    from scripts.research.representation_v2.transformer.probe import verify_registered_files,write_json
    from scripts.research.representation_v2.transformer.cache import file_hash
    path=tmp_path/'module.py'; path.write_text('x=1\n')
    payload={'code_files':{'module.py':file_hash(path)}}
    verify_registered_files(tmp_path,payload)
    path.write_text('x=2\n')
    with pytest.raises(RuntimeError,match='changed'):
        verify_registered_files(tmp_path,payload)


def test_runtime_config_accepts_json_tuple_roundtrip_but_not_changed_values():
    from scripts.research.representation_v2.transformer.probe import validate_runtime_config
    from scripts.research.representation_v2.event_probe import registered_e03_config
    from dataclasses import asdict
    config=registered_e03_config(0,workers=4)
    recorded=json.loads(json.dumps(asdict(config)))
    validate_runtime_config(config,recorded)
    recorded['seed']+=1
    with pytest.raises(RuntimeError,match='config changed'):
        validate_runtime_config(config,recorded)


@pytest.mark.parametrize('kind',['neural','encoding'])
def test_current_work_is_counted_before_cumulative_limit(tmp_path,kind):
    from scripts.research.representation_v2.transformer.probe import Resources
    resources=Resources(tmp_path,tmp_path,new=True)
    limit=12*3600 if kind=='neural' else 2*3600
    resources.row[kind+'_seconds']=limit-1
    with pytest.raises(RuntimeError,match='resource budget'):
        resources.check(**{'active_'+kind+'_seconds':2.})
