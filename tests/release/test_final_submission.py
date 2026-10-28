"""The final package cannot silently select the historical reproduction model."""
import importlib
from importlib.util import find_spec
import json
from pathlib import Path
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[2]


def api(name='final_submission_entry'):
    module='scripts.'+name
    assert find_spec(module) is not None, 'final submission entry/build module missing'
    return importlib.import_module(module)


@pytest.fixture
def package(tmp_path):
    (tmp_path/'release').mkdir()
    shutil.copyfile(ROOT/'release/final_submission.json',tmp_path/'release/final_submission.json')
    source=ROOT/'models/event_stack/official_deployment_f1_policy_v3/deployment'
    dest=tmp_path/'models/event_stack/official_deployment_f1_policy_v3/deployment'
    shutil.copytree(source,dest)
    return tmp_path


def test_selected_bundle_is_official_best_not_historical(package):
    selection,bundle=api().selected_model(package)
    assert selection['run_key']=='official_deployment_f1_policy_v3'
    assert bundle==package/'models/event_stack/official_deployment_f1_policy_v3/deployment'
    assert selection['official_metrics']['f1']==pytest.approx(.5333333333333333)


def test_missing_selection_never_falls_back_to_reproduction(package):
    (package/'release/final_submission.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)): api().selected_model(package)


def test_tampered_model_and_selection_escape_are_rejected(package):
    model=package/'models/event_stack/official_deployment_f1_policy_v3/deployment/policy.json'
    model.write_text('{}')
    with pytest.raises(ValueError): api().selected_model(package)
    pointer=package/'release/final_submission.json'
    data=json.loads(pointer.read_text());data['bundle']='../other/deployment'
    pointer.write_text(json.dumps(data))
    with pytest.raises(ValueError): api().selected_model(package)


def test_self_check_reports_selected_inference_version(package,capsys):
    assert api().main(['--self-check'],base=package)==0
    result=json.loads(capsys.readouterr().out)
    assert result['run_key']=='official_deployment_f1_policy_v3'
    assert result['device']=='cpu'


def test_final_binary_source_is_the_exact_official_best():
    source=api('build_final_submission').final_executable(ROOT)
    from scripts.build_competition_exe import sha
    assert sha(source)=='fa6987b6244fe73c76ee741ea352fcc74e8e4de06de332812e99b1f1d963aa44'


def test_final_binary_rejects_different_executable(package):
    (package/'EatingDetection.exe').write_bytes(b'MZwrong model')
    with pytest.raises(ValueError): api('build_final_submission').final_executable(package)


def test_model_copy_preserves_json_line_endings_and_binary_bytes(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'policy.json').write_bytes(b'{\r\n  "threshold": 0.4582664763345077\r\n}\r\n')
    (source/'macro.joblib').write_bytes(b'\x00\xff\r\nmodel')
    target=tmp_path/'copied'
    api('build_final_submission').copy_selected_bundle(source,target)
    assert (target/'policy.json').read_bytes()==b'{\r\n  "threshold": 0.4582664763345077\r\n}\r\n'
    assert (target/'macro.joblib').read_bytes()==b'\x00\xff\r\nmodel'


def test_final_inference_implementation_matches_evaluated_executable_source():
    from scripts.build_competition_exe import sha
    evidence=json.loads((ROOT/'outputs/official_recovery/deployment_f1_policy_v3_export/summary.json').read_text())
    for name,digest in evidence['source_sha256'].items():
        if name.startswith('src/'):
            assert sha(ROOT/name)==digest, f'inference implementation changed: {name}'
