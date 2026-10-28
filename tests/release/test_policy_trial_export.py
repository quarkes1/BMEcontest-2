"""Independent policy trials preserve the accepted four model files."""
from pathlib import Path
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[2]


def test_policy_trial_paths_are_fixed_and_not_formal_submission(tmp_path):
    from scripts.research.official_recovery.policy_trial_export import trial_paths
    p=trial_paths(tmp_path,'deploy_f1')
    assert p['version']=='official_deployment_f1_policy_v3'
    assert p['bundle']==tmp_path/'models/event_stack/official_deployment_f1_policy_v3/deployment'
    assert p['exe_directory']==tmp_path/'dist/official_trials/deployment_f1_policy_v3'
    with pytest.raises(ValueError): trial_paths(tmp_path,'../../submission')


def test_failed_export_preflight_records_incomplete_without_publishing(tmp_path):
    import json
    from scripts.research.official_recovery.policy_trial_export import export,trial_paths
    with pytest.raises(Exception): export(tmp_path,'deploy_f1')
    paths=trial_paths(tmp_path,'deploy_f1')
    assert json.loads((paths['export_report']/'state.json').read_text())['state']=='INCOMPLETE'
    assert not paths['bundle'].exists()


def test_missing_ancestor_that_is_a_dangling_link_is_rejected(tmp_path,monkeypatch):
    from scripts.research.official_recovery import policy_trial_export as delivery
    # Windows may deny symlink creation; exercise the link-detection boundary directly.
    link=tmp_path/'models'
    assert not link.exists()
    monkeypatch.setattr(delivery,'_is_link_or_reparse_point',lambda path:Path(path)==link)
    with pytest.raises(ValueError,match='linked'):
        delivery.ensure_new(tmp_path,link/'event_stack/new/deployment')


def test_policy_copy_preserves_models_and_schema_byte_for_byte_and_refuses_overwrite(tmp_path):
    import json
    from scripts.research.official_recovery.policy_trial_export import copy_policy_bundle,trial_paths
    from scripts.research.official_recovery.scope_audit import _sha
    source='models/event_stack/official_consistent_refit_v1/deployment'
    report='outputs/official_recovery/consistent_refit_v1/summary.json'
    if not (ROOT/source).is_dir(): pytest.skip('accepted trial model unavailable')
    shutil.copytree(ROOT/source,tmp_path/source)
    (tmp_path/report).parent.mkdir(parents=True);shutil.copyfile(ROOT/report,tmp_path/report)
    original=json.loads((tmp_path/source/'policy.json').read_text())
    bundle=copy_policy_bundle(tmp_path,'deploy_f1',.45,dict(thresholds=[.3,.4,.45,.5,.6],evidence_kind='TEST'))
    assert bundle==trial_paths(tmp_path,'deploy_f1')['bundle']
    for name in ('macro.joblib','micro.joblib','verifier_logistic.joblib','verifier_lgbm.joblib','feature_schema.json'):
        assert _sha(bundle/name)==_sha(tmp_path/source/name)
    policy=json.loads((bundle/'policy.json').read_text())
    assert policy==dict(original,threshold=.45)
    assert json.loads((bundle/'run_config.json').read_text())['trial_version']=='official_deployment_f1_policy_v3'
    with pytest.raises(FileExistsError): copy_policy_bundle(tmp_path,'deploy_f1',.44,{})
    with pytest.raises(ValueError): copy_policy_bundle(tmp_path,'deploy_f15',.9,{})
    assert json.loads((tmp_path/source/'policy.json').read_text())==original


@pytest.mark.parametrize('mode,threshold',[('deploy_f1',.4582664763345077),('deploy_f15',.26383368823631487)])
def test_completed_diagnostic_has_recomputed_gate_and_training_threshold_median(mode,threshold):
    import json
    import numpy as np
    from scripts.research.official_recovery.policy_trial_export import validated_diagnostic
    stem='f1' if mode=='deploy_f1' else 'f15'
    if not (ROOT/f'outputs/official_recovery/deployment_{stem}_policy_v3_five_attempt2/summary.json').is_file():
        pytest.skip('completed policy diagnostic unavailable')
    folder=ROOT/f'outputs/official_recovery/deployment_{stem}_policy_v3_five_attempt2'
    if (ROOT/'release/final_submission.json').is_file():
        # Final submission replaced the formerly protected historical EXE by
        # explicit release selection. Old experiment exports must now refuse,
        # while their immutable scores and training thresholds remain evidence.
        from scripts.research.official_recovery.negative_clean_probe import protected_sources
        from scripts.research.official_recovery.relative_score_probe import gate
        before=json.loads((folder/'plan.json').read_text())['protected']
        current=protected_sources(ROOT)
        assert set(current)==set(before)
        assert {p for p in current if current[p]!=before[p]}=={'dist/submission/EatingDetection.exe'}
        assert current['dist/submission/EatingDetection.exe']=='fa6987b6244fe73c76ee741ea352fcc74e8e4de06de332812e99b1f1d963aa44'
        with pytest.raises(ValueError,match='protected first/formal trial changed'):
            validated_diagnostic(ROOT,mode)
        records=[json.loads((folder/f'fold{k}_report.json').read_text()) for k in range(5)]
        result=gate(records,'five',mode)
        assert result['passed'] is True
        count=result['aggregate']['G']
        assert (count['tp'],count['fp'],count['fn'])==((133,59,121) if mode=='deploy_f1' else (139,79,115))
        thresholds=[json.loads((folder/f'fold{k}_policy.json').read_text())['policy']['threshold'] for k in range(5)]
        assert float(np.median(thresholds))==threshold
        return
    data=validated_diagnostic(ROOT,mode)
    assert data['threshold']==threshold and len(data['thresholds'])==5
    assert data['diagnostic']['passed'] and data['diagnostic']['actual_exe_cv'] is False
