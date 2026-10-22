from pathlib import Path

import pytest


def test_policy_build_requires_exact_export_identity_and_all_model_hashes(tmp_path):
    import json
    from scripts.research.official_recovery.policy_trial_exe import validated_export
    from scripts.research.official_recovery.policy_trial_export import trial_paths
    p=trial_paths(tmp_path,'deploy_f1');p['export_report'].mkdir(parents=True)
    (p['export_report']/'summary.json').write_text(json.dumps(dict(state='COMPLETE',version='WRONG')))
    with pytest.raises(ValueError): validated_export(tmp_path,'deploy_f1')


def test_policy_build_is_nonoverwriting_and_does_not_target_formal_submission(tmp_path):
    from scripts.research.official_recovery.policy_trial_exe import build
    from scripts.research.official_recovery.policy_trial_export import trial_paths
    p=trial_paths(tmp_path,'deploy_f1');p['exe_directory'].mkdir(parents=True)
    (p['exe_directory']/'existing.exe').write_bytes(b'keep')
    with pytest.raises(FileExistsError): build(tmp_path,'deploy_f1')
    assert (p['exe_directory']/'existing.exe').read_bytes()==b'keep'
    assert not (tmp_path/'dist/submission').exists()


def test_policy_acceptance_refuses_missing_trial_binary_without_creating_cache(tmp_path):
    from scripts.research.official_recovery.policy_trial_acceptance import accept
    with pytest.raises(FileNotFoundError): accept(tmp_path,'deploy_f15')
    assert not (tmp_path/'cache').exists()
