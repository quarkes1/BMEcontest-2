import json
from pathlib import Path
import pytest


def test_package_manifest_rejects_extra_data_or_tampered_exe(tmp_path):
    from scripts.build_competition_exe import write_manifest, verify_submission
    p = tmp_path/'submission'; p.mkdir()
    (p/'EatingDetection.exe').write_bytes(b'MZtest')
    (p/'README.md').write_text('instructions')
    write_manifest(p, {'release_run_key': 'test'})
    verify_submission(p)
    (p/'results.csv').write_text('old results')
    with pytest.raises(ValueError): verify_submission(p)
    (p/'results.csv').unlink(); (p/'EatingDetection.exe').write_bytes(b'MZchanged')
    with pytest.raises(ValueError): verify_submission(p)


def test_repack_can_preserve_only_readme_edits_not_executable_edits(tmp_path):
    from scripts.build_competition_exe import write_manifest, verify_submission
    (tmp_path/'EatingDetection.exe').write_bytes(b'MZtest')
    (tmp_path/'README.md').write_text('original')
    write_manifest(tmp_path, {})
    (tmp_path/'README.md').write_text('user edited reproduction instructions')
    with pytest.raises(ValueError): verify_submission(tmp_path)
    verify_submission(tmp_path, allow_readme_changes=True)
    (tmp_path/'EatingDetection.exe').write_bytes(b'MZtampered')
    with pytest.raises(ValueError): verify_submission(tmp_path, allow_readme_changes=True)


def test_nested_manifest_name_cannot_escape_file_ledger(tmp_path):
    from scripts.build_competition_exe import write_manifest, verify_submission
    (tmp_path/'EatingDetection.exe').write_bytes(b'MZtest')
    (tmp_path/'README.md').write_text('instructions')
    (tmp_path/'licenses').mkdir()
    write_manifest(tmp_path, {})
    (tmp_path/'licenses/build_manifest.json').write_text('unexpected untracked file')
    with pytest.raises(ValueError): verify_submission(tmp_path)


def test_license_bytes_survive_git_lf_checkout(tmp_path):
    from scripts.build_competition_exe import copy_licenses
    copy_licenses(tmp_path/'licenses')
    for p in (tmp_path/'licenses').rglob('*'):
        if p.is_file(): assert b'\r' not in p.read_bytes(), p


def test_reused_runtime_comparison_accepts_only_line_ending_normalization(tmp_path):
    import hashlib
    from scripts.build_competition_exe import reusable_runtime_matches
    source = tmp_path/'src/example.py'; source.parent.mkdir()
    source.write_bytes(b'value = 1\r\n')
    metadata = {'source_delivery': True, 'source_sha256': {
        'src/example.py': hashlib.sha256(b'value = 1\n').hexdigest()}}
    assert reusable_runtime_matches(tmp_path, metadata)
    source.write_bytes(b'value = 2\r\n')
    assert not reusable_runtime_matches(tmp_path, metadata)
    assert not reusable_runtime_matches(tmp_path, {'source_sha256': {}})


def test_atomic_publish_initial_rename_failure_never_removes_existing_package(tmp_path, monkeypatch):
    from scripts.build_inference_distribution import _atomic_replace
    import os
    destination = tmp_path/'submission'; destination.mkdir()
    (destination/'README.md').write_text('user content must survive')
    staging = tmp_path/'.submission.staging-test'; staging.mkdir()
    (staging/'README.md').write_text('new generated content')
    original = os.replace
    def locked_old_directory(source, target):
        if Path(source) == destination:
            raise PermissionError('Windows directory is temporarily locked')
        return original(source, target)
    monkeypatch.setattr(os, 'replace', locked_old_directory)
    with pytest.raises(PermissionError):
        _atomic_replace(staging, destination)
    assert (destination/'README.md').read_text() == 'user content must survive'


def test_builder_refuses_workspace_root_or_external_destination_before_build(tmp_path):
    from scripts.build_competition_exe import validate_destination
    (tmp_path/'dist').mkdir()
    assert validate_destination(tmp_path,tmp_path/'dist'/'submission') == tmp_path/'dist'/'submission'
    for p in (tmp_path, tmp_path/'dist', tmp_path/'models', tmp_path.parent/'outside'):
        with pytest.raises(ValueError): validate_destination(tmp_path,p)


def test_clean_environment_keeps_system_dll_search_but_no_python_path():
    from scripts.build_competition_exe import clean_environment
    e = clean_environment({'SystemRoot': r'C:\Windows', 'PATH': r'D:\Anaconda3;C:\Windows',
                           'PYTHONPATH': 'repo', 'PYTHONHOME':'python', 'CONDA_PREFIX':'env'})
    assert e['PATH'] == r'C:\Windows\System32;C:\Windows'
    assert not any(k in e for k in ('PYTHONPATH','PYTHONHOME','CONDA_PREFIX'))


def test_official_api_matches_recorded_source_hashes():
    import hashlib
    root = Path(__file__).resolve().parents[2]
    provenance = json.loads((root/'src/pipeline/inference/official_api/provenance.json').read_text())
    for name, expected in provenance['api_sha256_lf'].items():
        data = (root/'src/pipeline/inference/official_api'/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(data).hexdigest() == expected


def test_compiled_python_platform_not_optional_architecture_env(monkeypatch):
    from scripts.build_competition_exe import build_platform
    monkeypatch.delenv('PROCESSOR_ARCHITECTURE', raising=False)
    monkeypatch.delenv('PROCESSOR_ARCHITEW6432', raising=False)
    assert build_platform() == 'win-amd64'
    import sysconfig
    monkeypatch.setattr(sysconfig, 'get_platform', lambda: 'win-arm64')
    with pytest.raises(ValueError): build_platform()


def test_release_preflight_distinguishes_role_and_experiment_key():
    from scripts.build_competition_exe import release_preflight
    root = Path(__file__).resolve().parents[2]
    release, bundle = release_preflight(root)
    assert release['run_key'] == '160afaf81debf1ee'
    assert bundle.name == 'deployment'
    assert release['aggregate']['outer_metrics']['f1'] == pytest.approx(.6514285714285714)


def test_release_verifier_handles_official_exe_and_detects_tampering(tmp_path):
    from scripts.build_competition_exe import write_manifest
    from scripts.reproduce_release import verify_distributions
    p = tmp_path/'dist/submission'; p.mkdir(parents=True)
    (p/'EatingDetection.exe').write_bytes(b'MZtest')
    (p/'README.md').write_text('instructions')
    write_manifest(p, {'release_run_key':'test'})
    assert verify_distributions(tmp_path) == []
    (p/'EatingDetection.exe').write_bytes(b'MZchanged')
    assert any('dist/submission:' in problem for problem in verify_distributions(tmp_path))
