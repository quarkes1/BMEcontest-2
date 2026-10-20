"""Independent EXE exports cannot replace the formal submission."""
from pathlib import Path
import subprocess
import pytest


def test_trial_target_is_exact_nonexisting_and_not_linked(tmp_path):
    from scripts.research.official_recovery.trial_exe import validate_destination
    allowed=tmp_path/'dist/official_trials/consistent_refit_v1'
    assert validate_destination(tmp_path,allowed)==allowed
    with pytest.raises(ValueError): validate_destination(tmp_path,tmp_path/'dist/submission')
    with pytest.raises(ValueError): validate_destination(tmp_path,tmp_path/'dist/official_trials/other')
    allowed.mkdir(parents=True)
    with pytest.raises(FileExistsError): validate_destination(tmp_path,allowed)


def test_trial_build_command_embeds_only_the_trial_bundle_and_identity(tmp_path):
    from scripts.research.official_recovery.trial_exe import build_command
    bundle=tmp_path/'models/event_stack/official_consistent_refit_v1/deployment'
    identity=tmp_path/'work/event_stack_incumbent.json'
    args=build_command(tmp_path,bundle,tmp_path/'staging',tmp_path/'work',identity)
    data=[args[i+1] for i,x in enumerate(args[:-1]) if x=='--add-data']
    assert data==[str(bundle)+':deployment',str(identity)+':release']
    assert '--onefile' in args and '--console' in args
    assert args[-1]==str(tmp_path/'scripts/competition_entry.py')
    assert str(tmp_path/'release/event_stack_incumbent.json')+':release' not in args


def test_trial_export_requires_committed_sources(tmp_path):
    from scripts.research.official_recovery.trial_exe import require_clean_tree
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'uncommitted.py').write_text('pass')
    with pytest.raises(RuntimeError,match='commit'): require_clean_tree(tmp_path)


def test_source_snapshot_detects_imported_build_helper_changes(tmp_path):
    from scripts.research.official_recovery.trial_exe import source_snapshot
    (tmp_path/'src').mkdir();(tmp_path/'src/a.py').write_text('pass')
    (tmp_path/'scripts').mkdir();helper=tmp_path/'scripts/build_competition_exe.py';helper.write_text('first')
    first=source_snapshot(tmp_path)
    assert 'scripts/build_competition_exe.py' in first and 'src/a.py' in first
    helper.write_text('second')
    assert source_snapshot(tmp_path)!=first
