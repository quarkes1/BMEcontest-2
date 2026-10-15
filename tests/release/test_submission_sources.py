"""Source delivery and consolidated dependency notice checks."""
import json
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_consolidated_notice_retains_complete_license_texts(tmp_path):
    from scripts.build_competition_exe import copy_licenses
    from scripts.submission_sources import consolidate_notices
    copy_licenses(tmp_path/'licenses')
    originals = {p.relative_to(tmp_path/'licenses').as_posix(): p.read_text(encoding='utf-8')
                 for p in (tmp_path/'licenses').rglob('*') if p.is_file()}
    consolidate_notices(tmp_path)
    assert not (tmp_path/'licenses').exists()
    text = (tmp_path/'THIRD_PARTY_NOTICES.txt').read_text(encoding='utf-8')
    for name, content in originals.items():
        assert name in text
        assert content in text


def test_current_source_export_runs_without_repository_path(tmp_path):
    from scripts.submission_sources import add_sources
    add_sources(ROOT, tmp_path, '160afaf81debf1ee')
    assert (tmp_path/'src/pipeline/inference/official_runner.py').is_file()
    assert not (tmp_path/'src/pipeline/challenger').exists()
    assert not (tmp_path/'scripts/research').exists()
    assert not list(tmp_path.rglob('__pycache__'))
    assert (tmp_path/'scripts/evaluate_event_stack.py').is_file()
    forbidden = re.compile(r'\b(?:agent|subagent|codex|claude|chatgpt|terra|luna)\b|提示词|主代理|子代理', re.I)
    for path in tmp_path.rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.md', '.json', '.txt'):
            assert not forbidden.search(path.read_text(encoding='utf-8-sig')), path
    check = subprocess.run([sys.executable, '-I', str(tmp_path/'main.py'), '--self-check'],
                           cwd=tmp_path.parent, capture_output=True, text=True, timeout=120)
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)['run_key'] == '160afaf81debf1ee'
    verify = subprocess.run([sys.executable, '-I', str(tmp_path/'scripts/reproduce_release.py')],
                            cwd=tmp_path.parent, capture_output=True, text=True, timeout=120)
    assert verify.returncode == 0, verify.stderr
    for name in ('evaluate_event_stack.py', 'train_event_stack.py', 'build_micro_features.py'):
        help_result = subprocess.run([sys.executable, '-I', str(tmp_path/'scripts'/name), '--help'],
                                     cwd=tmp_path.parent, capture_output=True, text=True, timeout=120)
        assert help_result.returncode == 0, help_result.stderr
