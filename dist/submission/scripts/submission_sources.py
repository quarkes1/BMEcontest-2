"""Export current-model inference, training and evaluation source files."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import shutil


SOURCE_SCRIPTS = (
    'build_competition_exe.py', 'submission_sources.py', 'competition_entry.py',
    'build_inference_distribution.py', 'reproduce_release.py', 'build_micro_features.py',
    'evaluate_event_stack.py', 'crossfit_event_stack.py', 'train_event_stack.py',
    'promote_event_stack.py', 'release_event_stack.py', 'package_event_stack.py',
    'official_iou_eval.py', 'requirements-build.txt', 'official_submission_readme.md',
)
TEXT_SUFFIXES = {'.py', '.md', '.json', '.txt', '.bat', '.yml', '.yaml'}


def copy_file(source: Path, destination: Path):
    if source.is_symlink():
        raise ValueError('source export cannot contain symlinks')
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    if source.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    destination.write_bytes(data)


def copy_tree(source: Path, destination: Path):
    for file in sorted(source.rglob('*')):
        if file.is_symlink():
            raise ValueError('source export cannot contain symlinks')
        if file.is_file() and '__pycache__' not in file.parts and file.suffix != '.pyc':
            copy_file(file, destination/file.relative_to(source))


def consolidate_notices(package: Path):
    """Keep complete upstream notices in one file rather than a directory tree."""
    notices = package/'licenses'
    sections = ['Third-party software notices\n\nUpstream license texts are reproduced below without modification.\n\n']
    for path in sorted(notices.rglob('*')):
        if path.is_file():
            sections += ['='*72+'\n'+path.relative_to(notices).as_posix()+'\n'+'='*72+'\n',
                         path.read_text(encoding='utf-8'), '\n\n']
    if len(sections) == 1:
        raise ValueError('third-party notices are missing')
    (package/'THIRD_PARTY_NOTICES.txt').write_text(''.join(sections), encoding='utf-8', newline='\n')
    # Exact task-owned staging child, never a user directory or workspace root.
    if notices.parent != package or notices.is_symlink():
        raise ValueError('unexpected notices directory')
    shutil.rmtree(notices)


def add_sources(root: Path, package: Path, run_key: str):
    for file in sorted((root/'src').rglob('*.py')):
        if '__pycache__' not in file.parts and 'challenger' not in file.parts:
            copy_file(file, package/file.relative_to(root))
    provenance = root/'src/pipeline/inference/official_api/provenance.json'
    copy_file(provenance, package/provenance.relative_to(root))
    for name in SOURCE_SCRIPTS:
        copy_file(root/'scripts'/name, package/'scripts'/name)
    for source, target in ((root/'models/event_stack'/run_key, package/'models/event_stack'/run_key),
                           (root/'release', package/'release'),
                           (root/'outputs/crossfit', package/'outputs/crossfit')):
        copy_tree(source, target)
    # Root source entry follows the same official no-argument contract as the EXE.
    entry = '"""Official competition source entry point."""\n'
    entry += 'from pathlib import Path\nimport sys\n'
    entry += 'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
    entry += 'from scripts.competition_entry import main\n'
    entry += 'if __name__ == "__main__":\n    raise SystemExit(main())\n'
    (package/'main.py').write_text(entry, encoding='utf-8', newline='\n')
    packages = ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'threadpoolctl', 'packaging', 'pandas')
    (package/'requirements.txt').write_text(''.join(f'{name}=={version(name)}\n' for name in packages),
                                          encoding='utf-8', newline='\n')
