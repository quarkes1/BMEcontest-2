"""Build the Windows executable and self-contained current-model source package."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

EXE = 'EatingDetection.exe'
MANIFEST = 'build_manifest.json'
RUNTIME_PACKAGES = ('numpy','scipy','scikit-learn','lightgbm','joblib','threadpoolctl','packaging')


def build_platform():
    # platform.machine() may be empty when the execution host strips optional
    # PROCESSOR_ARCHITECTURE env vars; sysconfig describes this interpreter ABI.
    target = sysconfig.get_platform()
    if os.name != 'nt' or target != 'win-amd64':
        raise ValueError('build the Windows x64 EXE using Windows x64 Python')
    return target


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def files(root):
    found = {}
    for p in Path(root).rglob('*'):
        if p.is_symlink(): raise ValueError('submission cannot contain symlinks')
        if p.is_file() and p.relative_to(root).as_posix() != MANIFEST:
            found[p.relative_to(root).as_posix()] = sha(p)
    return found


def write_manifest(root, metadata):
    (Path(root)/MANIFEST).write_text(json.dumps({**metadata, 'package_files': files(root)},
        ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8', newline='\n')


def verify_submission(root, *, allow_readme_changes=False):
    root = Path(root); m = json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    actual, expected = files(root), dict(m['package_files'])
    if allow_readme_changes:
        actual.pop('README.md', None); expected.pop('README.md', None)
    if actual != expected: raise ValueError('submission file set or SHA differs')
    allowed = {EXE,'README.md',MANIFEST,'licenses','official_api', 'THIRD_PARTY_NOTICES.txt',
               'main.py', 'requirements.txt', 'src', 'scripts', 'models', 'release', 'outputs'}
    if {p.name for p in root.iterdir()}-allowed or not (root/EXE).is_file() or not (root/'README.md').is_file():
        raise ValueError('unexpected submission files; no test data or results may be shipped')
    if (root/EXE).read_bytes()[:2] != b'MZ': raise ValueError('not a Windows executable')
    if m.get('source_delivery'):
        required = ('main.py', 'requirements.txt', 'src/pipeline/inference/official_runner.py',
                    'scripts/reproduce_release.py', 'scripts/evaluate_event_stack.py',
                    'THIRD_PARTY_NOTICES.txt',
                    f"models/event_stack/{m['release_run_key']}/deployment/manifest.json")
        if any(not (root/name).is_file() for name in required):
            raise ValueError('current-model source delivery is incomplete')
        if (root/'licenses').exists(): raise ValueError('use the consolidated third-party notice')
    return m


def validate_destination(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination != root/'dist'/'submission':
        raise ValueError('official builder may replace only this repository dist/submission')
    for p in (root/'dist', root/'dist'/'submission'):
        if p.is_symlink(): raise ValueError('submission target must not be a symlink')
    if (destination/'ACC_test_data').exists() or (destination/'results.csv').exists():
        raise ValueError('remove your runtime test data/results before rebuilding; they will not be deleted')
    return destination


def clean_environment(original=None):
    env = dict(os.environ if original is None else original)
    for key in list(env):
        if key.upper().startswith(('PYTHON','CONDA','_CONDA','VIRTUAL_ENV')): env.pop(key)
    windows = env.get('SystemRoot', r'C:\Windows')
    env['PATH'] = windows+'\\System32;'+windows
    env['PYTHONNOUSERSITE'] = '1'
    return env


def copy_licenses(target):
    target.mkdir(parents=True)
    python_license = Path(sys.base_prefix)/'LICENSE_PYTHON.txt'
    if not python_license.is_file(): raise ValueError('CPython license not found')
    def copy_text(source, dest):
        # License text must have the same hashes after Git's LF checkout.
        dest.write_bytes(source.read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n'))
    copy_text(python_license, target/'Python.txt')
    for name in (*RUNTIME_PACKAGES, 'pyinstaller'):
        dist = distribution(name); count = 0
        for relative in dist.files or []:
            parts = Path(relative).parts
            if '..' in parts: continue
            lower = str(relative).lower()
            if not any(word in Path(relative).name.lower() for word in ('license','copying','copyright','notice')): continue
            source = Path(dist.locate_file(relative))
            if not source.is_file(): continue
            dest = target/name/Path(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True); copy_text(source, dest); count += 1
        if not count: raise ValueError('third-party license missing: '+name)


def release_preflight(root):
    from src.pipeline.artifacts import load_current_promoted_release, load_event_stack_bundle
    release = load_current_promoted_release(root)
    bundle = Path(root)/'models/event_stack'/str(release['run_key'])/'deployment'
    # Bundle manifests identify their role-directory (deployment); the parent
    # experiment key is independently attested by load_current_promoted_release.
    load_event_stack_bundle(bundle, expected_role='deployment', expected_run_key='deployment')
    return release, bundle


def build(root=ROOT, destination=None, *, reuse_exe=False):
    from scripts.build_inference_distribution import _atomic_replace
    from scripts.submission_sources import add_sources, consolidate_notices
    root = Path(root).resolve(); dest = validate_destination(root, destination or root/'dist/submission')
    build_platform()
    if version('pyinstaller') != '6.22.3' or version('pyinstaller-hooks-contrib') != '2026.8':
        raise ValueError('install scripts/requirements-build.txt in the frozen release environment')
    release, bundle = release_preflight(root); run_key = str(release['run_key'])
    previous = None
    if reuse_exe:
        previous = verify_submission(dest, allow_readme_changes=True)
        if previous['release_run_key'] != run_key or previous['deployment_sha256'] != {
                p.name:sha(p) for p in bundle.iterdir() if p.is_file()}:
            raise ValueError('cannot reuse executable from another model')
        runtime_sources = {name: value for name, value in previous['source_sha256'].items()
                           if name.startswith('src/') or name == 'scripts/competition_entry.py'}
        if not runtime_sources or any(not (root/name).is_file() or sha(root/name) != value
                                      for name, value in runtime_sources.items()):
            raise ValueError('runtime source changed; rebuild executable without --reuse-exe')
    api = root/'src/pipeline/inference/official_api'
    provenance = json.loads((api/'provenance.json').read_text(encoding='utf-8'))
    for name, expected in provenance['api_sha256_lf'].items():
        if hashlib.sha256((api/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != expected:
            raise ValueError('official API source differs: '+name)
    if shutil.disk_usage(root).free < 4*1024**3: raise ValueError('need at least 4GiB free for build')
    parent = root/'cache'/'delivery_build'; parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='official-exe-', dir=parent))
    staging = dest.parent/f'.submission.staging-{uuid.uuid4().hex}'
    staging.mkdir(parents=True)
    logs = root/'outputs/delivery/official_20261015'; logs.mkdir(parents=True, exist_ok=True)
    try:
        args = [sys.executable,'-m','PyInstaller','--noconfirm','--clean','--onefile','--console','--noupx',
                '--name','EatingDetection','--paths',str(root),'--distpath',str(staging),
                '--workpath',str(work/'build'),'--specpath',str(work), '--add-data',str(bundle)+':deployment',
                '--add-data',str(root/'release/event_stack_incumbent.json')+':release']
        for module in ('sklearn.ensemble._hist_gradient_boosting.gradient_boosting', 'sklearn.linear_model._logistic',
                       'sklearn.impute._base','sklearn.preprocessing._data','sklearn.pipeline','sklearn._loss.loss',
                       'lightgbm.sklearn'):
            args += ['--hidden-import',module]
        for module in ('torch','tensorflow','matplotlib','pandas','IPython','pytest','tkinter', 'sklearn.tests','scipy.tests','numpy.tests'):
            args += ['--exclude-module',module]
        args += [str(root/'scripts/competition_entry.py')]
        env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(work/'config'))
        env.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
        if reuse_exe:
            shutil.copyfile(dest/EXE, staging/EXE)
        else:
            print('Building standalone Windows EXE...', flush=True)
            result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, errors='replace', timeout=1800)
            (logs/'pyinstaller.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
            if result.returncode: raise RuntimeError('PyInstaller failed; see outputs/delivery/official_20261015/pyinstaller.log')
        # Keep submission-specific README edits; the template serves new exports.
        readme = dest/'README.md' if (dest/'README.md').is_file() else root/'scripts/official_submission_readme.md'
        shutil.copyfile(readme, staging/'README.md')
        copy_licenses(staging/'licenses')
        consolidate_notices(staging)
        add_sources(root, staging, run_key)
        (staging/'official_api').mkdir()
        for name in ('input.py','output.py','provenance.json'):
            text = (api/name).read_text(encoding='utf-8')
            (staging/'official_api'/name).write_text(text,encoding='utf-8',newline='\n')
        neutral = work/'neutral'; neutral.mkdir()
        probe = subprocess.run([str(staging/EXE),'--self-check'], cwd=neutral, env=clean_environment(),
                               capture_output=True, text=True, errors='replace', timeout=180)
        (logs/'self_check.log').write_text(probe.stdout+'\n'+probe.stderr,encoding='utf-8')
        if probe.returncode or json.loads(probe.stdout.strip().splitlines()[-1]).get('run_key') != run_key:
            raise RuntimeError('clean-room EXE self-check failed; old submission retained')
        source_files = sorted(p for tree in ('src', 'scripts') for p in (staging/tree).rglob('*.py'))
        revision = (subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
                    if shutil.which('git') else subprocess.CompletedProcess([], 1, '', ''))
        metadata = dict(format='official-20261015-windows-x64-onefile', release_run_key=run_key,
            source_delivery=True, executable_reused=bool(reuse_exe),
            release_development_f1=release['aggregate']['outer_metrics']['f1'], no_new_f1=True, python=platform.python_version(),
            build_tool={'pyinstaller': version('pyinstaller'),'hooks':version('pyinstaller-hooks-contrib')},
            dependencies={n:version(n) for n in RUNTIME_PACKAGES},
            code_commit=revision.stdout.strip() if revision.returncode == 0 else 'source-package',
            executable_code_commit=(previous.get('executable_code_commit', previous['code_commit'])
                                    if previous else (revision.stdout.strip() if revision.returncode == 0 else 'source-package')),
            source_sha256={p.relative_to(staging).as_posix():sha(p) for p in source_files},
            deployment_sha256={p.name:sha(p) for p in bundle.iterdir() if p.is_file()},
            official_api=provenance, clean_environment_self_check='PASS',
            clean_environment_scope='Same Windows machine, neutral cwd, PATH stripped of development environments; not a second physical PC')
        write_manifest(staging, metadata); verify_submission(staging)
        print(f'EXE: {(staging/EXE).stat().st_size/1024**2:.2f} MiB; self-check passed',flush=True)
        return _atomic_replace(staging, dest)
    finally:
        # Both targets were created by this invocation below validated task-owned parents.
        if work.parent != parent or not work.name.startswith('official-exe-'): raise RuntimeError('unexpected build cleanup target')
        shutil.rmtree(work)
        if staging.exists(): shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--reuse-exe',action='store_true', help='repackage a verified executable only if model and runtime sources are unchanged')
    args = parser.parse_args(argv)
    try:
        if args.verify_only: verify_submission(ROOT/'dist/submission'); print('Submission SHA and file set: OK')
        else: print(build(reuse_exe=args.reuse_exe))
        return 0
    except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as exc:
        print(f'Official EXE build refused: {exc}',file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
