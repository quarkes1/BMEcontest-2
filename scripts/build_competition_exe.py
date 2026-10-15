"""Build and atomically publish the official standalone Windows onefile submission."""
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
        if p.is_file() and p.name != MANIFEST: found[p.relative_to(root).as_posix()] = sha(p)
    return found


def write_manifest(root, metadata):
    (Path(root)/MANIFEST).write_text(json.dumps({**metadata, 'package_files': files(root)},
        ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')


def verify_submission(root):
    root = Path(root); m = json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    if files(root) != m['package_files']: raise ValueError('submission file set or SHA differs')
    allowed = {EXE,'README.md',MANIFEST,'licenses','official_api'}
    if {p.name for p in root.iterdir()}-allowed or not (root/EXE).is_file() or not (root/'README.md').is_file():
        raise ValueError('unexpected submission files; no test data or results may be shipped')
    if (root/EXE).read_bytes()[:2] != b'MZ': raise ValueError('not a Windows executable')
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
    shutil.copyfile(python_license, target/'Python.txt')
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
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest); count += 1
        if not count: raise ValueError('third-party license missing: '+name)


def build(root=ROOT, destination=None):
    from src.pipeline.artifacts import load_current_promoted_release, load_event_stack_bundle
    from scripts.build_inference_distribution import _atomic_replace
    root = Path(root).resolve(); dest = validate_destination(root, destination or root/'dist/submission')
    build_platform()
    if version('pyinstaller') != '6.22.3' or version('pyinstaller-hooks-contrib') != '2026.8':
        raise ValueError('install scripts/requirements-build.txt in the frozen release environment')
    release = load_current_promoted_release(root); run_key = str(release['run_key'])
    bundle = root/'models/event_stack'/run_key/'deployment'
    load_event_stack_bundle(bundle, expected_role='deployment', expected_run_key=run_key)
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
                '--workpath',str(work/'build'),'--specpath',str(work), '--add-data',str(bundle)+':deployment']
        for module in ('sklearn.ensemble._hist_gradient_boosting.gradient_boosting', 'sklearn.linear_model._logistic',
                       'sklearn.impute._base','sklearn.preprocessing._data','sklearn.pipeline','sklearn._loss.loss',
                       'lightgbm.sklearn'):
            args += ['--hidden-import',module]
        for module in ('torch','tensorflow','matplotlib','pandas','IPython','pytest','tkinter', 'sklearn.tests','scipy.tests','numpy.tests'):
            args += ['--exclude-module',module]
        args += [str(root/'scripts/competition_entry.py')]
        env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(work/'config'))
        env.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
        print('Building standalone Windows EXE...', flush=True)
        result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, errors='replace', timeout=1800)
        (logs/'pyinstaller.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        if result.returncode: raise RuntimeError('PyInstaller failed; see outputs/delivery/official_20261015/pyinstaller.log')
        shutil.copyfile(root/'scripts/official_submission_readme.md', staging/'README.md')
        copy_licenses(staging/'licenses')
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
        tracked = subprocess.check_output(['git','ls-files','src','scripts'],cwd=root,text=True).splitlines()
        metadata = dict(format='official-20261015-windows-x64-onefile', release_run_key=run_key,
            release_development_f1=release['f1'], no_new_f1=True, python=platform.python_version(),
            build_tool={'pyinstaller': version('pyinstaller'),'hooks':version('pyinstaller-hooks-contrib')},
            dependencies={n:version(n) for n in RUNTIME_PACKAGES},
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            source_sha256={n:sha(root/n) for n in tracked if n.endswith('.py')},
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
    args = parser.parse_args(argv)
    try:
        if args.verify_only: verify_submission(ROOT/'dist/submission'); print('Submission SHA and file set: OK')
        else: print(build())
        return 0
    except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as exc:
        print(f'Official EXE build refused: {exc}',file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
