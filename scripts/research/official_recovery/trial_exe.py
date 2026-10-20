"""Export one independent official-trial EXE, never the formal submission."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from .consistent_refit import VERSION,install_new_directory
from .scope_audit import _sha,protection
from scripts.build_competition_exe import build_platform,clean_environment,RUNTIME_PACKAGES
from src.pipeline.artifacts import load_event_stack_bundle,_is_link_or_reparse_point


def _read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def _write(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def validate_destination(root,destination):
    root,destination=Path(root).absolute(),Path(destination).absolute()
    if destination!=root/'dist/official_trials/consistent_refit_v1':
        raise ValueError('trial export may write only its independent destination')
    for path in (root,root/'dist',destination.parent,destination):
        if path.exists() and _is_link_or_reparse_point(path): raise ValueError('linked trial destination forbidden')
    if destination.exists(): raise FileExistsError('trial destination already exists; do not overwrite')
    return destination


def require_clean_tree(root):
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit source and prior evidence before exporting')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()


def source_snapshot(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():_sha(p) for tree in ('src','scripts') for p in (root/tree).rglob('*.py')}


def build_command(root,bundle,staging,work,identity):
    args=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--onefile','--console','--noupx',
        '--name','EatingDetection','--paths',str(root),'--distpath',str(staging),'--workpath',str(work/'build'),
        '--specpath',str(work),'--add-data',str(bundle)+':deployment','--add-data',str(identity)+':release']
    for module in ('sklearn.ensemble._hist_gradient_boosting.gradient_boosting','sklearn.linear_model._logistic',
        'sklearn.impute._base','sklearn.preprocessing._data','sklearn.pipeline','sklearn._loss.loss','lightgbm.sklearn'):
        args+=['--hidden-import',module]
    for module in ('torch','tensorflow','matplotlib','pandas','IPython','pytest','tkinter','sklearn.tests','scipy.tests','numpy.tests'):
        args+=['--exclude-module',module]
    return args+[str(root/'scripts/competition_entry.py')]


def build(root,training_report,output):
    root,training_report,output=map(lambda p:Path(p).resolve(),(root,training_report,output))
    destination=validate_destination(root,root/'dist/official_trials/consistent_refit_v1')
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();work=staging=None
    try:
        revision=require_clean_tree(root);sources=source_snapshot(root)
        build_platform()
        if version('pyinstaller')!='6.22.3' or version('pyinstaller-hooks-contrib')!='2026.8':
            raise ValueError('use the frozen build environment')
        trained=_read(training_report/'summary.json')
        if (trained.get('state')!='COMPLETE' or trained.get('version')!=VERSION or
            trained.get('protected_unchanged') is not True or trained.get('no_formal_promotion') is not True):
            raise ValueError('completed independent training evidence required')
        bundle=root/'models/event_stack'/VERSION/'deployment'
        before={p.name:_sha(p) for p in bundle.iterdir() if p.is_file()}
        if before!=trained['bundle_sha256']: raise ValueError('trial bundle differs from fitted evidence')
        loaded=load_event_stack_bundle(bundle,expected_role='deployment')
        if loaded.run_config.get('trial_version')!=VERSION: raise ValueError('wrong model identity')
        api=root/'src/pipeline/inference/official_api';provenance=_read(api/'provenance.json')
        import hashlib
        for name,digest in provenance['api_sha256_lf'].items():
            if hashlib.sha256((api/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
                raise ValueError('official API differs: '+name)
        protected=protection(root)
        if shutil.disk_usage(root).free<10*1024**3: raise ValueError('need 10GiB free for isolated build')
        parent=root/'cache/delivery_build';parent.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix='consistent-trial-',dir=parent))
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging=destination.parent/f'.consistent_refit_v1.staging-{uuid.uuid4().hex}';staging.mkdir()
        identity=work/'event_stack_incumbent.json'
        _write(identity,dict(run_key=VERSION,status='INDEPENDENT_OFFICIAL_TRIAL_NOT_PROMOTED'))
        arguments=build_command(root,bundle,staging,work,identity)
        env=dict(os.environ,PYINSTALLER_CONFIG_DIR=str(work/'config'));env.setdefault('PROCESSOR_ARCHITECTURE','AMD64')
        print('Building independent self-contained Windows x64 EXE',flush=True)
        result=subprocess.run(arguments,cwd=root,env=env,capture_output=True,text=True,errors='replace',timeout=1800)
        (output/'pyinstaller.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError('independent PyInstaller build failed')
        exe=staging/'EatingDetection.exe'
        if exe.read_bytes()[:2]!=b'MZ' or not 0<exe.stat().st_size<512*1024**2: raise ValueError('invalid trial executable')
        neutral=work/'neutral';neutral.mkdir()
        probe=subprocess.run([str(exe),'--self-check'],cwd=neutral,env=clean_environment(),capture_output=True,
            text=True,errors='replace',timeout=180)
        (output/'self_check.log').write_text(probe.stdout+'\n'+probe.stderr,encoding='utf-8')
        if probe.returncode or json.loads(probe.stdout.strip().splitlines()[-1]).get('run_key')!=VERSION:
            raise RuntimeError('isolated trial self-check failed')
        if (protection(root)!=protected or before!={p.name:_sha(p) for p in bundle.iterdir() if p.is_file()}
            or source_snapshot(root)!=sources):
            raise ValueError('protected release, trial model or source changed during build')
        metadata=dict(state='COMPLETE',trial_version=VERSION,formal_release_unchanged=True,no_official_score=True,
            executable=str((destination/exe.name).relative_to(root)),exe_sha256=_sha(exe),exe_bytes=exe.stat().st_size,
            training_summary_sha256=_sha(training_report/'summary.json'),bundle_sha256=before,source_sha256=sources,
            python=sys.version,build_tool=dict(pyinstaller=version('pyinstaller'),hooks=version('pyinstaller-hooks-contrib')),
            dependencies={name:version(name) for name in RUNTIME_PACKAGES},
            code_sha=revision,clean_worktree_at_start=True,
            self_check='PASS',elapsed_seconds=time.perf_counter()-start)
        if destination.exists(): raise FileExistsError('trial destination appeared during build')
        install_new_directory(staging,destination);staging=None
        _write(output/'summary.json',metadata);return metadata
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc)));raise
    finally:
        if work is not None:
            if work.parent!=root/'cache/delivery_build' or not work.name.startswith('consistent-trial-'):
                raise RuntimeError('unsafe trial build cleanup')
            shutil.rmtree(work)
        if staging is not None and staging.exists():
            if staging.parent!=destination.parent or not staging.name.startswith('.consistent_refit_v1.staging-'):
                raise RuntimeError('unsafe trial staging cleanup')
            shutil.rmtree(staging)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[3],args.training_report,args.output)),flush=True)
