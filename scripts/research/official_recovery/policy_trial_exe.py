"""Build an EXE-only independent policy trial without altering any prior release."""
from __future__ import annotations

import argparse
import hashlib
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

from .consistent_refit import install_new_directory
from .negative_clean_probe import _read,_write,protected_sources
from .policy_trial_export import trial_paths,ensure_new,cleanup_owned
from .scope_audit import _sha
from .trial_exe import build_command,require_clean_tree,source_snapshot
from scripts.build_competition_exe import build_platform,clean_environment,RUNTIME_PACKAGES
from src.pipeline.artifacts import load_event_stack_bundle


def validated_export(root,mode):
    root=Path(root).resolve();paths=trial_paths(root,mode);summary=_read(paths['export_report']/'summary.json')
    if (summary.get('state')!='COMPLETE' or summary.get('version')!=paths['version'] or summary.get('mode')!=mode
        or summary.get('model_bytes_unchanged') is not True or summary.get('protected_unchanged') is not True
        or summary.get('no_formal_promotion') is not True): raise ValueError('completed matching policy copy required')
    bundle=paths['bundle'];actual={p.name:_sha(p) for p in bundle.iterdir() if p.is_file()}
    if actual!=summary['bundle_sha256'] or str(bundle.relative_to(root))!=summary['bundle']:
        raise ValueError('exported policy bundle changed')
    loaded=load_event_stack_bundle(bundle,expected_role='deployment')
    if loaded.run_config.get('trial_version')!=paths['version'] or loaded.policy['threshold']!=summary['threshold']:
        raise ValueError('trial model identity or final threshold differs')
    if protected_sources(root)!=summary['protected_sha256'] or source_snapshot(root)!=summary['source_sha256']:
        raise ValueError('source/protected drift since policy export')
    return summary


def build(root,mode):
    root=Path(root).resolve();paths=trial_paths(root,mode);destination=paths['exe_directory'];output=paths['build_report']
    ensure_new(root,destination);ensure_new(root,output);output.mkdir(parents=True)
    work=staging=None;started=time.perf_counter()
    try:
        revision=require_clean_tree(root);sources=source_snapshot(root);exported=validated_export(root,mode)
        protected=protected_sources(root);build_platform()
        if version('pyinstaller')!='6.22.3' or version('pyinstaller-hooks-contrib')!='2026.8':
            raise ValueError('frozen PyInstaller environment required')
        api=root/'src/pipeline/inference/official_api'
        for name,digest in _read(api/'provenance.json')['api_sha256_lf'].items():
            if hashlib.sha256((api/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
                raise ValueError('official API hash differs')
        if shutil.disk_usage(root).free<10*1024**3: raise RuntimeError('need 10GiB build space')
        scratch=root/'cache/delivery_build';ensure_new(root,scratch/('probe-'+uuid.uuid4().hex));scratch.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix='policy-trial-',dir=scratch))
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=destination.parent))
        identity=work/'event_stack_incumbent.json';_write(identity,dict(run_key=paths['version'],status='INDEPENDENT_POLICY_TRIAL_NOT_PROMOTED'))
        command=build_command(root,paths['bundle'],staging,work,identity)
        env=dict(os.environ,PYINSTALLER_CONFIG_DIR=str(work/'config'));env.setdefault('PROCESSOR_ARCHITECTURE','AMD64')
        print('Building '+paths['version'],flush=True)
        result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,errors='replace',timeout=1800)
        (output/'pyinstaller.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError('policy trial PyInstaller failed')
        exe=staging/'EatingDetection.exe'
        with exe.open('rb') as handle: magic=handle.read(2)
        if magic!=b'MZ' or not 0<exe.stat().st_size<512*1024**2: raise ValueError('invalid Windows trial executable')
        neutral=work/'neutral';neutral.mkdir()
        probe=subprocess.run([str(exe),'--self-check'],cwd=neutral,env=clean_environment(),capture_output=True,
            text=True,encoding='utf-8',errors='replace',timeout=180)
        (output/'self_check.log').write_text(probe.stdout+'\n'+probe.stderr,encoding='utf-8')
        if probe.returncode or json.loads(probe.stdout.strip().splitlines()[-1]).get('run_key')!=paths['version']:
            raise RuntimeError('isolated policy EXE self-check failed')
        if (protected_sources(root)!=protected or source_snapshot(root)!=sources
            or {p.name:_sha(p) for p in paths['bundle'].iterdir() if p.is_file()}!=exported['bundle_sha256']):
            raise ValueError('source/model/protection changed during build')
        metadata=dict(state='COMPLETE',version=paths['version'],mode=mode,code_sha=revision,source_sha256=sources,
            executable=str((destination/exe.name).relative_to(root)),exe_bytes=exe.stat().st_size,exe_sha256=_sha(exe),
            threshold=exported['threshold'],bundle_sha256=exported['bundle_sha256'],protected_sha256=protected,
            export_summary_sha256=_sha(paths['export_report']/'summary.json'),self_check='PASS',
            no_official_score=True,no_formal_promotion=True,actual_exe_cv=False,
            dependencies={name:version(name) for name in RUNTIME_PACKAGES},elapsed_seconds=time.perf_counter()-started)
        install_new_directory(staging,destination);staging=None
        _write(output/'summary.json',metadata);return metadata
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc)));raise
    finally:
        if work is not None: cleanup_owned(work,root/'cache/delivery_build','policy-trial-')
        if staging is not None: cleanup_owned(staging,destination.parent,'.'+destination.name+'.staging-')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('deploy_f1','deploy_f15'),required=True)
    args=parser.parse_args();print(build(Path(__file__).resolve().parents[3],args.mode))
