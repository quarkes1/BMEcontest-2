"""EXE-only Chinese-path and real multifile parity acceptance for policy trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

from .negative_clean_probe import _read,_write,protected_sources
from .policy_trial_export import trial_paths,ensure_new,cleanup_owned
from .policy_trial_exe import validated_export
from .scope_audit import _sha
from scripts.build_competition_exe import clean_environment


def invoke(exe,cwd,*args):
    result=subprocess.run([str(exe),*args],cwd=cwd,env=clean_environment(),capture_output=True,
        text=True,encoding='utf-8',errors='replace',timeout=300)
    return result


def accept(root,mode):
    root=Path(root).resolve();paths=trial_paths(root,mode);exe=paths['exe_directory']/'EatingDetection.exe'
    if not exe.is_file(): raise FileNotFoundError('build the independent policy EXE first')
    exported=validated_export(root,mode);built=_read(paths['build_report']/'summary.json')
    if (built.get('state')!='COMPLETE' or built.get('version')!=paths['version'] or built['exe_sha256']!=_sha(exe)
        or built['bundle_sha256']!=exported['bundle_sha256']
        or built['export_summary_sha256']!=_sha(paths['export_report']/'summary.json')):
        raise ValueError('completed matching executable evidence required')
    output=paths['acceptance_report'];ensure_new(root,output);output.mkdir(parents=True)
    protected=protected_sources(root);started=time.perf_counter();work=None
    try:
        from src.pipeline.inference import Predictor
        from src.pipeline.inference.official_runner import run
        from src.pipeline.inference.official_api.output import validate_result
        from src import config
        scratch=root/'cache/delivery_build';ensure_new(root,scratch/('probe-'+uuid.uuid4().hex));scratch.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix='policy-accept-',dir=scratch));room=work/'中文 独立 程序';room.mkdir()
        isolated=room/exe.name;shutil.copyfile(exe,isolated);neutral=work/'neutral';neutral.mkdir()
        selfcheck=invoke(isolated,neutral,'--self-check')
        if selfcheck.returncode or json.loads(selfcheck.stdout.strip().splitlines()[-1])['run_key']!=paths['version']:
            raise RuntimeError('EXE-only identity self-check failed')
        missing=invoke(isolated,neutral)
        if missing.returncode!=2 or (room/'results.csv').exists(): raise RuntimeError('missing-input behavior differs')
        data=room/'ACC_test_data';data.mkdir()
        metadata='externalid,height,weight,age,gender\nHNU00001,172,68,29,男\n'
        (data/'userinfobean.txt').write_text(metadata,encoding='utf-8')
        empty=invoke(isolated,neutral)
        if empty.returncode or validate_result(room/'results.csv')!=0: raise RuntimeError('empty input behavior differs')
        preserved=(room/'results.csv').read_bytes();(data/'userinfobean.txt').write_text('invalid\ninvalid\n')
        invalid=invoke(isolated,neutral)
        if invalid.returncode!=2 or (room/'results.csv').read_bytes()!=preserved: raise RuntimeError('invalid input overwrote prior results')
        (data/'userinfobean.txt').write_text(metadata,encoding='utf-8');person=data/'hnu 00001';person.mkdir();fixtures={}
        for sid in ('sensorData-1784557876964-97236e0d0dda409f90e8088c128caa77',
                    'sensorData-1784557693483-00f4c85af3dc4b1fa4e745661d472c8e'):
            choices=sorted((config.SENSOR_DIR/sid).glob('collect_data*.txt'))
            if not choices: raise FileNotFoundError('required real raw parity fixture missing')
            source=choices[0];shutil.copyfile(source,person/source.name);fixtures[source.name]=_sha(source)
        (person/'empty.txt').write_text('')
        expected_summary=run(room,paths['bundle'],predictor=Predictor.from_bundle(paths['bundle'],run_key=paths['version'],device='cpu'))
        expected=(room/'results.csv').read_bytes();events=validate_result(room/'results.csv')
        if expected_summary['input_files']!=3 or expected_summary['subjects']!=1 or events<1:
            raise RuntimeError('real acceptance fixture did not exercise a nonempty multifile event')
        actual=invoke(isolated,neutral)
        (output/'exe_run.log').write_text(actual.stdout+'\n'+actual.stderr,encoding='utf-8')
        if (actual.returncode or (room/'results.csv').read_bytes()!=expected
            or json.loads(actual.stdout.strip().splitlines()[-1])!=expected_summary or (neutral/'results.csv').exists()):
            raise RuntimeError('real Python/EXE multifile CSV parity failed')
        if protected_sources(root)!=protected or _sha(exe)!=built['exe_sha256']: raise ValueError('protected files or EXE changed')
        result=dict(state='PASS',version=paths['version'],exe_sha256=_sha(exe),threshold=exported['threshold'],
            self_check='PASS',empty_input='PASS',malformed_input_preserves_output='PASS',missing_input='PASS',
            chinese_exe_only_neutral_cwd='PASS',real_multifile_python_exe_csv_byte_parity='PASS',result=expected_summary,
            fixture_sha256=fixtures,protected_unchanged=True,official_original_sample='NOT_AVAILABLE_NOT_TESTED',
            no_official_score=True,elapsed_seconds=time.perf_counter()-started)
        _write(output/'summary.json',result);return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc)));raise
    finally:
        if work is not None: cleanup_owned(work,root/'cache/delivery_build','policy-accept-')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('deploy_f1','deploy_f15'),required=True)
    args=parser.parse_args();print(accept(Path(__file__).resolve().parents[3],args.mode))
