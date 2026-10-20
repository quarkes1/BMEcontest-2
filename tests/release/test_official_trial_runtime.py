"""EXE-only acceptance for the independently versioned consistent refit."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

ROOT=Path(__file__).resolve().parents[2]
EXE=ROOT/'dist/official_trials/consistent_refit_v1/EatingDetection.exe'
BUNDLE=ROOT/'models/event_stack/official_consistent_refit_v1/deployment'


def invoke(exe,cwd,*args):
    from scripts.build_competition_exe import clean_environment
    return subprocess.run([str(exe),*args],cwd=cwd,env=clean_environment(),capture_output=True,
        text=True,encoding='utf-8',errors='replace',timeout=300)


def metadata(data):
    data.mkdir(parents=True,exist_ok=True)
    (data/'userinfobean.txt').write_text('externalid,height,weight,age,gender\nHNU00001,172,68,29,男\n',encoding='utf-8')


@pytest.mark.skipif(not EXE.is_file(),reason='independent trial EXE not built')
def test_trial_exe_only_contract_and_real_multifile_parity(tmp_path):
    from src.pipeline.inference import Predictor
    from src.pipeline.inference.official_runner import run
    from src.pipeline.inference.official_api.output import validate_result
    from src import config
    from scripts.research.official_recovery.scope_audit import _sha
    start=time.perf_counter()
    room=tmp_path/'中文 独立 EXE';room.mkdir();exe=room/EXE.name;shutil.copyfile(EXE,exe)
    neutral=tmp_path/'无仓库工作目录';neutral.mkdir()
    probe=invoke(exe,neutral,'--self-check')
    assert probe.returncode==0,probe.stderr
    assert json.loads(probe.stdout)['run_key']=='official_consistent_refit_v1'
    missing=invoke(exe,neutral)
    assert missing.returncode==2 and not (room/'results.csv').exists()
    data=room/'ACC_test_data';metadata(data)
    (room/'results.csv').write_text('invalid old output')
    empty=invoke(exe,neutral)
    assert empty.returncode==0,empty.stderr
    assert validate_result(room/'results.csv')==0
    preserved=(room/'results.csv').read_bytes()
    (data/'userinfobean.txt').write_text('invalid_header\ninvalid_row\n')
    malformed=invoke(exe,neutral)
    assert malformed.returncode==2
    assert (room/'results.csv').read_bytes()==preserved
    metadata(data)
    person=data/'hnu 00001';person.mkdir()
    # Fixed consecutive original files of one person, not an accuracy or tuning set.
    sids=['sensorData-1784557876964-97236e0d0dda409f90e8088c128caa77',
          'sensorData-1784557693483-00f4c85af3dc4b1fa4e745661d472c8e']
    source_files=[]
    for sid in sids:
        paths=sorted((config.SENSOR_DIR/sid).glob('collect_data*.txt'))
        if not paths: pytest.skip('original raw parity fixtures unavailable')
        source_files.append(paths[0]);shutil.copyfile(paths[0],person/paths[0].name)
    (person/'empty.txt').write_text('')
    python=run(room,BUNDLE,predictor=Predictor.from_bundle(BUNDLE,run_key='official_consistent_refit_v1',device='cpu'))
    expected=(room/'results.csv').read_bytes()
    assert python['input_files']==3 and python['subjects']==1
    assert validate_result(room/'results.csv')>0, 'real fixture must exercise nonempty output'
    result=invoke(exe,neutral)
    assert result.returncode==0,result.stderr
    assert (room/'results.csv').read_bytes()==expected
    actual=json.loads(result.stdout.strip().splitlines()[-1])
    assert actual==python and not (neutral/'results.csv').exists()
    with (room/'results.csv').open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.reader(handle))
    assert rows[0]==['externalid','beforeTime','afterTime']
    assert all(row[0]=='HNU00001' and int(row[2])>int(row[1]) for row in rows[1:])
    report=dict(state='PASS',exe_sha256=_sha(EXE),self_check='PASS',missing_input='PASS',empty_output='PASS',
        malformed_metadata_preserves_previous_output='PASS',real_python_exe_csv_byte_parity='PASS',
        multi_file_same_person='PASS',real_fixture_sha256={p.name:_sha(p) for p in source_files},
        result=actual,csv_sha256=_sha(room/'results.csv'),elapsed_seconds=time.perf_counter()-start,
        environment='same Windows machine; EXE-only copy, neutral cwd, development PATH/PYTHON/CONDA stripped')
    destination=os.environ.get('BME_TRIAL_ACCEPTANCE_REPORT')
    if destination:
        path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


@pytest.mark.skipif(not EXE.is_file(),reason='independent trial EXE not built')
def test_trial_official_provided_example_if_available(tmp_path):
    from src.pipeline.inference.official_api.output import validate_result
    source=os.environ.get('BME_OFFICIAL_SAMPLE_ROOT')
    if not source: pytest.skip('official provided sample no longer available locally; set BME_OFFICIAL_SAMPLE_ROOT')
    shutil.copyfile(EXE,tmp_path/EXE.name);shutil.copytree(source,tmp_path/'ACC_test_data')
    result=invoke(tmp_path/EXE.name,tmp_path)
    assert result.returncode==0,result.stderr
    assert validate_result(tmp_path/'results.csv')==0
