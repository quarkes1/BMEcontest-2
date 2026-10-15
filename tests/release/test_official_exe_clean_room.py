"""ZIP-only EXE integration: no development PATH, neutral cwd, official CSV."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT/'dist/submission'


@pytest.fixture(scope='module')
def exe():
    p = PACKAGE/'EatingDetection.exe'
    if not p.is_file(): pytest.skip('build official Windows EXE first')
    return p


def invoke(exe, cwd, *args):
    from scripts.build_competition_exe import clean_environment
    return subprocess.run([str(exe),*args],cwd=cwd,env=clean_environment(),
                          capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)


def metadata(data):
    data.mkdir(parents=True,exist_ok=True)
    (data/'userinfobean.txt').write_text('externalid,height,weight,age,gender\nHNU00001,172,68,29,男\n',encoding='utf-8')


def test_zip_roundtrip_relocated_exe_and_self_check(exe, tmp_path):
    from scripts.build_competition_exe import verify_submission
    with tempfile.TemporaryDirectory(prefix='official-zip-',dir=tmp_path) as temporary:
        base = Path(temporary); archive = base/'submission.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for p in PACKAGE.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(PACKAGE).as_posix())
        relocated = base/'中文 路径'/'提交包'; relocated.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z: z.extractall(relocated)
        verify_submission(relocated)
        neutral = base/'outside'; neutral.mkdir()
        result = invoke(relocated/exe.name,neutral,'--self-check')
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['run_key'] == '160afaf81debf1ee'


def test_only_exe_runs_without_python_and_overwrites_empty_results(exe, tmp_path):
    from src.pipeline.inference.official_api.output import validate_result
    with tempfile.TemporaryDirectory(prefix='official-empty-',dir=tmp_path) as temporary:
        room = Path(temporary)/'只有 EXE'; room.mkdir()
        local_exe = room/exe.name; shutil.copyfile(exe,local_exe)
        neutral = Path(temporary)/'different cwd'; neutral.mkdir()
        metadata(room/'ACC_test_data')
        (room/'results.csv').write_text('old invalid result')
        for _ in range(2):
            result = invoke(local_exe,neutral)
            assert result.returncode == 0, result.stderr
            assert validate_result(room/'results.csv') == 0
            assert not (neutral/'results.csv').exists()
        shutil.rmtree(room/'ACC_test_data')
        failure = invoke(local_exe,neutral)
        assert failure.returncode == 2 and 'ACC_test_data' in failure.stderr
        assert validate_result(room/'results.csv') == 0


def test_official_provided_synthetic_sample(exe, tmp_path):
    from src.pipeline.inference.official_api.output import validate_result
    source = os.environ.get('BME_OFFICIAL_SAMPLE_ROOT')
    if not source: pytest.skip('set BME_OFFICIAL_SAMPLE_ROOT to the provided ACC_test_data')
    with tempfile.TemporaryDirectory(prefix='official-sample-',dir=tmp_path) as temporary:
        room = Path(temporary)/'example'; room.mkdir()
        shutil.copyfile(exe,room/exe.name)
        shutil.copytree(source,room/'ACC_test_data')
        result = invoke(room/exe.name,Path(temporary))
        assert result.returncode == 0, result.stderr
        assert validate_result(room/'results.csv') == 0


def test_nonempty_prediction_parity_against_canonical_release(exe, tmp_path):
    import src.config as config
    from src.pipeline.inference import Predictor
    from src.pipeline.inference.official_api.output import validate_result
    # Fixed existing published-positive session, not an accuracy evaluation or tuning set.
    sid = 'sensorData-1784557876964-97236e0d0dda409f90e8088c128caa77'
    paths = sorted((config.SENSOR_DIR/sid).glob('collect_data*.txt'))
    if not paths: pytest.skip('local original fixture not available')
    raw = paths[0]
    pred = Predictor.from_bundle(ROOT/'models/event_stack/160afaf81debf1ee/deployment')
    expected = pred.trace_file(raw,subject_id='HNU00001').events
    assert expected, 'fixture must exercise non-empty model output'
    with tempfile.TemporaryDirectory(prefix='official-parity-',dir=tmp_path) as temporary:
        room = Path(temporary)/'standalone'; room.mkdir()
        shutil.copyfile(exe,room/exe.name)
        data = room/'ACC_test_data'; metadata(data)
        person = data/'HNU00001'; person.mkdir()
        shutil.copyfile(raw,person/raw.name)
        (person/'empty.txt').write_text('')
        result = invoke(room/exe.name,Path(temporary))
        assert result.returncode == 0, result.stderr
        assert validate_result(room/'results.csv') == len(expected)
        with (room/'results.csv').open(encoding='utf-8-sig',newline='') as f: rows = list(csv.reader(f))
        assert rows == [['externalid','beforeTime','afterTime']]+[
            ['HNU00001',str(e['start_ms']),str(e['end_ms'])] for e in expected]
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        assert summary['input_files'] == 2
