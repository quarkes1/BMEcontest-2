"""Final selected source and EXE agree on real multi-file input outside the repo."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.build_competition_exe import clean_environment, sha

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'dist/submission'


def test_final_source_and_exe_only_have_real_multifile_csv_byte_parity(tmp_path):
    import src.config as config
    from src.pipeline.inference.official_api.output import validate_result
    from scripts.build_final_submission import EXPECTED_EXE, verify_final
    verify_final(PACKAGE)
    room=tmp_path/'中文 最终源代码';shutil.copytree(PACKAGE,room)
    alone=tmp_path/'只有 最终程序';alone.mkdir()
    shutil.copyfile(PACKAGE/'EatingDetection.exe',alone/'EatingDetection.exe')
    neutral=tmp_path/'其他工作目录';neutral.mkdir()
    data=room/'ACC_test_data';data.mkdir()
    (data/'userinfobean.txt').write_text('externalid,height,weight,age,gender\nHNU00001,172,68,29,男\n',encoding='utf-8')
    person=data/'hnu 00001';person.mkdir()
    for sid in ('sensorData-1784557876964-97236e0d0dda409f90e8088c128caa77',
                'sensorData-1784557693483-00f4c85af3dc4b1fa4e745661d472c8e'):
        source=sorted((config.SENSOR_DIR/sid).glob('collect_data*.txt'))[0]
        shutil.copyfile(source,person/source.name)
    (person/'empty.txt').write_text('')
    shutil.copytree(data,alone/'ACC_test_data')
    source=subprocess.run([sys.executable,'-I','-B',str(room/'main.py')],cwd=neutral,
        env=clean_environment(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
    assert source.returncode==0, source.stdout+source.stderr
    expected=(room/'results.csv').read_bytes()
    native=subprocess.run([str(alone/'EatingDetection.exe')],cwd=neutral,
        env=clean_environment(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
    assert native.returncode==0, native.stdout+native.stderr
    assert (alone/'results.csv').read_bytes()==expected
    assert validate_result(alone/'results.csv')>=1
    assert json.loads(native.stdout.strip().splitlines()[-1])==json.loads(source.stdout.strip().splitlines()[-1])
    assert json.loads(source.stdout.strip().splitlines()[-1])['input_files']==3
    assert not (neutral/'results.csv').exists()
    assert not list(room.rglob('__pycache__'))
    assert sha(alone/'EatingDetection.exe')==EXPECTED_EXE
