"""Relocated submission visualization and native launcher integration."""
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT/'dist/submission'


def start_server(command, cwd, env=None):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace')
    lines = queue.Queue()
    def read_lines():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)
    threading.Thread(target=read_lines, daemon=True).start()
    output = []
    try:
        while True:
            line = lines.get(timeout=90)
            if line is None:
                raise AssertionError('server exited before startup: '+''.join(output))
            output.append(line)
            match = re.search(r'http://127\.0\.0\.1:(\d+)/', line)
            if match:
                return process, 'http://127.0.0.1:'+match.group(1)
    except BaseException:
        process.kill(); process.wait(timeout=20)
        raise


def exercise_bridge(base):
    def request(path, body=None, headers=None):
        with urllib.request.urlopen(urllib.request.Request(base+path, data=body, headers=headers or {}), timeout=60) as r:
            return r.read()
    health = json.loads(request('/api/health'))
    assert health['status'] == 'ok' and health['run_key'] == '160afaf81debf1ee'
    assert hashlib.sha256(request('/')).digest() == hashlib.sha256((ROOT/'dist/visual/index.html').read_bytes()).digest()
    assert json.loads(request('/runtime/release-metadata.json'))['run_key'] == '160afaf81debf1ee'
    header = ['ACC_TIME','PPG_TIME','GYRO_TIME']+[f'PPG{i}' for i in range(1,45)]+['ACC_X','ACC_Y','ACC_Z','GYRO_X','GYRO_Y','GYRO_Z']
    rows = ['\t'.join(header)]
    for i in range(8):
        t = 1760000000000+i*10
        rows.append('\t'.join(map(str,[t,t,t]+[100]*44+[0,0,1,0,0,0])))
    upload = json.loads(request('/api/upload', ('\n'.join(rows)+'\n').encode(),
                               {'Content-Type':'application/octet-stream','X-Session-Name':'collect_data1_2_3.txt'}))
    result = json.loads(request('/api/analyze', json.dumps({'tokens':[upload['token']]}).encode(),
                               {'Content-Type':'application/json'}))
    assert result['prediction']['model']['run_key'] == '160afaf81debf1ee'
    assert result['prediction']['events'] == []
    assert result['sessions'][0]['motion']['manifest']['sample_count'] == 8
    assert len(request(result['sessions'][0]['motion']['bin_url'])) == 8*32


def test_source_visualization_runs_from_its_own_tree(tmp_path):
    from scripts.submission_sources import add_sources
    package = tmp_path/'中文 提交包'; package.mkdir()
    add_sources(ROOT, package, '160afaf81debf1ee')
    assert (package/'serve.py').is_file()
    assert (package/'start.bat').is_file()
    assert (package/'visual/app/package.json').is_file()
    assert not list(package.rglob('node_modules'))
    process, base = start_server([sys.executable,'-I','-B',str(package/'serve.py'),'--port','0'], tmp_path)
    try:
        exercise_bridge(base)
        assert not (package/'results.csv').exists()
    finally:
        process.kill(); process.wait(timeout=20)


def test_native_start_bat_needs_no_python_or_repository(tmp_path):
    from scripts.build_competition_exe import clean_environment
    if not (PACKAGE/'EatingDetection.exe').is_file():
        pytest.skip('build the submission EXE first')
    package = tmp_path/'独立 & 离线!'; package.mkdir()
    for name in ('EatingDetection.exe','start.bat'):
        shutil.copyfile(PACKAGE/name, package/name)
    shutil.copytree(PACKAGE/'visual', package/'visual')
    # No src, model directory, Python, Node or original parent is available.
    shell = os.environ.get('COMSPEC', r'C:\Windows\System32\cmd.exe')
    command = [shell,'/d','/c',f'"{package / "start.bat"}" --port 0 --no-open']
    process, base = start_server(command, tmp_path, clean_environment())
    try:
        exercise_bridge(base)
        assert not (package/'results.csv').exists()
    finally:
        # This test owns the cmd -> frozen executable process tree.
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
        process.wait(timeout=20)
