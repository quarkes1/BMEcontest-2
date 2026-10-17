import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT/'models/event_stack/160afaf81debf1ee/deployment'


def raw_text(rows=5):
    header = ['ACC_TIME', 'PPG_TIME', 'GYRO_TIME']+[f'PPG{i}' for i in range(1,45)]+['ACC_X','ACC_Y','ACC_Z','GYRO_X','GYRO_Y','GYRO_Z']
    lines = ['\t'.join(header)]
    for i in range(rows):
        t = 1760000000000+i*10
        lines.append('\t'.join(map(str, [t,t,t]+[100]*44+[.01,.02,.98,.1,0,-.1])))
    return '\r\n'.join(lines)+'\r\n'


def data_root(base):
    root = base/'ACC_test_data'; root.mkdir()
    (root/'userinfobean.txt').write_text('externalid,height,weight,age,gender\nHNU00001,172,68,29,男\n',encoding='utf-8')
    return root


def test_text_parser_exactly_reuses_file_parser(tmp_path):
    from src.pipeline.io.raw_session import RawSessionSource, load_raw_session, load_raw_text
    p = tmp_path/'collect_data0_1760000000_1760000030.txt'; text = raw_text(); p.write_bytes(text.encode())
    source = RawSessionSource(p, 's', 'HNU00001')
    a, b = load_raw_session(source), load_raw_text(source, text)
    for name in ('acc','gyro','ppg','t_acc','t_ppg','imu_valid','ppg_valid'):
        assert np.array_equal(getattr(a,name),getattr(b,name))
    assert a.meta == b.meta


def test_texts_trace_parity_with_files_and_empty_input(tmp_path):
    from src.pipeline.inference import Predictor
    from src.pipeline.io.raw_session import RawSessionSource
    p = tmp_path/'collect_data0_1760000000_1760000030.txt'; text = raw_text(); p.write_text(text)
    pred = Predictor.from_bundle(BUNDLE)
    expected = pred._trace_sources([RawSessionSource(p,'HNU00001:'+p.stem,'HNU00001')])
    actual = pred.trace_texts(iter([(p.stem,text), ('empty','')]), subject_id='HNU00001')
    assert actual.events == expected.events
    assert actual.spans == expected.spans
    assert np.array_equal(actual.macro_features_63, expected.macro_features_63)
    assert pred.trace_texts(iter([]), subject_id='HNU00001').events == ()


def test_official_run_empty_predictions_and_overwrites_previous(tmp_path):
    from src.pipeline.inference.official_runner import run
    from src.pipeline.inference.official_api.output import validate_result
    root = data_root(tmp_path); person = root/'HNU00001'; person.mkdir()
    (person/'collect_data0_1760000000_1760000030.txt').write_text(raw_text())
    (person/'empty.txt').write_text('')
    (root/'HNU00002').mkdir()
    (tmp_path/'results.csv').write_text('old invalid output')
    result = run(tmp_path, BUNDLE)
    assert result['subjects'] == 2 and result['input_files'] == 2
    assert validate_result(tmp_path/'results.csv') == 0
    assert (tmp_path/'results.csv').read_text(encoding='utf-8-sig') == 'externalid,beforeTime,afterTime\n'
    run(tmp_path, BUNDLE)
    assert validate_result(tmp_path/'results.csv') == 0


def test_official_run_outputs_real_contract_and_groups_all_files(tmp_path):
    from src.pipeline.inference.official_runner import run
    from src.pipeline.inference.official_api.output import validate_result
    root = data_root(tmp_path); person = root/'hnu00001'; person.mkdir()
    for name in ('b.txt','a.txt'): (person/name).write_text(raw_text())
    class DeterministicBackend:
        def trace_texts(self, records, *, subject_id):
            files = list(records)
            if [p[0] for p in files] != ['a','b'] or subject_id != 'HNU00001':
                raise ValueError('all files must be one normalized subject inference')
            return SimpleNamespace(events=({'start_ms':1760000000000,'end_ms':1760000600000},))
    run(tmp_path, BUNDLE, predictor=DeterministicBackend())
    with (tmp_path/'results.csv').open(encoding='utf-8-sig',newline='') as f:
        assert list(csv.reader(f)) == [['externalid','beforeTime','afterTime'],['HNU00001','1760000000000','1760000600000']]
    run(tmp_path, BUNDLE, predictor=DeterministicBackend())
    assert validate_result(tmp_path/'results.csv') == 1


def test_failed_run_preserves_previous_complete_result_and_cleans_temp(tmp_path):
    from src.pipeline.inference.official_runner import run
    root = data_root(tmp_path); person = root/'HNU00001'; person.mkdir()
    (person/'bad.txt').write_text('not a sensor header')
    old = b'externalid,beforeTime,afterTime\r\n'; (tmp_path/'results.csv').write_bytes(old)
    with pytest.raises((ValueError, AssertionError)): run(tmp_path, BUNDLE)
    assert (tmp_path/'results.csv').read_bytes() == old
    assert not list(tmp_path.glob('.results-*.tmp'))


def test_missing_metadata_and_normalized_collision_refused(tmp_path):
    from src.pipeline.inference.official_runner import run
    root = data_root(tmp_path)
    (root/'HNU00001').mkdir(); (root/'HNU 00001').mkdir()
    with pytest.raises(ValueError, match='duplicate'): run(tmp_path, BUNDLE)
    (root/'userinfobean.txt').unlink()
    with pytest.raises(FileNotFoundError): run(tmp_path, BUNDLE)


def test_base_directory_uses_executable_not_cwd(monkeypatch, tmp_path):
    from src.pipeline.inference.official_runner import executable_directory
    import sys
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path/'中文路径'/'EatingDetection.exe'))
    assert executable_directory() == tmp_path/'中文路径'


def test_visual_flags_do_not_accidentally_run_official_batch():
    from src.pipeline.inference.official_runner import main
    for arguments in (['--open'], ['--port','0'], ['--visual','--port','65536']):
        with pytest.raises(SystemExit) as refused:
            main(arguments)
        assert refused.value.code == 2


def test_missing_visual_assets_fails_without_creating_official_results(monkeypatch, tmp_path):
    from src.pipeline.inference import official_runner
    monkeypatch.setattr(official_runner, 'executable_directory', lambda: tmp_path)
    assert official_runner.main(['--visual','--no-open']) == 2
    assert not (tmp_path/'results.csv').exists()
