import subprocess
import numpy as np
import pytest


def test_budget_boundaries():
    from scripts.research.representation_v2.ppg.audit import check_budget
    check_budget(10, 1024, 100, 30*2**30)
    for args in [(7201, 0, 0, 30*2**30), (0, 17*2**30, 0, 30*2**30),
                 (0, 0, 129*2**20, 30*2**30), (0, 0, 0, 19*2**30)]:
        with pytest.raises(RuntimeError):
            check_budget(*args)


def test_state_and_hash_mismatch(tmp_path):
    from scripts.research.representation_v2.ppg.audit import assert_state, verify_hashes
    from scripts.research.representation_v2.ppg.contract import file_hash
    with pytest.raises(RuntimeError):
        assert_state({'status': 'INCOMPLETE'}, 'REGISTERED')
    f = tmp_path/'x'; f.write_text('before')
    hashes = {'x': file_hash(f)}
    verify_hashes(tmp_path, hashes)
    f.write_text('after')
    with pytest.raises(RuntimeError):
        verify_hashes(tmp_path, hashes)


def test_require_committed(tmp_path):
    from scripts.research.representation_v2.ppg.audit import require_committed
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    f = tmp_path/'x'; f.write_text('unsealed')
    with pytest.raises(RuntimeError):
        require_committed(tmp_path, f)


def test_strict_json_and_no_overwrite(tmp_path):
    from scripts.research.representation_v2.ppg.audit import write_json, create_output
    from scripts.research.representation_v2.ppg.contract import read_json
    write_json(tmp_path/'x.json', {'x': np.nan})
    assert read_json(tmp_path/'x.json') == {'x': None}
    with pytest.raises(FileExistsError):
        create_output(tmp_path)


def test_array_contract():
    from scripts.research.representation_v2.ppg.audit import validate_arrays
    q = np.zeros((2, 9)); p = np.full((2, 6), np.nan)
    validate_arrays(q, p, 2)
    with pytest.raises(ValueError):
        validate_arrays(q[:, :8], p, 2)
    q[0, 0] = 1.01
    with pytest.raises(ValueError):
        validate_arrays(q, p, 2)
