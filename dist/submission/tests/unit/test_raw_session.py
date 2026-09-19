from pathlib import Path

import numpy as np
import pytest


def _write_collect_data(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "ACC_TIME\tPPG_TIME\tGYRO_TIME\t" + "\t".join(f"v{i}" for i in range(50)) + "\n"
    values = ["1"] * 44 + ["1", "2", "3", "4", "5", "6"]
    rows = ["100\t200\t100\t" + "\t".join(values), "150\t-1\t150\t" + "\t".join(values)]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_discover_raw_session_folder(tmp_path: Path):
    from src.pipeline.io.raw_session import RawSessionSource, discover_raw_sessions

    raw = _write_collect_data(tmp_path / "S01" / "collect_data1_2_3.txt")
    assert discover_raw_sessions(raw.parent) == (RawSessionSource(raw, "S01", None),)


def test_discover_raw_sessions_assigns_stable_unique_ids_for_multiple_files(tmp_path: Path):
    """Collapsing multi-file folders into one session would cross-contaminate events."""
    from src.pipeline.io.raw_session import discover_raw_sessions

    folder = tmp_path / "S01"
    later = _write_collect_data(folder / "collect_data9_9_9.txt")
    earlier = _write_collect_data(folder / "collect_data1_2_3.txt")
    sources = discover_raw_sessions(folder)
    assert [source.path for source in sources] == [earlier, later]
    assert [source.session_id for source in sources] == ["S01:collect_data1_2_3", "S01:collect_data9_9_9"]
    assert {source.subject_id for source in sources} == {None}


def test_directory_source_loads_legacy_sorted_first_collect_data_file(tmp_path: Path):
    from src.pipeline.io.raw_session import RawSessionSource, load_raw_session

    directory = tmp_path / "S01"
    _write_collect_data(directory / "collect_data9_9_9.txt")
    first = _write_collect_data(directory / "collect_data1_2_3.txt")
    session = load_raw_session(RawSessionSource(directory, "S01", None))
    assert session.meta["path"] == str(first)


def test_direct_source_rejects_non_collect_data_name(tmp_path: Path):
    from src.pipeline.io.raw_session import RawSessionSource, load_raw_session

    raw = _write_collect_data(tmp_path / "not_collect_data.txt")
    with pytest.raises(ValueError, match="collect_data"):
        load_raw_session(RawSessionSource(raw, "fixture", None))


def test_canonical_reader_preserves_legacy_parser_arrays(tmp_path: Path):
    from src.data.loader import load_session_tsv
    from src.pipeline.io.raw_session import RawSessionSource, load_raw_session

    raw = _write_collect_data(tmp_path / "collect_data1_2_3.txt")
    legacy = load_session_tsv(raw)
    canonical = load_raw_session(RawSessionSource(raw, "fixture", None))
    for name in ("acc", "gyro", "ppg", "t_acc", "t_ppg", "imu_valid", "ppg_valid"):
        np.testing.assert_array_equal(getattr(legacy, name), getattr(canonical, name))
    assert legacy.meta["row_rate"] == canonical.meta["row_rate"]


def test_batch_directory_with_sensordata_subdirectories_is_discovered(tmp_path):
    """The competition dataset layout (batch/sensorData-*/txt) must be accepted as input."""
    from src.pipeline.io.raw_session import discover_raw_sessions

    batch = tmp_path / "t_x_sensororiginaldata_system"
    for name in ("sensorData-aaa", "sensorData-bbb"):
        (batch / name).mkdir(parents=True)
        (batch / name / "collect_data1_2_3.txt").write_text("ACC_TIME\n", encoding="utf-8")
    assert [source.session_id for source in discover_raw_sessions(batch)] == ["sensorData-aaa", "sensorData-bbb"]
    # direct-hit behaviour is unchanged
    assert [source.session_id for source in discover_raw_sessions(batch / "sensorData-aaa")] == ["sensorData-aaa"]
