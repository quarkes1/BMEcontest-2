import csv
from datetime import datetime
from pathlib import Path

import pytest


def _event(id_: int, start_ms: int, end_ms: int) -> dict:
    return {"id": id_, "session_id": "collect_data1_2_3", "start_ms": start_ms, "end_ms": end_ms,
            "duration_s": (end_ms - start_ms) / 1000.0, "confidence": 0.9}


def test_csv_adapter_writes_four_columns_sorted_by_start(tmp_path: Path):
    from src.pipeline.inference.competition_adapter import CsvEventAdapter

    target = tmp_path / "out" / "predict.csv"
    CsvEventAdapter().dump({"events": [_event(1, 1_784_490_666_000, 1_784_490_918_000),
                                       _event(0, 1_784_468_604_500, 1_784_469_521_200)]}, target)
    with target.open(encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["start_time", "end_time", "start_ms", "end_ms"]
    assert [row[2:] for row in rows[1:]] == [["1784468604500", "1784469521200"],
                                             ["1784490666000", "1784490918000"]]
    expected = datetime.fromtimestamp(1_784_468_604.5).strftime("%Y-%m-%d %H:%M:%S") + ".500"
    assert rows[1][0] == expected


def test_resolve_output_path_uses_input_name(tmp_path: Path):
    from src.pipeline.inference.competition_adapter import resolve_output_path

    session = Path("X/sensorData-a/collect_data1_2_3.txt")
    assert resolve_output_path(session, None) == Path("predict") / "predict_collect_data1_2_3.csv"
    assert resolve_output_path(Path("X/sensorData-a"), None).name == "predict_sensorData-a.csv"
    directory = tmp_path / "designated"
    directory.mkdir()
    assert resolve_output_path(session, directory) == directory / "predict_collect_data1_2_3.csv"
    trailing = tmp_path / "trailing"
    trailing.mkdir()
    assert resolve_output_path(session, str(trailing) + "/") == trailing / "predict_collect_data1_2_3.csv"
    fresh = tmp_path / "not-created-yet"
    assert resolve_output_path(session, str(fresh) + "/") == fresh / "predict_collect_data1_2_3.csv"
    explicit = tmp_path / "named.csv"
    assert resolve_output_path(session, explicit) == explicit


def test_csv_adapter_dump_writes_into_directory_targets(tmp_path: Path):
    from src.pipeline.inference.competition_adapter import CsvEventAdapter

    directory = tmp_path / "designated" / "fold"
    directory.mkdir(parents=True)
    CsvEventAdapter().dump({"events": []}, directory)
    assert (directory / "predict_fold.csv").is_file()


def test_csv_adapter_load_accepts_sessions_and_refuses_others(tmp_path: Path):
    from src.pipeline.inference.competition_adapter import CsvEventAdapter

    adapter = CsvEventAdapter()
    session = tmp_path / "S" / "collect_data1_2_3.txt"
    session.parent.mkdir(parents=True)
    session.write_text("ACC_TIME\n", encoding="utf-8")
    assert adapter.load(session) == (session, None)
    assert adapter.load(tmp_path) == (tmp_path, None)

    other = tmp_path / "notes.txt"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a collect_data"):
        adapter.load(other)
    with pytest.raises(ValueError, match="does not exist"):
        adapter.load(tmp_path / "missing")


def test_registered_adapter_defaults_to_csv_and_keeps_unknown_names_refused():
    from src.pipeline.inference import competition_adapter as module

    assert isinstance(module.registered_adapter(), module.CsvEventAdapter)
    assert isinstance(module.registered_adapter("csv"), module.CsvEventAdapter)
    with pytest.raises(NotImplementedError, match="not registered"):
        module.registered_adapter("official")
    try:
        module.register_adapter("official", module.CsvEventAdapter())
        assert isinstance(module.registered_adapter("official"), module.CsvEventAdapter)
    finally:
        module._REGISTERED.pop("official", None)
    with pytest.raises(NotImplementedError, match="not registered"):
        module.registered_adapter("official")
