from pathlib import Path


def _raw_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "ACC_TIME\tPPG_TIME\tGYRO_TIME\t" + "\t".join(f"v{i}" for i in range(50)) + "\n"
    values = ["1"] * 44 + ["1", "2", "3", "4", "5", "6"]
    path.write_text(
        header
        + "100\t200\t100\t" + "\t".join(values) + "\n"
        + "150\t250\t150\t" + "\t".join(values) + "\n",
        encoding="utf-8",
    )
    return path


def _constant_raw_file(path: Path, rows: int = 25_200) -> Path:
    """A real 240-second, low-information IMU session with frozen sampling rate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "ACC_TIME\tPPG_TIME\tGYRO_TIME\t" + "\t".join(f"v{i}" for i in range(50)) + "\n"
    values = "\t".join(["1"] * 44 + ["1", "2", "3", "4", "5", "6"])
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for index in range(rows):
            timestamp = 1_000 + index * 10
            handle.write(f"{timestamp}\t{timestamp}\t{timestamp}\t{values}\n")
    return path


def _gapped_raw_file(path: Path) -> Path:
    path = _raw_file(path)
    values = "\t".join(["1"] * 44 + ["1", "2", "3", "4", "5", "6"])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"10000\t10000\t10000\t{values}\n10050\t10050\t10050\t{values}\n")
    return path


def test_predictor_raw_file_needs_no_precomputed_feature_payload(tmp_path: Path):
    """Replacing raw inference with a feature-payload requirement is a regression."""
    from src.pipeline.inference import Predictor
    from src.pipeline.inference.schema import validate_prediction

    raw = _raw_file(tmp_path / "S01" / "collect_data1_2_3.txt")
    bundle = Path("dist/event_stack/bundle")
    result = Predictor.from_bundle(bundle).predict_file(raw, subject_id="fixture-subject")
    validate_prediction(result)
    assert result["model"]["run_key"] == "160afaf81debf1ee"
    assert result["input"]["source"] == str(raw)


def test_predictor_rejects_unregistered_cuda(tmp_path: Path):
    """Changing forced CUDA from refusal to silent CPU fallback is unsafe."""
    from src.pipeline.inference import Predictor

    raw = _raw_file(tmp_path / "S01" / "collect_data1_2_3.txt")
    predictor = Predictor.from_bundle(Path("dist/event_stack/bundle"))
    try:
        predictor.predict_file(raw, options=predictor.options(device="cuda"))
    except RuntimeError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("forced CUDA must be refused without an audited adapter")


def test_constant_raw_session_uses_frozen_model_imputation_without_cache_writes(tmp_path: Path):
    """Replacing bundle imputers with inference-side filling would change frozen scores."""
    from src.pipeline.inference import Predictor
    from src.pipeline.inference.schema import validate_prediction

    raw = _constant_raw_file(tmp_path / "S01" / "collect_data1_2_3.txt")
    result = Predictor.from_bundle(Path("dist/event_stack/bundle")).predict_file(raw)
    validate_prediction(result)
    assert result["diagnostics"]["coverage"] == 1.0


def test_predictor_include_flags_emit_valid_structured_debug_data(tmp_path: Path):
    """Changing include flags into untyped ad-hoc JSON would break visualization."""
    from src.pipeline.inference import PredictionOptions, Predictor
    from src.pipeline.inference.schema import validate_prediction

    raw = _gapped_raw_file(tmp_path / "S01" / "collect_data1_2_3.txt")
    result = Predictor.from_bundle(Path("dist/event_stack/bundle")).predict_file(
        raw, options=PredictionOptions(include_timeline=True, include_candidates=True)
    )
    validate_prediction(result)
    assert result["timeline"]["session_ids"] == ["collect_data1_2_3"]
    assert result["gaps"] == [{"session_id": "collect_data1_2_3", "start_ms": 150, "end_ms": 10000}]
    assert result["candidates"] == []


def test_predictor_folder_keeps_multiple_files_as_distinct_sessions(tmp_path: Path):
    """Folder inference must never merge two acquisition files into one session."""
    from src.pipeline.inference import PredictionOptions, Predictor
    from src.pipeline.inference.schema import validate_prediction

    folder = tmp_path / "S01"
    _raw_file(folder / "collect_data9_9_9.txt")
    _raw_file(folder / "collect_data1_2_3.txt")
    result = Predictor.from_bundle(Path("dist/event_stack/bundle")).predict_folder(
        folder, subject_id="person-a", options=PredictionOptions(include_timeline=True)
    )
    validate_prediction(result)
    assert result["timeline"]["session_ids"] == ["S01:collect_data1_2_3", "S01:collect_data9_9_9"]


def _regressed_raw_file(path: Path) -> Path:
    """A long session whose clock rewinds mid-file (duplicate acquisition run).

    The rewound run starts inside the interval already covered by the first run
    (120 s into a 252 s span) and extends 67 s past its end, so the two spans
    overlap in wall-clock time — the case that used to break the document's
    diagnostics/gaps invariants.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "ACC_TIME\tPPG_TIME\tGYRO_TIME\t" + "\t".join(f"v{i}" for i in range(50)) + "\n"
    values = "\t".join(["1"] * 44 + ["1", "2", "3", "4", "5", "6"])
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for index in range(25_200):
            timestamp = 1_000 + index * 10
            handle.write(f"{timestamp}\t{timestamp}\t{timestamp}\t{values}\n")
        for index in range(20_000):
            timestamp = 120_000 + index * 10
            handle.write(f"{timestamp}\t{timestamp}\t{timestamp}\t{values}\n")
    return path


def test_regressed_raw_session_yields_a_valid_document(tmp_path: Path):
    """A rewound clock splits spans; the emitted document must stay schema-valid."""
    from src.pipeline.inference import PredictionOptions, Predictor
    from src.pipeline.inference.schema import validate_prediction

    raw = _regressed_raw_file(tmp_path / "S01" / "collect_data1_2_3.txt")
    result = Predictor.from_bundle(Path("dist/event_stack/bundle")).predict_file(
        raw, options=PredictionOptions(include_timeline=True, include_candidates=True)
    )
    validate_prediction(result)
    assert 0.0 <= result["diagnostics"]["coverage"] <= 1.0
    for gap in result.get("gaps", []):
        assert gap["end_ms"] > gap["start_ms"], "gap rows must be strictly positive"
