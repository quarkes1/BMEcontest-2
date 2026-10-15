"""Official no-argument Windows competition runner (2026-10-15 contract)."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

from threadpoolctl import threadpool_limits

from .official_api.input import DataReader, InfoReader
from .official_api.output import HEADER, save_result, validate_result
from .predictor import Predictor


def executable_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def default_bundle() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)/"deployment"
    from src.pipeline.artifacts import load_current_promoted_release
    root = Path(__file__).resolve().parents[3]
    release = load_current_promoted_release(root)
    return root/"models"/"event_stack"/str(release["run_key"])/"deployment"


def run(base: Path, bundle: Path, *, predictor=None) -> dict:
    base = Path(base).resolve(); data = base/"ACC_test_data"
    if not data.is_dir():
        raise FileNotFoundError(f"ACC_test_data must be next to the EXE: {data}")
    info = InfoReader(data/"userinfobean.txt")
    subjects = []
    seen = set()
    for directory in sorted(p for p in data.iterdir() if p.is_dir()):
        subject = ''.join(directory.name.split()).upper()
        if not subject or subject in seen:
            raise ValueError("duplicate or empty normalized subject folder ID")
        seen.add(subject); subjects.append((directory, subject))
    if predictor is None:
        predictor = Predictor.from_bundle(bundle, device="cpu")
    summary = dict(subjects=0, input_files=0, eating_events=0, missing_info=[], conflicting_info=[])
    target = base/"results.csv"
    handle = tempfile.NamedTemporaryFile(prefix=".results-", suffix=".tmp", dir=base, delete=False)
    temporary = Path(handle.name); handle.close()
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(HEADER)
        with threadpool_limits(limits=min(8, os.cpu_count() or 1)):
            for directory, subject in subjects:
                rows = info.read_info(subject)  # Official interface; never infer hand from metadata.
                if not rows: summary['missing_info'].append(subject)
                if len(rows) > 1: summary['conflicting_info'].append(subject)
                def records():
                    for file_id, text in DataReader(directory).iter_data():
                        summary['input_files'] += 1
                        yield file_id, text
                trace = predictor.trace_texts(records(), subject_id=subject)
                for event in trace.events:
                    save_result(subject, event['start_ms'], event['end_ms'], 1, output_path=temporary)
                summary['subjects'] += 1
                print(f"{subject}: {len(trace.events)} eating events", flush=True)
        summary['eating_events'] = validate_result(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Place ACC_test_data next to EatingDetection.exe; run without arguments; writes results.csv.")
    parser.add_argument('--self-check', action='store_true', help='validate bundled release and dependencies without reading test data')
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            predictor = Predictor.from_bundle(default_bundle(), device="cpu")
            print(json.dumps(dict(status="OK", run_key=predictor._run_key, device="cpu")))
            return 0
        summary = run(executable_directory(), default_bundle())
        print(json.dumps(summary, ensure_ascii=True))
        return 0
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError, TypeError) as exc:
        print(f"Competition inference failed: {exc}. No partial results.csv was published.", file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
