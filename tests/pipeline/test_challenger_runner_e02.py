"""CLI-level E02 guard tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


def test_e02_rejects_output_outside_the_experiment_root(tmp_path: Path):
    from scripts.run_challenger_e02 import validate_e02_output

    with pytest.raises(ValueError, match="outputs/challenger_70/experiments/E02"):
        validate_e02_output(tmp_path, tmp_path / "outputs" / "crossfit")


def test_full_phase_is_intentionally_unavailable():
    from scripts.run_challenger_e02 import run_e02

    with pytest.raises(RuntimeError, match="not implemented"):
        run_e02(Path("."), Path("outputs/challenger_70/experiments/E02"), phase="full", workers=1)


def test_script_bootstraps_project_import_from_scripts_directory():
    completed = subprocess.run(
        [sys.executable, "scripts/run_challenger_e02.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
