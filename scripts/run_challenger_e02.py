"""Command boundary for the E02 Context-v2 inner gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.challenger.context_v2 import E02_ID, _validate_experiment_id, run_e02_inner_gate


def validate_e02_output(root: Path, output: Path, *, experiment_id: str = E02_ID) -> Path:
    root = Path(root).resolve()
    expected = root / "outputs" / "challenger_70" / "experiments" / _validate_experiment_id(experiment_id)
    if Path(output).resolve(strict=False) != expected:
        raise ValueError("E02 output must be exactly outputs/challenger_70/experiments/E02")
    return expected


def run_e02(root: Path, output: Path, *, phase: str, workers: int, experiment_id: str = E02_ID, derived_from: str | None = None) -> dict[str, object]:
    output = validate_e02_output(root, output, experiment_id=experiment_id)
    if phase == "inner-gate":
        return run_e02_inner_gate(root, output, workers=workers, experiment_id=experiment_id, derived_from=derived_from)
    if phase == "full":
        raise RuntimeError("E02 full outer execution is not implemented in this task")
    raise ValueError("phase must be inner-gate or full")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered E02 challenger evidence")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("inner-gate", "full"), required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--experiment-id", default=E02_ID)
    parser.add_argument("--derived-from")
    args = parser.parse_args()
    run_e02(args.root, args.output, phase=args.phase, workers=args.workers, experiment_id=args.experiment_id, derived_from=args.derived_from)


if __name__ == "__main__":
    main()
