"""Command boundary for the E02 Context-v2 inner gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.challenger.context_v2 import E02_ID, run_e02_inner_gate


def validate_e02_output(root: Path, output: Path) -> Path:
    root = Path(root).resolve()
    expected = root / "outputs" / "challenger_70" / "experiments" / E02_ID
    if Path(output).resolve(strict=False) != expected:
        raise ValueError("E02 output must be exactly outputs/challenger_70/experiments/E02")
    return expected


def run_e02(root: Path, output: Path, *, phase: str, workers: int) -> dict[str, object]:
    output = validate_e02_output(root, output)
    if phase == "inner-gate":
        return run_e02_inner_gate(root, output, workers=workers)
    if phase == "full":
        raise RuntimeError("E02 full outer execution is not implemented in this task")
    raise ValueError("phase must be inner-gate or full")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered E02 challenger evidence")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("inner-gate", "full"), required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    run_e02(args.root, args.output, phase=args.phase, workers=args.workers)


if __name__ == "__main__":
    main()
