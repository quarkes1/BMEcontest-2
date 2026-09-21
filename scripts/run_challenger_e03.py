"""Command boundary for the preregistered E03 hard-negative gate."""
from __future__ import annotations
import argparse
from pathlib import Path
from src.pipeline.challenger.e03 import E03_ID, run_e03_inner_gate, write_e03_preregistration

def main() -> None:
    parser = argparse.ArgumentParser(description="Run E03 challenger evidence")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default=E03_ID)
    parser.add_argument("--derived-from", default=None)
    parser.add_argument("--phase", choices=("prereg", "inner-gate", "full"), required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.phase == "prereg":
        write_e03_preregistration(args.root, args.output, experiment_id=args.experiment_id, derived_from=args.derived_from)
        return
    if args.phase != "inner-gate":
        raise RuntimeError("E03 full outer execution is not implemented in this task")
    run_e03_inner_gate(args.root, args.output, workers=args.workers, experiment_id=args.experiment_id)

if __name__ == "__main__":
    main()
