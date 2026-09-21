"""Command boundary for the preregistered E03 hard-negative gate."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.pipeline.challenger.e03 import run_e03_inner_gate, run_e03_preflight, write_e03_preregistration

def main() -> None:
    parser = argparse.ArgumentParser(description="Run E03 challenger evidence")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--derived-from", default=None)
    parser.add_argument("--phase", choices=("prereg", "preflight", "inner-gate", "full"), required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.phase == "prereg":
        write_e03_preregistration(args.root, args.output, experiment_id=args.experiment_id, derived_from=args.derived_from)
        return
    if args.phase == "preflight":
        report = run_e03_preflight(args.root, args.output, workers=args.workers, experiment_id=args.experiment_id)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return
    if args.phase != "inner-gate":
        raise RuntimeError("E03 full outer execution is not implemented in this task")
    run_e03_inner_gate(args.root, args.output, workers=args.workers, experiment_id=args.experiment_id)

if __name__ == "__main__":
    main()
