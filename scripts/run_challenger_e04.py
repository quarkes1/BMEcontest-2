"""Run Amendment A2's immutable E04 verifier audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.challenger.verifier_audit_run import run_verifier_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run immutable E04 verifier audit")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/challenger_70/e04_verifier_audit"))
    args = parser.parse_args()
    print(run_verifier_audit(args.root, args.output))


if __name__ == "__main__":
    main()
