"""Gate event-stack artifact promotion on a registered aggregate experiment.

This command intentionally does not synthesize a full-target training dataset.
That would be an unreviewable leakage risk.  Task 6 must register a
``PromotionTrainer`` implementation and call :func:`promote_summary` after the
five-fold experiment has passed its strict gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as project_config
from src.pipeline.artifacts import PromotionContractError, promote_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an aggregate event-stack result before artifact promotion."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_config.ROOT_DIR / "models",
        help="models root; no directory is created until all promotion contracts pass",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # No trainer is registered at this layer.  Calling the shared contract is
        # nevertheless intentional: it verifies the strict F1 gate before any
        # output directory can be created and gives callers one authoritative
        # rejection message until Task 6 supplies legal full-target fitting.
        promote_summary(args.summary, output_root=args.output_root)
    except (OSError, ValueError, PromotionContractError) as exc:
        print(f"promotion refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
