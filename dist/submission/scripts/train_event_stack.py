"""Documented entry point: legal full-target training and promotion of the event stack.

Thin wrapper — all training/promotion logic lives in
``scripts/promote_event_stack.py``; this module only exists to give the
competition repository one obvious training entry point.

Usage:
  python scripts/train_event_stack.py --summary outputs/crossfit/summary_<run_key>.json
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promote_event_stack import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
