"""Documented entry point: strict subject-disjoint nested-CV evaluation of the event stack.

Thin wrapper — the evaluation implementation lives in
``scripts/crossfit_event_stack.py``; this module only exists to give the
competition repository one obvious evaluation entry point.

Usage:
  python scripts/evaluate_event_stack.py --fold 0     # one outer fold
  (pass --help for the full option set)
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.crossfit_event_stack import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
