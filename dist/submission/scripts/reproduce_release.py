"""Verify the promoted release chain without retraining anything.

Checks, in order: the incumbent registry and its promotion attestation, every
outer-fold evidence bundle and the deployment bundle, the frozen aggregate
metrics, and (when present) the built ``dist/inference`` and ``dist/submission``
distributions against their manifests.  Prints the frozen metrics for release
reporting.

Usage:
  python scripts/reproduce_release.py [--run-key 160afaf81debf1ee]

Exit codes: 0 = the release chain verifies; 2 = a verification problem was found.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.artifacts import (  # noqa: E402
    load_current_promoted_release,
    verify_bundle_manifest,
)


def verify_release_chain(root: Path, *, expected_run_key: str | None = None) -> tuple[list[str], dict[str, object]]:
    """Return (problems, summary) for the active promoted release."""
    root = Path(root)
    problems: list[str] = []
    release = load_current_promoted_release(root)
    run_key = str(release["run_key"])
    if expected_run_key is not None and run_key != expected_run_key:
        problems.append(f"active run key {run_key} does not match requested {expected_run_key}")
    run_root = root / "models" / "event_stack" / run_key
    for fold in range(5):
        bundle = run_root / f"outer-fold-{fold}"
        problems.extend(f"outer-fold-{fold}: {problem}"
                        for problem in verify_bundle_manifest(bundle))
    problems.extend(f"deployment: {problem}"
                    for problem in verify_bundle_manifest(run_root / "deployment"))
    return problems, dict(release["aggregate"])


def verify_distributions(root: Path) -> list[str]:
    """Verify built distributions when they exist; their manifests are hash ledgers."""
    from scripts.build_inference_distribution import verify_distribution_manifest

    problems: list[str] = []
    for name, entrypoint in (("inference", "predict.py"), ("submission", "main.py")):
        package = Path(root) / "dist" / name
        if not package.is_dir():
            continue
        # The submission ships its runtime in app/ (metadata in meta/); the inference
        # distribution keeps both at the package root.
        required = ("app", "meta", "models") if (package / "app" / "event_stack").is_dir() else ("event_stack", "models")
        try:
            verify_distribution_manifest(package, entrypoint=entrypoint, required_roots=required)
        except (OSError, ValueError) as exc:
            problems.append(f"dist/{name}: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", default=None, help="expected active release run key")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        problems, summary = verify_release_chain(args.root, expected_run_key=args.run_key)
        problems.extend(verify_distributions(args.root))
    except (OSError, ValueError) as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 2
    if problems:
        for problem in problems:
            print(f"release verification problem: {problem}", file=sys.stderr)
        return 2
    outer = summary["outer_metrics"]
    print(f"run key: {args.run_key or 'active'}")
    print(f"aggregate F1: {outer['f1']}")
    print(f"TP: {outer['n_tp']}")
    print(f"FP: {outer['n_pred'] - outer['n_tp']}")
    print(f"predictions: {outer['n_pred']}")
    print(f"eligible truths: {outer['n_true']}")
    print(f"recall: {outer['sensitivity']}")
    print(f"PPV: {outer['ppv']}")
    print("release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
