# -*- coding: utf-8 -*-
"""Run leakage-safe nested event-stack evaluation on one or all outer folds."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as project_config
from src.pipeline.event_stack import DensityConfig
from src.pipeline.runner import (
    RunConfig,
    fold_result_to_dict,
    run_folds,
    write_json_atomic,
)


def parse_subject_cap_grid(value: str) -> tuple[int, ...]:
    stripped = value.strip()
    if not stripped:
        return ()
    try:
        parsed = tuple(int(item.strip()) for item in stripped.split(","))
    except ValueError as exc:
        raise ValueError("subject cap grid must contain positive unique integers") from exc
    if any(item < 1 for item in parsed) or len(set(parsed)) != len(parsed):
        raise ValueError("subject cap grid must contain positive unique integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested subject-disjoint window and event verification evaluation."
    )
    parser.add_argument(
        "--fold",
        choices=("0", "1", "2", "3", "4", "all"),
        default="all",
        help="outer fold to run (default: all)",
    )
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument(
        "--coverage-fix",
        action="store_true",
        help="use timestamp-local coverage and five coverage verifier features",
    )
    parser.add_argument(
        "--no-tcn",
        action="store_true",
        default=True,
        help="run the CPU-only feature stack (currently required)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="fold processes; 0 uses up to the physical-core/fold minimum",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--subject-cap-grid",
        type=parse_subject_cap_grid,
        default=(),
        help="inner-OOF per-subject event caps, e.g. 2,3,4,5,6",
    )
    parser.add_argument(
        "--force", action="store_true", help="ignore matching fold-result caches"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda":
        raise SystemExit(
            "--device cuda is unavailable in the no-TCN foundation runner; "
            "use --device auto/cpu"
        )
    fold_indices = range(5) if args.fold == "all" else (int(args.fold),)
    configs = [
        RunConfig(
            outer_fold=fold,
            inner_splits=args.inner_splits,
            no_tcn=args.no_tcn,
            workers=args.workers,
            device=args.device,
            subject_cap_grid=args.subject_cap_grid,
            density=DensityConfig(coverage_fix=args.coverage_fix),
        )
        for fold in fold_indices
    ]
    results = run_folds(configs, workers=args.workers, force=args.force)
    output_directory = project_config.OUTPUT_DIR / "crossfit"
    for config, result in zip(configs, results):
        output_path = output_directory / (
            f"fold{config.outer_fold}_{result.config_hash}.json"
        )
        payload = fold_result_to_dict(result)
        payload["run_config"] = asdict(config)
        write_json_atomic(output_path, payload)
        metrics = result.outer_metrics
        cache_label = "cache" if result.cache_hits.get("fold_result") else "trained"
        cap_label = (
            str(result.max_events_per_subject)
            if result.max_events_per_subject is not None
            else "none"
        )
        print(
            f"fold {config.outer_fold}: F1={metrics.f1:.3f} "
            f"sens={metrics.sensitivity:.3f} ppv={metrics.ppv:.3f} "
            f"TP={metrics.n_tp}/{metrics.n_true} pred={metrics.n_pred} "
            f"threshold={result.threshold:.6f} cap={cap_label} [{cache_label}]"
        )
        print(f"  output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
