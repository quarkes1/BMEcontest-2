# -*- coding: utf-8 -*-
"""Run leakage-safe nested event-stack evaluation on one or all outer folds."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as project_config
from src.pipeline.event_stack import DensityConfig
from src.pipeline.runner import (
    RunConfig,
    aggregate_fold_results,
    experiment_key,
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


def parse_verifier_c_grid(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(
            "verifier C grid must contain positive unique finite values"
        ) from exc
    if (
        not parsed
        or any(value <= 0 or not math.isfinite(value) for value in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise ValueError("verifier C grid must contain positive unique finite values")
    return parsed


def parse_probability_grid(value: str) -> tuple[float, ...]:
    message = "grid must contain unique finite probabilities in [0, 1]"
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(message) from exc
    if (
        not parsed
        or any(not math.isfinite(item) or not 0 <= item <= 1 for item in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise ValueError(message)
    return tuple(sorted(parsed))


def parse_middle_fraction(value: str) -> float | None:
    if value.strip().lower() == "none":
        return None
    message = "middle fraction must be none or a finite value in (0, 1]"
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise ValueError(message)
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
        "--verifier-features",
        choices=("probability", "raw_summary"),
        default="probability",
    )
    parser.add_argument(
        "--verifier-c-grid",
        type=parse_verifier_c_grid,
        default=(0.1,),
        help="inner-OOF LogisticRegression C values, e.g. 0.001,0.01,0.1",
    )
    parser.add_argument(
        "--force", action="store_true", help="ignore matching fold-result caches"
    )
    parser.add_argument("--micro-enabled", action="store_true")
    parser.add_argument(
        "--micro-threshold-grid", type=parse_probability_grid,
        default=(0.10, 0.20, 0.30, 0.40, 0.50),
    )
    parser.add_argument(
        "--micro-gravity-align", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--micro-positive-middle-fraction", type=parse_middle_fraction, default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda":
        raise SystemExit(
            "--device cuda is unavailable in the no-TCN foundation runner; "
            "use --device auto/cpu"
        )
    if args.micro_enabled and args.verifier_features == "raw_summary":
        raise SystemExit("--micro-enabled does not support --verifier-features raw_summary")
    fold_indices = range(5) if args.fold == "all" else (int(args.fold),)
    configs = [
        RunConfig(
            outer_fold=fold,
            inner_splits=args.inner_splits,
            no_tcn=args.no_tcn,
            workers=args.workers,
            device=args.device,
            subject_cap_grid=args.subject_cap_grid,
            verifier_feature_mode=args.verifier_features,
            verifier_c_grid=args.verifier_c_grid,
            density=DensityConfig(coverage_fix=args.coverage_fix),
            micro_enabled=args.micro_enabled,
            micro_threshold_grid=args.micro_threshold_grid,
            micro_gravity_align=args.micro_gravity_align,
            micro_positive_middle_fraction=args.micro_positive_middle_fraction,
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
            f"threshold={result.threshold:.6f} cap={cap_label} "
            f"C={result.verifier_c:g} features={result.verifier_feature_count} "
            f"[{cache_label}]"
        )
        print(f"  output: {output_path}")
    if args.fold == "all":
        summary = aggregate_fold_results(configs, results)
        summary["run_configs"] = [asdict(config) for config in configs]
        summary_path = output_directory / f"summary_{experiment_key(configs)}.json"
        write_json_atomic(summary_path, summary)
        metrics = summary["outer_metrics"]
        print(
            f"all folds: F1={metrics['f1']:.3f} "
            f"sens={metrics['sensitivity']:.3f} ppv={metrics['ppv']:.3f} "
            f"TP={metrics['n_tp']}/{metrics['n_true']} pred={metrics['n_pred']}"
        )
        print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
