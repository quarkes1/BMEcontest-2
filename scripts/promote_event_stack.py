"""Gate event-stack artifact promotion on a registered aggregate experiment.

This command intentionally does not synthesize a full-target training dataset.
That would be an unreviewable leakage risk.  Task 6 must register a
``PromotionTrainer`` implementation and call :func:`promote_summary` after the
five-fold experiment has passed its strict gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as project_config
from src.pipeline.artifacts import (
    PROMOTION_F1_FLOOR,
    EventStackBundle,
    PromotionContractError,
    promote_summary,
)
from src.pipeline.runner import (
    FilesystemDataSource,
    RunConfig,
    _fold_result_from_dict,
    aggregate_fold_results,
    build_full_target_dataset,
    cache_key,
    experiment_key,
    fit_full_target_deployment,
    fit_outer_fold_for_promotion,
)
from src.pipeline.event_stack import DensityConfig, MicroCandidateConfig


_REGISTERED_EXPERIMENT_KEY = "a7396a9aa7c38f42"


def _canonical_mode(values: Sequence[object], name: str) -> object:
    """Equal-fold mode with a documented deterministic tie break.

    This function deliberately receives only fields selected inside each outer
    fold's training partition.  It must never be passed outer metrics.
    """

    if not values:
        raise ValueError(f"{name} needs at least one selected fold value")
    counts = Counter(values)
    maximum = max(counts.values())
    tied = [value for value, count in counts.items() if count == maximum]
    # ``None`` denotes no cap and is ordered before finite canonical values.
    return min(tied, key=lambda value: (value is not None, value))


def _canonical_median(values: Sequence[object], name: str) -> float:
    if not values:
        raise ValueError(f"{name} needs at least one selected fold value")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"{name} must contain finite selected fold values")
    return sorted(numeric)[len(numeric) // 2]


def canonical_deployment_policy(
    fold_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate frozen fold policies without inspecting validation performance.

    Fixed rule: all five folds receive equal weight; modes select discrete
    fields, the upper median selects the continuous event threshold, and ties
    choose the smallest canonical value (``None`` before integer caps).
    """

    if len(fold_records) != 5:
        raise ValueError("deployment policy needs exactly five frozen fold records")
    required = (
        "micro_threshold", "selected_blend_weight", "selected_admission_nms_iou",
        "selected_admission_threshold", "selected_admission_subject_cap",
        "threshold", "max_events_per_subject", "verifier_c",
    )
    if any(any(name not in record for name in required) for record in fold_records):
        raise ValueError("each fold record must carry every frozen train-only policy field")
    return {
        "micro_threshold": _canonical_mode([record["micro_threshold"] for record in fold_records], "micro_threshold"),
        "blend_weight": _canonical_mode([record["selected_blend_weight"] for record in fold_records], "selected_blend_weight"),
        "nms_iou": _canonical_mode([record["selected_admission_nms_iou"] for record in fold_records], "selected_admission_nms_iou"),
        "admission_threshold": _canonical_mode([record["selected_admission_threshold"] for record in fold_records], "selected_admission_threshold"),
        "max_candidates_per_subject": _canonical_mode([record["selected_admission_subject_cap"] for record in fold_records], "selected_admission_subject_cap"),
        "threshold": _canonical_median([record["threshold"] for record in fold_records], "threshold"),
        "max_events_per_group": _canonical_mode([record["max_events_per_subject"] for record in fold_records], "max_events_per_subject"),
        "verifier_c": _canonical_mode([record["verifier_c"] for record in fold_records], "verifier_c"),
        "aggregation": "equal-fold canonical median/mode; ties use the smallest canonical value",
    }


def _config_from_summary(payload: Mapping[str, object]) -> RunConfig:
    raw = dict(payload)
    try:
        raw["density"] = DensityConfig(**dict(raw["density"]))
        raw["micro_candidate"] = MicroCandidateConfig(**dict(raw["micro_candidate"]))
        for name in (
            "subject_cap_grid", "verifier_c_grid", "micro_threshold_grid",
            "external_fd_weight_grid", "admission_nms_iou_grid",
            "admission_threshold_grid", "admission_subject_cap_grid",
            "verifier_blend_weight_grid",
        ):
            raw[name] = tuple(raw[name])
        return RunConfig(**raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionContractError("registered summary has an invalid run configuration") from exc


def _model_mapping(fit: object) -> dict[str, object]:
    models = {
        "macro": getattr(fit, "macro"),
        "micro": getattr(fit, "micro"),
        "verifier_logistic": getattr(fit, "verifier_logistic"),
        "verifier_lgbm": getattr(fit, "verifier_lgbm"),
    }
    if any(value is None for value in models.values()):
        raise PromotionContractError("registered multiscale promotion fit is incomplete")
    return models


def _terminal_estimator(model: object) -> object:
    """Return the fitted estimator behind a sklearn-style pipeline."""

    steps = getattr(model, "steps", None)
    if steps:
        return _terminal_estimator(steps[-1][1])
    estimator = getattr(model, "estimator", None)
    return _terminal_estimator(estimator) if estimator is not None else model


def _fitted_input_width(model: object, name: str) -> int:
    width = getattr(_terminal_estimator(model), "n_features_in_", None)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise PromotionContractError(f"fitted {name} model does not expose a valid n_features_in_")
    return width


def _validated_feature_schema(fit: object) -> dict[str, int]:
    models = _model_mapping(fit)
    schema = {
        "macro": _fitted_input_width(models["macro"], "macro"),
        "micro": _fitted_input_width(models["micro"], "micro"),
        "verifier": _fitted_input_width(models["verifier_logistic"], "verifier_logistic"),
    }
    lgbm_width = _fitted_input_width(models["verifier_lgbm"], "verifier_lgbm")
    if schema != {"macro": 63, "micro": 47, "verifier": 56} or lgbm_width != schema["verifier"]:
        raise PromotionContractError("fitted promotion models do not match the registered 63/47/56 feature schema")
    return schema


def _runtime_policy_from_result(result: object) -> dict[str, object]:
    policy = {
        "blend_weight": getattr(result, "selected_blend_weight"),
        "admission_threshold": getattr(result, "selected_admission_threshold"),
        "nms_iou": getattr(result, "selected_admission_nms_iou"),
        "max_candidates_per_subject": getattr(result, "selected_admission_subject_cap"),
        "threshold": getattr(result, "threshold"),
        "max_events_per_group": getattr(result, "max_events_per_subject"),
    }
    if any(value is None for name, value in policy.items() if name != "max_events_per_group"):
        raise PromotionContractError("outer-fold fit did not produce a complete candidate-control policy")
    return policy


def _fingerprints(paths: Sequence[Path]) -> tuple[dict[str, object], ...]:
    result = []
    for path in sorted({Path(item).resolve() for item in paths}, key=lambda item: str(item)):
        stat = path.stat()
        result.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return tuple(result)


def _same_selected_policy(result: object, record: Mapping[str, object]) -> bool:
    fields = (
        "micro_threshold", "selected_blend_weight", "selected_admission_nms_iou",
        "selected_admission_threshold", "selected_admission_subject_cap", "threshold",
        "max_events_per_subject", "verifier_c",
    )
    return all(getattr(result, field) == record[field] for field in fields)


def _assert_evidence_equal(expected: object, observed: object, name: str) -> None:
    """Compare JSON evidence exactly, permitting only arithmetic roundoff."""

    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        if set(expected) != set(observed):
            raise PromotionContractError(f"registered summary {name} keys differ from fold evidence")
        for key in expected:
            _assert_evidence_equal(expected[key], observed[key], f"{name}.{key}")
        return
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            raise PromotionContractError(f"registered summary {name} differs from fold evidence")
        for index, (left, right) in enumerate(zip(expected, observed)):
            _assert_evidence_equal(left, right, f"{name}[{index}]")
        return
    if isinstance(expected, float) or isinstance(observed, float):
        try:
            left, right = float(expected), float(observed)
        except (TypeError, ValueError) as exc:
            raise PromotionContractError(f"registered summary {name} differs from fold evidence") from exc
        if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
            raise PromotionContractError(f"registered summary {name} differs from fold evidence")
        return
    if expected != observed:
        raise PromotionContractError(f"registered summary {name} differs from fold evidence")


def _validate_summary_aggregate(
    summary: Mapping[str, object], configs: Sequence[RunConfig], records: Sequence[Mapping[str, object]],
) -> None:
    try:
        results = tuple(_fold_result_from_dict(record) for record in records)
        recomputed = aggregate_fold_results(configs, results)
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionContractError("registered fold evidence cannot be aggregated") from exc
    for name, value in recomputed.items():
        _assert_evidence_equal(value, summary.get(name), name)


def _validate_current_cache_bindings(
    configs: Sequence[RunConfig], records: Sequence[Mapping[str, object]], source: object,
) -> None:
    input_files = getattr(source, "input_files", None)
    if not callable(input_files):
        raise PromotionContractError("registered promotion source cannot enumerate input files")
    for config, record in zip(configs, records):
        try:
            current_hash = cache_key(config, (63, 56, 47), input_files(config))
        except (OSError, TypeError, ValueError) as exc:
            raise PromotionContractError("registered promotion inputs cannot be fingerprinted") from exc
        if record.get("config_hash") != current_hash:
            raise PromotionContractError("registered fold evidence does not bind the current files and 63/56/47 schema")


def _registered_fold_records(summary: Mapping[str, object]) -> tuple[tuple[RunConfig, ...], tuple[Mapping[str, object], ...]]:
    if summary.get("experiment_key") != _REGISTERED_EXPERIMENT_KEY:
        raise PromotionContractError("summary is not the registered promotion summary a7396a9aa7c38f42")
    raw_configs = summary.get("run_configs")
    folds = summary.get("folds")
    if not isinstance(raw_configs, list) or not isinstance(folds, list) or len(raw_configs) != 5 or len(folds) != 5:
        raise PromotionContractError("registered summary must contain five run configurations and five fold records")
    configs = tuple(sorted((_config_from_summary(item) for item in raw_configs if isinstance(item, Mapping)), key=lambda item: item.outer_fold))
    if len(configs) != 5 or tuple(item.outer_fold for item in configs) != (0, 1, 2, 3, 4):
        raise PromotionContractError("registered summary must provide exactly outer folds 0 through 4")
    if experiment_key(configs) != _REGISTERED_EXPERIMENT_KEY:
        raise PromotionContractError("registered summary configurations do not reproduce experiment key a7396a9aa7c38f42")
    if any(not config.micro_enabled or not config.candidate_control_enabled or config.no_tcn is not True for config in configs):
        raise PromotionContractError("registered summary is not the frozen CPU multiscale candidate-control configuration")
    by_fold = {item.outer_fold: item for item in configs}
    records: list[Mapping[str, object]] = []
    for fold in sorted(folds, key=lambda item: int(item["outer_fold"]) if isinstance(item, Mapping) and "outer_fold" in item else -1):
        if not isinstance(fold, Mapping) or not isinstance(fold.get("outer_fold"), int) or not isinstance(fold.get("config_hash"), str):
            raise PromotionContractError("registered summary fold records are malformed")
        config = by_fold.get(fold["outer_fold"])
        if config is None:
            raise PromotionContractError("registered summary fold records do not match configurations")
        # The selected settings live in the compact fold evidence written beside
        # the canonical aggregate summary, not in outer metrics.
        evidence_path = project_config.OUTPUT_DIR / "crossfit" / f"fold{config.outer_fold}_{fold['config_hash']}.json"
        try:
            record = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PromotionContractError(f"registered fold evidence cannot be read: {evidence_path}") from exc
        if not isinstance(record, dict) or record.get("config_hash") != fold["config_hash"]:
            raise PromotionContractError("registered fold evidence does not match its summary config hash")
        if json.dumps(record.get("run_config"), sort_keys=True) != json.dumps(asdict(config), sort_keys=True):
            raise PromotionContractError("registered fold evidence run configuration differs from canonical summary")
        records.append(record)
    frozen_records = tuple(records)
    _validate_summary_aggregate(summary, configs, frozen_records)
    return configs, frozen_records


def registered_filesystem_trainer(summary: Mapping[str, object]) -> Mapping[str, EventStackBundle]:
    """The sole Task-6 trainer registered for the approved experiment summary."""

    configs, records = _registered_fold_records(summary)
    source = FilesystemDataSource(project_config.ROOT_DIR)
    _validate_current_cache_bindings(configs, records, source)
    bundles: dict[str, EventStackBundle] = {}
    for config, record in zip(configs, records):
        result, fit, _ = fit_outer_fold_for_promotion(config, source)
        if not _same_selected_policy(result, record):
            raise PromotionContractError("retrained outer-fold policy differs from frozen registered evidence")
        bundles[f"outer-fold-{config.outer_fold}"] = EventStackBundle(
            models=_model_mapping(fit),
            policy=_runtime_policy_from_result(result),
            run_config=asdict(config),
            feature_schema=_validated_feature_schema(fit),
            metrics=asdict(result.outer_metrics),
            source_fingerprints=_fingerprints(source.input_files(config)),
            role="outer-fold-evidence",
        )

    deployment_policy = canonical_deployment_policy(records)
    full_target = build_full_target_dataset(configs, source, expected_truth_count=153)
    fit, full_candidate_count = fit_full_target_deployment(configs[0], full_target, deployment_policy)
    # Deployment uses the canonical held-out aggregate as promotion evidence;
    # full-target fit metrics are intentionally not reported as a new test score.
    aggregate_metrics = dict(summary["outer_metrics"])
    aggregate_metrics["full_target_candidate_count"] = full_candidate_count
    bundles["deployment"] = EventStackBundle(
        models=_model_mapping(fit),
        policy={key: deployment_policy[key] for key in (
            "blend_weight", "admission_threshold", "nms_iou",
            "max_candidates_per_subject", "threshold", "max_events_per_group",
        )},
        run_config={
            "deployment_training": "union of five disjoint outer-validation partitions",
            "policy_aggregation": deployment_policy["aggregation"],
            "micro_threshold": deployment_policy["micro_threshold"],
            "verifier_c": deployment_policy["verifier_c"],
            "fold_configs": [asdict(config) for config in configs],
        },
        feature_schema=_validated_feature_schema(fit),
        metrics=aggregate_metrics,
        source_fingerprints=_fingerprints(source.deployment_input_files(configs)),
        role="deployment",
    )
    return bundles


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
        payload = json.loads(args.summary.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise PromotionContractError("registered summary must be a JSON object")
        claimed_f1 = float(payload.get("outer_metrics", {}).get("f1")) if isinstance(payload.get("outer_metrics"), Mapping) else None
        if claimed_f1 is not None and math.isfinite(claimed_f1) and claimed_f1 <= PROMOTION_F1_FLOOR:
            raise PromotionContractError(
                f"aggregate F1 must be strictly greater than {PROMOTION_F1_FLOOR:.10f}; got {claimed_f1:.10f}"
            )
        # Validate before ``promote_summary`` evaluates its F1 gate: the gate
        # must be fed re-derived fold evidence, never a self-reported summary.
        _registered_fold_records(payload)
        promote_summary(
            args.summary,
            output_root=args.output_root,
            trainer=registered_filesystem_trainer,
        )
    except (OSError, ValueError, PromotionContractError) as exc:
        print(f"promotion refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
