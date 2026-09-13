"""Persistence contracts for immutable event-stack model bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.pipeline.artifacts import (
    PROMOTION_F1_FLOOR,
    EventStackBundle,
    PromotionContractError,
    cleanup_stale_bundle_temporary_directories,
    load_event_stack_bundle,
    promote_summary,
    verify_bundle_manifest,
    write_event_stack_bundle,
)
from src.pipeline.event_stack import DensityConfig, EventMetrics
from src.pipeline.runner import FoldResult, RunConfig, aggregate_fold_results, cache_key, experiment_key, fold_result_to_dict


def _promotion_policy_record(**overrides):
    """A frozen outer-train policy record; outer metrics are deliberately noise."""

    record = {
        "micro_threshold": 0.1,
        "selected_blend_weight": 0.75,
        "selected_admission_nms_iou": 0.3,
        "selected_admission_threshold": 0.2,
        "selected_admission_subject_cap": 8,
        "threshold": 0.6,
        "max_events_per_subject": None,
        "verifier_c": 0.1,
        "outer_metrics": {"f1": 0.01, "n_tp": 0, "n_pred": 99, "n_true": 1},
    }
    record.update(overrides)
    return record


def test_canonical_deployment_policy_is_independent_of_outer_metrics():
    from scripts.promote_event_stack import canonical_deployment_policy

    records = [
        _promotion_policy_record(threshold=value)
        for value in (0.9, 0.6, 0.7, 0.5, 0.8)
    ]
    expected = canonical_deployment_policy(records)
    mutated = [
        {**record, "outer_metrics": {"f1": 1.0 - index / 10, "n_tp": 99, "n_pred": 1, "n_true": 1}}
        for index, record in enumerate(records)
    ]

    assert canonical_deployment_policy(mutated) == expected
    assert expected == {
        "micro_threshold": 0.1,
        "blend_weight": 0.75,
        "nms_iou": 0.3,
        "admission_threshold": 0.2,
        "max_candidates_per_subject": 8,
        "threshold": 0.7,
        "max_events_per_group": None,
        "verifier_c": 0.1,
        "aggregation": "equal-fold canonical median/mode; ties use the smallest canonical value",
    }


def test_registered_filesystem_trainer_refuses_any_summary_other_than_the_locked_experiment():
    from scripts.promote_event_stack import registered_filesystem_trainer

    with pytest.raises(PromotionContractError, match="registered promotion summary"):
        registered_filesystem_trainer({
            "experiment_key": "not-a7396a9aa7c38f42",
            "outer_metrics": {"f1": 0.9},
            "folds": [{"outer_fold": fold, "config_hash": str(fold)} for fold in range(5)],
            "run_configs": [],
        })


def _registered_promotion_fixture(tmp_path: Path, monkeypatch):
    """Create five attested records without touching repository results."""
    import scripts.promote_event_stack as promotion

    configs = tuple(
        RunConfig(
            outer_fold=fold,
            inner_splits=4,
            workers=0,
            micro_enabled=True,
            candidate_control_enabled=True,
            density=DensityConfig(window_threshold=0.28838),
        )
        for fold in range(5)
    )
    assert experiment_key(configs) == "a7396a9aa7c38f42"
    input_file = tmp_path / "registered-input.npz"
    input_file.write_bytes(b"registered-input")
    results = []
    records = []
    for config in configs:
        metrics = EventMetrics(2, 3, 3, 2 / 3, 2 / 3, 2 / 3)
        result = FoldResult(
            config_hash=cache_key(config, (63, 56, 47), (input_file,)),
            threshold=0.6,
            max_events_per_subject=None,
            verifier_c=0.1,
            verifier_feature_count=56,
            inner_metrics=metrics,
            outer_metrics=metrics,
            candidate_count=4,
            candidate_match_recall=2 / 3,
            slices={"duration_lt10": metrics},
            timings_seconds={"total": 1.0},
            cache_hits={"fold_result": False},
            outer_subjects=frozenset({f"outer-{config.outer_fold}"}),
            window_fit_subjects=frozenset({"fit"}),
            verifier_fit_subjects=frozenset({"fit"}),
            micro_threshold=0.1,
            micro_candidate_count=5,
            micro_candidate_match_recall=2 / 3,
            short_meal_candidate_recall=2 / 3,
            macro_window_feature_count=63,
            micro_window_feature_count=47,
            micro_window_fit_subjects=frozenset({"fit"}),
            selected_blend_weight=0.75,
            selected_admission_nms_iou=0.3,
            selected_admission_threshold=0.2,
            selected_admission_subject_cap=8,
        )
        results.append(result)
        record = fold_result_to_dict(result)
        record["run_config"] = asdict(config)
        records.append(record)
    summary = aggregate_fold_results(configs, results)
    summary["experiment_key"] = experiment_key(configs)
    summary["run_configs"] = [asdict(config) for config in configs]
    output_dir = tmp_path / "outputs"
    crossfit = output_dir / "crossfit"
    crossfit.mkdir(parents=True)
    for record in records:
        (crossfit / f"fold{record['run_config']['outer_fold']}_{record['config_hash']}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    monkeypatch.setattr(promotion.project_config, "OUTPUT_DIR", output_dir)

    class Source:
        def input_files(self, _config):
            return (input_file,)

    monkeypatch.setattr(promotion, "FilesystemDataSource", lambda _root: Source())
    return promotion, summary, records


@pytest.mark.parametrize("mutate", ("f1", "workers"))
def test_registered_trainer_rejects_summary_or_evidence_tampering_before_any_write(
    tmp_path: Path, monkeypatch, mutate: str,
):
    promotion, summary, records = _registered_promotion_fixture(tmp_path, monkeypatch)
    if mutate == "f1":
        summary["outer_metrics"] = {**summary["outer_metrics"], "f1": 0.99}
    else:
        evidence = tmp_path / "outputs" / "crossfit" / f"fold0_{records[0]['config_hash']}.json"
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        payload["run_config"]["workers"] = 1
        evidence.write_text(json.dumps(payload), encoding="utf-8")

    output_root = tmp_path / "models"
    with pytest.raises(PromotionContractError):
        promotion.registered_filesystem_trainer(summary)
    assert not output_root.exists()


def fitted_tiny_bundle(role: str = "outer-fold-evidence") -> EventStackBundle:
    model = DummyClassifier(strategy="prior").fit(
        np.array([[0.0], [1.0]]), np.array([0, 1])
    )
    return EventStackBundle(
        models={
            name: model
            for name in ("macro", "micro", "verifier_logistic", "verifier_lgbm")
        },
        policy={"blend_weight": 0.25, "admission_threshold": 0.5},
        run_config={"outer_fold": 0, "candidate_control_enabled": True},
        feature_schema={"macro": 63, "micro": 47, "verifier": 56},
        metrics={"n_tp": 1, "n_true": 1, "n_pred": 1, "f1": 1.0},
        source_fingerprints=({"path": "fixture.npz", "size": 1, "mtime_ns": 1},),
        role=role,
    )


def test_bundle_round_trip_preserves_predictions_and_manifest(tmp_path: Path):
    bundle = fitted_tiny_bundle()
    destination = tmp_path / "models" / "event_stack" / "run-key"

    write_event_stack_bundle(destination, bundle, event_stack_root=destination.parent)
    loaded = load_event_stack_bundle(destination)

    probe = np.array([[0.25], [0.75]])
    for name in bundle.models:
        assert np.array_equal(
            loaded.models[name].predict_proba(probe),
            bundle.models[name].predict_proba(probe),
        )
    assert verify_bundle_manifest(destination) == ()


def test_tampered_model_is_rejected(tmp_path: Path):
    destination = tmp_path / "models" / "event_stack" / "run-key"
    write_event_stack_bundle(
        destination, fitted_tiny_bundle(), event_stack_root=destination.parent
    )
    (destination / "verifier_lgbm.joblib").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_event_stack_bundle(destination)


def test_manifest_model_path_escape_is_rejected_before_deserialization(tmp_path: Path):
    destination = tmp_path / "models" / "event_stack" / "run-key"
    write_event_stack_bundle(
        destination, fitted_tiny_bundle(), event_stack_root=destination.parent
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"] = ["../outside"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest model entries"):
        load_event_stack_bundle(destination)


def test_failed_directory_swap_restores_exact_previous_bundle(tmp_path: Path, monkeypatch):
    destination = tmp_path / "models" / "event_stack" / "run-key"
    write_event_stack_bundle(
        destination, fitted_tiny_bundle(), event_stack_root=destination.parent
    )
    previous = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }

    import src.pipeline.artifacts as artifacts

    real_replace = artifacts.os.replace
    calls = 0

    def fail_only_new_bundle(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replacement failure")
        return real_replace(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail_only_new_bundle)
    with pytest.raises(OSError, match="injected replacement failure"):
        write_event_stack_bundle(
            destination, fitted_tiny_bundle(), event_stack_root=destination.parent
        )

    current = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert current == previous
    assert verify_bundle_manifest(destination) == ()


def test_failed_post_install_verification_restores_previous_bundle(tmp_path: Path, monkeypatch):
    destination = tmp_path / "models" / "event_stack" / "run-key"
    write_event_stack_bundle(
        destination, fitted_tiny_bundle(), event_stack_root=destination.parent
    )
    previous = (destination / "manifest.json").read_bytes()

    import src.pipeline.artifacts as artifacts

    real_verify = artifacts.verify_bundle_manifest

    def reject_only_installed(path, **kwargs):
        if Path(path) == destination:
            return ("injected post-install verification failure",)
        return real_verify(path, **kwargs)

    monkeypatch.setattr(artifacts, "verify_bundle_manifest", reject_only_installed)
    with pytest.raises(ValueError, match="installed bundle failed verification"):
        write_event_stack_bundle(
            destination,
            replace(fitted_tiny_bundle(), metrics={"f1": 0.5}),
            event_stack_root=destination.parent,
        )

    assert (destination / "manifest.json").read_bytes() == previous


def test_stale_cleanup_is_scoped_and_never_follows_external_symlink(tmp_path: Path):
    event_stack = tmp_path / "models" / "event_stack"
    event_stack.mkdir(parents=True)
    external = tmp_path / "outside"
    external.mkdir()
    protected = external / "keep.txt"
    protected.write_text("do not touch", encoding="utf-8")
    stale = event_stack / ".run-key.tmp-stale"
    stale.mkdir()
    (stale / "temporary.txt").write_text("remove", encoding="utf-8")
    other = event_stack / ".other-key.tmp-stale"
    other.mkdir()
    linked = event_stack / ".run-key.tmp-link"
    try:
        linked.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are not available: {exc}")

    removed = cleanup_stale_bundle_temporary_directories(
        event_stack / "run-key", event_stack_root=event_stack
    )

    assert removed == (stale, linked)
    assert not stale.exists()
    assert not linked.exists()
    assert other.exists()
    assert protected.read_text(encoding="utf-8") == "do not touch"


def test_non_improving_summary_performs_zero_writes(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"outer_metrics": {"f1": PROMOTION_F1_FLOOR}}), encoding="utf-8"
    )
    output_root = tmp_path / "models"

    with pytest.raises(PromotionContractError, match="strictly greater"):
        promote_summary(summary, output_root=output_root, trainer=lambda _: ())

    assert not output_root.exists()


def test_deployment_cannot_be_loaded_as_outer_fold_evidence(tmp_path: Path):
    destination = tmp_path / "models" / "event_stack" / "deployment"
    write_event_stack_bundle(
        destination,
        fitted_tiny_bundle(role="deployment"),
        event_stack_root=destination.parent,
    )

    with pytest.raises(ValueError, match="outer-fold-evidence"):
        load_event_stack_bundle(destination, expected_role="outer-fold-evidence")


def test_qualified_promotion_requires_registered_trainer_before_any_write(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"outer_metrics": {"f1": PROMOTION_F1_FLOOR + 0.01}}),
        encoding="utf-8",
    )
    output_root = tmp_path / "models"

    with pytest.raises(PromotionContractError, match="no legal full-target trainer"):
        promote_summary(summary, output_root=output_root)

    assert not output_root.exists()


def test_promotion_cli_enforces_f1_gate_without_creating_output(tmp_path: Path, capsys):
    from scripts.promote_event_stack import main

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"outer_metrics": {"f1": PROMOTION_F1_FLOOR}}), encoding="utf-8"
    )
    output_root = tmp_path / "models"

    assert main(["--summary", str(summary), "--output-root", str(output_root)]) == 2
    assert "strictly greater" in capsys.readouterr().err
    assert not output_root.exists()


def test_qualified_promotion_writes_five_evidence_bundles_and_one_deployment(
    tmp_path: Path,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "experiment_key": "registered-key",
                "outer_metrics": {"f1": PROMOTION_F1_FLOOR + 0.01},
                "folds": [
                    {"outer_fold": fold, "config_hash": f"fold-{fold}"}
                    for fold in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )

    def trainer(_summary):
        return {
            **{f"outer-fold-{fold}": fitted_tiny_bundle() for fold in range(5)},
            "deployment": fitted_tiny_bundle(role="deployment"),
        }

    written = promote_summary(summary, output_root=tmp_path / "models", trainer=trainer)

    assert len(written) == 6
    assert all(path.parent.name == "registered-key" for path in written)
    deployment = next(path for path in written if path.name == "deployment")
    assert load_event_stack_bundle(deployment, expected_role="deployment").role == "deployment"


def test_promotion_attestation_binds_canonical_summary_and_every_bundle_manifest(tmp_path: Path):
    summary = tmp_path / "summary.json"
    source = {
        "experiment_key": "registered-key",
        "outer_metrics": {"f1": PROMOTION_F1_FLOOR + 0.01},
        "folds": [
            {"outer_fold": fold, "config_hash": f"fold-{fold}"}
            for fold in range(5)
        ],
    }
    summary.write_text(json.dumps(source), encoding="utf-8")
    promote_summary(summary, output_root=tmp_path / "models", trainer=_six_bundle_trainer)
    run_root = tmp_path / "models" / "event_stack" / "registered-key"
    attestation = json.loads((run_root / "promotion_attestation.json").read_text(encoding="utf-8"))
    canonical_summary = (run_root / "promotion_summary.json").read_bytes()

    assert attestation["run_key"] == "registered-key"
    assert attestation["aggregate_summary"]["sha256"] == hashlib.sha256(canonical_summary).hexdigest()
    assert attestation["gate"] == {
        "version": 1,
        "floor": PROMOTION_F1_FLOOR,
        "f1": PROMOTION_F1_FLOOR + 0.01,
    }
    assert set(attestation["bundles"]) == {
        "deployment", "outer-fold-0", "outer-fold-1", "outer-fold-2", "outer-fold-3", "outer-fold-4"
    }
    for key, entry in attestation["bundles"].items():
        assert entry["manifest_sha256"] == hashlib.sha256(
            (run_root / key / "manifest.json").read_bytes()
        ).hexdigest()


def test_bundle_write_rejects_same_named_directory_outside_trusted_root(tmp_path: Path):
    trusted_root = tmp_path / "models" / "event_stack"
    outside_destination = tmp_path / "outside" / "event_stack" / "run-key"

    with pytest.raises(ValueError, match="trusted event_stack_root"):
        write_event_stack_bundle(
            outside_destination,
            fitted_tiny_bundle(),
            event_stack_root=trusted_root,
        )

    assert not outside_destination.exists()


def test_manifest_requires_all_metadata_before_model_deserialization(tmp_path: Path):
    destination = tmp_path / "models" / "event_stack" / "run-key"
    write_event_stack_bundle(
        destination, fitted_tiny_bundle(), event_stack_root=destination.parent
    )
    (destination / "policy.json").unlink()
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop("policy.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert any("required metadata" in item for item in verify_bundle_manifest(destination))
    with pytest.raises(ValueError, match="required metadata"):
        load_event_stack_bundle(destination)


@pytest.mark.parametrize(
    "metadata_name", ("policy.json", "run_config.json", "feature_schema.json")
)
def test_metadata_must_be_json_object(tmp_path: Path, metadata_name: str):
    destination = tmp_path / "models" / "event_stack" / "run-key"
    write_event_stack_bundle(
        destination, fitted_tiny_bundle(), event_stack_root=destination.parent
    )
    metadata_path = destination / metadata_name
    metadata_path.write_text("[]", encoding="utf-8")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][metadata_name] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert any(
        "metadata JSON must be an object" in item
        for item in verify_bundle_manifest(destination)
    )
    with pytest.raises(ValueError, match="metadata JSON must be an object"):
        load_event_stack_bundle(destination)


@pytest.mark.parametrize("f1", (float("nan"), float("inf"), -0.1, 1.1))
def test_invalid_aggregate_f1_performs_zero_writes(tmp_path: Path, f1: float):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"outer_metrics": {"f1": f1}}), encoding="utf-8")
    output_root = tmp_path / "models"

    with pytest.raises(PromotionContractError, match="finite"):
        promote_summary(summary, output_root=output_root, trainer=lambda _: {})

    assert not output_root.exists()


def _qualified_summary(tmp_path: Path) -> Path:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "experiment_key": "registered-key",
                "outer_metrics": {"f1": PROMOTION_F1_FLOOR + 0.01},
                "folds": [
                    {"outer_fold": fold, "config_hash": f"fold-{fold}"}
                    for fold in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )
    return summary


def _six_bundle_trainer(_summary):
    return {
        **{f"outer-fold-{fold}": fitted_tiny_bundle() for fold in range(5)},
        "deployment": fitted_tiny_bundle(role="deployment"),
    }


def test_promotion_failure_while_staging_third_bundle_leaves_no_run_key(
    tmp_path: Path, monkeypatch
):
    import src.pipeline.artifacts as artifacts

    real_write = artifacts._write_bundle_contents
    calls = 0

    def fail_third_bundle(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected third bundle failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(artifacts, "_write_bundle_contents", fail_third_bundle)
    output_root = tmp_path / "models"
    with pytest.raises(OSError, match="injected third bundle failure"):
        promote_summary(
            _qualified_summary(tmp_path),
            output_root=output_root,
            trainer=_six_bundle_trainer,
        )

    event_stack = output_root / "event_stack"
    assert not (event_stack / "registered-key").exists()
    assert not tuple(event_stack.glob(".registered-key.*"))


def test_promotion_staging_failure_keeps_existing_run_key_byte_identical(
    tmp_path: Path, monkeypatch
):
    summary = _qualified_summary(tmp_path)
    output_root = tmp_path / "models"
    promote_summary(summary, output_root=output_root, trainer=_six_bundle_trainer)
    run_root = output_root / "event_stack" / "registered-key"
    previous = {
        path.relative_to(run_root).as_posix(): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }

    import src.pipeline.artifacts as artifacts

    real_write = artifacts._write_bundle_contents
    calls = 0

    def fail_third_bundle(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected third bundle failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(artifacts, "_write_bundle_contents", fail_third_bundle)
    with pytest.raises(OSError, match="injected third bundle failure"):
        promote_summary(summary, output_root=output_root, trainer=_six_bundle_trainer)

    current = {
        path.relative_to(run_root).as_posix(): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }
    assert current == previous
