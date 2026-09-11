"""Tests for the standalone, verified event-stack inference package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import package_event_stack as package_module
from scripts.package_event_stack import (
    EXPECTED_EVENT_STACK_PATHS,
    package_event_stack,
    required_paths,
)
from scripts.predict_event_stack import (
    assert_backend_parity,
    build_smoke_fixture,
    predict_feature_payload,
    resolve_device,
)
from src.pipeline.artifacts import EventStackBundle, promote_summary, write_event_stack_bundle


def _schema_hash(schema: dict[str, int]) -> str:
    payload = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fitted_deployment_bundle() -> EventStackBundle:
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit(
        np.array([[0.0], [1.0]]), np.array([0, 1])
    )
    return EventStackBundle(
        models={
            name: model
            for name in ("macro", "micro", "verifier_logistic", "verifier_lgbm")
        },
        policy={
            "blend_weight": 0.25,
            "admission_threshold": 0.0,
            "threshold": 0.0,
            "nms_iou": 0.5,
            "max_candidates_per_subject": 2,
            "max_events_per_group": 1,
        },
        run_config={"candidate_control_enabled": True},
        feature_schema={"macro": 1, "micro": 1, "verifier": 1},
        metrics={"n_tp": 1, "n_true": 1, "n_pred": 1, "f1": 1.0},
        source_fingerprints=({"path": "fixture.npz", "size": 1, "mtime_ns": 1},),
        role="deployment",
    )


def fitted_scored_deployment_bundle() -> EventStackBundle:
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(
        np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1])
    )
    return EventStackBundle(
        models={name: model for name in ("macro", "micro", "verifier_logistic", "verifier_lgbm")},
        policy={
            "blend_weight": 0.25,
            "admission_threshold": 0.5,
            "threshold": 0.5,
            "nms_iou": 0.5,
            "max_candidates_per_subject": 2,
            "max_events_per_group": 1,
        },
        run_config={"candidate_control_enabled": True},
        feature_schema={"macro": 1, "micro": 1, "verifier": 1},
        metrics={"n_tp": 1, "n_true": 1, "n_pred": 1, "f1": 1.0},
        source_fingerprints=({"path": "fixture.npz", "size": 1, "mtime_ns": 1},),
        role="deployment",
    )


def package_fixture_bundle(
    tmp_path: Path, bundle_factory=fitted_deployment_bundle
) -> tuple[Path, Path]:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "experiment_key": "run-key",
        "outer_metrics": {"f1": 0.9},
        "folds": [
            {"outer_fold": fold, "config_hash": f"fold-{fold}"}
            for fold in range(5)
        ],
    }), encoding="utf-8")
    written = promote_summary(
        summary,
        output_root=tmp_path / "models",
        trainer=lambda _summary: {
            **{
                f"outer-fold-{fold}": EventStackBundle(
                    **{**bundle_factory().__dict__, "role": "outer-fold-evidence"}
                )
                for fold in range(5)
            },
            "deployment": bundle_factory(),
        },
    )
    bundle_root = next(path for path in written if path.name == "deployment")
    destination = tmp_path / "dist" / "event_stack"
    package_event_stack(
        bundle_path=bundle_root,
        destination=destination,
        trusted_dist_root=destination.parent,
    )
    return destination, bundle_root


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("packaged_predict_event_stack", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_cli(script: Path, bundle: Path, payload: Path, output: Path) -> tuple[str, bytes]:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script.resolve()),
            "--bundle",
            str(bundle),
            "--input-features",
            str(payload),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=output.parent,
    )
    return completed.stdout, output.read_bytes()


def test_dist_package_is_complete_and_matches_repository_prediction(tmp_path: Path):
    dist, bundle = package_fixture_bundle(tmp_path)
    assert required_paths(dist) == EXPECTED_EVENT_STACK_PATHS

    fixture = build_smoke_fixture({"macro": 1, "micro": 1, "verifier": 1})
    fixture["schema_hash"] = _schema_hash(fixture["feature_schema"])
    payload = tmp_path / "fixture.json"
    payload.write_text(json.dumps(fixture), encoding="utf-8")
    repo_output = tmp_path / "repo.json"
    dist_output = tmp_path / "dist.json"
    _, repository_bytes = _run_cli(
        Path("scripts/predict_event_stack.py"), bundle, payload, repo_output
    )
    _, packaged_bytes = _run_cli(
        dist / "predict_event_stack.py", dist / "bundle", payload, dist_output
    )
    assert packaged_bytes == repository_bytes


def test_package_rejects_non_deployment_and_incomplete_bundles_without_destination(tmp_path: Path):
    bundle = fitted_deployment_bundle()
    outer = tmp_path / "models" / "event_stack" / "run" / "outer-fold-0"
    outer.parent.mkdir(parents=True)
    write_event_stack_bundle(
        outer,
        EventStackBundle(**{**bundle.__dict__, "role": "outer-fold-evidence"}),
        event_stack_root=outer.parents[1],
    )
    destination = tmp_path / "dist" / "event_stack"
    with pytest.raises(ValueError, match="deployment"):
        package_event_stack(bundle_path=outer, destination=destination, trusted_dist_root=destination.parent)
    assert not destination.exists()

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="manifest"):
        package_event_stack(bundle_path=incomplete, destination=destination, trusted_dist_root=destination.parent)
    assert not destination.exists()


def test_package_rejects_deployment_bundle_below_promotion_gate(tmp_path: Path):
    from src.pipeline.artifacts import PROMOTION_F1_FLOOR

    bundle_path = tmp_path / "models" / "event_stack" / "run" / "deployment"
    bundle_path.parent.mkdir(parents=True)
    bundle = fitted_deployment_bundle()
    write_event_stack_bundle(
        bundle_path,
        EventStackBundle(**{**bundle.__dict__, "metrics": {"f1": PROMOTION_F1_FLOOR}}),
        event_stack_root=bundle_path.parents[1],
    )
    with pytest.raises(ValueError, match="promotion F1 gate"):
        package_event_stack(bundle_path=bundle_path, destination=tmp_path / "dist" / "event_stack", trusted_dist_root=tmp_path / "dist")


def test_failed_package_build_keeps_previous_dist_and_legacy_files(tmp_path: Path, monkeypatch):
    destination, bundle = package_fixture_bundle(tmp_path)
    dist_root = destination.parent
    (destination / "previous.txt").write_text("keep", encoding="utf-8")
    legacy = dist_root / "predict.py"
    legacy.write_text("legacy", encoding="utf-8")

    def raise_error(*_args, **_kwargs):
        raise RuntimeError("injected package verification failure")

    monkeypatch.setattr(package_module, "verify_packaged_bundle", raise_error)
    with pytest.raises(RuntimeError, match="injected package"):
        package_event_stack(bundle_path=bundle, destination=destination, trusted_dist_root=destination.parent)
    assert (destination / "previous.txt").read_text(encoding="utf-8") == "keep"
    assert legacy.read_text(encoding="utf-8") == "legacy"


def test_device_resolution_is_explicit_and_torch_is_lazy(monkeypatch):
    import scripts.predict_event_stack as predict_module

    monkeypatch.setattr(predict_module, "_cuda_available", lambda: False)
    assert resolve_device("cpu", has_cuda_component=False) == "cpu"
    assert resolve_device("auto", has_cuda_component=False) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA-capable component"):
        resolve_device("gpu", has_cuda_component=False)
    with pytest.raises(RuntimeError, match="CUDA-capable component"):
        resolve_device("cuda", has_cuda_component=False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda", has_cuda_component=True)
    with pytest.raises(ValueError, match="auto, cpu, gpu, cuda"):
        resolve_device("metal", has_cuda_component=True)


def test_future_cuda_backend_parity_hook_requires_score_tolerance_and_exact_geometry():
    cpu = {"events": [{"sid": "s", "start_ms": 0, "end_ms": 100, "score": 0.5}]}
    cuda_close = {"events": [{"sid": "s", "start_ms": 0, "end_ms": 100, "score": 0.500009}]}
    assert_backend_parity(cpu, cuda_close)

    with pytest.raises(ValueError, match="score"):
        assert_backend_parity(cpu, {"events": [{"sid": "s", "start_ms": 0, "end_ms": 100, "score": 0.50002}]})
    with pytest.raises(ValueError, match="geometry"):
        assert_backend_parity(cpu, {"events": [{"sid": "s", "start_ms": 1, "end_ms": 100, "score": 0.5}]})


def test_precomputed_feature_contract_rejects_schema_or_manifest_hash_mismatch(tmp_path: Path):
    dist, _ = package_fixture_bundle(tmp_path)
    packaged = _load_module(dist / "predict_event_stack.py")
    fixture = build_smoke_fixture({"macro": 1, "micro": 1, "verifier": 1})

    with pytest.raises(ValueError, match="schema hash"):
        packaged.predict_feature_payload(dist / "bundle", fixture, device="cpu")

    fixture["schema_hash"] = _schema_hash(fixture["feature_schema"])
    result = packaged.predict_feature_payload(dist / "bundle", fixture, device="auto")
    assert result == {
        "events": [
            {"end_ms": 1000, "score": 0.5, "sid": "fixture-session", "start_ms": 0}
        ],
        "resolved_device": "cpu",
    }

    manifest = dist / "bundle" / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        packaged.predict_feature_payload(dist / "bundle", fixture, device="cpu")


def test_packaged_runtime_manifest_rejects_tampered_runtime_file(tmp_path: Path):
    dist, _ = package_fixture_bundle(tmp_path)
    packaged = _load_module(dist / "predict_event_stack.py")
    fixture = build_smoke_fixture({"macro": 1, "micro": 1, "verifier": 1})
    fixture["schema_hash"] = _schema_hash(fixture["feature_schema"])
    (dist / "requirements.txt").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime manifest.*checksum"):
        packaged.predict_feature_payload(dist / "bundle", fixture, device="cpu")


def test_packaged_entrypoint_runs_in_isolated_subprocess_and_reports_resolved_device(tmp_path: Path):
    dist, _ = package_fixture_bundle(tmp_path)
    fixture = build_smoke_fixture({"macro": 1, "micro": 1, "verifier": 1})
    fixture["schema_hash"] = _schema_hash(fixture["feature_schema"])
    payload = tmp_path / "fixture.json"
    output = tmp_path / "out.json"
    payload.write_text(json.dumps(fixture), encoding="utf-8")

    stdout, contents = _run_cli(dist / "predict_event_stack.py", dist / "bundle", payload, output)
    assert "resolved_device=cpu" in stdout
    assert contents == (
        b'{"events":[{"end_ms":1000,"score":0.5,"sid":"fixture-session","start_ms":0}],'
        b'"resolved_device":"cpu"}\n'
    )


def test_packager_cli_is_importable_when_executed_as_a_script():
    completed = subprocess.run(
        [sys.executable, "scripts/package_event_stack.py", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Task-4 deployment bundle directory" in completed.stdout


def test_runtime_requires_explicit_subject_ids_and_applies_the_frozen_two_stage_policy(
    tmp_path: Path,
):
    dist, _ = package_fixture_bundle(tmp_path)
    packaged = _load_module(dist / "predict_event_stack.py")
    schema = {"macro": 1, "micro": 1, "verifier": 1}
    payload = {
        "feature_schema": schema,
        "schema_hash": _schema_hash(schema),
        "sessions": [{
            "subject_id": "person-a",
            "sid": "a-morning",
            "candidates": [
                {"start_ms": 0, "end_ms": 100, "macro": [0.0], "micro": [0.0], "verifier": [0.0]},
                {"start_ms": 10, "end_ms": 110, "macro": [0.0], "micro": [0.0], "verifier": [0.0]},
                {"start_ms": 200, "end_ms": 300, "macro": [0.0], "micro": [0.0], "verifier": [0.0]},
            ],
        }, {
            "subject_id": "person-b",
            "sid": "b-evening",
            "candidates": [
                {"start_ms": 0, "end_ms": 100, "macro": [0.0], "micro": [0.0], "verifier": [0.0]},
                {"start_ms": 200, "end_ms": 300, "macro": [0.0], "micro": [0.0], "verifier": [0.0]},
            ],
        }],
    }
    result = packaged.predict_feature_payload(dist / "bundle", payload, device="cpu")
    assert [(row["sid"], row["start_ms"]) for row in result["events"]] == [
        ("a-morning", 0), ("b-evening", 0)
    ]

    missing_subject = {**payload, "sessions": [{
        key: value for key, value in payload["sessions"][0].items() if key != "subject_id"
    }]}
    with pytest.raises(ValueError, match="subject_id"):
        packaged.predict_feature_payload(dist / "bundle", missing_subject, device="cpu")
    old_group_field = {**payload, "sessions": [{
        **payload["sessions"][0], "group": "person-a"
    }]}
    with pytest.raises(ValueError, match="unknown.*group"):
        packaged.predict_feature_payload(dist / "bundle", old_group_field, device="cpu")


def test_runtime_golden_policy_is_subject_scoped_across_multiple_sessions(tmp_path: Path):
    dist, _ = package_fixture_bundle(tmp_path, fitted_scored_deployment_bundle)
    packaged = _load_module(dist / "predict_event_stack.py")
    schema = {"macro": 1, "micro": 1, "verifier": 1}

    def candidate(start_ms: int, end_ms: int, score_feature: float) -> dict[str, object]:
        return {"start_ms": start_ms, "end_ms": end_ms, "macro": [score_feature], "micro": [score_feature], "verifier": [score_feature]}

    payload = {
        "feature_schema": schema,
        "schema_hash": _schema_hash(schema),
        "sessions": [
            {"subject_id": "a", "sid": "a-1", "candidates": [candidate(0, 100, 2.0), candidate(10, 110, 2.0), candidate(200, 300, 2.0), candidate(400, 500, 0.0)]},
            {"subject_id": "a", "sid": "a-2", "candidates": [candidate(0, 100, 2.0)]},
            {"subject_id": "b", "sid": "b-1", "candidates": [candidate(0, 100, 2.0)]},
            {"subject_id": "b", "sid": "b-2", "candidates": [candidate(0, 100, 2.0)]},
        ],
    }
    # Hand calculation: feature 2 scores 1 and feature 0 scores 0.  NMS removes
    # a-1[10,110]; subject a's candidate cap retains a-1[0,100], a-1[200,300],
    # then its event cap retains only the first.  The same event cap leaves b-1.
    assert packaged.predict_feature_payload(dist / "bundle", payload, device="cpu")["events"] == [
        {"sid": "a-1", "start_ms": 0, "end_ms": 100, "score": 1.0},
        {"sid": "b-1", "start_ms": 0, "end_ms": 100, "score": 1.0},
    ]


def test_packager_rejects_self_reported_deployment_without_promotion_attestation(tmp_path: Path):
    bundle = tmp_path / "models" / "event_stack" / "run" / "deployment"
    bundle.parent.mkdir(parents=True)
    write_event_stack_bundle(bundle, fitted_deployment_bundle(), event_stack_root=bundle.parents[1])

    with pytest.raises(ValueError, match="promotion attestation"):
        package_event_stack(
            bundle_path=bundle,
            destination=tmp_path / "dist" / "event_stack",
            trusted_dist_root=tmp_path / "dist",
        )


def test_packager_destination_is_exactly_anchored_to_one_trusted_dist_root(tmp_path: Path):
    _, bundle = package_fixture_bundle(tmp_path)
    trusted = tmp_path / "dist"
    with pytest.raises(ValueError, match="trusted_dist_root"):
        package_event_stack(
            bundle_path=bundle,
            destination=tmp_path / "outside" / "event_stack",
            trusted_dist_root=trusted,
        )
    with pytest.raises(ValueError, match="exact.*event_stack"):
        package_event_stack(
            bundle_path=bundle,
            destination=trusted,
            trusted_dist_root=trusted,
        )
    linked_root = tmp_path / "linked" / "dist"
    external = tmp_path / "external-dist"
    external.mkdir()
    linked_root.parent.mkdir()
    try:
        linked_root.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are not available: {exc}")
    with pytest.raises(ValueError, match="symlink/reparse"):
        package_event_stack(
            bundle_path=bundle,
            destination=linked_root / "event_stack",
            trusted_dist_root=linked_root,
        )


def test_runtime_manifest_cuda_boolean_cannot_claim_an_unregistered_adapter(tmp_path: Path, monkeypatch):
    dist, _ = package_fixture_bundle(tmp_path)
    packaged = _load_module(dist / "predict_event_stack.py")
    runtime_manifest = dist / "runtime_manifest.json"
    runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    runtime["capabilities"]["has_cuda_component"] = True
    runtime_manifest.write_text(json.dumps(runtime), encoding="utf-8")
    monkeypatch.setattr(packaged, "_cuda_available", lambda: True)
    fixture = build_smoke_fixture({"macro": 1, "micro": 1, "verifier": 1})
    fixture["schema_hash"] = _schema_hash(fixture["feature_schema"])

    with pytest.raises(ValueError, match="registered CUDA adapter"):
        packaged.predict_feature_payload(dist / "bundle", fixture, device="cuda")
