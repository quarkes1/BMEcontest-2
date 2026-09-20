"""Crash-recovery contracts for the sole active event-stack release command."""

from __future__ import annotations

import json
import shutil
import hashlib
import subprocess
import sys
import os
from pathlib import Path

import pytest


def _release_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "project"
    shutil.copytree("models/event_stack/160afaf81debf1ee", root / "models/event_stack/160afaf81debf1ee")
    shutil.copytree("dist/event_stack", root / "dist/event_stack")
    registry = json.loads(Path("release/event_stack_incumbent.json").read_text(encoding="utf-8"))
    (root / "release").mkdir()
    (root / "release/event_stack_incumbent.json").write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return root, registry


def _distinct_candidate(root: Path) -> dict[str, object]:
    """Create a separately attested run and copied package for interruption tests."""
    from src.pipeline.artifacts import _stable_json_bytes, incumbent_registry_payload

    old_key = "160afaf81debf1ee"
    new_key = "candidate-run-1"
    candidate_root = root / "models/event_stack" / new_key
    shutil.copytree(root / "models/event_stack" / old_key, candidate_root)
    manifest_path = candidate_root / "deployment/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git_sha"] = "candidate"
    manifest_path.write_bytes(_stable_json_bytes(manifest))
    summary_path = candidate_root / "promotion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment_key"] = new_key
    summary_path.write_bytes(_stable_json_bytes(summary))
    bundles = {}
    for name in [*(f"outer-fold-{fold}" for fold in range(5)), "deployment"]:
        entry = {"role": "deployment" if name == "deployment" else "outer-fold-evidence",
                 "manifest_sha256": hashlib.sha256((candidate_root / name / "manifest.json").read_bytes()).hexdigest()}
        if name != "deployment":
            entry["diagnostics_sha256"] = hashlib.sha256((candidate_root / name / "diagnostics.json").read_bytes()).hexdigest()
        bundles[name] = entry
    attestation = json.loads((candidate_root / "promotion_attestation.json").read_text(encoding="utf-8"))
    attestation["run_key"] = new_key
    attestation["aggregate_summary"]["sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    attestation["bundles"] = bundles
    (candidate_root / "promotion_attestation.json").write_bytes(_stable_json_bytes(attestation))
    candidate_dist = root / "dist/event_stack"
    shutil.rmtree(candidate_dist)
    shutil.copytree(root / "models/event_stack" / new_key / "deployment", candidate_dist / "bundle")
    # Rebuild the runtime package by borrowing non-bundle runtime files from source dist.
    source_dist = Path("dist/event_stack")
    for name in ("predict_event_stack.py", "requirements.txt"):
        shutil.copy2(source_dist / name, candidate_dist / name)
    runtime = json.loads((source_dist / "runtime_manifest.json").read_text(encoding="utf-8"))
    runtime["bundle_manifest_sha256"] = hashlib.sha256((candidate_dist / "bundle/manifest.json").read_bytes()).hexdigest()
    runtime["files"]["bundle/manifest.json"] = runtime["bundle_manifest_sha256"]
    (candidate_dist / "runtime_manifest.json").write_text(json.dumps(runtime, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return incumbent_registry_payload(candidate_root)


def test_recovery_after_prepared_phase_retains_old_verified_release(tmp_path: Path):
    from scripts.release_event_stack import _journal_path, _stable_json_bytes, recover_release_transaction

    root, registry = _release_fixture(tmp_path)
    journal = {
        "schema_version": 1,
        "phase": "prepared",
        "previous_registry": registry,
        "candidate_registry": registry,
        "candidate_dist_sha256": None,
        "backup_dist_path": None,
    }
    _journal_path(root).write_bytes(_stable_json_bytes(journal))

    recover_release_transaction(root)

    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry


def test_active_dist_candidate_match_reads_runtime_manifest(tmp_path: Path):
    from scripts.release_event_stack import _active_dist_matches_candidate

    root, registry = _release_fixture(tmp_path)
    assert _active_dist_matches_candidate(root, registry) is True


def test_release_refuses_context_v2_before_promotion_or_dist_write(tmp_path: Path, monkeypatch):
    """Research-only E02 summaries cannot enter the active release transaction."""

    from scripts import release_event_stack as release
    from src.pipeline.artifacts import PromotionContractError

    root, registry = _release_fixture(tmp_path)
    summary = root / "context-v2.json"
    summary.write_text(json.dumps({
        "outer_metrics": {"f1": 0.9},
        "run_configs": [{"context_features_version": "v2"}],
    }), encoding="utf-8")
    monkeypatch.setattr(
        release, "promote_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("promotion called")),
    )

    with pytest.raises(PromotionContractError, match="challenger requires human promotion/runtime implementation"):
        release.release_summary(summary, root=root)
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry


def test_new_process_recovers_package_swap_before_tree_hash_with_distinct_pair(tmp_path: Path):
    """The post-swap/pre-hash interruption must finish the new, not mixed, pair."""

    from scripts.release_event_stack import _journal_path, _stable_json_bytes

    root, old_registry = _release_fixture(tmp_path)
    candidate = _distinct_candidate(root)
    assert candidate["run_key"] != old_registry["run_key"]
    _journal_path(root).write_bytes(_stable_json_bytes({
        "schema_version": 1,
        "phase": "prepared",
        "previous_registry": old_registry,
        "candidate_registry": candidate,
        "candidate_dist_sha256": None,
        "backup_dist_path": None,
    }))

    completed = subprocess.run(
        [sys.executable, "-c", f"from scripts.release_event_stack import recover_release_transaction; recover_release_transaction(r'{root}')"],
        cwd=Path.cwd(), text=True, capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == candidate


@pytest.mark.parametrize("phase", ["dist_backup", "dist_replaced", "registry_replaced", "verified"])
def test_new_process_recovers_each_later_candidate_phase(tmp_path: Path, phase: str):
    from scripts.release_event_stack import _journal_path, _stable_json_bytes, _tree_hash

    root, old_registry = _release_fixture(tmp_path)
    candidate = _distinct_candidate(root)
    _journal_path(root).write_bytes(_stable_json_bytes({
        "schema_version": 1, "phase": phase, "previous_registry": old_registry,
        "candidate_registry": candidate,
        "candidate_dist_sha256": None if phase == "dist_backup" else _tree_hash(root / "dist/event_stack"),
        "backup_dist_path": None,
    }))

    completed = subprocess.run(
        [sys.executable, "-c", f"from scripts.release_event_stack import recover_release_transaction; recover_release_transaction(r'{root}')"],
        cwd=Path.cwd(), text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == candidate


def test_new_process_restores_real_verified_backup_from_dist_backup_phase(tmp_path: Path):
    """Crash after active dist -> backup replace restores the verified old pair."""

    from scripts.release_event_stack import _journal_path, _stable_json_bytes

    root, registry = _release_fixture(tmp_path)
    active = root / "dist/event_stack"
    backup = root / "dist/.event_stack.backup-real"
    os.replace(active, backup)
    assert not active.exists()
    _journal_path(root).write_bytes(_stable_json_bytes({
        "schema_version": 1, "phase": "dist_backup",
        "previous_registry": registry, "candidate_registry": registry,
        "candidate_dist_sha256": None, "backup_dist_path": str(backup.resolve()),
    }))

    completed = subprocess.run(
        [sys.executable, "-c", f"from scripts.release_event_stack import recover_release_transaction; recover_release_transaction(r'{root}')"],
        cwd=Path.cwd(), text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert active.is_dir() and not backup.exists()
    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry


def test_real_release_callback_persists_phase_before_package_replaces(tmp_path: Path, monkeypatch):
    """The callback wired into release_summary writes a durable phase before swaps."""
    from scripts.package_event_stack import package_event_stack
    from tests.pipeline.test_event_stack_dist import package_fixture_bundle
    import scripts.release_event_stack as release_module

    destination, bundle = package_fixture_bundle(tmp_path)
    root = tmp_path / "project"
    journal = {"schema_version": 1, "phase": "prepared", "previous_registry": {},
               "candidate_registry": {}, "candidate_dist_sha256": None, "backup_dist_path": None}
    callback = release_module.make_dist_journal_callback(root, journal)
    original_replace = __import__("scripts.package_event_stack", fromlist=["x"]).os.replace
    observed = []
    def inspect_replace(source, target):
        if Path(target).parent != destination.parent:
            return original_replace(source, target)
        persisted = json.loads(release_module._journal_path(root).read_text(encoding="utf-8"))
        observed.append((Path(source).name, Path(target).name, persisted["phase"]))
        return original_replace(source, target)
    monkeypatch.setattr(__import__("scripts.package_event_stack", fromlist=["x"]).os, "replace", inspect_replace)
    package_event_stack(bundle_path=bundle, destination=destination, trusted_dist_root=destination.parent, before_replace=callback)
    assert [item[2] for item in observed[:2]] == ["dist_backup", "dist_replaced"]


def test_orchestrator_equality_gate_writes_nothing(tmp_path: Path):
    """Equal F1 is rejected before any trainer/package side effect."""

    from src.pipeline.artifacts import PromotionContractError
    from scripts.release_event_stack import release_summary

    root, registry = _release_fixture(tmp_path)
    summary = root / "equal.json"
    summary.write_text(json.dumps({"outer_metrics": {"f1": registry["f1"]}}), encoding="utf-8")
    before = (root / "release/event_stack_incumbent.json").read_bytes()
    with pytest.raises(PromotionContractError, match="does not exceed incumbent"):
        release_summary(summary, root=root)
    assert (root / "release/event_stack_incumbent.json").read_bytes() == before
    assert not (root / "release/.event-stack-transaction.json").exists()


def test_orchestrator_refuses_invalid_incumbent_before_training(tmp_path: Path):
    from src.pipeline.artifacts import PromotionContractError
    from scripts.release_event_stack import release_summary

    root, registry = _release_fixture(tmp_path)
    bad = {**registry, "attestation_sha256": "0" * 64}
    (root / "release/event_stack_incumbent.json").write_text(json.dumps(bad), encoding="utf-8")
    summary = root / "candidate.json"
    summary.write_text(json.dumps({"outer_metrics": {"f1": 0.9}}), encoding="utf-8")
    with pytest.raises(PromotionContractError, match="incumbent registry"):
        release_summary(summary, root=root)


def test_orchestrator_refuses_invalid_diagnostic_set_before_training(tmp_path: Path):
    from src.pipeline.artifacts import PromotionContractError
    from scripts.release_event_stack import release_summary

    root, registry = _release_fixture(tmp_path)
    bad = {**registry, "diagnostic_set_sha256": "0" * 64}
    (root / "release/event_stack_incumbent.json").write_text(json.dumps(bad), encoding="utf-8")
    summary = root / "candidate.json"
    summary.write_text(json.dumps({"outer_metrics": {"f1": 0.9}}), encoding="utf-8")
    with pytest.raises(PromotionContractError, match="incumbent registry"):
        release_summary(summary, root=root)


def test_package_failure_leaves_recoverable_old_release(tmp_path: Path, monkeypatch):
    """Failure after candidate preparation leaves a durable journal and old pair."""

    import scripts.release_event_stack as release_module

    root, registry = _release_fixture(tmp_path)
    candidate = _distinct_candidate(root)
    # _distinct_candidate models an already-built candidate package; restore old active dist.
    shutil.rmtree(root / "dist/event_stack")
    shutil.copytree(Path("dist/event_stack"), root / "dist/event_stack")
    summary = root / "candidate.json"
    summary.write_text(json.dumps({"outer_metrics": {"f1": 0.9}}), encoding="utf-8")
    candidate_deployment = root / "models/event_stack" / str(candidate["run_key"]) / "deployment"
    monkeypatch.setattr(release_module, "promote_summary", lambda *_args, **_kwargs: (candidate_deployment,))
    monkeypatch.setattr(release_module, "package_event_stack", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("package failure")))

    with pytest.raises(RuntimeError, match="package failure"):
        release_module.release_summary(summary, root=root)
    assert release_module._journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry
    release_module.recover_release_transaction(root)
    assert not release_module._journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry


def test_registry_write_failure_recovers_complete_new_pair(tmp_path: Path, monkeypatch):
    """Once dist is installed, a registry write failure resumes the candidate pair."""

    import scripts.release_event_stack as release_module

    root, _ = _release_fixture(tmp_path)
    candidate = _distinct_candidate(root)
    shutil.rmtree(root / "dist/event_stack")
    shutil.copytree(Path("dist/event_stack"), root / "dist/event_stack")
    summary = root / "candidate.json"
    summary.write_text(json.dumps({"outer_metrics": {"f1": 0.9}}), encoding="utf-8")
    deployment = root / "models/event_stack" / str(candidate["run_key"]) / "deployment"
    monkeypatch.setattr(release_module, "promote_summary", lambda *_args, **_kwargs: (deployment,))
    original_write = release_module._atomic_write
    failed = {"value": False}
    def fail_registry(path, contents):
        if path.name == "event_stack_incumbent.json" and not failed["value"]:
            failed["value"] = True
            raise OSError("registry write failure")
        return original_write(path, contents)
    monkeypatch.setattr(release_module, "_atomic_write", fail_registry)

    with pytest.raises(OSError, match="registry write failure"):
        release_module.release_summary(summary, root=root)
    assert release_module._journal_path(root).exists()
    monkeypatch.setattr(release_module, "_atomic_write", original_write)
    release_module.recover_release_transaction(root)
    assert not release_module._journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == candidate


def test_post_copy_manifest_failure_recovers_complete_new_pair(tmp_path: Path, monkeypatch):
    """A package error after its swap leaves prepared journal, then detects new dist."""

    import scripts.release_event_stack as release_module

    root, _ = _release_fixture(tmp_path)
    candidate = _distinct_candidate(root)  # models a successfully copied new dist
    summary = root / "candidate.json"
    summary.write_text(json.dumps({"outer_metrics": {"f1": 0.9}}), encoding="utf-8")
    deployment = root / "models/event_stack" / str(candidate["run_key"]) / "deployment"
    monkeypatch.setattr(release_module, "promote_summary", lambda *_args, **_kwargs: (deployment,))
    monkeypatch.setattr(release_module, "package_event_stack", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("post-copy manifest failure")))

    with pytest.raises(RuntimeError, match="post-copy manifest failure"):
        release_module.release_summary(summary, root=root)
    assert release_module._journal_path(root).exists()
    release_module.recover_release_transaction(root)
    assert not release_module._journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == candidate


def test_recovery_after_dist_phase_finishes_a_verified_new_pair(tmp_path: Path):
    from scripts.release_event_stack import _journal_path, _stable_json_bytes, _tree_hash, recover_release_transaction

    root, registry = _release_fixture(tmp_path)
    journal = {
        "schema_version": 1,
        "phase": "dist_replaced",
        "previous_registry": registry,
        "candidate_registry": registry,
        "candidate_dist_sha256": _tree_hash(root / "dist/event_stack"),
        "backup_dist_path": None,
    }
    _journal_path(root).write_bytes(_stable_json_bytes(journal))

    recover_release_transaction(root)

    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry


def test_recovery_with_tampered_replaced_dist_retains_journal_for_safe_repair(tmp_path: Path):
    """A corrupt active package must never be silently paired with old metadata."""

    from src.pipeline.artifacts import PromotionContractError
    from scripts.release_event_stack import _journal_path, _stable_json_bytes, _tree_hash, recover_release_transaction

    root, registry = _release_fixture(tmp_path)
    journal = {
        "schema_version": 1,
        "phase": "dist_replaced",
        "previous_registry": registry,
        "candidate_registry": registry,
        "candidate_dist_sha256": _tree_hash(root / "dist/event_stack"),
        "backup_dist_path": None,
    }
    _journal_path(root).write_bytes(_stable_json_bytes(journal))
    (root / "dist/event_stack/requirements.txt").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(PromotionContractError, match="does not match"):
        recover_release_transaction(root)

    assert _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry


def test_recovery_after_post_copy_rollback_keeps_old_pair(tmp_path: Path):
    """A failed package can leave a callback journal but must not advance registry."""

    from scripts.release_event_stack import _journal_path, _stable_json_bytes, recover_release_transaction

    root, registry = _release_fixture(tmp_path)
    _journal_path(root).write_bytes(_stable_json_bytes({
        "schema_version": 1,
        "phase": "dist_replaced",
        "previous_registry": registry,
        "candidate_registry": registry,
        "candidate_dist_sha256": None,
        "backup_dist_path": None,
    }))

    recover_release_transaction(root)

    assert not _journal_path(root).exists()
    assert json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8")) == registry
