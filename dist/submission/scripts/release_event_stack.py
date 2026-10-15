"""Atomically advance the active event-stack release from registered evidence.

This is deliberately the only command which may replace both ``dist/event_stack``
and ``release/event_stack_incumbent.json``.  Its journal is durable enough to
finish a transaction after an interrupted process without ever trusting a
self-reported metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Mapping


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.package_event_stack import package_event_stack, verify_packaged_bundle
from scripts.promote_event_stack import registered_filesystem_trainer
from src.pipeline.artifacts import (
    PromotionContractError,
    _stable_json_bytes,
    incumbent_registry_payload,
    issue_release_transaction_token,
    load_incumbent_registry,
    promote_summary,
    validate_candidate_against_incumbent,
)


_JOURNAL = ".event-stack-transaction.json"


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        raise PromotionContractError("active dist is missing")
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    # Windows rejects ``fsync`` for a read-only CRT descriptor.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_parent(path: Path) -> None:
    """Best-effort directory durability; Windows may not expose directory FDs."""
    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_file(path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _journal_path(root: Path) -> Path:
    return root / "release" / _JOURNAL


def _write_journal(root: Path, payload: Mapping[str, object]) -> None:
    _atomic_write(_journal_path(root), _stable_json_bytes(dict(payload)))


def make_dist_journal_callback(root: Path, journal: dict[str, object]):
    """Return the sole fsynced callback used before every package replacement."""

    root = Path(root)
    def before_dist_replace(action: str, backup: Path) -> None:
        journal["backup_dist_path"] = str(backup.resolve())
        journal["phase"] = "dist_backup" if action == "backup" else "dist_replaced"
        _write_journal(root, journal)
    return before_dist_replace


def _load_journal(root: Path) -> dict[str, object] | None:
    path = _journal_path(root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionContractError("release journal cannot be read") from exc
    if not isinstance(payload, dict):
        raise PromotionContractError("release journal must be an object")
    return payload


def _active_dist_matches_candidate(root: Path, candidate: Mapping[str, object]) -> bool:
    """Recognize a fully copied candidate even before its tree hash is journaled."""

    run_key = candidate.get("run_key")
    if not isinstance(run_key, str):
        return False
    try:
        runtime = json.loads((root / "dist" / "event_stack" / "runtime_manifest.json").read_text(encoding="utf-8"))
        expected_manifest = hashlib.sha256(
            (root / "models" / "event_stack" / run_key / "deployment" / "manifest.json").read_bytes()
        ).hexdigest()
        return runtime.get("bundle_manifest_sha256") == expected_manifest
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def _journal_backup(root: Path, journal: Mapping[str, object]) -> Path | None:
    value = journal.get("backup_dist_path")
    if value is None:
        return None
    if not isinstance(value, str):
        raise PromotionContractError("release journal backup path is invalid")
    backup = Path(value)
    dist = (root / "dist").resolve()
    try:
        backup.resolve(strict=False).relative_to(dist)
    except ValueError as exc:
        raise PromotionContractError("release journal backup path is outside dist") from exc
    if not backup.name.startswith(".event_stack.backup-"):
        raise PromotionContractError("release journal backup path is invalid")
    return backup


def _restore_backup(root: Path, journal: Mapping[str, object], registry_path: Path, previous: Mapping[str, object]) -> None:
    backup = _journal_backup(root, journal)
    destination = root / "dist" / "event_stack"
    if backup is None or not backup.is_dir():
        raise PromotionContractError("active dist does not match transaction and no verified backup exists")
    verify_packaged_bundle(backup)
    _atomic_write(registry_path, _stable_json_bytes(previous))
    rollback = {**journal, "phase": "prepared"}
    _write_journal(root, rollback)  # fsync immediately before rollback replace
    if destination.exists():
        raise PromotionContractError("cannot restore backup over an unexpected active dist")
    os.replace(backup, destination)
    _journal_path(root).unlink()


def recover_release_transaction(root: Path = _ROOT) -> None:
    """Complete a verified interrupted transaction, or retain the old registry."""

    root = Path(root)
    journal = _load_journal(root)
    if journal is None:
        return
    required = {
        "schema_version", "phase", "previous_registry", "candidate_registry",
        "candidate_dist_sha256", "backup_dist_path",
    }
    if set(journal) != required or journal.get("schema_version") != 1:
        raise PromotionContractError("release journal schema is invalid")
    phase = journal["phase"]
    if phase not in {"prepared", "dist_backup", "dist_replaced", "registry_replaced", "verified"}:
        raise PromotionContractError("release journal phase is invalid")
    previous = journal["previous_registry"]
    candidate = journal["candidate_registry"]
    if not isinstance(previous, dict) or not isinstance(candidate, dict):
        raise PromotionContractError("release journal registry payload is invalid")
    registry_path = root / "release" / "event_stack_incumbent.json"
    if phase == "prepared":
        # An interruption may happen after the package's own atomic swap but
        # before this orchestrator records its tree hash.  Finish that new
        # release only when the copied deployment manifest identifies it.
        if not _active_dist_matches_candidate(root, candidate):
            if (root / "dist" / "event_stack").is_dir():
                verify_packaged_bundle(root / "dist" / "event_stack")
                _atomic_write(registry_path, _stable_json_bytes(previous))
                _journal_path(root).unlink()
                return
            _restore_backup(root, journal, registry_path, previous)
            return
        phase = "dist_replaced"
    if phase == "dist_backup" and not (root / "dist" / "event_stack").exists():
        _restore_backup(root, journal, registry_path, previous)
        return
    active_dist = root / "dist" / "event_stack"
    expected_tree = journal["candidate_dist_sha256"]
    if expected_tree is None and not _active_dist_matches_candidate(root, candidate):
        # The packager may have rolled back after a post-copy verification
        # failure.  Its callback left a durable journal, but the old verified
        # package is now active again, so retain the old registry and finish.
        if active_dist.is_dir():
            verify_packaged_bundle(active_dist)
            _atomic_write(registry_path, _stable_json_bytes(previous))
            _journal_path(root).unlink()
            return
        _restore_backup(root, journal, registry_path, previous)
        return
    if expected_tree is not None and _tree_hash(active_dist) != expected_tree:
        # We cannot prove whether this is the old package or a corrupted
        # partially replaced candidate.  Retain the journal and fail closed;
        # deleting it would make a mixed dist/registry pair look completed.
        if not active_dist.exists():
            _restore_backup(root, journal, registry_path, previous)
            return
        raise PromotionContractError("active dist does not match the transaction journal; recovery retained")
    verify_packaged_bundle(active_dist)
    run_key = candidate.get("run_key")
    if not isinstance(run_key, str):
        raise PromotionContractError("release journal candidate run key is invalid")
    expected = incumbent_registry_payload(root / "models" / "event_stack" / run_key)
    if candidate != expected:
        raise PromotionContractError("release journal candidate registry is not attested")
    _atomic_write(registry_path, _stable_json_bytes(candidate))
    load_incumbent_registry(registry_path, run_root=root / "models" / "event_stack" / run_key)
    final = {**journal, "phase": "verified"}
    _write_journal(root, final)
    backup = _journal_backup(root, journal)
    if backup is not None and backup.exists():
        shutil.rmtree(backup)
    _journal_path(root).unlink()


def release_summary(summary_path: Path, *, root: Path = _ROOT) -> Path:
    """Promote, package and atomically record one strictly improved summary."""

    root = Path(root)
    recover_release_transaction(root)
    registry_path = root / "release" / "event_stack_incumbent.json"
    incumbent = load_incumbent_registry(registry_path)
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        candidate_f1 = float(summary["outer_metrics"]["f1"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PromotionContractError("release summary cannot provide aggregate F1") from exc
    if isinstance(summary.get("run_configs"), list) and any(
        isinstance(item, Mapping) and item.get("context_features_version") == "v2"
        for item in summary["run_configs"]
    ):
        raise PromotionContractError(
            "context-v2 challenger requires human promotion/runtime implementation"
        )
    validate_candidate_against_incumbent(candidate_f1, float(incumbent["f1"]))

    # The trainer re-validates fold evidence before producing the candidate run.
    written = promote_summary(
        Path(summary_path), output_root=root / "models",
        trainer=registered_filesystem_trainer,
        release_token=issue_release_transaction_token(),
        active_release=True,
    )
    deployment = next(path for path in written if path.name == "deployment")
    candidate = incumbent_registry_payload(deployment.parent)
    journal = {
        "schema_version": 1,
        "phase": "prepared",
        "previous_registry": incumbent,
        "candidate_registry": candidate,
        "candidate_dist_sha256": None,
        "backup_dist_path": None,
    }
    _write_journal(root, journal)
    package_event_stack(
        bundle_path=deployment,
        destination=root / "dist" / "event_stack",
        trusted_dist_root=root / "dist",
        release_token=issue_release_transaction_token(),
        active_release=True,
        before_replace=make_dist_journal_callback(root, journal),
        retain_backup=True,
    )
    journal["candidate_dist_sha256"] = _tree_hash(root / "dist" / "event_stack")
    journal["phase"] = "dist_replaced"
    _write_journal(root, journal)
    _atomic_write(registry_path, _stable_json_bytes(candidate))
    journal["phase"] = "registry_replaced"
    _write_journal(root, journal)
    recover_release_transaction(root)
    return root / "dist" / "event_stack"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--recover", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.recover:
            recover_release_transaction()
        elif args.summary is not None:
            release_summary(args.summary)
        else:
            raise PromotionContractError("provide --summary or --recover")
    except (OSError, ValueError, RuntimeError, PromotionContractError) as exc:
        print(f"release refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
