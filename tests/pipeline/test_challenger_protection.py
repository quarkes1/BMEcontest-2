"""Behavioral contracts for the read-only challenger boundary."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from src.pipeline.challenger.protection import (
    assert_snapshot_unchanged,
    confined_output,
    protected_snapshot,
    validate_fingerprints,
)


_RUN_KEY = "160afaf81debf1ee"


def make_protected_fixture(tmp_path: Path) -> Path:
    """Build precisely the protected tree, with one ordinary file per root."""

    root = tmp_path / "project"
    paths = (
        "release/event_stack_incumbent.json",
        f"models/event_stack/{_RUN_KEY}/promotion_summary.json",
        "dist/inference/predict.py",
        "dist/submission/README.md",
        "dist/visual/index.html",
        "dist/event_stack/manifest.json",
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    return root


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_confined_output_rejects_parent_escape(tmp_path: Path):
    """A caller cannot send challenger output into crossfit evidence."""

    with pytest.raises(ValueError, match="outside challenger output root"):
        confined_output(tmp_path / "outputs/challenger_70", "../crossfit/x.json")


def test_confined_output_rejects_output_root_itself(tmp_path: Path):
    """A file-writing caller cannot treat the output directory as a file."""

    with pytest.raises(ValueError, match="outside challenger output root"):
        confined_output(tmp_path / "outputs/challenger_70", ".")


def test_snapshot_reports_modified_release(tmp_path: Path):
    """An accidental incumbent content edit stops the challenger immediately."""

    root = make_protected_fixture(tmp_path)
    before = protected_snapshot(root)
    (root / "release/event_stack_incumbent.json").write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="protected release input changed"):
        assert_snapshot_unchanged(before, root)


def test_snapshot_rejects_added_file_and_file_directory_replacement(tmp_path: Path):
    """The guard catches both a new dist artifact and a path-type substitution."""

    root = make_protected_fixture(tmp_path)
    before = protected_snapshot(root)
    (root / "dist/visual/new.js").write_text("new", encoding="utf-8")

    with pytest.raises(RuntimeError, match="protected release input changed"):
        assert_snapshot_unchanged(before, root)

    root = make_protected_fixture(tmp_path / "second")
    before = protected_snapshot(root)
    target = root / "dist/submission/README.md"
    target.unlink()
    target.mkdir()

    with pytest.raises(RuntimeError, match="protected release input changed"):
        assert_snapshot_unchanged(before, root)


def test_validate_fingerprints_rejects_changed_content(tmp_path: Path):
    """A stale digest cannot silently authorize source data to be replayed."""

    path = tmp_path / "source.npz"
    path.write_bytes(b"before")
    rows = [fingerprint(path)]
    before = path.stat()
    path.write_bytes(b"after!")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(ValueError, match="SHA-256"):
        validate_fingerprints(rows)


def test_validate_fingerprints_rejects_changed_mtime(tmp_path: Path):
    """The replay identity includes nanosecond modification time, not just bytes."""

    path = tmp_path / "source.npz"
    path.write_bytes(b"same")
    rows = [fingerprint(path)]
    original = path.stat().st_mtime_ns
    os.utime(path, ns=(path.stat().st_atime_ns, original + 1_000_000_000))

    with pytest.raises(ValueError, match="mtime_ns"):
        validate_fingerprints(rows)


def test_validate_fingerprints_rejects_duplicate_path(tmp_path: Path):
    """Duplicate source identity rows make an evidence set ambiguous."""

    path = tmp_path / "source.npz"
    path.write_bytes(b"bytes")
    row = fingerprint(path)

    with pytest.raises(ValueError, match="duplicate"):
        validate_fingerprints([row, dict(row)])


def test_validate_fingerprints_rejects_unexpected_present_path_for_missing_row(tmp_path: Path):
    """A claimed-missing source must actually be absent at replay time."""

    path = tmp_path / "appeared.npz"
    path.write_bytes(b"unexpected")
    rows = [{"path": str(path.resolve()), "missing": True}]

    with pytest.raises(ValueError, match="expected missing"):
        validate_fingerprints(rows)


def test_validate_fingerprints_rejects_missing_present_path(tmp_path: Path):
    """A required evidence input cannot disappear after the bundle was sealed."""

    path = tmp_path / "gone.npz"
    path.write_bytes(b"present")
    rows = [fingerprint(path)]
    path.unlink()

    with pytest.raises(ValueError, match="required source input is missing"):
        validate_fingerprints(rows)
