"""Fail-closed I/O guards for challenger E00/E01 research.

The challenger may inspect a frozen release and emit diagnostics below its own
output root.  This module deliberately has no writer besides path confinement:
callers remain responsible for atomically writing the files they create.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PureWindowsPath
from typing import Mapping, Sequence

from src.pipeline.artifacts import _is_source_fingerprint


_RUN_KEY = "160afaf81debf1ee"
_PROTECTED_PATHS = (
    "release/event_stack_incumbent.json",
    f"models/event_stack/{_RUN_KEY}",
    "dist/inference",
    "dist/submission",
    "dist/visual",
    "dist/event_stack",
)
_REPARSE_POINT = 0x400  # Windows FILE_ATTRIBUTE_REPARSE_POINT


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & _REPARSE_POINT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unsafe(path: Path) -> RuntimeError:
    return RuntimeError(f"protected release input changed: symlink/reparse point at {path}")


def _record_tree(root: Path, target: Path, snapshot: dict[str, tuple[str, str | None]]) -> None:
    """Record a protected target recursively without following link-like nodes."""

    if _is_link_or_reparse(target):
        raise _unsafe(target)
    if target.is_file():
        snapshot[target.relative_to(root).as_posix()] = ("file", _sha256(target))
        return
    if not target.is_dir():
        raise RuntimeError(f"protected release input changed: missing or unsafe path {target}")

    snapshot[target.relative_to(root).as_posix()] = ("directory", None)
    with os.scandir(target) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            child = Path(entry.path)
            if _is_link_or_reparse(child):
                raise _unsafe(child)
            if entry.is_dir(follow_symlinks=False):
                _record_tree(root, child, snapshot)
            elif entry.is_file(follow_symlinks=False):
                snapshot[child.relative_to(root).as_posix()] = ("file", _sha256(child))
            else:
                raise RuntimeError(
                    f"protected release input changed: non-ordinary path {child}"
                )


def protected_snapshot(root: Path) -> dict[str, tuple[str, str | None]]:
    """Hash the complete, fixed incumbent and distribution input surface.

    The snapshot intentionally excludes ``outputs/challenger_70`` and every
    unlisted project path, so challenger diagnostics cannot trigger a false
    release mutation alarm.
    """

    base = Path(root).absolute()
    if _is_link_or_reparse(base):
        raise _unsafe(base)
    if not base.is_dir():
        raise RuntimeError(f"protected release input changed: project root is missing {base}")
    snapshot: dict[str, tuple[str, str | None]] = {}
    for relative in _PROTECTED_PATHS:
        _record_tree(base, base / relative, snapshot)
    return snapshot


def assert_snapshot_unchanged(
    before: Mapping[str, tuple[str, str | None]], root: Path
) -> None:
    """Raise if any protected release/dist file tree differs from ``before``."""

    after = protected_snapshot(root)
    if dict(before) != after:
        raise RuntimeError("protected release input changed")


def validate_fingerprints(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate canonical evidence fingerprints against the current filesystem.

    This validates exactly the contract used by persisted bundle manifests.  A
    present record must still identify the same ordinary file; a missing record
    must still be absent.  Cross-checking against a fold's *expected* path set
    is intentionally performed by replay, where the relevant RunConfig exists.
    """

    normalized: list[dict[str, object]] = []
    for row in rows:
        record = dict(row)
        if not _is_source_fingerprint(record):
            raise ValueError("source fingerprint violates the canonical contract")
        normalized.append(record)
    paths = [str(row["path"]) for row in normalized]
    if len(paths) != len(set(paths)):
        raise ValueError("source fingerprint set contains a duplicate path")

    for row in normalized:
        path = Path(str(row["path"]))
        exists = os.path.lexists(path)
        claimed_missing = set(row) == {"path", "missing"}
        if claimed_missing:
            if exists:
                raise ValueError(f"source fingerprint expected missing path is present: {path}")
            continue
        if not exists:
            raise ValueError(f"required source input is missing: {path}")
        if _is_link_or_reparse(path):
            raise ValueError(f"source fingerprint path is unsafe: {path}")
        if not path.is_file():
            raise ValueError(f"required source input is not an ordinary file: {path}")
        stat = path.stat()
        if stat.st_size != row["size"]:
            raise ValueError(f"source fingerprint size does not match: {path}")
        if stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError(f"source fingerprint mtime_ns does not match: {path}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"source fingerprint SHA-256 does not match: {path}")
    return tuple(normalized)


def confined_output(root: Path, relative: str) -> Path:
    """Resolve one challenger output file strictly below ``root``.

    ``relative`` may not name the root itself, an absolute path, or a path which
    resolves through ``..`` outside the challenger output directory.
    """

    windows_relative = PureWindowsPath(relative)
    if Path(relative).is_absolute() or windows_relative.drive or windows_relative.root:
        raise ValueError("outside challenger output root")
    output_root = Path(root).resolve(strict=False)
    candidate = (output_root / relative).resolve(strict=False)
    if candidate == output_root or output_root not in candidate.parents:
        raise ValueError("outside challenger output root")
    return candidate
