"""Fail-closed source and learned-ancestor checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePosixPath


_ALLOWED_PREFIXES = (
    "cache/sessions/",
    "cache/splits/",
    "cache/micro15/",
    "FDdatasets/doi-10.48804-CN8VBB/v1.0/",
)


def validate_source(source: str) -> str:
    if not isinstance(source, str) or not source:
        raise ValueError("forbidden source")
    normalized = source.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not normalized.startswith(_ALLOWED_PREFIXES):
        raise ValueError(f"forbidden source: {source}")
    return normalized


def cache_identity(payload: Mapping[str, object]) -> str:
    required = {"schema", "source", "fit_subjects"}
    if not required <= payload.keys() or any(payload[key] is None for key in required):
        raise ValueError("cache identity lacks required provenance")
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def assert_no_heldout_ancestors(
    artifact_id: str, heldout_subjects: set[str], graph: Mapping[str, Mapping[str, object]]
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError(f"provenance cycle at {node_id}")
        if node_id in visited:
            return
        node = graph.get(node_id)
        if node is None:
            raise ValueError(f"unknown provenance parent: {node_id}")
        if set(node) != {"fit_subjects", "parents"}:
            raise ValueError(f"incomplete provenance node: {node_id}")
        subjects = node["fit_subjects"]
        parents = node["parents"]
        if not isinstance(subjects, (list, tuple, set)) or not isinstance(parents, (list, tuple)):
            raise ValueError(f"malformed provenance node: {node_id}")
        if any(not isinstance(subject, str) or not subject for subject in subjects):
            raise ValueError(f"malformed fit subject in {node_id}")
        overlap = set(subjects) & heldout_subjects
        if overlap:
            raise ValueError(f"held-out subject in learned ancestors: {sorted(overlap)}")
        visiting.add(node_id)
        for parent in parents:
            if not isinstance(parent, str) or not parent:
                raise ValueError(f"malformed parent in {node_id}")
            walk(parent)
        visiting.remove(node_id)
        visited.add(node_id)

    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("missing artifact id")
    walk(artifact_id)
