"""Canonical per-subject event diagnostics for promoted experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Mapping, Sequence

from src.pipeline.event_stack import EventRef, compute_event_metrics


DIAGNOSTICS_SCHEMA_VERSION = 1


def _linear_percentile(values: Sequence[float], percentile: float) -> float | None:
    """Return the explicitly defined linear-interpolated percentile."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def subject_diagnostics(
    predictions: Sequence[EventRef],
    truths: Sequence[EventRef],
    subject_by_session: Mapping[str, str],
    *,
    runtime: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compute official event metrics per subject with explicit empty semantics.

    A subject with neither predictions nor truths has no defined F1 and is
    excluded from the F1 distribution.  Subjects with false positives or false
    negatives remain included with F1 zero.
    """

    grouped: dict[str, dict[str, list[EventRef]]] = {
        str(subject): {"predictions": [], "truths": []}
        for subject in subject_by_session.values()
    }
    for label, events in (("predictions", predictions), ("truths", truths)):
        for event in events:
            try:
                subject = subject_by_session[event.sid]
            except KeyError as exc:
                raise ValueError(f"{label} session is absent from subject mapping: {event.sid}") from exc
            grouped.setdefault(str(subject), {"predictions": [], "truths": []})[label].append(event)

    rows: dict[str, dict[str, object]] = {}
    defined_f1: list[float] = []
    for subject in sorted(grouped):
        events = grouped[subject]
        metrics = asdict(compute_event_metrics(events["predictions"], events["truths"]))
        empty = metrics["n_pred"] == 0 and metrics["n_true"] == 0
        metrics["f1"] = None if empty else metrics["f1"]
        rows[subject] = metrics
        if metrics["f1"] is not None:
            defined_f1.append(float(metrics["f1"]))
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "subjects": rows,
        "distribution": {
            "included": len(defined_f1),
            "excluded_empty": len(rows) - len(defined_f1),
            "f1_percentiles": {
                f"p{percentile}": _linear_percentile(defined_f1, percentile)
                for percentile in (0, 25, 50, 75, 100)
            },
        },
        "runtime": dict(runtime or {}),
    }


def paired_subject_diagnostics(
    incumbent: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, int]:
    """Compare subject F1 values over the union of both diagnostic records."""

    old_rows = incumbent.get("subjects", {})
    new_rows = candidate.get("subjects", {})
    if not isinstance(old_rows, Mapping) or not isinstance(new_rows, Mapping):
        raise ValueError("paired diagnostics require subject mappings")
    counts = {"improved": 0, "unchanged": 0, "worsened": 0}
    for subject in sorted(set(old_rows) | set(new_rows)):
        old = old_rows.get(subject, {})
        new = new_rows.get(subject, {})
        old_f1 = old.get("f1") if isinstance(old, Mapping) else None
        new_f1 = new.get("f1") if isinstance(new, Mapping) else None
        old_value = float(old_f1) if old_f1 is not None else None
        new_value = float(new_f1) if new_f1 is not None else None
        if old_value == new_value:
            counts["unchanged"] += 1
        elif old_value is None or (new_value is not None and new_value > old_value):
            counts["improved"] += 1
        else:
            counts["worsened"] += 1
    return counts


def aggregate_subject_diagnostics(
    diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Merge disjoint fold subject rows and recompute distribution statistics."""

    subjects: dict[str, object] = {}
    for payload in diagnostics:
        rows = payload.get("subjects", {})
        if not isinstance(rows, Mapping):
            raise ValueError("diagnostics subjects must be an object")
        overlap = set(subjects) & set(rows)
        if overlap:
            raise ValueError("subject diagnostics overlap across folds: " + ", ".join(sorted(overlap)))
        subjects.update({str(subject): row for subject, row in rows.items()})
    values = [
        float(row["f1"])
        for row in subjects.values()
        if isinstance(row, Mapping) and row.get("f1") is not None
    ]
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "subjects": {subject: subjects[subject] for subject in sorted(subjects)},
        "distribution": {
            "included": len(values),
            "excluded_empty": len(subjects) - len(values),
            "f1_percentiles": {
                f"p{percentile}": _linear_percentile(values, percentile)
                for percentile in (0, 25, 50, 75, 100)
            },
        },
    }


def canonical_diagnostics_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize a diagnostic record once for cache, CLI, and promotion use."""

    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
