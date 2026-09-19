"""Canonical raw-data inference for a verified frozen event-stack bundle."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from event_stack.artifacts import EventStackBundle, load_event_stack_bundle
from event_stack.candidate_control import CandidateAdmissionConfig, admit_candidates
from event_stack.event_stack import (
    DensityConfig, EventRef, MicroCandidateConfig, apply_event_policy,
    density_candidates, micro_candidates, multiscale_verifier_features, union_candidates,
)
from event_stack.features.macro import MacroFeatureConfig, add_time_prior, extract_macro_windows
from event_stack.features.micro import extract_micro_windows
from event_stack.imu_features import MicroFeatureConfig
from event_stack.io.raw_session import RawSessionSource, discover_raw_sessions, load_raw_session
from event_stack.preprocessing.timeline import valid_imu_spans

from .legacy_payload import resolve_device
from .schema import make_prediction_result, validate_prediction


@dataclass(frozen=True)
class PredictionOptions:
    include_timeline: bool = False
    include_candidates: bool = False
    device: Literal["auto", "cpu", "gpu", "cuda"] = "auto"


@dataclass(frozen=True)
class InferenceTrace:
    """Read-only, layer-by-layer evidence from the frozen raw inference graph.

    Threshold comparisons in the frozen policy use ``score >= threshold``.
    The thresholds and tie rule are retained here so a score within the
    numerical boundary neighbourhood is auditable rather than silently
    excluded from a parity report.
    """

    spans: tuple[tuple[str, int, int], ...]
    macro_features_62: np.ndarray
    macro_features_63: np.ndarray
    micro_features: np.ndarray
    context_features: np.ndarray
    macro_probabilities: np.ndarray
    micro_probabilities: np.ndarray
    candidates: tuple[dict[str, object], ...]
    admitted: tuple[dict[str, object], ...]
    verifier_scores: np.ndarray
    events: tuple[dict[str, object], ...]
    admission_threshold: float
    event_threshold: float
    threshold_tie_rule: str = "score >= threshold"
    decision_boundary_epsilon: float = 1e-12
    # Optional window references (session_id, start_ms, end_ms) aligned with
    # macro/micro probabilities; used only to emit the optional, presentation-only
    # timeline series. They never feed candidates, scoring, or the event decoder.
    macro_window_refs: tuple[tuple[str, int, int], ...] = ()
    micro_window_refs: tuple[tuple[str, int, int], ...] = ()


def _positive_probability(model: object, features: np.ndarray, width: int, name: str) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} model features must have width {width}")
    if not len(values):
        return np.empty(0, dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    classes = np.asarray(model.classes_)
    column = np.flatnonzero(classes == 1)
    if probabilities.shape != (len(values), 2) or len(column) != 1 or not np.isfinite(probabilities).all():
        raise ValueError(f"{name} model returned incompatible probabilities")
    positive = probabilities[:, int(column[0])]
    if np.any(positive < 0.0) or np.any(positive > 1.0):
        raise ValueError(f"{name} model returned incompatible probabilities")
    return positive


def _by_session(windows: Sequence[EventRef], scores: np.ndarray) -> dict[str, list[tuple[int, int, float]]]:
    if len(windows) != len(scores):
        raise ValueError("window scores must align with windows")
    result: dict[str, list[tuple[int, int, float]]] = {}
    for window, score in zip(windows, scores):
        result.setdefault(window.sid, []).append((window.start_ms, window.end_ms, float(score)))
    return result


def _timeline_series(trace: "InferenceTrace") -> list[dict[str, object]]:
    """Optional per-timestamp probability series (presentation-only, schema v1.0).

    Macro and micro windows have different cadences; the series samples the union of
    their window starts and holds each stream's latest value (a step function). This is
    deterministic, label-free, and never alters candidates, scores, or decoded events.
    """
    by_session: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for refs, probabilities, key in (
        (trace.macro_window_refs, trace.macro_probabilities, "macro_probability"),
        (trace.micro_window_refs, trace.micro_probabilities, "micro_probability"),
    ):
        if len(refs) != len(probabilities):
            return []
        for (sid, start_ms, _end_ms), value in zip(refs, probabilities):
            by_session.setdefault(str(sid), {"macro_probability": [], "micro_probability": []})[key].append((int(start_ms), float(value)))
    points: list[dict[str, object]] = []
    for sid in sorted(by_session):
        streams = by_session[sid]
        macro = dict(streams["macro_probability"])
        micro = dict(streams["micro_probability"])
        last_macro = last_micro = 0.0
        for timestamp in sorted(set(macro) | set(micro)):
            if timestamp in macro:
                last_macro = macro[timestamp]
            if timestamp in micro:
                last_micro = micro[timestamp]
            points.append({
                "session_id": sid, "timestamp_ms": timestamp,
                "macro_probability": last_macro, "micro_probability": last_micro,
                "valid": True, "gap": False,
            })
    return points


def _config(raw: object, cls):
    if not isinstance(raw, dict):
        raise ValueError("frozen run config is malformed")
    return cls(**raw)


class Predictor:
    """Inference-only facade; it never reads caches, splits, labels, or training data."""

    def __init__(self, bundle: EventStackBundle, *, bundle_path: Path, run_key: str, device: str = "auto") -> None:
        self._bundle = bundle
        self._bundle_path = Path(bundle_path)
        self._run_key = run_key
        self._device = resolve_device(device)
        schema = bundle.feature_schema
        widths = schema["widths"] if "widths" in schema else schema
        if dict(widths) != {"macro": 63, "micro": 47, "verifier": 116}:
            raise ValueError("raw Predictor supports only the frozen 63/47/116 release schema")
        configs = bundle.run_config.get("fold_configs")
        if not isinstance(configs, list) or not configs or not isinstance(configs[0], dict):
            raise ValueError("bundle has no frozen fold configuration")
        frozen = configs[0]
        self._density = _config(frozen.get("density"), DensityConfig)
        self._micro_candidate = _config(frozen.get("micro_candidate"), MicroCandidateConfig)
        self._macro_config = MacroFeatureConfig(
            window_ms=self._density.window_ms, stride_ms=self._density.stride_ms,
            coverage_min=self._density.coverage_min,
        )
        self._micro_config = MicroFeatureConfig(
            window_ms=self._micro_candidate.window_ms, stride_ms=self._micro_candidate.stride_ms,
            coverage_min=self._density.coverage_min, gravity_align=bool(frozen.get("micro_gravity_align")),
        )

    @classmethod
    def from_bundle(cls, path: Path, *, device: str = "auto", run_key: str | None = None) -> "Predictor":
        bundle_path = Path(path)
        try:
            declared_run_key = str(json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))["run_key"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("bundle manifest run key cannot be read") from exc
        bundle = load_event_stack_bundle(bundle_path, expected_role="deployment", expected_run_key=declared_run_key)
        if run_key is not None:
            # Standalone distributions carry the frozen release key in the
            # package manifest; the caller passes it explicitly.
            if not isinstance(run_key, str) or not run_key:
                raise ValueError("explicit run key must be a non-empty string")
            return cls(bundle, bundle_path=bundle_path, run_key=run_key, device=device)
        run_key = bundle_path.parent.name
        # Repository use can point at dist/event_stack/bundle; its release key
        # is tracked by the immutable registry, never inferred from features.
        for parent in (bundle_path, *bundle_path.parents):
            registry = parent / "release" / "event_stack_incumbent.json"
            if registry.is_file():
                try:
                    run_key = str(json.loads(registry.read_text(encoding="utf-8"))["run_key"])
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                break
        return cls(bundle, bundle_path=bundle_path, run_key=run_key, device=device)

    @staticmethod
    def options(**kwargs: object) -> PredictionOptions:
        return replace(PredictionOptions(), **kwargs)

    def predict_file(self, path: Path, *, subject_id: str | None = None,
                     options: PredictionOptions = PredictionOptions()) -> dict[str, object]:
        path = Path(path)
        return self.predict_sources((RawSessionSource(path, path.stem, subject_id),), options)

    def trace_file(self, path: Path, *, subject_id: str | None = None) -> InferenceTrace:
        """Expose frozen intermediate values for parity tests and release audits."""
        path = Path(path)
        return self._trace_sources((RawSessionSource(path, path.stem, subject_id),))

    def predict_folder(self, path: Path, *, subject_id: str | None = None,
                       options: PredictionOptions = PredictionOptions()) -> dict[str, object]:
        sources = tuple(replace(source, subject_id=subject_id) for source in discover_raw_sessions(Path(path)))
        return self.predict_sources(sources, options)

    def predict_sources(self, sources: Sequence[RawSessionSource], options: PredictionOptions = PredictionOptions()) -> dict[str, object]:
        if not sources:
            raise ValueError("at least one raw source is required")
        resolved_device = resolve_device(options.device)
        # A future audited GPU adapter belongs here; cpu is the registered release runtime.
        if resolved_device != self._device and self._device != "cpu":
            raise RuntimeError("Predictor device state is inconsistent")
        trace = self._trace_sources(sources)
        source_names = [str(source.path) for source in sources]
        total_duration, covered_duration, gap_rows = self._source_diagnostics(sources)
        events = [dict(event) for event in trace.events]
        result = make_prediction_result(run_key=self._run_key, source=source_names[0] if len(source_names) == 1 else str(self._bundle_path), duration_seconds=total_duration, events=events)
        result["diagnostics"] = {
            "coverage": covered_duration / total_duration if total_duration else 0.0,
            "warnings": [], "resolved_device": resolved_device,
        }
        if options.include_candidates:
            admitted_keys = {(str(row["session_id"]), int(row["start_ms"]), int(row["end_ms"])) for row in trace.admitted}
            # Public rows carry exactly the published schema keys; the internal
            # trace keeps its wider diagnostic rows (has_macro/has_micro).
            result["candidates"] = [
                {
                    "session_id": str(row["session_id"]),
                    "start_ms": int(row["start_ms"]),
                    "end_ms": int(row["end_ms"]),
                    "score": float(row["score"]),
                    "admitted": (str(row["session_id"]), int(row["start_ms"]), int(row["end_ms"])) in admitted_keys,
                }
                for row in trace.candidates
            ]
        if options.include_timeline:
            result["timeline"] = {
                "session_ids": [source.session_id for source in sources],
                "macro_windows": len(trace.macro_features_62), "micro_windows": len(trace.micro_features),
            }
            series = _timeline_series(trace)
            if series:
                result["timeline"]["series"] = series
            result["gaps"] = gap_rows
        validate_prediction(result)
        return result

    def _source_diagnostics(self, sources: Sequence[RawSessionSource]) -> tuple[float, float, list[dict[str, object]]]:
        total_duration = covered_duration = 0.0
        gap_rows: list[dict[str, object]] = []
        for source in sources:
            spans = valid_imu_spans(load_raw_session(source))
            if not spans:
                continue
            start, end = min(span.start_ms for span in spans), max(span.end_ms for span in spans)
            total_duration += (end - start) / 1000.0
            covered_duration += sum((span.end_ms - span.start_ms) / 1000.0 for span in spans)
            for left, right in zip(spans, spans[1:]):
                gap_rows.append({"session_id": source.session_id, "start_ms": left.end_ms, "end_ms": right.start_ms})
        return total_duration, covered_duration, gap_rows

    def _trace_sources(self, sources: Sequence[RawSessionSource], *, session_loader=load_raw_session) -> InferenceTrace:
        """Run the one canonical graph, optionally with a legacy raw loader for audit."""
        if not sources:
            raise ValueError("at least one raw source is required")
        macro_windows: list[EventRef] = []
        macro_rows: list[np.ndarray] = []
        micro_windows: list[EventRef] = []
        micro_rows: list[np.ndarray] = []
        bounds: dict[str, tuple[int, int]] = {}
        spans: list[tuple[str, int, int]] = []
        subject_by_sid: dict[str, str] = {}
        for source in sources:
            session = session_loader(source)
            sid = source.session_id
            if sid in subject_by_sid:
                raise ValueError("raw session identifiers must be unique")
            valid_spans = valid_imu_spans(session)
            if valid_spans:
                bounds[sid] = (min(span.start_ms for span in valid_spans), max(span.end_ms for span in valid_spans))
                spans.extend((sid, int(span.start_ms), int(span.end_ms)) for span in valid_spans)
            subject_by_sid[sid] = source.subject_id or sid
            macro = extract_macro_windows(session, session_id=sid, config=self._macro_config)
            micro = extract_micro_windows(session, session_id=sid, config=self._micro_config)
            macro_windows.extend(macro.windows); macro_rows.append(macro.features)
            micro_windows.extend(micro.windows); micro_rows.append(micro.features)
        macro_features_62 = np.concatenate(macro_rows, axis=0) if macro_rows else np.empty((0, 62), dtype=np.float32)
        macro_features = add_time_prior(macro_features_62, macro_windows)
        micro_features = np.concatenate(micro_rows, axis=0) if micro_rows else np.empty((0, 47), dtype=np.float32)
        macro_scores = _positive_probability(self._bundle.models["macro"], macro_features, 63, "macro")
        micro_scores = _positive_probability(self._bundle.models["micro"], micro_features, 47, "micro")
        macro_by_sid = _by_session(macro_windows, macro_scores)
        micro_by_sid = _by_session(micro_windows, micro_scores)
        candidates = union_candidates(density_candidates(macro_by_sid, self._density), micro_candidates(micro_by_sid, float(self._bundle.run_config["micro_threshold"]), self._micro_candidate))
        scored: np.ndarray
        if candidates:
            features = multiscale_verifier_features(candidates, macro_by_sid, micro_by_sid, context_features_version="v1", session_bounds_by_sid=bounds)
            if features.shape[1] != 116:
                raise RuntimeError("canonical verifier did not produce frozen 116-D features")
            logistic = _positive_probability(self._bundle.models["verifier_logistic"], features, 116, "verifier")
            lgbm = _positive_probability(self._bundle.models["verifier_lgbm"], features, 116, "verifier")
            weight = float(self._bundle.policy["blend_weight"])
            scored = weight * logistic + (1.0 - weight) * lgbm
        else:
            scored = np.empty(0, dtype=np.float64)
            features = np.empty((0, 116), dtype=np.float64)
        admission = CandidateAdmissionConfig(float(self._bundle.policy["nms_iou"]), float(self._bundle.policy["admission_threshold"]), self._bundle.policy["max_candidates_per_subject"])
        admitted_indices = admit_candidates(candidates, scored, [subject_by_sid[c.event.sid] for c in candidates], admission)
        admitted = [candidates[index] for index in admitted_indices]
        admitted_scores = scored[list(admitted_indices)] if admitted_indices else np.empty(0, dtype=np.float64)
        final = apply_event_policy([candidate.event for candidate in admitted], admitted_scores, [subject_by_sid[candidate.event.sid] for candidate in admitted], float(self._bundle.policy["threshold"]), self._bundle.policy["max_events_per_group"])
        confidence_by_event = {(candidate.event.sid, candidate.event.start_ms, candidate.event.end_ms): float(score) for candidate, score in zip(admitted, admitted_scores)}
        events = tuple({"id": index, "session_id": event.sid, "start_ms": event.start_ms, "end_ms": event.end_ms,
                   "duration_s": (event.end_ms - event.start_ms) / 1000.0,
                   "confidence": confidence_by_event[(event.sid, event.start_ms, event.end_ms)]}
                  for index, event in enumerate(final))
        candidate_rows = tuple(
            {"session_id": candidate.event.sid, "start_ms": candidate.event.start_ms,
             "end_ms": candidate.event.end_ms, "score": float(score),
             "has_macro": candidate.macro is not None, "has_micro": candidate.micro is not None}
            for candidate, score in zip(candidates, scored)
        )
        admitted_rows = tuple(candidate_rows[index] for index in admitted_indices)
        return InferenceTrace(
            spans=tuple(spans), macro_features_62=macro_features_62, macro_features_63=macro_features,
            micro_features=micro_features, context_features=features[:, 56:],
            macro_probabilities=macro_scores, micro_probabilities=micro_scores,
            candidates=candidate_rows, admitted=admitted_rows, verifier_scores=scored,
            events=events, admission_threshold=float(admission.threshold),
            event_threshold=float(self._bundle.policy["threshold"]),
            macro_window_refs=tuple((w.sid, int(w.start_ms), int(w.end_ms)) for w in macro_windows),
            micro_window_refs=tuple((w.sid, int(w.start_ms), int(w.end_ms)) for w in micro_windows),
        )
