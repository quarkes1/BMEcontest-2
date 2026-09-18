"""Local inference bridge for the visualization application (stdlib only).

A thin HTTP adapter around the canonical :class:`Predictor`: it accepts uploaded raw
``collect_data*.txt`` sessions plus optional static visualization files, runs the
canonical pipeline, and returns the published prediction contract together with
visualization-only motion telemetry (``motion.bin``, record format
``f64_ms_6xf32_le``).  It never re-implements parsing, features, candidates,
verification, or decoding — the dependency direction is strictly

    local inference bridge -> canonical Predictor

Security posture (single user, localhost only): binds 127.0.0.1, stages uploads in
collision-safe temporary directories, sanitizes every client-provided name, refuses
path traversal, and never modifies the user's original files.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid
import webbrowser

import numpy as np

from ..io.raw_session import RawSessionSource, discover_raw_sessions, load_raw_session
from ..preprocessing.timeline import timeline_regressions, valid_imu_spans
from .predictor import PredictionOptions, Predictor
from .schema import make_prediction_result

SESSION_FILENAME = re.compile(r"^collect_data\d+_\d+_\d+\.txt$")
SEGMENT_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_RELATIVE_SEGMENTS = 6
SERVICE_NAME = "eatingsense-local-inference"


def sanitize_segment(name: str, *, fallback: str = "session", limit: int = 80) -> str:
    """Client-provided names never reach the filesystem unsanitized."""
    cleaned = SEGMENT_UNSAFE.sub("_", str(name)).strip("._")
    return cleaned[:limit] or fallback


def safe_relative(relative: str | None) -> list[str]:
    """Split a browser-provided relative path into safe segments (traversal refused)."""
    if not relative:
        return []
    parts = [part for part in re.split(r"[\\/]+", str(relative)) if part not in ("", ".", "..")]
    return [sanitize_segment(part) for part in parts[:MAX_RELATIVE_SEGMENTS]]


def motion_records(session: object) -> tuple[bytes, int]:
    """Little-endian records: float64 timestamp (ms) + six float32 raw IMU channels.

    Mirrors ``dist/visual/tools/export-motion.mjs`` (ACC_TIME timestamps, gyro paired
    by source row, raw ADC values passed through unchanged).
    """
    acc = np.asarray(getattr(session, "acc"), dtype=np.float32)
    gyro = np.asarray(getattr(session, "gyro"), dtype=np.float32)
    timestamps = np.asarray(getattr(session, "t_acc"))
    valid = np.asarray(getattr(session, "imu_valid"), dtype=bool) & (timestamps > 0)
    rows = np.nonzero(valid)[0]
    if rows.size:
        finite = np.isfinite(acc[:, rows]).all(axis=0) & np.isfinite(gyro[:, rows]).all(axis=0)
        rows = rows[finite]
    skipped = int(len(timestamps) - rows.size)
    n = int(rows.size)
    if not n:
        return b"", skipped
    records = np.zeros((n, 32), dtype=np.uint8)
    records[:, :8] = np.ascontiguousarray(timestamps[rows], dtype="<f8").view(np.uint8).reshape(n, 8)
    channels = np.vstack([acc[:, rows], gyro[:, rows]]).T
    records[:, 8:] = np.ascontiguousarray(channels, dtype="<f4").view(np.uint8).reshape(n, 24)
    return records.tobytes(), skipped


def estimate_gravity_calibration(session: object) -> dict[str, object] | None:
    """Estimate raw ACC counts/g from at least three nonoverlapping quiet 10 s spans."""
    acc = np.asarray(session.acc, dtype=np.float64)
    estimates: list[float] = []
    for span in valid_imu_spans(session):
        timestamps = span.timestamps_ms
        rows = span.row_indices
        start = 0
        while start < len(rows):
            stop = int(np.searchsorted(timestamps, timestamps[start] + 10_000, side="left"))
            if stop >= len(rows):
                break
            magnitudes = np.linalg.norm(acc[:, rows[start:stop + 1]], axis=0)
            mean = float(np.mean(magnitudes))
            if mean > 0 and np.isfinite(magnitudes).all() and float(np.std(magnitudes)) / mean < .05:
                estimates.append(float(np.median(magnitudes)))
            start = stop + 1
    if len(estimates) < 3:
        return None
    return {"acceleration_counts_per_g": float(np.median(estimates)),
            "viewer_from_sensor": [1, 0, 0, 0, 0, 1, 0, -1, 0]}


class LocalInferenceService:
    """Owns the canonical Predictor, staging directories, and per-analysis artifacts."""

    def __init__(self, bundle_path: Path, *, run_key: str, visual_dir: Path | None = None,
                 temp_root: Path | None = None) -> None:
        self.bundle_path = Path(bundle_path)
        self.run_key = str(run_key)
        self.visual_dir = Path(visual_dir).resolve() if visual_dir else None
        self.temp_root = Path(temp_root) if temp_root else Path(tempfile.mkdtemp(prefix="eatingsense-"))
        (self.temp_root / "uploads").mkdir(parents=True, exist_ok=True)
        (self.temp_root / "analyses").mkdir(parents=True, exist_ok=True)
        self._uploads: dict[str, dict[str, object]] = {}
        self._analyses: dict[str, dict[str, object]] = {}
        self._lock = threading.Lock()
        self._predictor: Predictor | None = None

    # ------------------------------------------------------------- inference
    def predictor(self) -> Predictor:
        if self._predictor is None:
            self._predictor = Predictor.from_bundle(self.bundle_path, device="cpu", run_key=self.run_key)
        return self._predictor

    def stage_upload(self, name: str, body: bytes, relative: str | None) -> str:
        token = uuid.uuid4().hex
        target = self.temp_root / "uploads" / f"{token}.bin"
        target.write_bytes(body)
        self._uploads[token] = {"path": target, "name": str(name), "relative": relative}
        return token

    def analyze(self, tokens: list[str], *, include_timeline: bool = True, include_candidates: bool = True) -> dict[str, object]:
        staged: list[dict[str, object]] = []
        warnings: list[str] = []
        for token in tokens:
            entry = self._uploads.get(str(token))
            if entry is None:
                warnings.append(f"Unknown upload token {token}.")
                continue
            staged.append(entry)
        if not staged:
            raise ValueError("No uploaded session files to analyze.")
        analysis_id = uuid.uuid4().hex[:16]
        root = self.temp_root / "analyses" / analysis_id / "src"
        session_dirs: dict[Path, list[Path]] = {}
        seen_folders: set[str] = set()
        for entry in staged:
            name = str(entry["name"])
            if not SESSION_FILENAME.fullmatch(name):
                warnings.append(f"Ignored {name}: not a collect_data*.txt session file.")
                continue
            segments = safe_relative(entry["relative"] if isinstance(entry["relative"], str) else None)
            if segments and SESSION_FILENAME.fullmatch(segments[-1]):
                segments = segments[:-1]
            if not segments:
                stem = sanitize_segment(Path(name).stem)
                folder = stem
                suffix = 2
                while folder in seen_folders:
                    folder = f"{stem}-{suffix}"
                    suffix += 1
                seen_folders.add(folder)
                segments = [folder]
            folder_path = root.joinpath(*segments)
            target = folder_path / name
            if target.exists():
                # Keep the canonical filename intact (discovery depends on it) and
                # disambiguate by isolating the duplicate in its own folder instead.
                folder_path = folder_path.with_name(f"{folder_path.name}-{uuid.uuid4().hex[:4]}")
                target = folder_path / name
                warnings.append(f"Duplicate name {name} stored in {folder_path.name}/.")
            folder_path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(entry["path"]), str(target))
            session_dirs.setdefault(folder_path, []).append(target)
        sources: list[RawSessionSource] = []
        for folder_path in sorted(session_dirs):
            sources.extend(discover_raw_sessions(folder_path))
        if not sources:
            raise ValueError("No collect_data*.txt session files were uploaded.")
        healthy_sources: list[RawSessionSource] = []
        for source in sources:
            try:
                session = load_raw_session(source)
                regressions = timeline_regressions(session)
                if not regressions:
                    valid_imu_spans(session)
            except (ValueError, AssertionError) as exc:
                warnings.append(f"{source.session_id}：会话数据无效（{exc}），已跳过")
                continue
            if regressions:
                warnings.append(f"{source.session_id}：采集时间轴损坏（{regressions} 处时间戳回退），已跳过")
            else:
                healthy_sources.append(source)
        sources = healthy_sources
        if not sources:
            result = make_prediction_result(run_key=self.run_key, source="uploaded sessions", duration_seconds=0, events=[])
            result["diagnostics"]["warnings"] = warnings.copy()
            if include_candidates:
                result["candidates"] = []
            if include_timeline:
                result["timeline"] = {"session_ids": [], "macro_windows": 0, "micro_windows": 0, "series": []}
            return {"analysis_id": analysis_id, "run_key": self.run_key, "prediction": result, "sessions": [], "warnings": warnings}
        options = PredictionOptions(include_timeline=include_timeline, include_candidates=include_candidates, device="cpu")
        with self._lock:
            predictor = self.predictor()
            result = predictor.predict_sources(tuple(sources), options)
            telemetry: dict[str, dict[str, object]] = {}
            sessions: list[dict[str, object]] = []
            telemetry_dir = root.parent / "telemetry"
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(sources):
                session = load_raw_session(source)
                data, skipped = motion_records(session)
                bin_path = telemetry_dir / f"{index}.bin"
                bin_path.write_bytes(data)
                timestamps = np.asarray(session.t_acc)
                valid = np.asarray(session.imu_valid) & (timestamps > 0)
                start_ms = int(timestamps[valid].min()) if valid.any() else 0
                end_ms = int(timestamps[valid].max()) if valid.any() else 0
                manifest = {
                    "telemetry_version": "1.0",
                    "session_id": source.session_id,
                    "record_format": "f64_ms_6xf32_le",
                    "sample_count": len(data) // 32,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "binary_file": "motion.bin",
                    "units": {"acceleration": "raw_adc", "gyroscope": "raw_adc"},
                    "provenance": {"source_file": Path(source.path).name, "skipped_rows": skipped, "timestamps": "ACC_TIME"},
                }
                calibration = estimate_gravity_calibration(session)
                if calibration is not None:
                    manifest["calibration"] = calibration
                telemetry[str(source.session_id)] = manifest
                sessions.append({
                    "session_id": source.session_id,
                    "source_name": Path(source.path).name,
                    "sample_count": manifest["sample_count"],
                    "motion": {"manifest": manifest, "bin_url": f"/api/artifacts/{analysis_id}/{index}/motion.bin"},
                })
        self._analyses[analysis_id] = {"root": root.parent, "telemetry": telemetry, "created": time.time()}
        return {
            "analysis_id": analysis_id,
            "run_key": self.run_key,
            "prediction": result,
            "sessions": sessions,
            "warnings": warnings,
        }

    def telemetry_path(self, analysis_id: str, index: int) -> Path | None:
        analysis = self._analyses.get(analysis_id)
        if analysis is None:
            return None
        if not (0 <= index < len(analysis["telemetry"])):
            return None
        path = Path(analysis["root"]) / "telemetry" / f"{index}.bin"
        return path if path.is_file() else None


class _Handler(BaseHTTPRequestHandler):
    server_version = "EatingSenseLocal/1.0"
    protocol_version = "HTTP/1.1"
    service: LocalInferenceService

    def log_message(self, format: str, *args: object) -> None:  # quiet, instrument-style
        return

    # ------------------------------------------------------------ utilities
    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return b""
        if length > MAX_UPLOAD_BYTES:
            self.connection.settimeout(5)
            try:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except (OSError, TimeoutError):
                pass
            self.close_connection = True
            raise ValueError("Upload exceeds the size limit.")
        return self.rfile.read(length)

    def _static(self, url_path: str) -> None:
        visual = self.service.visual_dir
        if visual is None:
            self._error(HTTPStatus.NOT_FOUND, "No visualization directory is served by this instance.")
            return
        relative = url_path.split("?", 1)[0].lstrip("/") or "index.html"
        candidate = (visual / relative).resolve()
        try:
            candidate.relative_to(visual)
        except ValueError:
            self._error(HTTPStatus.FORBIDDEN, "Path traversal is refused.")
            return
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"Not found: {relative}")
            return
        content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------- routing
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok", "service": SERVICE_NAME, "run_key": self.service.run_key})
        elif path == "/api/capabilities":
            self._json(HTTPStatus.OK, {
                "inference": True, "service": SERVICE_NAME, "run_key": self.service.run_key,
                "prediction_schema_version": "1.0", "telemetry": "motion.bin v1 (f64_ms_6xf32_le)",
                "visual_served": self.service.visual_dir is not None,
            })
        elif path.startswith("/api/artifacts/"):
            parts = path.strip("/").split("/")
            try:
                analysis_id, index = parts[2], int(parts[3])
            except (IndexError, ValueError):
                self._error(HTTPStatus.BAD_REQUEST, "Malformed artifact path.")
                return
            telemetry = self.service.telemetry_path(sanitize_segment(analysis_id), index)
            if telemetry is None:
                self._error(HTTPStatus.NOT_FOUND, "Unknown analysis artifact.")
                return
            data = telemetry.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path.startswith("/api/"):
            self._error(HTTPStatus.NOT_FOUND, f"Unknown API route: {path}")
        else:
            self._static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/upload":
                name = self.headers.get("X-Session-Name", "")
                relative = self.headers.get("X-Relative-Path")
                body = self._body()
                if not name:
                    self._error(HTTPStatus.BAD_REQUEST, "X-Session-Name header is required.")
                    return
                token = self.service.stage_upload(name, body, relative)
                self._json(HTTPStatus.OK, {"token": token, "name": name})
            elif path == "/api/analyze":
                payload = json.loads(self._body().decode("utf-8") or "{}")
                tokens = payload.get("tokens")
                if not isinstance(tokens, list) or not tokens:
                    self._error(HTTPStatus.BAD_REQUEST, "tokens must be a non-empty list.")
                    return
                result = self.service.analyze([str(token) for token in tokens])
                self._json(HTTPStatus.OK, result)
            else:
                self._error(HTTPStatus.NOT_FOUND, f"Unknown API route: {path}")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # keep the service alive; report restrained detail
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Inference failed: {exc}")


def build_server(service: LocalInferenceService, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (_Handler,), {"service": service})
    server_class = type("LocalThreadingHTTPServer", (ThreadingHTTPServer,), {"request_queue_size": 32})
    server = server_class((host, port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local canonical-inference bridge for the visual application.")
    parser.add_argument("--bundle", type=Path, default=None, help="deployment bundle directory (required; serve.py fills it in)")
    parser.add_argument("--run-key", default=None, help="release run key reported to clients")
    parser.add_argument("--visual-dir", type=Path, default=None, help="static visualization directory to serve")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (localhost only by default)")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--open", action="store_true", help="open the application in the default browser")
    args = parser.parse_args(argv)
    if args.bundle is None:
        print("--bundle is required (point it at a deployment bundle directory).", flush=True)
        return 2
    bundle = Path(args.bundle)
    run_key = args.run_key
    if run_key is None:
        try:
            manifest = json.loads((Path(bundle).parent / "manifest.json").read_text(encoding="utf-8"))
            run_key = str(manifest.get("release_run_key"))
        except (OSError, ValueError):
            run_key = Path(bundle).parent.name
    service = LocalInferenceService(Path(bundle), run_key=run_key, visual_dir=args.visual_dir)
    try:
        server = build_server(service, args.host, args.port)
    except OSError as exc:
        print(f"Local inference service cannot bind {args.host}:{args.port}: {exc}", flush=True)
        print("Another process may already use this port; close it or pass --port.", flush=True)
        return 2
    host, port = server.server_address[0], server.server_address[1]
    print(f"Local inference service: http://{host}:{port}/ (Ctrl+C to stop)", flush=True)
    if args.visual_dir:
        print(f"Serving visualization from: {args.visual_dir}", flush=True)
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
