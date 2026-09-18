"""Local inference bridge: canonical prediction contract + telemetry, stdlib only.

The bridge is a thin adapter around the canonical Predictor. These tests prove the
end-to-end path ``TXT -> bridge -> canonical Predictor -> valid prediction contract``
with the real promoted model on a real session fixture, plus upload hygiene.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "models/event_stack/160afaf81debf1ee/deployment"


@pytest.fixture()
def bridge(tmp_path):
    from src.pipeline.inference.local_server import LocalInferenceService, build_server

    service = LocalInferenceService(BUNDLE, run_key="160afaf81debf1ee", visual_dir=None, temp_root=tmp_path / "bridge")
    server = build_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield service, base
    server.shutdown()
    server.server_close()


def _request(url: str, method: str = "GET", body: bytes | None = None, headers: dict | None = None):
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _raw_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def _tiny_session(rows: int = 120) -> bytes:
    header = "ACC_TIME\tPPG_TIME\tGYRO_TIME\t" + "\t".join(f"v{i}" for i in range(50)) + "\n"
    values = ["1"] * 44 + ["100", "200", "300", "1", "2", "3"]
    body = []
    for i in range(rows):
        t = 1_700_000_000_000 + i * 10
        body.append(f"{t}\t{t}\t{t}\t" + "\t".join(values))
    return (header + "\n".join(body) + "\n").encode("utf-8")


def _regressed_session() -> bytes:
    lines = _tiny_session(120).decode("utf-8").splitlines()
    lines[61] = lines[11]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _upload(base: str, name: str, data: bytes, relative: str | None = None) -> str:
    headers = {"Content-Type": "application/octet-stream", "X-Session-Name": name}
    if relative:
        headers["X-Relative-Path"] = relative
    status, body = _request(f"{base}/api/upload", "POST", data, headers)
    assert status == 200, body
    return body["token"]


def test_health_and_capabilities(bridge):
    service, base = bridge
    from src.pipeline.inference.local_server import build_server
    server = build_server(service, "127.0.0.1", 0)
    assert server.request_queue_size == 32
    assert server.RequestHandlerClass.protocol_version == "HTTP/1.1"
    server.server_close()
    status, health = _request(f"{base}/api/health")
    assert status == 200 and health["status"] == "ok" and health["run_key"] == "160afaf81debf1ee"
    status, capabilities = _request(f"{base}/api/capabilities")
    assert status == 200 and capabilities["inference"] is True
    assert capabilities["prediction_schema_version"] == "1.0"


def test_oversized_upload_returns_json_error(bridge):
    import http.client
    from src.pipeline.inference.local_server import MAX_UPLOAD_BYTES

    _, base = bridge
    port = int(base.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        connection.putrequest("POST", "/api/upload")
        connection.putheader("Content-Length", str(MAX_UPLOAD_BYTES + 1))
        connection.putheader("X-Session-Name", "collect_data1_2_3.txt")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 400
        assert "size limit" in json.loads(response.read())["error"]
    finally:
        connection.close()


def test_analyze_tiny_session_returns_contract_and_telemetry(bridge):
    from src.pipeline.inference.schema import validate_prediction

    _, base = bridge
    token = _upload(base, "collect_data1_2_3.txt", _tiny_session())
    status, body = _request(f"{base}/api/analyze", "POST", json.dumps({"tokens": [token]}).encode("utf-8"),
                            {"Content-Type": "application/json"})
    assert status == 200, body
    validate_prediction(body["prediction"])
    assert body["prediction"]["model"]["run_key"] == "160afaf81debf1ee"
    assert body["sessions"][0]["session_id"] == "collect_data1_2_3"
    manifest = body["sessions"][0]["motion"]["manifest"]
    assert manifest["record_format"] == "f64_ms_6xf32_le"
    assert manifest["units"] == {"acceleration": "raw_adc", "gyroscope": "raw_adc"}
    data = _raw_bytes(base + body["sessions"][0]["motion"]["bin_url"])
    assert len(data) == manifest["sample_count"] * 32
    timestamp = np.frombuffer(data[:8], dtype="<f8")[0]
    assert timestamp == manifest["start_ms"]


def test_regressed_session_is_skipped_without_losing_healthy_peer(bridge):
    from src.pipeline.inference.schema import validate_prediction

    _, base = bridge
    good = _upload(base, "collect_data1_2_3.txt", _tiny_session())
    bad = _upload(base, "collect_data4_5_6.txt", _regressed_session())
    status, body = _request(f"{base}/api/analyze", "POST", json.dumps({"tokens": [good, bad]}).encode(),
                            {"Content-Type": "application/json"})
    assert status == 200, body
    validate_prediction(body["prediction"])
    assert len(body["sessions"]) == 1
    assert "已跳过" in " ".join(body["warnings"])

    status, body = _request(f"{base}/api/analyze", "POST", json.dumps({"tokens": [bad]}).encode(),
                            {"Content-Type": "application/json"})
    assert status == 200, body
    validate_prediction(body["prediction"])
    assert body["sessions"] == [] and body["prediction"]["events"] == []
    assert "已跳过" in " ".join(body["warnings"])


def test_bridge_matches_canonical_predictor_on_the_release_fixture(bridge):
    import src.config as config
    from src.pipeline.inference import Predictor, PredictionOptions

    _, base = bridge
    fixture = json.loads((ROOT / "tests/fixtures/release_160afaf81debf1ee/fixture_manifest.json").read_text(encoding="utf-8"))
    session_id = str(fixture["macro_golden"]["session_id"])
    raw = next((config.SENSOR_DIR / session_id).glob("collect_data*.txt"))
    token = _upload(base, raw.name, raw.read_bytes())
    status, body = _request(f"{base}/api/analyze", "POST", json.dumps({"tokens": [token]}).encode("utf-8"),
                            {"Content-Type": "application/json"})
    assert status == 200, body
    expected = Predictor.from_bundle(BUNDLE).predict_file(
        raw, options=PredictionOptions(include_timeline=True, include_candidates=True))
    assert [(e["session_id"], e["start_ms"], e["end_ms"], e["confidence"]) for e in body["prediction"]["events"]] \
        == [(e["session_id"], e["start_ms"], e["end_ms"], e["confidence"]) for e in expected["events"]]
    assert body["prediction"]["candidates"] == expected["candidates"]


def test_upload_hygiene_and_input_validation(bridge):
    _, base = bridge
    token = _upload(base, "collect_data9_9_9.txt", _tiny_session(30), relative="../../escape/collect_data9_9_9.txt")
    service, _ = bridge
    staged = service._uploads[token]["path"]
    assert Path(staged).resolve().is_relative_to(service.temp_root.resolve())
    status, body = _request(f"{base}/api/analyze", "POST", json.dumps({"tokens": [token]}).encode("utf-8"), {"Content-Type": "application/json"})
    assert status == 200, body
    assert str(service.temp_root.resolve()) in str(sorted((service.temp_root / "analyses").rglob("collect_data9_9_9.txt"))[0].resolve())
    status, body = _request(f"{base}/api/analyze", "POST", json.dumps({"tokens": ["missing"]}).encode("utf-8"), {"Content-Type": "application/json"})
    assert status == 400
    notes = _upload(base, "notes.txt", b"not a session")
    status, body = _request(f"{base}/api/analyze", "POST", json.dumps({"tokens": [notes]}).encode("utf-8"), {"Content-Type": "application/json"})
    assert status == 400 and "collect_data" in body["error"]


def test_static_serving_stays_inside_the_visual_directory(tmp_path):
    from src.pipeline.inference.local_server import LocalInferenceService, build_server

    visual = tmp_path / "visual"
    visual.mkdir()
    (visual / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    service = LocalInferenceService(BUNDLE, run_key="160afaf81debf1ee", visual_dir=visual, temp_root=tmp_path / "bridge")
    server = build_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/", timeout=30) as response:
            assert b"ok" in response.read()
        for path in ("/%2e%2e/secret.txt", "/..%2fsecret.txt"):
            try:
                urllib.request.urlopen(base + path, timeout=30)
                raise AssertionError(f"traversal served for {path}")
            except urllib.error.HTTPError as error:
                assert error.code in (403, 404)
    finally:
        server.shutdown()
        server.server_close()


def test_unconsumed_error_body_closes_the_connection(bridge):
    """HTTP/1.1 keep-alive must never leave an unread request body in the socket.

    A 404 for an unknown POST route used to answer while the body stayed queued; the
    next request on that connection was then parsed from the leftover bytes (observed
    as ``501 Unsupported method '{"tokens":["x"]}GET'``).  Error responses that never
    read the body must end the connection instead.
    """
    import http.client

    _, base = bridge
    port = int(base.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        body = b'{"tokens":["x"]}'
        connection.putrequest("POST", "/api/unknown-route")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(len(body)))
        connection.endheaders()
        connection.send(body)
        response = connection.getresponse()
        assert response.status == 404
        assert response.getheader("Connection") == "close"
        response.read()
    finally:
        connection.close()
