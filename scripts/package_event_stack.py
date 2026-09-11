"""Atomically assemble a standalone event-stack deployment package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.artifacts import (
    PROMOTION_F1_FLOOR,
    load_event_stack_bundle,
    verify_bundle_manifest,
    verify_promotion_attestation,
)


_PREDICTOR = _ROOT / "scripts" / "predict_event_stack.py"
_RUNTIME_FILES = ("predict_event_stack.py", "requirements.txt", "runtime_manifest.json")
_MODEL_FILES = (
    "macro.joblib", "micro.joblib", "verifier_logistic.joblib", "verifier_lgbm.joblib",
    "policy.json", "run_config.json", "feature_schema.json", "manifest.json",
)
EXPECTED_EVENT_STACK_PATHS = frozenset(
    set(_RUNTIME_FILES) | {f"bundle/{name}" for name in _MODEL_FILES}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_paths(destination: Path) -> frozenset[str]:
    """Return the package file set excluding interpreter caches."""

    root = Path(destination)
    return frozenset(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_requirements(destination: Path) -> None:
    destination.write_text(
        "numpy>=1.26\njoblib>=1.3\nscikit-learn>=1.4\nlightgbm>=4.3\n",
        encoding="utf-8",
    )


def _build_runtime_manifest(staging: Path, *, has_cuda_adapter: bool) -> None:
    bundle_manifest = staging / "bundle" / "manifest.json"
    files = {
        path.relative_to(staging).as_posix(): _sha256(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "runtime_manifest.json"
    }
    payload = {
        "package_version": 1,
        "bundle_manifest_sha256": _sha256(bundle_manifest),
        "capabilities": {
            "cuda_adapter": (
                {"module": "cuda_adapter.py", "factory": "load_adapter"}
                if has_cuda_adapter else None
            ),
            "components": {name: "cpu" for name in ("macro", "micro", "verifier_logistic", "verifier_lgbm")},
            "future_cuda_parity": {"score_absolute_error_max": 1e-5, "event_geometry": "exact"},
        },
        "files": files,
    }
    (staging / "runtime_manifest.json").write_bytes(_canonical_json(payload))


def verify_packaged_bundle(destination: Path) -> None:
    """Verify the complete package and the copied Task-4 deployment manifest."""

    destination = Path(destination)
    try:
        runtime = json.loads((destination / "runtime_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"runtime manifest cannot be read: {exc}") from exc
    capabilities = runtime.get("capabilities")
    adapter = capabilities.get("cuda_adapter") if isinstance(capabilities, dict) else None
    if not isinstance(capabilities, dict) or "has_cuda_component" in capabilities:
        raise ValueError("runtime manifest must bind a registered CUDA adapter or null")
    expected_paths = EXPECTED_EVENT_STACK_PATHS
    if adapter is not None:
        if not isinstance(adapter, dict) or adapter != {"module": "cuda_adapter.py", "factory": "load_adapter"}:
            raise ValueError("runtime manifest CUDA adapter is not registered")
        expected_paths = frozenset(set(expected_paths) | {"cuda_adapter.py"})
    if required_paths(destination) != expected_paths:
        raise ValueError("packaged file set is incomplete or contains unexpected files")
    expected_hash = runtime.get("bundle_manifest_sha256")
    if not isinstance(expected_hash, str) or _sha256(destination / "bundle" / "manifest.json") != expected_hash:
        raise ValueError("runtime manifest bundle manifest SHA-256 mismatch")
    listed = runtime.get("files")
    if not isinstance(listed, dict):
        raise ValueError("runtime manifest files must be an object")
    actual = {
        path.relative_to(destination).as_posix(): _sha256(path)
        for path in destination.rglob("*")
        if path.is_file() and path.name != "runtime_manifest.json" and "__pycache__" not in path.parts
    }
    if listed != actual:
        raise ValueError("runtime manifest file checksums do not match package contents")
    problems = verify_bundle_manifest(destination / "bundle", expected_run_key="deployment")
    if problems:
        raise ValueError("packaged deployment bundle verification failed: " + "; ".join(problems))
    # The package deliberately places the copied bundle below ``bundle/``;
    # retain its original run key (``deployment``) during verification instead
    # of asking Task-4's loader to infer a run key from that new directory name.


def _run_smoke(script: Path, bundle: Path, payload: Path, output: Path, *, device: str = "cpu") -> bytes:
    completed = subprocess.run(
        [sys.executable, "-I", str(script), "--bundle", str(bundle), "--input-features", str(payload), "--output", str(output), "--device", device],
        cwd=payload.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"isolated inference smoke failed: {completed.stderr.strip()}")
    if f"resolved_device={device}" not in completed.stdout:
        raise RuntimeError(f"isolated inference smoke did not report resolved {device} device")
    return output.read_bytes()


def _verify_fixture_parity(staging: Path, source_bundle: Path, *, has_cuda_adapter: bool) -> None:
    spec = __import__("importlib.util").util.spec_from_file_location("event_stack_repo_predict", _PREDICTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repository event-stack predictor for parity verification")
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema = json.loads((source_bundle / "feature_schema.json").read_text(encoding="utf-8"))
    fixture = module.build_smoke_fixture(schema)
    fixture["schema_hash"] = module._schema_hash(fixture["feature_schema"])
    scratch = Path(tempfile.mkdtemp(prefix="event-stack-parity-"))
    payload = scratch / "fixture.json"
    repo_output = scratch / "repo-output.json"
    dist_output = scratch / "dist-output.json"
    payload.write_bytes(_canonical_json(fixture))
    try:
        repository = _run_smoke(_PREDICTOR, source_bundle, payload, repo_output)
        packaged = _run_smoke(staging / "predict_event_stack.py", staging / "bundle", payload, dist_output)
        if repository != packaged:
            raise RuntimeError("repository and packaged fixture predictions differ")
        if has_cuda_adapter:
            cuda_output = staging / ".cuda-output.json"
            try:
                cuda = _run_smoke(
                    staging / "predict_event_stack.py", staging / "bundle", payload,
                    cuda_output, device="cuda"
                )
                module.assert_backend_parity(json.loads(packaged), json.loads(cuda))
            finally:
                if cuda_output.exists():
                    cuda_output.unlink()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _safe_remove(path: Path) -> None:
    if _is_link_or_reparse_point(path) or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _is_link_or_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return bool(path.stat(follow_symlinks=False).st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False


def _trusted_event_stack_destination(destination: Path, trusted_dist_root: Path) -> Path:
    root = Path(trusted_dist_root).absolute()
    destination = Path(destination).absolute()
    if root.name != "dist":
        raise ValueError("trusted_dist_root must be the exact dist root")
    current = root
    while True:
        if current.exists() and _is_link_or_reparse_point(current):
            raise ValueError("trusted_dist_root must not traverse symlink/reparse components")
        if current.parent == current:
            break
        current = current.parent
    expected = root / "event_stack"
    if destination != expected:
        raise ValueError("destination must be the exact trusted_dist_root/event_stack directory")
    if destination.exists() and _is_link_or_reparse_point(destination):
        raise ValueError("destination must not be a symlink/reparse point")
    return expected


def package_event_stack(
    *,
    bundle_path: Path,
    destination: Path,
    trusted_dist_root: Path | None = None,
    cuda_adapter_path: Path | None = None,
) -> Path:
    """Package one verified deployment bundle through a same-parent atomic swap."""

    bundle_path = Path(bundle_path).absolute()
    trusted_root = Path(trusted_dist_root or (_ROOT / "dist"))
    destination = _trusted_event_stack_destination(destination, trusted_root)
    problems = verify_bundle_manifest(bundle_path, expected_run_key="deployment")
    if problems:
        raise ValueError("deployment bundle manifest verification failed: " + "; ".join(problems))
    load_event_stack_bundle(bundle_path, expected_role="deployment")
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    try:
        f1 = float(manifest["metrics"]["f1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("deployment bundle manifest lacks aggregate promotion F1") from exc
    if not math.isfinite(f1) or not f1 > PROMOTION_F1_FLOOR:
        raise ValueError(
            "deployment bundle does not satisfy the strict promotion F1 gate "
            f"> {PROMOTION_F1_FLOOR:.10f}"
        )
    attestation_problems = verify_promotion_attestation(bundle_path.parent)
    if attestation_problems:
        raise ValueError("promotion attestation verification failed: " + "; ".join(attestation_problems))
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse_point(parent) or _is_link_or_reparse_point(destination):
        raise ValueError("destination must not traverse symlink/reparse components")
    staging = parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    backup = parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    moved_previous = False
    promoted = False
    try:
        staging.mkdir()
        shutil.copy2(_PREDICTOR, staging / "predict_event_stack.py")
        _write_requirements(staging / "requirements.txt")
        shutil.copytree(bundle_path, staging / "bundle", symlinks=False)
        has_cuda_adapter = cuda_adapter_path is not None
        if cuda_adapter_path is not None:
            adapter_source = Path(cuda_adapter_path).absolute()
            if adapter_source.name != "cuda_adapter.py" or not adapter_source.is_file() or _is_link_or_reparse_point(adapter_source):
                raise ValueError("cuda_adapter_path must be a regular file named cuda_adapter.py")
            shutil.copy2(adapter_source, staging / "cuda_adapter.py")
        _build_runtime_manifest(staging, has_cuda_adapter=has_cuda_adapter)
        _verify_fixture_parity(staging, bundle_path, has_cuda_adapter=has_cuda_adapter)
        verify_packaged_bundle(staging)
        subprocess.run([sys.executable, "-I", str(staging / "predict_event_stack.py"), "--help"], check=True, cwd=staging.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if destination.exists():
            os.replace(destination, backup)
            moved_previous = True
        os.replace(staging, destination)
        promoted = True
        verify_packaged_bundle(destination)
        if backup.exists():
            _safe_remove(backup)
        return destination
    except Exception:
        if promoted and (destination.exists() or _is_link_or_reparse_point(destination)):
            _safe_remove(destination)
        if moved_previous and (backup.exists() or _is_link_or_reparse_point(backup)):
            os.replace(backup, destination)
        raise
    finally:
        if staging.exists() or _is_link_or_reparse_point(staging):
            _safe_remove(staging)
        if (backup.exists() or _is_link_or_reparse_point(backup)) and destination.exists():
            _safe_remove(backup)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Task-4 deployment bundle directory")
    parser.add_argument("--destination", type=Path, default=_ROOT / "dist" / "event_stack")
    parser.add_argument("--cuda-adapter", type=Path, default=None, help="registered cuda_adapter.py; requires CPU/CUDA fixture parity")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        package_event_stack(bundle_path=args.bundle, destination=args.destination, cuda_adapter_path=args.cuda_adapter)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"event-stack package refused: {exc}", file=sys.stderr)
        return 2
    print(f"packaged deployment event stack: {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
