"""Build the standalone raw-data inference distribution from canonical source.

The generated ``dist/inference`` tree is deliberately a build product.  Its
``event_stack`` package is copied from, and mechanically import-rewritten from,
the canonical implementation; it is never a hand-maintained second algorithm.

The staging/manifest/probe/atomic-replace core is shared with the competition
submission builder (``scripts/build_submission.py``), which parameterizes the
entrypoint and vendored import closure instead of forking this machinery.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.artifacts import (  # noqa: E402
    load_current_promoted_release,
    verify_bundle_manifest,
)


_RUNTIME_ROOT_MODULES = ("src.pipeline.inference.predictor", "src.pipeline.inference.local_server")
_PIN_NAMES = (
    ("numpy", "numpy"),
    ("joblib", "joblib"),
    ("scikit_learn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
)
_GENERATED_BASE_FILES = frozenset({"manifest.json", "feature_schema.json", "requirements.txt", "README.md"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _module_path(root: Path, module: str) -> Path | None:
    """Return an existing repository-local source file for an absolute module."""
    parts = module.split(".")
    if not parts or parts[0] != "src":
        return None
    direct = root.joinpath(*parts).with_suffix(".py")
    if direct.is_file():
        return direct
    package = root.joinpath(*parts, "__init__.py")
    return package if package.is_file() else None


def _module_name(root: Path, source: Path) -> str:
    relative = source.relative_to(root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _top_level_local_imports(root: Path, source: Path) -> set[Path]:
    """Resolve only import-time local dependencies of a canonical module.

    Imports nested inside diagnostic-only functions are intentionally excluded:
    they are not part of raw predictor runtime closure and would drag research
    loaders/caches into the distribution.  Every module copied here is still
    discovered by recursively parsing its own import-time imports.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    current = _module_name(root, source).split(".")
    if source.name != "__init__.py":
        current = current[:-1]
    found: set[Path] = set()
    for node in tree.body:
        module: str | None = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                path = _module_path(root, alias.name)
                if path is not None:
                    found.add(path)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if node.level > len(current):
                raise ValueError(f"relative import escapes canonical source: {source}")
            prefix = current[: len(current) - node.level + 1]
            module = ".".join(prefix + (node.module.split(".") if node.module else []))
        else:
            module = node.module
        if module:
            path = _module_path(root, module)
            if path is not None:
                found.add(path)
            elif module.startswith("src"):
                raise ValueError(f"local import resolves outside runtime closure: {module} from {source}")
    return found


def runtime_source_closure(repository_root: Path, *,
                           roots: Sequence[str] = _RUNTIME_ROOT_MODULES) -> tuple[Path, ...]:
    """Compute the complete import-time canonical source closure for the given roots."""
    root = Path(repository_root).resolve()
    pending = [path for module in roots if (path := _module_path(root, module))]
    if len(pending) != len(roots):
        raise ValueError("canonical Predictor source is missing")
    closure: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in closure:
            continue
        if source.is_symlink():
            raise ValueError(f"canonical runtime source may not be a symlink: {source}")
        closure.add(source)
        pending.extend(sorted(_top_level_local_imports(root, source), key=str))
    # Packages must be explicit files in the generated closure, not accidental
    # namespace packages supplied by a repository parent.
    for source in tuple(closure):
        relative = source.relative_to(root)
        for parent in relative.parents:
            if parent == Path("src"):
                continue
            init = root / parent / "__init__.py"
            if init.is_file():
                closure.add(init)
    return tuple(sorted(closure, key=lambda item: item.relative_to(root).as_posix()))


def _destination_for_source(repository_root: Path, source: Path, runtime_root: Path) -> Path:
    relative = source.relative_to(repository_root)
    if relative.parts[:2] == ("src", "pipeline"):
        return runtime_root / Path(*relative.parts[2:])
    if relative == Path("src/config.py"):
        return runtime_root / "config.py"
    if relative.parts[:2] == ("src", "eval"):
        return runtime_root / "eval" / Path(*relative.parts[2:])
    raise ValueError(f"runtime source is outside permitted canonical namespaces: {relative}")


def _rewrite_imports(text: str) -> str:
    """Mechanically rewrite canonical imports for the vendored package name."""
    return (
        text.replace("from src.pipeline.", "from event_stack.")
        .replace("import src.pipeline.", "import event_stack.")
        .replace("from src.eval.", "from event_stack.eval.")
        .replace("import src.eval.", "import event_stack.eval.")
        .replace("import src.config as config", "from event_stack import config")
        .replace("from src import config", "from event_stack import config")
    )


def copy_canonical_runtime_source(repository_root: Path, destination: Path, *,
                                  roots: Sequence[str] = _RUNTIME_ROOT_MODULES) -> tuple[str, ...]:
    """Vendor exactly the parsed canonical closure under ``event_stack``."""
    root = Path(repository_root).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []
    for source in runtime_source_closure(root, roots=roots):
        target = _destination_for_source(root, source, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_rewrite_imports(source.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
        copied.append(target.relative_to(destination.parent).as_posix())
    return tuple(sorted(copied))


def _requirements(bundle_path: Path) -> str:
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    versions = manifest.get("dependency_versions")
    if not isinstance(versions, dict):
        raise ValueError("deployment manifest dependency versions are missing")
    lines: list[str] = []
    for key, distribution in _PIN_NAMES:
        value = versions.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"deployment manifest {key} pin is invalid")
        lines.append(f"{distribution}=={value}")
    return "\n".join(lines) + "\n"


_PREDICT_ENTRYPOINT = '''"""Standalone raw event-stack inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from event_stack.inference import Predictor

def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="collect_data*.txt file or folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-timeline", action="store_true")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
    return parser.parse_args(argv)

def main(argv=None):
    args = _args(argv)
    try:
        manifest = json.loads((_ROOT / "manifest.json").read_text(encoding="utf-8"))
        predictor = Predictor.from_bundle(_ROOT / "models", device=args.device,
                                          run_key=str(manifest["release_run_key"]))
        options = predictor.options(include_timeline=args.include_timeline, include_candidates=args.include_candidates, device=args.device)
        result = predictor.predict_folder(args.input, options=options) if args.input.is_dir() else predictor.predict_file(args.input, options=options)
        args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"event-stack inference refused: {exc}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


_SERVE_ENTRYPOINT = '''"""Local inference bridge entry: canonical inference API + static visual application.

Serves the visualization at http://127.0.0.1:PORT/ (default 4173) and the bridge API
under /api/*. Raw TXT selections from the browser are analyzed by the canonical
Predictor shipped in this package; numbers, events, and telemetry are never computed
in the browser.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from event_stack.inference import local_server


def _package_root() -> Path:
    """serve.py sits at the package root (dist/inference) or in app/ (submission)."""
    for base in (_ROOT, _ROOT.parent):
        if (base / "manifest.json").is_file() or (base / "meta" / "manifest.json").is_file():
            return base
    return _ROOT


def _manifest() -> dict:
    root = _package_root()
    for candidate in (root / "meta" / "manifest.json", root / "manifest.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError("manifest.json not found next to serve.py")


def _default_bundle() -> Path | None:
    """Bundle layout differs per package: dist/inference has models/ flat at the root;
    dist/submission nests models/event_stack/<run_key>/deployment."""
    flat = _package_root() / "models"
    if (flat / "manifest.json").is_file():
        return flat
    try:
        run_key = str(_manifest()["release_run_key"])
    except (OSError, ValueError, KeyError):
        return None
    nested = flat / "event_stack" / run_key / "deployment"
    return nested if (nested / "manifest.json").is_file() else None


def _default_visual() -> Path | None:
    for candidate in (_package_root() / "visual", _ROOT / "visual", _ROOT.parent / "visual"):
        if (candidate / "index.html").is_file():
            return candidate
    return None


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--visual-dir", type=Path, default=None)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    try:
        manifest = _manifest()
    except (OSError, ValueError) as exc:
        print(f"local inference bridge refused: {exc}", file=sys.stderr)
        return 2
    bundle = _default_bundle()
    if bundle is None:
        print("local inference bridge refused: no deployment bundle found next to serve.py", file=sys.stderr)
        return 2
    visual = args.visual_dir or _default_visual()
    forwarded = ["--bundle", str(bundle), "--run-key", str(manifest["release_run_key"]),
                 "--host", args.host, "--port", str(args.port)]
    if visual:
        forwarded += ["--visual-dir", str(visual)]
    if args.open:
        forwarded.append("--open")
    return local_server.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
'''


_README = """# 独立 event-stack 推理包

本目录由当前 promoted canonical release 自动生成，请勿手工修改 `event_stack/`；
重建命令：`python scripts/build_inference_distribution.py`。

## 包结构

```text
dist/inference/
├── predict.py            # 独立推理入口（原始 collect_data*.txt → prediction JSON）
├── event_stack/          # canonical 源码的机械 vendored 副本（勿手改）
├── models/               # 冻结 deployment bundle（macro/micro/verifier 模型 + policy）
├── manifest.json         # 逐文件 SHA-256 与 source/model 闭包
├── feature_schema.json   # schema v2（macro 63 / micro 47 / verifier 116）
├── requirements.txt      # 精确依赖 pin（来自 deployment manifest）
└── README.md
```

安装精确记录的依赖后，即可对官方原始 `collect_data*.txt` 文件或其所在目录做预测：

```bash
python -m pip install -r requirements.txt
python predict.py path/to/collect_data1_2_3.txt --output prediction.json
python predict.py path/to/subject-folder --output prediction.json --include-timeline
```

支持 `--device cpu`。当前发布没有经过审计的 CUDA 适配器，强制 `--device gpu` 或
`--device cuda` 会被明确拒绝，而不是静默回退到 CPU。输出 JSON 遵循
`dist/schema/prediction.schema.json`。

## 可视化本地服务（serve.py）

`python serve.py --open` 启动本地推理桥（仅绑定 127.0.0.1，默认端口 4173）：同源
提供 `../visual/` 的静态前端与 `/api/*` 接口（`/api/health`、`/api/upload`、
`/api/analyze`、`/api/artifacts/...`）。浏览器中选择 collect_data*.txt 文件/文件夹后，
由本包的 canonical Predictor 完成推理并返回预测契约与运动遥测；前端不实现任何模型逻辑。
`dist/start.bat` 即为该服务的一键启动器。
"""


def _generated_paths(meta_dir: str, entrypoint: str, extra_entries: Sequence[str]) -> set[str]:
    """Package-relative paths of the generated (and entry) files that must exist.

    README and requirements.txt always sit at the package root; the manifest and
    feature schema live under ``meta_dir`` when one is configured.
    """
    prefix = f"{meta_dir}/" if meta_dir else ""
    return {"requirements.txt", "README.md", f"{prefix}feature_schema.json",
            f"{prefix}manifest.json", entrypoint, *extra_entries}


def _distribution_files(root: Path) -> dict[str, str]:
    """Every shipped file except the manifest itself; local bytecode never counts."""
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and not path.is_symlink() and path.name != "manifest.json"
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _resolve_meta_dir(package: Path, meta_dir: str) -> str:
    """Accept the meta/ layout automatically so callers need not know the flavour."""
    if meta_dir:
        return meta_dir
    return "meta" if (package / "meta" / "manifest.json").is_file() else ""


def verify_distribution_manifest(package: Path, *, entrypoint: str = "predict.py",
                                 extra_entries: Sequence[str] = (),
                                 required_roots: Sequence[str] = ("event_stack", "models"),
                                 meta_dir: str = "") -> None:
    """Reject hash drift, symlinks and undeclared runtime files.

    ``source_files`` covers every declared file outside ``models/`` (vendored
    runtime, reproduction source, docs, visualization assets); ``model_files``
    covers everything under ``models/``.  ``meta_dir`` is where the generated
    ``manifest.json``/``feature_schema.json`` live ("meta" for the submission,
    "" for the inference distribution).
    """
    package = Path(package)
    if package.is_symlink() or not package.is_dir():
        raise ValueError("distribution package must be a real directory")
    meta_dir = _resolve_meta_dir(package, meta_dir)
    generated = _generated_paths(meta_dir, entrypoint, extra_entries)
    manifest = json.loads((package / meta_dir / "manifest.json").read_text(encoding="utf-8"))
    required = {"release_run_key", "model_version", "prediction_schema_version", "feature_schema", "python_version", "dependencies", "model_files", "source_files", "hashes"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("distribution manifest schema is invalid")
    for name in generated | set(required_roots):
        if not (package / name).exists():
            raise ValueError(f"distribution file is missing: {name}")
    if any(path.is_symlink() for path in package.rglob("*")):
        raise ValueError("distribution may not contain symlinks")
    actual = _distribution_files(package)
    if manifest["hashes"] != actual:
        raise ValueError("distribution manifest file checksums do not match")
    model_files = sorted(path for path in actual if path.startswith("models/"))
    source_files = sorted(path for path in actual if not path.startswith("models/"))
    if manifest["source_files"] != source_files or manifest["model_files"] != model_files:
        raise ValueError("distribution manifest source/model closure does not match")


def _atomic_replace(staging: Path, destination: Path) -> Path:
    destination = Path(destination)
    parent = destination.parent
    backup = parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    moved = False
    try:
        if destination.exists():
            if destination.is_symlink():
                raise ValueError("distribution destination may not be a symlink")
            os.replace(destination, backup)
            moved = True
        os.replace(staging, destination)
        if moved:
            shutil.rmtree(backup)
        return destination
    except Exception:
        if destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination)
        if moved and backup.exists():
            os.replace(backup, destination)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and destination.exists():
            shutil.rmtree(backup)


_TEXT_SUFFIXES = frozenset({
    ".py", ".md", ".json", ".txt", ".html", ".htm", ".css", ".js", ".mjs", ".cjs",
    ".ts", ".tsx", ".jsx", ".mts", ".cts", ".scss", ".yml", ".yaml", ".toml", ".cfg",
    ".ini", ".sh", ".svg", ".map",
})


def _copy_file(source: Path, target: Path) -> None:
    """Copy one file; text files are normalized to LF.

    Byte stability matters: the distribution manifest hashes worktree bytes,
    and git checks these files out with LF (``* text=auto eol=lf``).  Copying
    CRLF worktree bytes would make a fresh clone fail manifest verification.
    """
    source = Path(source)
    target = Path(target)
    if source.suffix.lower() in _TEXT_SUFFIXES:
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(source, target)
            return
        target.write_text(text.replace("\r\n", "\n").replace("\r", "\n"),
                          encoding="utf-8", newline="\n")
        return
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path, *, extra_ignore: Sequence[str] = ()) -> None:
    """Copy a file or directory into the staging tree without bytecode."""
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        _copy_file(source, target)
        return
    shutil.copytree(source, target, symlinks=False, copy_function=_copy_file,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache",
                                                  "node_modules", ".vite", "coverage",
                                                  *extra_ignore))


def build_distribution(*, repository_root: Path, bundle_path: Path, destination: Path,
                       entrypoint: str, entrypoint_text: str, readme_text: str,
                       extra_entrypoints: Sequence[tuple[str, str]] = (),
                       closure_roots: Sequence[str] = _RUNTIME_ROOT_MODULES,
                       models_source: Path | None = None, models_destination: str = "models",
                       extra_trees: Sequence[tuple[Path, str]] = (),
                       required_roots: Sequence[str] = ("event_stack", "models"),
                       meta_dir: str = "", runtime_dir: str = "") -> Path:
    """Atomically construct a verified distribution package for the active release.

    Shared by the inference distribution (``predict.py``) and the competition
    submission (``main.py``; the submission supplies the full model root as
    ``models_source`` plus reproduction-source and visualization ``extra_trees``).
    """
    root = Path(repository_root).resolve()
    release = load_current_promoted_release(root)
    bundle = Path(bundle_path).resolve()
    expected = root / "models" / "event_stack" / str(release["run_key"]) / "deployment"
    if bundle != expected:
        raise ValueError("distribution must use the current promoted deployment bundle")
    problems = verify_bundle_manifest(bundle, expected_run_key="deployment")
    if problems:
        raise ValueError("deployment bundle manifest verification failed: " + "; ".join(problems))
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or destination.is_symlink():
        raise ValueError("distribution destination may not traverse symlinks")
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        runtime_root = staging / runtime_dir if runtime_dir else staging
        runtime_root.mkdir(parents=True, exist_ok=True)
        copy_canonical_runtime_source(root, runtime_root / "event_stack", roots=closure_roots)
        _copy_tree(models_source if models_source is not None else bundle, staging / models_destination)
        for source, relative, *ignore in extra_trees:
            _copy_tree(source, staging / relative, extra_ignore=ignore[0] if ignore else ())
        (staging / entrypoint).write_text(entrypoint_text, encoding="utf-8", newline="\n")
        for extra_name, extra_text in extra_entrypoints:
            extra_target = staging / extra_name
            extra_target.parent.mkdir(parents=True, exist_ok=True)
            extra_target.write_text(extra_text, encoding="utf-8", newline="\n")
        (staging / "requirements.txt").write_text(_requirements(bundle), encoding="utf-8", newline="\n")
        meta = staging / meta_dir if meta_dir else staging
        meta.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / "feature_schema.json", meta / "feature_schema.json")
        (staging / "README.md").write_text(readme_text, encoding="utf-8", newline="\n")
        model_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest = {
            "release_run_key": str(release["run_key"]),
            "model_version": model_manifest.get("bundle_version"),
            "prediction_schema_version": "1.0",
            "feature_schema": json.loads((bundle / "feature_schema.json").read_text(encoding="utf-8")),
            "python_version": model_manifest["dependency_versions"]["python"],
            "dependencies": {key: model_manifest["dependency_versions"][key] for key, _ in _PIN_NAMES},
            "model_files": [], "source_files": [], "hashes": {},
        }
        manifest["hashes"] = _distribution_files(staging)
        manifest["model_files"] = sorted(path for path in manifest["hashes"] if path.startswith("models/"))
        manifest["source_files"] = sorted(path for path in manifest["hashes"] if not path.startswith("models/"))
        (meta / "manifest.json").write_bytes(_json_bytes(manifest))
        extra_names = tuple(name for name, _ in extra_entrypoints)
        verify_distribution_manifest(staging, entrypoint=entrypoint, extra_entries=extra_names,
                                     required_roots=required_roots, meta_dir=meta_dir)
        probe_environment = dict(os.environ)
        probe_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for probe_entry in (entrypoint, *extra_names):
            probe = subprocess.run(
                [sys.executable, "-I", probe_entry, "--help"], cwd=staging,
                capture_output=True, text=True, env=probe_environment,
            )
            if probe.returncode != 0:
                raise RuntimeError(f"isolated distribution import probe failed ({probe_entry}): " + probe.stderr.strip())
        # A distribution is source/model only.  The isolated import probe must
        # not make bytecode a generated runtime dependency.
        for cache in staging.rglob("__pycache__"):
            shutil.rmtree(cache)
        verify_distribution_manifest(staging, entrypoint=entrypoint, extra_entries=extra_names,
                                     required_roots=required_roots, meta_dir=meta_dir)
        return _atomic_replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_inference_distribution(*, repository_root: Path, bundle_path: Path, destination: Path) -> Path:
    """Atomically construct a verified raw-inference package for the active release."""
    return build_distribution(
        repository_root=repository_root, bundle_path=bundle_path, destination=destination,
        entrypoint="predict.py", entrypoint_text=_PREDICT_ENTRYPOINT, readme_text=_README,
        extra_entrypoints=(("serve.py", _SERVE_ENTRYPOINT),),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/event_stack/160afaf81debf1ee/deployment")
    parser.add_argument("--destination", type=Path, default=ROOT / "dist/inference")
    args = parser.parse_args(argv)
    try:
        built = build_inference_distribution(repository_root=ROOT, bundle_path=args.bundle, destination=args.destination)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"inference distribution build refused: {exc}", file=sys.stderr)
        return 2
    print(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
