"""Local inference bridge entry: canonical inference API + static visual application.

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
