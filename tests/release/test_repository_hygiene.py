"""Release gate: promoted-release reproduction works and every deleted path is proven dead.

Deletion proof policy: ``tests/fixtures/deletion_manifest.json`` is the ledger.
Each entry claims its path is gone and that six consumer categories
(``imports``/``tests``/``reproduction``/``release``/``submission``/``references``)
find no trace of the path in the final tree.  ``references`` is a total scan of
the source/test/distribution/documentation surface, so a false ``references``
claim implies the five narrower claims.

Two kinds of documents are excluded from the scan because they are the
refactor's own process artifacts, not consumers: the deletion ledger itself
(including packaged copies inside generated distributions, matched by its
``tests/fixtures/deletion_manifest.json`` suffix) and the delivery
specification (``docs/BMEcontest-2 ... .md``) plus the historical plan files
under ``docs/superpowers/``, which name cleanup categories by design.

Bare-basename search applies only to distinctive stems (containing ``_``);
generic basenames (``predict.py``, ``config.py``, ``metrics.py``, ``ranker.py``)
are searched by repository-relative path only, because bare mentions are
ambiguous with same-named files inside kept distributions such as
``dist/inference/predict.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/fixtures/deletion_manifest.json"

_EXCLUDED_FILES = {
    "tests/fixtures/deletion_manifest.json",
    "docs/BMEcontest-2 竞赛交付与仓库收口重构任务书.md",
    "docs/superpowers/plans/2026-09-15-competition-delivery-refactor.md",
    "docs/superpowers/plans/2026-09-14-context-v1-implementation.md",
}
_EXCLUDED_FILE_SUFFIXES = ("tests/fixtures/deletion_manifest.json",)
_EXCLUDED_DIR_PREFIXES = ("docs/superpowers/",)
_SKIP_SUFFIXES = {".pt", ".joblib", ".npz", ".7z", ".png", ".jpg", ".ico", ".pyc"}
_MAX_TEXT_BYTES = 2_000_000

_SCOPES: dict[str, tuple[str, ...]] = {
    "imports": ("src", "scripts", "tests"),
    "tests": ("tests",),
    "reproduction": ("README.md", "docs", "scripts/train_event_stack.py",
                      "scripts/evaluate_event_stack.py", "scripts/reproduce_release.py"),
    "release": ("release", "models", "scripts/release_event_stack.py",
                "scripts/promote_event_stack.py", "scripts/package_event_stack.py"),
    "submission": ("dist",),
    "references": ("README.md", "docs", "src", "scripts", "tests", "dist", "release", "models"),
}


def _scan_texts(scopes: tuple[str, ...]) -> list[tuple[str, str]]:
    files: list[Path] = []
    for scope in scopes:
        target = ROOT / scope
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(path for path in target.rglob("*") if path.is_file())
    found: list[tuple[str, str]] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if (relative in _EXCLUDED_FILES
                or relative.endswith(_EXCLUDED_FILE_SUFFIXES)
                or relative.startswith(_EXCLUDED_DIR_PREFIXES)):
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES or path.stat().st_size > _MAX_TEXT_BYTES:
            continue
        found.append((relative, path.read_text(encoding="utf-8", errors="ignore")))
    return found


def _terms(path_str: str) -> tuple[str, ...]:
    path = Path(path_str)
    terms = [re.escape(path_str)]
    if path_str.startswith("scripts/") and path_str.endswith(".py"):
        terms.append(re.escape("scripts." + path.stem) + r"\b")
    if "_" in path.stem:
        terms.append(r"(?<![\w/.\\-])" + re.escape(path.name) + r"\b")
    return tuple(terms)


def _category_hits(path_str: str, category: str, *, texts=None) -> list[str]:
    if texts is None:
        texts = _scan_texts(_SCOPES[category])
    path = Path(path_str)
    # Every existing regex requires at least one of these literal substrings.
    # Fast rejection cannot remove a genuine match; regexes still decide hits.
    needles = (path_str, path.name, 'scripts.' + path.stem)
    hits: list[str] = []
    for term in _terms(path_str):
        pattern = re.compile(term)
        for relative, text in texts:
            if relative == path_str:
                continue
            if not any(needle in text for needle in needles):
                continue
            if category == "imports" and not re.search(r"^\s*(from|import)\b", text, re.MULTILINE):
                continue
            if pattern.search(text):
                hits.append(relative)
                break
    return sorted(set(hits))


def test_reproduce_release_verifies_without_training():
    done = subprocess.run(
        [sys.executable, "scripts/reproduce_release.py", "--run-key", "160afaf81debf1ee"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "0.6514285714285715" in done.stdout


def test_deleted_paths_have_no_consumers():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["deleted"], "deletion ledger must not be empty"
    proof_fields = ("imports", "tests", "reproduction", "release", "submission", "references")
    # One consistent read-only snapshot per consumer scope for this audit only.
    # Do not reread the entire workspace for every retired path and category.
    snapshots = {category: _scan_texts(_SCOPES[category]) for category in proof_fields}
    for item in manifest["deleted"]:
        assert set(item) == {"path", *proof_fields}, item
        assert all(type(item[name]) is bool and item[name] is False for name in proof_fields), item
        assert not (ROOT / item["path"]).exists(), f"ledger path still exists: {item['path']}"
        for category in proof_fields:
            hits = _category_hits(item["path"], category, texts=snapshots[category])
            assert hits == [], f"{item['path']} still referenced ({category}): {hits[:5]}"


def test_gitignore_covers_rebuildable_trash():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("__pycache__/", "*.pyc", ".pytest_cache/", "outputs/tmp/", "outputs/experiments/",
                    "*.staging-*/", "*.backup-*/"):
        assert pattern in text, f".gitignore must cover {pattern}"


def test_git_status_has_no_rebuildable_trash():
    done = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    for line in done.stdout.splitlines():
        name = line[3:].strip('"')
        assert "__pycache__" not in name, line
        assert not name.endswith(".pyc"), line
        assert ".staging-" not in name and ".backup-" not in name, line


def test_preloaded_consumer_scope_preserves_reference_matching():
    path = 'scripts/example_retired_tool.py'
    snapshot = [('README.md', 'python scripts/example_retired_tool.py'),
                ('scripts/example_retired_tool.py', 'self reference')]
    assert _category_hits(path, 'references', texts=snapshot) == ['README.md']
    assert _category_hits(path, 'references', texts=[]) == []


def test_preloaded_import_scope_preserves_noncode_filter():
    path = 'scripts/example_retired_tool.py'
    snapshot = [('README.md', 'python scripts/example_retired_tool.py'),
                ('src/consumer.py', 'from scripts.example_retired_tool import main')]
    assert _category_hits(path, 'imports', texts=snapshot) == ['src/consumer.py']


def test_consumer_prefilter_keeps_exact_regex_boundaries_and_module_names():
    snapshot = [('src/use.py', 'from scripts.example_retired_tool import main'),
                ('README.md', 'old example_retired_tool.py'),
                ('unrelated.md', 'not_example_retired_tool.py'),
                ('generic.md', 'config.py')]
    assert _category_hits('scripts/example_retired_tool.py', 'references', texts=snapshot) == [
        'README.md', 'src/use.py']
    assert _category_hits('src/config.py', 'references', texts=snapshot) == []
