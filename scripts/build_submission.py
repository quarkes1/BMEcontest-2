"""Build the competition submission bundle from canonical source.

``dist/submission`` 是竞赛最终交付物，由本脚本从仓库唯一真源确定性生成：

- 推理接口：``main.py``（raw 模式；官方模式默认 csv adapter）与机械 vendored 的
  canonical 运行时 ``app/event_stack/``（本地桥 ``app/serve.py``）；
- 完整模型资产：整个冻结模型根 ``models/event_stack/<run_key>/``（deployment
  bundle + 五个 outer-fold evidence bundle + promotion summary 与 attestation）；
- 复现代码：``src/``（canonical 算法源码）、``scripts/``（训练/评估/发布/构建全链）
  与 ``tests/``；``release/`` 与 ``outputs/crossfit/``（发布指针与五折证据）；
- 可视化：``visual/``（dist/visual 工作区，队友产出原样打包）以及 ``schema/``、
  ``examples/``（预测契约与安全示例）；``meta/``（manifest 与特征 schema）。

README 由 ``scripts/submission_readme.md`` 提供——直接编辑该 markdown 文件，
重建即生效。

The audited release materials do not define the official competition I/O contract,
so the documented default is the ``csv`` adapter (one row per detected eating
event).  Implement a concrete adapter and call ``register_adapter`` in
``src/pipeline/inference/competition_adapter.py`` when the real contract appears.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_inference_distribution import _SERVE_ENTRYPOINT, build_distribution  # noqa: E402
from src.pipeline.artifacts import load_current_promoted_release  # noqa: E402


_SUBMISSION_CLOSURE_ROOTS = (
    "src.pipeline.inference.predictor",
    "src.pipeline.inference.competition_adapter",
    "src.pipeline.inference.local_server",
)

_SUBMISSION_REQUIRED_ROOTS = (
    "app", "meta", "models", "src", "scripts", "tests", "visual", "schema", "examples",
    "release", "start.bat",
)

_MAIN_ENTRYPOINT = '''"""竞赛提交入口：冻结 event-stack 发布的完整推理接口。

raw 模式完整可用，输出与 canonical Predictor 文档逐值一致。官方竞赛模式使用文档化
的默认 csv adapter：每条进食事件一行 CSV（start_time,end_time,start_ms,end_ms，
ISO 本地时间 + 原始毫秒），缺省写入 ./predict/predict_<输入名>.csv。官方真实契约
发布后，实现 adapter 并 register_adapter(...) 即可替换（见 README「自定义 adapter」）。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

_ROOT = Path(__file__).resolve().parent
for _extra in (_ROOT / "app", _ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from event_stack.inference import Predictor
from event_stack.inference.competition_adapter import (
    DEFAULT_ADAPTER_NAME, registered_adapter, resolve_output_path,
)


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--raw", type=Path, help="collect_data*.txt 文件或目录（canonical raw 模式）")
    mode.add_argument("--official-input", type=Path, help="官方竞赛输入（需要已注册 adapter）")
    parser.add_argument("--output", default=None,
                        help="输出路径：raw 模式必填（JSON）；official 模式缺省 ./predict/predict_<输入名>.csv"
                             "（目录以 / 结尾，或指向已存在的目录）")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER_NAME,
                        help="official 模式的 adapter 名（默认 csv；注册自定义后可用其他名）")
    parser.add_argument("--cls", action="store_true", help="official 模式：先清空 ./predict 再写出本次结果")
    parser.add_argument("--include-timeline", action="store_true")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _predict(predictor, path, *, subject_id=None, options):
    path = Path(path)
    if path.is_dir():
        return predictor.predict_folder(path, subject_id=subject_id, options=options)
    return predictor.predict_file(path, subject_id=subject_id, options=options)


def main(argv=None):
    args = _args(argv)
    try:
        manifest = json.loads((_ROOT / "meta" / "manifest.json").read_text(encoding="utf-8"))
        run_key = str(manifest["release_run_key"])
        predictor = Predictor.from_bundle(
            _ROOT / "models" / "event_stack" / run_key / "deployment",
            device=args.device, run_key=run_key,
        )
        options = predictor.options(include_timeline=args.include_timeline,
                                    include_candidates=args.include_candidates,
                                    device=args.device)
        if args.raw is not None:
            if args.output is None:
                print("competition submission refused: --raw requires --output", file=sys.stderr)
                return 2
            out_path = Path(args.output)
            result = _predict(predictor, args.raw, options=options)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
            return 0
        if args.cls:
            shutil.rmtree("predict", ignore_errors=True)
        adapter = registered_adapter(args.adapter)
        raw_input, subject_id = adapter.load(args.official_input)
        output = resolve_output_path(args.official_input, args.output)
        result = _predict(predictor, raw_input, subject_id=subject_id, options=options)
        adapter.dump(result, output)
        print(f"wrote {output}")
        return 0
    except NotImplementedError as exc:
        print(f"competition submission refused: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"competition submission refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

_SUBMISSION_README = (Path(__file__).resolve().parent / "submission_readme.md").read_text(encoding="utf-8")


def _extra_trees(root: Path) -> tuple[tuple, ...]:
    trees: list[tuple] = [
        (root / "src", "src"),
        # 报告生成工具依赖 python-docx/matplotlib，不属于推理/复现链，随仓库维护。
        (root / "scripts", "scripts", ("report",)),
        (root / "tests", "tests"),
        (root / "dist" / "visual", "visual"),
        (root / "dist" / "schema", "schema"),
        (root / "dist" / "examples", "examples"),
        (root / "dist" / "start.bat", "start.bat"),
        # 复现链所需的发布指针与五折证据（体积小，让包内 repro 可直接运行）
        (root / "release", "release"),
        (root / "outputs" / "crossfit", "outputs/crossfit"),
    ]
    missing = [str(source) for source, *_ in trees if not (root / source).exists()]
    if missing:
        raise ValueError("submission source tree is missing: " + ", ".join(missing))
    return tuple(trees)


def build_submission(*, repository_root: Path, destination: Path,
                     bundle_path: Path | None = None) -> Path:
    """Atomically construct the verified competition submission package."""
    root = Path(repository_root).resolve()
    release = load_current_promoted_release(root)
    run_key = str(release["run_key"])
    if bundle_path is None:
        bundle_path = root / "models" / "event_stack" / run_key / "deployment"
    return build_distribution(
        repository_root=root, bundle_path=Path(bundle_path), destination=destination,
        entrypoint="main.py", entrypoint_text=_MAIN_ENTRYPOINT, readme_text=_SUBMISSION_README,
        closure_roots=_SUBMISSION_CLOSURE_ROOTS,
        models_source=root / "models" / "event_stack" / run_key,
        models_destination=f"models/event_stack/{run_key}",
        extra_entrypoints=(("app/serve.py", _SERVE_ENTRYPOINT),),
        extra_trees=_extra_trees(root),
        required_roots=_SUBMISSION_REQUIRED_ROOTS,
        meta_dir="meta",
        runtime_dir="app",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--destination", type=Path, default=ROOT / "dist/submission")
    args = parser.parse_args(argv)
    try:
        built = build_submission(repository_root=ROOT, destination=args.destination,
                                 bundle_path=args.bundle)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"submission build refused: {exc}", file=sys.stderr)
        return 2
    print(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
