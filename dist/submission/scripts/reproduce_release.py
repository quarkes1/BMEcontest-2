"""不重新训练，校验已发布模型的完整证据链。

依次检查版本登记及发布证明、外层各折证据模型、部署模型和冻结聚合指标。
若存在 dist/inference 或 dist/submission，也按清单检查构建包，并打印冻结指标。
用法：python scripts/reproduce_release.py [--run-key 160afaf81debf1ee]
退出码：0 表示校验通过，2 表示发现校验问题。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.artifacts import (  # noqa: E402
    load_current_promoted_release,
    verify_bundle_manifest,
)


def verify_release_chain(root: Path, *, expected_run_key: str | None = None) -> tuple[list[str], dict[str, object]]:
    """返回当前已发布版本的校验问题及结果摘要。"""
    root = Path(root)
    problems: list[str] = []
    release = load_current_promoted_release(root)
    run_key = str(release["run_key"])
    if expected_run_key is not None and run_key != expected_run_key:
        problems.append(f"active run key {run_key} does not match requested {expected_run_key}")
    run_root = root / "models" / "event_stack" / run_key
    for fold in range(5):
        bundle = run_root / f"outer-fold-{fold}"
        problems.extend(f"outer-fold-{fold}: {problem}"
                        for problem in verify_bundle_manifest(bundle))
    problems.extend(f"deployment: {problem}"
                    for problem in verify_bundle_manifest(run_root / "deployment"))
    return problems, dict(release["aggregate"])


def verify_distributions(root: Path) -> list[str]:
    """按各自的哈希清单校验已经构建的推理包。"""
    from scripts.build_inference_distribution import verify_distribution_manifest

    problems: list[str] = []
    for name, entrypoint in (("inference", "predict.py"), ("submission", "main.py")):
        package = Path(root) / "dist" / name
        if not package.is_dir():
            continue
        # 提交包的运行时位于 app/，元数据位于 meta/。
        # 独立推理包则将两者放在包根目录。
        required = ("app", "meta", "models") if (package / "app" / "event_stack").is_dir() else ("event_stack", "models")
        try:
            if name == "submission" and (package / "build_manifest.json").is_file():
                from scripts.build_competition_exe import verify_submission
                verify_submission(package)
            else:
                verify_distribution_manifest(package, entrypoint=entrypoint, required_roots=required)
        except (OSError, ValueError) as exc:
            problems.append(f"dist/{name}: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", default=None, help="expected active release run key")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        problems, summary = verify_release_chain(args.root, expected_run_key=args.run_key)
        problems.extend(verify_distributions(args.root))
    except (OSError, ValueError) as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 2
    if problems:
        for problem in problems:
            print(f"release verification problem: {problem}", file=sys.stderr)
        return 2
    outer = summary["outer_metrics"]
    print(f"run key: {args.run_key or 'active'}")
    print(f"aggregate F1: {outer['f1']}")
    print(f"TP: {outer['n_tp']}")
    print(f"FP: {outer['n_pred'] - outer['n_tp']}")
    print(f"predictions: {outer['n_pred']}")
    print(f"eligible truths: {outer['n_true']}")
    print(f"recall: {outer['sensitivity']}")
    print(f"PPV: {outer['ppv']}")
    print("release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
