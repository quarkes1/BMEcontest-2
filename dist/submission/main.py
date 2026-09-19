"""竞赛提交入口：冻结 event-stack 发布的完整推理接口。

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
            out_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
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
