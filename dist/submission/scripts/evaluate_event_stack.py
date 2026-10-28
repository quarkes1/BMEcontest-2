"""事件模型的受试者隔离嵌套交叉验证入口。

具体评估逻辑位于 scripts/crossfit_event_stack.py。
示例：python scripts/evaluate_event_stack.py --fold 0
完整参数见 --help。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.crossfit_event_stack import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
