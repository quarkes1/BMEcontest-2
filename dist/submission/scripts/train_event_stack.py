"""事件模型的全量目标数据训练和发布入口。

具体训练及发布逻辑位于 scripts/promote_event_stack.py。
示例：python scripts/train_event_stack.py --summary outputs/crossfit/summary_<run_key>.json"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promote_event_stack import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
