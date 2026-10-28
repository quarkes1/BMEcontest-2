"""打包后可视化界面及本地推理接口的源码入口。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if not (ROOT/'src').is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.inference.official_runner import main

if __name__ == '__main__':
    raise SystemExit(main(['--visual', *sys.argv[1:]]))
