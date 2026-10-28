"""本地可视化服务源码入口。"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts.final_submission_entry import main
if __name__ == "__main__": raise SystemExit(main(["--visual",*sys.argv[1:]]))
