"""竞赛正式推理入口。"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts.final_submission_entry import main
if __name__ == "__main__": raise SystemExit(main())
