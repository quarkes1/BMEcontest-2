"""Source entry point for the packaged visualization and local inference API."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if not (ROOT/'src').is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.inference.official_runner import main

if __name__ == '__main__':
    raise SystemExit(main(['--visual', *sys.argv[1:]]))
