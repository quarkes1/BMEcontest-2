"""Build versioned 15-second ACC+GYRO feature caches."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as project_config
from src.pipeline.imu_features import MicroFeatureConfig
from src.pipeline.micro_cache import build_micro_split


def parse_args():
    parser = argparse.ArgumentParser(description="Build versioned 15-second ACC+GYRO feature caches.")
    parser.add_argument("--fold", choices=("0", "1", "2", "3", "4", "all"), default="all")
    parser.add_argument("--split", choices=("train", "meal_train", "no_meal_train", "val", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-gravity-align", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = range(5) if args.fold == "all" else (int(args.fold),)
    splits = ("train", "meal_train", "no_meal_train", "val") if args.split == "all" else (args.split,)
    feature_config = MicroFeatureConfig(gravity_align=not args.no_gravity_align)
    for fold in folds:
        for split in splits:
            output = build_micro_split(
                project_config.ROOT_DIR,
                fold=fold,
                split=split,
                config=feature_config,
                workers=args.workers,
                limit=args.limit,
                force=args.force,
            )
            print(output)


if __name__ == "__main__":
    main()
