#!/usr/bin/env python3
"""CLI entry point for training.

Usage:
    python scripts/train.py --config configs/config.yaml --dataset-dir data/SkinDisease
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skin_disease.config import AppConfig  # noqa: E402
from skin_disease.train import train  # noqa: E402
from skin_disease.utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hybrid skin disease classifier.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Overrides data.dataset_dir in the config.")
    args = parser.parse_args()

    setup_logging()
    config = AppConfig.load(args.config)
    train(config, dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
