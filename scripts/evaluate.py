#!/usr/bin/env python3
"""CLI entry point for final held-out test evaluation.

Usage:
    python scripts/evaluate.py --config configs/config.yaml --checkpoint models/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skin_disease.config import AppConfig  # noqa: E402
from skin_disease.evaluate import evaluate_test_set  # noqa: E402
from skin_disease.utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the held-out test set.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    config = AppConfig.load(args.config)
    evaluate_test_set(config, checkpoint_path=args.checkpoint, dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
