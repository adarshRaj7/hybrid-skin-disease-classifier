#!/usr/bin/env python3
"""CLI entry point for a single local prediction.

Usage:
    python scripts/predict.py --checkpoint models/best_model.pt --image path/to/image.jpg
    python scripts/predict.py --checkpoint models/best_model.pt --image path/to/image.jpg --gradcam out.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skin_disease.inference import SkinDiseasePredictor  # noqa: E402
from skin_disease.utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single local prediction.")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--gradcam", type=str, default=None, help="If set, save a Grad-CAM overlay PNG to this path.")
    args = parser.parse_args()

    setup_logging()
    predictor = SkinDiseasePredictor(checkpoint_path=args.checkpoint)
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=2))

    if args.gradcam:
        import numpy as np
        from PIL import Image

        cam_result = predictor.generate_gradcam(args.image)
        overlay = (np.clip(cam_result["overlay"], 0, 1) * 255).astype("uint8")
        Image.fromarray(overlay).save(args.gradcam)
        print(f"Saved Grad-CAM overlay to {args.gradcam} (target class: {cam_result['target_class_name']})")


if __name__ == "__main__":
    main()
