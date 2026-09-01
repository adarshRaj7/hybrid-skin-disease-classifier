#!/usr/bin/env python3
"""Package a trained checkpoint into a Hugging Face model repository layout,
and optionally upload it.

This script never hardcodes a Hugging Face username, repo name, or token.
Supply them via CLI flags or environment variables:

    export HF_TOKEN=hf_xxx
    python scripts/package_for_huggingface.py \\
        --checkpoint models/best_model.pt \\
        --repo-id your-username/skin-disease-classifier \\
        --push
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skin_disease.utils import load_model  # noqa: E402

MODEL_CARD_TEMPLATE = """---
license: other
tags:
  - image-classification
  - pytorch
  - resnet
  - densenet
  - medical-imaging
---

# Skin Disease Classifier (ResNet50 + DenseNet121 Hybrid)

## Medical disclaimer

This tool provides an AI-generated classification based on an uploaded skin
image. It is intended for research and decision-support purposes only and
is **not** a substitute for evaluation by a qualified healthcare
professional. Model confidence does not represent medical certainty. It has
**not** been clinically validated and has **no** regulatory approval. If you
have a concerning or changing skin lesion or symptom, seek professional
medical advice.

## Model

- Backbone: ResNet50 (2048-d) + DenseNet121 (1024-d), concatenated to a
  3072-d feature vector, both pretrained on ImageNet.
- Head: 4-layer fully connected classifier (3072→2048→1024→512→num_classes)
  with BatchNorm, ReLU, and Dropout.
- Classes: {num_classes}
- Model version: {model_version}

## Files

- `model.pt`: self-describing checkpoint (weights + architecture config +
  class names + preprocessing config + training config + version).
- `class_names.json`: class index -> name mapping.

## Usage

```python
from skin_disease.inference import SkinDiseasePredictor

predictor = SkinDiseasePredictor(hf_repo_id="{repo_id}")
result = predictor.predict("path/to/image.jpg")
print(result)
```

## Severity metadata

Class predictions are optionally annotated with a MILD / MODERATE / SEVERE /
OTHER severity bucket. This is predefined **project metadata for display
purposes only** and has not been clinically validated -- it must not be
interpreted as a medical severity assessment.

## Evaluation

See `test_metrics.json` in the training repository's `outputs/` directory
for macro/weighted precision, recall, F1, per-class metrics, and
severe-class recall on the held-out test set. Model selection during
training prioritized validation macro recall (see `train.py`) because false
negatives are particularly costly in a medical-imaging context; this
correspondingly means precision must be checked explicitly rather than
assumed.

## Limitations

- Trained on a single public dataset; performance may not generalize to
  images taken with different cameras, lighting, or skin tones not well
  represented in the training data.
- Class boundaries between visually similar conditions can be ambiguous.
- Not validated in a clinical setting.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Package (and optionally upload) a model for Hugging Face.")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="hf_package")
    parser.add_argument("--repo-id", type=str, default=os.environ.get("HF_MODEL_REPO_ID"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    if args.push and not args.repo_id:
        raise SystemExit("--push requires --repo-id (or HF_MODEL_REPO_ID env var).")

    _, metadata = load_model(args.checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.checkpoint, output_dir / "model.pt")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump({"classes": metadata["class_names"]}, f, indent=2)

    with open(output_dir / "preprocessing.json", "w", encoding="utf-8") as f:
        json.dump(metadata["preprocessing"], f, indent=2)

    card = MODEL_CARD_TEMPLATE.format(
        num_classes=len(metadata["class_names"]),
        model_version=metadata.get("model_version", "unknown"),
        repo_id=args.repo_id or "<your-username>/<your-repo>",
    )
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(card)

    print(f"Packaged model artifact at: {output_dir}")

    if args.push:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN in your environment to push to the Hub.")
        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, exist_ok=True)
        api.upload_folder(folder_path=str(output_dir), repo_id=args.repo_id)
        print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
