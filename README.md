# Skin Disease Classifier — ResNet50 + DenseNet121 Hybrid

> **Medical disclaimer:** This tool provides an AI-generated classification
> based on an uploaded image. It is intended for **research and
> decision-support purposes only** and is **not** a substitute for
> evaluation by a qualified healthcare professional. Model confidence does
> not represent medical certainty. The system has **not** been clinically
> validated, has **no** regulatory approval, and its severity labels are
> project metadata, not a medical assessment. If you have a concerning or
> changing skin lesion or symptom, seek professional medical advice.

This repository is a production rewrite of a research notebook that trained
a hybrid ResNet50 + DenseNet121 image classifier for skin-disease
categories. The core scientific idea — two pretrained CNN backbones,
concatenated features, a deep classification head — is preserved. What
changed is everything around it: data handling, evaluation methodology,
checkpointing, testing, and deployment. See "What changed from the original
notebook" below for the full list.

## Overview

Two ImageNet-pretrained backbones extract complementary features from the
same input image; a four-layer classifier head turns the concatenated
feature vector into class logits.

## Architecture

```text
Input (3x224x224)
      │
      ├──────────────→ ResNet50 (no FC layer)   ──→ 2048-d
      │
      └──────────────→ DenseNet121 (no classifier) → 1024-d
                           │
                           ▼
                     concatenate → 3072-d
                           │
                           ▼
              Linear(3072→2048) BN ReLU Dropout
              Linear(2048→1024) BN ReLU Dropout
              Linear(1024→512)  BN ReLU Dropout
              Linear(512→num_classes)
                           │
                           ▼
                    disease logits → softmax
```

There is exactly one `ResNet50` and one `DenseNet121` instance in the model
(`src/skin_disease/model.py::HybridBackbone`), shared by a single forward
path (`HybridSkinDiseaseModel`). See "What changed" for why this is called
out explicitly.

## Dataset

The model expects an ImageFolder-style layout:

```text
<dataset_dir>/
├── train/
│   ├── Acne/
│   ├── Eczema/
│   └── ... one folder per class
└── test/
    ├── Acne/
    ├── Eczema/
    └── ...
```

A validation split is carved out of `train/` automatically at training time
(stratified, seeded — see `configs/config.yaml: data.val_fraction`). The
`test/` directory is reserved for final evaluation only (`scripts/evaluate.py`)
and is never touched during training or model selection.

Class indices are **not** derived from filesystem order at inference time.
`ClassLabels.from_directory()` builds the mapping once (alphabetically
sorted, matching the original project's convention) and writes it to
`models/class_names.json`; every other stage (training, evaluation,
checkpoint loading, inference, the API) loads that file rather than
re-deriving it.

## Installation

```bash
git clone <your-repo-url> skin-disease-classifier
cd skin-disease-classifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dataset preparation

Dataset downloading is a separate, explicit step — the model package itself
never downloads training data, and no credentials live in this repository.

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
python scripts/prepare_dataset.py --output-dir data
```

> **If you are the original project owner:** the Kaggle API key that was
> embedded in the source notebook (`adarshraj77` / a plaintext key) must be
> treated as compromised. Rotate/revoke it on kaggle.com. This repository
> does not contain it anywhere, including in `notebooks/`.

This also runs an explicit, offline validation pass and **reports** (does
not silently delete) unreadable image files.

## Training

```bash
python scripts/train.py --config configs/config.yaml --dataset-dir data/SkinDisease
```

This writes:
- `models/best_model.pt` — self-describing checkpoint (see "Checkpoint format")
- `models/class_names.json` — the deterministic class mapping
- `outputs/training_history.json` — per-epoch train/val metrics

Key training decisions (see `configs/config.yaml` to change any of them):

| Setting | Default | Why |
|---|---|---|
| Validation source | stratified split of `train/` | test set must never influence model selection |
| Checkpoint metric | `val_macro_recall` | see "Why macro recall" below |
| Backbone LR / classifier LR | `1e-5` / `1e-4` | differential fine-tuning: backbones adapt slowly, head learns faster |
| `freeze_backbones` | `false` | fine-tuning both backbones outperforms frozen features in most transfer-learning setups for domain-specific medical images; set `true` if you have very little data or want faster/cheaper training |
| Class weights | off by default | only enable (`train.use_class_weights: true`) after checking the training class distribution — see "Class weighting" below |
| Loss | `CrossEntropyLoss` | no evidence in the source material justified a different loss |

### Augmentation strategy

Critically re-examined rather than copied verbatim:

- **Horizontal/vertical flip, rotation**: kept. Close-up lesion crops have no
  canonical "up" orientation the way natural scene photos do.
- **Brightness/contrast/saturation jitter**: kept at mild levels to absorb
  camera and lighting variation between source images.
- **Hue jitter reduced from 0.1 → 0.02**: hue encodes clinically relevant
  signal (erythema/redness, pigmentation) that is often exactly what
  distinguishes one class from another. Aggressively randomizing it risks
  training the model to disregard a genuinely diagnostic feature. A small
  amount is kept only to absorb minor white-balance variation.
- **Affine translation**: kept mild, to simulate imperfect framing.

### Class weighting

Class weighting is **not** applied automatically, and it is **not** tied to
the `SEVERE` severity bucket — severity is presentation metadata, not a
training signal (see `src/skin_disease/labels.py`). If you enable
`train.use_class_weights`, weights are computed via inverse class frequency
from the *training split only* and logged at the start of training; check
`outputs/training_history.json` for their effect on validation
macro-precision, since blind class weighting can trade precision for
recall.

## Evaluation

```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint models/best_model.pt
```

Run this exactly once, after training is finished and no more
hyperparameter/checkpoint decisions remain. Writes:
- `outputs/test_metrics.json` — accuracy, macro/weighted precision/recall/F1,
  severe-class recall, confusion matrix, and the corresponding validation
  numbers for comparison
- `outputs/per_class_metrics.csv` — per-class precision/recall/F1/support

### Why macro recall (and not just accuracy)

This is a medical decision-support application, where a missed disease
(false negative) tends to matter more than a percentage point of overall
accuracy — and overall accuracy on an imbalanced dataset can look good while
hiding poor recall on rare or severe classes. The primary
checkpoint-selection metric is therefore **validation macro recall**. That
is deliberately *not* the same as maximizing recall unconditionally: macro
F1 and macro precision are logged every epoch specifically so a degenerate
"predict everything as the same class" solution (which can spike recall on
one class while destroying precision) would be visible before a model is
shipped. The final report separately breaks out:
1. Macro recall · 2. Per-class recall · 3. Severe-class recall
4. Macro F1 · 5. Macro precision · 6. Weighted F1 · 7. Accuracy

No hyperparameter, threshold, or checkpoint decision in this project is
made using the test set — only the validation split.

## Local inference

```bash
python scripts/predict.py --checkpoint models/best_model.pt --image path/to/image.jpg
python scripts/predict.py --checkpoint models/best_model.pt --image path/to/image.jpg --gradcam overlay.png
```

```python
from skin_disease.inference import SkinDiseasePredictor

predictor = SkinDiseasePredictor(checkpoint_path="models/best_model.pt")
result = predictor.predict("path/to/image.jpg")
# {
#   "predicted_class": "...",
#   "confidence": 0.94,             # model confidence, not medical certainty
#   "top_3_predictions": [...],
#   "severity": "...",              # project metadata, not a clinical assessment
#   "model_version": "1.0.0"
# }
```

Handles JPEG/PNG/WebP, grayscale, RGBA, and RGB images; raises
`InvalidImageError` (never an unhandled exception) for corrupt data,
oversized files, or missing files.

## Grad-CAM

```python
overlay = predictor.generate_gradcam("path/to/image.jpg")  # defaults to predicted class, DenseNet branch
overlay = predictor.generate_gradcam("path/to/image.jpg", target_class=3, branch="resnet")
```

Grad-CAM is refactored onto the model's actual (single) backbone layers —
`backbone.resnet_target_layer` / `backbone.densenet_target_layer` — instead
of the original notebook's duplicated feature extractor. It is never
required for a normal prediction; call it explicitly when you want an
explanation.

## API

```bash
export MODEL_CHECKPOINT_PATH=models/best_model.pt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```text
GET  /health           → {"status": "ok", "model_loaded": true, "model_version": "...", "num_classes": N, "device": "..."}
POST /predict           (multipart file upload) → prediction JSON + medical disclaimer
POST /predict/gradcam    (multipart file upload) → prediction JSON + base64 PNG overlay
```

The model is loaded exactly once at process startup (FastAPI lifespan
handler), not per-request. Invalid/oversized/unsupported uploads return
`4xx` with a generic message; inference failures return `500` with a
generic message — no stack traces are ever returned to the client (they are
logged server-side instead).

## Frontend

A minimal static frontend (`frontend/index.html`) posts an uploaded image to
`/predict` and renders the result, including the disclaimer and an optional
Grad-CAM overlay. It talks to whatever API base URL you set at the top of
the file — open it directly in a browser once the API is running locally,
or serve it from any static host.

## Hugging Face

```bash
python scripts/package_for_huggingface.py --checkpoint models/best_model.pt --output-dir hf_package
# review hf_package/ (model.pt, class_names.json, preprocessing.json, README.md)

export HF_TOKEN=hf_your_token
python scripts/package_for_huggingface.py \
    --checkpoint models/best_model.pt \
    --repo-id your-username/skin-disease-classifier \
    --push
```

At inference time, point the predictor at the Hub instead of a local file:

```python
predictor = SkinDiseasePredictor(hf_repo_id="your-username/skin-disease-classifier")
```

or set `HF_MODEL_REPO_ID` for the API. No username, repo name, or token is
hardcoded anywhere in this repository.

## Checkpoint format

`models/best_model.pt` is a single `torch.save`d dict containing:
`state_dict`, `architecture` (num_classes, dropout_rate), `class_names`,
`preprocessing` (image_size, mean, std), `model_version`,
`training_config` (the full `AppConfig` used), `checkpoint_metric`, and
`best_metric_value`. `skin_disease.utils.load_model()` reconstructs a ready
model + metadata dict from this file alone — no notebook or original
training script is required.

## Testing

```bash
pip install -r requirements.txt  # includes pytest, httpx
pytest -q
```

46 tests across model shape/parameter-count, transforms (RGB/grayscale/RGBA,
determinism), checkpoint round-trips, inference (top-3, confidence bounds,
invalid/oversized/missing-file handling), Grad-CAM (both branches, hook
cleanup), dataset/stratified-split correctness, and the API (`/health`,
`/predict`, `/predict/gradcam`, error codes, model-not-loaded fallback).
Tests use synthetic images and a randomly initialized tiny model — they
never require the real dataset, a GPU, or network access.

## What changed from the original notebook

### Preserved
- The ResNet50 + DenseNet121 hybrid architecture and 3072-d concatenated
  feature vector.
- The 4-layer classifier head shape (3072→2048→1024→512→num_classes) with
  BatchNorm/ReLU/Dropout, including the halved dropout on the last hidden
  layer.
- ImageNet normalization constants, 224×224 input size.
- The `SEVERITY_MAPPING` / `SEVERITY_PRIORITY` metadata, verbatim.
- CrossEntropyLoss, AdamW, and Grad-CAM as the explainability method.

### Fixed
- **Duplicate feature extractors**: the original notebook built one
  `HybridFeatureExtractor` at module scope (used for novelty detection) and
  a *second, separate* one inside `SkinDiseaseClassifier.__init__` (used for
  classification) — two ResNet50s and two DenseNet121s in memory, trained
  independently of each other. There is now exactly one of each, shared by
  one model.
- **Train/validation/test leakage**: the training loop's "validation" phase
  ran directly over `test_loader`, and the best checkpoint (and the final
  "test evaluation" reported afterward) were selected/computed using that
  same test set throughout. A proper stratified train/validation split is
  now carved out of `train/` only; `test/` is used exactly once, at the end.
- **Non-deterministic class indices**: classes were implicitly ordered by
  `os.listdir()`/`ImageFolder` at whatever time each script ran. A
  `class_names.json` mapping is now built once and reused everywhere.
- **Corrupted-image handling**: `__getitem__` silently recursed to
  `(idx + 1) % len(self)` with no logging and no bound, risking silent
  infinite recursion; it now logs and falls back within a bounded number of
  deterministic alternates.

### Removed
- PCA dimensionality reduction, and the Mahalanobis / One-Class SVM / Deep
  SVDD / ensemble novelty-detection pipeline. These were explicitly out of
  scope for the production system per project requirements; production
  inference goes directly from the 3072-d feature vector to the classifier,
  with no artifacts to save or load beyond the model checkpoint itself.
- Kaggle credentials embedded in source. Dataset download is now a separate
  script that reads `KAGGLE_USERNAME`/`KAGGLE_KEY` from the environment (or
  an existing `~/.kaggle/kaggle.json`) and never accepts or stores a
  credential inside the repository.
- Notebook shell commands (`!pip`, `!mkdir`, `!kaggle`, `!unzip`) from all
  production code paths.

### Added
- Modular `src/skin_disease/` package, FastAPI backend, CLI scripts,
  YAML configuration, self-describing checkpoints, a pytest suite,
  structured logging, reproducibility seeding, a minimal static frontend,
  and Hugging Face packaging tooling.

## Limitations

- Evaluated only on the single dataset used to train it; performance may
  not generalize to images from different cameras, lighting conditions, or
  skin tones not well represented in the training data.
- Some class boundaries (e.g., between visually similar inflammatory
  conditions) are inherently ambiguous even for expert clinicians from a
  single image.
- The `SEVERITY_MAPPING` used for display grouping was authored as project
  metadata and has not been independently, clinically validated.
- No fairness/subgroup performance analysis (e.g., across Fitzpatrick skin
  types) has been performed; treat any deployment beyond research use with
  substantial additional caution and evaluation.
- This README and codebase do not claim clinical validation, diagnostic
  accuracy in clinical practice, or regulatory approval of any kind.

## Remaining decisions for the project owner

These require a judgment call this repository cannot make on your behalf:

1. **Real training run**: no GPU/dataset was available in the environment
   this was built in, so the checkpoint you use in production has not
   actually been trained here — you need to run `scripts/train.py` (and
   `scripts/evaluate.py`) yourself against the real dataset and inspect the
   resulting metrics before deploying.
2. **`freeze_backbones`**: whether to fine-tune both backbones or freeze
   them depends on your dataset size and compute budget; the default
   (`false`, fine-tune both with a small backbone LR) is a reasonable
   starting point, not a guarantee of the best result.
3. **Class weighting**: enable only after inspecting the real class
   distribution and confirming via the validation set that it improves
   macro recall without an unacceptable drop in macro precision.
4. **Confidence threshold** for any user-facing "low confidence" warning in
   the frontend: none is currently hardcoded; if you want one, select it
   from validation-set data, not the test set.
5. **Hugging Face repo naming/visibility** and whether the model repo
   should be public or gated, given it is a health-adjacent model.
