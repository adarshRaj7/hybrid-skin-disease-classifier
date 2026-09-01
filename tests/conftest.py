"""Shared pytest fixtures.

Tests never require the real (multi-GB, Kaggle-hosted) dataset or internet
access to download pretrained ImageNet weights. Instead they build tiny
synthetic images on disk and instantiate the model with
``pretrained_backbones=False`` (random init), which is enough to exercise
every code path -- shapes, checkpoint round-trips, Grad-CAM hook wiring, API
plumbing -- without requiring network access or GPU time.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from skin_disease.labels import ClassLabels
from skin_disease.model import HybridSkinDiseaseModel
from skin_disease.utils import save_checkpoint

TEST_CLASSES = ["Acne", "Eczema", "Psoriasis", "SkinCancer"]  # includes one SEVERE class


@pytest.fixture()
def class_labels() -> ClassLabels:
    return ClassLabels(classes=TEST_CLASSES)


@pytest.fixture()
def tiny_model(class_labels) -> HybridSkinDiseaseModel:
    torch.manual_seed(0)
    return HybridSkinDiseaseModel(
        num_classes=class_labels.num_classes,
        dropout_rate=0.5,
        pretrained_backbones=False,
        freeze_backbones=False,
    )


def _make_pil_image(mode: str = "RGB", size=(64, 64), color=(120, 60, 60)) -> Image.Image:
    if mode == "RGB":
        return Image.new("RGB", size, color)
    if mode == "L":
        return Image.new("L", size, 100)
    if mode == "RGBA":
        return Image.new("RGBA", size, color + (255,))
    raise ValueError(mode)


@pytest.fixture()
def sample_rgb_image() -> Image.Image:
    return _make_pil_image("RGB")


@pytest.fixture()
def sample_grayscale_image() -> Image.Image:
    return _make_pil_image("L")


@pytest.fixture()
def sample_rgba_image() -> Image.Image:
    return _make_pil_image("RGBA")


@pytest.fixture()
def sample_image_bytes(sample_rgb_image) -> bytes:
    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def synthetic_dataset_dir(tmp_path: Path, class_labels: ClassLabels) -> Path:
    """A tiny ImageFolder-style dataset: a few images per class, deterministic content."""
    rng = np.random.default_rng(0)
    root = tmp_path / "SkinDisease"
    for split, n_per_class in (("train", 6), ("test", 3)):
        for class_name in class_labels.classes:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
                Image.fromarray(arr).save(class_dir / f"{class_name}_{i}.jpg")
    return root


@pytest.fixture()
def tiny_checkpoint(tmp_path: Path, tiny_model, class_labels: ClassLabels) -> Path:
    path = tmp_path / "models" / "tiny_model.pt"
    save_checkpoint(
        model=tiny_model,
        path=path,
        class_names=class_labels.classes,
        image_size=64,
        training_config={"note": "test fixture, not a real training run"},
        checkpoint_metric="val_macro_recall",
        best_metric_value=0.0,
    )
    return path
