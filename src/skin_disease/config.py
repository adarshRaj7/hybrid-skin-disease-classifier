"""Typed configuration objects loaded from configs/config.yaml.

Keeping configuration in dataclasses (rather than passing loose dicts
around) means the model, training, and inference code always agree on
field names and defaults, and the whole configuration can be dropped
straight into a checkpoint for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_VERSION = "1.0.0"


@dataclass
class DataConfig:
    dataset_dir: str = "data/SkinDisease"
    train_subdir: str = "train"
    test_subdir: str = "test"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    val_fraction: float = 0.15
    split_seed: int = 42


@dataclass
class AugmentationConfig:
    # See README "Augmentation strategy" for the rationale behind each value.
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.3
    rotation_degrees: float = 30.0
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    # Hue jitter perturbs the exact color/redness/pigmentation signal that is
    # often diagnostic for skin conditions (erythema, pigmentation changes).
    # Reduced sharply from the original 0.1 -- kept nonzero only to absorb
    # small, realistic camera/white-balance variation.
    color_jitter_hue: float = 0.02
    affine_translate: float = 0.1


@dataclass
class ModelConfig:
    num_classes: int = 22
    dropout_rate: float = 0.5
    freeze_backbones: bool = False
    pretrained_backbones: bool = True
    model_version: str = MODEL_VERSION


@dataclass
class TrainConfig:
    epochs: int = 30
    backbone_lr: float = 1e-5
    classifier_lr: float = 1e-4
    weight_decay: float = 0.01
    use_class_weights: bool = False
    label_smoothing: float = 0.0
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 8
    checkpoint_metric: str = "val_macro_recall"  # documented model-selection metric
    secondary_metrics: tuple = ("val_macro_f1", "val_macro_precision")
    mixed_precision: bool = True
    seed: int = 42
    grad_clip_norm: Optional[float] = 5.0


@dataclass
class PathsConfig:
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    class_names_file: str = "models/class_names.json"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppConfig":
        return cls(
            data=DataConfig(**d.get("data", {})),
            augmentation=AugmentationConfig(**d.get("augmentation", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            paths=PathsConfig(**d.get("paths", {})),
        )

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AppConfig":
        if path is None:
            return cls()
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
