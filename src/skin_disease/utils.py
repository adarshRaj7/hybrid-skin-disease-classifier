"""Shared utilities: reproducibility, logging setup, and checkpoint I/O.

The checkpoint format bundles everything needed to reconstruct the model
without the original notebook: weights, architecture configuration, class
mapping, preprocessing parameters, and training/version metadata (see
``save_checkpoint`` / ``load_model``).
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .config import IMAGENET_MEAN, IMAGENET_STD, MODEL_VERSION
from .model import HybridSkinDiseaseModel


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all relevant RNG seeds.

    ``deterministic=True`` trades some throughput for reproducibility by
    asking cuDNN to use deterministic algorithms. Set to False if training
    speed matters more than bit-for-bit reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: HybridSkinDiseaseModel,
    path: Union[str, Path],
    class_names: list,
    image_size: int = 224,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    model_version: str = MODEL_VERSION,
    training_config: Optional[Dict[str, Any]] = None,
    checkpoint_metric: Optional[str] = None,
    best_metric_value: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a self-describing checkpoint (weights + full reconstruction metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "architecture": {
            "num_classes": model.num_classes,
            "dropout_rate": model.dropout_rate,
        },
        "class_names": list(class_names),
        "preprocessing": {
            "image_size": image_size,
            "mean": list(mean),
            "std": list(std),
        },
        "model_version": model_version,
        "training_config": training_config or {},
        "checkpoint_metric": checkpoint_metric,
        "best_metric_value": best_metric_value,
    }
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)


def load_model(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
) -> "tuple[HybridSkinDiseaseModel, Dict[str, Any]]":
    """Reconstruct a model + its metadata from a checkpoint produced by ``save_checkpoint``.

    Returns ``(model, metadata)`` where ``metadata`` contains everything
    other than the raw ``state_dict`` (class names, preprocessing config,
    version, training config, etc.). The model is returned in ``eval()``
    mode with pretrained-backbone downloading disabled (weights come from
    the checkpoint, not from torchvision's pretrained weights).
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location or "cpu", weights_only=False)

    arch = checkpoint["architecture"]
    model = HybridSkinDiseaseModel(
        num_classes=arch["num_classes"],
        dropout_rate=arch.get("dropout_rate", 0.5),
        pretrained_backbones=False,  # weights come from the checkpoint
        freeze_backbones=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return model, metadata
