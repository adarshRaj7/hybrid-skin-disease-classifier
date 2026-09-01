"""Final test-set evaluation. Run exactly once, after all model-selection
decisions have already been made on the validation set.

Writes:
  - outputs/test_metrics.json     (overall + severe-class metrics, machine readable)
  - outputs/per_class_metrics.csv (per-class precision/recall/F1/support)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .config import AppConfig
from .dataset import SkinLesionDataset
from .labels import ClassLabels, severe_class_names
from .transforms import build_eval_transform
from .utils import get_device, load_model

logger = logging.getLogger(__name__)


def _collect_predictions(model, loader, device):
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_labels, all_preds, all_probs


def evaluate_test_set(
    config: AppConfig,
    checkpoint_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    validation_history_path: Optional[str] = None,
) -> Dict[str, Any]:
    device = get_device()
    checkpoint_path = checkpoint_path or str(Path(config.paths.models_dir) / "best_model.pt")
    model, metadata = load_model(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    class_labels = ClassLabels(classes=metadata["class_names"])
    preprocessing = metadata["preprocessing"]
    eval_tf = build_eval_transform(
        image_size=preprocessing["image_size"],
        mean=tuple(preprocessing["mean"]),
        std=tuple(preprocessing["std"]),
    )

    dataset_dir = Path(dataset_dir or config.data.dataset_dir)
    test_dir = dataset_dir / config.data.test_subdir
    test_dataset = SkinLesionDataset(test_dir, class_labels, transform=eval_tf)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    logger.info("Evaluating on %d held-out test images (untouched during training)", len(test_dataset))
    labels, preds, probs = _collect_predictions(model, test_loader, device)

    accuracy = float(np.mean(np.array(preds) == np.array(labels)))
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_precision = precision_score(labels, preds, average="weighted", zero_division=0)
    weighted_recall = recall_score(labels, preds, average="weighted", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    per_class_precision = precision_score(labels, preds, average=None, zero_division=0, labels=range(class_labels.num_classes))
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0, labels=range(class_labels.num_classes))
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=range(class_labels.num_classes))
    support = np.bincount(labels, minlength=class_labels.num_classes)

    per_class_rows = []
    severe_recalls = {}
    for idx, class_name in enumerate(class_labels.classes):
        row = {
            "class": class_name,
            "precision": float(per_class_precision[idx]),
            "recall": float(per_class_recall[idx]),
            "f1": float(per_class_f1[idx]),
            "support": int(support[idx]),
        }
        per_class_rows.append(row)
        if class_name in severe_class_names():
            severe_recalls[class_name] = row["recall"]

    cm = confusion_matrix(labels, preds, labels=range(class_labels.num_classes)).tolist()

    validation_summary = None
    validation_history_path = validation_history_path or str(Path(config.paths.outputs_dir) / "training_history.json")
    if Path(validation_history_path).exists():
        with open(validation_history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if history:
            best_epoch = max(history, key=lambda r: r.get(config.train.checkpoint_metric, -1))
            validation_summary = {
                "best_epoch": best_epoch["epoch"],
                "val_accuracy": best_epoch["val_accuracy"],
                "val_macro_recall": best_epoch["val_macro_recall"],
                "val_macro_precision": best_epoch["val_macro_precision"],
                "val_macro_f1": best_epoch["val_macro_f1"],
            }

    results = {
        "note": (
            "This report reflects FINAL test-set performance, evaluated once "
            "after all model-selection decisions were made on the validation "
            "set. It is not a clinical validation study."
        ),
        "validation_performance_for_reference": validation_summary,
        "test_performance": {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "num_test_samples": len(labels),
        },
        "severe_class_recall": severe_recalls,
        "confusion_matrix": cm,
        "class_order": class_labels.classes,
    }

    outputs_dir = Path(config.paths.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(outputs_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(per_class_rows)

    logger.info(
        "Test results: accuracy=%.4f macro_recall=%.4f macro_f1=%.4f macro_precision=%.4f",
        accuracy,
        macro_recall,
        macro_f1,
        macro_precision,
    )
    if severe_recalls:
        logger.info("Severe-class recall: %s", severe_recalls)

    return results
