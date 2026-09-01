"""Training pipeline.

Fixes relative to the original notebook:

* The test directory is never used during training. A stratified
  validation split is carved out of the *training* directory
  (``make_stratified_split``) and used for checkpoint selection; the test
  set is reserved entirely for ``evaluate.py``.
* The checkpoint-selection metric is **validation macro recall** by default
  (configurable), not validation accuracy, because in a medical
  decision-support setting a missed disease (false negative) tends to
  matter more than a percentage point of overall accuracy. Macro F1 and
  macro precision are tracked alongside it every epoch specifically so a
  pathological "predict everything as one class to maximize recall"
  degenerate solution would be visible and could be caught before shipping.
* Class weighting (if enabled) is computed once from observed training
  class counts (inverse frequency) and logged, rather than applied
  invisibly.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset

from .config import AppConfig
from .dataset import SkinLesionDataset, make_stratified_split
from .labels import ClassLabels
from .model import HybridSkinDiseaseModel, count_parameters
from .transforms import build_eval_transform, build_train_transform
from .utils import get_device, save_checkpoint, set_seed

logger = logging.getLogger(__name__)


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, computed from the training split only."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid division by zero for absent classes
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_split(
    model: HybridSkinDiseaseModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels))) if n else 0.0
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
    }


def train(config: AppConfig, dataset_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run the full training loop and return a summary dict.

    Also writes:
      - ``{paths.models_dir}/best_model.pt``     (best checkpoint, self-describing)
      - ``{paths.class_names_file}``              (deterministic class mapping)
      - ``{paths.outputs_dir}/training_history.json``
    """
    set_seed(config.train.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    dataset_dir = Path(dataset_dir or config.data.dataset_dir)
    train_dir = dataset_dir / config.data.train_subdir

    class_labels = ClassLabels.from_directory(train_dir)
    class_labels.save(config.paths.class_names_file)
    config.model.num_classes = class_labels.num_classes
    logger.info("Discovered %d classes", class_labels.num_classes)

    train_tf = build_train_transform(config.data.image_size, config.augmentation)
    eval_tf = build_eval_transform(config.data.image_size)

    full_train_dataset_for_labels = SkinLesionDataset(train_dir, class_labels, transform=None)
    train_idx, val_idx = make_stratified_split(
        full_train_dataset_for_labels.labels,
        val_fraction=config.data.val_fraction,
        seed=config.data.split_seed,
    )
    logger.info("Train/val split: %d train, %d val (seed=%d)", len(train_idx), len(val_idx), config.data.split_seed)

    train_dataset = Subset(SkinLesionDataset(train_dir, class_labels, transform=train_tf), train_idx)
    val_dataset = Subset(SkinLesionDataset(train_dir, class_labels, transform=eval_tf), val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    model = HybridSkinDiseaseModel(
        num_classes=class_labels.num_classes,
        dropout_rate=config.model.dropout_rate,
        pretrained_backbones=config.model.pretrained_backbones,
        freeze_backbones=config.model.freeze_backbones,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info("Model parameters: total=%d trainable=%d", total_params, trainable_params)

    class_weights = None
    if config.train.use_class_weights:
        train_labels = [full_train_dataset_for_labels.labels[i] for i in train_idx]
        class_weights = compute_class_weights(train_labels, class_labels.num_classes).to(device)
        logger.info("Using inverse-frequency class weights: %s", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.train.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameter_groups(config.train.backbone_lr, config.train.classifier_lr),
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.train.scheduler_factor, patience=config.train.scheduler_patience
    )

    use_amp = config.train.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: List[Dict[str, Any]] = []
    best_metric_value = -float("inf")
    epochs_without_improvement = 0
    models_dir = Path(config.paths.models_dir)
    best_checkpoint_path = models_dir / "best_model.pt"

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if config.train.grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_metrics = evaluate_split(model, val_loader, device, criterion)
        checkpoint_metric_value = val_metrics[config.train.checkpoint_metric.replace("val_", "")]
        current_lr = optimizer.param_groups[-1]["lr"]
        scheduler.step(checkpoint_metric_value)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": current_lr,
            "seconds": time.time() - start,
        }
        history.append(epoch_record)

        is_best = checkpoint_metric_value > best_metric_value
        logger.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f val_%s=%.4f val_macro_f1=%.4f val_macro_precision=%.4f%s",
            epoch,
            train_loss,
            train_acc,
            config.train.checkpoint_metric.replace("val_", ""),
            checkpoint_metric_value,
            val_metrics["macro_f1"],
            val_metrics["macro_precision"],
            " (best)" if is_best else "",
        )

        if is_best:
            best_metric_value = checkpoint_metric_value
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                path=best_checkpoint_path,
                class_names=class_labels.classes,
                image_size=config.data.image_size,
                training_config=config.to_dict(),
                checkpoint_metric=config.train.checkpoint_metric,
                best_metric_value=best_metric_value,
                extra={"epoch": epoch},
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.train.early_stopping_patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, epochs_without_improvement)
                break

    outputs_dir = Path(config.paths.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_checkpoint": str(best_checkpoint_path),
        "checkpoint_metric": config.train.checkpoint_metric,
        "best_metric_value": best_metric_value,
        "epochs_run": len(history),
        "class_names_file": config.paths.class_names_file,
    }
    logger.info("Training complete: %s", summary)
    return summary
