"""Dataset module.

Key production fixes relative to the original notebook:

* Class indices come from :class:`~skin_disease.labels.ClassLabels`, which is
  built once and persisted to JSON -- never re-derived from
  ``os.listdir()`` ordering at load time.
* Corrupted images are logged (not silently swallowed) and the dataset
  falls back to a *different* deterministic sample rather than recursively
  calling ``__getitem__`` on ``(idx + 1) % len(self)``, which in the
  original code could recurse arbitrarily deep (or infinitely, if every
  image happened to be corrupt) with no logging at all.
* A reproducible, stratified train/validation split is provided so that the
  test set is never used as a validation signal during training (see
  ``make_stratified_split``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .labels import ClassLabels

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SkinLesionDataset(Dataset):
    """An ImageFolder-style dataset with a fixed, externally supplied class mapping."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        class_labels: ClassLabels,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_labels = class_labels
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        for class_name in class_labels.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                logger.warning("Class directory missing, skipping: %s", class_dir)
                continue
            for entry in sorted(os.scandir(class_dir), key=lambda e: e.name):
                if not entry.is_file():
                    continue
                if Path(entry.name).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                self.samples.append((entry.path, class_labels.index_of(class_name)))

        if not self.samples:
            raise ValueError(f"No images found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> List[int]:
        return [label for _, label in self.samples]

    def _load_image(self, path: str) -> Optional[Image.Image]:
        try:
            with Image.open(path) as img:
                img.load()
                return img.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - we want to catch any decode error
            logger.warning("Skipping unreadable image %s (%s)", path, exc)
            return None

    def __getitem__(self, idx: int):
        n = len(self.samples)
        # Try the requested sample, then a bounded number of deterministic
        # fallbacks. This avoids both silent infinite recursion and crashing
        # a whole training run because of one bad file.
        for attempt in range(min(n, 10)):
            path, label = self.samples[(idx + attempt) % n]
            image = self._load_image(path)
            if image is not None:
                if self.transform:
                    image = self.transform(image)
                return image, label
        raise RuntimeError(
            f"Could not load a valid image after {min(n, 10)} attempts starting at index {idx}"
        )


def make_stratified_split(
    labels: Sequence[int],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Return (train_indices, val_indices), stratified by class label.

    This is used to carve a validation set out of the *training* directory
    only. The held-out test directory is never touched by this function and
    must only be used for final evaluation (see evaluate.py).
    """
    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=list(labels),
    )
    return train_idx, val_idx


def validate_directory(directory: Union[str, Path]) -> Tuple[List[str], int]:
    """Scan a dataset directory and report (unreadable_paths, total_count).

    This is an explicit, offline data-quality step (see
    ``scripts/prepare_dataset.py``) rather than something that happens
    silently inside training. It reports problems instead of deleting files,
    leaving the decision of what to do with bad files to the user.
    """
    directory = Path(directory)
    unreadable: List[str] = []
    total = 0
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        for entry in sorted(os.scandir(class_dir), key=lambda e: e.name):
            if not entry.is_file() or Path(entry.name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            total += 1
            try:
                with Image.open(entry.path) as img:
                    img.verify()
            except Exception:  # noqa: BLE001
                unreadable.append(entry.path)
    return unreadable, total
