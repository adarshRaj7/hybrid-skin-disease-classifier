"""Class-label and severity-metadata utilities.

The severity mapping below is copied verbatim from the original research
notebook. It is **application metadata / presentation logic only** -- it was
authored by the project team to group classes for display purposes and has
not been clinically validated. It must never be interpreted as, or
communicated as, a medical severity assessment. See the README "Medical
disclaimer" section.

This module also defines the deterministic class-index mapping used
throughout training, evaluation, checkpointing and inference. Production
code must never rely on ``os.listdir()`` ordering -- the mapping is fixed
once at dataset-preparation time and persisted to ``class_names.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

# ---------------------------------------------------------------------------
# Severity metadata (preserved from the original notebook).
# ---------------------------------------------------------------------------

SEVERITY_MAPPING: Dict[str, str] = {
    # Mild
    "Acne": "MILD",
    "Candidiasis": "MILD",
    "Eczema": "MILD",
    "Infestations_Bites": "MILD",
    "Rosacea": "MILD",
    "Warts": "MILD",
    "Tinea": "MILD",
    # Moderate
    "Bullous": "MODERATE",
    "DrugEruption": "MODERATE",
    "Lichen": "MODERATE",
    "Moles": "MODERATE",
    "Seborrh_Keratoses": "MODERATE",
    "Benign_tumors": "MODERATE",
    # Severe
    "Actinic_Keratosis": "SEVERE",
    "Lupus": "SEVERE",
    "Psoriasis": "SEVERE",
    "SkinCancer": "SEVERE",
    "Vasculitis": "SEVERE",
    "Vascular_Tumors": "SEVERE",
    # Other
    "Vitiligo": "OTHER",
    "Sun_Sunlight_Damage": "OTHER",
    "Unknown_Normal": "OTHER",
}

SEVERITY_COLORS: Dict[str, str] = {
    "MILD": "#4CAF50",
    "MODERATE": "#FF9800",
    "SEVERE": "#F44336",
    "OTHER": "#9E9E9E",
}

SEVERITY_ORDER: List[str] = ["MILD", "MODERATE", "SEVERE", "OTHER"]

SEVERITY_PRIORITY: Dict[str, int] = {
    "Actinic_Keratosis": 1,
    "Lupus": 2,
    "Psoriasis": 3,
    "SkinCancer": 4,
    "Vasculitis": 5,
    "Vascular_Tumors": 6,
    "Bullous": 7,
    "DrugEruption": 8,
    "Lichen": 9,
    "Moles": 10,
    "Seborrh_Keratoses": 11,
    "Benign_tumors": 12,
    "Acne": 13,
    "Candidiasis": 14,
    "Eczema": 15,
    "Infestations_Bites": 16,
    "Rosacea": 17,
    "Warts": 18,
    "Tinea": 19,
    "Unknown_Normal": 20,
    "Vitiligo": 21,
    "Sun_Sunlight_Damage": 22,
}


def get_severity(class_name: str) -> str:
    """Return the (non-clinical) severity bucket for a class name."""
    return SEVERITY_MAPPING.get(class_name, "OTHER")


def get_priority(class_name: str) -> int:
    """Return the display-priority rank for a class name (lower = higher priority)."""
    return SEVERITY_PRIORITY.get(class_name, 99)


def severe_class_names() -> List[str]:
    """Return the class names currently categorized as SEVERE."""
    return [name for name, sev in SEVERITY_MAPPING.items() if sev == "SEVERE"]


# ---------------------------------------------------------------------------
# Deterministic class <-> index mapping.
# ---------------------------------------------------------------------------


@dataclass
class ClassLabels:
    """Deterministic, persisted mapping between class names and indices.

    The mapping is built once (sorted alphabetically, matching the original
    project's convention) when a training dataset is first prepared, then
    written to ``class_names.json`` and reused everywhere afterwards so that
    training, evaluation, checkpoint loading, inference and API responses can
    never disagree about what index N means.
    """

    classes: List[str]

    def __post_init__(self) -> None:
        if len(self.classes) != len(set(self.classes)):
            raise ValueError("Class names must be unique.")
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def index_of(self, class_name: str) -> int:
        return self.class_to_idx[class_name]

    def name_of(self, index: int) -> str:
        return self.classes[index]

    @classmethod
    def from_directory(cls, root_dir: Union[str, Path]) -> "ClassLabels":
        """Build a deterministic mapping from an ImageFolder-style directory.

        Subdirectory names are sorted alphabetically. This should only be
        called once, at dataset-preparation time; the result should then be
        saved with :meth:`save` and reloaded with :meth:`load` everywhere
        else so the mapping never silently drifts due to filesystem
        ordering or a changed dataset layout.
        """
        root_dir = Path(root_dir)
        classes = sorted(
            entry.name for entry in os.scandir(root_dir) if entry.is_dir()
        )
        if not classes:
            raise ValueError(f"No class subdirectories found under {root_dir}")
        return cls(classes=classes)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClassLabels":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(classes=data["classes"])

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"classes": self.classes}, f, indent=2)
