"""Image preprocessing pipelines.

Validation, test, and production inference MUST use the exact same
deterministic pipeline (``build_eval_transform``) so that reported metrics
are representative of what the deployed model actually sees. Only training
uses augmentation.
"""

from __future__ import annotations

from typing import Tuple

from torchvision import transforms

from .config import AugmentationConfig, IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(
    image_size: int = 224,
    aug: AugmentationConfig | None = None,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Training transform: resize + augmentation + normalization.

    Augmentation choices (see README for full rationale):
      - Horizontal/vertical flip and rotation are kept: close-up lesion crops
        have no canonical "up" orientation, unlike natural scene photos.
      - Color jitter brightness/contrast/saturation are kept at mild levels
        to absorb camera and lighting variation.
      - Hue jitter is reduced sharply (0.1 -> 0.02) because hue encodes
        clinically relevant signal (erythema/redness, pigmentation) that the
        model is meant to learn from; aggressively randomizing it risks
        teaching the model to ignore a genuinely diagnostic feature.
      - Affine translation is kept mild to simulate imperfect framing.
    """
    aug = aug or AugmentationConfig()
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=aug.horizontal_flip_p),
            transforms.RandomVerticalFlip(p=aug.vertical_flip_p),
            transforms.RandomRotation(degrees=aug.rotation_degrees),
            transforms.ColorJitter(
                brightness=aug.color_jitter_brightness,
                contrast=aug.color_jitter_contrast,
                saturation=aug.color_jitter_saturation,
                hue=aug.color_jitter_hue,
            ),
            transforms.RandomAffine(degrees=0, translate=(aug.affine_translate, aug.affine_translate)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_eval_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Deterministic transform shared by validation, test, and inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
