"""Production inference module.

    predictor = SkinDiseasePredictor(checkpoint_path="models/model.pt")
    result = predictor.predict(image_bytes_or_path_or_pil_image)

``result['confidence']`` is a *model confidence*, not a medical certainty --
see the README medical disclaimer. Severity is predefined project metadata,
not a clinical assessment (see ``skin_disease.labels``).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image, UnidentifiedImageError

from .gradcam import GradCAM, overlay_heatmap
from .labels import get_severity
from .transforms import build_eval_transform
from .utils import get_device, load_model

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB, mirrors the API's own upload limit


class InvalidImageError(ValueError):
    """Raised when the supplied image cannot be safely decoded."""


ImageInput = Union[str, Path, bytes, Image.Image]


class SkinDiseasePredictor:
    """Loads a checkpoint once and serves predictions."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        hf_repo_id: Optional[str] = None,
        hf_filename: str = "model.pt",
        device: Optional[torch.device] = None,
    ) -> None:
        """Load a model either from a local checkpoint or a Hugging Face repo.

        Exactly one of ``checkpoint_path`` or ``hf_repo_id`` should be given.
        Local filesystem loading is used for development and for tests;
        ``hf_repo_id`` is the path used once the model is published to
        Hugging Face (see scripts/package_for_huggingface.py).
        """
        if checkpoint_path is None and hf_repo_id is None:
            raise ValueError("Provide either checkpoint_path or hf_repo_id")

        if checkpoint_path is None:
            checkpoint_path = self._download_from_hub(hf_repo_id, hf_filename)

        self.device = device or get_device()
        self.model, self.metadata = load_model(checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.class_names: List[str] = self.metadata["class_names"]
        preprocessing = self.metadata["preprocessing"]
        self.image_size = preprocessing["image_size"]
        self.transform = build_eval_transform(
            image_size=self.image_size,
            mean=tuple(preprocessing["mean"]),
            std=tuple(preprocessing["std"]),
        )
        self.model_version = self.metadata.get("model_version", "unknown")
        logger.info(
            "Loaded model version=%s classes=%d device=%s",
            self.model_version,
            len(self.class_names),
            self.device,
        )

    @staticmethod
    def _download_from_hub(repo_id: str, filename: str) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise ImportError(
                "huggingface_hub is required to load a model from the Hugging Face Hub. "
                "Install it with `pip install huggingface_hub`."
            ) from exc
        # Do not hardcode a username/repo/token here -- repo_id and any auth
        # token come from configuration/environment (see README "Hugging Face").
        return hf_hub_download(repo_id=repo_id, filename=filename)

    # ------------------------------------------------------------------
    # Image decoding
    # ------------------------------------------------------------------

    def _load_image(self, image: ImageInput) -> Image.Image:
        try:
            if isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, (bytes, bytearray)):
                if len(image) > MAX_IMAGE_BYTES:
                    raise InvalidImageError("Image exceeds maximum allowed size.")
                pil_image = Image.open(io.BytesIO(image))
                pil_image.load()
            else:  # path-like
                path = Path(image)
                if not path.exists():
                    raise InvalidImageError(f"File not found: {path}")
                if path.stat().st_size > MAX_IMAGE_BYTES:
                    raise InvalidImageError("Image exceeds maximum allowed size.")
                pil_image = Image.open(path)
                pil_image.load()
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(f"Could not decode image: {exc}") from exc

        # Normalizes RGB, RGBA, grayscale (L), and palette (P) images alike.
        return pil_image.convert("RGB")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, image: ImageInput, top_k: int = 3) -> Dict[str, Any]:
        pil_image = self._load_image(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probs, top_k)

        top_predictions = [
            {"class": self.class_names[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top_probs, top_indices)
        ]
        predicted_class = top_predictions[0]["class"]

        return {
            "predicted_class": predicted_class,
            "confidence": top_predictions[0]["confidence"],
            "top_3_predictions": top_predictions,
            "severity": get_severity(predicted_class),
            "model_version": self.model_version,
        }

    def generate_gradcam(
        self,
        image: ImageInput,
        target_class: Optional[int] = None,
        branch: str = "densenet",
    ) -> Dict[str, Any]:
        """Generate a Grad-CAM overlay for an image.

        Returns a dict with the overlaid RGB image (as a float array in
        [0, 1], HxWx3) and the class index/name it explains. Grad-CAM is
        never required for a normal ``predict()`` call -- callers opt in
        explicitly.
        """
        import numpy as np

        pil_image = self._load_image(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        cam = GradCAM(self.model, branch=branch)
        try:
            heatmap, resolved_class = cam.generate(tensor, target_class=target_class)
        finally:
            cam.remove_hooks()

        resized = pil_image.resize((self.image_size, self.image_size))
        image_np = np.asarray(resized).astype(np.float32) / 255.0
        overlay = overlay_heatmap(image_np, heatmap)

        return {
            "overlay": overlay,
            "target_class_index": resolved_class,
            "target_class_name": self.class_names[resolved_class],
        }
