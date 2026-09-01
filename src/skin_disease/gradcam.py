"""Grad-CAM explainability, refactored for the single-backbone hybrid model.

The original notebook's Grad-CAM was wired to
``model.feature_extractor.densenet_last_conv`` on the (duplicated) inner
feature extractor. Here it is wired directly to the one backbone the model
actually uses, and can target either branch since ResNet50 and DenseNet121
have different spatial resolutions and see the image differently.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import HybridSkinDiseaseModel

Branch = Literal["resnet", "densenet"]


class GradCAM:
    """Grad-CAM for one branch of :class:`HybridSkinDiseaseModel`."""

    def __init__(self, model: HybridSkinDiseaseModel, branch: Branch = "densenet") -> None:
        if branch not in ("resnet", "densenet"):
            raise ValueError("branch must be 'resnet' or 'densenet'")
        self.model = model
        self.branch = branch
        target_layer = (
            model.backbone.resnet_target_layer
            if branch == "resnet"
            else model.backbone.densenet_target_layer
        )
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """Generate a Grad-CAM heatmap for a single image.

        Args:
            image_tensor: (1, 3, H, W) preprocessed tensor, on the same
                device as the model.
            target_class: class index to explain. Defaults to the model's
                own predicted class.

        Returns:
            (heatmap, target_class) where heatmap is a float32 array in
            [0, 1] at the target layer's spatial resolution (the caller is
            expected to resize it to the input resolution for overlay).
        """
        if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
            raise ValueError("generate() expects a single-image batch of shape (1, 3, H, W)")

        was_training = self.model.training
        self.model.eval()
        image_tensor = image_tensor.clone().requires_grad_(True)

        logits = self.model(image_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, target_class]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire; check the target layer.")

        activations = self._activations[0]
        gradients = self._gradients[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        if was_training:
            self.model.train()

        return cam.cpu().numpy(), target_class

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Resize ``heatmap`` to ``image_rgb``'s resolution and alpha-blend a jet colormap.

    Args:
        image_rgb: (H, W, 3) float array in [0, 1].
        heatmap: (h, w) float array in [0, 1] (Grad-CAM output).
        alpha: blend strength for the heatmap.

    Returns:
        (H, W, 3) float array in [0, 1].
    """
    from PIL import Image

    h, w = image_rgb.shape[:2]
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heatmap_resized = np.asarray(heatmap_img).astype(np.float32) / 255.0

    # Simple jet-like colormap without an extra dependency (blue -> green -> red).
    r = np.clip(1.5 - np.abs(4 * heatmap_resized - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heatmap_resized - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heatmap_resized - 1), 0, 1)
    colored = np.stack([r, g, b], axis=-1)

    blended = (1 - alpha) * image_rgb + alpha * colored
    return np.clip(blended, 0, 1)
