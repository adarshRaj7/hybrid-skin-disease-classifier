import numpy as np
import torch

from skin_disease.gradcam import GradCAM, overlay_heatmap


def test_target_layers_exist(tiny_model):
    assert tiny_model.backbone.resnet_target_layer is not None
    assert tiny_model.backbone.densenet_target_layer is not None


def test_gradcam_generates_heatmap_densenet_branch(tiny_model):
    x = torch.randn(1, 3, 64, 64)
    cam = GradCAM(tiny_model, branch="densenet")
    try:
        heatmap, target_class = cam.generate(x)
    finally:
        cam.remove_hooks()

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert isinstance(target_class, int)


def test_gradcam_generates_heatmap_resnet_branch(tiny_model):
    x = torch.randn(1, 3, 64, 64)
    cam = GradCAM(tiny_model, branch="resnet")
    try:
        heatmap, _ = cam.generate(x)
    finally:
        cam.remove_hooks()
    assert heatmap.ndim == 2


def test_gradcam_explicit_target_class(tiny_model, class_labels):
    x = torch.randn(1, 3, 64, 64)
    cam = GradCAM(tiny_model, branch="densenet")
    try:
        _, resolved_class = cam.generate(x, target_class=1)
    finally:
        cam.remove_hooks()
    assert resolved_class == 1


def test_gradcam_context_manager_removes_hooks(tiny_model):
    x = torch.randn(1, 3, 64, 64)
    with GradCAM(tiny_model, branch="densenet") as cam:
        cam.generate(x)
    # After the context exits, hooks should be removed; the underlying
    # torch modules should have no remaining forward/backward hooks left by us.
    assert cam._forward_handle.id not in tiny_model.backbone.densenet_target_layer._forward_hooks


def test_overlay_heatmap_output_shape():
    image = np.random.rand(64, 64, 3).astype(np.float32)
    heatmap = np.random.rand(4, 4).astype(np.float32)
    overlay = overlay_heatmap(image, heatmap)
    assert overlay.shape == (64, 64, 3)
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0 + 1e-6
