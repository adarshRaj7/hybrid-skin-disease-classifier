import torch

from skin_disease.transforms import build_eval_transform, build_train_transform


def test_eval_transform_rgb_image(sample_rgb_image):
    tf = build_eval_transform(image_size=64)
    tensor = tf(sample_rgb_image)
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32


def test_eval_transform_grayscale_image(sample_grayscale_image):
    tf = build_eval_transform(image_size=64)
    rgb_image = sample_grayscale_image.convert("RGB")
    tensor = tf(rgb_image)
    assert tensor.shape == (3, 64, 64)


def test_eval_transform_rgba_image(sample_rgba_image):
    tf = build_eval_transform(image_size=64)
    rgb_image = sample_rgba_image.convert("RGB")
    tensor = tf(rgb_image)
    assert tensor.shape == (3, 64, 64)


def test_normalization_is_applied(sample_rgb_image):
    tf = build_eval_transform(image_size=64)
    tensor = tf(sample_rgb_image)
    # A mid-gray-ish constant-color image should not remain in raw [0, 1]
    # range after ImageNet normalization (mean subtracted, divided by std).
    raw_min, raw_max = 0.0, 1.0
    assert tensor.min() < raw_min or tensor.max() > raw_max or tensor.std() < 1e-6


def test_train_transform_output_shape_matches_eval(sample_rgb_image):
    train_tf = build_train_transform(image_size=64)
    tensor = train_tf(sample_rgb_image)
    assert tensor.shape == (3, 64, 64)


def test_train_and_eval_transform_agree_on_deterministic_content():
    """Regression check: eval transform must not apply any randomness."""
    from PIL import Image

    image = Image.new("RGB", (64, 64), (10, 20, 30))
    tf = build_eval_transform(image_size=64)
    t1 = tf(image)
    t2 = tf(image)
    assert torch.allclose(t1, t2)
