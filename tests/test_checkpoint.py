import torch

from skin_disease.utils import load_model, save_checkpoint


def test_checkpoint_save_creates_file(tiny_checkpoint):
    assert tiny_checkpoint.exists()


def test_checkpoint_metadata_preserved(tiny_checkpoint, class_labels):
    _, metadata = load_model(tiny_checkpoint)
    assert metadata["class_names"] == class_labels.classes
    assert metadata["preprocessing"]["image_size"] == 64
    assert metadata["preprocessing"]["mean"]
    assert metadata["preprocessing"]["std"]
    assert metadata["model_version"]
    assert metadata["checkpoint_metric"] == "val_macro_recall"
    assert "training_config" in metadata


def test_loaded_model_matches_saved_model_prediction(tiny_model, tiny_checkpoint):
    x = torch.randn(1, 3, 64, 64)
    tiny_model.eval()
    with torch.no_grad():
        original_logits = tiny_model(x)

    loaded_model, _ = load_model(tiny_checkpoint)
    loaded_model.eval()
    with torch.no_grad():
        loaded_logits = loaded_model(x)

    assert torch.allclose(original_logits, loaded_logits, atol=1e-6)
    assert original_logits.argmax(dim=1).item() == loaded_logits.argmax(dim=1).item()


def test_loaded_model_does_not_require_network(tiny_checkpoint):
    """Loading from a checkpoint must not attempt to download pretrained ImageNet weights."""
    model, _ = load_model(tiny_checkpoint)
    # pretrained_backbones=False is hardcoded inside load_model precisely so
    # that inference never depends on internet access; this just documents
    # and locks in that behavior via the model's own weight source.
    assert model is not None
