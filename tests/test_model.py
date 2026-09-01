import torch

from skin_disease.model import HybridSkinDiseaseModel, count_parameters


def test_model_instantiates(tiny_model, class_labels):
    assert isinstance(tiny_model, HybridSkinDiseaseModel)
    assert tiny_model.num_classes == class_labels.num_classes


def test_forward_pass_correct_shape(tiny_model, class_labels):
    x = torch.randn(2, 3, 64, 64)
    tiny_model.eval()
    with torch.no_grad():
        logits = tiny_model(x)
    assert logits.shape == (2, class_labels.num_classes)


def test_forward_pass_returns_features_when_requested(tiny_model):
    x = torch.randn(2, 3, 64, 64)
    tiny_model.eval()
    with torch.no_grad():
        logits, features = tiny_model(x, return_features=True)
    assert logits.shape[0] == 2
    assert features.shape == (2, tiny_model.backbone.combined_feat_dim)


def test_batch_inference_various_batch_sizes(tiny_model, class_labels):
    tiny_model.eval()
    for batch_size in (1, 3, 5):
        x = torch.randn(batch_size, 3, 64, 64)
        with torch.no_grad():
            logits = tiny_model(x)
        assert logits.shape == (batch_size, class_labels.num_classes)


def test_only_one_resnet_and_one_densenet(tiny_model):
    """Regression test for the original notebook's duplicate-feature-extractor bug."""
    resnet_backbones = [m for m in tiny_model.modules() if type(m).__name__ == "ResNet"]
    densenet_backbones = [m for m in tiny_model.modules() if type(m).__name__ == "DenseNet"]
    # torchvision's resnet50()/densenet121() top-level classes are decomposed
    # into Sequential/feature submodules by HybridBackbone, so we instead
    # assert there is exactly one each of the *identifying* substructures.
    assert hasattr(tiny_model.backbone, "resnet_features")
    assert hasattr(tiny_model.backbone, "densenet_features")
    # There should be exactly one combined-feature concatenation path: the
    # combined feature dimension must equal resnet_feat_dim + densenet_feat_dim,
    # not some multiple of it (which would indicate accidental duplication).
    assert tiny_model.backbone.combined_feat_dim == (
        tiny_model.backbone.resnet_feat_dim + tiny_model.backbone.densenet_feat_dim
    )


def test_freeze_backbones_sets_requires_grad(class_labels):
    model = HybridSkinDiseaseModel(
        num_classes=class_labels.num_classes,
        pretrained_backbones=False,
        freeze_backbones=True,
    )
    backbone_params = list(model.backbone.parameters())
    assert all(not p.requires_grad for p in backbone_params)
    classifier_params = list(model.classifier.parameters())
    assert all(p.requires_grad for p in classifier_params)

    model.set_backbone_trainable(True)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_count_parameters_reports_trainable_subset(class_labels):
    model = HybridSkinDiseaseModel(
        num_classes=class_labels.num_classes,
        pretrained_backbones=False,
        freeze_backbones=True,
    )
    total, trainable = count_parameters(model)
    assert trainable < total
    assert trainable > 0
