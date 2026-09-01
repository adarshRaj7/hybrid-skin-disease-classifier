"""The hybrid ResNet50 + DenseNet121 model.

This replaces the original notebook's two separate ``HybridFeatureExtractor``
instantiations (one at module scope for feature extraction / novelty
detection, one created again inside ``SkinDiseaseClassifier.__init__``) with
a single backbone that is built once and shared by the whole model. There is
now exactly one ResNet50 and one DenseNet121 in the model, and one coherent
forward path from image to logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class BackboneConfig:
    pretrained: bool = True
    freeze: bool = False


class HybridBackbone(nn.Module):
    """ResNet50 + DenseNet121 feature extractor (single instance of each).

    Input:  (B, 3, H, W)
    Output: (B, 3072) = concat(resnet_feat (2048), densenet_feat (1024))
    """

    resnet_feat_dim = 2048
    densenet_feat_dim = 1024
    combined_feat_dim = resnet_feat_dim + densenet_feat_dim

    def __init__(self, pretrained: bool = True, freeze: bool = False) -> None:
        super().__init__()

        weights = "DEFAULT" if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Drop the final FC layer; keep everything up to global average pool.
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        # Kept for Grad-CAM: the last conv block before global pooling.
        self.resnet_target_layer = resnet.layer4

        densenet = models.densenet121(weights=weights)
        self.densenet_features = densenet.features
        # Kept for Grad-CAM: DenseNet's final dense block.
        self.densenet_target_layer = densenet.features.denseblock4

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=False)

        if freeze:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.resnet_features.parameters():
            p.requires_grad = trainable
        for p in self.densenet_features.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resnet_feat = self.resnet_features(x)
        resnet_feat = torch.flatten(resnet_feat, 1)

        densenet_feat = self.densenet_features(x)
        # DenseNet's raw feature map needs ReLU + pooling (torchvision's own
        # classifier head does the same before its final Linear layer).
        densenet_feat = self.relu(densenet_feat)
        densenet_feat = self.adaptive_pool(densenet_feat)
        densenet_feat = torch.flatten(densenet_feat, 1)

        return torch.cat([resnet_feat, densenet_feat], dim=1)


class ClassifierHead(nn.Module):
    """The 4-linear-layer classification head, preserved from the original project."""

    def __init__(self, feature_dim: int, num_classes: int, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridSkinDiseaseModel(nn.Module):
    """Single coherent model: HybridBackbone -> ClassifierHead -> logits."""

    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.5,
        pretrained_backbones: bool = True,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = HybridBackbone(pretrained=pretrained_backbones, freeze=freeze_backbones)
        self.classifier = ClassifierHead(
            feature_dim=self.backbone.combined_feat_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]":
        features = self.backbone(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone.set_backbone_trainable(trainable)

    def parameter_groups(self, backbone_lr: float, classifier_lr: float):
        """Differential-learning-rate parameter groups for the optimizer."""
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr},
        ]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
