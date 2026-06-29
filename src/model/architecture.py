"""Arquitectura del clasificador: EfficientNet-B0 con transfer learning.

Por qué EfficientNet-B0:
  - excelente relación precisión/costo -> liviano para inferencia y contenedor,
  - pesos preentrenados en ImageNet -> convergencia rápida con pocos datos por clase,
  - apto para edge/campo (modelo pequeño).
La cabeza se deja con `num_classes` salidas (2 para binario) para poder extender
a multiclase (madurez, tipo de enfermedad) sin reescribir la arquitectura.
"""
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_model(num_classes: int = 2, pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        set_backbone_trainable(model, False)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool = True) -> None:
    """Congela/descongela el extractor de features (todo menos la cabeza)."""
    for param in model.features.parameters():
        param.requires_grad = trainable
