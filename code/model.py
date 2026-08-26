"""Lightweight edge classifier for the MedMNIST study: MobileNetV3-Small (a realistic
on-device backbone) with ImageNet pretraining, adapted for variable channel count
and class count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_model(n_classes: int, n_channels: int = 3, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    if n_channels == 1:
        # Replace the first conv to accept 1 channel; seed it with the mean of
        # the pretrained RGB filters so we keep the pretrained signal.
        old = model.features[0][0]
        new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        model.features[0][0] = new

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, n_classes)
    return model
