from __future__ import annotations

from typing import Any

import torch
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install timm: pip install timm") from exc


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet/DenseNet/etc. through timm.

    Examples:
        efficientnet_b0, efficientnet_b3, densenet121, resnet50
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def get_last_conv_layer(model: nn.Module) -> tuple[str, nn.Module]:
    """Find the last Conv2d layer, useful for Grad-CAM."""
    last_name = ""
    last_module: nn.Module | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module
    if last_module is None:
        raise ValueError("Could not find a Conv2d layer in the model.")
    return last_name, last_module


def load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
