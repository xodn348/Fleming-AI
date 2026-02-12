"""
Model loading and feature extraction for DeiT-Small and ResNet-34.

Uses timm library for both architectures. Supports pretrained (ImageNet)
and randomly initialized weights with frozen backbone for probing.
"""

import logging
from typing import Optional

import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# timm model names for our two architectures
MODEL_REGISTRY: dict[str, str] = {
    "deit_small": "deit_small_patch16_224",
    "resnet34": "resnet34",
}

# Feature dimensions from penultimate layer
FEATURE_DIMS: dict[str, int] = {
    "deit_small": 384,
    "resnet34": 512,
}


def load_model(
    arch: str,
    pretrained: bool,
    num_classes: int,
) -> nn.Module:
    """
    Load model with frozen backbone + trainable classification head.

    Args:
        arch: "deit_small" or "resnet34"
        pretrained: True for ImageNet weights, False for random init
        num_classes: Number of output classes for classification head

    Returns:
        nn.Module with frozen backbone and trainable head

    Raises:
        ValueError: If arch not in MODEL_REGISTRY
    """
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    model_name = MODEL_REGISTRY[arch]
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classification head
    head = _get_head(model, arch)
    for param in head.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Loaded {arch} (pretrained={pretrained}): "
        f"{total_params / 1e6:.1f}M total, {trainable_params / 1e6:.3f}M trainable"
    )

    return model


def _get_head(model: nn.Module, arch: str) -> nn.Module:
    """Return the classification head module for a given architecture."""
    head_attr = "head" if arch == "deit_small" else "fc" if arch == "resnet34" else None
    if head_attr is None:
        raise ValueError(f"No head mapping for architecture: {arch}")
    head = getattr(model, head_attr)
    if not isinstance(head, nn.Module):
        raise TypeError(f"Expected nn.Module for head, got {type(head)}")
    return head


def load_model_unfrozen(
    arch: str,
    pretrained: bool,
    num_classes: int,
) -> nn.Module:
    """
    Load model with ALL parameters trainable (for from-scratch training).

    Args:
        arch: "deit_small" or "resnet34"
        pretrained: True for ImageNet weights, False for random init
        num_classes: Number of output classes

    Returns:
        nn.Module with all parameters trainable
    """
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    model_name = MODEL_REGISTRY[arch]
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded {arch} unfrozen (pretrained={pretrained}): {total_params / 1e6:.1f}M params"
    )

    return model


def extract_features(
    model: nn.Module,
    dataloader: "DataLoader[tuple[Tensor, Tensor]]",
    device: torch.device,
    arch: Optional[str] = None,
) -> tuple[Tensor, Tensor]:
    """
    Extract penultimate layer features for linear probing / k-NN.

    - DeiT-Small: [CLS] token from last transformer layer (N x 384)
    - ResNet-34: global average pool output (N x 512)

    Args:
        model: Model with frozen backbone
        dataloader: Data to extract features from
        device: torch.device (MPS or CPU)
        arch: Architecture hint; auto-detected if None

    Returns:
        (features, labels) — features shape (N, D), labels shape (N,)
    """
    model = model.to(device)
    model.eval()

    all_features: list[Tensor] = []
    all_labels: list[Tensor] = []

    # Use timm's forward_features to get penultimate representations
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            forward_features = getattr(model, "forward_features")
            features: Tensor = forward_features(images)

            # DeiT returns (B, num_patches+1, D) — take [CLS] token at index 0
            if features.dim() == 3:
                features = features[:, 0]

            # ResNet returns (B, C, H, W) after final conv — global avg pool
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])

            all_features.append(features.cpu())
            all_labels.append(labels)

            # Periodic MPS cache cleanup
            if device.type == "mps" and (batch_idx + 1) % 50 == 0:
                torch.mps.empty_cache()

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    logger.info(f"Extracted features: {features_tensor.shape}")

    return features_tensor, labels_tensor
