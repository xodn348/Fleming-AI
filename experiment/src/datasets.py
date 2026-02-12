"""
Dataset loading and transforms for experiment datasets via HuggingFace.
All images resized to 224x224 for compatibility with both DeiT and ResNet.
"""

import logging
from typing import Any, Optional

import datasets as hf_datasets
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# HuggingFace dataset IDs and their metadata
DATASET_REGISTRY: dict[str, dict] = {
    "cifar10": {
        "hf_id": "uoft-cs/cifar10",
        "num_classes": 10,
        "image_key": "img",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
    },
    "cifar100": {
        "hf_id": "uoft-cs/cifar100",
        "num_classes": 100,
        "image_key": "img",
        "label_key": "fine_label",
        "train_split": "train",
        "test_split": "test",
    },
    "stl10": {
        "hf_id": "tanganke/stl10",
        "num_classes": 10,
        "image_key": "image",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
    },
    "flowers102": {
        "hf_id": "nelorth/oxford-flowers",
        "num_classes": 102,
        "image_key": "image",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
    },
    "oxford_pets": {
        "hf_id": "timm/oxford-iiit-pet",
        "num_classes": 37,
        "image_key": "image",
        "label_key": "label",
        "train_split": "train",
        "test_split": "test",
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class HFImageDataset(Dataset):
    """Wraps a HuggingFace dataset as a PyTorch Dataset with transforms."""

    def __init__(
        self,
        hf_dataset,
        image_key: str,
        label_key: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.hf_dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        item = self.hf_dataset[idx]
        image = item[self.image_key]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Ensure RGB (some datasets have grayscale or RGBA)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(item[self.label_key])
        return image, label


def load_dataset(
    name: str,
    split: str,
    transform: Optional[transforms.Compose] = None,
) -> Dataset:
    """
    Load dataset from HuggingFace and wrap as PyTorch Dataset.

    Args:
        name: Dataset name — one of "cifar10", "cifar100", "stl10", "flowers102", "oxford_pets"
        split: "train" or "test"
        transform: Optional torchvision transform pipeline

    Returns:
        PyTorch Dataset yielding (image_tensor, label) tuples

    Raises:
        ValueError: If name not in DATASET_REGISTRY
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_REGISTRY.keys())}")

    meta = DATASET_REGISTRY[name]
    split_name = meta["train_split"] if split == "train" else meta["test_split"]

    logger.info(f"Loading {name} ({split}) from HuggingFace: {meta['hf_id']}")
    hf_ds = hf_datasets.load_dataset(meta["hf_id"], split=split_name)

    dataset = HFImageDataset(
        hf_dataset=hf_ds,
        image_key=meta["image_key"],
        label_key=meta["label_key"],
        transform=transform,
    )
    logger.info(f"Loaded {name} ({split}): {len(dataset)} samples, {meta['num_classes']} classes")

    return dataset


def get_transforms(pretrained: bool, train: bool) -> transforms.Compose:
    """
    Build transform pipeline for training or evaluation.

    Args:
        pretrained: True = use ImageNet normalization stats
        train: True = apply data augmentation, False = center crop only

    Returns:
        torchvision.transforms.Compose pipeline
    """
    mean = IMAGENET_MEAN if pretrained else [0.5, 0.5, 0.5]
    std = IMAGENET_STD if pretrained else [0.5, 0.5, 0.5]

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def get_num_classes(name: str) -> int:
    """Return the number of classes for a dataset."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'")
    return DATASET_REGISTRY[name]["num_classes"]
