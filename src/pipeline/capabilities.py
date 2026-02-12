"""
Single source of truth for experimental capabilities.

This module defines the CAPABILITIES dictionary that serves as the authoritative
reference for all available models, datasets, metrics, and constraints used in
the Fleming-AI pipeline. All components should reference this dictionary to ensure
consistency and prevent hypothesis-experiment mismatches.
"""

from typing import Any, Dict

CAPABILITIES: Dict[str, Any] = {
    "task": "image_classification",
    "models": ["deit_small", "resnet34"],
    "datasets": ["cifar10", "cifar100", "stl10", "flowers102", "oxford_pets"],
    "metrics": ["top1_accuracy", "loss"],
    "max_runtime_minutes": 30,
}
