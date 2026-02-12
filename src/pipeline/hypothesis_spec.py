"""HypothesisSpec: Structured schema for experiment hypotheses.

Replaces free-text hypotheses with a structured JSON format to enable
validation, serialization, and programmatic analysis.

Example JSON:
{
    "hypothesis": "DeiT-Small outperforms ResNet-34 on Flowers102 due to self-attention",
    "confidence": 0.7,
    "task": "image_classification",
    "dataset": "flowers102",
    "baseline": {"model": "resnet34", "pretrain": "imagenet"},
    "variant": {"model": "deit_small", "pretrain": "imagenet"},
    "metric": "top1_accuracy",
    "expected_effect": {"direction": "increase", "min_delta_points": 2}
}
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class HypothesisSpec:
    """Structured specification for an experiment hypothesis.

    Attributes:
        hypothesis: Natural language description of the hypothesis
        confidence: Confidence level (0.0-1.0) in the hypothesis
        task: Task type (e.g., "image_classification", "object_detection")
        dataset: Dataset name (e.g., "flowers102", "coco")
        baseline: Dict with baseline model config {"model": str, "pretrain": str}
        variant: Dict with variant model config {"model": str, "pretrain": str}
        metric: Evaluation metric (e.g., "top1_accuracy", "mAP")
        expected_effect: Dict with expected outcome {"direction": str, "min_delta_points": float}
    """

    hypothesis: str
    confidence: float
    task: str
    dataset: str
    baseline: Dict[str, Any]
    variant: Dict[str, Any]
    metric: str
    expected_effect: Dict[str, Any]

    def validate(self) -> bool:
        """Validate that all required fields are present and non-empty.

        Returns:
            bool: True if all required fields are present, False otherwise

        Raises:
            ValueError: If any required field is missing or invalid
        """
        required_fields = {
            "hypothesis": str,
            "confidence": (float, int),
            "task": str,
            "dataset": str,
            "baseline": dict,
            "variant": dict,
            "metric": str,
            "expected_effect": dict,
        }

        for field_name, expected_type in required_fields.items():
            if not hasattr(self, field_name):
                raise ValueError(f"Missing required field: {field_name}")

            value = getattr(self, field_name)

            if value is None:
                raise ValueError(f"Field '{field_name}' cannot be None")

            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Field '{field_name}' has invalid type. "
                        f"Expected {expected_type}, got {type(value)}"
                    )
            else:
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Field '{field_name}' has invalid type. "
                        f"Expected {expected_type}, got {type(value)}"
                    )

            # Check non-empty for string fields
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"Field '{field_name}' cannot be empty")

            # Check non-empty for dict fields
            if isinstance(value, dict) and not value:
                raise ValueError(f"Field '{field_name}' cannot be empty")

        # Validate confidence is in valid range
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert HypothesisSpec to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the hypothesis spec
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypothesisSpec":
        """Create HypothesisSpec from dictionary.

        Args:
            data: Dictionary with hypothesis spec fields

        Returns:
            HypothesisSpec: Instance created from dictionary

        Raises:
            TypeError: If required fields are missing from dictionary
        """
        try:
            return cls(**data)
        except TypeError as e:
            raise TypeError(f"Failed to create HypothesisSpec from dict: {e}")
