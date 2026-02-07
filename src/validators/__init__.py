"""
Validation module for Fleming-AI
Hypothesis classification and validation pipeline
"""

from src.validators.classifier import HypothesisClassifier
from src.validators.pipeline import ValidationPipeline
from src.validators.result import (
    CLASS_COMPUTATIONAL,
    CLASS_DATA_DRIVEN,
    CLASS_EXPERIMENTAL,
    CLASS_THEORETICAL,
    STATUS_INCONCLUSIVE,
    STATUS_NOT_TESTABLE,
    STATUS_REFUTED,
    STATUS_VERIFIED,
    ValidationResult,
)

__all__ = [
    "HypothesisClassifier",
    "ValidationPipeline",
    "ValidationResult",
    "CLASS_COMPUTATIONAL",
    "CLASS_DATA_DRIVEN",
    "CLASS_EXPERIMENTAL",
    "CLASS_THEORETICAL",
    "STATUS_VERIFIED",
    "STATUS_REFUTED",
    "STATUS_INCONCLUSIVE",
    "STATUS_NOT_TESTABLE",
]
