"""
Validation Result for Fleming-AI
Data structures for hypothesis validation outcomes
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# Validation status constants
STATUS_VERIFIED = "verified"
STATUS_REFUTED = "refuted"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_NOT_TESTABLE = "not_testable"

VALID_STATUSES = {STATUS_VERIFIED, STATUS_REFUTED, STATUS_INCONCLUSIVE, STATUS_NOT_TESTABLE}


# Hypothesis classification types
CLASS_COMPUTATIONAL = "computational"
CLASS_DATA_DRIVEN = "data_driven"
CLASS_EXPERIMENTAL = "experimental"
CLASS_THEORETICAL = "theoretical"

VALID_CLASSIFICATIONS = {
    CLASS_COMPUTATIONAL,
    CLASS_DATA_DRIVEN,
    CLASS_EXPERIMENTAL,
    CLASS_THEORETICAL,
}


@dataclass
class ValidationResult:
    """
    Represents the result of hypothesis validation.

    Attributes:
        hypothesis_id: ID of the validated hypothesis
        status: Validation status (verified, refuted, inconclusive, not_testable)
        evidence: Dictionary containing evidence data
        logs: List of validation log messages
        validated_at: Timestamp of validation
        classification: How the hypothesis was classified
        execution_time_ms: Time taken for validation in milliseconds
        code_executed: Optional code that was executed for validation
        error: Optional error message if validation failed
    """

    hypothesis_id: str
    status: str  # verified, refuted, inconclusive, not_testable
    evidence: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)
    classification: str = ""
    execution_time_ms: int = 0
    code_executed: str = ""
    error: str = ""

    def __post_init__(self):
        """Validate status after initialization."""
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of: {VALID_STATUSES}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "status": self.status,
            "evidence": self.evidence,
            "logs": self.logs,
            "validated_at": self.validated_at.isoformat(),
            "classification": self.classification,
            "execution_time_ms": self.execution_time_ms,
            "code_executed": self.code_executed,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        """Create ValidationResult from dictionary."""
        validated_at = data.get("validated_at")
        if isinstance(validated_at, str):
            validated_at = datetime.fromisoformat(validated_at)
        elif validated_at is None:
            validated_at = datetime.now()

        return cls(
            hypothesis_id=data["hypothesis_id"],
            status=data["status"],
            evidence=data.get("evidence", {}),
            logs=data.get("logs", []),
            validated_at=validated_at,
            classification=data.get("classification", ""),
            execution_time_ms=data.get("execution_time_ms", 0),
            code_executed=data.get("code_executed", ""),
            error=data.get("error", ""),
        )

    def is_verified(self) -> bool:
        """Check if hypothesis was verified."""
        return self.status == STATUS_VERIFIED

    def is_refuted(self) -> bool:
        """Check if hypothesis was refuted."""
        return self.status == STATUS_REFUTED

    def is_testable(self) -> bool:
        """Check if hypothesis was testable."""
        return self.status != STATUS_NOT_TESTABLE

    def add_log(self, message: str) -> None:
        """Add a log message."""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")

    def add_evidence(self, key: str, value: Any) -> None:
        """Add evidence to the result."""
        self.evidence[key] = value
