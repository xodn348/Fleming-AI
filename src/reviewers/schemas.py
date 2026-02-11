"""
Review Schemas for Fleming-AI
Data structures for hypothesis review outcomes and conversation state
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ReviewResult:
    """
    Represents the result of a hypothesis review.

    Attributes:
        review_id: Unique identifier for this review
        hypothesis_id: ID of the reviewed hypothesis (optional)
        stage: Current stage of the pipeline (e.g., "hypothesis_generation", "validation")
        turn_number: Turn number in the review conversation
        verdict: Review verdict (PASS, REVISE, RESTART_STAGE, RESTART_PIPELINE)
        strengths: List of identified strengths
        weaknesses: List of identified weaknesses
        questions: List of clarifying questions
        suggestions: List of improvement suggestions
        scores: Dictionary of numerical scores (e.g., clarity, rigor, novelty)
        requested_experiments: List of requested experiments (optional)
        timestamp: ISO format timestamp of review
    """

    review_id: str
    hypothesis_id: Optional[str]
    stage: str
    turn_number: int
    verdict: str  # PASS, REVISE, RESTART_STAGE, RESTART_PIPELINE
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    requested_experiments: Optional[list[dict[str, Any]]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate verdict after initialization."""
        valid_verdicts = {"PASS", "REVISE", "RESTART_STAGE", "RESTART_PIPELINE"}
        if self.verdict not in valid_verdicts:
            raise ValueError(f"Invalid verdict '{self.verdict}'. Must be one of: {valid_verdicts}")


@dataclass
class ReviewTurn:
    """
    Represents a single turn in a review conversation.

    Attributes:
        turn_id: Unique identifier for this turn
        speaker: Who spoke (fleming or alex)
        content: The text content of the turn
        structured_data: Structured data extracted from the turn
        timestamp: ISO format timestamp of the turn
    """

    turn_id: str
    speaker: str  # fleming or alex
    content: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationState:
    """
    Represents the state of a review conversation.

    Attributes:
        session_id: Unique identifier for this review session
        stage: Current pipeline stage
        turns: List of review turns in this conversation
        current_score: Current overall score (optional)
        is_converged: Whether the conversation has converged to a decision
    """

    session_id: str
    stage: str
    turns: list[ReviewTurn] = field(default_factory=list)
    current_score: Optional[float] = None
    is_converged: bool = False
