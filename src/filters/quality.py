"""
Quality Filter for Fleming-AI
Scores academic papers based on research quality patterns
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.filters.pattern_extractor import PatternExtractor

logger = logging.getLogger(__name__)

# Patterns file path
PATTERNS_FILE = Path(__file__).parent / "patterns.json"


@dataclass
class QualityAnalysis:
    """Detailed quality analysis result"""

    overall_score: float
    clarity_score: float
    methodology_score: float
    evidence_score: float
    novelty_score: float
    impact_score: float
    patterns: dict
    details: dict

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "overall_score": self.overall_score,
            "clarity_score": self.clarity_score,
            "methodology_score": self.methodology_score,
            "evidence_score": self.evidence_score,
            "novelty_score": self.novelty_score,
            "impact_score": self.impact_score,
            "patterns": self.patterns,
            "details": self.details,
        }


class QualityFilter:
    """Filters and scores academic papers based on quality patterns"""

    def __init__(
        self,
        patterns_file: Optional[Path] = None,
        model: str = "qwen2.5:7b",
    ):
        """
        Initialize the QualityFilter

        Args:
            patterns_file: Path to patterns JSON file
            model: Ollama model to use for LLM-based extraction
        """
        self.patterns_file = patterns_file or PATTERNS_FILE
        self.patterns = self._load_patterns()
        self.pattern_extractor = PatternExtractor(model=model, patterns_file=self.patterns_file)

    def _load_patterns(self) -> dict:
        """Load patterns from JSON file"""
        try:
            with open(self.patterns_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Patterns file not found: {self.patterns_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing patterns file: {e}")
            return {}

    def score(self, paper_text: str) -> float:
        """
        Calculate quality score for a paper (synchronous, heuristic-based)

        Args:
            paper_text: Full text of the academic paper

        Returns:
            Quality score between 0.0 and 1.0
        """
        analysis = self.analyze_heuristic(paper_text)
        return analysis.overall_score

    async def score_async(self, paper_text: str) -> float:
        """
        Calculate quality score for a paper (async, LLM-based)

        Args:
            paper_text: Full text of the academic paper

        Returns:
            Quality score between 0.0 and 1.0
        """
        analysis = await self.analyze(paper_text)
        return analysis.overall_score

    async def analyze(self, paper_text: str) -> QualityAnalysis:
        """
        Perform detailed quality analysis of a paper (LLM-based)

        Args:
            paper_text: Full text of the academic paper

        Returns:
            QualityAnalysis with detailed scores and patterns
        """
        # Extract patterns using LLM
        patterns = await self.pattern_extractor.extract_patterns(paper_text)

        # Calculate component scores
        clarity_score = self._calculate_clarity_score(paper_text, patterns)
        methodology_score = self._calculate_methodology_score(paper_text, patterns)
        evidence_score = self._calculate_evidence_score(paper_text, patterns)
        novelty_score = self._calculate_novelty_score(paper_text, patterns)
        impact_score = self._calculate_impact_score(paper_text, patterns)

        # Get weights from patterns
        weights = self.patterns.get("quality_criteria", {})
        clarity_weight = weights.get("clarity", {}).get("weight", 0.2)
        methodology_weight = weights.get("methodology_rigor", {}).get("weight", 0.25)
        evidence_weight = weights.get("evidence_strength", {}).get("weight", 0.25)
        novelty_weight = weights.get("novelty", {}).get("weight", 0.15)
        impact_weight = weights.get("impact", {}).get("weight", 0.15)

        # Calculate weighted overall score
        overall_score = (
            clarity_score * clarity_weight
            + methodology_score * methodology_weight
            + evidence_score * evidence_weight
            + novelty_score * novelty_weight
            + impact_score * impact_weight
        )

        # Build details
        details = {
            "clarity": {
                "score": clarity_score,
                "weight": clarity_weight,
                "has_clear_question": self._has_clear_research_question(paper_text),
                "has_objectives": self._has_objectives(paper_text),
            },
            "methodology": {
                "score": methodology_score,
                "weight": methodology_weight,
                "detected_type": patterns.get("methodology", "unknown"),
            },
            "evidence": {
                "score": evidence_score,
                "weight": evidence_weight,
                "detected_type": patterns.get("evidence_type", "unknown"),
            },
            "novelty": {
                "score": novelty_score,
                "weight": novelty_weight,
                "claim": patterns.get("novelty_claim", ""),
            },
            "impact": {
                "score": impact_score,
                "weight": impact_weight,
            },
        }

        return QualityAnalysis(
            overall_score=round(overall_score, 3),
            clarity_score=round(clarity_score, 3),
            methodology_score=round(methodology_score, 3),
            evidence_score=round(evidence_score, 3),
            novelty_score=round(novelty_score, 3),
            impact_score=round(impact_score, 3),
            patterns=patterns,
            details=details,
        )

    def analyze_heuristic(self, paper_text: str) -> QualityAnalysis:
        """
        Perform quality analysis using only heuristics (no LLM)

        Args:
            paper_text: Full text of the academic paper

        Returns:
            QualityAnalysis with detailed scores and patterns
        """
        # Extract patterns using heuristics
        patterns = self.pattern_extractor.extract_patterns_heuristic(paper_text)

        # Calculate component scores
        clarity_score = self._calculate_clarity_score(paper_text, patterns)
        methodology_score = self._calculate_methodology_score(paper_text, patterns)
        evidence_score = self._calculate_evidence_score(paper_text, patterns)
        novelty_score = self._calculate_novelty_score(paper_text, patterns)
        impact_score = self._calculate_impact_score(paper_text, patterns)

        # Get weights from patterns
        weights = self.patterns.get("quality_criteria", {})
        clarity_weight = weights.get("clarity", {}).get("weight", 0.2)
        methodology_weight = weights.get("methodology_rigor", {}).get("weight", 0.25)
        evidence_weight = weights.get("evidence_strength", {}).get("weight", 0.25)
        novelty_weight = weights.get("novelty", {}).get("weight", 0.15)
        impact_weight = weights.get("impact", {}).get("weight", 0.15)

        # Calculate weighted overall score
        overall_score = (
            clarity_score * clarity_weight
            + methodology_score * methodology_weight
            + evidence_score * evidence_weight
            + novelty_score * novelty_weight
            + impact_score * impact_weight
        )

        # Build details
        details = {
            "clarity": {
                "score": clarity_score,
                "weight": clarity_weight,
                "has_clear_question": self._has_clear_research_question(paper_text),
                "has_objectives": self._has_objectives(paper_text),
            },
            "methodology": {
                "score": methodology_score,
                "weight": methodology_weight,
                "detected_type": patterns.get("methodology", "unknown"),
            },
            "evidence": {
                "score": evidence_score,
                "weight": evidence_weight,
                "detected_type": patterns.get("evidence_type", "unknown"),
            },
            "novelty": {
                "score": novelty_score,
                "weight": novelty_weight,
                "claim": patterns.get("novelty_claim", ""),
            },
            "impact": {
                "score": impact_score,
                "weight": impact_weight,
            },
        }

        return QualityAnalysis(
            overall_score=round(overall_score, 3),
            clarity_score=round(clarity_score, 3),
            methodology_score=round(methodology_score, 3),
            evidence_score=round(evidence_score, 3),
            novelty_score=round(novelty_score, 3),
            impact_score=round(impact_score, 3),
            patterns=patterns,
            details=details,
        )

    def _calculate_clarity_score(self, paper_text: str, patterns: dict) -> float:
        """Calculate clarity score based on research question and objectives"""
        score = 0.0
        text_lower = paper_text.lower()

        # Check for clear research question
        if self._has_clear_research_question(text_lower):
            score += 0.4

        # Check for clear objectives
        if self._has_objectives(text_lower):
            score += 0.3

        # Check if research question type is identified
        if patterns.get("research_question_type") != "unknown":
            score += 0.3

        return min(score, 1.0)

    def _calculate_methodology_score(self, paper_text: str, patterns: dict) -> float:
        """Calculate methodology score"""
        score = 0.0
        text_lower = paper_text.lower()

        methodology = patterns.get("methodology", "unknown")
        methodology_patterns = self.patterns.get("methodology_patterns", {})

        # Base score from methodology type
        if methodology in methodology_patterns:
            method_info = methodology_patterns[methodology]
            score += method_info.get("strength", 0.5) * 0.5

            # Check for required elements
            required_elements = method_info.get("required_elements", [])
            elements_found = sum(1 for elem in required_elements if elem in text_lower)
            if required_elements:
                score += 0.5 * (elements_found / len(required_elements))
        elif methodology != "unknown":
            score += 0.3  # Unknown but detected methodology

        # Bonus for reproducibility mentions
        reproducibility_terms = ["reproducible", "code available", "open source", "replication"]
        if any(term in text_lower for term in reproducibility_terms):
            score += 0.1

        return min(score, 1.0)

    def _calculate_evidence_score(self, paper_text: str, patterns: dict) -> float:
        """Calculate evidence strength score"""
        score = 0.0
        text_lower = paper_text.lower()

        evidence_type = patterns.get("evidence_type", "unknown")
        evidence_patterns = self.patterns.get("evidence_patterns", {})

        # Base score from evidence type
        if evidence_type in evidence_patterns:
            evidence_info = evidence_patterns[evidence_type]
            score += evidence_info.get("strength", 0.5) * 0.6

        # Check for statistical rigor
        statistical_terms = [
            "p-value",
            "statistical significance",
            "confidence interval",
            "effect size",
        ]
        stat_count = sum(1 for term in statistical_terms if term in text_lower)
        score += min(stat_count * 0.1, 0.2)

        # Check for quantitative results
        if re.search(r"\d+\.?\d*%", paper_text):
            score += 0.1

        # Check for comparison with baselines
        if "baseline" in text_lower or "state-of-the-art" in text_lower:
            score += 0.1

        return min(score, 1.0)

    def _calculate_novelty_score(self, paper_text: str, patterns: dict) -> float:
        """Calculate novelty score"""
        score = 0.0
        text_lower = paper_text.lower()

        novelty_indicators = self.patterns.get("novelty_indicators", {})

        # Check for strong novelty indicators
        strong_indicators = novelty_indicators.get("strong", {}).get("phrases", [])
        moderate_indicators = novelty_indicators.get("moderate", {}).get("phrases", [])
        incremental_indicators = novelty_indicators.get("incremental", {}).get("phrases", [])

        if any(phrase in text_lower for phrase in strong_indicators):
            score = 0.8
        elif any(phrase in text_lower for phrase in moderate_indicators):
            score = 0.6
        elif any(phrase in text_lower for phrase in incremental_indicators):
            score = 0.4

        # Bonus for explicit contribution statement
        if "contribution" in text_lower or "we propose" in text_lower:
            score += 0.1

        # Check if novelty claim was extracted
        if patterns.get("novelty_claim"):
            score += 0.1

        return min(score, 1.0)

    def _calculate_impact_score(self, paper_text: str, patterns: dict) -> float:  # noqa: ARG002
        """Calculate potential impact score"""
        score = 0.0
        text_lower = paper_text.lower()

        # Check for practical implications
        practical_terms = ["practical", "application", "real-world", "industry", "deployment"]
        if any(term in text_lower for term in practical_terms):
            score += 0.3

        # Check for future work/directions
        future_terms = ["future work", "future directions", "further research", "open problems"]
        if any(term in text_lower for term in future_terms):
            score += 0.2

        # Check for broader impact discussion
        impact_terms = ["broader impact", "societal", "implications", "significance"]
        if any(term in text_lower for term in impact_terms):
            score += 0.2

        # Check for limitations acknowledgment (shows rigor)
        if "limitation" in text_lower:
            score += 0.15

        # Check for ethical considerations
        if "ethical" in text_lower or "ethics" in text_lower:
            score += 0.15

        return min(score, 1.0)

    def _has_clear_research_question(self, text: str) -> bool:
        """Check if paper has a clear research question"""
        text_lower = text.lower() if isinstance(text, str) else text
        question_indicators = [
            "research question",
            "we investigate",
            "we study",
            "we address",
            "we examine",
            "we explore",
            "we ask",
            "this paper investigates",
            "this work investigates",
        ]
        return any(indicator in text_lower for indicator in question_indicators)

    def _has_objectives(self, text: str) -> bool:
        """Check if paper has clear objectives"""
        text_lower = text.lower() if isinstance(text, str) else text
        objective_indicators = [
            "objective",
            "aim",
            "goal",
            "purpose",
            "we aim to",
            "our goal is",
            "we propose to",
        ]
        return any(indicator in text_lower for indicator in objective_indicators)

    def filter_papers(
        self,
        papers: list[str],
        min_score: float = 0.5,
    ) -> list[tuple[int, str, float]]:
        """
        Filter papers by quality score

        Args:
            papers: List of paper texts
            min_score: Minimum quality score to pass filter

        Returns:
            List of tuples (index, paper_text, score) for papers above threshold
        """
        results = []
        for i, paper_text in enumerate(papers):
            score = self.score(paper_text)
            if score >= min_score:
                results.append((i, paper_text, score))

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results
