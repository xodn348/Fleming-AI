"""
Hypothesis Classifier for Fleming-AI
Classifies hypotheses by validation method feasibility
"""

import json
import logging
from typing import Any, Optional

from src.generators.hypothesis import Hypothesis
from src.validators.result import (
    CLASS_COMPUTATIONAL,
    CLASS_DATA_DRIVEN,
    CLASS_EXPERIMENTAL,
    CLASS_THEORETICAL,
)

logger = logging.getLogger(__name__)


# Keywords indicating computational validation is possible
COMPUTATIONAL_KEYWORDS = [
    "algorithm",
    "computational",
    "simulation",
    "model",
    "predict",
    "calculate",
    "compute",
    "benchmark",
    "performance",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "optimization",
    "machine learning",
    "neural network",
    "deep learning",
    "regression",
    "classification",
    "clustering",
    "statistical",
    "correlation",
    "mathematical",
]

# Keywords indicating data-driven validation is possible
DATA_DRIVEN_KEYWORDS = [
    "dataset",
    "database",
    "public data",
    "open data",
    "clinical trial",
    "survey",
    "population study",
    "epidemiological",
    "census",
    "registry",
    "cohort",
    "longitudinal",
    "meta-analysis",
    "systematic review",
    "literature review",
    "bibliometric",
]

# Keywords indicating experimental validation is required
EXPERIMENTAL_KEYWORDS = [
    "laboratory",
    "experiment",
    "in vitro",
    "in vivo",
    "clinical trial",
    "randomized controlled",
    "synthesis",
    "chemical reaction",
    "biological assay",
    "cell culture",
    "animal model",
    "human subjects",
    "physical measurement",
    "spectroscopy",
    "chromatography",
    "imaging",
    "biopsy",
    "wet lab",
]

# Keywords indicating theoretical validation
THEORETICAL_KEYWORDS = [
    "theoretical",
    "proof",
    "theorem",
    "axiom",
    "conjecture",
    "philosophical",
    "conceptual",
    "framework",
    "paradigm",
    "qualitative analysis",
    "interpretive",
    "phenomenological",
    "hermeneutic",
    "critical analysis",
    "discourse analysis",
]


class HypothesisClassifier:
    """
    Classifies hypotheses by the type of validation method required.

    Classification types:
    - computational: Can be validated with code/simulations
    - data_driven: Can be validated with public datasets
    - experimental: Requires physical experiments (not programmatically testable)
    - theoretical: Requires theoretical/formal analysis
    """

    def __init__(
        self,
        ollama_client: Optional[Any] = None,
        use_llm: bool = False,
    ):
        """
        Initialize the HypothesisClassifier.

        Args:
            ollama_client: Optional OllamaClient for LLM-based classification
            use_llm: Whether to use LLM for classification (default: heuristic)
        """
        self.ollama = ollama_client
        self.use_llm = use_llm and ollama_client is not None

    def classify(self, hypothesis: Hypothesis) -> str:
        """
        Classify a hypothesis by validation method.

        Args:
            hypothesis: Hypothesis object to classify

        Returns:
            Classification string: "computational", "data_driven",
                                  "experimental", or "theoretical"
        """
        text = hypothesis.hypothesis_text.lower()
        connection = hypothesis.connection

        # Also consider bridging concept for classification
        bridging_concept = connection.get("bridging_concept", "").lower()
        concept_a = connection.get("concept_a", "").lower()
        concept_b = connection.get("concept_b", "").lower()

        full_text = f"{text} {bridging_concept} {concept_a} {concept_b}"

        return self._classify_by_keywords(full_text)

    async def classify_async(self, hypothesis: Hypothesis) -> str:
        """
        Classify a hypothesis using LLM (async).

        Args:
            hypothesis: Hypothesis object to classify

        Returns:
            Classification string
        """
        if not self.use_llm or self.ollama is None:
            return self.classify(hypothesis)

        return await self._classify_with_llm(hypothesis)

    def _classify_by_keywords(self, text: str) -> str:
        """
        Classify hypothesis using keyword matching.

        Args:
            text: Combined text from hypothesis

        Returns:
            Classification string
        """
        scores = {
            CLASS_COMPUTATIONAL: 0,
            CLASS_DATA_DRIVEN: 0,
            CLASS_EXPERIMENTAL: 0,
            CLASS_THEORETICAL: 0,
        }

        # Count keyword matches for each category
        for keyword in COMPUTATIONAL_KEYWORDS:
            if keyword in text:
                scores[CLASS_COMPUTATIONAL] += 1

        for keyword in DATA_DRIVEN_KEYWORDS:
            if keyword in text:
                scores[CLASS_DATA_DRIVEN] += 1

        for keyword in EXPERIMENTAL_KEYWORDS:
            if keyword in text:
                scores[CLASS_EXPERIMENTAL] += 1

        for keyword in THEORETICAL_KEYWORDS:
            if keyword in text:
                scores[CLASS_THEORETICAL] += 1

        # Find highest scoring category
        max_score = max(scores.values())

        if max_score == 0:
            # Default to theoretical if no keywords match
            return CLASS_THEORETICAL

        # Return the category with highest score
        # Priority: computational > data_driven > experimental > theoretical
        # (prefer testable categories when tied)
        for category in [
            CLASS_COMPUTATIONAL,
            CLASS_DATA_DRIVEN,
            CLASS_EXPERIMENTAL,
            CLASS_THEORETICAL,
        ]:
            if scores[category] == max_score:
                return category

        return CLASS_THEORETICAL

    async def _classify_with_llm(self, hypothesis: Hypothesis) -> str:
        """
        Classify hypothesis using LLM.

        Args:
            hypothesis: Hypothesis object to classify

        Returns:
            Classification string
        """
        prompt = f"""Classify this scientific hypothesis by the type of validation method required.

Hypothesis: {hypothesis.hypothesis_text}

Connection:
- Concept A: {hypothesis.connection.get("concept_a", "N/A")}
- Concept B: {hypothesis.connection.get("concept_b", "N/A")}
- Bridging Concept: {hypothesis.connection.get("bridging_concept", "N/A")}

Classification options:
1. "computational": Can be validated with code, simulations, or mathematical analysis
2. "data_driven": Can be validated with existing public datasets
3. "experimental": Requires physical laboratory experiments or clinical trials
4. "theoretical": Requires formal theoretical proof or philosophical analysis

Return ONLY a JSON object:
{{"classification": "<one of the four options>", "reasoning": "<brief explanation>"}}
"""

        ollama_client = self.ollama
        if ollama_client is None:
            return self.classify(hypothesis)

        try:
            response = await ollama_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )

            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0]

            result = json.loads(response)
            classification = result.get("classification", "")

            if classification in [
                CLASS_COMPUTATIONAL,
                CLASS_DATA_DRIVEN,
                CLASS_EXPERIMENTAL,
                CLASS_THEORETICAL,
            ]:
                return classification

            logger.warning(f"Invalid LLM classification: {classification}")
            return self.classify(hypothesis)  # Fallback to heuristic

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM classification failed: {e}")
            return self.classify(hypothesis)  # Fallback to heuristic

    def classify_batch(self, hypotheses: list[Hypothesis]) -> dict[str, str]:
        """
        Classify multiple hypotheses.

        Args:
            hypotheses: List of Hypothesis objects

        Returns:
            Dictionary mapping hypothesis ID to classification
        """
        return {h.id: self.classify(h) for h in hypotheses}

    async def classify_batch_async(
        self,
        hypotheses: list[Hypothesis],
    ) -> dict[str, str]:
        """
        Classify multiple hypotheses using LLM (async).

        Args:
            hypotheses: List of Hypothesis objects

        Returns:
            Dictionary mapping hypothesis ID to classification
        """
        results = {}
        for h in hypotheses:
            results[h.id] = await self.classify_async(h)
        return results

    def get_classification_stats(
        self,
        hypotheses: list[Hypothesis],
    ) -> dict[str, int]:
        """
        Get statistics on hypothesis classifications.

        Args:
            hypotheses: List of Hypothesis objects

        Returns:
            Dictionary with counts for each classification
        """
        stats = {
            CLASS_COMPUTATIONAL: 0,
            CLASS_DATA_DRIVEN: 0,
            CLASS_EXPERIMENTAL: 0,
            CLASS_THEORETICAL: 0,
        }

        for h in hypotheses:
            classification = self.classify(h)
            stats[classification] += 1

        return stats

    def is_programmatically_testable(self, hypothesis: Hypothesis) -> bool:
        """
        Check if hypothesis can be tested programmatically.

        Args:
            hypothesis: Hypothesis to check

        Returns:
            True if computational or data_driven, False otherwise
        """
        classification = self.classify(hypothesis)
        return classification in [CLASS_COMPUTATIONAL, CLASS_DATA_DRIVEN]
