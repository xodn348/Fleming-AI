"""
Pattern Extractor for Fleming-AI
Extracts research patterns from academic papers using Ollama LLM
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Patterns file path
PATTERNS_FILE = Path(__file__).parent / "patterns.json"

# Extraction prompt template
EXTRACTION_PROMPT = """Analyze the following academic paper text and extract research patterns.

Paper Text:
{paper_text}

Extract and return a JSON object with the following structure:
{{
    "research_question_type": "what" | "how" | "why" (the type of research question),
    "methodology": "experimental" | "theoretical" | "empirical" | "survey" | "case_study",
    "novelty_claim": "description of the main novel contribution",
    "evidence_type": "quantitative" | "qualitative" | "formal_proof" | "comparative"
}}

Return ONLY the JSON object, no additional text."""

SYSTEM_PROMPT = """You are an expert academic paper analyzer. 
Your task is to extract key research patterns from papers.
Always respond with valid JSON only, no markdown or explanations."""


class PatternExtractor:
    """Extracts research patterns from academic papers using LLM"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        patterns_file: Optional[Path] = None,
    ):
        """
        Initialize the PatternExtractor

        Args:
            model: Ollama model to use for extraction
            patterns_file: Path to patterns JSON file
        """
        self.model = model
        self.patterns_file = patterns_file or PATTERNS_FILE
        self.patterns = self._load_patterns()

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

    async def extract_patterns(self, paper_text: str) -> dict:
        """
        Extract patterns from paper text using LLM

        Args:
            paper_text: Full text of the academic paper

        Returns:
            Dict with extracted patterns:
            {
                "research_question_type": "what/how/why",
                "methodology": "experimental/theoretical/survey/etc",
                "novelty_claim": str,
                "evidence_type": "empirical/formal/case_study/etc"
            }
        """
        # Truncate paper text to reasonable length for LLM
        max_chars = 8000
        truncated_text = paper_text[:max_chars]
        if len(paper_text) > max_chars:
            truncated_text += "\n\n[... paper continues ...]"

        prompt = EXTRACTION_PROMPT.format(paper_text=truncated_text)

        async with OllamaClient(model=self.model) as client:
            try:
                response = await client.generate(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=500,
                    stream=False,
                )

                # Parse JSON response (response is always str when stream=False)
                extracted = self._parse_extraction_response(str(response))
                return extracted

            except Exception as e:
                logger.error(f"Error extracting patterns: {e}")
                return self._get_default_patterns()

    def _parse_extraction_response(self, response: str) -> dict:
        """Parse LLM response into structured patterns"""
        try:
            # Try to extract JSON from response
            # Handle potential markdown code blocks
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                # Validate and normalize extracted data
                return {
                    "research_question_type": self._validate_question_type(
                        data.get("research_question_type", "")
                    ),
                    "methodology": self._validate_methodology(data.get("methodology", "")),
                    "novelty_claim": str(data.get("novelty_claim", ""))[:500],
                    "evidence_type": self._validate_evidence_type(data.get("evidence_type", "")),
                }
            else:
                logger.warning("No JSON found in LLM response")
                return self._get_default_patterns()

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._get_default_patterns()

    def _validate_question_type(self, value: str) -> str:
        """Validate research question type"""
        valid_types = ["what", "how", "why"]
        value_lower = value.lower().strip()
        return value_lower if value_lower in valid_types else "unknown"

    def _validate_methodology(self, value: str) -> str:
        """Validate methodology type"""
        valid_methods = ["experimental", "theoretical", "empirical", "survey", "case_study"]
        value_lower = value.lower().strip().replace(" ", "_")
        return value_lower if value_lower in valid_methods else "unknown"

    def _validate_evidence_type(self, value: str) -> str:
        """Validate evidence type"""
        valid_types = ["quantitative", "qualitative", "formal_proof", "comparative"]
        value_lower = value.lower().strip().replace(" ", "_")
        return value_lower if value_lower in valid_types else "unknown"

    def _get_default_patterns(self) -> dict:
        """Return default patterns when extraction fails"""
        return {
            "research_question_type": "unknown",
            "methodology": "unknown",
            "novelty_claim": "",
            "evidence_type": "unknown",
        }

    def extract_patterns_heuristic(self, paper_text: str) -> dict:
        """
        Extract patterns using heuristic matching (no LLM required)

        Args:
            paper_text: Full text of the academic paper

        Returns:
            Dict with extracted patterns using heuristic matching
        """
        text_lower = paper_text.lower()

        # Detect research question type
        question_type = self._detect_question_type_heuristic(text_lower)

        # Detect methodology
        methodology = self._detect_methodology_heuristic(text_lower)

        # Detect evidence type
        evidence_type = self._detect_evidence_type_heuristic(text_lower)

        return {
            "research_question_type": question_type,
            "methodology": methodology,
            "novelty_claim": "",  # Cannot extract without LLM
            "evidence_type": evidence_type,
        }

    def _detect_question_type_heuristic(self, text: str) -> str:
        """Detect research question type using heuristics"""
        scores = {"what": 0, "how": 0, "why": 0}

        for q_type, pattern_info in self.patterns.get("research_question_types", {}).items():
            for indicator in pattern_info.get("indicators", []):
                if indicator in text:
                    scores[q_type] += 1

        if all(v == 0 for v in scores.values()):
            return "unknown"
        return max(scores, key=lambda k: scores[k])

    def _detect_methodology_heuristic(self, text: str) -> str:
        """Detect methodology type using heuristics"""
        scores: dict[str, int] = {}

        for method, pattern_info in self.patterns.get("methodology_patterns", {}).items():
            score = 0
            for indicator in pattern_info.get("indicators", []):
                if indicator in text:
                    score += 1
            scores[method] = score

        if not scores or all(v == 0 for v in scores.values()):
            return "unknown"
        return max(scores, key=lambda k: scores[k])

    def _detect_evidence_type_heuristic(self, text: str) -> str:
        """Detect evidence type using heuristics"""
        scores: dict[str, int] = {}

        for ev_type, pattern_info in self.patterns.get("evidence_patterns", {}).items():
            score = 0
            for indicator in pattern_info.get("indicators", []):
                if indicator in text:
                    score += 1
            scores[ev_type] = score

        if not scores or all(v == 0 for v in scores.values()):
            return "unknown"
        return max(scores, key=lambda k: scores[k])
