"""
Alex Reviewer Module for Fleming-AI
Expert ML paper reviewer with anti-sycophancy protocol
"""

import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.llm.groq_client import GroqClient
from src.reviewers.schemas import ReviewResult, ReviewTurn
from src.reviewers import knowledge

logger = logging.getLogger(__name__)


class Alex:
    """
    Expert ML paper reviewer implementing NeurIPS/ICML review standards.

    Reviews 4 stages: hypothesis, experiment design, results, paper
    Returns structured ReviewResult with verdict, strengths, weaknesses, scores
    """

    def __init__(self, llm_client: GroqClient):
        """
        Initialize Alex reviewer.

        Args:
            llm_client: GroqClient instance for LLM backend
        """
        self.llm = llm_client
        self.rate_limit_delay = 2.0  # seconds between Groq API calls
        self.review_patterns = self._load_review_patterns()

        # Map stage names to prompts
        self.stage_prompts = {
            "hypothesis": knowledge.HYPOTHESIS_REVIEW_PROMPT,
            "experiment_design": knowledge.EXPERIMENT_DESIGN_REVIEW_PROMPT,
            "results": knowledge.RESULTS_REVIEW_PROMPT,
            "paper": knowledge.PAPER_REVIEW_PROMPT,
        }

    def _load_review_patterns(self) -> dict:
        """Load review patterns from patterns.json for heuristic checks."""
        try:
            patterns_path = Path(__file__).parent.parent / "filters" / "patterns.json"
            with open(patterns_path) as f:
                data = json.load(f)
                return data.get("review_patterns", {})
        except Exception as e:
            logger.warning(f"Failed to load review patterns: {e}")
            return {}

    def _check_heuristic_patterns(self, artifact: str) -> list[str]:
        """
        Check artifact text for heuristic pattern indicators.

        Args:
            artifact: Text to analyze

        Returns:
            List of pattern names found (e.g., ["MISSING_BASELINE", "OVERCLAIMING"])
        """
        found_patterns = []
        artifact_lower = artifact.lower()

        for pattern_name, pattern_info in self.review_patterns.items():
            indicators = pattern_info.get("indicators", [])
            for indicator in indicators:
                if indicator.lower() in artifact_lower:
                    found_patterns.append(pattern_name)
                    break

        return found_patterns

    async def review(
        self,
        stage: str,
        artifact: str,
        conversation_history: Optional[list[ReviewTurn]] = None,
        hypothesis: Optional[str] = None,
        experiment_design: Optional[str] = None,
        results: Optional[str] = None,
    ) -> ReviewResult:
        """
        Core review method for any stage.

        Args:
            stage: Pipeline stage (hypothesis, experiment_design, results, paper)
            artifact: The artifact to review (text or JSON string)
            conversation_history: Previous turns in the review conversation
            hypothesis: Hypothesis text for context (optional)
            experiment_design: Experiment design for context (optional)
            results: Results for context (optional)

        Returns:
            ReviewResult with verdict, scores, feedback
        """
        conversation_history = conversation_history or []

        # 1. Select stage-specific prompt
        if stage not in self.stage_prompts:
            raise ValueError(
                f"Unknown stage '{stage}'. Must be one of: {list(self.stage_prompts.keys())}"
            )

        prompt_template = self.stage_prompts[stage]

        # 2. Check for heuristic patterns
        heuristic_issues = self._check_heuristic_patterns(artifact)
        heuristic_context = ""
        if heuristic_issues:
            heuristic_context = (
                f"\n\nHeuristic analysis found potential issues: {', '.join(heuristic_issues)}"
            )

        # 3. Format conversation history
        previous_reviews = self._format_conversation_history(conversation_history)

        # 4. Fill in prompt template placeholders
        full_prompt = prompt_template
        full_prompt = full_prompt.replace("{hypothesis}", hypothesis or artifact)
        full_prompt = full_prompt.replace("{previous_reviews}", previous_reviews)
        full_prompt = full_prompt.replace("{experiment_design}", experiment_design or "")
        full_prompt = full_prompt.replace("{results}", results or "")
        full_prompt = full_prompt.replace("{paper}", artifact)
        full_prompt = full_prompt + heuristic_context

        # 5. Call LLM
        try:
            logger.info(f"Alex reviewing {stage}, turn {len(conversation_history) + 1}")
            response = await self.llm.generate(
                prompt=full_prompt,
                max_tokens=2000,
                temperature=0.3,
            )

            # 4. Parse JSON with retry
            try:
                data = await self._parse_review_json(response)
            except Exception as e:
                # Retry once with explicit instruction
                logger.warning(
                    f"Initial JSON parse failed: {e}. Retrying with explicit instruction."
                )
                retry_prompt = f"""{full_prompt}

IMPORTANT: Your previous response could not be parsed as JSON. Please return ONLY valid JSON in this exact format:
{{
  "verdict": "PASS" | "REVISE" | "RESTART_STAGE" | "RESTART_PIPELINE",
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "questions": ["question 1"],
  "suggestions": ["suggestion 1"],
  "scores": {{"score_name": 0.8}},
  "requested_experiments": null
}}"""

                retry_response = await self.llm.generate(
                    prompt=retry_prompt,
                    max_tokens=2000,
                    temperature=0.7,
                )

                # Use retry_prompt parameter to trigger fallback if still fails
                data = await self._parse_review_json(retry_response, retry_prompt="retry")

            # 5. Convert JSON to ReviewResult
            result = ReviewResult(
                review_id=f"review_{stage}_{datetime.now().timestamp()}",
                hypothesis_id=None,  # Will be set by orchestrator
                stage=stage,
                turn_number=len(conversation_history) + 1,
                verdict=data["verdict"],
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                questions=data.get("questions", []),
                suggestions=data.get("suggestions", []),
                scores=data.get("scores", {}),
                requested_experiments=data.get("requested_experiments"),
                timestamp=datetime.now().isoformat(),
            )

            # 6. Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

            logger.info(f"Alex verdict: {result.verdict} (scores: {result.scores})")

            # 7. Return
            return result

        except Exception as e:
            logger.error(f"Alex review failed: {e}")
            raise

    async def _parse_review_json(self, response: str, retry_prompt: str = None) -> dict:
        """
        Parse JSON response from LLM with robust error handling.

        Args:
            response: Raw LLM response
            retry_prompt: If set, use fallback instead of raising error

        Returns:
            Parsed JSON dictionary
        """
        try:
            json_str = response.strip()

            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                parts = json_str.split("```")
                for part in parts[1::2]:
                    stripped = part.strip()
                    if stripped.startswith("{"):
                        json_str = stripped
                        break

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                import re

                first_brace = response.find("{")
                last_brace = response.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = response[first_brace : last_brace + 1]
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        json_str = re.sub(r"[\n\r\t]+", " ", json_str)
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError:
                            in_string = False
                            escaped = False
                            fixed = []
                            for i, char in enumerate(json_str):
                                if escaped:
                                    fixed.append(char)
                                    escaped = False
                                elif char == "\\":
                                    fixed.append(char)
                                    escaped = True
                                elif char == '"':
                                    in_string = not in_string
                                    fixed.append(char)
                                elif char in "\n\r\t" and in_string:
                                    fixed.append(" ")
                                else:
                                    fixed.append(char)
                            json_str = "".join(fixed)
                            data = json.loads(json_str)
                else:
                    raise

            # Validate required fields
            required = ["verdict", "strengths", "weaknesses", "scores"]
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            # Validate verdict
            valid_verdicts = {"PASS", "REVISE", "RESTART_STAGE", "RESTART_PIPELINE"}
            if data["verdict"] not in valid_verdicts:
                raise ValueError(
                    f"Invalid verdict '{data['verdict']}'. Must be one of: {valid_verdicts}"
                )

            return data

        except Exception as e:
            if retry_prompt:
                # Already retried, use fallback
                logger.error(f"Review parsing failed after retry: {e}")
                return {
                    "verdict": "REVISE",
                    "strengths": [],
                    "weaknesses": [f"Review parsing failed: {str(e)}"],
                    "questions": [],
                    "suggestions": ["Please provide more structured feedback"],
                    "scores": {"overall": 0.5},
                    "requested_experiments": None,
                }
            else:
                # Let caller retry
                raise

    def _format_conversation_history(self, turns: list[ReviewTurn]) -> str:
        """
        Format conversation history for prompt.

        Args:
            turns: List of ReviewTurn objects

        Returns:
            Formatted history string
        """
        if not turns:
            return "No previous conversation."

        formatted = []
        for turn in turns:
            speaker = "Fleming" if turn.speaker == "fleming" else "Alex"
            # Truncate long content to keep prompt manageable
            content_preview = turn.content[:500]
            if len(turn.content) > 500:
                content_preview += "..."
            formatted.append(
                f"[Turn {getattr(turn, 'turn_number', '?')} - {speaker}]: {content_preview}"
            )

        return "\n".join(formatted)

    # Convenience methods for specific stages

    async def review_hypothesis(
        self,
        hypothesis_text: str,
        connection: Optional[dict] = None,
        conversation_history: Optional[list[ReviewTurn]] = None,
    ) -> ReviewResult:
        """
        Review a hypothesis.

        Args:
            hypothesis_text: The hypothesis text to review
            connection: Optional connection metadata (not used in review)
            conversation_history: Previous review turns (optional)

        Returns:
            ReviewResult
        """
        return await self.review(
            stage="hypothesis",
            artifact=hypothesis_text,
            conversation_history=conversation_history,
        )

    async def review_experiment_design(
        self,
        design: dict,
        hypothesis: Optional[str] = None,
        conversation_history: Optional[list[ReviewTurn]] = None,
    ) -> ReviewResult:
        """
        Review an experiment design.

        Args:
            design: Design dictionary with models, datasets, baselines, etc.
            hypothesis: Associated hypothesis (optional)
            conversation_history: Previous review turns (optional)

        Returns:
            ReviewResult
        """
        return await self.review(
            stage="experiment_design",
            artifact=json.dumps(design, indent=2),
            conversation_history=conversation_history,
            hypothesis=hypothesis,
        )

    async def review_results(
        self,
        analysis: dict,
        hypothesis: Optional[str] = None,
        experiment_design: Optional[str] = None,
        conversation_history: Optional[list[ReviewTurn]] = None,
    ) -> ReviewResult:
        """
        Review experimental results.

        Args:
            analysis: Results dictionary with metrics, statistical tests, etc.
            hypothesis: Associated hypothesis (optional)
            experiment_design: Associated experiment design (optional)
            conversation_history: Previous review turns (optional)

        Returns:
            ReviewResult
        """
        return await self.review(
            stage="results",
            artifact=json.dumps(analysis, indent=2),
            conversation_history=conversation_history,
            hypothesis=hypothesis,
            experiment_design=experiment_design,
        )

    async def review_paper(
        self,
        paper_text: str,
        claims: Optional[list] = None,
        hypothesis: Optional[str] = None,
        experiment_design: Optional[str] = None,
        results: Optional[str] = None,
        conversation_history: Optional[list[ReviewTurn]] = None,
    ) -> ReviewResult:
        """
        Review a complete paper draft.

        Args:
            paper_text: Full paper text
            claims: Optional list of claims to verify (not used in review)
            hypothesis: Associated hypothesis (optional)
            experiment_design: Associated experiment design (optional)
            results: Associated results (optional)
            conversation_history: Previous review turns (optional)

        Returns:
            ReviewResult
        """
        return await self.review(
            stage="paper",
            artifact=paper_text,
            conversation_history=conversation_history,
            hypothesis=hypothesis,
            experiment_design=experiment_design,
            results=results,
        )
