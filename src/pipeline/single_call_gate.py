"""
Single-Call Review Gate for Fleming-AI
Reduces API calls by 75% (from ~36 to 4 calls per paper) using consolidated review.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from src.reviewers import knowledge
from src.reviewers.schemas import ReviewResult


class SingleCallReviewer:
    """
    Single-call reviewer that consolidates all review criteria into one LLM call.

    Reduces API overhead from multi-turn conversations (2-6 turns × 2 calls/turn)
    to a single comprehensive review per stage.

    Usage:
        reviewer = SingleCallReviewer(llm_client)
        result = await reviewer.review(artifact_text, "hypothesis")
    """

    def __init__(self, llm_client):
        """
        Initialize single-call reviewer with LLM client.

        Args:
            llm_client: LLM client for generating reviews
        """
        self.llm = llm_client
        self.stage_prompts = {
            "hypothesis": knowledge.HYPOTHESIS_REVIEW_PROMPT,
            "experiment_design": knowledge.EXPERIMENT_DESIGN_REVIEW_PROMPT,
            "results": knowledge.RESULTS_REVIEW_PROMPT,
            "paper": knowledge.PAPER_REVIEW_PROMPT,
        }

    async def review(
        self, artifact: str, stage: str, hypothesis_id: Optional[str] = None
    ) -> ReviewResult:
        """
        Perform single-call review of artifact for given stage.

        Args:
            artifact: Text to review (hypothesis, experiment design, results, or paper)
            stage: Pipeline stage ("hypothesis", "experiment_design", "results", "paper")
            hypothesis_id: Optional hypothesis ID for tracking

        Returns:
            ReviewResult with verdict, scores, and feedback

        Raises:
            ValueError: If stage is invalid or parsing fails
        """
        if stage not in self.stage_prompts:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of: {list(self.stage_prompts.keys())}"
            )

        prompt = self.stage_prompts[stage]
        full_prompt = f"{prompt}\n\nARTIFACT TO REVIEW:\n{artifact}"

        response = await self.llm.generate(full_prompt)

        try:
            review_data = self._parse_review_json(response)
        except Exception as e:
            retry_prompt = (
                f"{full_prompt}\n\n"
                "IMPORTANT: Your previous response had a parsing error. "
                "Please respond with ONLY valid JSON matching the exact format specified."
            )
            retry_response = await self.llm.generate(retry_prompt)
            review_data = self._parse_review_json(retry_response, retry_prompt=True)

        review_id = f"review_{uuid.uuid4().hex[:8]}"
        result = ReviewResult(
            review_id=review_id,
            hypothesis_id=hypothesis_id,
            stage=stage,
            turn_number=1,  # Single-call mode always uses turn 1
            verdict=review_data["verdict"],
            strengths=review_data.get("strengths", []),
            weaknesses=review_data.get("weaknesses", []),
            questions=review_data.get("questions", []),
            suggestions=review_data.get("suggestions", []),
            scores=review_data.get("scores", {}),
            requested_experiments=review_data.get("requested_experiments"),
            timestamp=datetime.now().isoformat(),
        )

        return result

    async def review_paper_full(
        self, paper_text: str, hypothesis_id: Optional[str] = None
    ) -> ReviewResult:
        """
        Convenience method for full paper review using consolidated prompt.

        Args:
            paper_text: Complete paper draft to review
            hypothesis_id: Optional hypothesis ID for tracking

        Returns:
            ReviewResult with comprehensive evaluation across all 9 criteria
        """
        full_prompt = f"{knowledge.CONSOLIDATED_REVIEW_PROMPT}\n\nPAPER TO REVIEW:\n{paper_text}"

        response = await self.llm.generate(full_prompt)

        try:
            review_data = self._parse_review_json(response)
        except Exception as e:
            retry_prompt = (
                f"{full_prompt}\n\n"
                "IMPORTANT: Your previous response had a parsing error. "
                "Please respond with ONLY valid JSON matching the exact format specified."
            )
            retry_response = await self.llm.generate(retry_prompt)
            review_data = self._parse_review_json(retry_response, retry_prompt=True)

        review_id = f"review_{uuid.uuid4().hex[:8]}"
        result = ReviewResult(
            review_id=review_id,
            hypothesis_id=hypothesis_id,
            stage="paper",
            turn_number=1,
            verdict=review_data["verdict"],
            strengths=review_data.get("strengths", []),
            weaknesses=review_data.get("weaknesses", []),
            questions=review_data.get("questions", []),
            suggestions=review_data.get("suggestions", []),
            scores=review_data.get("scores", {}),
            requested_experiments=review_data.get("requested_experiments"),
            timestamp=datetime.now().isoformat(),
        )

        return result

    def _parse_review_json(self, response: str, retry_prompt: str = None) -> dict:
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

            required = ["verdict", "strengths", "weaknesses", "scores"]
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            valid_verdicts = {"PASS", "REVISE", "RESTART_STAGE", "RESTART_PIPELINE"}
            if data["verdict"] not in valid_verdicts:
                raise ValueError(
                    f"Invalid verdict '{data['verdict']}'. Must be one of: {valid_verdicts}"
                )

            return data

        except Exception as e:
            if retry_prompt:
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
                raise
