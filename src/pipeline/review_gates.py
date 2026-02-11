"""
Review Gates Module for Fleming-AI
Implements 4-stage review loop orchestration with Alex reviewer and Fleming revisions
"""

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.reviewers.alex import Alex
from src.reviewers.conversation import ConversationManager
from src.reviewers.schemas import ReviewResult
from src.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of passing through a review gate."""

    final_artifact: str
    final_review: ReviewResult
    num_turns: int
    converged: bool
    escalation_reason: Optional[str] = None


# Fleming Revision Functions
# ============================================================================


async def revise_hypothesis(llm: GroqClient, original: str, feedback: ReviewResult) -> str:
    """Revise hypothesis based on Alex's feedback.

    Args:
        llm: LLM client
        original: Original hypothesis text
        feedback: Alex's review with suggestions

    Returns:
        Revised hypothesis text
    """
    weaknesses_str = "\n".join(f"- {w}" for w in feedback.weaknesses)
    suggestions_str = "\n".join(f"- {s}" for s in feedback.suggestions)

    prompt = f"""You are Fleming, an ML researcher. Revise your hypothesis based on reviewer feedback.

ORIGINAL HYPOTHESIS:
{original}

REVIEWER FEEDBACK (Weaknesses):
{weaknesses_str}

REVIEWER SUGGESTIONS:
{suggestions_str}

Instructions:
1. Address ALL weaknesses mentioned
2. Incorporate suggestions where applicable
3. Keep the hypothesis concise (1-2 sentences)
4. Make it more specific, testable, and clear

Output ONLY the revised hypothesis text, no preamble."""

    logger.info("Fleming revising hypothesis")
    revised = await llm.generate(prompt, max_tokens=800, temperature=0.7)

    revised = revised.strip()
    if revised.startswith('"') and revised.endswith('"'):
        revised = revised[1:-1]

    # Validate completeness
    if revised and revised[-1] not in ".!?":
        for i in range(len(revised) - 1, -1, -1):
            if revised[i] in ".!?":
                revised = revised[: i + 1]
                break
        else:
            revised = revised.rstrip(",;: ") + "."

    await asyncio.sleep(2.0)

    return revised


async def revise_experiment_design(llm: GroqClient, original: dict, feedback: ReviewResult) -> dict:
    """Revise experiment design based on feedback.

    Args:
        llm: LLM client
        original: Original design dict
        feedback: Alex's review

    Returns:
        Revised design dict
    """
    weaknesses_str = "\n".join(f"- {w}" for w in feedback.weaknesses)
    suggestions_str = "\n".join(f"- {s}" for s in feedback.suggestions)

    prompt = f"""You are Fleming. Revise your experiment design based on reviewer feedback.

ORIGINAL DESIGN:
{json.dumps(original, indent=2)}

WEAKNESSES:
{weaknesses_str}

SUGGESTIONS:
{suggestions_str}

Instructions:
1. Address weaknesses (add baselines, diversify datasets, increase repeatability, add ablations)
2. Keep JSON structure compatible with original
3. Add missing elements, don't remove existing ones

Output ONLY the revised design as valid JSON."""

    logger.info("Fleming revising experiment design")
    revised_json = await llm.generate(prompt, max_tokens=2000, temperature=0.7)

    try:
        if "```json" in revised_json:
            revised_json = revised_json.split("```json")[1].split("```")[0]
        elif "```" in revised_json:
            revised_json = revised_json.split("```")[1].split("```")[0]

        revised_json = revised_json.strip()
        # Validate completeness
        if revised_json and revised_json[-1] not in ".!?}":
            for i in range(len(revised_json) - 1, -1, -1):
                if revised_json[i] in ".!?}":
                    revised_json = revised_json[: i + 1]
                    break
            else:
                revised_json = revised_json.rstrip(",;: ") + "}"

        revised = json.loads(revised_json)
    except Exception as e:
        logger.warning(f"Failed to parse revised design JSON: {e}. Using fallback.")
        revised = original.copy()
        revised["reviewer_suggestions"] = feedback.suggestions

    await asyncio.sleep(2.0)

    return revised


async def revise_results(llm: GroqClient, original: dict, feedback: ReviewResult) -> dict:
    """Revise analysis results based on feedback.

    Args:
        llm: LLM client
        original: Original results dict
        feedback: Alex's review

    Returns:
        Revised results dict with additional analysis
    """
    weaknesses_str = "\n".join(f"- {w}" for w in feedback.weaknesses)

    prompt = f"""You are Fleming. Revise your results analysis based on reviewer feedback.

ORIGINAL RESULTS:
{json.dumps(original, indent=2)[:2000]}

WEAKNESSES TO ADDRESS:
{weaknesses_str}

REQUESTED EXPERIMENTS:
{json.dumps(feedback.requested_experiments, indent=2) if feedback.requested_experiments else "None"}

Instructions:
1. Add claim-evidence alignment notes
2. Report effect sizes explicitly
3. Mention negative results if any
4. Add statistical rigor notes
5. Keep original data, add "revised_analysis" section

Output as JSON."""

    logger.info("Fleming revising results analysis")
    revised_json = await llm.generate(prompt, max_tokens=2500, temperature=0.7)

    try:
        if "```json" in revised_json:
            revised_json = revised_json.split("```json")[1].split("```")[0]
        elif "```" in revised_json:
            revised_json = revised_json.split("```")[1].split("```")[0]

        revised_json = revised_json.strip()
        # Validate completeness
        if revised_json and revised_json[-1] not in ".!?}":
            for i in range(len(revised_json) - 1, -1, -1):
                if revised_json[i] in ".!?}":
                    revised_json = revised_json[: i + 1]
                    break
            else:
                revised_json = revised_json.rstrip(",;: ") + "}"

        revised = json.loads(revised_json)
    except Exception as e:
        logger.warning(f"Failed to parse revised results JSON: {e}. Using fallback.")
        revised = original.copy()
        revised["alex_feedback"] = weaknesses_str

    await asyncio.sleep(2.0)

    return revised


async def revise_paper(llm: GroqClient, original: str, feedback: ReviewResult) -> str:
    """Revise paper draft based on feedback.

    Args:
        llm: LLM client
        original: Original paper text (LaTeX or markdown)
        feedback: Alex's review

    Returns:
        Revised paper text
    """
    weaknesses_str = "\n".join(f"- {w}" for w in feedback.weaknesses)
    suggestions_str = "\n".join(f"- {s}" for s in feedback.suggestions)

    original_preview = original[:3000]

    prompt = f"""You are Fleming. Revise your paper draft based on reviewer feedback.

ORIGINAL PAPER (preview):
{original_preview}

WEAKNESSES:
{weaknesses_str}

SUGGESTIONS:
{suggestions_str}

Instructions:
1. Fix overclaiming → make claims more conservative
2. Add Limitations section if missing
3. Improve Related Work if weak
4. Enhance Structure if needed
5. Add reproducibility details

Output the revised version of the ENTIRE paper (or specific sections that need changes)."""

    logger.info("Fleming revising paper draft")
    revised = await llm.generate(prompt, max_tokens=4096, temperature=0.7)

    revised = revised.strip()
    if revised.startswith('"') and revised.endswith('"'):
        revised = revised[1:-1]

    # Validate completeness
    if revised and revised[-1] not in ".!?":
        for i in range(len(revised) - 1, -1, -1):
            if revised[i] in ".!?":
                revised = revised[: i + 1]
                break
        else:
            revised = revised.rstrip(",;: ") + "."

    await asyncio.sleep(2.0)

    return revised


class ReviewGate(ABC):
    """Base class for review gates."""

    def __init__(self, alex: Alex, conversation: ConversationManager, llm: GroqClient):
        """Initialize review gate.

        Args:
            alex: Alex reviewer instance
            conversation: Conversation manager for tracking dialogue
            llm: LLM client for Fleming's revisions
        """
        self.alex = alex
        self.conversation = conversation
        self.llm = llm
        self.stage = self._get_stage_name()
        self.conversation.state.stage = self.stage

    @abstractmethod
    def _get_stage_name(self) -> str:
        """Return stage name (hypothesis, experiment_design, results, paper)."""
        pass

    @abstractmethod
    async def _review_artifact(self, artifact: str) -> ReviewResult:
        """Call appropriate Alex review method for this stage."""
        pass

    @abstractmethod
    async def _revise_artifact(self, artifact: str, feedback: ReviewResult) -> str:
        """Call Fleming revision function for this stage."""
        pass

    async def run_gate(self, initial_artifact: str) -> GateResult:
        """Run review loop: Alex reviews → Fleming revises → ... → converge.

        Args:
            initial_artifact: Initial version to review

        Returns:
            GateResult with final artifact and review metadata
        """
        current_artifact = initial_artifact
        logger.info(f"Starting {self.stage} gate with artifact length: {len(current_artifact)}")

        while not self.conversation.is_converged():
            logger.info(
                f"{self.stage} gate: Alex reviewing (turn {len(self.conversation.state.turns) + 1})"
            )
            review = await self._review_artifact(current_artifact)

            self.conversation.add_turn(
                speaker="alex",
                content=f"{review.verdict}. Strengths: {len(review.strengths)}. Weaknesses: {len(review.weaknesses)}.",
                structured=review.__dict__,
            )

            logger.info(f"Alex verdict: {review.verdict}, score: {review.scores}")

            if self.conversation.is_converged():
                escalation = self.conversation.should_escalate()
                logger.info(f"{self.stage} gate converged. Escalation: {escalation}")
                return GateResult(
                    final_artifact=current_artifact,
                    final_review=review,
                    num_turns=len(self.conversation.state.turns),
                    converged=True,
                    escalation_reason=escalation,
                )

            logger.info(f"{self.stage} gate: Fleming revising")
            revised_artifact = await self._revise_artifact(current_artifact, review)

            self.conversation.add_turn(
                speaker="fleming",
                content=f"Revised artifact (length: {len(revised_artifact)} chars)",
                structured={"artifact": revised_artifact[:500]},
            )

            current_artifact = revised_artifact

        logger.info(f"{self.stage} gate: max turns reached, getting final review")
        final_review = await self._review_artifact(current_artifact)
        escalation = self.conversation.should_escalate()

        return GateResult(
            final_artifact=current_artifact,
            final_review=final_review,
            num_turns=len(self.conversation.state.turns),
            converged=True,
            escalation_reason=escalation,
        )


class HypothesisGate(ReviewGate):
    """Review gate for hypothesis stage."""

    def _get_stage_name(self) -> str:
        return "hypothesis"

    async def _review_artifact(self, artifact: str) -> ReviewResult:
        return await self.alex.review(
            stage="hypothesis",
            artifact=artifact,
            conversation_history=self.conversation.state.turns,
        )

    async def _revise_artifact(self, artifact: str, feedback: ReviewResult) -> str:
        return await revise_hypothesis(self.llm, artifact, feedback)


class ExperimentDesignGate(ReviewGate):
    """Review gate for experiment design stage."""

    def _get_stage_name(self) -> str:
        return "experiment_design"

    async def _review_artifact(self, artifact: str) -> ReviewResult:
        design = json.loads(artifact) if isinstance(artifact, str) else artifact
        artifact_str = json.dumps(design, indent=2) if not isinstance(artifact, str) else artifact

        return await self.alex.review(
            stage="experiment_design",
            artifact=artifact_str,
            conversation_history=self.conversation.state.turns,
        )

    async def _revise_artifact(self, artifact: str, feedback: ReviewResult) -> str:
        design = json.loads(artifact) if isinstance(artifact, str) else artifact
        revised = await revise_experiment_design(self.llm, design, feedback)
        return json.dumps(revised, indent=2)


class ResultsGate(ReviewGate):
    """Review gate for results analysis stage."""

    def _get_stage_name(self) -> str:
        return "results"

    async def _review_artifact(self, artifact: str) -> ReviewResult:
        analysis = json.loads(artifact) if isinstance(artifact, str) else artifact
        artifact_str = json.dumps(analysis, indent=2) if not isinstance(artifact, str) else artifact

        return await self.alex.review(
            stage="results",
            artifact=artifact_str,
            conversation_history=self.conversation.state.turns,
        )

    async def _revise_artifact(self, artifact: str, feedback: ReviewResult) -> str:
        analysis = json.loads(artifact) if isinstance(artifact, str) else artifact
        revised = await revise_results(self.llm, analysis, feedback)
        return json.dumps(revised, indent=2)


class PaperGate(ReviewGate):
    """Review gate for paper draft stage."""

    def _get_stage_name(self) -> str:
        return "paper"

    async def _review_artifact(self, artifact: str) -> ReviewResult:
        return await self.alex.review(
            stage="paper", artifact=artifact, conversation_history=self.conversation.state.turns
        )

    async def _revise_artifact(self, artifact: str, feedback: ReviewResult) -> str:
        return await revise_paper(self.llm, artifact, feedback)
