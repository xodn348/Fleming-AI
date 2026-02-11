"""
Orchestrator Module for Fleming-AI
Orchestrates full 4-stage pipeline with escalation handling and conversation logging
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any

from src.llm.groq_client import GroqClient
from src.reviewers.alex import Alex
from src.reviewers.conversation import ConversationManager
from src.pipeline.review_gates import HypothesisGate, ExperimentDesignGate, ResultsGate, PaperGate

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of full Fleming-Alex pipeline."""

    final_hypothesis: Optional[str] = None
    final_design: Optional[dict] = None
    final_results: Optional[dict] = None
    final_paper: Optional[str] = None

    total_turns: int = 0
    total_llm_calls: int = 0

    gate_results: dict[str, str] = field(default_factory=dict)
    conversation_logs: dict[str, list] = field(default_factory=dict)

    escalations: list[str] = field(default_factory=list)
    pipeline_restarts: int = 0

    run_id: str = ""
    start_time: str = ""
    end_time: str = ""


class FlemingAlexOrchestrator:
    """Orchestrates full 4-stage Fleming-Alex review pipeline."""

    def __init__(self, groq_client: GroqClient):
        """Initialize orchestrator with LLM client."""
        self.llm = groq_client
        self.alex = Alex(groq_client)
        self.max_stage_retries = 1
        self.max_pipeline_restarts = 1

    async def run_full_pipeline(self, initial_hypothesis: str) -> PipelineResult:
        """Run full 4-stage pipeline: hypothesis → design → results → paper."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = PipelineResult(run_id=run_id, start_time=datetime.now().isoformat())

        pipeline_attempts = 0

        while pipeline_attempts <= self.max_pipeline_restarts:
            try:
                hypothesis, hypothesis_log = await self._run_stage(
                    "hypothesis", initial_hypothesis, HypothesisGate
                )
                result.final_hypothesis = hypothesis
                result.conversation_logs["hypothesis"] = hypothesis_log

                mock_design = {"datasets": ["ImageNet"], "models": ["ViT", "ResNet"]}
                design, design_log = await self._run_stage(
                    "experiment_design", mock_design, ExperimentDesignGate
                )
                result.final_design = design
                result.conversation_logs["experiment_design"] = design_log

                mock_results = {"accuracy": {"ViT": 0.85, "ResNet": 0.82}}
                results, results_log = await self._run_stage("results", mock_results, ResultsGate)
                result.final_results = results
                result.conversation_logs["results"] = results_log

                mock_paper = f"Abstract: {hypothesis}. Methods: {design}. Results: {results}."
                paper, paper_log = await self._run_stage("paper", mock_paper, PaperGate)
                result.final_paper = paper
                result.conversation_logs["paper"] = paper_log

                break

            except PipelineRestartRequired:
                pipeline_attempts += 1
                result.pipeline_restarts += 1
                logger.warning(f"Pipeline restart #{pipeline_attempts}")
                if pipeline_attempts > self.max_pipeline_restarts:
                    logger.error("Max pipeline restarts reached")
                    raise

        result.end_time = datetime.now().isoformat()
        result.total_turns = sum(len(log) for log in result.conversation_logs.values())

        self._save_conversation_log(result)

        return result

    async def _run_stage(
        self, stage_name: str, artifact: Any, gate_class: type
    ) -> tuple[Any, list]:
        stage_attempts = 0

        while stage_attempts <= self.max_stage_retries:
            conversation = ConversationManager(max_turns=6)
            gate = gate_class(self.alex, conversation, self.llm)

            if isinstance(artifact, dict):
                artifact_str = json.dumps(artifact, indent=2)
            else:
                artifact_str = artifact

            gate_result = await gate.run_gate(artifact_str)

            if gate_result.escalation_reason == "RESTART_STAGE":
                stage_attempts += 1
                logger.warning(f"{stage_name}: RESTART_STAGE (attempt #{stage_attempts})")
                if stage_attempts > self.max_stage_retries:
                    logger.error(f"{stage_name}: Max stage retries reached")
                    break
                continue

            elif gate_result.escalation_reason == "RESTART_PIPELINE":
                logger.warning(f"{stage_name}: RESTART_PIPELINE requested")
                raise PipelineRestartRequired()

            final_artifact = gate_result.final_artifact
            if stage_name in ["experiment_design", "results"]:
                try:
                    final_artifact = json.loads(final_artifact)
                except Exception:
                    pass

            return final_artifact, conversation.state.turns

        final_artifact = gate_result.final_artifact
        if stage_name in ["experiment_design", "results"]:
            try:
                final_artifact = json.loads(final_artifact)
            except Exception:
                pass

        return final_artifact, conversation.state.turns

    async def run_hypothesis_review(self, hypothesis: str) -> PipelineResult:
        """Run only hypothesis review stage (for src/ pipeline integration)."""
        run_id = f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = PipelineResult(run_id=run_id, start_time=datetime.now().isoformat())

        hypothesis, log = await self._run_stage("hypothesis", hypothesis, HypothesisGate)
        result.final_hypothesis = hypothesis
        result.conversation_logs["hypothesis"] = log
        result.end_time = datetime.now().isoformat()

        self._save_conversation_log(result)
        return result

    async def run_paper_review(
        self, hypothesis: str, design: dict, results: dict, draft: str
    ) -> PipelineResult:
        """Run only paper review stage (for experiment/ pipeline integration)."""
        run_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = PipelineResult(run_id=run_id, start_time=datetime.now().isoformat())

        paper, log = await self._run_stage("paper", draft, PaperGate)
        result.final_paper = paper
        result.conversation_logs["paper"] = log
        result.end_time = datetime.now().isoformat()

        self._save_conversation_log(result)
        return result

    def _save_conversation_log(self, result: PipelineResult):
        log_dir = Path(f"runs/{result.run_id}")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "conversation.json"

        serializable_logs = {}
        for stage, turns in result.conversation_logs.items():
            serializable_logs[stage] = [
                {
                    "turn_id": t.turn_id,
                    "speaker": t.speaker,
                    "content": t.content,
                    "structured_data": t.structured_data,
                    "timestamp": t.timestamp,
                }
                for t in turns
            ]

        log_data = {
            "run_id": result.run_id,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "total_turns": result.total_turns,
            "pipeline_restarts": result.pipeline_restarts,
            "final_hypothesis": result.final_hypothesis,
            "conversation_logs": serializable_logs,
        }

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved conversation log to {log_file}")


class PipelineRestartRequired(Exception):
    """Raised when RESTART_PIPELINE escalation is triggered."""

    pass
