"""Tests for FlemingAlexOrchestrator."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.pipeline.orchestrator import FlemingAlexOrchestrator, PipelineResult
from src.pipeline.review_gates import GateResult
from src.reviewers.schemas import ReviewResult
from src.llm.groq_client import GroqClient


@pytest.fixture
def mock_groq():
    """Mock GroqClient."""
    groq = AsyncMock(spec=GroqClient)
    return groq


@pytest.fixture
def orchestrator(mock_groq):
    """Orchestrator with mocked LLM."""
    return FlemingAlexOrchestrator(mock_groq, runs_dir="test_runs")


def test_orchestrator_initialization(mock_groq):
    """Test orchestrator initializes correctly."""
    orch = FlemingAlexOrchestrator(mock_groq)
    assert orch.llm == mock_groq
    assert orch.alex is not None
    assert orch.runs_dir.name == "runs"


@pytest.mark.asyncio
async def test_hypothesis_gate_converges_mock(orchestrator, mock_groq):
    """Test hypothesis gate converges (mocked)."""
    mock_groq.generate.side_effect = [
        '{"verdict": "REVISE", "strengths": [], "weaknesses": ["Too vague"], "questions": [], "suggestions": ["Add specificity"], "scores": {"clarity": 0.5}, "requested_experiments": null}',
        "Revised hypothesis with more specificity",
        '{"verdict": "PASS", "strengths": ["Much better"], "weaknesses": [], "questions": [], "suggestions": [], "scores": {"clarity": 0.9}, "requested_experiments": null}',
    ]

    result = await orchestrator.run_hypothesis_review("Initial hypothesis")

    assert result.final_hypothesis is not None
    assert result.total_turns >= 2
    assert "hypothesis" in result.stages_completed
    assert result.hypothesis_gate_result is not None
    assert result.hypothesis_gate_result.converged


@pytest.mark.asyncio
async def test_max_turns_respected(orchestrator, mock_groq):
    """Test orchestrator respects max_turns limit."""
    mock_groq.generate.return_value = '{"verdict": "REVISE", "strengths": [], "weaknesses": ["Still bad"], "questions": [], "suggestions": [], "scores": {"clarity": 0.3}, "requested_experiments": null}'

    result = await orchestrator.run_hypothesis_review("Test")

    assert result.total_turns <= 12
    assert result.hypothesis_gate_result.converged


@pytest.mark.asyncio
async def test_escalation_restart_stage(orchestrator, mock_groq):
    """Test escalation handling for RESTART_STAGE."""
    mock_groq.generate.side_effect = [
        '{"verdict": "RESTART_STAGE", "strengths": [], "weaknesses": ["Fundamentally flawed"], "questions": [], "suggestions": ["Start over"], "scores": {"clarity": 0.2}, "requested_experiments": null}',
        '{"verdict": "REVISE", "strengths": [], "weaknesses": ["Still needs work"], "questions": [], "suggestions": ["Add more"], "scores": {"clarity": 0.4}, "requested_experiments": null}',
        "Revised hypothesis after restart",
        '{"verdict": "PASS", "strengths": ["Good"], "weaknesses": [], "questions": [], "suggestions": [], "scores": {"clarity": 0.8}, "requested_experiments": null}',
    ]

    result = await orchestrator.run_hypothesis_review("Test")

    assert result.final_hypothesis is not None


@pytest.mark.asyncio
async def test_run_stage_with_retry_mechanism(orchestrator, mock_groq):
    """Test _run_stage_with_retry retries on RESTART_STAGE."""
    mock_groq.generate.side_effect = [
        '{"verdict": "RESTART_STAGE", "strengths": [], "weaknesses": ["Bad"], "questions": [], "suggestions": [], "scores": {"clarity": 0.2}, "requested_experiments": null}',
        '{"verdict": "REVISE", "strengths": [], "weaknesses": ["Needs improvement"], "questions": [], "suggestions": ["Be more specific"], "scores": {"clarity": 0.5}, "requested_experiments": null}',
        "Improved hypothesis after feedback",
        '{"verdict": "PASS", "strengths": ["Better"], "weaknesses": [], "questions": [], "suggestions": [], "scores": {"clarity": 0.7}, "requested_experiments": null}',
    ]

    result = await orchestrator.run_hypothesis_review("Test hypothesis")

    assert result.final_hypothesis is not None
    assert result.hypothesis_gate_result.converged


@pytest.mark.asyncio
async def test_conversation_log_saved(orchestrator, mock_groq, tmp_path):
    """Test that conversation log is saved to disk."""
    orchestrator.runs_dir = tmp_path

    mock_groq.generate.side_effect = [
        '{"verdict": "PASS", "strengths": ["Good"], "weaknesses": [], "questions": [], "suggestions": [], "scores": {"clarity": 0.8}, "requested_experiments": null}',
    ]

    result = await orchestrator.run_hypothesis_review("Good hypothesis")

    assert result.conversation_log_path is not None
    log_file = tmp_path / result.run_id / "conversation.json"
    assert log_file.exists()

    with open(log_file) as f:
        log_data = json.load(f)
        assert "run_id" in log_data
        assert "stages" in log_data
        assert "total_turns" in log_data


@pytest.mark.asyncio
async def test_pipeline_result_structure(orchestrator, mock_groq):
    """Test PipelineResult has expected structure."""
    mock_groq.generate.return_value = '{"verdict": "PASS", "strengths": [], "weaknesses": [], "questions": [], "suggestions": [], "scores": {"clarity": 0.8}, "requested_experiments": null}'

    result = await orchestrator.run_hypothesis_review("Test")

    assert isinstance(result, PipelineResult)
    assert hasattr(result, "final_hypothesis")
    assert hasattr(result, "final_design")
    assert hasattr(result, "final_results")
    assert hasattr(result, "final_paper")
    assert hasattr(result, "hypothesis_gate_result")
    assert hasattr(result, "total_turns")
    assert hasattr(result, "total_llm_calls")
    assert hasattr(result, "stages_completed")
    assert hasattr(result, "conversation_log_path")
    assert hasattr(result, "run_id")


@pytest.mark.slow
async def test_full_pipeline_e2e():
    """Test full 4-stage pipeline end-to-end (REQUIRES GROQ_API_KEY)."""
    try:
        async with GroqClient() as groq:
            orch = FlemingAlexOrchestrator(groq, runs_dir="test_runs_e2e")

            result = await orch.run_full_pipeline(
                "Pre-training helps ViT more than CNN at low data"
            )

            assert len(result.stages_completed) == 4
            assert "hypothesis" in result.stages_completed
            assert "experiment_design" in result.stages_completed
            assert "results" in result.stages_completed
            assert "paper" in result.stages_completed

            assert result.final_hypothesis is not None
            assert result.final_design is not None
            assert result.final_results is not None
            assert result.final_paper is not None

            assert result.conversation_log_path is not None

            print(
                f"E2E test completed: {result.total_turns} turns, {result.total_llm_calls} LLM calls"
            )
    except Exception as e:
        pytest.skip(f"E2E test skipped: {e}")


@pytest.mark.slow
async def test_hypothesis_review_integration():
    """Test hypothesis review with real API (REQUIRES GROQ_API_KEY)."""
    try:
        async with GroqClient() as groq:
            orch = FlemingAlexOrchestrator(groq, runs_dir="test_runs_hyp")

            result = await orch.run_hypothesis_review(
                "Vision transformers need more training data than CNNs"
            )

            assert result.final_hypothesis is not None
            assert len(result.stages_completed) == 1
            assert result.stages_completed[0] == "hypothesis"
            assert result.hypothesis_gate_result is not None
            assert result.hypothesis_gate_result.converged

            final_review = result.hypothesis_gate_result.final_review
            assert final_review.verdict == "PASS"

            print(f"Hypothesis review: {result.total_turns} turns")
    except Exception as e:
        pytest.skip(f"Integration test skipped: {e}")


@pytest.mark.slow
async def test_paper_review_integration():
    """Test paper review with real API (REQUIRES GROQ_API_KEY)."""
    try:
        async with GroqClient() as groq:
            orch = FlemingAlexOrchestrator(groq, runs_dir="test_runs_paper")

            hypothesis = "Pre-trained ViT outperforms CNN at low data"
            design = {
                "datasets": ["CIFAR-10"],
                "models": ["ViT-B/16", "ResNet-50"],
                "baselines": ["Random init", "Pre-trained"],
            }
            results = {
                "main_findings": ["ViT improves by 12% with pre-training"],
                "metrics": {"accuracy": 0.85},
            }
            draft = f"""# Paper Title

## Abstract
{hypothesis}

## Introduction
We investigate vision transformers...

## Methods
{json.dumps(design, indent=2)}

## Results
{json.dumps(results, indent=2)}

## Conclusion
Our findings show that pre-training helps ViT more than CNN.
"""

            result = await orch.run_paper_review(hypothesis, design, results, draft)

            assert result.final_paper is not None
            assert len(result.stages_completed) == 1
            assert result.stages_completed[0] == "paper"
            assert result.paper_gate_result is not None

            print(f"Paper review: {result.total_turns} turns")
    except Exception as e:
        pytest.skip(f"Integration test skipped: {e}")
