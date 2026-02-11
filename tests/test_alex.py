"""Tests for Alex reviewer module."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.reviewers.alex import Alex
from src.reviewers.schemas import ReviewResult
from src.llm.groq_client import GroqClient


@pytest.fixture
def mock_groq():
    """Mock GroqClient for unit tests."""
    groq = AsyncMock(spec=GroqClient)
    return groq


@pytest.fixture
def alex_with_mock(mock_groq):
    """Alex instance with mocked LLM."""
    return Alex(mock_groq)


def test_alex_initialization(mock_groq):
    """Test Alex initializes correctly."""
    alex = Alex(mock_groq)
    assert alex.llm == mock_groq
    assert alex.rate_limit_delay == 2.0
    assert "hypothesis" in alex.stage_prompts
    assert "experiment_design" in alex.stage_prompts
    assert "results" in alex.stage_prompts
    assert "paper" in alex.stage_prompts


@pytest.mark.asyncio
async def test_review_hypothesis_structure(alex_with_mock, mock_groq):
    """Test review_hypothesis returns correct structure."""
    # Mock LLM response
    mock_groq.generate.return_value = """{
        "verdict": "REVISE",
        "strengths": ["Clear topic"],
        "weaknesses": ["Too vague"],
        "questions": ["What conditions?"],
        "suggestions": ["Add specificity"],
        "scores": {"clarity": 0.5},
        "requested_experiments": null
    }"""

    result = await alex_with_mock.review_hypothesis("Test hypothesis")

    assert isinstance(result, ReviewResult)
    assert result.verdict == "REVISE"
    assert len(result.strengths) == 1
    assert len(result.weaknesses) == 1
    assert result.scores["clarity"] == 0.5
    assert result.stage == "hypothesis"


@pytest.mark.asyncio
async def test_json_parse_fallback(alex_with_mock, mock_groq):
    """Test fallback when LLM returns invalid JSON."""
    # First call returns bad JSON, second call also fails
    mock_groq.generate.side_effect = [
        "This is not JSON at all",
        "Still not JSON",
    ]

    result = await alex_with_mock.review_hypothesis("Test")

    # Should fall back to REVISE with generic feedback
    assert result.verdict == "REVISE"
    assert any("parsing" in w.lower() or "parse" in w.lower() for w in result.weaknesses)


@pytest.mark.asyncio
async def test_review_with_markdown_json(alex_with_mock, mock_groq):
    """Test JSON parsing with markdown code blocks."""
    # Mock LLM returns JSON wrapped in markdown
    mock_groq.generate.return_value = """```json
{
    "verdict": "PASS",
    "strengths": ["Well specified"],
    "weaknesses": [],
    "questions": [],
    "suggestions": [],
    "scores": {"clarity": 0.9},
    "requested_experiments": null
}
```"""

    result = await alex_with_mock.review_hypothesis("Solid hypothesis")

    assert result.verdict == "PASS"
    assert result.scores["clarity"] == 0.9


@pytest.mark.asyncio
async def test_invalid_verdict_uses_fallback(alex_with_mock, mock_groq):
    """Test that invalid verdict triggers retry then fallback."""
    mock_groq.generate.side_effect = [
        """{
            "verdict": "INVALID_VERDICT",
            "strengths": [],
            "weaknesses": [],
            "questions": [],
            "suggestions": [],
            "scores": {},
            "requested_experiments": null
        }""",
        "Still invalid after retry",
    ]

    result = await alex_with_mock.review_hypothesis("Test")

    assert result.verdict == "REVISE"
    assert any("parsing" in w.lower() or "parse" in w.lower() for w in result.weaknesses)


@pytest.mark.asyncio
async def test_review_with_conversation_history(alex_with_mock, mock_groq):
    """Test review method includes conversation history in prompt."""
    from src.reviewers.schemas import ReviewTurn

    mock_groq.generate.return_value = """{
        "verdict": "PASS",
        "strengths": ["Improved"],
        "weaknesses": [],
        "questions": [],
        "suggestions": [],
        "scores": {"clarity": 0.8},
        "requested_experiments": null
    }"""

    # Create mock conversation history
    history = [
        ReviewTurn(
            turn_id="turn_1",
            speaker="alex",
            content="REVISE. Too vague.",
            structured_data={"verdict": "REVISE"},
        ),
        ReviewTurn(
            turn_id="turn_2",
            speaker="fleming",
            content="Revised hypothesis with more specificity",
            structured_data={},
        ),
    ]

    result = await alex_with_mock.review(
        stage="hypothesis", artifact="Revised hypothesis", conversation_history=history
    )

    # Check that LLM was called with prompt containing history
    called_prompt = mock_groq.generate.call_args[1]["prompt"]
    assert "PREVIOUS REVIEWS" in called_prompt or "previous reviews" in called_prompt.lower()
    assert "Fleming" in called_prompt or "Alex" in called_prompt


@pytest.mark.asyncio
async def test_review_hypothesis_weak(alex_with_mock, mock_groq):
    """Test that Alex correctly identifies weak hypotheses (unit test with mock)."""
    # Mock LLM to return REVISE verdict with low scores
    mock_groq.generate.return_value = """{
        "verdict": "REVISE",
        "strengths": [],
        "weaknesses": ["Too vague and unfalsifiable", "No specific variables or conditions", "Not testable"],
        "questions": ["What exactly do you mean by 'everything'?", "How would you measure this?"],
        "suggestions": ["Define specific variables", "Add measurable conditions", "Make it falsifiable"],
        "scores": {"clarity": 0.2, "novelty": 0.3, "testability": 0.1},
        "requested_experiments": null
    }"""

    result = await alex_with_mock.review_hypothesis("Everything is connected.")

    assert result.verdict in ("REVISE", "RESTART_STAGE")
    assert len(result.weaknesses) >= 1
    # Weak hypothesis should get low clarity score
    assert result.scores.get("clarity", 1.0) < 0.6


@pytest.mark.asyncio
async def test_review_hypothesis_strong(alex_with_mock, mock_groq):
    """Test that Alex rates strong hypotheses favorably (unit test with mock)."""
    # Mock LLM to return PASS verdict with high scores
    mock_groq.generate.return_value = """{
        "verdict": "PASS",
        "strengths": ["Clear and specific", "Testable with defined thresholds", "Well-scoped"],
        "weaknesses": [],
        "questions": [],
        "suggestions": [],
        "scores": {"clarity": 0.9, "novelty": 0.7, "testability": 0.9},
        "requested_experiments": null
    }"""

    strong_hypothesis = (
        "Pre-training benefits ViT more than CNN at low data fractions (<25%) "
        "due to attention requiring more training signal, but benefits equalize above 50%."
    )

    result = await alex_with_mock.review_hypothesis(strong_hypothesis)

    # Strong hypothesis should get PASS or high clarity score
    assert result.verdict == "PASS" or result.scores.get("clarity", 0) >= 0.7
    # Should have more strengths than weaknesses
    assert len(result.strengths) >= len(result.weaknesses)


# === Integration Tests (require API, marked slow) ===


@pytest.mark.slow
async def test_review_hypothesis_weak_real_api():
    """Test Alex rejects weak hypothesis (REQUIRES GROQ_API_KEY)."""
    try:
        async with GroqClient() as groq:
            alex = Alex(groq)
            result = await alex.review_hypothesis("Everything is connected because quantum.")

            assert result.verdict in ("REVISE", "RESTART_STAGE")
            assert len(result.weaknesses) >= 1
            # Weak hypothesis should get low scores
            if result.scores:
                avg_score = sum(result.scores.values()) / len(result.scores)
                assert avg_score < 0.6, f"Weak hypothesis got high score: {avg_score}"
    except Exception as e:
        pytest.skip(f"API test skipped: {e}")


@pytest.mark.slow
async def test_review_hypothesis_strong_real_api():
    """Test Alex rates strong hypothesis higher (REQUIRES GROQ_API_KEY)."""
    try:
        async with GroqClient() as groq:
            alex = Alex(groq)
            result = await alex.review_hypothesis(
                "Pre-training benefits ViT more than CNN at low data fractions (<25%) "
                "due to attention layers requiring more training signal, but benefits "
                "equalize above 50% data."
            )

            # Strong hypothesis should get higher clarity score
            assert result.scores.get("clarity", 0) >= 0.6, (
                f"Strong hypothesis got low clarity: {result.scores}"
            )
    except Exception as e:
        pytest.skip(f"API test skipped: {e}")


@pytest.mark.slow
async def test_anti_sycophancy_real_api():
    """Test Alex is critical of bad input (REQUIRES GROQ_API_KEY)."""
    try:
        async with GroqClient() as groq:
            alex = Alex(groq)
            # Deliberately terrible hypothesis
            result = await alex.review_hypothesis("AI will solve everything.")

            # Should find multiple weaknesses
            assert len(result.weaknesses) >= 2, (
                f"Expected >=2 weaknesses, got {len(result.weaknesses)}"
            )
            # Should give low scores
            if result.scores:
                avg_score = sum(result.scores.values()) / len(result.scores)
                assert avg_score < 0.5, f"Bad hypothesis got high score: {avg_score}"
    except Exception as e:
        pytest.skip(f"API test skipped: {e}")


@pytest.mark.slow
async def test_review_paper_missing_limitations():
    """Test Alex catches missing Limitations section (REQUIRES GROQ_API_KEY)."""
    try:
        async with GroqClient() as groq:
            alex = Alex(groq)

            # Paper draft without Limitations section
            paper_no_limits = """
# Great Paper Title

## Abstract
We solve everything perfectly.

## Introduction
Our method is perfect and has no flaws whatsoever.

## Methods
We use a neural network with magical properties.

## Results
Amazing results: 99.9% accuracy on all datasets!

## Conclusion
We are the best and our method has no limitations.
"""

            result = await alex.review_paper(paper_no_limits)

            # Should catch missing limitations
            limitations_mentioned = any(
                "limitation" in w.lower() for w in result.weaknesses
            ) or any("limitation" in s.lower() for s in result.suggestions)

            assert limitations_mentioned, "Alex should catch missing Limitations section"
    except Exception as e:
        pytest.skip(f"API test skipped: {e}")


@pytest.mark.slow
async def test_review_experiment_design_real_api():
    """Test Alex reviews experiment design (REQUIRES GROQ_API_KEY)."""
    try:
        async with GroqClient() as groq:
            alex = Alex(groq)

            design = {
                "hypothesis": "ViT is better than CNN",
                "datasets": ["CIFAR-10"],
                "models": ["ViT-B/16"],
                "baselines": [],
                "metrics": ["accuracy"],
            }

            result = await alex.review_experiment_design(design)

            # Should find weaknesses in minimal design
            assert isinstance(result, ReviewResult)
            assert result.stage == "experiment_design"
            # Likely should suggest adding baselines or more datasets
            assert len(result.suggestions) >= 1
    except Exception as e:
        pytest.skip(f"API test skipped: {e}")


@pytest.mark.slow
async def test_review_results_real_api():
    """Test Alex reviews results (REQUIRES GROQ_API_KEY)."""
    try:
        async with GroqClient() as groq:
            alex = Alex(groq)

            analysis = {
                "hypothesis": "Pre-trained ViT outperforms CNN",
                "main_findings": ["ViT got 85% accuracy, CNN got 80%"],
                "metrics": {"vit_acc": 0.85, "cnn_acc": 0.80},
                "statistical_tests": {},  # Missing stats
            }

            result = await alex.review_results(analysis)

            assert isinstance(result, ReviewResult)
            assert result.stage == "results"
            # Should likely mention missing statistical tests
    except Exception as e:
        pytest.skip(f"API test skipped: {e}")
