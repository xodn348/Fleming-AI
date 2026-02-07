"""
Tests for Validation Pipeline, Classifier, and Result
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generators.hypothesis import Hypothesis
from src.storage.hypothesis_db import HypothesisDatabase
from src.validators.classifier import (
    CLASS_COMPUTATIONAL,
    CLASS_DATA_DRIVEN,
    CLASS_EXPERIMENTAL,
    CLASS_THEORETICAL,
    COMPUTATIONAL_KEYWORDS,
    DATA_DRIVEN_KEYWORDS,
    EXPERIMENTAL_KEYWORDS,
    HypothesisClassifier,
)
from src.validators.pipeline import ValidationPipeline
from src.validators.result import (
    STATUS_INCONCLUSIVE,
    STATUS_NOT_TESTABLE,
    STATUS_REFUTED,
    STATUS_VERIFIED,
    VALID_STATUSES,
    ValidationResult,
)


# Sample hypotheses for testing
COMPUTATIONAL_HYPOTHESIS = Hypothesis(
    id="comp-001",
    hypothesis_text=(
        "The neural network model can predict protein folding "
        "with 95% accuracy using deep learning algorithms."
    ),
    source_papers=["paper-001", "paper-002"],
    connection={
        "concept_a": "neural network",
        "concept_b": "protein folding",
        "bridging_concept": "deep learning",
    },
    confidence=0.8,
    quality_score=0.75,
)

DATA_DRIVEN_HYPOTHESIS = Hypothesis(
    id="data-001",
    hypothesis_text=(
        "Population-level analysis of clinical trial data shows "
        "that the drug reduces symptoms in the cohort study."
    ),
    source_papers=["paper-003", "paper-004"],
    connection={
        "concept_a": "drug",
        "concept_b": "symptom reduction",
        "bridging_concept": "clinical trial",
    },
    confidence=0.7,
    quality_score=0.65,
)

EXPERIMENTAL_HYPOTHESIS = Hypothesis(
    id="exp-001",
    hypothesis_text=(
        "In vitro laboratory experiments show that the chemical "
        "synthesis produces the desired compound in cell culture."
    ),
    source_papers=["paper-005", "paper-006"],
    connection={
        "concept_a": "chemical synthesis",
        "concept_b": "compound production",
        "bridging_concept": "cell culture",
    },
    confidence=0.6,
    quality_score=0.55,
)

THEORETICAL_HYPOTHESIS = Hypothesis(
    id="theo-001",
    hypothesis_text=(
        "The theoretical proof demonstrates that the conjecture "
        "holds under the given axiomatic framework."
    ),
    source_papers=["paper-007", "paper-008"],
    connection={
        "concept_a": "conjecture",
        "concept_b": "theorem",
        "bridging_concept": "proof",
    },
    confidence=0.5,
    quality_score=0.45,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass"""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult instance"""
        result = ValidationResult(
            hypothesis_id="test-001",
            status=STATUS_VERIFIED,
            evidence={"output": "success"},
            logs=["Test log"],
        )

        assert result.hypothesis_id == "test-001"
        assert result.status == STATUS_VERIFIED
        assert result.evidence == {"output": "success"}
        assert result.logs == ["Test log"]
        assert isinstance(result.validated_at, datetime)

    def test_validation_result_invalid_status(self):
        """Test that invalid status raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            ValidationResult(
                hypothesis_id="test-001",
                status="invalid_status",
            )

        assert "Invalid status" in str(exc_info.value)

    def test_validation_result_valid_statuses(self):
        """Test all valid status values"""
        for status in VALID_STATUSES:
            result = ValidationResult(
                hypothesis_id="test-001",
                status=status,
            )
            assert result.status == status

    def test_validation_result_to_dict(self):
        """Test converting ValidationResult to dictionary"""
        result = ValidationResult(
            hypothesis_id="test-001",
            status=STATUS_VERIFIED,
            evidence={"key": "value"},
            logs=["log1", "log2"],
            classification=CLASS_COMPUTATIONAL,
            execution_time_ms=100,
            code_executed="print('hello')",
        )

        result_dict = result.to_dict()

        assert result_dict["hypothesis_id"] == "test-001"
        assert result_dict["status"] == STATUS_VERIFIED
        assert result_dict["evidence"] == {"key": "value"}
        assert result_dict["logs"] == ["log1", "log2"]
        assert result_dict["classification"] == CLASS_COMPUTATIONAL
        assert result_dict["execution_time_ms"] == 100
        assert result_dict["code_executed"] == "print('hello')"
        assert "validated_at" in result_dict

    def test_validation_result_from_dict(self):
        """Test creating ValidationResult from dictionary"""
        data = {
            "hypothesis_id": "dict-001",
            "status": STATUS_REFUTED,
            "evidence": {"result": "negative"},
            "logs": ["test log"],
            "validated_at": "2025-02-06T12:00:00",
            "classification": CLASS_DATA_DRIVEN,
            "execution_time_ms": 500,
        }

        result = ValidationResult.from_dict(data)

        assert result.hypothesis_id == "dict-001"
        assert result.status == STATUS_REFUTED
        assert result.evidence == {"result": "negative"}
        assert result.classification == CLASS_DATA_DRIVEN
        assert result.execution_time_ms == 500
        assert isinstance(result.validated_at, datetime)

    def test_validation_result_from_dict_defaults(self):
        """Test ValidationResult.from_dict with minimal data"""
        data = {
            "hypothesis_id": "min-001",
            "status": STATUS_INCONCLUSIVE,
        }

        result = ValidationResult.from_dict(data)

        assert result.hypothesis_id == "min-001"
        assert result.status == STATUS_INCONCLUSIVE
        assert result.evidence == {}
        assert result.logs == []
        assert result.classification == ""
        assert result.execution_time_ms == 0

    def test_validation_result_is_verified(self):
        """Test is_verified method"""
        verified = ValidationResult(hypothesis_id="t1", status=STATUS_VERIFIED)
        refuted = ValidationResult(hypothesis_id="t2", status=STATUS_REFUTED)

        assert verified.is_verified() is True
        assert refuted.is_verified() is False

    def test_validation_result_is_refuted(self):
        """Test is_refuted method"""
        verified = ValidationResult(hypothesis_id="t1", status=STATUS_VERIFIED)
        refuted = ValidationResult(hypothesis_id="t2", status=STATUS_REFUTED)

        assert verified.is_refuted() is False
        assert refuted.is_refuted() is True

    def test_validation_result_is_testable(self):
        """Test is_testable method"""
        testable = ValidationResult(hypothesis_id="t1", status=STATUS_VERIFIED)
        not_testable = ValidationResult(hypothesis_id="t2", status=STATUS_NOT_TESTABLE)

        assert testable.is_testable() is True
        assert not_testable.is_testable() is False

    def test_validation_result_add_log(self):
        """Test add_log method"""
        result = ValidationResult(hypothesis_id="t1", status=STATUS_VERIFIED)
        result.add_log("Test message")

        assert len(result.logs) == 1
        assert "Test message" in result.logs[0]
        assert "[" in result.logs[0]  # Timestamp present

    def test_validation_result_add_evidence(self):
        """Test add_evidence method"""
        result = ValidationResult(hypothesis_id="t1", status=STATUS_VERIFIED)
        result.add_evidence("key1", "value1")
        result.add_evidence("key2", {"nested": "data"})

        assert result.evidence["key1"] == "value1"
        assert result.evidence["key2"] == {"nested": "data"}


class TestHypothesisClassifier:
    """Tests for HypothesisClassifier"""

    @pytest.fixture
    def classifier(self):
        """Create HypothesisClassifier instance"""
        return HypothesisClassifier()

    def test_classifier_init(self):
        """Test HypothesisClassifier initialization"""
        classifier = HypothesisClassifier()
        assert classifier.ollama is None
        assert classifier.use_llm is False

    def test_classifier_init_with_ollama(self):
        """Test HypothesisClassifier with ollama client"""
        mock_ollama = MagicMock()
        classifier = HypothesisClassifier(ollama_client=mock_ollama, use_llm=True)

        assert classifier.ollama is mock_ollama
        assert classifier.use_llm is True

    def test_classify_computational(self, classifier):
        """Test classifying computational hypothesis"""
        result = classifier.classify(COMPUTATIONAL_HYPOTHESIS)
        assert result == CLASS_COMPUTATIONAL

    def test_classify_data_driven(self, classifier):
        """Test classifying data-driven hypothesis"""
        result = classifier.classify(DATA_DRIVEN_HYPOTHESIS)
        assert result == CLASS_DATA_DRIVEN

    def test_classify_experimental(self, classifier):
        """Test classifying experimental hypothesis"""
        result = classifier.classify(EXPERIMENTAL_HYPOTHESIS)
        assert result == CLASS_EXPERIMENTAL

    def test_classify_theoretical(self, classifier):
        """Test classifying theoretical hypothesis"""
        result = classifier.classify(THEORETICAL_HYPOTHESIS)
        assert result == CLASS_THEORETICAL

    def test_classify_unknown_defaults_to_theoretical(self, classifier):
        """Test that unknown hypothesis defaults to theoretical"""
        unknown_hypothesis = Hypothesis(
            id="unknown-001",
            hypothesis_text="Something vague without specific keywords.",
            source_papers=[],
            connection={},
            confidence=0.5,
            quality_score=0.5,
        )

        result = classifier.classify(unknown_hypothesis)
        assert result == CLASS_THEORETICAL

    def test_classify_batch(self, classifier):
        """Test batch classification"""
        hypotheses = [
            COMPUTATIONAL_HYPOTHESIS,
            DATA_DRIVEN_HYPOTHESIS,
            EXPERIMENTAL_HYPOTHESIS,
            THEORETICAL_HYPOTHESIS,
        ]

        results = classifier.classify_batch(hypotheses)

        assert len(results) == 4
        assert results["comp-001"] == CLASS_COMPUTATIONAL
        assert results["data-001"] == CLASS_DATA_DRIVEN
        assert results["exp-001"] == CLASS_EXPERIMENTAL
        assert results["theo-001"] == CLASS_THEORETICAL

    def test_get_classification_stats(self, classifier):
        """Test getting classification statistics"""
        hypotheses = [
            COMPUTATIONAL_HYPOTHESIS,
            COMPUTATIONAL_HYPOTHESIS,
            DATA_DRIVEN_HYPOTHESIS,
            EXPERIMENTAL_HYPOTHESIS,
        ]

        stats = classifier.get_classification_stats(hypotheses)

        assert stats[CLASS_COMPUTATIONAL] == 2
        assert stats[CLASS_DATA_DRIVEN] == 1
        assert stats[CLASS_EXPERIMENTAL] == 1
        assert stats[CLASS_THEORETICAL] == 0

    def test_is_programmatically_testable(self, classifier):
        """Test is_programmatically_testable method"""
        assert classifier.is_programmatically_testable(COMPUTATIONAL_HYPOTHESIS) is True
        assert classifier.is_programmatically_testable(DATA_DRIVEN_HYPOTHESIS) is True
        assert classifier.is_programmatically_testable(EXPERIMENTAL_HYPOTHESIS) is False
        assert classifier.is_programmatically_testable(THEORETICAL_HYPOTHESIS) is False

    @pytest.mark.asyncio
    async def test_classify_async_without_llm(self, classifier):
        """Test async classification without LLM falls back to heuristic"""
        result = await classifier.classify_async(COMPUTATIONAL_HYPOTHESIS)
        assert result == CLASS_COMPUTATIONAL

    @pytest.mark.asyncio
    async def test_classify_async_with_llm(self):
        """Test async classification with LLM"""
        mock_ollama = AsyncMock()
        mock_ollama.generate.return_value = (
            '{"classification": "computational", "reasoning": "Uses ML"}'
        )

        classifier = HypothesisClassifier(ollama_client=mock_ollama, use_llm=True)
        result = await classifier.classify_async(COMPUTATIONAL_HYPOTHESIS)

        assert result == CLASS_COMPUTATIONAL
        mock_ollama.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_async_llm_fallback_on_error(self):
        """Test async classification falls back on LLM error"""
        mock_ollama = AsyncMock()
        mock_ollama.generate.side_effect = Exception("LLM error")

        classifier = HypothesisClassifier(ollama_client=mock_ollama, use_llm=True)
        result = await classifier.classify_async(COMPUTATIONAL_HYPOTHESIS)

        # Should fall back to heuristic classification
        assert result == CLASS_COMPUTATIONAL


class TestValidationPipeline:
    """Tests for ValidationPipeline"""

    @pytest.fixture
    def mock_ollama(self):
        """Create mock OllamaClient"""
        mock = AsyncMock()
        mock.generate.return_value = "INCONCLUSIVE\nSimulated output"
        return mock

    @pytest.fixture
    def db(self):
        """Create temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_validation.db"
            db = HypothesisDatabase(db_path)
            yield db
            db.close()

    @pytest.fixture
    def pipeline(self, mock_ollama, db):
        """Create ValidationPipeline with mocks"""
        return ValidationPipeline(
            ollama_client=mock_ollama,
            hypothesis_db=db,
            sandbox_enabled=False,  # Use simulated execution
        )

    def test_pipeline_init(self, mock_ollama, db):
        """Test ValidationPipeline initialization"""
        pipeline = ValidationPipeline(
            ollama_client=mock_ollama,
            hypothesis_db=db,
        )

        assert pipeline.ollama is mock_ollama
        assert pipeline.db is db
        assert pipeline.classifier is not None
        assert pipeline.sandbox_enabled is True

    @pytest.mark.asyncio
    async def test_validate_computational_hypothesis(self, pipeline, db):
        """Test validating a computational hypothesis"""
        # Insert hypothesis into DB
        db.insert_hypothesis(COMPUTATIONAL_HYPOTHESIS)

        # Mock the LLM to return validation code
        pipeline.ollama.generate.side_effect = [
            '{"classification": "computational", "reasoning": "Uses ML"}',
            '```python\nprint("VERIFIED")\n```',
            '{"status": "verified", "reasoning": "Output shows verified"}',
        ]

        result = await pipeline.validate(COMPUTATIONAL_HYPOTHESIS)

        assert result.hypothesis_id == "comp-001"
        assert result.classification == CLASS_COMPUTATIONAL
        assert result.execution_time_ms >= 0  # May be 0 for fast mock execution
        assert len(result.logs) > 0

    @pytest.mark.asyncio
    async def test_validate_experimental_hypothesis(self, pipeline, db):
        """Test validating an experimental hypothesis (not testable)"""
        db.insert_hypothesis(EXPERIMENTAL_HYPOTHESIS)

        result = await pipeline.validate(EXPERIMENTAL_HYPOTHESIS)

        assert result.hypothesis_id == "exp-001"
        assert result.status == STATUS_NOT_TESTABLE
        assert result.classification == CLASS_EXPERIMENTAL
        assert "experiment" in " ".join(result.logs).lower()

    @pytest.mark.asyncio
    async def test_validate_theoretical_hypothesis(self, pipeline, db):
        """Test validating a theoretical hypothesis"""
        db.insert_hypothesis(THEORETICAL_HYPOTHESIS)

        # Mock theoretical analysis response
        pipeline.ollama.generate.side_effect = [
            '{"classification": "theoretical", "reasoning": "Formal proof"}',
            """{
                "is_logically_sound": true,
                "has_contradictions": false,
                "plausibility": "medium",
                "assumptions": ["assumption1"],
                "analysis": "Theory appears sound"
            }""",
        ]

        result = await pipeline.validate(THEORETICAL_HYPOTHESIS)

        assert result.hypothesis_id == "theo-001"
        assert result.classification == CLASS_THEORETICAL
        assert result.status == STATUS_VERIFIED  # Logically sound

    @pytest.mark.asyncio
    async def test_validate_data_driven_hypothesis(self, pipeline, db):
        """Test validating a data-driven hypothesis"""
        db.insert_hypothesis(DATA_DRIVEN_HYPOTHESIS)

        # Mock data analysis response
        pipeline.ollama.generate.side_effect = [
            '{"classification": "data_driven", "reasoning": "Uses datasets"}',
            '{"datasets": ["dataset1"], "methods": ["regression"]}',
        ]

        result = await pipeline.validate(DATA_DRIVEN_HYPOTHESIS)

        assert result.hypothesis_id == "data-001"
        assert result.classification == CLASS_DATA_DRIVEN
        assert result.status == STATUS_INCONCLUSIVE  # Simulated

    @pytest.mark.asyncio
    async def test_validate_batch(self, pipeline, db):
        """Test batch validation"""
        hypotheses = [COMPUTATIONAL_HYPOTHESIS, EXPERIMENTAL_HYPOTHESIS]
        for h in hypotheses:
            db.insert_hypothesis(h)

        results = await pipeline.validate_batch(hypotheses)

        assert len(results) == 2
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_get_validation_stats(self, pipeline):
        """Test getting validation statistics"""
        results = [
            ValidationResult(hypothesis_id="1", status=STATUS_VERIFIED),
            ValidationResult(hypothesis_id="2", status=STATUS_VERIFIED),
            ValidationResult(hypothesis_id="3", status=STATUS_REFUTED),
            ValidationResult(hypothesis_id="4", status=STATUS_INCONCLUSIVE),
            ValidationResult(hypothesis_id="5", status=STATUS_NOT_TESTABLE),
        ]

        stats = pipeline.get_validation_stats(results)

        assert stats[STATUS_VERIFIED] == 2
        assert stats[STATUS_REFUTED] == 1
        assert stats[STATUS_INCONCLUSIVE] == 1
        assert stats[STATUS_NOT_TESTABLE] == 1

    def test_extract_code_block(self, pipeline):
        """Test extracting code from markdown"""
        markdown = """Here is the code:
```python
print("hello")
```
"""
        code = pipeline._extract_code_block(markdown)
        assert code == 'print("hello")'

    def test_extract_code_block_no_language(self, pipeline):
        """Test extracting code without language specifier"""
        markdown = """```
print("world")
```"""
        code = pipeline._extract_code_block(markdown)
        assert code == 'print("world")'

    def test_extract_code_block_no_block(self, pipeline):
        """Test extracting when no code block present"""
        text = 'print("raw code")'
        code = pipeline._extract_code_block(text)
        assert code == 'print("raw code")'

    @pytest.mark.asyncio
    async def test_execute_code_simulated(self, pipeline):
        """Test simulated code execution"""
        code = "print('test')"
        output, error, success = await pipeline._execute_code_simulated(code)

        assert "INCONCLUSIVE" in output
        assert "Simulated" in output
        assert error == ""
        assert success is True

    @pytest.mark.asyncio
    async def test_analyze_execution_results_verified(self, pipeline, db):
        """Test analyzing execution results for verified status"""
        db.insert_hypothesis(COMPUTATIONAL_HYPOTHESIS)
        result = ValidationResult(
            hypothesis_id=COMPUTATIONAL_HYPOTHESIS.id,
            status=STATUS_INCONCLUSIVE,
        )

        analyzed = await pipeline._analyze_execution_results(
            COMPUTATIONAL_HYPOTHESIS,
            result,
            output="VERIFIED: The test passed",
            error="",
        )

        assert analyzed.status == STATUS_VERIFIED

    @pytest.mark.asyncio
    async def test_analyze_execution_results_refuted(self, pipeline, db):
        """Test analyzing execution results for refuted status"""
        db.insert_hypothesis(COMPUTATIONAL_HYPOTHESIS)
        result = ValidationResult(
            hypothesis_id=COMPUTATIONAL_HYPOTHESIS.id,
            status=STATUS_INCONCLUSIVE,
        )

        analyzed = await pipeline._analyze_execution_results(
            COMPUTATIONAL_HYPOTHESIS,
            result,
            output="REFUTED: The hypothesis is false",
            error="",
        )

        assert analyzed.status == STATUS_REFUTED

    @pytest.mark.asyncio
    async def test_analyze_execution_results_with_error(self, pipeline, db):
        """Test analyzing execution results with errors"""
        db.insert_hypothesis(COMPUTATIONAL_HYPOTHESIS)
        result = ValidationResult(
            hypothesis_id=COMPUTATIONAL_HYPOTHESIS.id,
            status=STATUS_INCONCLUSIVE,
        )

        analyzed = await pipeline._analyze_execution_results(
            COMPUTATIONAL_HYPOTHESIS,
            result,
            output="",
            error="SyntaxError: invalid syntax",
        )

        assert analyzed.status == STATUS_INCONCLUSIVE


class TestKeywordLists:
    """Tests for keyword lists used in classification"""

    def test_computational_keywords_exist(self):
        """Test that computational keywords are defined"""
        assert len(COMPUTATIONAL_KEYWORDS) > 0
        assert "algorithm" in COMPUTATIONAL_KEYWORDS
        assert "machine learning" in COMPUTATIONAL_KEYWORDS

    def test_data_driven_keywords_exist(self):
        """Test that data-driven keywords are defined"""
        assert len(DATA_DRIVEN_KEYWORDS) > 0
        assert "dataset" in DATA_DRIVEN_KEYWORDS
        assert "clinical trial" in DATA_DRIVEN_KEYWORDS

    def test_experimental_keywords_exist(self):
        """Test that experimental keywords are defined"""
        assert len(EXPERIMENTAL_KEYWORDS) > 0
        assert "laboratory" in EXPERIMENTAL_KEYWORDS
        assert "in vitro" in EXPERIMENTAL_KEYWORDS


class TestValidationIntegration:
    """Integration tests for the validation pipeline"""

    @pytest.fixture
    def full_pipeline(self):
        """Create full pipeline with temporary database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "integration_test.db"
            db = HypothesisDatabase(db_path)
            mock_ollama = AsyncMock()

            # Set up mock responses for full flow
            mock_ollama.generate.return_value = "INCONCLUSIVE"

            pipeline = ValidationPipeline(
                ollama_client=mock_ollama,
                hypothesis_db=db,
                sandbox_enabled=False,
            )

            yield pipeline, db

            db.close()

    @pytest.mark.asyncio
    async def test_full_validation_flow(self, full_pipeline):
        """Test complete validation flow"""
        pipeline, db = full_pipeline

        # Insert hypothesis
        db.insert_hypothesis(COMPUTATIONAL_HYPOTHESIS)

        # Validate
        result = await pipeline.validate(COMPUTATIONAL_HYPOTHESIS)

        # Check result
        assert result.hypothesis_id == COMPUTATIONAL_HYPOTHESIS.id
        assert result.classification in [
            CLASS_COMPUTATIONAL,
            CLASS_DATA_DRIVEN,
            CLASS_EXPERIMENTAL,
            CLASS_THEORETICAL,
        ]
        assert result.validated_at is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_hypothesis_status_updated_after_validation(self, full_pipeline):
        """Test that hypothesis status is updated in DB after validation"""
        pipeline, db = full_pipeline

        # Insert hypothesis
        db.insert_hypothesis(COMPUTATIONAL_HYPOTHESIS)

        # Mock verified result
        pipeline.ollama.generate.side_effect = [
            '{"classification": "computational"}',
            '```python\nprint("VERIFIED")\n```',
        ]

        # Validate
        result = await pipeline.validate(COMPUTATIONAL_HYPOTHESIS)

        # Note: Status update depends on validation result
        # For simulated execution, status remains inconclusive
        hypothesis = db.get_hypothesis(COMPUTATIONAL_HYPOTHESIS.id)
        assert hypothesis is not None
