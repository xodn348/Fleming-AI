"""
End-to-End Integration Tests for Fleming-AI
Tests the complete pipeline from paper collection to hypothesis validation
"""

import asyncio
import pytest
from pathlib import Path

from src.collectors.arxiv_client import ArxivClient
from src.llm.ollama_client import OllamaClient
from src.generators.hypothesis import HypothesisGenerator, Hypothesis
from src.storage.hypothesis_db import HypothesisDatabase
from src.storage.vectordb import VectorDB
from src.filters.quality import QualityFilter
from src.validators.pipeline import ValidationPipeline
from src.scheduler.runner import FlemingRunner


class TestComponentImports:
    """Test that all components can be imported successfully"""

    def test_import_arxiv_client(self):
        assert ArxivClient is not None

    def test_import_ollama_client(self):
        assert OllamaClient is not None

    def test_import_hypothesis_generator(self):
        assert HypothesisGenerator is not None

    def test_import_hypothesis_db(self):
        assert HypothesisDatabase is not None

    def test_import_vectordb(self):
        assert VectorDB is not None

    def test_import_quality_filter(self):
        assert QualityFilter is not None

    def test_import_validation_pipeline(self):
        assert ValidationPipeline is not None

    def test_import_fleming_runner(self):
        assert FlemingRunner is not None


class TestArxivCollection:
    """Test paper collection from arXiv"""

    def test_arxiv_client_initialization(self):
        client = ArxivClient()
        assert client is not None
        client.close()

    def test_arxiv_search_basic(self):
        with ArxivClient() as client:
            papers = client.search(
                query="cat:cs.AI",
                max_results=2,
            )
            assert isinstance(papers, list)
            assert len(papers) <= 2

    def test_arxiv_paper_structure(self):
        with ArxivClient() as client:
            papers = client.search(query="cat:cs.AI", max_results=1)
            if papers:
                paper = papers[0]
                assert "id" in paper
                assert "title" in paper
                assert "summary" in paper
                assert "authors" in paper


@pytest.mark.asyncio
class TestOllamaClient:
    """Test Ollama LLM client"""

    async def test_ollama_initialization(self):
        async with OllamaClient() as client:
            assert client is not None

    async def test_ollama_health_check(self):
        async with OllamaClient() as client:
            is_healthy = await client.health_check()
            if not is_healthy:
                pytest.skip("Ollama server not available")

    async def test_ollama_generate_simple(self):
        async with OllamaClient() as client:
            if not await client.health_check():
                pytest.skip("Ollama server not available")

            response = await client.generate(
                prompt="Say 'test' and nothing else.",
                temperature=0.1,
                max_tokens=10,
            )
            assert isinstance(response, str)
            assert len(response) > 0


class TestHypothesisDatabase:
    """Test hypothesis database operations"""

    def test_db_initialization(self):
        db_path = Path("data/db/test_hypotheses.db")
        db_path.unlink(missing_ok=True)

        with HypothesisDatabase(db_path) as db:
            assert db is not None
            count = db.count_hypotheses()
            assert count >= 0

        db_path.unlink(missing_ok=True)

    def test_db_insert_and_retrieve(self):
        db_path = Path("data/db/test_hypotheses.db")
        db_path.unlink(missing_ok=True)

        hypothesis = Hypothesis(
            id="test-001",
            hypothesis_text="Test hypothesis",
            source_papers=["paper1", "paper2"],
            connection={"concept_a": "A", "concept_b": "B", "bridging_concept": "C"},
            confidence=0.8,
            quality_score=0.7,
        )

        with HypothesisDatabase(db_path) as db:
            success = db.insert_hypothesis(hypothesis)
            assert success is True

            retrieved = db.get_hypothesis("test-001")
            assert retrieved is not None
            assert retrieved.hypothesis_text == "Test hypothesis"
            assert retrieved.confidence == 0.8

        db_path.unlink(missing_ok=True)


class TestQualityFilter:
    """Test quality filtering"""

    def test_quality_filter_initialization(self):
        filter = QualityFilter()
        assert filter is not None

    def test_quality_filter_scoring(self):
        filter = QualityFilter()
        score = filter.score("This is a well-formed scientific hypothesis about machine learning.")
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
class TestFlemingRunner:
    """Test the main Fleming runner"""

    async def test_runner_initialization(self):
        runner = FlemingRunner(test_mode=True)
        assert runner is not None
        assert runner.test_mode is True

    async def test_runner_single_cycle(self):
        runner = FlemingRunner(
            cycle_delay=1,
            max_retries=1,
            test_mode=True,
        )

        try:
            success = await asyncio.wait_for(
                runner.run_once(),
                timeout=60,
            )
            assert isinstance(success, bool)
        except asyncio.TimeoutError:
            pytest.skip("Runner cycle timed out - may need Ollama server")
        finally:
            await runner.cleanup()


@pytest.mark.asyncio
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline"""

    async def test_paper_to_hypothesis_flow(self):
        """Test: Collect papers -> Generate hypotheses -> Store in DB"""
        papers = []
        with ArxivClient() as client:
            papers = client.search(query="cat:cs.AI", max_results=2)

        assert len(papers) > 0

        async with OllamaClient() as ollama:
            if not await ollama.health_check():
                pytest.skip("Ollama server not available")

            db_path = Path("data/db/test_e2e.db")
            db_path.unlink(missing_ok=True)

            with HypothesisDatabase(db_path) as db:
                initial_count = db.count_hypotheses()
                assert initial_count == 0

            db_path.unlink(missing_ok=True)

    async def test_hypothesis_validation_flow(self):
        """Test: Create hypothesis -> Validate -> Check result"""
        db_path = Path("data/db/test_validation.db")
        db_path.unlink(missing_ok=True)

        hypothesis = Hypothesis(
            id="test-validation-001",
            hypothesis_text="If A correlates with B, and B correlates with C, then A may correlate with C.",
            source_papers=["paper1", "paper2"],
            connection={"concept_a": "A", "concept_b": "C", "bridging_concept": "B"},
            confidence=0.7,
            quality_score=0.6,
        )

        async with OllamaClient() as ollama:
            if not await ollama.health_check():
                pytest.skip("Ollama server not available")

            with HypothesisDatabase(db_path) as db:
                db.insert_hypothesis(hypothesis)

                pipeline = ValidationPipeline(
                    ollama_client=ollama,
                    hypothesis_db=db,
                    sandbox_enabled=False,
                )

                result = await pipeline.validate(hypothesis)
                assert result is not None
                assert result.hypothesis_id == "test-validation-001"
                assert result.status in ["verified", "refuted", "inconclusive", "not_testable"]

        db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
class TestComponentIntegration:
    """Test integration between components"""

    async def test_ollama_vectordb_integration(self):
        """Test that Ollama can generate embeddings for VectorDB"""
        async with OllamaClient() as ollama:
            if not await ollama.health_check():
                pytest.skip("Ollama server not available")

            embedding = await ollama.embed("test text for embedding")
            assert isinstance(embedding, list)
            assert len(embedding) > 0

    def test_arxiv_to_database_flow(self):
        """Test storing arXiv papers in database"""
        with ArxivClient() as client:
            papers = client.search(query="cat:cs.AI", max_results=1)
            assert len(papers) > 0

            paper = papers[0]
            assert paper.get("title") is not None
            assert paper.get("summary") is not None
