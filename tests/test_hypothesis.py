"""
Tests for Hypothesis Generator and Hypothesis Database
"""

import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generators.hypothesis import ConceptPair, Hypothesis, HypothesisGenerator
from src.storage.hypothesis_db import HypothesisDatabase


# Sample data for testing
SAMPLE_HYPOTHESIS = Hypothesis(
    id="test-hypo-001",
    hypothesis_text="Fish oil omega-3 fatty acids may reduce inflammation in migraine patients.",
    source_papers=["paper-001", "paper-002"],
    connection={
        "concept_a": "fish oil",
        "concept_b": "migraine",
        "bridging_concept": "inflammation",
    },
    confidence=0.75,
    quality_score=0.68,
    created_at=datetime(2025, 2, 6, 12, 0, 0),
    status="pending",
)

SAMPLE_PAPERS = [
    {
        "paper_id": "paper-001",
        "text": """Title: Omega-3 Fatty Acids and Inflammation
        
Abstract: This study investigates the anti-inflammatory effects of omega-3 fatty acids 
found in fish oil. We demonstrate that EPA and DHA significantly reduce inflammatory 
markers including IL-6 and TNF-alpha in clinical trials.

Methods: Randomized controlled trial with 200 participants over 12 weeks.

Results: Fish oil supplementation reduced inflammation by 35% compared to placebo.""",
        "title": "Omega-3 Fatty Acids and Inflammation",
    },
    {
        "paper_id": "paper-002",
        "text": """Title: Inflammation in Chronic Migraine Pathophysiology

Abstract: We examine the role of neuroinflammation in migraine. Our findings show that 
inflammatory markers are elevated during migraine attacks. Targeting inflammation 
may provide therapeutic benefits for migraine patients.

Methods: Case-control study with 150 migraine patients.

Results: IL-6 and TNF-alpha were significantly elevated in migraine patients.""",
        "title": "Inflammation in Chronic Migraine Pathophysiology",
    },
]


class TestHypothesisDataclass:
    """Tests for Hypothesis dataclass"""

    def test_hypothesis_creation(self):
        """Test creating a Hypothesis instance"""
        hypo = Hypothesis(
            id="test-001",
            hypothesis_text="Test hypothesis",
            source_papers=["paper1", "paper2"],
            connection={"concept_a": "A", "concept_b": "B", "bridging_concept": "C"},
            confidence=0.8,
            quality_score=0.7,
        )

        assert hypo.id == "test-001"
        assert hypo.hypothesis_text == "Test hypothesis"
        assert hypo.source_papers == ["paper1", "paper2"]
        assert hypo.confidence == 0.8
        assert hypo.quality_score == 0.7
        assert hypo.status == "pending"

    def test_hypothesis_to_dict(self):
        """Test converting Hypothesis to dictionary"""
        hypo_dict = SAMPLE_HYPOTHESIS.to_dict()

        assert isinstance(hypo_dict, dict)
        assert hypo_dict["id"] == "test-hypo-001"
        assert hypo_dict["hypothesis_text"] == SAMPLE_HYPOTHESIS.hypothesis_text
        assert hypo_dict["source_papers"] == ["paper-001", "paper-002"]
        assert hypo_dict["connection"]["bridging_concept"] == "inflammation"
        assert hypo_dict["confidence"] == 0.75
        assert hypo_dict["quality_score"] == 0.68
        assert "created_at" in hypo_dict
        assert hypo_dict["status"] == "pending"

    def test_hypothesis_from_dict(self):
        """Test creating Hypothesis from dictionary"""
        hypo_dict = {
            "id": "dict-001",
            "hypothesis_text": "Hypothesis from dict",
            "source_papers": ["p1", "p2"],
            "connection": {"concept_a": "X", "concept_b": "Y", "bridging_concept": "Z"},
            "confidence": 0.6,
            "quality_score": 0.55,
            "created_at": "2025-02-06T10:00:00",
            "status": "validated",
        }

        hypo = Hypothesis.from_dict(hypo_dict)

        assert hypo.id == "dict-001"
        assert hypo.hypothesis_text == "Hypothesis from dict"
        assert hypo.confidence == 0.6
        assert hypo.status == "validated"
        assert isinstance(hypo.created_at, datetime)

    def test_hypothesis_from_dict_defaults(self):
        """Test Hypothesis.from_dict with minimal data"""
        hypo_dict = {"hypothesis_text": "Minimal hypothesis"}

        hypo = Hypothesis.from_dict(hypo_dict)

        assert hypo.hypothesis_text == "Minimal hypothesis"
        assert hypo.source_papers == []
        assert hypo.connection == {}
        assert hypo.confidence == 0.0
        assert hypo.quality_score == 0.0
        assert hypo.status == "pending"
        assert isinstance(hypo.id, str)
        assert isinstance(hypo.created_at, datetime)

    def test_hypothesis_default_datetime(self):
        """Test that created_at defaults to now"""
        before = datetime.now()
        hypo = Hypothesis(
            id="time-test",
            hypothesis_text="Test",
            source_papers=[],
            connection={},
            confidence=0.5,
            quality_score=0.5,
        )
        after = datetime.now()

        assert before <= hypo.created_at <= after


class TestConceptPair:
    """Tests for ConceptPair dataclass"""

    def test_concept_pair_creation(self):
        """Test creating a ConceptPair instance"""
        pair = ConceptPair(
            concept_a="fish oil",
            concept_b="migraine",
            bridging_concept="inflammation",
            paper_a_id="paper-001",
            paper_b_id="paper-002",
            strength=0.75,
        )

        assert pair.concept_a == "fish oil"
        assert pair.concept_b == "migraine"
        assert pair.bridging_concept == "inflammation"
        assert pair.paper_a_id == "paper-001"
        assert pair.paper_b_id == "paper-002"
        assert pair.strength == 0.75

    def test_concept_pair_default_strength(self):
        """Test ConceptPair default strength"""
        pair = ConceptPair(
            concept_a="A",
            concept_b="B",
            bridging_concept="C",
            paper_a_id="p1",
            paper_b_id="p2",
        )

        assert pair.strength == 0.0


class TestHypothesisDatabase:
    """Tests for HypothesisDatabase"""

    @pytest.fixture
    def db(self):
        """Create temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_hypotheses.db"
            db = HypothesisDatabase(db_path)
            yield db
            db.close()

    def test_database_init(self, db):
        """Test database initialization"""
        assert db.conn is not None
        assert db.db_path.exists()

    def test_insert_hypothesis(self, db):
        """Test inserting a hypothesis"""
        result = db.insert_hypothesis(SAMPLE_HYPOTHESIS)

        assert result is True
        assert db.count_hypotheses() == 1

    def test_insert_duplicate_hypothesis(self, db):
        """Test inserting duplicate hypothesis returns False"""
        db.insert_hypothesis(SAMPLE_HYPOTHESIS)
        result = db.insert_hypothesis(SAMPLE_HYPOTHESIS)

        assert result is False
        assert db.count_hypotheses() == 1

    def test_insert_hypotheses_batch(self, db):
        """Test batch insertion"""
        hypotheses = [
            Hypothesis(
                id=f"batch-{i}",
                hypothesis_text=f"Hypothesis {i}",
                source_papers=["p1"],
                connection={"concept_a": "A", "concept_b": "B", "bridging_concept": "C"},
                confidence=0.5 + i * 0.1,
                quality_score=0.6,
            )
            for i in range(5)
        ]

        inserted, skipped = db.insert_hypotheses_batch(hypotheses)

        assert inserted == 5
        assert skipped == 0
        assert db.count_hypotheses() == 5

    def test_get_hypothesis(self, db):
        """Test retrieving a hypothesis by ID"""
        db.insert_hypothesis(SAMPLE_HYPOTHESIS)
        retrieved = db.get_hypothesis("test-hypo-001")

        assert retrieved is not None
        assert retrieved.id == "test-hypo-001"
        assert retrieved.hypothesis_text == SAMPLE_HYPOTHESIS.hypothesis_text
        assert retrieved.source_papers == ["paper-001", "paper-002"]
        assert retrieved.connection["bridging_concept"] == "inflammation"

    def test_get_hypothesis_not_found(self, db):
        """Test retrieving non-existent hypothesis"""
        result = db.get_hypothesis("non-existent-id")

        assert result is None

    def test_get_hypotheses_by_status(self, db):
        """Test filtering by status"""
        # Insert hypotheses with different statuses
        for status in ["pending", "validated", "rejected"]:
            db.insert_hypothesis(
                Hypothesis(
                    id=f"status-{status}",
                    hypothesis_text=f"Hypothesis {status}",
                    source_papers=[],
                    connection={},
                    confidence=0.5,
                    quality_score=0.5,
                    status=status,
                )
            )

        pending = db.get_hypotheses_by_status("pending")
        validated = db.get_hypotheses_by_status("validated")

        assert len(pending) == 1
        assert pending[0].status == "pending"
        assert len(validated) == 1
        assert validated[0].status == "validated"

    def test_get_top_hypotheses(self, db):
        """Test retrieving top hypotheses by score"""
        # Insert hypotheses with varying scores
        for i in range(10):
            db.insert_hypothesis(
                Hypothesis(
                    id=f"score-{i}",
                    hypothesis_text=f"Hypothesis {i}",
                    source_papers=[],
                    connection={},
                    confidence=0.1 * i,
                    quality_score=0.1 * i,
                )
            )

        top5 = db.get_top_hypotheses(limit=5)

        assert len(top5) == 5
        # Should be sorted by combined score descending
        scores = [(h.confidence + h.quality_score) / 2 for h in top5]
        assert scores == sorted(scores, reverse=True)

    def test_get_top_hypotheses_with_threshold(self, db):
        """Test top hypotheses with score thresholds"""
        for i in range(10):
            db.insert_hypothesis(
                Hypothesis(
                    id=f"thresh-{i}",
                    hypothesis_text=f"Hypothesis {i}",
                    source_papers=[],
                    connection={},
                    confidence=0.1 * i,
                    quality_score=0.1 * i,
                )
            )

        # Only get hypotheses with confidence >= 0.5
        top = db.get_top_hypotheses(limit=10, min_confidence=0.5)

        assert all(h.confidence >= 0.5 for h in top)

    def test_search_hypotheses(self, db):
        """Test searching hypotheses by text"""
        db.insert_hypothesis(SAMPLE_HYPOTHESIS)
        db.insert_hypothesis(
            Hypothesis(
                id="unrelated",
                hypothesis_text="Something about quantum physics",
                source_papers=[],
                connection={"concept_a": "quark", "concept_b": "photon"},
                confidence=0.5,
                quality_score=0.5,
            )
        )

        results = db.search_hypotheses("inflammation")

        assert len(results) == 1
        assert (
            "inflammation" in results[0].hypothesis_text.lower()
            or "inflammation" in str(results[0].connection).lower()
        )

    def test_update_status(self, db):
        """Test updating hypothesis status"""
        db.insert_hypothesis(SAMPLE_HYPOTHESIS)

        result = db.update_status("test-hypo-001", "validated")

        assert result is True
        updated = db.get_hypothesis("test-hypo-001")
        assert updated.status == "validated"

    def test_update_status_not_found(self, db):
        """Test updating non-existent hypothesis"""
        result = db.update_status("non-existent", "validated")

        assert result is False

    def test_update_scores(self, db):
        """Test updating hypothesis scores"""
        db.insert_hypothesis(SAMPLE_HYPOTHESIS)

        result = db.update_scores("test-hypo-001", confidence=0.9, quality_score=0.85)

        assert result is True
        updated = db.get_hypothesis("test-hypo-001")
        assert updated.confidence == 0.9
        assert updated.quality_score == 0.85

    def test_update_scores_partial(self, db):
        """Test updating only one score"""
        db.insert_hypothesis(SAMPLE_HYPOTHESIS)

        db.update_scores("test-hypo-001", confidence=0.95)

        updated = db.get_hypothesis("test-hypo-001")
        assert updated.confidence == 0.95
        assert updated.quality_score == SAMPLE_HYPOTHESIS.quality_score

    def test_delete_hypothesis(self, db):
        """Test deleting a hypothesis"""
        db.insert_hypothesis(SAMPLE_HYPOTHESIS)
        assert db.count_hypotheses() == 1

        result = db.delete_hypothesis("test-hypo-001")

        assert result is True
        assert db.count_hypotheses() == 0

    def test_delete_hypothesis_not_found(self, db):
        """Test deleting non-existent hypothesis"""
        result = db.delete_hypothesis("non-existent")

        assert result is False

    def test_count_hypotheses(self, db):
        """Test counting hypotheses"""
        assert db.count_hypotheses() == 0

        for i in range(5):
            db.insert_hypothesis(
                Hypothesis(
                    id=f"count-{i}",
                    hypothesis_text=f"H{i}",
                    source_papers=[],
                    connection={},
                    confidence=0.5,
                    quality_score=0.5,
                    status="pending" if i < 3 else "validated",
                )
            )

        assert db.count_hypotheses() == 5
        assert db.count_hypotheses(status="pending") == 3
        assert db.count_hypotheses(status="validated") == 2

    def test_context_manager(self):
        """Test database context manager"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context_test.db"

            with HypothesisDatabase(db_path) as db:
                db.insert_hypothesis(SAMPLE_HYPOTHESIS)
                assert db.count_hypotheses() == 1

    def test_get_all_hypotheses(self, db):
        """Test getting all hypotheses"""
        for i in range(5):
            db.insert_hypothesis(
                Hypothesis(
                    id=f"all-{i}",
                    hypothesis_text=f"H{i}",
                    source_papers=[],
                    connection={},
                    confidence=0.5,
                    quality_score=0.5,
                )
            )

        all_hypos = db.get_all_hypotheses()
        limited = db.get_all_hypotheses(limit=3)

        assert len(all_hypos) == 5
        assert len(limited) == 3


class TestHypothesisGenerator:
    """Tests for HypothesisGenerator"""

    @pytest.fixture
    def mock_ollama(self):
        """Create mock OllamaClient"""
        mock = AsyncMock()
        mock.generate = AsyncMock(return_value='["fish oil", "omega-3", "inflammation"]')
        return mock

    @pytest.fixture
    def mock_vectordb(self):
        """Create mock VectorDB"""
        mock = MagicMock()
        mock.search.return_value = [
            {"id": "paper-001_abstract", "metadata": {"paper_id": "paper-001"}},
            {"id": "paper-002_abstract", "metadata": {"paper_id": "paper-002"}},
        ]
        mock.get_paper.side_effect = lambda paper_id: {
            "paper_id": paper_id,
            "chunks": [
                {
                    "text": next(p["text"] for p in SAMPLE_PAPERS if p["paper_id"] == paper_id),
                    "metadata": {"title": f"Paper {paper_id}"},
                }
            ],
        }
        return mock

    @pytest.fixture
    def mock_quality_filter(self):
        """Create mock QualityFilter"""
        mock = MagicMock()
        mock.score.return_value = 0.7
        return mock

    @pytest.fixture
    def generator(self, mock_ollama, mock_vectordb, mock_quality_filter):
        """Create HypothesisGenerator with mocks"""
        return HypothesisGenerator(
            ollama_client=mock_ollama,
            vectordb=mock_vectordb,
            quality_filter=mock_quality_filter,
        )

    def test_generator_init(self, generator, mock_ollama, mock_vectordb, mock_quality_filter):
        """Test HypothesisGenerator initialization"""
        assert generator.ollama is mock_ollama
        assert generator.vectordb is mock_vectordb
        assert generator.quality_filter is mock_quality_filter

    @pytest.mark.asyncio
    async def test_extract_concepts(self, generator):
        """Test concept extraction"""
        generator.ollama.generate.return_value = '["concept1", "concept2", "concept3"]'

        concepts = await generator.extract_concepts("Sample paper text")

        assert concepts == ["concept1", "concept2", "concept3"]
        generator.ollama.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_concepts_handles_markdown(self, generator):
        """Test concept extraction with markdown code blocks"""
        generator.ollama.generate.return_value = '```json\n["concept1", "concept2"]\n```'

        concepts = await generator.extract_concepts("Sample text")

        assert concepts == ["concept1", "concept2"]

    @pytest.mark.asyncio
    async def test_extract_concepts_handles_error(self, generator):
        """Test concept extraction handles invalid JSON"""
        generator.ollama.generate.return_value = "invalid json"

        concepts = await generator.extract_concepts("Sample text")

        assert concepts == []

    @pytest.mark.asyncio
    async def test_find_concept_connections(self, generator):
        """Test building concept connection graph"""
        papers = [
            {"paper_id": "p1", "text": "Paper about A and B"},
            {"paper_id": "p2", "text": "Paper about B and C"},
        ]
        generator.ollama.generate.side_effect = [
            '["concept_A", "concept_B"]',
            '["concept_B", "concept_C"]',
        ]

        graph, paper_concepts = await generator.find_concept_connections(papers)

        assert "concept_B" in graph
        assert "p1" in paper_concepts
        assert "p2" in paper_concepts

    @pytest.mark.asyncio
    async def test_find_abc_patterns(self, generator):
        """Test finding ABC patterns (undiscovered connections)"""
        concept_graph = {
            "fish_oil": [("inflammation", "paper-001")],
            "inflammation": [("fish_oil", "paper-001"), ("migraine", "paper-002")],
            "migraine": [("inflammation", "paper-002")],
        }
        paper_concepts = {
            "paper-001": ["fish_oil", "inflammation"],
            "paper-002": ["inflammation", "migraine"],
        }

        patterns = await generator.find_abc_patterns(concept_graph, paper_concepts)

        # Should find fish_oil -> migraine via inflammation
        assert len(patterns) >= 1
        found_pattern = any(
            p.concept_a in ["fish_oil", "migraine"]
            and p.concept_b in ["fish_oil", "migraine"]
            and p.bridging_concept == "inflammation"
            for p in patterns
        )
        assert found_pattern

    @pytest.mark.asyncio
    async def test_generate_hypothesis_text(self, generator):
        """Test generating hypothesis text from concept pair"""
        pair = ConceptPair(
            concept_a="fish oil",
            concept_b="migraine",
            bridging_concept="inflammation",
            paper_a_id="paper-001",
            paper_b_id="paper-002",
        )
        generator.ollama.generate.return_value = """
        {"hypothesis": "Fish oil may help treat migraine through anti-inflammatory effects.", 
         "confidence": 0.75, "reasoning": "Both linked to inflammation."}
        """

        text, confidence = await generator.generate_hypothesis_text(
            pair, "Paper 1 text", "Paper 2 text"
        )

        assert "Fish oil" in text or "fish oil" in text.lower()
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_generate_hypothesis_text_fallback(self, generator):
        """Test hypothesis generation fallback on error"""
        pair = ConceptPair(
            concept_a="A",
            concept_b="B",
            bridging_concept="C",
            paper_a_id="p1",
            paper_b_id="p2",
        )
        generator.ollama.generate.return_value = "invalid json response"

        text, confidence = await generator.generate_hypothesis_text(pair, "", "")

        assert "A" in text and "B" in text and "C" in text
        assert confidence == 0.3

    @pytest.mark.asyncio
    async def test_generate_hypotheses(self, generator):
        """Test full hypothesis generation pipeline"""
        generator.ollama.generate.side_effect = [
            # First call: concept extraction for paper 1
            '["fish_oil", "omega3", "inflammation"]',
            # Second call: concept extraction for paper 2
            '["inflammation", "migraine", "pain"]',
            # Third call: hypothesis generation
            '{"hypothesis": "Fish oil may reduce migraine via anti-inflammatory pathway.", "confidence": 0.8}',
        ]

        hypotheses = await generator.generate_hypotheses("fish oil migraine", k=5)

        # Should have called vectordb.search
        generator.vectordb.search.assert_called_once()
        # Should return hypothesis list
        assert isinstance(hypotheses, list)

    @pytest.mark.asyncio
    async def test_generate_hypotheses_no_papers(self, generator):
        """Test hypothesis generation with no papers found"""
        generator.vectordb.search.return_value = []

        hypotheses = await generator.generate_hypotheses("nonexistent topic")

        assert hypotheses == []

    @pytest.mark.asyncio
    async def test_generate_from_papers_insufficient_papers(self, generator):
        """Test generation with less than 2 papers"""
        generator.vectordb.get_paper.side_effect = None
        generator.vectordb.get_paper.return_value = None

        hypotheses = await generator.generate_from_papers(["single-paper"])

        assert hypotheses == []

    @pytest.mark.asyncio
    async def test_hypothesis_has_required_fields(self, generator):
        """Test that generated hypotheses have all required fields"""
        generator.ollama.generate.side_effect = [
            '["concept_a", "concept_b", "bridge"]',
            '["bridge", "concept_c", "other"]',
            '{"hypothesis": "A may connect to C.", "confidence": 0.7}',
        ]

        hypotheses = await generator.generate_hypotheses("test query", k=5)

        for hypo in hypotheses:
            assert hypo.id is not None
            assert hypo.hypothesis_text is not None
            assert isinstance(hypo.source_papers, list)
            assert isinstance(hypo.connection, dict)
            assert 0.0 <= hypo.confidence <= 1.0
            assert 0.0 <= hypo.quality_score <= 1.0
            assert isinstance(hypo.created_at, datetime)


class TestHypothesisGeneratorIntegration:
    """Integration tests (require Ollama)"""

    @pytest.fixture
    def real_generator(self):
        """Create generator with real dependencies for integration testing"""
        try:
            from src.llm.ollama_client import OllamaClient
            from src.storage.vectordb import VectorDB
            from src.filters.quality import QualityFilter

            ollama = OllamaClient()
            vectordb = VectorDB(persist_dir="data/db/test_chromadb")
            quality_filter = QualityFilter()

            return HypothesisGenerator(ollama, vectordb, quality_filter)
        except Exception:
            pytest.skip("Dependencies not available for integration test")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_concept_extraction(self, real_generator):
        """Test concept extraction with real Ollama (integration test)"""
        text = """
        This study investigates the relationship between omega-3 fatty acids and
        cardiovascular health. We found that fish oil supplementation reduces
        inflammation markers and improves heart function.
        """

        concepts = await real_generator.extract_concepts(text)

        assert isinstance(concepts, list)
        # Should extract relevant concepts
        if concepts:
            text_lower = " ".join(concepts).lower()
            assert any(
                term in text_lower
                for term in ["omega", "fish", "inflammation", "heart", "cardiovascular"]
            )
