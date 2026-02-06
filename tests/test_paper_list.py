"""
Tests for great papers collection and database storage.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.collectors.great_papers import GreatPapersCollector
from src.storage.database import PaperDatabase


class TestPaperDatabase:
    """Tests for PaperDatabase class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_database_init(self, temp_db):
        """Test database initialization and schema creation."""
        db = PaperDatabase(temp_db)
        assert db.db_path == Path(temp_db)

        # Check that tables were created
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='great_papers'")
        assert cursor.fetchone() is not None
        db.close()

    def test_insert_paper(self, temp_db):
        """Test inserting a single paper."""
        with PaperDatabase(temp_db) as db:
            paper = {
                "title": "Test Paper",
                "authors": "John Doe, Jane Smith",
                "year": 2020,
                "arxiv_id": "2001.12345",
                "doi": "10.1234/test",
                "citations": 100,
                "source": "test",
                "s2_paper_id": "s2_test_id",
                "abstract": "This is a test abstract.",
                "venue": "Test Conference",
            }

            row_id = db.insert_paper(paper)
            assert row_id is not None
            assert row_id > 0

    def test_insert_duplicate_paper(self, temp_db):
        """Test that duplicate papers (same title + year) are rejected."""
        with PaperDatabase(temp_db) as db:
            paper = {
                "title": "Test Paper",
                "authors": "John Doe",
                "year": 2020,
                "source": "test",
            }

            # First insert should succeed
            row_id1 = db.insert_paper(paper)
            assert row_id1 is not None

            # Second insert should fail (duplicate)
            row_id2 = db.insert_paper(paper)
            assert row_id2 is None

    def test_insert_papers_batch(self, temp_db):
        """Test batch insertion of papers."""
        with PaperDatabase(temp_db) as db:
            papers = [
                {"title": "Paper 1", "year": 2020, "source": "test"},
                {"title": "Paper 2", "year": 2021, "source": "test"},
                {"title": "Paper 3", "year": 2022, "source": "test"},
            ]

            inserted, skipped = db.insert_papers_batch(papers)
            assert inserted == 3
            assert skipped == 0

    def test_get_paper_by_arxiv_id(self, temp_db):
        """Test retrieving a paper by arXiv ID."""
        with PaperDatabase(temp_db) as db:
            paper = {
                "title": "Test Paper",
                "arxiv_id": "2001.12345",
                "year": 2020,
                "source": "test",
            }
            db.insert_paper(paper)

            retrieved = db.get_paper_by_arxiv_id("2001.12345")
            assert retrieved is not None
            assert retrieved["title"] == "Test Paper"
            assert retrieved["arxiv_id"] == "2001.12345"

    def test_get_papers_by_source(self, temp_db):
        """Test retrieving papers by source."""
        with PaperDatabase(temp_db) as db:
            papers = [
                {"title": "Paper 1", "year": 2020, "source": "seminal"},
                {"title": "Paper 2", "year": 2021, "source": "seminal"},
                {"title": "Paper 3", "year": 2022, "source": "nobel"},
            ]
            db.insert_papers_batch(papers)

            seminal_papers = db.get_papers_by_source("seminal")
            assert len(seminal_papers) == 2

            nobel_papers = db.get_papers_by_source("nobel")
            assert len(nobel_papers) == 1

    def test_count_papers(self, temp_db):
        """Test counting papers."""
        with PaperDatabase(temp_db) as db:
            papers = [
                {"title": "Paper 1", "year": 2020, "source": "seminal"},
                {"title": "Paper 2", "year": 2021, "source": "seminal"},
                {"title": "Paper 3", "year": 2022, "source": "nobel"},
            ]
            db.insert_papers_batch(papers)

            assert db.count_papers() == 3
            assert db.count_papers("seminal") == 2
            assert db.count_papers("nobel") == 1

    def test_update_citations(self, temp_db):
        """Test updating citation count."""
        with PaperDatabase(temp_db) as db:
            paper = {"title": "Test Paper", "year": 2020, "source": "test", "citations": 100}
            paper_id = db.insert_paper(paper)

            db.update_paper_citations(paper_id, 200)

            papers = db.get_all_papers()
            assert papers[0]["citations"] == 200


class TestGreatPapersCollector:
    """Tests for GreatPapersCollector class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_collect_seminal_papers(self):
        """Test collecting seminal papers list."""
        collector = GreatPapersCollector()
        papers = collector.collect_seminal_papers()

        # Should have 100+ papers
        assert len(papers) >= 100

        # All papers should have required fields
        for paper in papers:
            assert "title" in paper
            assert "source" in paper
            assert paper["source"] == "seminal"
            assert "year" in paper or "arxiv_id" in paper

    def test_collect_seminal_papers_content(self):
        """Test that seminal papers list contains expected papers."""
        collector = GreatPapersCollector()
        papers = collector.collect_seminal_papers()

        titles = [p["title"] for p in papers]

        # Check for some well-known papers
        assert "Attention Is All You Need" in titles
        assert "Generative Adversarial Networks" in titles
        assert "Deep Residual Learning for Image Recognition" in titles

    def test_save_to_db(self, temp_db):
        """Test saving papers to database."""
        collector = GreatPapersCollector()
        papers = collector.collect_seminal_papers()[:10]  # Only test with 10 papers

        inserted, skipped = collector.save_to_db(papers, temp_db)

        assert inserted == 10
        assert skipped == 0

        # Verify papers are in database
        with PaperDatabase(temp_db) as db:
            assert db.count_papers() == 10

    def test_collect_and_save_without_enrichment(self, temp_db):
        """Test full collection and save without enrichment."""
        collector = GreatPapersCollector()

        # Collect without enrichment (faster for testing)
        summary = collector.collect_and_save(temp_db, enrich=False)

        assert summary["total_papers"] >= 100
        assert summary["inserted"] >= 100
        assert summary["enriched"] == 0
        assert summary["db_path"] == str(temp_db)

        # Verify database
        with PaperDatabase(temp_db) as db:
            assert db.count_papers() >= 100
            assert db.count_papers("seminal") >= 100


# Integration test (skipped by default due to API rate limits)
@pytest.mark.skip(reason="Requires API access and is slow")
def test_enrich_with_citations():
    """Test enriching papers with Semantic Scholar data.

    This test is skipped by default because it:
    1. Requires Semantic Scholar API access
    2. Takes a long time due to rate limiting
    3. May fail if API is unavailable

    To run: pytest tests/test_paper_list.py::test_enrich_with_citations -v
    """
    collector = GreatPapersCollector()

    # Test with just a few papers
    papers = [
        {
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762",
            "year": 2017,
            "source": "test",
        },
        {
            "title": "Generative Adversarial Networks",
            "arxiv_id": "1406.2661",
            "year": 2014,
            "source": "test",
        },
    ]

    enriched = collector.enrich_with_citations(papers, delay=1.0, max_papers=2)

    assert len(enriched) == 2

    # Both papers should have citations
    for paper in enriched:
        assert paper.get("citations", 0) > 0
        assert paper.get("s2_paper_id") is not None
