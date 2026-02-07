"""
Tests for VectorDB module.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from storage.vectordb import VectorDB


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    db = VectorDB(persist_dir=str(tmp_path / "test_chromadb"))
    yield db
    # Cleanup
    db.reset()


@pytest.fixture
def sample_papers():
    """Sample paper data for testing."""
    return [
        {
            "paper_id": "test_paper_1",
            "title": "Deep Learning for Image Recognition",
            "abstract": "This paper presents a deep learning approach for image recognition using convolutional neural networks.",
            "introduction": "Deep learning has revolutionized computer vision. CNNs are particularly effective for image classification tasks.",
            "method": "We use a VGG-style architecture with multiple convolutional layers and max pooling.",
            "results": "Our model achieves 95% accuracy on the ImageNet dataset.",
            "conclusion": "Deep learning is highly effective for image recognition tasks.",
        },
        {
            "paper_id": "test_paper_2",
            "title": "Attention Mechanisms in Neural Networks",
            "abstract": "We explore attention mechanisms for improving neural network performance on sequence tasks.",
            "introduction": "Attention allows models to focus on relevant parts of the input.",
            "method": "We implement self-attention and multi-head attention mechanisms.",
            "results": "Attention improves performance by 10% on machine translation tasks.",
            "conclusion": "Attention is a powerful technique for sequence modeling.",
        },
    ]


def test_vectordb_initialization(test_db):
    """Test VectorDB initialization."""
    assert test_db is not None
    assert test_db.collection is not None
    assert test_db.count() == 0


def test_add_single_paper(test_db):
    """Test adding a single paper chunk."""
    test_db.add_paper(
        paper_id="test_1",
        text="This is a test paper about machine learning.",
        metadata={"section": "abstract", "title": "Test Paper"},
    )

    assert test_db.count() == 1


def test_add_multiple_papers(test_db, sample_papers):
    """Test adding multiple papers."""
    chunks_added = test_db.add_papers(sample_papers)

    assert chunks_added > 0
    assert test_db.count() == chunks_added
    # Each paper should have at least abstract + some sections
    assert chunks_added >= len(sample_papers)


def test_search_functionality(test_db, sample_papers):
    """Test semantic search."""
    # Add papers
    test_db.add_papers(sample_papers)

    # Search for deep learning related content
    results = test_db.search("convolutional neural networks for images", k=3)

    assert len(results) > 0
    assert len(results) <= 3

    # Check result structure
    first_result = results[0]
    assert "id" in first_result
    assert "text" in first_result
    assert "metadata" in first_result
    assert "paper_id" in first_result["metadata"]

    # The first result should be from the deep learning paper
    assert "deep learning" in first_result["text"].lower() or "cnn" in first_result["text"].lower()


def test_search_attention_paper(test_db, sample_papers):
    """Test search returns correct paper for attention query."""
    test_db.add_papers(sample_papers)

    results = test_db.search("attention mechanisms for sequences", k=2)

    assert len(results) > 0
    # Should find the attention paper
    found_attention = any("attention" in r["text"].lower() for r in results)
    assert found_attention


def test_get_paper(test_db, sample_papers):
    """Test retrieving all chunks for a specific paper."""
    test_db.add_papers(sample_papers)

    paper_data = test_db.get_paper("test_paper_1")

    assert paper_data["paper_id"] == "test_paper_1"
    assert len(paper_data["chunks"]) > 0

    # Check chunk structure
    chunk = paper_data["chunks"][0]
    assert "id" in chunk
    assert "text" in chunk
    assert "metadata" in chunk


def test_chunking_long_sections(test_db):
    """Test that long sections are properly chunked."""
    # Create text with paragraph breaks that will be split
    long_text = "\n\n".join(
        ["This is a test paragraph. " * 20 for _ in range(5)]
    )  # Create text > 1000 chars with breaks

    paper = {
        "paper_id": "long_paper",
        "title": "Long Paper",
        "abstract": "Short abstract",
        "introduction": long_text,
    }

    chunks_added = test_db.add_papers([paper])

    # Should create multiple chunks for the long introduction
    assert chunks_added > 2  # abstract + multiple introduction chunks


def test_empty_search(test_db):
    """Test search on empty database."""
    results = test_db.search("test query", k=5)

    assert len(results) == 0


def test_count(test_db, sample_papers):
    """Test count functionality."""
    assert test_db.count() == 0

    test_db.add_papers(sample_papers)

    count = test_db.count()
    assert count > 0


def test_reset(test_db, sample_papers):
    """Test database reset."""
    test_db.add_papers(sample_papers)
    assert test_db.count() > 0

    test_db.reset()
    assert test_db.count() == 0


def test_metadata_preservation(test_db):
    """Test that metadata is preserved correctly."""
    test_db.add_paper(
        paper_id="meta_test",
        text="Test content",
        metadata={
            "section": "introduction",
            "title": "Metadata Test",
            "custom_field": "custom_value",
        },
    )

    results = test_db.search("test content", k=1)

    assert len(results) == 1
    metadata = results[0]["metadata"]
    assert metadata["section"] == "introduction"
    assert metadata["title"] == "Metadata Test"
    assert metadata["custom_field"] == "custom_value"
