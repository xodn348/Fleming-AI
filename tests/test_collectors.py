"""
Tests for API clients (arXiv, Semantic Scholar, OpenAlex).
"""

import os
from unittest.mock import Mock, patch

import pytest

from src.collectors.arxiv_client import ArxivClient
from src.collectors.openalex_client import OpenAlexClient
from src.collectors.semantic_scholar_client import SemanticScholarClient


class TestArxivClient:
    """Tests for ArxivClient."""

    def test_init(self):
        """Test client initialization."""
        client = ArxivClient()
        assert client.BASE_URL == "http://export.arxiv.org/api/query"
        assert client.RATE_LIMIT_DELAY == 3
        client.close()

    def test_context_manager(self):
        """Test context manager usage."""
        with ArxivClient() as client:
            assert client is not None

    @pytest.mark.skipif(os.getenv("SKIP_LIVE_TESTS") == "1", reason="Skipping live API tests")
    def test_search_live(self):
        """Test search with live API (skipped by default)."""
        with ArxivClient() as client:
            results = client.search("cat:cs.AI", max_results=2)
            assert isinstance(results, list)
            assert len(results) <= 2
            if results:
                paper = results[0]
                assert "id" in paper
                assert "title" in paper
                assert "authors" in paper

    @pytest.mark.skipif(os.getenv("SKIP_LIVE_TESTS") == "1", reason="Skipping live API tests")
    def test_get_paper_live(self):
        """Test get_paper with live API (skipped by default)."""
        with ArxivClient() as client:
            # Use a well-known paper ID
            paper = client.get_paper("1706.03762")  # "Attention is All You Need"
            assert paper is not None
            assert "title" in paper
            assert "authors" in paper

    def test_parse_entry(self):
        """Test XML entry parsing."""
        import xml.etree.ElementTree as ET

        xml_str = """
        <entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
            <id>http://arxiv.org/abs/1234.5678v1</id>
            <title>Test Paper</title>
            <summary>Test abstract</summary>
            <published>2023-01-01T00:00:00Z</published>
            <updated>2023-01-02T00:00:00Z</updated>
            <author><name>John Doe</name></author>
            <author><name>Jane Smith</name></author>
            <category term="cs.AI"/>
            <category term="cs.LG"/>
            <arxiv:primary_category term="cs.AI"/>
            <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1"/>
        </entry>
        """

        entry = ET.fromstring(xml_str)
        client = ArxivClient()
        paper = client._parse_entry(entry)
        client.close()

        assert paper["id"] == "http://arxiv.org/abs/1234.5678v1"
        assert paper["title"] == "Test Paper"
        assert paper["summary"] == "Test abstract"
        assert paper["authors"] == ["John Doe", "Jane Smith"]
        assert paper["categories"] == ["cs.AI", "cs.LG"]
        assert paper["primary_category"] == "cs.AI"
        assert paper["pdf_url"] == "http://arxiv.org/pdf/1234.5678v1"


class TestSemanticScholarClient:
    """Tests for SemanticScholarClient."""

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = SemanticScholarClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert "x-api-key" in client.client.headers
        client.close()

    def test_init_from_env(self):
        """Test client initialization from environment variable."""
        with patch.dict(os.environ, {"S2_API_KEY": "env_key"}):
            client = SemanticScholarClient()
            assert client.api_key == "env_key"
            client.close()

    def test_context_manager(self):
        """Test context manager usage."""
        with SemanticScholarClient() as client:
            assert client is not None

    @pytest.mark.skipif(
        not os.getenv("S2_API_KEY") or os.getenv("SKIP_LIVE_TESTS") == "1",
        reason="S2_API_KEY not set or skipping live tests",
    )
    def test_get_paper_live(self):
        """Test get_paper with live API (requires API key)."""
        with SemanticScholarClient() as client:
            # Use a well-known paper ID
            paper = client.get_paper("649def34f8be52c8b66281af98ae884c09aef38b")
            assert paper is not None
            assert "paperId" in paper
            assert "title" in paper

    @pytest.mark.skipif(
        not os.getenv("S2_API_KEY") or os.getenv("SKIP_LIVE_TESTS") == "1",
        reason="S2_API_KEY not set or skipping live tests",
    )
    def test_search_live(self):
        """Test search with live API (requires API key)."""
        with SemanticScholarClient() as client:
            results = client.search("machine learning", limit=2)
            assert "data" in results
            assert isinstance(results["data"], list)
            assert len(results["data"]) <= 2


class TestOpenAlexClient:
    """Tests for OpenAlexClient."""

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = OpenAlexClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert "Authorization" in client.client.headers
        client.close()

    def test_init_with_email(self):
        """Test client initialization with email."""
        client = OpenAlexClient(email="test@example.com")
        assert client.email == "test@example.com"
        assert "mailto" in client.client.params
        client.close()

    def test_init_requires_auth(self):
        """Test that initialization requires either API key or email."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing env vars
            if "OPENALEX_API_KEY" in os.environ:
                del os.environ["OPENALEX_API_KEY"]
            if "OPENALEX_EMAIL" in os.environ:
                del os.environ["OPENALEX_EMAIL"]

            with pytest.raises(ValueError, match="requires either an API key"):
                OpenAlexClient()

    def test_context_manager(self):
        """Test context manager usage."""
        with OpenAlexClient(email="test@example.com") as client:
            assert client is not None

    @pytest.mark.skipif(
        (not os.getenv("OPENALEX_API_KEY") and not os.getenv("OPENALEX_EMAIL"))
        or os.getenv("SKIP_LIVE_TESTS") == "1",
        reason="OPENALEX_API_KEY or OPENALEX_EMAIL not set or skipping live tests",
    )
    def test_get_work_live(self):
        """Test get_work with live API (requires API key or email)."""
        api_key = os.getenv("OPENALEX_API_KEY")
        email = os.getenv("OPENALEX_EMAIL")

        with OpenAlexClient(api_key=api_key, email=email) as client:
            # Use a well-known work ID
            work = client.get_work("W2741809807")
            assert work is not None
            assert "id" in work
            assert "title" in work or "display_name" in work

    @pytest.mark.skipif(
        (not os.getenv("OPENALEX_API_KEY") and not os.getenv("OPENALEX_EMAIL"))
        or os.getenv("SKIP_LIVE_TESTS") == "1",
        reason="OPENALEX_API_KEY or OPENALEX_EMAIL not set or skipping live tests",
    )
    def test_search_live(self):
        """Test search with live API (requires API key or email)."""
        api_key = os.getenv("OPENALEX_API_KEY")
        email = os.getenv("OPENALEX_EMAIL")

        with OpenAlexClient(api_key=api_key, email=email) as client:
            results = client.search(query="machine learning", per_page=2)
            assert "results" in results
            assert isinstance(results["results"], list)
            assert len(results["results"]) <= 2


# Integration test that can run without API keys
def test_all_clients_can_initialize():
    """Test that all clients can be initialized (without making API calls)."""
    # ArxivClient doesn't need API key
    arxiv = ArxivClient()
    arxiv.close()

    # SemanticScholarClient can work without API key (lower rate limits)
    s2 = SemanticScholarClient()
    s2.close()

    # OpenAlexClient needs either API key or email
    with pytest.raises(ValueError):
        with patch.dict(os.environ, {}, clear=True):
            if "OPENALEX_API_KEY" in os.environ:
                del os.environ["OPENALEX_API_KEY"]
            if "OPENALEX_EMAIL" in os.environ:
                del os.environ["OPENALEX_EMAIL"]
            OpenAlexClient()
