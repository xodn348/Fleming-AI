"""
Tests for PDF downloading and parsing.
"""

import asyncio
from pathlib import Path

import pytest

from src.collectors.pdf_downloader import PDFDownloader
from src.parsers.pdf_parser import PDFParser


class TestPDFDownloader:
    """Test PDF downloading functionality."""

    @pytest.fixture
    def downloader(self, tmp_path):
        """Create a PDFDownloader instance with temp directory."""
        return PDFDownloader(output_dir=tmp_path / "papers")

    @pytest.mark.asyncio
    async def test_download_single_paper(self, downloader, tmp_path):
        """Test downloading a single paper."""
        # Use a well-known arXiv paper (Attention Is All You Need)
        arxiv_id = "1706.03762"

        pdf_path = await downloader.download_paper(arxiv_id)

        assert Path(pdf_path).exists()
        assert Path(pdf_path).suffix == ".pdf"
        assert arxiv_id in pdf_path

    @pytest.mark.asyncio
    async def test_download_already_exists(self, downloader, tmp_path):
        """Test that re-downloading skips existing files."""
        arxiv_id = "1706.03762"

        # First download
        pdf_path1 = await downloader.download_paper(arxiv_id)

        # Second download (should skip)
        pdf_path2 = await downloader.download_paper(arxiv_id)

        assert pdf_path1 == pdf_path2
        assert Path(pdf_path1).exists()

    @pytest.mark.asyncio
    async def test_batch_download(self, downloader, tmp_path):
        """Test batch downloading multiple papers."""
        arxiv_ids = ["1706.03762", "1810.04805"]  # Attention, BERT

        results = await downloader.batch_download(arxiv_ids)

        assert len(results) == 2
        for result in results:
            assert result["status"] == "success"
            assert result["path"] is not None
            assert Path(result["path"]).exists()

    @pytest.mark.asyncio
    async def test_download_invalid_arxiv_id(self, downloader, tmp_path):
        """Test handling of invalid arXiv ID."""
        arxiv_id = "invalid.id.9999"

        results = await downloader.batch_download([arxiv_id])

        assert results[0]["status"] == "failed"
        assert results[0]["error"] is not None


class TestPDFParser:
    """Test PDF parsing functionality."""

    @pytest.fixture
    async def sample_pdf(self, tmp_path):
        """Download a sample PDF for testing."""
        downloader = PDFDownloader(output_dir=tmp_path / "papers")
        pdf_path = await downloader.download_paper("1706.03762")
        return pdf_path

    def test_parse_pdf(self, sample_pdf):
        """Test parsing a PDF file."""
        parser = PDFParser()
        result = parser.parse(sample_pdf)

        # Check all expected keys are present
        expected_keys = [
            "title",
            "abstract",
            "introduction",
            "method",
            "results",
            "conclusion",
            "full_text",
            "references",
        ]
        for key in expected_keys:
            assert key in result

        # Check that we extracted some content
        assert len(result["full_text"]) > 0
        assert len(result["title"]) > 0

    def test_parse_nonexistent_pdf(self):
        """Test handling of nonexistent PDF."""
        parser = PDFParser()

        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.pdf")

    def test_extract_sections(self):
        """Test section extraction from text."""
        parser = PDFParser()

        sample_text = """
        ABSTRACT
        This is the abstract.
        
        INTRODUCTION
        This is the introduction.
        
        METHOD
        This is the method section.
        
        RESULTS
        These are the results.
        
        CONCLUSION
        This is the conclusion.
        
        REFERENCES
        [1] Reference 1
        """

        sections = parser.extract_sections(sample_text)

        assert "abstract" in sections
        assert "introduction" in sections
        assert "method" in sections
        assert "results" in sections
        assert "conclusion" in sections
        assert "references" in sections

        # Check content is extracted (excluding headers)
        assert "This is the abstract" in sections["abstract"]
        assert "This is the introduction" in sections["introduction"]
