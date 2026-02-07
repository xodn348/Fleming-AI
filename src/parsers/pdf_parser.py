"""
PDF parser for extracting text and sections from research papers.
"""

import re
from pathlib import Path
from typing import Dict

import pymupdf  # PyMuPDF


class PDFParser:
    """Parse PDF files and extract structured content from research papers."""

    # Common section headers in research papers
    SECTION_PATTERNS = {
        "abstract": re.compile(r"\bABSTRACT\b", re.IGNORECASE),
        "introduction": re.compile(r"\b(?:1\.?\s*)?INTRODUCTION\b", re.IGNORECASE),
        "method": re.compile(
            r"\b(?:\d+\.?\s*)?(?:METHOD|METHODOLOGY|APPROACH|MODEL)\b", re.IGNORECASE
        ),
        "results": re.compile(
            r"\b(?:\d+\.?\s*)?(?:RESULTS?|EXPERIMENTS?|EVALUATION)\b", re.IGNORECASE
        ),
        "conclusion": re.compile(
            r"\b(?:\d+\.?\s*)?(?:CONCLUSION|DISCUSSION|SUMMARY)\b", re.IGNORECASE
        ),
        "references": re.compile(r"\b(?:REFERENCES|BIBLIOGRAPHY)\b", re.IGNORECASE),
    }

    def parse(self, pdf_path: str | Path) -> Dict[str, str]:
        """Parse a PDF file and extract text content.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with keys: title, abstract, introduction, method,
            results, conclusion, full_text, references
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Extract full text
        full_text = self._extract_text(pdf_path)

        # Extract title (first non-empty line, often in larger font)
        title = self._extract_title(pdf_path)

        # Extract sections
        sections = self.extract_sections(full_text)

        return {
            "title": title,
            "abstract": sections.get("abstract", ""),
            "introduction": sections.get("introduction", ""),
            "method": sections.get("method", ""),
            "results": sections.get("results", ""),
            "conclusion": sections.get("conclusion", ""),
            "references": sections.get("references", ""),
            "full_text": full_text,
        }

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract all text from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Full text content
        """
        doc = pymupdf.open(str(pdf_path))
        text_parts = []

        for page in doc:
            text_parts.append(page.get_text())

        doc.close()

        return "\n".join(text_parts)

    def _extract_title(self, pdf_path: Path) -> str:
        """Extract title from PDF (first page, largest text).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Title text
        """
        doc = pymupdf.open(str(pdf_path))

        if len(doc) == 0:
            doc.close()
            return ""

        # Get first page text
        first_page = doc[0]
        text = first_page.get_text()
        doc.close()

        # Try to get first non-empty line as title
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            # Skip arXiv identifier if present
            for line in lines:
                if not re.match(r"^arXiv:", line, re.IGNORECASE):
                    return line

        return lines[0] if lines else ""

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from paper text.

        Args:
            text: Full text content

        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}

        # Find section boundaries
        section_positions = []
        for section_name, pattern in self.SECTION_PATTERNS.items():
            match = pattern.search(text)
            if match:
                section_positions.append((match.start(), section_name))

        # Sort by position
        section_positions.sort()

        # Extract content between section headers
        for i, (start_pos, section_name) in enumerate(section_positions):
            # Determine end position (next section or end of text)
            if i + 1 < len(section_positions):
                end_pos = section_positions[i + 1][0]
            else:
                end_pos = len(text)

            # Extract section content
            section_content = text[start_pos:end_pos].strip()

            # Remove section header from content
            lines = section_content.split("\n")
            if lines:
                # Skip first line (header)
                section_content = "\n".join(lines[1:]).strip()

            sections[section_name] = section_content

        return sections
