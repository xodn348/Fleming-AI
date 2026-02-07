"""
ArXiv API client for fetching research papers.
"""

import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx


class ArxivClient:
    """Client for interacting with the arXiv API."""

    BASE_URL = "https://export.arxiv.org/api/query"
    RATE_LIMIT_DELAY = 3  # seconds

    def __init__(self):
        """Initialize the ArXiv client."""
        self.client = httpx.Client(timeout=30.0, follow_redirects=True)
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _parse_entry(self, entry: ET.Element) -> Dict[str, Any]:
        """Parse an arXiv entry XML element into a dictionary.

        Args:
            entry: XML element representing a paper entry

        Returns:
            Dictionary containing paper metadata
        """
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

        # Extract basic fields
        id_elem = entry.find("atom:id", ns)
        paper_id = id_elem.text if id_elem is not None else None

        title_elem = entry.find("atom:title", ns)
        title = title_elem.text if title_elem is not None else None

        summary_elem = entry.find("atom:summary", ns)
        summary = summary_elem.text if summary_elem is not None else None

        published_elem = entry.find("atom:published", ns)
        published = published_elem.text if published_elem is not None else None

        updated_elem = entry.find("atom:updated", ns)
        updated = updated_elem.text if updated_elem is not None else None

        # Extract authors
        authors = []
        for author in entry.findall("atom:author", ns):
            name_elem = author.find("atom:name", ns)
            if name_elem is not None:
                authors.append(name_elem.text)

        # Extract categories
        categories = []
        for category in entry.findall("atom:category", ns):
            term = category.get("term")
            if term:
                categories.append(term)

        # Extract primary category
        primary_category = None
        primary_cat_elem = entry.find("arxiv:primary_category", ns)
        if primary_cat_elem is not None:
            primary_category = primary_cat_elem.get("term")

        # Extract PDF link
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break

        return {
            "id": paper_id,
            "title": title.strip() if title else None,
            "summary": summary.strip() if summary else None,
            "authors": authors,
            "published": published,
            "updated": updated,
            "categories": categories,
            "primary_category": primary_category,
            "pdf_url": pdf_url,
        }

    def search(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> List[Dict[str, Any]]:
        """Search for papers on arXiv.

        Args:
            query: Search query string (e.g., "cat:cs.AI AND ti:transformer")
            max_results: Maximum number of results to return
            start: Starting index for pagination
            sort_by: Sort criterion ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: Sort order ("ascending" or "descending")

        Returns:
            List of paper dictionaries
        """
        self._rate_limit()

        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        url = f"{self.BASE_URL}?{urlencode(params)}"
        response = self.client.get(url)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entries = root.findall("atom:entry", ns)
        return [self._parse_entry(entry) for entry in entries]

    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific paper by arXiv ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.12345" or "cs/0001234")

        Returns:
            Paper dictionary or None if not found
        """
        self._rate_limit()

        params = {
            "id_list": arxiv_id,
        }

        url = f"{self.BASE_URL}?{urlencode(params)}"
        response = self.client.get(url)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        return self._parse_entry(entry)

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
