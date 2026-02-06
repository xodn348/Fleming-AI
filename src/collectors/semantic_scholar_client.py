"""
Semantic Scholar API client for fetching research papers and citations.
"""

import os
import time
from typing import Any, Dict, List, Optional

import httpx


class SemanticScholarClient:
    """Client for interacting with the Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    RATE_LIMIT_DELAY = 3  # seconds

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Semantic Scholar client.

        Args:
            api_key: Optional API key. If not provided, will try to read from S2_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("S2_API_KEY")

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        self.client = httpx.Client(timeout=30.0, headers=headers)
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def get_paper(
        self, paper_id: str, fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a specific paper by ID.

        Args:
            paper_id: Paper ID (can be S2 ID, DOI, arXiv ID, etc.)
            fields: List of fields to return. If None, returns default fields.
                   Available fields: paperId, externalIds, url, title, abstract, venue,
                   year, referenceCount, citationCount, influentialCitationCount,
                   isOpenAccess, fieldsOfStudy, authors, citations, references

        Returns:
            Paper dictionary or None if not found
        """
        self._rate_limit()

        if fields is None:
            fields = [
                "paperId",
                "externalIds",
                "url",
                "title",
                "abstract",
                "venue",
                "year",
                "referenceCount",
                "citationCount",
                "influentialCitationCount",
                "isOpenAccess",
                "fieldsOfStudy",
                "authors",
            ]

        params = {"fields": ",".join(fields)}
        url = f"{self.BASE_URL}/paper/{paper_id}"

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def get_citations(
        self, paper_id: str, limit: int = 100, offset: int = 0, fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get citations for a paper.

        Args:
            paper_id: Paper ID
            limit: Maximum number of citations to return (max 1000)
            offset: Offset for pagination
            fields: List of fields to return for each citation

        Returns:
            Dictionary with 'data' (list of citations) and 'next' (offset for next page)
        """
        self._rate_limit()

        if fields is None:
            fields = [
                "paperId",
                "title",
                "year",
                "authors",
                "citationCount",
                "influentialCitationCount",
                "isOpenAccess",
            ]

        params = {"fields": ",".join(fields), "limit": min(limit, 1000), "offset": offset}

        url = f"{self.BASE_URL}/paper/{paper_id}/citations"
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        fields: Optional[List[str]] = None,
        year: Optional[str] = None,
        venue: Optional[List[str]] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search for papers.

        Args:
            query: Search query string
            limit: Maximum number of results to return (max 100)
            offset: Offset for pagination
            fields: List of fields to return for each paper
            year: Year or year range (e.g., "2020" or "2019-2021")
            venue: List of venue names to filter by
            fields_of_study: List of fields of study to filter by

        Returns:
            Dictionary with 'data' (list of papers), 'total' (total results),
            and 'next' (offset for next page)
        """
        self._rate_limit()

        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "year",
                "authors",
                "venue",
                "citationCount",
                "influentialCitationCount",
                "isOpenAccess",
            ]

        params = {
            "query": query,
            "fields": ",".join(fields),
            "limit": min(limit, 100),
            "offset": offset,
        }

        if year:
            params["year"] = year
        if venue:
            params["venue"] = ",".join(venue)
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        url = f"{self.BASE_URL}/paper/search"
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
