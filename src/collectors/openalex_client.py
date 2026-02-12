"""
OpenAlex API client for fetching research papers and metadata.
"""

import os
from typing import Any, Dict, List, Optional

import httpx


class OpenAlexClient:
    """Client for interacting with the OpenAlex API."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """Initialize the OpenAlex client.

        Args:
            api_key: Optional API key. If not provided, will try to read from OPENALEX_API_KEY env var.
            email: Optional email for polite pool. If not provided, will try to read from OPENALEX_EMAIL env var.

        Note:
            OpenAlex requires either an API key or email for polite pool access.
            Without either, you'll be in the common pool with lower rate limits.
        """
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY")
        self.email = email or os.getenv("OPENALEX_EMAIL")

        if not self.api_key and not self.email:
            raise ValueError(
                "OpenAlex requires either an API key (OPENALEX_API_KEY) or "
                "email (OPENALEX_EMAIL) for API access. Please provide one."
            )

        headers = {"User-Agent": "Fleming-AI/0.1.0"}
        params = {}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.email:
            params["mailto"] = self.email

        self.client = httpx.Client(timeout=30.0, headers=headers, params=params)

    def get_work(
        self, work_id: str, select: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a specific work (paper) by ID.

        Args:
            work_id: Work ID (can be OpenAlex ID, DOI, PMID, etc.)
                    Examples: "W2741809807", "https://doi.org/10.1038/nature12373"
            select: List of fields to return. If None, returns all fields.
                   Available fields: id, doi, title, display_name, publication_year,
                   publication_date, type, cited_by_count, is_retracted, is_paratext,
                   cited_by_api_url, abstract_inverted_index, authorships, etc.

        Returns:
            Work dictionary or None if not found
        """
        # Normalize work_id to OpenAlex format if it's a URL
        if work_id.startswith("http"):
            # Extract the ID from URL
            if "doi.org" in work_id:
                work_id = f"https://doi.org/{work_id.split('doi.org/')[-1]}"
        elif not work_id.startswith("W") and not work_id.startswith("http"):
            # Assume it's a DOI without https://doi.org/ prefix
            work_id = f"https://doi.org/{work_id}"

        url = f"{self.BASE_URL}/works/{work_id}"
        params: Dict[str, Any] = {}

        if select:
            params["select"] = ",".join(select)

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def search(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort: Optional[str] = None,
        per_page: int = 25,
        page: Optional[int] = None,
        cursor: Optional[str] = None,
        select: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search for works.

        Args:
            query: Search query string (searches across title, abstract, fulltext)
            filters: Dictionary of filters to apply
                    Examples: {"publication_year": 2020, "type": "article"}
                    Available filters: publication_year, cited_by_count, is_oa,
                    type, authorships.author.id, concepts.id, etc.
            sort: Sort criterion (e.g., "cited_by_count:desc", "publication_date:asc")
            per_page: Number of results per page (max 200)
            page: Page number (1-indexed, only works up to 10k results)
            cursor: Cursor for pagination (use instead of page for >10k results)
            select: List of fields to return for each work

        Returns:
            Dictionary with 'results' (list of works), 'meta' (pagination info including next_cursor)
        """
        url = f"{self.BASE_URL}/works"
        params: Dict[str, Any] = {"per-page": min(per_page, 200)}

        if cursor is not None:
            params["cursor"] = cursor
        elif page is not None:
            params["page"] = page
        else:
            params["page"] = 1

        # Build filter string
        filter_parts = []
        if query:
            params["search"] = query

        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    # Multiple values for same filter (OR)
                    filter_parts.append(f"{key}:{'|'.join(str(v) for v in value)}")
                else:
                    filter_parts.append(f"{key}:{value}")

        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        if sort:
            params["sort"] = sort

        if select:
            params["select"] = ",".join(select)

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
