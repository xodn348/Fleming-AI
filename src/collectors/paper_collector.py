"""
Automated paper collector for discovering and storing high-quality research papers.
"""

import asyncio
import hashlib
from typing import Any, Dict, List, Optional

from src.collectors.openalex_client import OpenAlexClient
from src.collectors.semantic_scholar_client import SemanticScholarClient
from src.storage.database import PaperDatabase
from src.utils.scoring import VENUE_TIERS, calculate_quality_score


class PaperCollector:
    """Automated collector for discovering and storing high-quality research papers."""

    RATE_LIMIT_DELAY = 3  # seconds between API requests

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the paper collector.

        Args:
            config: Optional configuration dictionary with keys:
                - db_path: Path to SQLite database (default: "data/db/papers.db")
                - min_citations: Minimum citation count for discovery (default: 100)
                - quality_threshold: Minimum quality score for filtering (default: 60)
        """
        self.config = config or {}
        self.openalex = OpenAlexClient()
        self.s2 = SemanticScholarClient()

        db_path = self.config.get("db_path", "data/db/papers.db")
        self.db = PaperDatabase(db_path)

        self.min_citations = self.config.get("min_citations", 100)
        self.quality_threshold = self.config.get("quality_threshold", 60)

    async def discover(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Discover papers using OpenAlex API.

        Searches for papers with:
        - Citation count >= min_citations (default: 100)
        - Recent papers (last 7 years: 2019-2026)
        - Type: article (peer-reviewed)

        Quality filtering by venue tier happens in filter() step.

        Args:
            limit: Maximum number of papers to discover

        Returns:
            List of paper dictionaries with OpenAlex metadata
        """
        discovered_papers = []

        try:
            # Search with filters for high-quality papers
            response = self.openalex.search(
                filters={
                    "cited_by_count": f">{self.min_citations}",
                    "publication_year": "2019-2026",  # Last 7 years
                    "type": "article",
                },
                sort="cited_by_count:desc",
                per_page=min(limit, 200),  # OpenAlex max per_page is 200
                select=[
                    "id",
                    "doi",
                    "title",
                    "display_name",
                    "publication_year",
                    "cited_by_count",
                    "authorships",
                    "primary_location",
                    "abstract_inverted_index",
                ],
            )

            # Extract papers from response
            results = response.get("results", [])
            for work in results:
                # Extract venue
                primary_location = work.get("primary_location") or {}
                source = primary_location.get("source") or {}
                work_venue = source.get("display_name", "")

                # Extract paper metadata
                paper = {
                    "title": work.get("display_name") or work.get("title"),
                    "year": work.get("publication_year"),
                    "citations": work.get("cited_by_count", 0),
                    "venue": work_venue,
                    "doi": work.get("doi", "").replace("https://doi.org/", "")
                    if work.get("doi")
                    else None,
                    "openalex_id": work.get("id"),
                    "authors": None,  # Will be enriched later
                    "abstract": None,  # Will be enriched later
                }

                # Extract authors
                authorships = work.get("authorships", [])
                if authorships:
                    authors = []
                    for authorship in authorships:
                        author = authorship.get("author", {})
                        author_name = author.get("display_name")
                        if author_name:
                            authors.append(author_name)
                    paper["authors"] = ", ".join(authors) if authors else None

                # Extract abstract from inverted index
                abstract_inverted = work.get("abstract_inverted_index")
                if abstract_inverted:
                    # Reconstruct abstract from inverted index
                    word_positions = []
                    for word, positions in abstract_inverted.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort()
                    paper["abstract"] = " ".join([word for _, word in word_positions])

                discovered_papers.append(paper)

                if len(discovered_papers) >= limit:
                    break

        except Exception as e:
            print(f"⚠️  Error searching OpenAlex: {e}")

        return discovered_papers[:limit]

    async def enrich(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich papers with Semantic Scholar data.

        Fetches additional metadata from Semantic Scholar:
        - influentialCitationCount
        - s2_paper_id
        - Updated citation counts
        - Abstract (if missing)

        Respects 3-second rate limit between requests.

        Args:
            papers: List of paper dictionaries from discover()

        Returns:
            Enriched list of papers with S2 metadata
        """
        enriched_papers = []

        for i, paper in enumerate(papers):
            # Rate limiting
            if i > 0:
                await asyncio.sleep(self.RATE_LIMIT_DELAY)

            try:
                # Try to find paper by DOI first, then by title
                s2_paper = None

                if paper.get("doi"):
                    try:
                        s2_paper = self.s2.get_paper(
                            f"DOI:{paper['doi']}",
                            fields=[
                                "paperId",
                                "externalIds",
                                "title",
                                "abstract",
                                "year",
                                "citationCount",
                                "influentialCitationCount",
                                "authors",
                            ],
                        )
                    except Exception as e:
                        print(f"  ⚠️  Error fetching by DOI: {e}")

                # Fallback to title search
                if not s2_paper and paper.get("title"):
                    try:
                        search_results = self.s2.search(
                            paper["title"],
                            limit=1,
                            fields=[
                                "paperId",
                                "externalIds",
                                "title",
                                "abstract",
                                "year",
                                "citationCount",
                                "influentialCitationCount",
                                "authors",
                            ],
                        )
                        if search_results.get("data"):
                            s2_paper = search_results["data"][0]
                    except Exception as e:
                        print(f"  ⚠️  Error searching by title: {e}")

                # Enrich paper with S2 data
                if s2_paper:
                    paper["s2_paper_id"] = s2_paper.get("paperId")
                    paper["influential_citations"] = s2_paper.get("influentialCitationCount", 0)

                    # Update citation count if S2 has more recent data
                    s2_citations = s2_paper.get("citationCount", 0)
                    if s2_citations > paper.get("citations", 0):
                        paper["citations"] = s2_citations

                    # Add abstract if missing
                    if not paper.get("abstract") and s2_paper.get("abstract"):
                        paper["abstract"] = s2_paper["abstract"]

                    # Extract arXiv ID if available
                    external_ids = s2_paper.get("externalIds", {})
                    if external_ids.get("ArXiv"):
                        paper["arxiv_id"] = external_ids["ArXiv"]

                enriched_papers.append(paper)

            except Exception as e:
                print(f"⚠️  Error enriching paper '{paper.get('title', 'Unknown')}': {e}")
                # Add paper anyway, even if enrichment failed
                enriched_papers.append(paper)

        return enriched_papers

    async def filter(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter papers by quality score.

        Applies calculate_quality_score() to each paper and filters
        papers with score >= quality_threshold (default: 60).

        Args:
            papers: List of enriched paper dictionaries

        Returns:
            Filtered list of high-quality papers
        """
        filtered_papers = []

        for paper in papers:
            # Calculate quality score
            score = calculate_quality_score(paper)
            paper["quality_score"] = score

            # Filter by threshold
            if score >= self.quality_threshold:
                filtered_papers.append(paper)

        return filtered_papers

    async def store(self, papers: List[Dict[str, Any]]) -> int:
        """Store papers in database with deduplication.

        Deduplicates based on:
        1. DOI (if available)
        2. arXiv ID (if available)
        3. Title hash (normalized lowercase, stripped)

        Inserts papers into great_papers table with source="automated".

        Args:
            papers: List of filtered paper dictionaries

        Returns:
            Count of newly stored papers (excluding duplicates)
        """
        newly_stored = 0
        seen_identifiers = set()

        for paper in papers:
            # Generate deduplication identifiers
            identifiers = []

            # 1. DOI-based identifier
            if paper.get("doi"):
                identifiers.append(f"doi:{paper['doi']}")

            # 2. arXiv ID-based identifier
            if paper.get("arxiv_id"):
                identifiers.append(f"arxiv:{paper['arxiv_id']}")

            # 3. Title hash-based identifier
            if paper.get("title"):
                title_normalized = paper["title"].lower().strip()
                title_hash = hashlib.md5(title_normalized.encode()).hexdigest()
                identifiers.append(f"title_hash:{title_hash}")

            # Check if paper already seen in this batch
            is_duplicate = False
            for identifier in identifiers:
                if identifier in seen_identifiers:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # Mark identifiers as seen
            for identifier in identifiers:
                seen_identifiers.add(identifier)

            # Prepare paper for database insertion
            db_paper = {
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "year": paper.get("year"),
                "arxiv_id": paper.get("arxiv_id"),
                "doi": paper.get("doi"),
                "citations": paper.get("citations", 0),
                "source": "automated",
                "s2_paper_id": paper.get("s2_paper_id"),
                "abstract": paper.get("abstract"),
                "venue": paper.get("venue"),
            }

            # Insert into database
            result = self.db.insert_paper(db_paper)
            if result is not None:
                newly_stored += 1

        return newly_stored

    async def collect_and_store(
        self, discover_limit: int = 100, verbose: bool = True
    ) -> Dict[str, Any]:
        """Run full collection pipeline: discover -> enrich -> filter -> store.

        Args:
            discover_limit: Maximum number of papers to discover
            verbose: Whether to print progress messages

        Returns:
            Summary dictionary with statistics:
                - discovered: Number of papers discovered
                - enriched: Number of papers enriched
                - filtered: Number of papers passing quality filter
                - stored: Number of papers newly stored
        """
        if verbose:
            print("🔍 Discovering papers from OpenAlex...")

        discovered = await self.discover(limit=discover_limit)

        if verbose:
            print(f"✓ Discovered {len(discovered)} papers")
            print("\n📊 Enriching with Semantic Scholar data...")

        enriched = await self.enrich(discovered)

        if verbose:
            print(f"✓ Enriched {len(enriched)} papers")
            print("\n🎯 Filtering by quality score...")

        filtered = await self.filter(enriched)

        if verbose:
            print(
                f"✓ Filtered to {len(filtered)} high-quality papers (score >= {self.quality_threshold})"
            )
            print("\n💾 Storing in database...")

        stored = await self.store(filtered)

        if verbose:
            print(f"✓ Stored {stored} new papers")

        summary = {
            "discovered": len(discovered),
            "enriched": len(enriched),
            "filtered": len(filtered),
            "stored": stored,
        }

        return summary

    def close(self):
        """Close all client connections."""
        self.openalex.close()
        self.s2.close()
        self.db.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
