#!/usr/bin/env python3
"""
Enrich papers in great_papers table with citation data from Semantic Scholar.

This script:
1. Loads all papers from the great_papers table
2. For each paper, fetches citation count from Semantic Scholar
3. Updates citations, s2_paper_id, abstract, and venue fields
4. Respects 3-second rate limit between requests
5. Logs progress and provides summary
"""

import sqlite3
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from collectors.semantic_scholar_client import SemanticScholarClient


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Create database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_all_papers(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Fetch all papers from great_papers table."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, title, arxiv_id, doi, citations, s2_paper_id, abstract, venue
        FROM great_papers
        ORDER BY id
        """
    )
    return cursor.fetchall()


def update_paper(
    conn: sqlite3.Connection,
    paper_id: int,
    citations: int,
    s2_paper_id: str,
    abstract: str,
    venue: str,
) -> None:
    """Update paper with enriched data."""
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE great_papers
        SET citations = ?, s2_paper_id = ?, abstract = ?, venue = ?
        WHERE id = ?
        """,
        (citations, s2_paper_id, abstract, venue, paper_id),
    )
    conn.commit()


def enrich_papers(db_path: str) -> None:
    """Main enrichment function."""
    conn = get_db_connection(db_path)
    papers = fetch_all_papers(conn)

    print(f"Starting enrichment of {len(papers)} papers...")
    print("-" * 70)

    updated_count = 0
    failed_count = 0
    skipped_count = 0

    s2 = SemanticScholarClient()

    try:
        for idx, paper in enumerate(papers, 1):
            paper_id = paper["id"]
            title = paper["title"]
            arxiv_id = paper["arxiv_id"]
            existing_citations = paper["citations"]

            # Skip if already has citations
            if existing_citations > 0:
                skipped_count += 1
                print(
                    f"[{idx:3d}/106] SKIP: {title[:50]}... (already has {existing_citations} citations)"
                )
                continue

            paper_data = None
            error_msg = ""

            # Try arXiv ID first
            if arxiv_id:
                try:
                    paper_data = s2.get_paper(f"arXiv:{arxiv_id}")
                except Exception as e:
                    error_msg = f"arXiv lookup failed: {str(e)[:30]}"

            # Fallback to title search
            if not paper_data:
                try:
                    search_result = s2.search(title, limit=1)
                    if search_result.get("data"):
                        paper_data = search_result["data"][0]
                except Exception as e:
                    if not error_msg:
                        error_msg = f"Title search failed: {str(e)[:30]}"

            if paper_data:
                try:
                    citations = paper_data.get("citationCount", 0)
                    s2_paper_id = paper_data.get("paperId", "")
                    abstract = paper_data.get("abstract", "")
                    venue = paper_data.get("venue", "")

                    update_paper(conn, paper_id, citations, s2_paper_id, abstract, venue)
                    updated_count += 1
                    print(f"[{idx:3d}/106] OK: {title[:50]}... ({citations} citations)")
                except Exception as e:
                    failed_count += 1
                    print(f"[{idx:3d}/106] UPDATE ERROR: {title[:50]}... ({str(e)[:30]})")
            else:
                failed_count += 1
                if error_msg:
                    print(f"[{idx:3d}/106] FAIL: {title[:50]}... ({error_msg})")
                else:
                    print(f"[{idx:3d}/106] FAIL: {title[:50]}... (not found)")

    finally:
        s2.close()
        conn.close()

    print("-" * 70)
    print(f"\nEnrichment Summary:")
    print(f"  Updated:  {updated_count} papers")
    print(f"  Failed:   {failed_count} papers")
    print(f"  Skipped:  {skipped_count} papers")
    print(f"  Total:    {len(papers)} papers")


if __name__ == "__main__":
    db_path = Path(__file__).parent.parent / "data" / "db" / "papers.db"
    enrich_papers(str(db_path))
