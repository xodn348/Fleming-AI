"""
SQLite database management for Fleming-AI paper storage.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class PaperDatabase:
    """Database manager for storing and retrieving research papers."""

    def __init__(self, db_path: str | Path):
        """Initialize database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()

        # Great papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS great_papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                arxiv_id TEXT,
                doi TEXT,
                citations INTEGER DEFAULT 0,
                source TEXT NOT NULL,
                s2_paper_id TEXT,
                abstract TEXT,
                venue TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(title, year)
            )
        """)

        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_arxiv_id ON great_papers(arxiv_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_doi ON great_papers(doi)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON great_papers(source)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations ON great_papers(citations DESC)
        """)

        self.conn.commit()

    def insert_paper(self, paper: Dict[str, Any]) -> Optional[int]:
        """Insert a paper into the database.

        Args:
            paper: Paper dictionary with fields: title, authors, year, arxiv_id,
                   doi, citations, source, s2_paper_id, abstract, venue

        Returns:
            Row ID of inserted paper, or None if already exists
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO great_papers 
                (title, authors, year, arxiv_id, doi, citations, source, 
                 s2_paper_id, abstract, venue)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    paper.get("title"),
                    paper.get("authors"),
                    paper.get("year"),
                    paper.get("arxiv_id"),
                    paper.get("doi"),
                    paper.get("citations", 0),
                    paper.get("source"),
                    paper.get("s2_paper_id"),
                    paper.get("abstract"),
                    paper.get("venue"),
                ),
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Paper already exists (duplicate title + year)
            return None

    def insert_papers_batch(self, papers: List[Dict[str, Any]]) -> tuple[int, int]:
        """Insert multiple papers in a batch.

        Args:
            papers: List of paper dictionaries

        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        inserted = 0
        skipped = 0

        for paper in papers:
            result = self.insert_paper(paper)
            if result is not None:
                inserted += 1
            else:
                skipped += 1

        return inserted, skipped

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get a paper by its arXiv ID.

        Args:
            arxiv_id: arXiv ID

        Returns:
            Paper dictionary or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM great_papers WHERE arxiv_id = ?", (arxiv_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_papers_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all papers from a specific source.

        Args:
            source: Source identifier (e.g., 'turing', 'nobel', 'seminal')

        Returns:
            List of paper dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM great_papers WHERE source = ?", (source,))
        return [dict(row) for row in cursor.fetchall()]

    def get_all_papers(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all papers from the database.

        Args:
            limit: Maximum number of papers to return

        Returns:
            List of paper dictionaries
        """
        cursor = self.conn.cursor()
        query = "SELECT * FROM great_papers ORDER BY citations DESC"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def count_papers(self, source: Optional[str] = None) -> int:
        """Count papers in the database.

        Args:
            source: Optional source filter

        Returns:
            Number of papers
        """
        cursor = self.conn.cursor()
        if source:
            cursor.execute("SELECT COUNT(*) FROM great_papers WHERE source = ?", (source,))
        else:
            cursor.execute("SELECT COUNT(*) FROM great_papers")
        return cursor.fetchone()[0]

    def update_paper_citations(self, paper_id: int, citations: int):
        """Update citation count for a paper.

        Args:
            paper_id: Paper ID
            citations: New citation count
        """
        cursor = self.conn.cursor()
        cursor.execute("UPDATE great_papers SET citations = ? WHERE id = ?", (citations, paper_id))
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
