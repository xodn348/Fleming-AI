"""
SQLite database for hypothesis storage in Fleming-AI.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.generators.hypothesis import Hypothesis


class HypothesisDatabase:
    """Database manager for storing and retrieving generated hypotheses."""

    def __init__(self, db_path: str | Path = "data/db/hypotheses.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()

        # Hypotheses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                hypothesis_text TEXT NOT NULL,
                source_papers TEXT NOT NULL,
                connection_json TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                quality_score REAL NOT NULL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON hypotheses(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence ON hypotheses(confidence DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality ON hypotheses(quality_score DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created ON hypotheses(created_at DESC)
        """)

        self.conn.commit()

    def insert_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """
        Insert a hypothesis into the database.

        Args:
            hypothesis: Hypothesis object to insert

        Returns:
            True if inserted successfully, False if already exists
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO hypotheses 
                (id, hypothesis_text, source_papers, connection_json, 
                 confidence, quality_score, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    hypothesis.id,
                    hypothesis.hypothesis_text,
                    json.dumps(hypothesis.source_papers),
                    json.dumps(hypothesis.connection),
                    hypothesis.confidence,
                    hypothesis.quality_score,
                    hypothesis.created_at.isoformat(),
                    hypothesis.status,
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Hypothesis with this ID already exists
            return False

    def insert_hypotheses_batch(self, hypotheses: list[Hypothesis]) -> tuple[int, int]:
        """
        Insert multiple hypotheses in a batch.

        Args:
            hypotheses: List of Hypothesis objects

        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        inserted = 0
        skipped = 0

        for hypothesis in hypotheses:
            if self.insert_hypothesis(hypothesis):
                inserted += 1
            else:
                skipped += 1

        return inserted, skipped

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """
        Get a hypothesis by ID.

        Args:
            hypothesis_id: Hypothesis ID

        Returns:
            Hypothesis object or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_hypothesis(row)

    def get_hypotheses_by_status(
        self,
        status: str,
        limit: Optional[int] = None,
    ) -> list[Hypothesis]:
        """
        Get hypotheses by status.

        Args:
            status: Status filter ('pending', 'validated', 'rejected')
            limit: Maximum number to return

        Returns:
            List of Hypothesis objects
        """
        cursor = self.conn.cursor()
        query = """
            SELECT * FROM hypotheses 
            WHERE status = ? 
            ORDER BY (confidence + quality_score) / 2 DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, (status,))
        return [self._row_to_hypothesis(row) for row in cursor.fetchall()]

    def get_top_hypotheses(
        self,
        limit: int = 10,
        min_confidence: float = 0.0,
        min_quality: float = 0.0,
    ) -> list[Hypothesis]:
        """
        Get top hypotheses sorted by combined score.

        Args:
            limit: Maximum number to return
            min_confidence: Minimum confidence threshold
            min_quality: Minimum quality score threshold

        Returns:
            List of Hypothesis objects
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM hypotheses 
            WHERE confidence >= ? AND quality_score >= ?
            ORDER BY (confidence + quality_score) / 2 DESC
            LIMIT ?
        """,
            (min_confidence, min_quality, limit),
        )
        return [self._row_to_hypothesis(row) for row in cursor.fetchall()]

    def get_all_hypotheses(self, limit: Optional[int] = None) -> list[Hypothesis]:
        """
        Get all hypotheses.

        Args:
            limit: Maximum number to return

        Returns:
            List of Hypothesis objects
        """
        cursor = self.conn.cursor()
        query = "SELECT * FROM hypotheses ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        return [self._row_to_hypothesis(row) for row in cursor.fetchall()]

    def search_hypotheses(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Hypothesis]:
        """
        Search hypotheses by text content.

        Args:
            query: Search query
            limit: Maximum number to return

        Returns:
            List of matching Hypothesis objects
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM hypotheses 
            WHERE hypothesis_text LIKE ? OR connection_json LIKE ?
            ORDER BY (confidence + quality_score) / 2 DESC
            LIMIT ?
        """,
            (f"%{query}%", f"%{query}%", limit),
        )
        return [self._row_to_hypothesis(row) for row in cursor.fetchall()]

    def update_status(self, hypothesis_id: str, status: str) -> bool:
        """
        Update hypothesis status.

        Args:
            hypothesis_id: Hypothesis ID
            status: New status ('pending', 'validated', 'rejected')

        Returns:
            True if updated, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE hypotheses SET status = ? WHERE id = ?",
            (status, hypothesis_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def update_scores(
        self,
        hypothesis_id: str,
        confidence: Optional[float] = None,
        quality_score: Optional[float] = None,
    ) -> bool:
        """
        Update hypothesis scores.

        Args:
            hypothesis_id: Hypothesis ID
            confidence: New confidence score (optional)
            quality_score: New quality score (optional)

        Returns:
            True if updated, False if not found
        """
        cursor = self.conn.cursor()

        updates = []
        values = []

        if confidence is not None:
            updates.append("confidence = ?")
            values.append(confidence)

        if quality_score is not None:
            updates.append("quality_score = ?")
            values.append(quality_score)

        if not updates:
            return False

        values.append(hypothesis_id)
        query = f"UPDATE hypotheses SET {', '.join(updates)} WHERE id = ?"

        cursor.execute(query, values)
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_hypothesis(self, hypothesis_id: str) -> bool:
        """
        Delete a hypothesis.

        Args:
            hypothesis_id: Hypothesis ID

        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM hypotheses WHERE id = ?", (hypothesis_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def count_hypotheses(self, status: Optional[str] = None) -> int:
        """
        Count hypotheses in the database.

        Args:
            status: Optional status filter

        Returns:
            Number of hypotheses
        """
        cursor = self.conn.cursor()
        if status:
            cursor.execute("SELECT COUNT(*) FROM hypotheses WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT COUNT(*) FROM hypotheses")
        return cursor.fetchone()[0]

    def _row_to_hypothesis(self, row: sqlite3.Row) -> Hypothesis:
        """Convert database row to Hypothesis object."""
        return Hypothesis(
            id=row["id"],
            hypothesis_text=row["hypothesis_text"],
            source_papers=json.loads(row["source_papers"]),
            connection=json.loads(row["connection_json"]),
            confidence=row["confidence"],
            quality_score=row["quality_score"],
            created_at=datetime.fromisoformat(row["created_at"]),
            status=row["status"],
        )

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
