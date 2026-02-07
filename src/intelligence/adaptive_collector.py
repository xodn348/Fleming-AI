"""
Adaptive Paper Collector for Fleming-AI
Self-improving collection system with metrics tracking, threshold optimization, and feedback loops.

Implements continuous improvement ("끊임없이 발전") through:
1. MetricsTracker: Tracks paper quality, hypothesis success rates, venue performance
2. ThresholdOptimizer: Adjusts collection thresholds based on performance data
3. FeedbackLoop: Learns from hypothesis validation to improve paper selection
4. AdaptiveCollector: Orchestrates the self-improving collection pipeline
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.collectors.paper_collector import PaperCollector
from src.storage.hypothesis_db import HypothesisDatabase
from src.utils.scoring import VENUE_TIERS

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CollectionMetrics:
    """Metrics for a single collection cycle."""

    cycle_id: str
    timestamp: datetime
    discovered: int
    enriched: int
    filtered: int
    stored: int
    thresholds_used: dict[str, Any]
    strategy_id: str | None = None


@dataclass
class PaperPerformance:
    """Performance tracking for a single paper."""

    paper_id: int
    title: str
    venue: str
    year: int
    citations: int
    quality_score: float
    hypotheses_generated: int = 0
    hypotheses_validated: int = 0
    hypotheses_rejected: int = 0
    contribution_score: float = 0.0


@dataclass
class VenuePerformance:
    """Aggregated performance metrics for a publication venue."""

    venue: str
    papers_collected: int = 0
    total_hypotheses: int = 0
    validated_hypotheses: int = 0
    validation_rate: float = 0.0
    avg_quality_score: float = 0.0


@dataclass
class ThresholdConfig:
    """Configuration for collection thresholds."""

    min_citations: int = 100
    quality_threshold: float = 60.0
    venue_weights: dict[str, float] = field(default_factory=dict)
    recency_weight: float = 0.3
    influence_weight: float = 0.4
    velocity_weight: float = 0.3
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    justification: str = "Initial configuration"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "min_citations": self.min_citations,
            "quality_threshold": self.quality_threshold,
            "venue_weights": self.venue_weights,
            "recency_weight": self.recency_weight,
            "influence_weight": self.influence_weight,
            "velocity_weight": self.velocity_weight,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "justification": self.justification,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThresholdConfig":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            min_citations=data.get("min_citations", 100),
            quality_threshold=data.get("quality_threshold", 60.0),
            venue_weights=data.get("venue_weights", {}),
            recency_weight=data.get("recency_weight", 0.3),
            influence_weight=data.get("influence_weight", 0.4),
            velocity_weight=data.get("velocity_weight", 0.3),
            version=data.get("version", 1),
            created_at=created_at,
            justification=data.get("justification", ""),
        )


@dataclass
class ABTestResult:
    """Result of an A/B test between strategies."""

    test_id: str
    strategy_a: str
    strategy_b: str
    winner: str | None
    metric_a: float
    metric_b: float
    confidence: float
    sample_size: int
    started_at: datetime
    ended_at: datetime | None = None


# ============================================================================
# MetricsTracker
# ============================================================================


class MetricsTracker:
    """
    Tracks all metrics for the adaptive collection system.

    Stores metrics in SQLite for persistence and analysis.
    Tracks:
    - Collection cycle performance
    - Paper-level performance (hypotheses generated/validated)
    - Venue-level performance
    - Threshold adjustments history
    - A/B test results
    """

    def __init__(self, db_path: str | Path = "data/db/adaptive_metrics.db"):
        """Initialize metrics tracker with database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema for metrics."""
        cursor = self.conn.cursor()

        # Collection cycles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_cycles (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                discovered INTEGER NOT NULL,
                enriched INTEGER NOT NULL,
                filtered INTEGER NOT NULL,
                stored INTEGER NOT NULL,
                thresholds_json TEXT NOT NULL,
                strategy_id TEXT
            )
        """)

        # Paper performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_performance (
                paper_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                venue TEXT,
                year INTEGER,
                citations INTEGER,
                quality_score REAL,
                hypotheses_generated INTEGER DEFAULT 0,
                hypotheses_validated INTEGER DEFAULT 0,
                hypotheses_rejected INTEGER DEFAULT 0,
                contribution_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Venue performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS venue_performance (
                venue TEXT PRIMARY KEY,
                papers_collected INTEGER DEFAULT 0,
                total_hypotheses INTEGER DEFAULT 0,
                validated_hypotheses INTEGER DEFAULT 0,
                validation_rate REAL DEFAULT 0.0,
                avg_quality_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Threshold history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threshold_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                justification TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performance_before REAL,
                performance_after REAL
            )
        """)

        # A/B test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                strategy_a TEXT NOT NULL,
                strategy_b TEXT NOT NULL,
                winner TEXT,
                metric_a REAL,
                metric_b REAL,
                confidence REAL,
                sample_size INTEGER,
                started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP
            )
        """)

        # Concept diversity tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concept_diversity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                unique_concepts INTEGER NOT NULL,
                new_concepts INTEGER NOT NULL,
                concept_growth_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Citation prediction accuracy
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                predicted_citations INTEGER NOT NULL,
                actual_citations INTEGER,
                prediction_date TIMESTAMP NOT NULL,
                check_date TIMESTAMP,
                accuracy REAL,
                FOREIGN KEY (paper_id) REFERENCES paper_performance(paper_id)
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cycles_timestamp 
            ON collection_cycles(timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_venue 
            ON paper_performance(venue)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_quality 
            ON paper_performance(quality_score DESC)
        """)

        self.conn.commit()

    def record_collection_cycle(self, metrics: CollectionMetrics) -> None:
        """Record metrics from a collection cycle."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO collection_cycles 
            (id, timestamp, discovered, enriched, filtered, stored, thresholds_json, strategy_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.cycle_id,
                metrics.timestamp.isoformat(),
                metrics.discovered,
                metrics.enriched,
                metrics.filtered,
                metrics.stored,
                json.dumps(metrics.thresholds_used),
                metrics.strategy_id,
            ),
        )
        self.conn.commit()
        logger.info(f"Recorded collection cycle {metrics.cycle_id}")

    def record_paper_performance(self, paper: PaperPerformance) -> None:
        """Record or update paper performance metrics."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO paper_performance 
            (paper_id, title, venue, year, citations, quality_score,
             hypotheses_generated, hypotheses_validated, hypotheses_rejected, contribution_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                hypotheses_generated = hypotheses_generated + excluded.hypotheses_generated,
                hypotheses_validated = hypotheses_validated + excluded.hypotheses_validated,
                hypotheses_rejected = hypotheses_rejected + excluded.hypotheses_rejected,
                contribution_score = excluded.contribution_score,
                updated_at = CURRENT_TIMESTAMP
        """,
            (
                paper.paper_id,
                paper.title,
                paper.venue,
                paper.year,
                paper.citations,
                paper.quality_score,
                paper.hypotheses_generated,
                paper.hypotheses_validated,
                paper.hypotheses_rejected,
                paper.contribution_score,
            ),
        )
        self.conn.commit()

    def update_venue_performance(self, venue: str) -> VenuePerformance:
        """Recalculate and update venue performance metrics."""
        cursor = self.conn.cursor()

        # Aggregate paper performance by venue
        cursor.execute(
            """
            SELECT 
                COUNT(*) as papers_collected,
                SUM(hypotheses_generated) as total_hypotheses,
                SUM(hypotheses_validated) as validated_hypotheses,
                AVG(quality_score) as avg_quality_score
            FROM paper_performance
            WHERE venue = ?
        """,
            (venue,),
        )
        row = cursor.fetchone()

        papers = row["papers_collected"] or 0
        total = row["total_hypotheses"] or 0
        validated = row["validated_hypotheses"] or 0
        avg_quality = row["avg_quality_score"] or 0.0

        validation_rate = validated / total if total > 0 else 0.0

        # Update venue performance table
        cursor.execute(
            """
            INSERT INTO venue_performance 
            (venue, papers_collected, total_hypotheses, validated_hypotheses, 
             validation_rate, avg_quality_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(venue) DO UPDATE SET
                papers_collected = excluded.papers_collected,
                total_hypotheses = excluded.total_hypotheses,
                validated_hypotheses = excluded.validated_hypotheses,
                validation_rate = excluded.validation_rate,
                avg_quality_score = excluded.avg_quality_score,
                updated_at = CURRENT_TIMESTAMP
        """,
            (venue, papers, total, validated, validation_rate, avg_quality),
        )
        self.conn.commit()

        return VenuePerformance(
            venue=venue,
            papers_collected=papers,
            total_hypotheses=total,
            validated_hypotheses=validated,
            validation_rate=validation_rate,
            avg_quality_score=avg_quality,
        )

    def record_threshold_change(
        self,
        config: ThresholdConfig,
        performance_before: float | None = None,
        performance_after: float | None = None,
    ) -> None:
        """Record a threshold configuration change."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO threshold_history 
            (version, config_json, justification, performance_before, performance_after)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                config.version,
                json.dumps(config.to_dict()),
                config.justification,
                performance_before,
                performance_after,
            ),
        )
        self.conn.commit()
        logger.info(f"Recorded threshold change v{config.version}: {config.justification}")

    def get_recent_cycles(self, days: int = 30) -> list[CollectionMetrics]:
        """Get collection cycles from the last N days."""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute(
            """
            SELECT * FROM collection_cycles
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """,
            (cutoff,),
        )

        cycles = []
        for row in cursor.fetchall():
            cycles.append(
                CollectionMetrics(
                    cycle_id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    discovered=row["discovered"],
                    enriched=row["enriched"],
                    filtered=row["filtered"],
                    stored=row["stored"],
                    thresholds_used=json.loads(row["thresholds_json"]),
                    strategy_id=row["strategy_id"],
                )
            )
        return cycles

    def get_venue_rankings(self, min_papers: int = 5) -> list[VenuePerformance]:
        """Get venues ranked by validation rate (with minimum sample size)."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM venue_performance
            WHERE papers_collected >= ?
            ORDER BY validation_rate DESC
        """,
            (min_papers,),
        )

        venues = []
        for row in cursor.fetchall():
            venues.append(
                VenuePerformance(
                    venue=row["venue"],
                    papers_collected=row["papers_collected"],
                    total_hypotheses=row["total_hypotheses"],
                    validated_hypotheses=row["validated_hypotheses"],
                    validation_rate=row["validation_rate"],
                    avg_quality_score=row["avg_quality_score"],
                )
            )
        return venues

    def get_hypothesis_success_rate(self, days: int = 30) -> float:
        """Calculate overall hypothesis validation success rate."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT 
                SUM(hypotheses_validated) as validated,
                SUM(hypotheses_generated) as total
            FROM paper_performance
        """
        )
        row = cursor.fetchone()
        total = row["total"] or 0
        validated = row["validated"] or 0
        return validated / total if total > 0 else 0.0

    def get_current_threshold_version(self) -> int:
        """Get the current threshold version number."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(version) as max_version FROM threshold_history")
        row = cursor.fetchone()
        return row["max_version"] or 0

    def record_concept_diversity(
        self,
        unique_concepts: int,
        new_concepts: int,
        growth_rate: float,
    ) -> None:
        """Record concept diversity metrics."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO concept_diversity 
            (date, unique_concepts, new_concepts, concept_growth_rate)
            VALUES (DATE('now'), ?, ?, ?)
        """,
            (unique_concepts, new_concepts, growth_rate),
        )
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# ThresholdOptimizer
# ============================================================================


class ThresholdOptimizer:
    """
    Optimizes collection thresholds based on performance data.

    Adjustment strategies:
    1. Citation threshold: Increase if too many low-quality papers, decrease if missing good ones
    2. Quality threshold: Adjust based on hypothesis validation rate
    3. Venue weights: Learn which venues produce best hypotheses
    4. Scoring formula: Evolve weights based on outcomes
    """

    # Adjustment bounds to prevent extreme changes
    MIN_CITATIONS = 10
    MAX_CITATIONS = 500
    MIN_QUALITY = 30.0
    MAX_QUALITY = 90.0
    ADJUSTMENT_STEP = 0.1  # 10% adjustment per iteration

    def __init__(self, metrics: MetricsTracker):
        """Initialize optimizer with metrics tracker."""
        self.metrics = metrics
        self.current_config = self._load_current_config()

    def _load_current_config(self) -> ThresholdConfig:
        """Load the current threshold configuration from database."""
        cursor = self.metrics.conn.cursor()
        cursor.execute(
            """
            SELECT config_json FROM threshold_history
            ORDER BY version DESC LIMIT 1
        """
        )
        row = cursor.fetchone()

        if row:
            return ThresholdConfig.from_dict(json.loads(row["config_json"]))

        # Return default config if no history exists
        default_config = ThresholdConfig(
            min_citations=100,
            quality_threshold=60.0,
            venue_weights=self._build_initial_venue_weights(),
            justification="Initial default configuration",
        )
        self.metrics.record_threshold_change(default_config)
        return default_config

    def _build_initial_venue_weights(self) -> dict[str, float]:
        """Build initial venue weights from tier definitions."""
        weights: dict[str, float] = {}
        for tier_name, tier_data in VENUE_TIERS.items():
            weight = 1.0
            if tier_name == "tier_1":
                weight = 1.2
            elif tier_name == "tier_2":
                weight = 1.0
            elif tier_name == "tier_3":
                weight = 0.8

            conferences = tier_data.get("conferences", [])
            journals = tier_data.get("journals", [])

            if isinstance(conferences, list):
                for venue in conferences:
                    weights[venue] = weight
            if isinstance(journals, list):
                for venue in journals:
                    weights[venue] = weight

        return weights

    def analyze_performance(self) -> dict[str, Any]:
        """Analyze recent performance to determine adjustment needs."""
        recent_cycles = self.metrics.get_recent_cycles(days=7)
        success_rate = self.metrics.get_hypothesis_success_rate(days=30)
        venue_rankings = self.metrics.get_venue_rankings(min_papers=3)

        # Calculate key metrics
        avg_stored = (
            sum(c.stored for c in recent_cycles) / len(recent_cycles) if recent_cycles else 0
        )
        avg_filtered = (
            sum(c.filtered for c in recent_cycles) / len(recent_cycles) if recent_cycles else 0
        )
        filter_ratio = avg_stored / avg_filtered if avg_filtered > 0 else 0

        recommendations: list[str] = []

        # Generate recommendations
        if success_rate < 0.1:
            recommendations.append("Increase quality_threshold (low hypothesis success)")
        elif success_rate > 0.5:
            recommendations.append("Consider decreasing quality_threshold (room for more papers)")

        if filter_ratio < 0.3:
            recommendations.append("Decrease min_citations (too few papers stored)")
        elif filter_ratio > 0.9:
            recommendations.append("Increase min_citations (filtering not selective enough)")

        analysis: dict[str, Any] = {
            "recent_cycles_count": len(recent_cycles),
            "avg_papers_stored": avg_stored,
            "avg_papers_filtered": avg_filtered,
            "filter_to_store_ratio": filter_ratio,
            "hypothesis_success_rate": success_rate,
            "top_venues": [v.venue for v in venue_rankings[:5]],
            "bottom_venues": [v.venue for v in venue_rankings[-5:]]
            if len(venue_rankings) > 5
            else [],
            "recommendations": recommendations,
        }

        return analysis

    def optimize(self, force: bool = False) -> ThresholdConfig | None:
        """
        Run optimization and return new config if changes are warranted.

        Args:
            force: Force optimization even if performance is acceptable

        Returns:
            New ThresholdConfig if adjustments made, None otherwise
        """
        analysis = self.analyze_performance()
        success_rate = analysis["hypothesis_success_rate"]
        filter_ratio = analysis["filter_to_store_ratio"]

        # Check if optimization is needed
        if not force and 0.2 <= success_rate <= 0.4 and 0.4 <= filter_ratio <= 0.8:
            logger.info("Performance within acceptable range, no optimization needed")
            return None

        # Determine adjustments
        adjustments = []
        new_config = ThresholdConfig(
            min_citations=self.current_config.min_citations,
            quality_threshold=self.current_config.quality_threshold,
            venue_weights=dict(self.current_config.venue_weights),
            recency_weight=self.current_config.recency_weight,
            influence_weight=self.current_config.influence_weight,
            velocity_weight=self.current_config.velocity_weight,
            version=self.current_config.version + 1,
        )

        # Adjust citation threshold
        if filter_ratio < 0.3:
            old_val = new_config.min_citations
            new_config.min_citations = max(
                self.MIN_CITATIONS,
                int(new_config.min_citations * (1 - self.ADJUSTMENT_STEP)),
            )
            if new_config.min_citations != old_val:
                adjustments.append(f"min_citations: {old_val} -> {new_config.min_citations}")
        elif filter_ratio > 0.9:
            old_val = new_config.min_citations
            new_config.min_citations = min(
                self.MAX_CITATIONS,
                int(new_config.min_citations * (1 + self.ADJUSTMENT_STEP)),
            )
            if new_config.min_citations != old_val:
                adjustments.append(f"min_citations: {old_val} -> {new_config.min_citations}")

        # Adjust quality threshold
        if success_rate < 0.1:
            old_val = new_config.quality_threshold
            new_config.quality_threshold = min(
                self.MAX_QUALITY,
                new_config.quality_threshold + 5.0,
            )
            if new_config.quality_threshold != old_val:
                adjustments.append(
                    f"quality_threshold: {old_val} -> {new_config.quality_threshold}"
                )
        elif success_rate > 0.5:
            old_val = new_config.quality_threshold
            new_config.quality_threshold = max(
                self.MIN_QUALITY,
                new_config.quality_threshold - 5.0,
            )
            if new_config.quality_threshold != old_val:
                adjustments.append(
                    f"quality_threshold: {old_val} -> {new_config.quality_threshold}"
                )

        # Update venue weights based on performance
        venue_rankings = self.metrics.get_venue_rankings(min_papers=3)
        for venue_perf in venue_rankings:
            if venue_perf.validation_rate > 0.3:
                # Boost weight for high-performing venues
                current_weight = new_config.venue_weights.get(venue_perf.venue, 1.0)
                new_config.venue_weights[venue_perf.venue] = min(1.5, current_weight * 1.1)
            elif venue_perf.validation_rate < 0.05 and venue_perf.total_hypotheses >= 10:
                # Reduce weight for consistently poor venues
                current_weight = new_config.venue_weights.get(venue_perf.venue, 1.0)
                new_config.venue_weights[venue_perf.venue] = max(0.5, current_weight * 0.9)

        if not adjustments:
            logger.info("No threshold adjustments needed")
            return None

        # Build justification
        new_config.justification = f"Auto-optimization: {'; '.join(adjustments)}"

        # Record change
        self.metrics.record_threshold_change(
            new_config,
            performance_before=success_rate,
        )

        self.current_config = new_config
        logger.info(f"Optimized thresholds: {new_config.justification}")

        return new_config

    def get_current_config(self) -> ThresholdConfig:
        """Get the current threshold configuration."""
        return self.current_config


# ============================================================================
# FeedbackLoop
# ============================================================================


class FeedbackLoop:
    """
    Learns from hypothesis validation results to improve paper selection.

    Tracks which paper characteristics predict successful hypotheses:
    - Citation velocity
    - Venue
    - Concept diversity
    - Author reputation
    - Methodology type
    """

    def __init__(self, metrics: MetricsTracker, hypothesis_db: HypothesisDatabase):
        """Initialize feedback loop with dependencies."""
        self.metrics = metrics
        self.hypothesis_db = hypothesis_db

    def process_validation_result(
        self,
        hypothesis_id: str,
        is_validated: bool,
        source_paper_ids: list[int],
    ) -> None:
        """
        Process a hypothesis validation result and update paper metrics.

        Args:
            hypothesis_id: ID of the validated hypothesis
            is_validated: Whether the hypothesis was validated (True) or rejected (False)
            source_paper_ids: IDs of papers that contributed to this hypothesis
        """
        cursor = self.metrics.conn.cursor()

        for paper_id in source_paper_ids:
            if is_validated:
                cursor.execute(
                    """
                    UPDATE paper_performance 
                    SET hypotheses_validated = hypotheses_validated + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE paper_id = ?
                """,
                    (paper_id,),
                )
            else:
                cursor.execute(
                    """
                    UPDATE paper_performance 
                    SET hypotheses_rejected = hypotheses_rejected + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE paper_id = ?
                """,
                    (paper_id,),
                )

            # Recalculate contribution score
            cursor.execute(
                """
                SELECT hypotheses_generated, hypotheses_validated, hypotheses_rejected
                FROM paper_performance WHERE paper_id = ?
            """,
                (paper_id,),
            )
            row = cursor.fetchone()
            if row:
                generated = row["hypotheses_generated"] or 0
                validated = row["hypotheses_validated"] or 0
                rejected = row["hypotheses_rejected"] or 0

                # Contribution score: weighted validation rate
                # Higher weight for validated, penalty for rejected
                if generated > 0:
                    contribution = (validated * 1.0 - rejected * 0.5) / generated
                else:
                    contribution = 0.0

                cursor.execute(
                    """
                    UPDATE paper_performance 
                    SET contribution_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE paper_id = ?
                """,
                    (contribution, paper_id),
                )

        self.metrics.conn.commit()

        # Update venue performance for affected venues
        cursor.execute(
            """
            SELECT DISTINCT venue FROM paper_performance 
            WHERE paper_id IN ({})
        """.format(",".join("?" * len(source_paper_ids))),
            source_paper_ids,
        )
        for row in cursor.fetchall():
            if row["venue"]:
                self.metrics.update_venue_performance(row["venue"])

        logger.debug(f"Processed validation for hypothesis {hypothesis_id}")

    def sync_from_hypothesis_db(self) -> int:
        """
        Sync validation results from hypothesis database.

        Returns:
            Number of validations processed
        """
        # Get validated hypotheses
        validated = self.hypothesis_db.get_hypotheses_by_status("validated")
        rejected = self.hypothesis_db.get_hypotheses_by_status("rejected")

        processed = 0

        for hypothesis in validated:
            # Extract paper IDs from source_papers (which may be stored as strings or IDs)
            paper_ids = self._extract_paper_ids(hypothesis.source_papers)
            if paper_ids:
                self.process_validation_result(hypothesis.id, True, paper_ids)
                processed += 1

        for hypothesis in rejected:
            paper_ids = self._extract_paper_ids(hypothesis.source_papers)
            if paper_ids:
                self.process_validation_result(hypothesis.id, False, paper_ids)
                processed += 1

        logger.info(f"Synced {processed} validation results from hypothesis database")
        return processed

    def _extract_paper_ids(self, source_papers: list[str]) -> list[int]:
        """Extract numeric paper IDs from source_papers list."""
        ids = []
        for paper_ref in source_papers:
            try:
                # Try to parse as integer directly
                if isinstance(paper_ref, int):
                    ids.append(paper_ref)
                elif isinstance(paper_ref, str):
                    # Try to extract ID from string like "paper_123" or just "123"
                    if paper_ref.isdigit():
                        ids.append(int(paper_ref))
                    elif paper_ref.startswith("paper_"):
                        ids.append(int(paper_ref.split("_")[1]))
            except (ValueError, IndexError):
                continue
        return ids

    def identify_success_patterns(self) -> dict[str, Any]:
        """
        Identify patterns in papers that lead to successful hypotheses.

        Returns:
            Dictionary of identified patterns and their strengths
        """
        cursor = self.metrics.conn.cursor()

        high_performing_venues: list[dict[str, Any]] = []
        optimal_citation_range: dict[str, dict[str, Any]] = {}
        optimal_quality_range: dict[str, dict[str, Any]] = {}
        recommendations: list[str] = []

        # Find high-performing venues
        cursor.execute(
            """
            SELECT venue, validation_rate, papers_collected
            FROM venue_performance
            WHERE papers_collected >= 3 AND validation_rate > 0.2
            ORDER BY validation_rate DESC
            LIMIT 10
        """
        )
        high_performing_venues = [
            {"venue": row["venue"], "validation_rate": row["validation_rate"]}
            for row in cursor.fetchall()
        ]

        # Find optimal citation range
        cursor.execute(
            """
            SELECT 
                CASE 
                    WHEN citations < 50 THEN 'low'
                    WHEN citations < 200 THEN 'medium'
                    WHEN citations < 500 THEN 'high'
                    ELSE 'very_high'
                END as citation_bucket,
                AVG(CAST(hypotheses_validated AS FLOAT) / NULLIF(hypotheses_generated, 0)) as avg_success_rate,
                COUNT(*) as paper_count
            FROM paper_performance
            WHERE hypotheses_generated > 0
            GROUP BY citation_bucket
            ORDER BY avg_success_rate DESC
        """
        )
        for row in cursor.fetchall():
            optimal_citation_range[row["citation_bucket"]] = {
                "success_rate": row["avg_success_rate"] or 0,
                "paper_count": row["paper_count"],
            }

        # Find optimal quality score range
        cursor.execute(
            """
            SELECT 
                CASE 
                    WHEN quality_score < 50 THEN 'low'
                    WHEN quality_score < 70 THEN 'medium'
                    WHEN quality_score < 85 THEN 'high'
                    ELSE 'very_high'
                END as quality_bucket,
                AVG(CAST(hypotheses_validated AS FLOAT) / NULLIF(hypotheses_generated, 0)) as avg_success_rate,
                COUNT(*) as paper_count
            FROM paper_performance
            WHERE hypotheses_generated > 0
            GROUP BY quality_bucket
            ORDER BY avg_success_rate DESC
        """
        )
        for row in cursor.fetchall():
            optimal_quality_range[row["quality_bucket"]] = {
                "success_rate": row["avg_success_rate"] or 0,
                "paper_count": row["paper_count"],
            }

        # Generate recommendations
        if high_performing_venues:
            top_venues = [str(v["venue"]) for v in high_performing_venues[:3]]
            recommendations.append(f"Prioritize papers from: {', '.join(top_venues)}")

        if optimal_citation_range:
            best_bucket = max(
                optimal_citation_range.items(),
                key=lambda x: x[1].get("success_rate", 0),
            )
            recommendations.append(f"Focus on {best_bucket[0]} citation papers")

        return {
            "high_performing_venues": high_performing_venues,
            "optimal_citation_range": optimal_citation_range,
            "optimal_quality_range": optimal_quality_range,
            "recommendations": recommendations,
        }


# ============================================================================
# AdaptiveCollector
# ============================================================================


class AdaptiveCollector:
    """
    Self-improving paper collector that wraps PaperCollector.

    Orchestrates the adaptive collection pipeline:
    1. Collect papers using current thresholds
    2. Track which papers were selected
    3. Monitor hypothesis quality from these papers
    4. Adjust thresholds based on performance
    5. Log all decisions for transparency
    """

    def __init__(
        self,
        db_path: str = "data/db/papers.db",
        metrics_db_path: str = "data/db/adaptive_metrics.db",
        hypothesis_db_path: str = "data/db/hypotheses.db",
    ):
        """
        Initialize the adaptive collector with all dependencies.

        Args:
            db_path: Path to papers database
            metrics_db_path: Path to adaptive metrics database
            hypothesis_db_path: Path to hypotheses database
        """
        self.metrics = MetricsTracker(metrics_db_path)
        self.optimizer = ThresholdOptimizer(self.metrics)
        self.hypothesis_db = HypothesisDatabase(hypothesis_db_path)
        self.feedback = FeedbackLoop(self.metrics, self.hypothesis_db)

        # Track current A/B test if any
        self._active_ab_test: ABTestResult | None = None
        self._ab_test_strategy: str | None = None

        # Initialize collector with current optimal config
        self._update_collector(db_path)

        logger.info("AdaptiveCollector initialized with self-improvement capabilities")

    def _update_collector(self, db_path: str) -> None:
        """Update the internal collector with current thresholds."""
        config = self.optimizer.get_current_config()
        self.collector = PaperCollector(
            config={
                "db_path": db_path,
                "min_citations": config.min_citations,
                "quality_threshold": config.quality_threshold,
            }
        )

    async def collect_with_learning(
        self,
        discover_limit: int = 100,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Run collection pipeline with learning and metrics tracking.

        This is the main entry point for adaptive collection:
        1. Syncs feedback from hypothesis validation
        2. Optionally runs optimization
        3. Collects papers with current thresholds
        4. Records all metrics
        5. Returns summary

        Args:
            discover_limit: Maximum papers to discover
            verbose: Whether to print progress

        Returns:
            Summary dict with collection stats and learning info
        """
        cycle_id = str(uuid.uuid4())[:8]
        cycle_start = datetime.now()

        if verbose:
            logger.info(f"Starting adaptive collection cycle {cycle_id}")

        # Step 1: Sync feedback from hypothesis validation
        feedback_synced = self.feedback.sync_from_hypothesis_db()
        if verbose:
            logger.info(f"Synced {feedback_synced} hypothesis validation results")

        # Step 2: Check if optimization is needed
        performance_analysis = self.optimizer.analyze_performance()
        new_config = self.optimizer.optimize()

        if new_config and verbose:
            logger.info(f"Applied threshold optimization: {new_config.justification}")
            # Reinitialize collector with new thresholds
            self._update_collector(str(self.collector.db.db_path))

        # Step 3: Run collection pipeline
        current_config = self.optimizer.get_current_config()
        thresholds_used = current_config.to_dict()

        if verbose:
            logger.info(
                f"Collecting with thresholds: min_citations={current_config.min_citations}, "
                f"quality_threshold={current_config.quality_threshold}"
            )

        summary = await self.collector.collect_and_store(
            discover_limit=discover_limit,
            verbose=verbose,
        )

        # Step 4: Record collection metrics
        metrics = CollectionMetrics(
            cycle_id=cycle_id,
            timestamp=cycle_start,
            discovered=summary["discovered"],
            enriched=summary["enriched"],
            filtered=summary["filtered"],
            stored=summary["stored"],
            thresholds_used=thresholds_used,
            strategy_id=self._ab_test_strategy,
        )
        self.metrics.record_collection_cycle(metrics)

        # Step 5: Build comprehensive summary
        enhanced_summary = {
            **summary,
            "cycle_id": cycle_id,
            "thresholds": thresholds_used,
            "optimization_applied": new_config is not None,
            "performance_analysis": performance_analysis,
            "feedback_synced": feedback_synced,
        }

        if verbose:
            logger.info(
                f"Collection cycle {cycle_id} complete: "
                f"discovered={summary['discovered']}, stored={summary['stored']}"
            )

        return enhanced_summary

    def adjust_thresholds(self, performance_data: dict[str, Any]) -> ThresholdConfig | None:
        """
        Manually trigger threshold adjustment based on provided performance data.

        Args:
            performance_data: Dict with performance metrics to consider

        Returns:
            New ThresholdConfig if adjustments made, None otherwise
        """
        return self.optimizer.optimize(force=True)

    def learn_from_feedback(self, hypothesis_results: list[dict[str, Any]]) -> None:
        """
        Process hypothesis validation results for learning.

        Args:
            hypothesis_results: List of dicts with hypothesis_id, is_validated, source_paper_ids
        """
        for result in hypothesis_results:
            self.feedback.process_validation_result(
                hypothesis_id=result["hypothesis_id"],
                is_validated=result["is_validated"],
                source_paper_ids=result["source_paper_ids"],
            )

    def start_ab_test(
        self,
        strategy_a: str,
        strategy_b: str,
        config_a: ThresholdConfig,
        config_b: ThresholdConfig,
    ) -> str:
        """
        Start an A/B test between two collection strategies.

        Args:
            strategy_a: Name/description of strategy A
            strategy_b: Name/description of strategy B
            config_a: Threshold config for strategy A
            config_b: Threshold config for strategy B

        Returns:
            Test ID for tracking
        """
        test_id = str(uuid.uuid4())[:8]

        self._active_ab_test = ABTestResult(
            test_id=test_id,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            winner=None,
            metric_a=0.0,
            metric_b=0.0,
            confidence=0.0,
            sample_size=0,
            started_at=datetime.now(),
        )

        # Store test info
        cursor = self.metrics.conn.cursor()
        cursor.execute(
            """
            INSERT INTO ab_tests (test_id, strategy_a, strategy_b, started_at)
            VALUES (?, ?, ?, ?)
        """,
            (test_id, strategy_a, strategy_b, datetime.now().isoformat()),
        )
        self.metrics.conn.commit()

        logger.info(f"Started A/B test {test_id}: {strategy_a} vs {strategy_b}")
        return test_id

    def end_ab_test(self, test_id: str) -> ABTestResult | None:
        """
        End an A/B test and determine winner.

        Args:
            test_id: ID of the test to end

        Returns:
            ABTestResult with winner and metrics
        """
        if not self._active_ab_test or self._active_ab_test.test_id != test_id:
            logger.warning(f"No active A/B test with ID {test_id}")
            return None

        # Calculate metrics for each strategy
        cursor = self.metrics.conn.cursor()
        cursor.execute(
            """
            SELECT strategy_id, 
                   AVG(CAST(stored AS FLOAT) / NULLIF(discovered, 0)) as efficiency
            FROM collection_cycles
            WHERE strategy_id IN (?, ?)
            GROUP BY strategy_id
        """,
            (self._active_ab_test.strategy_a, self._active_ab_test.strategy_b),
        )

        metrics = {}
        for row in cursor.fetchall():
            metrics[row["strategy_id"]] = row["efficiency"] or 0

        metric_a = metrics.get(self._active_ab_test.strategy_a, 0)
        metric_b = metrics.get(self._active_ab_test.strategy_b, 0)

        # Determine winner
        winner = None
        if metric_a > metric_b * 1.1:  # 10% threshold
            winner = self._active_ab_test.strategy_a
        elif metric_b > metric_a * 1.1:
            winner = self._active_ab_test.strategy_b

        # Update result
        self._active_ab_test.metric_a = metric_a
        self._active_ab_test.metric_b = metric_b
        self._active_ab_test.winner = winner
        self._active_ab_test.ended_at = datetime.now()

        # Record in database
        cursor.execute(
            """
            UPDATE ab_tests 
            SET winner = ?, metric_a = ?, metric_b = ?, ended_at = ?
            WHERE test_id = ?
        """,
            (
                winner,
                metric_a,
                metric_b,
                datetime.now().isoformat(),
                test_id,
            ),
        )
        self.metrics.conn.commit()

        result = self._active_ab_test
        self._active_ab_test = None
        self._ab_test_strategy = None

        logger.info(
            f"Ended A/B test {test_id}: winner={winner}, A={metric_a:.3f}, B={metric_b:.3f}"
        )
        return result

    def get_performance_report(self, days: int = 30) -> dict[str, Any]:
        """
        Generate a comprehensive performance report.

        Args:
            days: Number of days to include in report

        Returns:
            Dict with performance metrics and insights
        """
        recent_cycles = self.metrics.get_recent_cycles(days)
        success_rate = self.metrics.get_hypothesis_success_rate(days)
        venue_rankings = self.metrics.get_venue_rankings()
        patterns = self.feedback.identify_success_patterns()
        current_config = self.optimizer.get_current_config()

        total_discovered = sum(c.discovered for c in recent_cycles)
        total_stored = sum(c.stored for c in recent_cycles)

        report = {
            "period_days": days,
            "collection_summary": {
                "total_cycles": len(recent_cycles),
                "total_discovered": total_discovered,
                "total_stored": total_stored,
                "storage_efficiency": total_stored / total_discovered
                if total_discovered > 0
                else 0,
            },
            "hypothesis_performance": {
                "success_rate": success_rate,
                "status": "good" if success_rate > 0.2 else "needs_improvement",
            },
            "current_thresholds": current_config.to_dict(),
            "top_venues": [
                {
                    "venue": v.venue,
                    "validation_rate": v.validation_rate,
                    "papers": v.papers_collected,
                }
                for v in venue_rankings[:5]
            ],
            "success_patterns": patterns,
            "recommendations": patterns.get("recommendations", []),
        }

        return report

    def close(self):
        """Close all database connections."""
        self.collector.close()
        self.metrics.close()
        self.hypothesis_db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
