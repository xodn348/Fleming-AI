"""
Hypothesis Exporter - Export hypotheses to JSON and Markdown formats
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.generators.hypothesis import Hypothesis
from src.storage.hypothesis_db import HypothesisDatabase

logger = logging.getLogger(__name__)


class HypothesisExporter:
    """Export hypotheses to various formats (JSON, Markdown)."""

    def __init__(self, output_dir: Path | str = "data/output"):
        """
        Initialize exporter.

        Args:
            output_dir: Directory to write export files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_json(
        self,
        hypotheses: list[Hypothesis],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export hypotheses to JSON format.

        Args:
            hypotheses: List of hypotheses to export
            filename: Output filename (default: hypotheses_YYYYMMDD_HHMMSS.json)

        Returns:
            Path to created JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hypotheses_{timestamp}.json"

        output_path = self.output_dir / filename

        # Convert to dict format
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_count": len(hypotheses),
            "hypotheses": [h.to_dict() for h in hypotheses],
        }

        # Write JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(hypotheses)} hypotheses to {output_path}")
        return output_path

    def export_to_markdown(
        self,
        hypotheses: list[Hypothesis],
        filename: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Path:
        """
        Export hypotheses to Markdown format (human-readable).

        Args:
            hypotheses: List of hypotheses to export
            filename: Output filename (default: hypotheses_YYYYMMDD_HHMMSS.md)
            include_metadata: Include metadata section at top

        Returns:
            Path to created Markdown file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hypotheses_{timestamp}.md"

        output_path = self.output_dir / filename

        lines = []

        # Header
        lines.append("# Fleming-AI Generated Hypotheses\n")

        # Metadata section
        if include_metadata:
            validated = sum(1 for h in hypotheses if h.status == "validated")
            pending = sum(1 for h in hypotheses if h.status == "pending")
            rejected = sum(1 for h in hypotheses if h.status == "rejected")

            lines.append("## Summary\n")
            lines.append(f"- **Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **Total Hypotheses:** {len(hypotheses)}")
            lines.append(f"- **Validated:** {validated}")
            lines.append(f"- **Pending:** {pending}")
            lines.append(f"- **Rejected:** {rejected}\n")
            lines.append("---\n")

        # Sort hypotheses by score (quality + confidence)
        sorted_hyps = sorted(
            hypotheses,
            key=lambda h: (h.quality_score + h.confidence) / 2,
            reverse=True,
        )

        # Group by status
        for status in ["validated", "pending", "rejected"]:
            status_hyps = [h for h in sorted_hyps if h.status == status]
            if not status_hyps:
                continue

            lines.append(f"## {status.capitalize()} Hypotheses ({len(status_hyps)})\n")

            for i, hyp in enumerate(status_hyps, 1):
                # Hypothesis header
                combined_score = (hyp.quality_score + hyp.confidence) / 2
                lines.append(f"### {i}. Hypothesis ID: `{hyp.id[:8]}`\n")

                # Scores
                lines.append("**Scores:**")
                lines.append(
                    f"- Confidence: {hyp.confidence:.2f} | "
                    f"Quality: {hyp.quality_score:.3f} | "
                    f"Combined: {combined_score:.3f}"
                )
                lines.append("")

                # Hypothesis text
                lines.append("**Hypothesis:**")
                lines.append(f"> {hyp.hypothesis_text}\n")

                # Connection details
                if hyp.connection:
                    lines.append("**Connection:**")
                    concept_a = hyp.connection.get("concept_a", "N/A")
                    concept_b = hyp.connection.get("concept_b", "N/A")
                    bridging = hyp.connection.get("bridging_concept", "N/A")
                    lines.append(f"- Concept A: `{concept_a}`")
                    lines.append(f"- Concept B: `{concept_b}`")
                    lines.append(f"- Bridging Concept: `{bridging}`")
                    lines.append("")

                # Source papers
                if hyp.source_papers:
                    lines.append(f"**Source Papers:** {len(hyp.source_papers)} papers")
                    for paper_id in hyp.source_papers[:3]:  # Show first 3
                        lines.append(f"- `{paper_id}`")
                    if len(hyp.source_papers) > 3:
                        lines.append(f"- ... and {len(hyp.source_papers) - 3} more")
                    lines.append("")

                # Metadata
                lines.append(f"**Created:** {hyp.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append("")
                lines.append("---\n")

        # Write Markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Exported {len(hypotheses)} hypotheses to {output_path}")
        return output_path

    def export_latest(
        self,
        db_path: str | Path = "data/db/hypotheses.db",
        json_filename: str = "hypotheses_latest.json",
        md_filename: str = "hypotheses_latest.md",
    ) -> tuple[Path, Path]:
        """
        Export all hypotheses from database to both JSON and Markdown.

        Args:
            db_path: Path to hypothesis database
            json_filename: JSON output filename
            md_filename: Markdown output filename

        Returns:
            Tuple of (json_path, markdown_path)
        """
        with HypothesisDatabase(db_path) as db:
            hypotheses = db.get_all_hypotheses()

        if not hypotheses:
            logger.warning("No hypotheses to export")
            return None, None

        json_path = self.export_to_json(hypotheses, json_filename)
        md_path = self.export_to_markdown(hypotheses, md_filename)

        return json_path, md_path

    def export_by_status(
        self,
        status: str,
        db_path: str | Path = "data/db/hypotheses.db",
        limit: Optional[int] = None,
    ) -> tuple[Path, Path]:
        """
        Export hypotheses filtered by status.

        Args:
            status: 'validated', 'pending', or 'rejected'
            db_path: Path to hypothesis database
            limit: Maximum number to export

        Returns:
            Tuple of (json_path, markdown_path)
        """
        with HypothesisDatabase(db_path) as db:
            hypotheses = db.get_hypotheses_by_status(status, limit=limit)

        if not hypotheses:
            logger.warning(f"No {status} hypotheses to export")
            return None, None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"hypotheses_{status}_{timestamp}.json"
        md_filename = f"hypotheses_{status}_{timestamp}.md"

        json_path = self.export_to_json(hypotheses, json_filename)
        md_path = self.export_to_markdown(hypotheses, md_filename)

        return json_path, md_path

    def export_top(
        self,
        limit: int = 10,
        db_path: str | Path = "data/db/hypotheses.db",
        min_confidence: float = 0.0,
        min_quality: float = 0.0,
    ) -> tuple[Path, Path]:
        """
        Export top N hypotheses by combined score.

        Args:
            limit: Number of top hypotheses to export
            db_path: Path to hypothesis database
            min_confidence: Minimum confidence threshold
            min_quality: Minimum quality threshold

        Returns:
            Tuple of (json_path, markdown_path)
        """
        with HypothesisDatabase(db_path) as db:
            hypotheses = db.get_top_hypotheses(
                limit=limit,
                min_confidence=min_confidence,
                min_quality=min_quality,
            )

        if not hypotheses:
            logger.warning("No hypotheses meet the criteria")
            return None, None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"hypotheses_top{limit}_{timestamp}.json"
        md_filename = f"hypotheses_top{limit}_{timestamp}.md"

        json_path = self.export_to_json(hypotheses, json_filename)
        md_path = self.export_to_markdown(hypotheses, md_filename)

        return json_path, md_path
