#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exporters.hypothesis_exporter import HypothesisExporter


def main():
    parser = argparse.ArgumentParser(description="Export Fleming-AI hypotheses")
    parser.add_argument(
        "--status",
        choices=["validated", "pending", "rejected", "all"],
        default="all",
        help="Filter by status (default: all)",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Export only top N by score",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Output directory (default: data/output)",
    )
    parser.add_argument(
        "--db-path",
        default="data/db/hypotheses.db",
        help="Database path (default: data/db/hypotheses.db)",
    )

    args = parser.parse_args()

    exporter = HypothesisExporter(args.output_dir)

    if args.top:
        print(f"Exporting top {args.top} hypotheses...")
        json_path, md_path = exporter.export_top(args.top, args.db_path)
    elif args.status == "all":
        print("Exporting all hypotheses...")
        json_path, md_path = exporter.export_latest(args.db_path)
    else:
        print(f"Exporting {args.status} hypotheses...")
        json_path, md_path = exporter.export_by_status(args.status, args.db_path)

    if json_path and md_path:
        print(f"✓ JSON: {json_path}")
        print(f"✓ Markdown: {md_path}")
    else:
        print("✗ No hypotheses to export")
        sys.exit(1)


if __name__ == "__main__":
    main()
