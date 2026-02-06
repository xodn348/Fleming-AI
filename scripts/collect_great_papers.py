#!/usr/bin/env python3
"""
Script to collect great papers and save to database.

Usage:
    python scripts/collect_great_papers.py [--enrich] [--max-papers N]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.collectors.great_papers import GreatPapersCollector


def main():
    parser = argparse.ArgumentParser(description="Collect great papers and save to database")
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Enrich papers with Semantic Scholar citation data (slow, requires API)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between API requests in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/db/papers.db",
        help="Path to SQLite database (default: data/db/papers.db)",
    )

    args = parser.parse_args()

    # Create collector
    collector = GreatPapersCollector()

    # Run collection
    print("=" * 70)
    print("Fleming-AI: Great Papers Collection")
    print("=" * 70)
    print()

    if args.enrich:
        print("⚠️  Enrichment enabled - this will take a while due to API rate limits")
        print(f"   Delay between requests: {args.delay} seconds")
        if args.max_papers:
            print(f"   Processing only first {args.max_papers} papers")
    else:
        print("ℹ️  Enrichment disabled - papers will be saved without citation data")
        print("   Use --enrich flag to enable citation enrichment from Semantic Scholar")

    print()

    summary = collector.collect_and_save(
        db_path=args.db_path, enrich=args.enrich, max_papers=args.max_papers, delay=args.delay
    )

    print()
    print("=" * 70)
    print("Collection Summary")
    print("=" * 70)
    print(f"Total papers collected:  {summary['total_papers']}")
    print(f"Papers inserted to DB:   {summary['inserted']}")
    print(f"Papers skipped (dupes):  {summary['skipped']}")
    if args.enrich:
        print(f"Papers enriched:         {summary['enriched']}")
    print(f"Database location:       {summary['db_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
