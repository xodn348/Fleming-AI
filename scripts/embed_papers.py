#!/usr/bin/env python3
"""
Script to embed parsed papers into ChromaDB vector database.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from storage.vectordb import VectorDB


def load_parsed_papers(parsed_dir: Path) -> list[dict]:
    """
    Load all parsed papers from JSON files.

    Args:
        parsed_dir: Directory containing parsed paper JSON files

    Returns:
        List of paper data dictionaries
    """
    papers = []

    for json_file in parsed_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
                # Add paper_id from filename
                paper_data["paper_id"] = json_file.stem
                papers.append(paper_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return papers


def main():
    """Main function to embed papers."""
    # Paths
    project_root = Path(__file__).parent.parent
    parsed_dir = project_root / "data" / "papers" / "parsed"

    if not parsed_dir.exists():
        print(f"Error: Parsed papers directory not found: {parsed_dir}")
        return

    # Load papers
    print(f"Loading papers from {parsed_dir}...")
    papers = load_parsed_papers(parsed_dir)
    print(f"Loaded {len(papers)} papers")

    if not papers:
        print("No papers found to embed")
        return

    # Initialize VectorDB
    print("\nInitializing VectorDB...")
    db = VectorDB()

    # Check if database already has data
    existing_count = db.count()
    if existing_count > 0:
        print(f"Database already contains {existing_count} chunks")
        response = input("Reset database and re-embed? (y/n): ")
        if response.lower() == "y":
            print("Resetting database...")
            db.reset()
        else:
            print("Keeping existing data")
            return

    # Embed papers
    print(f"\nEmbedding {len(papers)} papers...")
    print("This may take a few minutes...")

    total_chunks = db.add_papers(papers)

    print(f"\n✓ Successfully embedded {len(papers)} papers")
    print(f"✓ Total chunks stored: {total_chunks}")
    print(f"✓ Average chunks per paper: {total_chunks / len(papers):.1f}")

    # Test search
    print("\n--- Testing search functionality ---")
    test_query = "deep learning convolutional neural networks"
    print(f"Query: '{test_query}'")

    results = db.search(test_query, k=3)
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Paper: {result['metadata'].get('title', 'Unknown')}")
        print(f"   Section: {result['metadata'].get('section', 'Unknown')}")
        print(f"   Distance: {result.get('distance', 'N/A'):.4f}")
        print(f"   Text preview: {result['text'][:150]}...")


if __name__ == "__main__":
    main()
