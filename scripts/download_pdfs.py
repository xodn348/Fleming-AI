"""
Script to download PDFs from arXiv and parse them.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors.pdf_downloader import PDFDownloader
from src.parsers.pdf_parser import PDFParser
from src.storage.database import PaperDatabase


async def main():
    """Download PDFs for papers in database."""
    # Initialize database
    db_path = Path(__file__).parent.parent / "data" / "db" / "papers.db"
    db = PaperDatabase(db_path)

    # Get papers with arXiv IDs (limit to 10)
    papers = db.get_all_papers(limit=100)
    arxiv_papers = [p for p in papers if p.get("arxiv_id")][:10]

    print(f"Found {len(arxiv_papers)} papers with arXiv IDs (downloading 10)")

    # Initialize downloader
    downloader = PDFDownloader(output_dir=Path(__file__).parent.parent / "data" / "papers")

    # Download PDFs
    arxiv_ids = [p["arxiv_id"] for p in arxiv_papers]
    print(f"\nDownloading PDFs for: {', '.join(arxiv_ids[:3])}...")

    results = await downloader.batch_download(arxiv_ids)

    # Print results
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n✅ Downloaded {success_count}/{len(results)} PDFs")

    # Parse PDFs
    parser = PDFParser()
    parsed_dir = Path(__file__).parent.parent / "data" / "papers" / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    parse_success = 0
    for result in results:
        if result["status"] == "success":
            try:
                # Parse PDF
                parsed_data = parser.parse(result["path"])

                # Save parsed data
                arxiv_id = result["arxiv_id"]
                output_file = parsed_dir / f"{arxiv_id}.json"

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(parsed_data, f, indent=2, ensure_ascii=False)

                parse_success += 1
                print(f"✅ Parsed: {arxiv_id} - {parsed_data['title'][:60]}...")

            except Exception as e:
                print(f"❌ Failed to parse {result['arxiv_id']}: {e}")

    print(f"\n✅ Parsed {parse_success}/{success_count} PDFs")

    # Show summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total papers in DB: {len(papers)}")
    print(f"Papers with arXiv ID: {len([p for p in papers if p.get('arxiv_id')])}")
    print(f"PDFs downloaded: {success_count}/{len(results)}")
    print(f"PDFs parsed: {parse_success}/{success_count}")
    print(f"PDF location: data/papers/")
    print(f"Parsed data location: data/papers/parsed/")

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
