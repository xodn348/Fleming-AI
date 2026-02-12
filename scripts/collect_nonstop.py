#!/usr/bin/env python3
"""
Non-stop paper collection - no cycles, no delays, no enrichment bottlenecks
Continuously fetches from OpenAlex and stores directly
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import PaperDatabase
from src.storage.vectordb import VectorDB
from src.storage.direct_drive_storage import DirectDriveStorage
from src.collectors.tamu_proxy import TAMUProxy
from src.collectors.openalex_client import OpenAlexClient
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def main():
    logger.info("=" * 60)
    logger.info("📚 NON-STOP Paper Collection")
    logger.info("=" * 60)

    openalex = OpenAlexClient()
    vectordb = VectorDB()
    direct_storage = DirectDriveStorage()

    tamu_proxy = TAMUProxy()
    if tamu_proxy.is_available():
        logger.info("TAMU proxy available, authenticating...")
        await tamu_proxy.authenticate()
    else:
        tamu_proxy = None

    page_state_file = Path("data/collection_page.txt")
    if page_state_file.exists():
        page = int(page_state_file.read_text().strip())
        logger.info(f"Resuming from page {page}")
    else:
        page = 1

    collected_total = 0

    with PaperDatabase("data/db/papers.db") as db:
        existing = len(db.get_all_papers())
        logger.info(f"Starting with {existing} papers in DB")

    while True:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"🔍 Fetching page {page} from OpenAlex...")

            response = openalex.search(
                filters={"cited_by_count": ">50", "type": "article"},
                sort="cited_by_count:desc",
                per_page=50,
                page=page,
            )

            results = response.get("results", [])

            if not results:
                logger.info("No more results, resetting to page 1")
                page = 1
                continue

            logger.info(f"✓ Got {len(results)} papers")

            for i, work in enumerate(results, 1):
                try:
                    paper_data = {
                        "title": work.get("display_name") or work.get("title"),
                        "year": work.get("publication_year"),
                        "citations": work.get("cited_by_count", 0),
                        "doi": work.get("doi", "").replace("https://doi.org/", "")
                        if work.get("doi")
                        else None,
                    }

                    abstract_index = work.get("abstract_inverted_index", {})
                    if abstract_index:
                        words = []
                        for word, positions in abstract_index.items():
                            for pos in positions:
                                words.append((pos, word))
                        words.sort()
                        paper_data["abstract"] = " ".join([w[1] for w in words])

                    authorships = work.get("authorships", [])
                    if authorships:
                        authors = []
                        for auth in authorships:
                            author_info = auth.get("author", {})
                            name = author_info.get("display_name")
                            if name:
                                authors.append(name)
                        paper_data["authors"] = ", ".join(authors)

                    primary_location = work.get("primary_location", {})
                    pdf_url = primary_location.get("pdf_url")
                    if pdf_url and "arxiv.org" in pdf_url:
                        arxiv_id = pdf_url.split("/")[-1].replace(".pdf", "")
                        paper_data["arxiv_id"] = arxiv_id

                    with PaperDatabase("data/db/papers.db") as db:
                        if paper_data.get("doi"):
                            existing = [
                                p for p in db.get_all_papers() if p.get("doi") == paper_data["doi"]
                            ]
                            if existing:
                                logger.info(
                                    f"  [{i}/{len(results)}] Already exists: {paper_data['title'][:60]}..."
                                )
                                continue

                        paper_id = db.insert_paper(paper_data)
                        if paper_id:
                            logger.info(
                                f"  [{i}/{len(results)}] ✓ Stored: {paper_data['title'][:60]}..."
                            )
                            collected_total += 1
                        else:
                            logger.info(
                                f"  [{i}/{len(results)}] Already exists (duplicate title+year): {paper_data['title'][:60]}..."
                            )
                            continue

                    if paper_data.get("doi") or paper_data.get("arxiv_id"):
                        paper_id_str = paper_data.get("arxiv_id") or paper_data.get("doi")

                        if paper_id_str in vectordb.get_all_paper_ids():
                            logger.info(f"    Already embedded, skipping download")
                            continue

                        logger.info(f"    Downloading PDF...")
                        success = await direct_storage.download_and_upload_paper(
                            paper_data, vectordb, tamu_proxy
                        )

                        if success:
                            logger.info(f"    ✓ Downloaded, embedded, uploaded to Drive")
                        else:
                            logger.info(f"    ⚠ PDF unavailable, skipped")

                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"  [{i}/{len(results)}] Error processing paper: {e}")
                    continue

            with PaperDatabase("data/db/papers.db") as db:
                total_now = len(db.get_all_papers())
                logger.info(
                    f"\n📊 Progress: {total_now} papers in DB (+{collected_total} this session)"
                )

            page += 1
            page_state_file.write_text(str(page))

            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in collection loop: {e}", exc_info=True)
            await asyncio.sleep(10)
            continue

    if tamu_proxy:
        await tamu_proxy.close()


if __name__ == "__main__":
    asyncio.run(main())
