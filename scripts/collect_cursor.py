#!/usr/bin/env python3
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def main():
    logger.info("=" * 60)
    logger.info("📚 Cursor-Based Collection (Skip Duplicates)")
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

    cursor_state_file = Path("data/collection_cursor.txt")
    if cursor_state_file.exists():
        cursor = cursor_state_file.read_text().strip()
        if cursor == "DONE":
            logger.info("Collection complete")
            return
        logger.info(f"Resuming from saved cursor")
    else:
        cursor = "*"

    collected_total = 0
    consecutive_empty_batches = 0

    with PaperDatabase("data/db/papers.db") as db:
        existing = len(db.get_all_papers())
        logger.info(f"Starting with {existing} papers in DB")

    while cursor:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"🔍 Fetching batch...")

            response = openalex.search(
                filters={"cited_by_count": ">50", "type": "article"},
                sort="cited_by_count:desc",
                per_page=50,
                cursor=cursor,
            )

            results = response.get("results", [])
            if not results:
                logger.info("No more results")
                cursor_state_file.write_text("DONE")
                break

            logger.info(f"✓ Got {len(results)} papers")

            batch_new = 0
            with PaperDatabase("data/db/papers.db") as db:
                all_papers = db.get_all_papers()
                existing_dois = {p.get("doi") for p in all_papers if p.get("doi")}
                existing_titles = {
                    (p.get("title"), p.get("year")) for p in all_papers if p.get("title")
                }

            for i, work in enumerate(results, 1):
                doi = (
                    work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None
                )
                title = work.get("display_name") or work.get("title")
                year = work.get("publication_year")

                if doi and doi in existing_dois:
                    continue
                if (title, year) in existing_titles:
                    continue

                batch_new += 1
                logger.info(f"  [{i}/{len(results)}] NEW: {title[:60]}...")

            if batch_new == 0:
                consecutive_empty_batches += 1
                logger.info(
                    f"⚠️  No new papers ({consecutive_empty_batches} consecutive empty batches)"
                )

                if consecutive_empty_batches >= 10:
                    logger.warning(f"⏭️  Fast-forwarding 50 batches...")
                    for skip_count in range(50):
                        skip_response = openalex.search(
                            filters={"cited_by_count": ">50", "type": "article"},
                            sort="cited_by_count:desc",
                            per_page=50,
                            cursor=cursor,
                        )
                        cursor = skip_response.get("meta", {}).get("next_cursor")
                        if not cursor:
                            break
                        if (skip_count + 1) % 10 == 0:
                            logger.info(f"   Skipped {skip_count + 1} batches...")

                    consecutive_empty_batches = 0
                    if cursor:
                        cursor_state_file.write_text(cursor)
                        logger.info(f"✓ Saved fast-forwarded cursor")
                    else:
                        cursor_state_file.write_text("DONE")
                    continue
            else:
                consecutive_empty_batches = 0
                logger.info(f"✅ Found {batch_new} new papers in this batch!")
                break

            next_cursor = response.get("meta", {}).get("next_cursor")
            if next_cursor:
                cursor = next_cursor
                cursor_state_file.write_text(cursor)
            else:
                cursor_state_file.write_text("DONE")
                cursor = None

            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await asyncio.sleep(10)

    if cursor and cursor != "DONE":
        logger.info("=" * 60)
        logger.info(f"🎯 Found region with new papers!")
        logger.info(f"Restart collect_cursor.py to collect from here")
        logger.info("=" * 60)

    if tamu_proxy:
        await tamu_proxy.close()


if __name__ == "__main__":
    asyncio.run(main())
