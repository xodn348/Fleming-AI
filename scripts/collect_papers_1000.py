#!/usr/bin/env python3
"""
Collect papers until reaching 1000 total
Drive-first strategy: PDFs uploaded to Drive and deleted locally
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scheduler.runner import FlemingRunner
from src.storage.database import PaperDatabase
from src.storage.vectordb import VectorDB
from src.storage.lean_storage import LeanStorageManager
from src.storage.direct_drive_storage import DirectDriveStorage
from src.collectors.tamu_proxy import TAMUProxy
from src.monitoring.resource_monitor import ResourceMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path.home() / "Fleming-AI" / "logs" / "paper_collection.log"),
    ],
)

logger = logging.getLogger(__name__)


async def main():
    runner = FlemingRunner(cycle_delay=0, test_mode=False)

    base_dir = Path.home() / "Fleming-AI"
    papers_dir = base_dir / "data" / "papers"
    db_dir = base_dir / "data" / "db"

    lean_manager = LeanStorageManager(papers_dir, db_dir)
    direct_storage = DirectDriveStorage()
    vectordb = VectorDB()
    monitor = ResourceMonitor(cpu_threshold=70.0, memory_threshold=80.0)

    tamu_proxy = TAMUProxy()
    if tamu_proxy.is_available():
        logger.info("TAMU proxy available, authenticating...")
        await tamu_proxy.authenticate()
    else:
        tamu_proxy = None

    logger.info("=" * 60)
    logger.info("📚 Paper Collection Mission: 1000 papers")
    logger.info("=" * 60)

    cycle_count = 0

    while True:
        with PaperDatabase("data/db/papers.db") as db:
            papers = db.get_all_papers()
            count = len(papers)
            logger.info(f"Progress: {count}/1000 papers")

            if count >= 1000:
                logger.info("🎉 TARGET REACHED: 1000 papers collected!")
                logger.info("Final DB backup to Drive...")
                lean_manager.backup_db_to_drive()
                break

        try:
            if cycle_count % 5 == 0:
                monitor.log_stats()

            if monitor.should_backoff():
                logger.warning("System under load, skipping this cycle")
                await asyncio.sleep(300)
                continue

            logger.info("Starting paper collection cycle...")
            summary = await runner.run_collection_cycle()
            logger.info(f"Cycle complete: {summary}")

            cycle_count += 1

            stored_count = summary.get("stored", 0)

            if stored_count > 0:
                with PaperDatabase("data/db/papers.db") as db:
                    all_papers = db.get_all_papers()
                    recent_papers = all_papers[-stored_count:]

                    logger.info(f"Processing {len(recent_papers)} new papers (lean pipeline)...")
                    processed_count = 0

                    for i, paper in enumerate(recent_papers, 1):
                        paper_id = (
                            paper.get("arxiv_id") or paper.get("doi") or f"paper_{paper.get('id')}"
                        )

                        if paper_id in vectordb.get_all_paper_ids():
                            logger.info(
                                f"  [{i}/{len(recent_papers)}] Already embedded, skipping: {paper.get('title', 'unknown')[:60]}..."
                            )
                            continue

                        if not (paper.get("arxiv_id") or paper.get("doi")):
                            logger.warning(
                                f"  [{i}/{len(recent_papers)}] No download link, skipping: {paper.get('title', 'unknown')[:60]}..."
                            )
                            continue

                        logger.info(
                            f"  [{i}/{len(recent_papers)}] {paper.get('title', 'unknown')[:60]}..."
                        )

                        success = await direct_storage.download_and_upload_paper(
                            paper, vectordb, tamu_proxy
                        )
                        if success:
                            logger.info(f"    ✓ Downloaded, embedded, uploaded to Drive")
                            processed_count += 1
                        else:
                            logger.warning(f"    ✗ Failed to process")

                        await asyncio.sleep(1)

                    if processed_count > 0:
                        logger.info(
                            f"✓ {processed_count} papers processed (uploaded to Drive, deleted locally)"
                        )
            else:
                logger.info("No new papers stored, skipping lean pipeline")

            if cycle_count % 2 == 0:
                logger.info("Periodic DB backup to Drive...")
                lean_manager.backup_db_to_drive()

        except Exception as e:
            logger.error(f"Collection cycle failed: {e}", exc_info=True)

        await asyncio.sleep(60)

    if tamu_proxy:
        await tamu_proxy.close()

    logger.info("Paper collection complete. Exiting.")


if __name__ == "__main__":
    asyncio.run(main())
