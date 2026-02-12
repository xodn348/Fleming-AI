#!/usr/bin/env python3
"""
Continuously generate and validate hypotheses (runs forever)
Periodically backs up DB to Drive
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.llm.groq_client import GroqClient
from src.generators.hypothesis import HypothesisGenerator
from src.storage.vectordb import VectorDB
from src.storage.hypothesis_db import HypothesisDatabase
from src.filters.quality import QualityFilter
from src.validators.pipeline import ValidationPipeline
from src.storage.lean_storage import LeanStorageManager
from src.exporters.hypothesis_exporter import HypothesisExporter
from src.monitoring.resource_monitor import ResourceMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path.home() / "Fleming-AI" / "logs" / "hypothesis_generation.log"),
    ],
)

logger = logging.getLogger(__name__)


async def main():
    logger.info("=" * 60)
    logger.info("💡 Continuous Hypothesis Generation Started")
    logger.info("=" * 60)

    base_dir = Path.home() / "Fleming-AI"
    db_dir = base_dir / "data" / "db"
    papers_dir = base_dir / "data" / "papers"
    output_dir = base_dir / "data" / "output"

    lean_manager = LeanStorageManager(papers_dir, db_dir)
    exporter = HypothesisExporter(output_dir)
    monitor = ResourceMonitor(cpu_threshold=70.0, memory_threshold=80.0)

    async with GroqClient() as groq:
        vectordb = VectorDB()
        quality_filter = QualityFilter()

        generator = HypothesisGenerator(
            llm_client=groq,
            vectordb=vectordb,
            quality_filter=quality_filter,
        )

        cycle_count = 0
        batch_size = 10
        target_validated = 100
        processed_paper_idx = 0

        while True:
            cycle_count += 1

            if cycle_count % 5 == 0:
                monitor.log_stats()

            if monitor.should_backoff():
                logger.warning("System under load, waiting 5 minutes")
                await asyncio.sleep(300)
                continue

            with HypothesisDatabase() as db:
                validated_count = db.count_hypotheses("validated")

            if validated_count >= target_validated:
                logger.info(
                    f"Target reached! {validated_count}/{target_validated} validated hypotheses"
                )
                logger.info("Exporting final results...")
                exporter.export_latest()
                break

            logger.info(
                f"Cycle {cycle_count}: Progress {validated_count}/{target_validated} validated"
            )

            try:
                all_paper_ids = vectordb.get_all_paper_ids()
                if len(all_paper_ids) < 2:
                    logger.warning("Need at least 2 papers. Waiting...")
                    await asyncio.sleep(60)
                    continue

                if processed_paper_idx >= len(all_paper_ids):
                    processed_paper_idx = 0

                batch_end = min(processed_paper_idx + batch_size, len(all_paper_ids))
                batch_ids = all_paper_ids[processed_paper_idx:batch_end]
                processed_paper_idx = batch_end

                logger.info(
                    f"Processing papers {processed_paper_idx - len(batch_ids)}-{processed_paper_idx} of {len(all_paper_ids)}"
                )

                hypotheses = await asyncio.wait_for(
                    generator.generate_from_papers(batch_ids), timeout=300
                )

                await asyncio.sleep(2)

                with HypothesisDatabase() as db:
                    stored = 0
                    for hyp in hypotheses:
                        if db.insert_hypothesis(hyp):
                            stored += 1

                    total = db.count_hypotheses()
                    pending = db.count_hypotheses("pending")
                    validated = db.count_hypotheses("validated")

                    logger.info(f"Generated {len(hypotheses)}, Stored {stored}")
                    logger.info(f"Total: {total} ({pending} pending, {validated} validated)")

                logger.info("Validating pending hypotheses...")
                with HypothesisDatabase() as db:
                    pending_hyps = db.get_hypotheses_by_status("pending", limit=5)
                    if pending_hyps:
                        pipeline = ValidationPipeline(groq, db, sandbox_enabled=False)
                        results = await pipeline.validate_batch(pending_hyps)
                        logger.info(f"Validated {len(results)} hypotheses")

                await asyncio.sleep(2)

                logger.info("Exporting hypotheses to output/...")
                exporter.export_latest()
                logger.info("Syncing output to Drive...")
                lean_manager.gdrive.sync_to_drive(output_dir, "gdrive:Fleming-AI/output")

                if cycle_count % 5 == 0:
                    logger.info("Periodic DB backup to Drive...")
                    lean_manager.backup_db_to_drive()

            except asyncio.TimeoutError:
                logger.error("Hypothesis generation timed out (300s)")
            except Exception as e:
                logger.error(f"Cycle failed: {e}")

            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
