"""
Fleming-AI Scheduler Runner
Manages continuous execution cycles with error recovery
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class FlemingRunner:
    """
    Main runner for Fleming-AI automation
    Orchestrates the continuous cycle of:
    1. Paper collection
    2. Hypothesis generation
    3. Validation
    4. Storage
    5. Synchronization
    """

    def __init__(
        self,
        cycle_delay: int = 3600,  # 1 hour between cycles
        max_retries: int = 3,
        retry_delay: int = 60,
        test_mode: bool = False,
    ):
        """
        Initialize Fleming runner

        Args:
            cycle_delay: Seconds to wait between successful cycles
            max_retries: Maximum retry attempts on failure
            retry_delay: Seconds to wait before retry
            test_mode: If True, use minimal data for quick testing
        """
        self.cycle_delay = cycle_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.test_mode = test_mode
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def run_once(self) -> bool:
        """
        Execute one complete cycle

        Returns:
            True if cycle completed successfully, False otherwise
        """
        cycle_start = datetime.now()
        logger.info(f"Starting cycle at {cycle_start}")

        try:
            # Step 1: Collect papers
            logger.info("Step 1/5: Collecting papers from arXiv...")
            await self._collect_papers()

            # Step 2: Generate hypotheses
            logger.info("Step 2/5: Generating hypotheses...")
            await self._generate_hypotheses()

            # Step 3: Validate hypotheses
            logger.info("Step 3/5: Validating hypotheses...")
            await self._validate_hypotheses()

            # Step 4: Store results
            logger.info("Step 4/5: Storing results...")
            await self._store_results()

            # Step 5: Synchronize data
            logger.info("Step 5/5: Synchronizing data...")
            await self._sync_data()

            cycle_end = datetime.now()
            duration = (cycle_end - cycle_start).total_seconds()
            logger.info(f"Cycle completed successfully in {duration:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Cycle failed: {e}", exc_info=True)
            return False

    async def run_forever(self):
        """
        Run continuous cycles with error recovery
        Continues until stop() is called
        """
        self._running = True
        logger.info("Starting continuous execution mode")

        while self._running:
            retry_count = 0
            success = False

            # Retry loop for failed cycles
            while retry_count < self.max_retries and not success:
                if not self._running:
                    break

                success = await self.run_once()

                if not success:
                    retry_count += 1
                    if retry_count < self.max_retries:
                        logger.warning(
                            f"Cycle failed, retrying in {self.retry_delay}s "
                            f"(attempt {retry_count}/{self.max_retries})"
                        )
                        try:
                            await asyncio.wait_for(
                                self._shutdown_event.wait(), timeout=self.retry_delay
                            )
                            break  # Shutdown requested
                        except asyncio.TimeoutError:
                            continue  # Retry

            if not success:
                logger.error(
                    f"Cycle failed after {self.max_retries} attempts, "
                    f"waiting {self.cycle_delay}s before next cycle"
                )

            # Wait before next cycle (unless shutting down)
            if self._running:
                logger.info(f"Waiting {self.cycle_delay}s before next cycle...")
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=self.cycle_delay)
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue to next cycle

        logger.info("Continuous execution stopped")

    def stop(self):
        """Signal graceful shutdown"""
        logger.info("Stop signal received")
        self._running = False
        self._shutdown_event.set()

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        # TODO: Close database connections, HTTP clients, etc.
        await asyncio.sleep(0.1)  # Allow pending tasks to complete
        logger.info("Cleanup complete")

    # Placeholder methods for each pipeline step
    # These will be implemented with actual logic later

    async def _collect_papers(self):
        """Collect papers from arXiv"""
        from src.collectors.arxiv_client import ArxivClient

        max_results = 3 if self.test_mode else 10

        with ArxivClient() as client:
            papers = client.search(
                query="cat:cs.AI OR cat:cs.LG",
                max_results=max_results,
                sort_by="submittedDate",
            )
            logger.info(f"Collected {len(papers)} papers from arXiv")
            return papers

    async def _generate_hypotheses(self):
        """Generate hypotheses from collected papers"""
        from src.llm.ollama_client import OllamaClient
        from src.generators.hypothesis import HypothesisGenerator
        from src.storage.vectordb import VectorDB
        from src.filters.quality import QualityFilter

        async with OllamaClient() as ollama:
            vectordb = VectorDB()
            quality_filter = QualityFilter()

            generator = HypothesisGenerator(
                ollama_client=ollama,
                vectordb=vectordb,
                quality_filter=quality_filter,
            )

            query = "machine learning" if self.test_mode else "artificial intelligence"
            k = 3 if self.test_mode else 10

            hypotheses = await generator.generate_hypotheses(query=query, k=k)
            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses

    async def _validate_hypotheses(self):
        """Validate generated hypotheses"""
        from src.llm.ollama_client import OllamaClient
        from src.storage.hypothesis_db import HypothesisDatabase
        from src.validators.pipeline import ValidationPipeline

        async with OllamaClient() as ollama:
            with HypothesisDatabase() as db:
                pipeline = ValidationPipeline(
                    ollama_client=ollama,
                    hypothesis_db=db,
                    sandbox_enabled=not self.test_mode,
                )

                pending = db.get_hypotheses_by_status("pending", limit=3 if self.test_mode else 10)

                if not pending:
                    logger.info("No pending hypotheses to validate")
                    return []

                results = await pipeline.validate_batch(pending)
                stats = pipeline.get_validation_stats(results)
                logger.info(f"Validated {len(results)} hypotheses: {stats}")
                return results

    async def _store_results(self):
        """Store results to database"""
        from src.storage.hypothesis_db import HypothesisDatabase

        with HypothesisDatabase() as db:
            total = db.count_hypotheses()
            pending = db.count_hypotheses("pending")
            validated = db.count_hypotheses("validated")
            rejected = db.count_hypotheses("rejected")

            logger.info(
                f"Database stats - Total: {total}, "
                f"Pending: {pending}, Validated: {validated}, Rejected: {rejected}"
            )

            return {
                "total": total,
                "pending": pending,
                "validated": validated,
                "rejected": rejected,
            }

    async def _sync_data(self):
        """Synchronize data with external systems"""
        if self.test_mode:
            logger.info("Skipping sync in test mode")
            return

        logger.info("Data sync placeholder - implement cloud backup/export as needed")
        await asyncio.sleep(0.1)
