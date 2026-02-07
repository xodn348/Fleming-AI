"""
Fleming-AI Scheduler Runner
Manages continuous execution cycles with error recovery
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import Any, Dict, Optional

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
            # Step 1: Sample papers from VectorDB
            logger.info("Step 1/5: Sampling papers from VectorDB...")
            paper_ids = await self._sample_papers()

            # Step 2: Generate hypotheses
            logger.info("Step 2/5: Generating hypotheses...")
            await self._generate_hypotheses(paper_ids)

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

    async def add_quality_paper(self, paper_path: str) -> bool:
        """
        Manually add a quality paper to VectorDB.
        Called separately from main cycle, not during run_once().
        """
        from src.parsers.pdf_parser import PDFParser
        from src.storage.vectordb import VectorDB

        parser = PDFParser()
        paper_data = parser.parse(paper_path)

        if not paper_data:
            logger.error(f"Failed to parse: {paper_path}")
            return False

        vectordb = VectorDB()
        chunks_added = vectordb.add_papers([paper_data])
        logger.info(f"Added {chunks_added} chunks from {paper_path}")
        return True

    async def run_collection_cycle(self) -> Dict[str, Any]:
        """
        Run paper collection cycle (separate from hypothesis generation).
        Discovers new papers, enriches with citations, filters by quality, and stores.

        Returns:
            dict: Summary with discovered, enriched, filtered, stored counts
        """
        from src.collectors.paper_collector import PaperCollector

        logger.info("=== Starting Paper Collection Cycle ===")
        collector = PaperCollector()

        # Step 1: Discover candidates
        logger.info("Step 1/4: Discovering candidate papers...")
        limit = 10 if self.test_mode else 50
        candidates = await collector.discover(limit=limit)
        logger.info(f"  Discovered {len(candidates)} candidate papers")

        # Step 2: Enrich with citations
        logger.info("Step 2/4: Enriching with citation data...")
        enriched = await collector.enrich(candidates)
        logger.info(f"  Enriched {len(enriched)} papers")

        # Step 3: Filter by quality
        logger.info("Step 3/4: Filtering by quality score...")
        filtered = await collector.filter(enriched)
        logger.info(f"  {len(filtered)} papers passed quality filter")

        # Step 4: Store to database
        logger.info("Step 4/4: Storing to database...")
        stored_count = await collector.store(filtered)
        logger.info(f"  Stored {stored_count} new papers")

        summary = {
            "discovered": len(candidates),
            "enriched": len(enriched),
            "filtered": len(filtered),
            "stored": stored_count,
        }

        logger.info(f"=== Collection Cycle Complete: {summary} ===")
        return summary

    # Pipeline step methods

    async def _sample_papers(self) -> list[str]:
        """Sample papers from VectorDB for hypothesis exploration."""
        from src.storage.vectordb import VectorDB

        vectordb = VectorDB()
        all_paper_ids = vectordb.get_all_paper_ids()

        if len(all_paper_ids) < 2:
            logger.warning(f"Need at least 2 papers in VectorDB, found {len(all_paper_ids)}")
            return all_paper_ids

        sample_size = (
            min(5, len(all_paper_ids)) if not self.test_mode else min(3, len(all_paper_ids))
        )
        sampled = random.sample(all_paper_ids, sample_size)

        logger.info(f"Sampled {len(sampled)} papers from {len(all_paper_ids)} available")
        return sampled

    async def _collect_papers(self):
        """Collect papers from multiple sources (arXiv, Semantic Scholar, OpenAlex)"""
        from src.collectors.arxiv_client import ArxivClient
        from src.collectors.semantic_scholar_client import SemanticScholarClient
        from src.collectors.openalex_client import OpenAlexClient
        from src.storage.database import PaperDatabase
        import httpx

        max_results = 3 if self.test_mode else 10
        collected_papers = []

        # Try Source 1: arXiv
        try:
            with ArxivClient() as client:
                papers = client.search(
                    query="cat:cs.AI OR cat:cs.LG",
                    max_results=max_results,
                    sort_by="submittedDate",
                )
                collected_papers.extend(papers)
                logger.info(f"✓ arXiv: {len(papers)} papers")
        except Exception as e:
            logger.warning(f"✗ arXiv failed: {str(e)[:50]}")

        # Try Source 2: Semantic Scholar (if we need more)
        if len(collected_papers) < max_results:
            try:
                client = SemanticScholarClient()
                papers = await client.search(
                    query="machine learning",
                    limit=max_results - len(collected_papers),
                )
                collected_papers.extend(papers)
                logger.info(f"✓ Semantic Scholar: {len(papers)} papers")
                await client.close()
            except Exception as e:
                logger.warning(f"✗ Semantic Scholar failed: {str(e)[:50]}")

        # Try Source 3: OpenAlex (if we still need more)
        if len(collected_papers) < max_results:
            try:
                client = OpenAlexClient()
                papers = await client.search(
                    query="artificial intelligence",
                    limit=max_results - len(collected_papers),
                )
                collected_papers.extend(papers)
                logger.info(f"✓ OpenAlex: {len(papers)} papers")
                await client.close()
            except Exception as e:
                logger.warning(f"✗ OpenAlex failed: {str(e)[:50]}")

        # Fallback: Use existing papers from DB
        if not collected_papers:
            logger.warning("All sources failed, using existing papers from DB")
            with PaperDatabase("data/db/papers.db") as db:
                all_papers = db.get_all_papers()
                collected_papers = all_papers[:max_results] if all_papers else []
                logger.info(f"✓ Local DB: {len(collected_papers)} papers")

        logger.info(f"Total collected: {len(collected_papers)} papers from all sources")
        return collected_papers

    async def _generate_hypotheses(self, paper_ids: Optional[list[str]] = None):
        """Generate hypotheses from sampled papers."""
        from src.llm.ollama_client import OllamaClient
        from src.llm.advanced_llm import AdvancedLLM
        from src.generators.hypothesis import HypothesisGenerator
        from src.storage.vectordb import VectorDB
        from src.storage.hypothesis_db import HypothesisDatabase
        from src.filters.quality import QualityFilter

        async with OllamaClient() as ollama, AdvancedLLM() as advanced_llm:
            vectordb = VectorDB()
            quality_filter = QualityFilter()

            generator = HypothesisGenerator(
                ollama_client=ollama,
                vectordb=vectordb,
                quality_filter=quality_filter,
                advanced_llm=advanced_llm,
            )

            if paper_ids:
                hypotheses = await generator.generate_from_papers(paper_ids)
            else:
                sampled = await self._sample_papers()
                hypotheses = await generator.generate_from_papers(sampled)

            # Store hypotheses in database
            with HypothesisDatabase() as db:
                stored_count = 0
                for hyp in hypotheses:
                    if db.insert_hypothesis(hyp):
                        stored_count += 1
                logger.info(f"Stored {stored_count}/{len(hypotheses)} hypotheses in database")

            logger.info(
                f"Generated {len(hypotheses)} hypotheses from {len(paper_ids or [])} papers"
            )
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
