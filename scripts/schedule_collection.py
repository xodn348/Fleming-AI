#!/usr/bin/env python3
"""
Fleming-AI Automated Collection Scheduler
Runs paper collection on a configurable schedule (daily/weekly/monthly)
Can run as a daemon or cron job with graceful shutdown support
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scheduler.runner import FlemingRunner

# Configure logging
LOG_DIR = Path.home() / "Fleming-AI" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "collection.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)

logger = logging.getLogger(__name__)


class CollectionScheduler:
    """
    Manages scheduled paper collection cycles
    Supports daily, weekly, and monthly frequencies
    """

    # Frequency to seconds mapping
    FREQUENCIES = {
        "daily": 24 * 3600,  # 86400 seconds
        "weekly": 7 * 24 * 3600,  # 604800 seconds
        "monthly": 30 * 24 * 3600,  # 2592000 seconds
    }

    def __init__(self, frequency: str = "weekly", test_mode: bool = False):
        """
        Initialize scheduler

        Args:
            frequency: 'daily', 'weekly', or 'monthly'
            test_mode: If True, use minimal data for quick testing
        """
        if frequency not in self.FREQUENCIES:
            raise ValueError(
                f"Invalid frequency: {frequency}. Must be one of {list(self.FREQUENCIES.keys())}"
            )

        self.frequency = frequency
        self.interval_seconds = self.FREQUENCIES[frequency]
        self.test_mode = test_mode
        self.runner = FlemingRunner(test_mode=test_mode)
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_run = None

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers for SIGTERM and SIGINT"""

        def signal_handler(signum, frame):
            signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
            logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _check_paper_count(self) -> int:
        """Check current paper count in database"""
        from src.storage.database import PaperDatabase

        db_path = Path.home() / "Fleming-AI" / "data" / "db" / "papers.db"
        db = PaperDatabase(db_path)
        count = db.count_papers()
        db.close()
        return count

    async def run_collection(self) -> bool:
        """
        Execute one collection cycle
        Stops automatically if 1000 papers reached

        Returns:
            True if successful, False otherwise
        """
        try:
            paper_count = await self._check_paper_count()

            if paper_count >= 1000:
                logger.info("=" * 80)
                logger.info(f"🎯 Paper collection target reached: {paper_count}/1000")
                logger.info("Skipping paper collection (target met)")
                logger.info("Continuing hypothesis generation...")
                logger.info("=" * 80)
            else:
                logger.info("=" * 80)
                logger.info(f"Starting collection cycle ({self.frequency})")
                logger.info(f"Current papers: {paper_count}/1000")
                logger.info("=" * 80)

                collection_result = await self.runner.run_collection_cycle()

                logger.info("=" * 80)
                logger.info(f"Collection cycle completed: {collection_result}")
                logger.info("=" * 80)

            logger.info("=" * 80)
            logger.info("Starting hypothesis generation")
            logger.info("=" * 80)

            hypothesis_result = await self.runner.run_once()

            logger.info("=" * 80)
            logger.info(f"Hypothesis generation completed: {hypothesis_result}")
            logger.info("=" * 80)

            self._last_run = datetime.now()
            return True

        except Exception as e:
            logger.error(f"Collection cycle failed: {e}", exc_info=True)
            return False

    async def run_once(self):
        """Run collection once and exit (for cron jobs)"""
        logger.info(f"Running collection once (--once mode)")
        success = await self.run_collection()
        sys.exit(0 if success else 1)

    async def run_scheduled(self):
        """
        Run collection on schedule indefinitely
        Continues until stop() is called
        """
        self._running = True
        self._setup_signal_handlers()

        logger.info("=" * 80)
        logger.info(f"Collection Scheduler Started")
        logger.info(f"Frequency: {self.frequency} ({self.interval_seconds}s)")
        logger.info(f"Test Mode: {self.test_mode}")
        logger.info("=" * 80)

        next_run = datetime.now()

        while self._running:
            now = datetime.now()

            # Check if it's time to run
            if now >= next_run:
                success = await self.run_collection()

                # Schedule next run
                next_run = datetime.now() + timedelta(seconds=self.interval_seconds)
                logger.info(
                    f"Next collection scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            else:
                # Calculate wait time
                wait_seconds = (next_run - now).total_seconds()
                logger.info(f"Waiting {wait_seconds:.0f}s until next collection...")

                # Wait with interruptible timeout
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=min(wait_seconds, 60)
                    )
                    # Shutdown requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue

        logger.info("Collection scheduler stopped")

    def stop(self):
        """Signal graceful shutdown"""
        self._running = False
        self._shutdown_event.set()

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        await self.runner.cleanup()
        logger.info("Cleanup complete")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fleming-AI Collection Scheduler - Automated paper collection on schedule"
    )
    parser.add_argument(
        "--frequency",
        choices=["daily", "weekly", "monthly"],
        default="weekly",
        help="Collection frequency (default: weekly)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run collection once and exit (for cron jobs)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (minimal data)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Fleming-AI Collection Scheduler")
    logger.info(f"Frequency: {args.frequency}")
    logger.info(f"Once mode: {args.once}")
    logger.info(f"Test mode: {args.test}")
    logger.info("=" * 80)

    scheduler = CollectionScheduler(frequency=args.frequency, test_mode=args.test)

    try:
        if args.once:
            # Run once and exit
            await scheduler.run_once()
        else:
            # Run scheduled indefinitely
            await scheduler.run_scheduled()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Shutting down...")
        await scheduler.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
        sys.exit(0)
