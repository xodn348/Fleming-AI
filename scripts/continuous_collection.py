#!/usr/bin/env python3
"""
Fleming-AI Continuous Collection
Runs paper collection and hypothesis generation continuously
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scheduler.runner import FlemingRunner

LOG_DIR = Path.home() / "Fleming-AI" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "continuous.log"),
    ],
)

logger = logging.getLogger(__name__)


class ContinuousRunner:
    """Runs collection and hypothesis generation continuously"""

    def __init__(self):
        self.runner = FlemingRunner(
            cycle_delay=3600,  # 1 hour between cycles
            test_mode=False
        )
        self._running = False
        self._shutdown_event = asyncio.Event()

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def stop(self):
        logger.info("Stopping continuous runner...")
        self._running = False
        self._shutdown_event.set()
        self.runner.stop()

    async def run(self):
        """Run continuously"""
        self._setup_signal_handlers()
        self._running = True

        logger.info("=" * 80)
        logger.info("Fleming-AI Continuous Collection Started")
        logger.info("Mode: Continuous (1 hour cycle)")
        logger.info("=" * 80)

        try:
            await self.runner.run_forever()
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            await self.runner.cleanup()
            logger.info("Continuous runner stopped")


async def main():
    runner = ContinuousRunner()
    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
