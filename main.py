#!/usr/bin/env python3
"""
Fleming-AI Main Entry Point
Continuous execution loop for automated hypothesis generation and validation
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from src.scheduler.runner import FlemingRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path.home() / "Library/Logs/fleming-ai.log"),
    ],
)

logger = logging.getLogger(__name__)


def setup_signal_handlers(runner: FlemingRunner):
    """Setup graceful shutdown handlers"""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        runner.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point for Fleming-AI"""
    logger.info("=" * 80)
    logger.info("Fleming-AI Starting")
    logger.info("=" * 80)

    runner = FlemingRunner()
    setup_signal_handlers(runner)

    try:
        # Run forever with automatic error recovery
        await runner.run_forever()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Fleming-AI shutting down")
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
        sys.exit(0)
