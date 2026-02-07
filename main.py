#!/usr/bin/env python3
"""
Fleming-AI Main Entry Point
Continuous execution loop for automated hypothesis generation and validation
"""

import argparse
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


async def run_collection(args):
    logger.info("=" * 80)
    logger.info("Fleming-AI Paper Collection")
    logger.info("=" * 80)

    runner = FlemingRunner(test_mode=args.test_mode)

    try:
        summary = await runner.run_collection_cycle()

        logger.info("=" * 80)
        logger.info("Collection Cycle Summary:")
        logger.info(f"  Discovered: {summary['discovered']} papers")
        logger.info(f"  Enriched:   {summary['enriched']} papers")
        logger.info(f"  Filtered:   {summary['filtered']} papers")
        logger.info(f"  Stored:     {summary['stored']} new papers")
        logger.info("=" * 80)

        sys.exit(0)
    except Exception as e:
        logger.error(f"Collection cycle failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await runner.cleanup()


async def run_main_cycle(args):
    logger.info("=" * 80)
    logger.info("Fleming-AI Starting")
    if args.test:
        logger.info("Running in TEST MODE")
    logger.info("=" * 80)

    runner = FlemingRunner(
        cycle_delay=args.cycle_delay,
        max_retries=args.max_retries,
        test_mode=args.test,
    )
    setup_signal_handlers(runner)

    try:
        if args.test:
            logger.info("Running single test cycle...")
            success = await runner.run_once()
            if success:
                logger.info("Test cycle completed successfully")
                sys.exit(0)
            else:
                logger.error("Test cycle failed")
                sys.exit(1)
        else:
            await runner.run_forever()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Fleming-AI shutting down")
        await runner.cleanup()


async def main():
    """Main entry point for Fleming-AI"""
    parser = argparse.ArgumentParser(description="Fleming-AI - Automated Research Discovery")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Main cycle command (default behavior)
    main_parser = subparsers.add_parser("run", help="Run main hypothesis generation cycle")
    main_parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (single short cycle)",
    )
    main_parser.add_argument(
        "--cycle-delay",
        type=int,
        default=3600,
        help="Seconds between cycles (default: 3600)",
    )
    main_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts on failure (default: 3)",
    )

    # Collection command
    collect_parser = subparsers.add_parser("collect", help="Run paper collection cycle")
    collect_parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with smaller limits",
    )

    args = parser.parse_args()

    # If no command specified, default to 'run' for backward compatibility
    if not args.command:
        args.command = "run"
        args.test = False
        args.cycle_delay = 3600
        args.max_retries = 3

    if args.command == "collect":
        await run_collection(args)
    elif args.command == "run":
        await run_main_cycle(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
        sys.exit(0)
