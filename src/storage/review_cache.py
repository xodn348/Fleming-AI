"""
File-based review caching with TTL to avoid redundant API calls.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ReviewCache:
    """Cache manager for storing and retrieving review results with TTL."""

    def __init__(self, cache_dir: str | Path = ".cache/reviews", ttl_days: int = 7):
        """
        Initialize review cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_days: Time-to-live in days (default 7)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        self.prompt_version = self._compute_prompt_version()
        logger.debug(
            f"ReviewCache initialized: dir={self.cache_dir}, ttl={ttl_days}d, "
            f"prompt_version={self.prompt_version[:8]}..."
        )

    def _compute_prompt_version(self) -> str:
        """
        Compute SHA256 hash of concatenated prompts from knowledge.py.

        Returns:
            SHA256 hash of all prompts
        """
        try:
            from src.reviewers.knowledge import ALL_PROMPTS

            # Concatenate all prompts
            combined = "".join(ALL_PROMPTS)
            # Hash the combined string
            return hashlib.sha256(combined.encode()).hexdigest()
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to compute prompt version: {e}. Using fallback.")
            # Fallback: use a static hash if import fails
            return hashlib.sha256(b"fallback_prompt_version").hexdigest()

    def _cache_key(self, paper_hash: str, stage: str) -> str:
        """
        Generate cache key from paper_hash, stage, and prompt_version.

        Args:
            paper_hash: Hash of the paper
            stage: Review stage (e.g., 'hypothesis', 'experiment_design', 'results', 'paper')

        Returns:
            SHA256 hash used as cache key
        """
        key_input = f"{paper_hash}:{stage}:{self.prompt_version}"
        return hashlib.sha256(key_input.encode()).hexdigest()

    def store(self, paper_hash: str, stage: str, review_data: dict) -> bool:
        """
        Store review result in cache.

        Args:
            paper_hash: Hash of the paper
            stage: Review stage
            review_data: Review result dictionary (must have 'verdict' != 'error')

        Returns:
            True if stored successfully, False if verdict is 'error'
        """
        # Don't cache error verdicts
        if review_data.get("verdict") == "error":
            logger.debug(f"Skipping cache for error verdict: {paper_hash}/{stage}")
            return False

        cache_key = self._cache_key(paper_hash, stage)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Prepare cache entry with metadata
        cache_entry = {
            "paper_hash": paper_hash,
            "stage": stage,
            "prompt_version": self.prompt_version,
            "cached_at": datetime.now().isoformat(),
            "review_data": review_data,
        }

        try:
            with open(cache_file, "w") as f:
                json.dump(cache_entry, f, indent=2)
            logger.debug(f"Cached review: {paper_hash}/{stage} -> {cache_key[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to store cache: {e}")
            return False

    def lookup(self, paper_hash: str, stage: str) -> Optional[dict]:
        """
        Lookup review result in cache.

        Args:
            paper_hash: Hash of the paper
            stage: Review stage

        Returns:
            Review data dict if found and not expired, None otherwise
        """
        cache_key = self._cache_key(paper_hash, stage)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            logger.debug(f"Cache miss: {paper_hash}/{stage}")
            return None

        try:
            with open(cache_file, "r") as f:
                cache_entry = json.load(f)

            # Check TTL
            cached_at = datetime.fromisoformat(cache_entry["cached_at"])
            age = datetime.now() - cached_at

            if age > self.ttl:
                logger.debug(
                    f"Cache expired: {paper_hash}/{stage} (age={age.days}d, ttl={self.ttl.days}d)"
                )
                # Delete expired entry
                cache_file.unlink()
                return None

            logger.debug(f"Cache hit: {paper_hash}/{stage}")
            return cache_entry["review_data"]

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def invalidate(self, paper_hash: str) -> int:
        """
        Invalidate all cache entries for a paper.

        Args:
            paper_hash: Hash of the paper

        Returns:
            Number of cache files deleted
        """
        deleted_count = 0

        # Find all cache files for this paper_hash
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    cache_entry = json.load(f)

                if cache_entry.get("paper_hash") == paper_hash:
                    cache_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Invalidated: {cache_file.name}")

            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {e}")

        logger.info(f"Invalidated {deleted_count} cache entries for {paper_hash}")
        return deleted_count

    def clear_all(self) -> int:
        """
        Clear all cache files.

        Returns:
            Number of cache files deleted
        """
        deleted_count = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_file}: {e}")

        logger.info(f"Cleared {deleted_count} cache files")
        return deleted_count
