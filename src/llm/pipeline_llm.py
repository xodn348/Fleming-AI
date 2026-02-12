"""
Pipeline LLM - Unified LLM interface for Fleming-AI
Wraps BackendSwitcher to provide simple async LLM access with automatic fallback
"""

import logging
import os
from typing import Optional, AsyncIterator

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from src.llm.backend_switcher import BackendSwitcher
from src.llm.gemini_client import GeminiClient
from src.llm.groq_client import GroqClient
from src.llm.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class PipelineLLM:
    """
    Unified LLM interface for Fleming-AI pipeline

    Automatically loads API keys from .env and uses BackendSwitcher
    for reliable LLM access with fallback across multiple providers.

    Priority order: Gemini → Groq → OpenRouter

    Usage:
        llm = PipelineLLM()
        result = await llm.generate("Say hello", max_tokens=50)

        # Validate all backends
        status = await llm.validate_backends()
        print(status)  # {"gemini": True, "groq": False, "openrouter": True}
    """

    def __init__(self):
        """
        Initialize PipelineLLM with API keys from environment

        Loads keys from .env:
        - GOOGLE_API_KEY for Gemini
        - GROQ_API_KEY for Groq
        - OPENROUTER_API_KEY for OpenRouter
        """
        if load_dotenv is not None:
            load_dotenv()

        self.gemini_key = os.getenv("GOOGLE_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")

        available_keys = sum(
            [
                bool(self.gemini_key),
                bool(self.groq_key),
                bool(self.openrouter_key),
            ]
        )

        if available_keys == 0:
            raise ValueError(
                "No API keys found. Please set at least one of: "
                "GOOGLE_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY"
            )

        logger.info(f"PipelineLLM initialized with {available_keys}/3 API keys configured")

        self.switcher = BackendSwitcher()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the backend switcher and cleanup resources"""
        await self.switcher.close()

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> str | AsyncIterator[str]:
        """
        Generate text with automatic backend fallback

        Args:
            prompt: User prompt text
            system: System prompt (optional)
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            max_tokens: Maximum tokens to generate (default: backend-specific)
            stream: Whether to stream the response (default: False)
            **kwargs: Additional backend-specific parameters

        Returns:
            Generated text string, or AsyncIterator[str] if streaming

        Raises:
            RuntimeError: If all backends fail
        """
        return await self.switcher.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

    async def validate_backends(self) -> dict[str, bool]:
        """
        Test all three API backends with simple prompts

        Returns:
            Dictionary mapping backend name to availability status
            Example: {"gemini": True, "groq": False, "openrouter": True}
        """
        results = {}

        if self.gemini_key:
            try:
                client = GeminiClient(api_key=self.gemini_key)
                response = await client.generate("Hi", max_tokens=50)
                results["gemini"] = bool(response and len(response) > 0)
                logger.info("✓ Gemini backend validated")
            except Exception as e:
                logger.warning(f"✗ Gemini validation failed: {e}")
                results["gemini"] = False
        else:
            logger.info("○ Gemini key not configured")
            results["gemini"] = False

        if self.groq_key:
            try:
                async with GroqClient(api_key=self.groq_key) as client:
                    response = await client.generate("Hi", max_tokens=50)
                results["groq"] = bool(response and len(response) > 0)
                logger.info("✓ Groq backend validated")
            except Exception as e:
                logger.warning(f"✗ Groq validation failed: {e}")
                results["groq"] = False
        else:
            logger.info("○ Groq key not configured")
            results["groq"] = False

        if self.openrouter_key:
            try:
                async with OpenRouterClient(api_key=self.openrouter_key) as client:
                    response = await client.generate("Hi", max_tokens=50)
                results["openrouter"] = bool(response and len(response) > 0)
                logger.info("✓ OpenRouter backend validated")
            except Exception as e:
                logger.warning(f"✗ OpenRouter validation failed: {e}")
                results["openrouter"] = False
        else:
            logger.info("○ OpenRouter key not configured")
            results["openrouter"] = False

        working = sum(results.values())
        logger.info(f"Backend validation complete: {working}/3 backends working")

        if working == 0:
            logger.error("WARNING: No backends are working!")

        return results

    def get_backend_status(self) -> dict[str, str]:
        """
        Get current status of all backends in the switcher

        Returns:
            Dictionary mapping backend name to status
            Example: {"gemini": "available", "groq": "failed", "openrouter": "available"}
        """
        return self.switcher.get_backend_status()

    async def get_active_backend(self) -> Optional[str]:
        """
        Get the name of the currently active backend

        Returns:
            Backend name (e.g., "gemini") or None if no backend is active
        """
        return await self.switcher.get_active_backend()

    def reset_failures(self):
        """
        Reset the failed backends list

        Useful for starting a new session or after resolving API issues
        """
        self.switcher.reset_failures()
        logger.info("Backend failure list reset")
