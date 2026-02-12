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
from src.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)


class PipelineLLM:
    """
    Unified LLM interface for Fleming-AI pipeline

    Wraps BackendSwitcher (Groq-only) for Fleming's generation tasks.
    Note: For new code, prefer using GroqClient directly.

    Usage:
        llm = PipelineLLM()
        result = await llm.generate("Say hello", max_tokens=50)

        # Validate Groq backend
        status = await llm.validate_backends()
        print(status)  # {"groq": True}
    """

    def __init__(self):
        """
        Initialize PipelineLLM with Groq API key from environment

        Loads keys from .env:
        - GROQ_API_KEY for Groq (required)
        """
        if load_dotenv is not None:
            load_dotenv()

        self.groq_key = os.getenv("GROQ_API_KEY")

        if not self.groq_key:
            raise ValueError("No GROQ_API_KEY found. Please set GROQ_API_KEY in .env")

        logger.info("PipelineLLM initialized with Groq backend")

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
        Test Groq backend with simple prompt

        Returns:
            Dictionary mapping backend name to availability status
            Example: {"groq": True}
        """
        results = {}

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

        if not results.get("groq"):
            logger.error("WARNING: Groq backend is not working!")

        return results

    def get_backend_status(self) -> dict[str, str]:
        """
        Get current status of Groq backend in the switcher

        Returns:
            Dictionary mapping backend name to status
            Example: {"groq": "available"}
        """
        return self.switcher.get_backend_status()

    async def get_active_backend(self) -> Optional[str]:
        """
        Get the name of the currently active backend

        Returns:
            Backend name ("groq") or None if no backend is active
        """
        return await self.switcher.get_active_backend()

    def reset_failures(self):
        """
        Reset the failed backends list

        Useful for starting a new session or after resolving API issues
        """
        self.switcher.reset_failures()
        logger.info("Backend failure list reset")
