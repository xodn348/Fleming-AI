"""
Backend Switcher for Fleming-AI
Automatically switches between available LLM backends with fallback
"""

import logging
from typing import Optional, List, AsyncIterator, Any
from dataclasses import dataclass

from src.llm.claude_client import ClaudeClient
from src.llm.gemini_client import GeminiClient
from src.llm.groq_client import GroqClient
from src.llm.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


@dataclass
class BackendConfig:
    name: str
    client_class: type
    priority: int
    enabled: bool = True


class BackendSwitcher:
    """
    Smart LLM backend switcher with automatic fallback

    Priority order:
    1. Claude (highest quality, recommended)
    2. Gemini (free, high quality, 1500 RPD)
    3. Groq (free, fast, 30 RPM but rate-limited)
    4. OpenRouter (free, 200 RPD)

    Usage:
        switcher = BackendSwitcher()
        response = await switcher.generate("prompt")
    """

    def __init__(self):
        self.backends: List[BackendConfig] = [
            BackendConfig("claude", ClaudeClient, priority=1),
            BackendConfig("gemini", GeminiClient, priority=2),
            BackendConfig("groq", GroqClient, priority=3),
            BackendConfig("openrouter", OpenRouterClient, priority=4),
        ]

        self._active_client: Optional[Any] = None
        self._active_backend: Optional[str] = None
        self._failed_backends: set[str] = set()

        logger.info("BackendSwitcher initialized with 3 backends")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if self._active_client and hasattr(self._active_client, "close"):
            await self._active_client.close()

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

        Tries backends in priority order until one succeeds.
        Remembers failed backends within session.
        """
        sorted_backends = sorted(
            [b for b in self.backends if b.enabled and b.name not in self._failed_backends],
            key=lambda x: x.priority,
        )

        if not sorted_backends:
            self._failed_backends.clear()
            sorted_backends = sorted(
                [b for b in self.backends if b.enabled], key=lambda x: x.priority
            )
            logger.warning("All backends failed, retrying from scratch")

        last_error = None

        for backend in sorted_backends:
            try:
                logger.info(f"Trying backend: {backend.name} (priority {backend.priority})")

                client = await self._get_or_create_client(backend)

                response = await client.generate(
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    **kwargs,
                )

                self._active_backend = backend.name
                logger.info(f"✓ Success with {backend.name}")
                return response

            except Exception as e:
                logger.warning(f"✗ {backend.name} failed: {e}")
                self._failed_backends.add(backend.name)
                last_error = e
                continue

        error_msg = f"All backends failed. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def _get_or_create_client(self, backend: BackendConfig):
        """Get existing client or create new one"""
        if self._active_backend == backend.name and self._active_client:
            return self._active_client

        if self._active_client and hasattr(self._active_client, "close"):
            await self._active_client.close()

        try:
            self._active_client = backend.client_class()
            return self._active_client
        except ValueError as e:
            if "API" in str(e) and "not set" in str(e):
                logger.warning(f"{backend.name} API key not set, skipping")
                self._failed_backends.add(backend.name)
                raise
            raise

    async def get_active_backend(self) -> Optional[str]:
        """Get name of currently active backend"""
        return self._active_backend

    def reset_failures(self):
        """Reset failed backends list (for new sessions)"""
        self._failed_backends.clear()
        logger.info("Failed backends list reset")

    async def is_backend_available(self, backend_name: str) -> bool:
        """Check if a specific backend is available"""
        backend = next((b for b in self.backends if b.name == backend_name), None)
        if not backend or not backend.enabled:
            return False

        try:
            client = backend.client_class()
            if hasattr(client, "is_available"):
                return await client.is_available()
            return True
        except Exception as e:
            logger.warning(f"{backend_name} availability check failed: {e}")
            return False

    def get_backend_status(self) -> dict[str, str]:
        """Get status of all backends"""
        return {
            backend.name: "failed" if backend.name in self._failed_backends else "available"
            for backend in self.backends
            if backend.enabled
        }
