"""
OpenRouter API Client for Fleming-AI
Advanced reasoning with explicit thinking process
"""

import os
import logging
from typing import Optional, Any
import httpx

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """OpenRouter API client using OpenAI-compatible format"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: str | None = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        self.base_url: str = "https://openrouter.ai/api/v1"
        self.model: str = "arcee-ai/trinity-large-preview:free"
        logger.info("✓ Trinity Large Preview initialized")

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text with OpenRouter

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text (final content, not reasoning)
        """
        try:
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            headers: dict[str, str] = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/fleming-ai",
                "X-Title": "Fleming-AI",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                response.raise_for_status()
                data: dict[str, Any] = response.json()

            if "choices" not in data or not data["choices"]:
                raise ValueError("No choices in response")

            choice: dict[str, Any] = data["choices"][0]
            message: dict[str, Any] = choice.get("message", {})
            content: str = message.get("content", "")

            if not content:
                raise ValueError("No content in response")

            return content

        except httpx.HTTPError as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise
