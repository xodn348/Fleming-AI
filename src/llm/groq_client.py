"""
Groq Client for Fleming-AI
Provides ultra-fast LLM inference via Groq API
"""

import asyncio
import httpx
import json
import logging
import os
from typing import Optional, Dict, Any, List, AsyncIterator

logger = logging.getLogger(__name__)


class GroqClient:
    """Client for Groq API with OpenAI-compatible interface"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 120,
    ):
        """
        Initialize Groq client

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.model = model
        self.timeout = timeout
        self.base_url = "https://api.groq.com/openai/v1"
        self.client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

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
        Generate text completion

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Generated text or async iterator if streaming
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if stream:
            return self._stream_generate(payload)
        else:
            return await self._generate(payload)

    async def _generate(self, payload: Dict[str, Any]) -> str:
        """Non-streaming generation"""
        try:
            logger.debug(f"Groq API request: {self.model}")
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 60))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                return await self._generate(payload)

            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.debug(f"Groq API response received ({len(content)} chars)")
            return content
        except httpx.HTTPError as e:
            logger.error(f"Groq API error: {e}")
            raise

    async def _stream_generate(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        """Streaming generation"""
        payload["stream"] = True
        try:
            logger.debug(f"Groq streaming request: {self.model}")
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as response:
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("retry-after", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    async for chunk in self._stream_generate(payload):
                        yield chunk
                    return

                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        # Handle SSE format: "data: {...}"
                        if line.startswith("data: "):
                            line = line[6:]
                        if line == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                            content = (
                                data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            )
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as e:
            logger.error(f"Groq streaming error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream text completion (convenience method)

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Async iterator of text chunks
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        async for chunk in self._stream_generate(payload):
            yield chunk

    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Groq does not support embeddings

        Args:
            text: Text to embed
            model: Embedding model (unused)

        Raises:
            NotImplementedError: Groq API does not support embeddings
        """
        raise NotImplementedError("Groq API does not support embeddings")
