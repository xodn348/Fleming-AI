"""
Ollama Client for Fleming-AI
Provides optimized LLM inference with memory management
"""

import httpx
import json
from typing import Optional, Dict, Any, List, AsyncIterator
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API with memory optimization"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout: int = 120,
    ):
        """
        Initialize Ollama client

        Args:
            base_url: Ollama server URL
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
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
            **kwargs: Additional Ollama parameters

        Returns:
            Generated text or async iterator if streaming
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": temperature, **kwargs},
        }

        if system:
            payload["system"] = system

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        if stream:
            return self._stream_generate(payload)
        else:
            return await self._generate(payload)

    async def _generate(self, payload: Dict[str, Any]) -> str:
        """Non-streaming generation"""
        try:
            response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise

    async def _stream_generate(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        """Streaming generation"""
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> str | AsyncIterator[str]:
        """
        Chat completion with message history

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional Ollama parameters

        Returns:
            Generated response or async iterator if streaming
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature, **kwargs},
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        if stream:
            return self._stream_chat(payload)
        else:
            return await self._chat(payload)

    async def _chat(self, payload: Dict[str, Any]) -> str:
        """Non-streaming chat"""
        try:
            response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            message = result.get("message", {})
            return message.get("content", "")
        except httpx.HTTPError as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    async def _stream_chat(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        """Streaming chat"""
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        message = data.get("message", {})
                        if "content" in message:
                            yield message["content"]
        except httpx.HTTPError as e:
            logger.error(f"Ollama chat streaming error: {e}")
            raise

    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embeddings for text

        Args:
            text: Text to embed
            model: Embedding model (defaults to nomic-embed-text)

        Returns:
            List of embedding values
        """
        embed_model = model or "nomic-embed-text"

        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings", json={"model": embed_model, "prompt": text}
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except httpx.HTTPError as e:
            logger.error(f"Ollama embedding error: {e}")
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models

        Returns:
            List of model information dicts
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            result = response.json()
            return result.get("models", [])
        except httpx.HTTPError as e:
            logger.error(f"Ollama list models error: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if Ollama server is healthy

        Returns:
            True if server is responding
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
