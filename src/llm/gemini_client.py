"""
Gemini Client for Fleming-AI
Provides high-quality LLM inference via Google Gemini API
"""

import asyncio
import logging
import os
from typing import Optional, AsyncIterator
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Google Gemini API with async support"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",  # Best performance: gemini-2.5-flash (free tier)
        timeout: int = 120,
    ):
        """
        Initialize Gemini client

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Model name to use (default: gemini-2.0-flash-exp)
                   Options:
                   - gemini-2.0-flash-exp: Fastest, experimental (free tier)
                   - gemini-1.5-pro: Highest quality (paid tier)
                   - gemini-1.5-flash: Balanced (free tier)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.model = model
        self.timeout = timeout

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        logger.info(f"GeminiClient initialized with model: {model}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the client (no-op for Gemini SDK)"""
        pass

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
            system: System prompt (prepended to user prompt)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Generated text or async iterator if streaming
        """
        # Combine system and user prompts
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 8192,  # Default to 8K tokens
            response_modalities=["TEXT"],
        )

        if stream:
            return self._stream_generate(full_prompt, config)
        else:
            return await self._generate(full_prompt, config)

    async def _generate(self, prompt: str, config: types.GenerateContentConfig) -> str:
        """Non-streaming generation"""
        try:
            logger.debug(f"Gemini API request: {self.model}")

            # Run synchronous SDK call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                ),
            )

            content = response.text
            logger.debug(f"Gemini API response received ({len(content)} chars)")
            return content

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            # Handle rate limiting
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning("Rate limited, waiting 60s")
                await asyncio.sleep(60)
                return await self._generate(prompt, config)
            raise

    async def _stream_generate(
        self, prompt: str, config: types.GenerateContentConfig
    ) -> AsyncIterator[str]:
        """Streaming generation"""
        try:
            logger.debug(f"Gemini streaming request: {self.model}")

            # Run synchronous streaming SDK call in thread pool
            loop = asyncio.get_event_loop()

            # Create a queue for chunks
            chunk_queue = asyncio.Queue()

            def _stream_in_thread():
                """Run streaming in thread"""
                try:
                    for chunk in self.client.models.generate_content_stream(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    ):
                        if chunk.text:
                            # Put chunk in queue (thread-safe)
                            asyncio.run_coroutine_threadsafe(chunk_queue.put(chunk.text), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(chunk_queue.put(e), loop)
                finally:
                    # Signal end of stream
                    asyncio.run_coroutine_threadsafe(chunk_queue.put(None), loop)

            # Start streaming in background thread
            await loop.run_in_executor(None, _stream_in_thread)

            # Yield chunks from queue
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning("Rate limited, waiting 60s")
                await asyncio.sleep(60)
                async for chunk in self._stream_generate(prompt, config):
                    yield chunk
            else:
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
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 8192,
            response_modalities=["TEXT"],
        )

        async for chunk in self._stream_generate(full_prompt, config):
            yield chunk

    async def is_available(self) -> bool:
        """
        Check if the client is available (API key valid, not rate-limited)

        Returns:
            True if available, False otherwise
        """
        try:
            # Try a minimal request
            await self.generate("test", max_tokens=1)
            return True
        except Exception as e:
            logger.warning(f"Gemini availability check failed: {e}")
            return False
