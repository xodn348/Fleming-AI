"""
Claude Client for Fleming-AI
Provides high-quality LLM inference via Claude.ai session key (free, no API key needed)
"""

import asyncio
import logging
import os
import uuid
import json
from typing import Optional, AsyncIterator
import httpx

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Client for Claude.ai API using session key authentication"""

    def __init__(
        self,
        session_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: int = 120,
    ):
        """
        Initialize Claude client with session key

        Args:
            session_key: Claude.ai session key (defaults to CLAUDE_SESSION_KEY env var)
                        Get from: https://claude.ai → F12 → Application → Cookies → sessionKey
            model: Model name (for compatibility, not used with session key)
            timeout: Request timeout in seconds
        """
        self.session_key = session_key or os.getenv("CLAUDE_SESSION_KEY")
        if not self.session_key:
            raise ValueError(
                "CLAUDE_SESSION_KEY environment variable not set. "
                "Get it from: https://claude.ai → F12 → Application → Cookies → sessionKey"
            )

        self.model = model
        self._timeout = timeout
        self.base_url = "https://claude.ai/api"
        self.org_id = None

        logger.info("ClaudeClient initialized with session key authentication")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the client (no-op - we use context managers)"""
        pass  # httpx clients are created per-request

    async def _get_organization_id(self) -> str:
        """Get organization ID from Claude.ai API"""
        try:
            headers = self._get_headers()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{self.base_url}/organizations",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                org_id = data[0]["uuid"]
                logger.debug(f"Got organization ID: {org_id}")
                return org_id
        except Exception as e:
            logger.error(f"Failed to get organization ID: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                raise ValueError("Invalid session key. Please check your CLAUDE_SESSION_KEY.")
            raise

    async def _create_conversation(self) -> str:
        """Create a new conversation"""
        try:
            headers = self._get_headers()
            payload = {
                "uuid": str(uuid.uuid4()),
                "name": "Fleming-AI",
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self.base_url}/organizations/{self.org_id}/chat_conversations",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                conv_id = data["uuid"]
                logger.debug(f"Created conversation: {conv_id}")
                return conv_id
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise

    async def _send_message(
        self,
        conv_id: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> str:
        """Send message and get response"""
        try:
            headers = self._get_headers()
            payload = {
                "prompt": prompt,
                "attachments": [],
            }

            # Add system prompt if provided
            if system:
                payload["system"] = system

            async with httpx.AsyncClient(timeout=self._timeout * 2) as client:
                response = await client.post(
                    f"{self.base_url}/organizations/{self.org_id}/chat_conversations/{conv_id}/completion",
                    headers=headers,
                    json=payload,
                    timeout=self._timeout * 2,
                )
                response.raise_for_status()

                # Parse streaming response
                full_response = ""
                for line in response.text.split("\n"):
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "completion" in data:
                                full_response += data["completion"]
                        except json.JSONDecodeError:
                            continue

                logger.debug(f"Got response ({len(full_response)} chars)")
                return full_response

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                raise ValueError("Invalid session key. Please check your CLAUDE_SESSION_KEY.")
            raise

    async def _delete_conversation(self, conv_id: str) -> None:
        """Delete conversation (cleanup)"""
        try:
            headers = self._get_headers()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                await client.delete(
                    f"{self.base_url}/organizations/{self.org_id}/chat_conversations/{conv_id}",
                    headers=headers,
                )
            logger.debug(f"Deleted conversation: {conv_id}")
        except Exception as e:
            logger.warning(f"Failed to delete conversation: {e}")

    def _get_headers(self) -> dict:
        """Get request headers with session key"""
        return {
            "Cookie": f"sessionKey={self.session_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }

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
            temperature: Sampling temperature (0.0-1.0) - note: not used with session key
            max_tokens: Maximum tokens to generate - note: not used with session key
            stream: Whether to stream the response (not yet supported)
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        if stream:
            logger.warning("Streaming not yet supported with session key auth")

        return await self._generate(
            prompt=prompt,
            system=system,
            **kwargs,
        )

    async def _generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Non-streaming generation"""
        try:
            logger.debug("Claude API request via session key")

            if not self.org_id:
                self.org_id = await self._get_organization_id()

            conv_id = await self._create_conversation()

            response = await self._send_message(conv_id, prompt, system)

            await self._delete_conversation(conv_id)

            return response

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            if "429" in str(e) or "rate" in str(e).lower():
                logger.warning("Rate limited, waiting 60s")
                await asyncio.sleep(60)
                return await self._generate(
                    prompt=prompt,
                    system=system,
                    **kwargs,
                )
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
        Stream text completion (not yet supported with session key)

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Async iterator of text chunks
        """
        logger.warning("Streaming not yet supported with session key auth")
        response = await self.generate(
            prompt=prompt,
            system=system,
            **kwargs,
        )
        yield response

    async def is_available(self) -> bool:
        """
        Check if the client is available (session key valid, not rate-limited)

        Returns:
            True if available, False otherwise
        """
        try:
            await self.generate("test", max_tokens=1)
            return True
        except Exception as e:
            logger.warning(f"Claude availability check failed: {e}")
            return False
