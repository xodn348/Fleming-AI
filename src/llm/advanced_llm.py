"""
Advanced LLM Client for Fleming-AI
Supports Claude (primary) and KIMI (fallback) for hypothesis generation
"""

import httpx
import json
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AdvancedLLM:
    """Multi-provider LLM client with Claude primary and KIMI fallback"""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        claude_session_key: Optional[str] = None,
        kimi_api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.claude_session_key = claude_session_key or os.getenv("CLAUDE_SESSION_KEY")
        self.kimi_api_key = kimi_api_key or os.getenv("KIMI_API_KEY")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

        if not self.anthropic_api_key and not self.claude_session_key and not self.kimi_api_key:
            logger.warning(
                "No credentials found for Claude or KIMI - hypothesis generation may fail"
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self.client.aclose()

    async def _generate_claude_api(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate using Claude API"""
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system:
            payload["system"] = system

        response = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        return result["content"][0]["text"]

    async def _generate_claude_session(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate using Claude session cookie (Max plan)"""
        headers = {
            "Cookie": f"sessionKey={self.claude_session_key}",
            "Content-Type": "application/json",
            "anthropic-client-sha": "unknown",
            "anthropic-client-version": "unknown",
        }

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        payload = {
            "prompt": full_prompt,
            "model": "claude-sonnet-4",
            "temperature": temperature,
        }

        response = await self.client.post(
            "https://claude.ai/api/append_message",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("completion", "")

    async def _generate_claude(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate using Claude (API or session)"""
        if self.anthropic_api_key:
            return await self._generate_claude_api(prompt, system, temperature, max_tokens)
        elif self.claude_session_key:
            return await self._generate_claude_session(prompt, system, temperature, max_tokens)
        else:
            raise RuntimeError("No Claude credentials configured")

    async def _generate_kimi(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate using KIMI API"""
        headers = {
            "Authorization": f"Bearer {self.kimi_api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "moonshot-v1-128k",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = await self.client.post(
                "https://api.moonshot.cn/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            logger.error(f"KIMI API error: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Generate text with KIMI (primary) - Claude disabled to avoid session key conflicts

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # KIMI disabled - API authentication issues
        # Hypothesis generator will fallback to Ollama automatically
        logger.warning(
            "AdvancedLLM disabled (KIMI API issues), hypothesis generator will use Ollama"
        )
        raise RuntimeError("KIMI API unavailable - using Ollama fallback")
