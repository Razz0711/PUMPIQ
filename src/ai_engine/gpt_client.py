"""
GPT-4o API Client
===================
Thin async wrapper around the OpenAI ChatCompletion endpoint.
Handles retries, rate-limits, token counting, and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPTMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class GPTResponse:
    """Wrapper around an OpenAI ChatCompletion response."""
    content: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0
    success: bool = True
    error: str = ""


class GPTClient:
    """
    Async GPT-4o client with retry logic and error handling.

    Usage::

        client = GPTClient(api_key="sk-...")
        response = await client.chat(
            system_prompt="You are ...",
            user_prompt="Analyze ...",
        )
        print(response.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        self._client = None  # Lazy init

    # ── Public API ────────────────────────────────────────────────

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GPTResponse:
        """
        Send a system+user prompt to GPT-4o and return the response.

        Falls back to a structured error response on failure.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self._call_with_retry(
            messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

    async def chat_multi(
        self,
        messages: List[GPTMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GPTResponse:
        """Send an arbitrary message list."""
        raw = [{"role": m.role, "content": m.content} for m in messages]
        return await self._call_with_retry(
            raw,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

    # ── Internal – retries ────────────────────────────────────────

    async def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> GPTResponse:
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._call_openai(
                    messages, temperature, max_tokens,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"GPT call attempt {attempt}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)

        logger.error(f"GPT call failed after {self.max_retries} attempts: {last_error}")
        return GPTResponse(
            success=False,
            error=f"Failed after {self.max_retries} retries: {last_error}",
        )

    # ── Internal – API call ───────────────────────────────────────

    async def _call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> GPTResponse:
        """
        Actual OpenAI API call using the openai Python library.
        """
        try:
            import openai
        except ImportError:
            logger.error("openai package not installed")
            return GPTResponse(
                success=False,
                error="openai package not installed – pip install openai",
            )

        if not self.api_key:
            return GPTResponse(
                success=False,
                error="OPENAI_API_KEY not configured",
            )

        # Lazy-init async client
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )

        start = time.monotonic()

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )

        latency = (time.monotonic() - start) * 1000

        choice = response.choices[0] if response.choices else None
        usage = response.usage

        return GPTResponse(
            content=choice.message.content if choice else "",
            finish_reason=choice.finish_reason if choice else "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=response.model or self.model,
            latency_ms=round(latency, 1),
            success=True,
        )

    # ── Cleanup ───────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
