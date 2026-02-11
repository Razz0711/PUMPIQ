"""
Google Gemini AI Client
=========================
Async wrapper around the **Google GenAI** (Gemini) SDK.

Replaces the previous GPT-4o client while exposing the same
``chat()`` / ``chat_multi()`` interface so the Orchestrator does not
need changes.

Model       : gemini-2.0-flash  (free tier: 15 RPM / 1 M TPM)
Get API key : https://aistudio.google.com/apikey
Docs        : https://ai.google.dev/gemini-api/docs
Python pkg  : google-genai  (pip install google-genai)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ───────────────────────────── Data classes ─────────────────────────────

@dataclass
class GeminiMessage:
    """Compatible with GPTMessage for the orchestrator."""
    role: str       # "user" | "model"  (Gemini uses "model" not "assistant")
    content: str


@dataclass
class GeminiResponse:
    """
    Wrapper around a Gemini generate_content response.

    Field names mirror ``GPTResponse`` so the Orchestrator works unchanged.
    """
    content: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0
    success: bool = True
    error: str = ""


# ───────────────────────────── Client ──────────────────────────────────

class GeminiClient:
    """
    Async Gemini client with retry logic.

    Usage::

        client = GeminiClient(api_key="AIza...")
        resp = await client.chat(
            system_prompt="You are PumpIQ …",
            user_prompt="Analyze $BONK …",
        )
        print(resp.content)

    The ``chat()`` and ``chat_multi()`` methods have the exact same
    signature as the old ``GPTClient`` so the Orchestrator can swap
    in this client with zero changes.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None        # lazy init

    # ── lazy init ──────────────────────────────────────────────────

    def _ensure_client(self) -> Any:
        """Lazily initialise the google.genai client."""
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed.\n"
                "  pip install google-genai"
            )

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Get one free at https://aistudio.google.com/apikey"
            )

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    # ── public API (mirrors GPTClient) ─────────────────────────────

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GeminiResponse:
        """
        Send a system + user prompt to Gemini and return the response.

        Matches the ``GPTClient.chat()`` signature exactly.
        """
        combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        return await self._call_with_retry(
            combined_prompt, temperature, max_tokens,
        )

    async def chat_multi(
        self,
        messages: List[GeminiMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GeminiResponse:
        """Send a multi-turn conversation to Gemini."""
        parts = []
        for m in messages:
            role_tag = "User" if m.role == "user" else "Assistant"
            parts.append(f"[{role_tag}]\n{m.content}")
        combined = "\n\n".join(parts)
        return await self._call_with_retry(combined, temperature, max_tokens)

    # ── retry wrapper ──────────────────────────────────────────────

    async def _call_with_retry(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GeminiResponse:
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._call_gemini(prompt, temperature, max_tokens)
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Gemini attempt %d/%d failed: %s",
                    attempt, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)

        logger.error("Gemini failed after %d retries: %s", self.max_retries, last_error)
        return GeminiResponse(
            success=False,
            error=f"Failed after {self.max_retries} retries: {last_error}",
        )

    # ── actual API call ────────────────────────────────────────────

    async def _call_gemini(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GeminiResponse:
        client = self._ensure_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature or self.temperature,
            max_output_tokens=max_tokens or self.max_tokens,
            top_p=self.top_p,
        )

        start = time.monotonic()

        # google-genai uses aio for async calls
        response = await client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        latency = (time.monotonic() - start) * 1000

        # Parse response
        text = response.text or ""
        finish = ""
        if response.candidates:
            candidate = response.candidates[0]
            finish = str(candidate.finish_reason) if candidate.finish_reason else ""

        # Token counts
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
            completion_tokens = getattr(um, "candidates_token_count", 0) or 0

        return GeminiResponse(
            content=text,
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self.model_name,
            latency_ms=round(latency, 1),
            success=True,
        )

    # ── cleanup ────────────────────────────────────────────────────

    async def close(self) -> None:
        """No persistent connection to close for Gemini, but keeps API parity."""
        self._client = None
