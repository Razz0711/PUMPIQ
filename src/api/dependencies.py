"""
FastAPI Dependencies
======================
Step 5.1 – Shared dependency-injection functions for the API routes.

Provides:
  - ``get_service``     → PumpIQService instance
  - ``get_user_prefs``  → UserPreferences from header / DB / default
  - ``get_redis``       → Optional async Redis client
  - ``verify_api_key``  → Simple API-key guard
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

@lru_cache()
def _api_keys() -> set:
    """Load valid API keys from env (comma-separated)."""
    raw = os.getenv("PUMPIQ_API_KEYS", "")
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


# ══════════════════════════════════════════════════════════════════
# Auth / API Key
# ══════════════════════════════════════════════════════════════════

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """
    Validate the ``X-API-Key`` header.

    If no keys are configured (dev mode), any request is accepted.
    Returns the validated key or ``"dev"`` in open mode.
    """
    valid_keys = _api_keys()
    if not valid_keys:
        # Dev mode – no auth required
        return "dev"
    if not x_api_key or x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key


# ══════════════════════════════════════════════════════════════════
# User Preferences
# ══════════════════════════════════════════════════════════════════

async def get_user_prefs(
    request: Request,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
):
    """
    Resolve user preferences.

    Priority:
      1. Database look-up by ``X-User-Id`` header (if DB available)
      2. Request body ``preferences`` field (inline override)
      3. Default preferences

    Returns a ``UserPreferences`` instance.
    """
    from src.ui.user_config import UserPreferences, default_preferences

    # Future: query DB for user prefs by x_user_id
    # For now, return sensible defaults
    if x_user_id:
        logger.debug("Resolving preferences for user %s", x_user_id)
        # TODO: DB look-up
    return default_preferences()


# ══════════════════════════════════════════════════════════════════
# Redis (optional cache)
# ══════════════════════════════════════════════════════════════════

async def get_redis(request: Request):
    """Return the shared Redis client or None."""
    return getattr(request.app.state, "redis", None)


# ══════════════════════════════════════════════════════════════════
# PumpIQ Service (main integration point)
# ══════════════════════════════════════════════════════════════════

async def get_service(request: Request):
    """
    Build and return a fully-wired ``PumpIQService`` ready to
    handle recommendation / analysis / comparison requests.
    """
    from .service_layer import PumpIQService

    gpt_client = getattr(request.app.state, "gpt_client", None)
    redis = getattr(request.app.state, "redis", None)

    return PumpIQService(gpt_client=gpt_client, redis=redis)
