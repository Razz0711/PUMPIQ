"""
Health & Infrastructure Routes
=================================
GET /api/v1/health      – Liveness + dependency health checks.
GET /api/v1/health/deep – Detailed component-level status.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Response Models ───────────────────────────────────────────────

class ComponentHealth(BaseModel):
    """Status of a single infrastructure component."""
    name: str
    status: str = "unknown"         # healthy | degraded | down
    latency_ms: float = 0
    detail: str = ""


class HealthResponse(BaseModel):
    """Top-level health envelope."""
    status: str = "healthy"         # healthy | degraded | unhealthy
    version: str = "1.0.0"
    uptime_seconds: float = 0
    timestamp: str = ""
    components: list[ComponentHealth] = Field(default_factory=list)


# ── Quick liveness ────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Quick health check",
    description="Returns 200 if the API process is alive.",
)
async def health(request: Request):
    startup = getattr(request.app.state, "startup_time", time.time())
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - startup, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Deep check ────────────────────────────────────────────────────

@router.get(
    "/health/deep",
    response_model=HealthResponse,
    summary="Deep health check",
    description="Checks Redis, Database, and GPT client health.",
)
async def deep_health(request: Request):
    startup = getattr(request.app.state, "startup_time", time.time())
    components: list[ComponentHealth] = []
    overall = "healthy"

    # ── Redis ─────────────────────────────────────────────────────
    redis = getattr(request.app.state, "redis", None)
    if redis:
        try:
            t0 = time.perf_counter()
            await redis.ping()
            latency = (time.perf_counter() - t0) * 1000
            components.append(ComponentHealth(
                name="redis", status="healthy", latency_ms=round(latency, 1),
            ))
        except Exception as exc:
            components.append(ComponentHealth(
                name="redis", status="down", detail=str(exc),
            ))
            overall = "degraded"
    else:
        components.append(ComponentHealth(
            name="redis", status="down", detail="Not configured",
        ))

    # ── Database ──────────────────────────────────────────────────
    db_engine = getattr(request.app.state, "db_engine", None)
    if db_engine:
        try:
            from sqlalchemy import text

            t0 = time.perf_counter()
            async with db_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            latency = (time.perf_counter() - t0) * 1000
            components.append(ComponentHealth(
                name="database", status="healthy", latency_ms=round(latency, 1),
            ))
        except Exception as exc:
            components.append(ComponentHealth(
                name="database", status="down", detail=str(exc),
            ))
            overall = "degraded"
    else:
        components.append(ComponentHealth(
            name="database", status="down", detail="Not configured",
        ))

    # ── GPT client ────────────────────────────────────────────────
    gpt = getattr(request.app.state, "gpt_client", None)
    if gpt:
        components.append(ComponentHealth(
            name="gpt_client", status="healthy",
            detail="Client initialised (lazy connection)",
        ))
    else:
        components.append(ComponentHealth(
            name="gpt_client", status="down",
            detail="OPENAI_API_KEY not set",
        ))
        overall = "degraded"

    # ── OpenAI API key present ────────────────────────────────────
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    components.append(ComponentHealth(
        name="openai_api_key",
        status="healthy" if has_key else "down",
        detail="Set" if has_key else "Missing – AI synthesis disabled",
    ))

    return HealthResponse(
        status=overall,
        uptime_seconds=round(time.time() - startup, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )
