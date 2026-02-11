"""
FastAPI Application Factory
==============================
Step 5.1 – Main entry-point for the PumpIQ REST API.

Run with::

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .error_handlers import register_error_handlers

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Lifespan (startup / shutdown hooks)
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: initialise GPT client, DB pool, Redis cache.
    Shutdown: close connections gracefully.
    """
    logger.info("PumpIQ API starting up …")

    # ── Startup ───────────────────────────────────────────────────
    # Stored on app.state so DI dependencies can reach them.
    app.state.startup_time = time.time()
    app.state.ready = False

    try:
        # AI client: prefer Gemini (free), fall back to GPT-4o
        import os

        gemini_key = os.getenv("GEMINI_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")

        if gemini_key:
            from src.ai_engine.gemini_client import GeminiClient
            app.state.ai_client = GeminiClient(api_key=gemini_key)
            logger.info("AI backend: Google Gemini")
        elif openai_key:
            from src.ai_engine.gpt_client import GPTClient
            app.state.ai_client = GPTClient(api_key=openai_key)
            logger.info("AI backend: OpenAI GPT-4o (legacy)")
        else:
            app.state.ai_client = None
            logger.warning("No AI API key configured (GEMINI_API_KEY or OPENAI_API_KEY)")

        # Keep legacy alias for backward compat
        app.state.gpt_client = app.state.ai_client

        # Redis (optional – degrades gracefully)
        try:
            import redis.asyncio as aioredis

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            app.state.redis = aioredis.from_url(redis_url, decode_responses=True)
            await app.state.redis.ping()
            logger.info("Redis connected at %s", redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable – caching disabled: %s", exc)
            app.state.redis = None

        # Database pool (async – via asyncpg / SQLAlchemy)
        try:
            from sqlalchemy.ext.asyncio import create_async_engine

            db_url = os.getenv(
                "DATABASE_URL",
                "postgresql+asyncpg://pumpiq:pumpiq@localhost:5432/pumpiq",
            )
            app.state.db_engine = create_async_engine(db_url, pool_size=10)
            logger.info("Database engine created")
        except Exception as exc:
            logger.warning("Database unavailable: %s", exc)
            app.state.db_engine = None

        app.state.ready = True
        logger.info("PumpIQ API ready")

    except Exception as exc:
        logger.error("Startup failed: %s", exc, exc_info=True)

    yield  # ── Application runs ──

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("PumpIQ API shutting down …")
    if getattr(app.state, "redis", None):
        await app.state.redis.close()
    if getattr(app.state, "db_engine", None):
        await app.state.db_engine.dispose()
    logger.info("Shutdown complete")


# ══════════════════════════════════════════════════════════════════
# Application Factory
# ══════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    """
    Build and return the fully-configured FastAPI application.
    """
    app = FastAPI(
        title="PumpIQ API",
        description=(
            "AI-powered cryptocurrency trading recommendation engine. "
            "Combines news, on-chain, technical, and social data "
            "through a GPT-4o synthesis pipeline."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request-timing middleware ─────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    # ── Error handlers ────────────────────────────────────────────
    register_error_handlers(app)

    # ── Routers ───────────────────────────────────────────────────
    from .routes import (
        recommendations_router,
        analysis_router,
        portfolio_router,
        health_router,
    )

    app.include_router(
        recommendations_router,
        prefix="/api/v1",
        tags=["recommendations"],
    )
    app.include_router(
        analysis_router,
        prefix="/api/v1",
        tags=["analysis"],
    )
    app.include_router(
        portfolio_router,
        prefix="/api/v1",
        tags=["portfolio"],
    )
    app.include_router(
        health_router,
        prefix="/api/v1",
        tags=["health"],
    )

    return app


# ── Module-level app for ``uvicorn src.api.app:app`` ─────────────
app = create_app()
