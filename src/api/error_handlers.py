"""
Error Handlers
================
Step 5.1 – Centralised exception handling for the PumpIQ API.

Strategy:
  - Module failure      → log, continue with other modules, lower confidence
  - Critical failure    → return error, suggest alternatives
  - Partial data        → proceed with available, flag in response
  - GPT-4o failure      → retry 3× with exponential back-off → NLG fallback
  - Validation error    → 422 with structured detail
  - Unexpected errors   → 500 with generic message (no internals leaked)
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Custom Exceptions
# ══════════════════════════════════════════════════════════════════

class PumpIQError(Exception):
    """Base exception for all PumpIQ errors."""

    def __init__(
        self,
        message: str = "An error occurred",
        code: int = 500,
        detail: Optional[str] = None,
    ):
        self.message = message
        self.code = code
        self.detail = detail
        super().__init__(message)


class TokenNotFoundError(PumpIQError):
    """Raised when a requested token cannot be found."""

    def __init__(self, ticker: str):
        super().__init__(
            message=f"Token '{ticker}' not found",
            code=404,
            detail=f"No data available for token '{ticker}'. Check ticker spelling.",
        )


class InsufficientDataError(PumpIQError):
    """Raised when not enough data modules return usable data."""

    def __init__(self, available_modes: list[str], required: int = 1):
        super().__init__(
            message="Insufficient data for analysis",
            code=503,
            detail=(
                f"Only {len(available_modes)} data mode(s) returned data "
                f"({', '.join(available_modes) or 'none'}). "
                f"At least {required} required."
            ),
        )


class GPTSynthesisError(PumpIQError):
    """Raised when GPT-4o synthesis fails after retries."""

    def __init__(self, last_error: str = ""):
        super().__init__(
            message="AI synthesis temporarily unavailable",
            code=503,
            detail=(
                "GPT-4o synthesis failed after retries. "
                "Template-based fallback was used. "
                f"Last error: {last_error}"
            ),
        )


class RateLimitError(PumpIQError):
    """Raised when rate limits are exceeded."""

    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(
            message="Rate limit exceeded",
            code=429,
            detail=f"Too many requests. Retry after {retry_after} seconds.",
        )


class ModuleFailureError(PumpIQError):
    """
    Raised when one data module fails.
    This is non-critical – handled by lowering confidence.
    """

    def __init__(self, module_name: str, reason: str = ""):
        self.module_name = module_name
        super().__init__(
            message=f"Module '{module_name}' failed",
            code=206,       # Partial Content
            detail=reason,
        )


# ══════════════════════════════════════════════════════════════════
# Handler Registration
# ══════════════════════════════════════════════════════════════════

def register_error_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI app."""

    @app.exception_handler(PumpIQError)
    async def pumpiq_error_handler(request: Request, exc: PumpIQError):
        logger.error(
            "PumpIQError %d: %s – %s",
            exc.code, exc.message, exc.detail,
        )
        return JSONResponse(
            status_code=exc.code,
            content={
                "error": exc.message,
                "detail": exc.detail,
                "code": exc.code,
            },
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc: RateLimitError):
        return JSONResponse(
            status_code=429,
            content={
                "error": exc.message,
                "detail": exc.detail,
                "code": 429,
            },
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation error: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "detail": _format_validation_errors(exc.errors()),
                "code": 422,
            },
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(request: Request, exc: ValidationError):
        logger.warning("Pydantic validation error: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "detail": _format_validation_errors(exc.errors()),
                "code": 422,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method, request.url.path,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred. Please try again later.",
                "code": 500,
            },
        )


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _format_validation_errors(errors: list) -> str:
    """
    Convert Pydantic / FastAPI validation errors into a
    human-readable string.
    """
    parts = []
    for err in errors:
        loc = " → ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "")
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts) if parts else "Unknown validation error"
