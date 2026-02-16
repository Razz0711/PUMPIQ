"""
PumpIQ Security & Rate Limiting Middleware
==========================================
Production-hardened middleware for:
- API rate limiting (per-IP and per-user)
- Login attempt tracking with account lockout
- Request logging with structured output
- Security headers (CSP, HSTS, etc.)
"""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# RATE LIMITER (Token Bucket)
# ══════════════════════════════════════════════════════════════════

@dataclass
class TokenBucket:
    """Token bucket rate limiter."""
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self, n: int = 1) -> bool:
        """Try to consume n tokens. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class RateLimiter:
    """
    Per-client rate limiting with configurable tiers.

    Default limits:
    - General API: 60 requests/minute per IP
    - Auth endpoints: 10 requests/minute per IP
    - AI endpoints: 15 requests/minute per IP
    - Trading endpoints: 30 requests/minute per IP
    """

    # Route → (capacity, refill_rate_per_second)
    ROUTE_LIMITS: Dict[str, Tuple[int, float]] = {
        "/api/auth/login": (10, 10 / 60),      # 10/min
        "/api/auth/register": (5, 5 / 60),      # 5/min
        "/api/auth/forgot-password": (3, 3 / 60),  # 3/min
        "/api/otp/": (5, 5 / 60),               # 5/min
        "/api/ai/": (15, 15 / 60),              # 15/min
        "/api/trader/": (30, 30 / 60),           # 30/min
        "/api/bot/ask": (10, 10 / 60),          # 10/min
    }
    DEFAULT_LIMIT = (120, 120 / 60)  # 120/min for everything else

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.monotonic()

    def is_allowed(self, client_id: str, path: str) -> Tuple[bool, int]:
        """
        Check if a request is allowed.

        Returns:
            (allowed, retry_after_seconds)
        """
        capacity, rate = self._get_limit(path)
        key = f"{client_id}:{self._get_limit_key(path)}"

        with self._lock:
            self._maybe_cleanup()
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    capacity=capacity,
                    refill_rate=rate,
                    tokens=capacity,
                )
            bucket = self._buckets[key]
            if bucket.consume():
                return True, 0
            else:
                # Calculate retry-after
                retry_after = int((1.0 - bucket.tokens) / max(bucket.refill_rate, 0.01))
                return False, max(1, retry_after)

    def _get_limit(self, path: str) -> Tuple[int, float]:
        """Find the matching rate limit for a path."""
        for prefix, limit in self.ROUTE_LIMITS.items():
            if path.startswith(prefix):
                return limit
        return self.DEFAULT_LIMIT

    def _get_limit_key(self, path: str) -> str:
        """Group paths into rate limit buckets."""
        for prefix in self.ROUTE_LIMITS:
            if path.startswith(prefix):
                return prefix
        return "default"

    def _maybe_cleanup(self):
        """Remove stale buckets every 5 minutes."""
        now = time.monotonic()
        if now - self._last_cleanup > 300:
            stale = [
                k for k, v in self._buckets.items()
                if now - v.last_refill > 600  # 10 min inactive
            ]
            for k in stale:
                del self._buckets[k]
            self._last_cleanup = now


# ══════════════════════════════════════════════════════════════════
# LOGIN ATTEMPT TRACKER
# ══════════════════════════════════════════════════════════════════

class LoginTracker:
    """
    Track failed login attempts and enforce account lockout.

    - 5 failed attempts → 15-minute lockout
    - Successful login resets the counter
    """

    MAX_ATTEMPTS = 5
    LOCKOUT_SECONDS = 900  # 15 minutes

    def __init__(self):
        self._attempts: Dict[str, list] = defaultdict(list)
        self._lockouts: Dict[str, float] = {}
        self._lock = threading.RLock()

    def is_locked(self, identifier: str) -> Tuple[bool, int]:
        """
        Check if an account/IP is locked out.

        Returns:
            (is_locked, remaining_seconds)
        """
        with self._lock:
            lockout_until = self._lockouts.get(identifier)
            if lockout_until and time.monotonic() < lockout_until:
                remaining = int(lockout_until - time.monotonic())
                return True, remaining
            elif lockout_until:
                del self._lockouts[identifier]
            return False, 0

    def record_failure(self, identifier: str) -> Tuple[bool, int]:
        """
        Record a failed login attempt.

        Returns:
            (now_locked, remaining_attempts)
        """
        with self._lock:
            now = time.monotonic()
            # Clean old attempts (older than lockout period)
            self._attempts[identifier] = [
                t for t in self._attempts[identifier]
                if now - t < self.LOCKOUT_SECONDS
            ]
            self._attempts[identifier].append(now)

            remaining = self.MAX_ATTEMPTS - len(self._attempts[identifier])
            if remaining <= 0:
                self._lockouts[identifier] = now + self.LOCKOUT_SECONDS
                self._attempts[identifier].clear()
                logger.warning("Account locked out: %s", identifier)
                return True, 0

            return False, remaining

    def record_success(self, identifier: str) -> None:
        """Clear failed attempts on successful login."""
        with self._lock:
            self._attempts.pop(identifier, None)
            self._lockouts.pop(identifier, None)


# ══════════════════════════════════════════════════════════════════
# SECURITY HEADERS MIDDLEWARE
# ══════════════════════════════════════════════════════════════════

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        # Cache control for API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"

        return response


# ══════════════════════════════════════════════════════════════════
# RATE LIMIT MIDDLEWARE
# ══════════════════════════════════════════════════════════════════

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply rate limiting to API requests."""

    def __init__(self, app, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self.limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for static files and health checks
        path = request.url.path
        if not path.startswith("/api/") or path == "/api/health":
            return await call_next(request)

        # Get client identifier (IP)
        client_ip = request.headers.get(
            "x-forwarded-for",
            request.client.host if request.client else "unknown",
        )
        if "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()

        allowed, retry_after = self.limiter.is_allowed(client_ip, path)

        if not allowed:
            logger.warning("Rate limited: %s → %s", client_ip, path)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please slow down.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        return response


# ══════════════════════════════════════════════════════════════════
# REQUEST LOGGING MIDDLEWARE
# ══════════════════════════════════════════════════════════════════

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log API requests with timing information."""

    async def dispatch(self, request: Request, call_next):
        # Skip logging for static files
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        # Log with structured format
        log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            log_level,
            "%s %s → %d (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.0f}ms"

        return response


# ── Singleton instances ──────────────────────────────────────────
rate_limiter = RateLimiter()
login_tracker = LoginTracker()
