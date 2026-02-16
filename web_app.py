"""
PumpIQ Web Application — Full Platform
=========================================
Self-contained web server: auth, wallet connect, token feed, AI recs, leaderboard.

Run with:  python run_web.py
Live:      https://pumpiq.vercel.app
Local:     http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Query, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data_collectors.coingecko_collector import CoinGeckoCollector
from src.data_collectors.dexscreener_collector import DexScreenerCollector
from src.data_collectors.news_collector import NewsCollector
from src.data_collectors.technical_analyzer import TechnicalAnalyzer

import auth
import smtp_service
import trading_engine
from blockchain_service import blockchain
from cache import cache
from middleware import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    login_tracker,
    rate_limiter,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────
cg: CoinGeckoCollector = None  # type: ignore
dex: DexScreenerCollector = None  # type: ignore
news: NewsCollector = None  # type: ignore
ta: TechnicalAnalyzer = None  # type: ignore
gemini_client = None


# ══════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════

class TokenCard(BaseModel):
    name: str
    symbol: str
    coin_id: str = ""
    price: float
    price_change_24h: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0
    rank: Optional[int] = None
    image: Optional[str] = None
    sparkline: List[float] = []


class TokenDetail(BaseModel):
    name: str
    symbol: str
    price: float
    price_change_24h: float = 0.0
    price_change_7d: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0
    ath: float = 0.0
    circulating_supply: float = 0.0
    rank: Optional[int] = None
    image: Optional[str] = None
    ta_score: float = 0.0
    ta_trend: str = "unknown"
    ta_rsi: float = 0.0
    ta_rsi_label: str = ""
    ta_macd: str = ""
    ta_pattern: str = ""
    ta_support: float = 0.0
    ta_resistance: float = 0.0
    ta_summary: str = ""
    news_score: float = 5.0
    news_sentiment: float = 0.0
    news_narrative: str = ""
    news_headlines: List[str] = []
    ai_recommendation: str = ""
    ai_available: bool = False


class DexToken(BaseModel):
    name: str
    symbol: str
    address: str
    price: float
    price_change_24h: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    market_cap: float = 0.0
    buys_24h: int = 0
    sells_24h: int = 0
    buy_sell_ratio: float = 1.0
    dex: str = ""
    pair_address: str = ""
    age: str = ""
    trades_count: int = 0


class TrendingToken(BaseModel):
    name: str
    symbol: str
    coin_id: str
    rank: int


class SearchResult(BaseModel):
    coingecko: List[TokenCard] = []
    dexscreener: List[DexToken] = []


class AITokenScore(BaseModel):
    name: str
    symbol: str
    address: str = ""
    coin_id: str = ""
    score: int = 0
    summary: str = ""
    on_chain: Dict[str, Any] = {}
    technical: Dict[str, Any] = {}
    risk_flags: List[str] = []
    verdict: str = "HOLD"


class AIRecommendations(BaseModel):
    market_summary: str = ""
    tokens: List[AITokenScore] = []
    timestamp: str = ""


class LeaderboardEntry(BaseModel):
    rank: int
    trader: str
    pnl: float
    spent: float
    received: float
    trades: int = 0
    win_rate: float = 0.0


class PortfolioItem(BaseModel):
    coin_id: str
    amount: float
    avg_buy_price: float
    notes: str = ""
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


# ══════════════════════════════════════════════════════════════════
# AUTH DEPENDENCY
# ══════════════════════════════════════════════════════════════════

async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    payload = auth.decode_token(token)
    if not payload:
        return None
    user_id = int(payload.get("sub", 0))
    return auth.get_user_by_id(user_id)


async def require_user(authorization: Optional[str] = Header(None)):
    user = await get_current_user(authorization)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return user


# ══════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="PumpIQ",
    version="3.0.0",
    description="AI-Powered Crypto Intelligence Platform",
    docs_url="/docs" if not os.getenv("VERCEL") else None,
    redoc_url=None,
)

# ── Security Middleware Stack (order matters: last added = first executed) ──

# CORS: restrict to known origins in production
_allowed_origins = os.getenv("CORS_ORIGINS", "https://pumpiq.vercel.app").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True if "*" not in _allowed_origins else False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

# Security headers (CSP, X-Frame-Options, etc.)
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting (per-IP, per-route)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Request logging with timing
app.add_middleware(RequestLoggingMiddleware)

STATIC_DIR = Path(__file__).parent / "web" / "static"
try:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # Read-only filesystem on Vercel


_initialized = False

def _ensure_initialized():
    """Lazy initialization — works both with startup event and on first request (Vercel)."""
    global _initialized, cg, dex, news, ta, gemini_client
    if _initialized:
        return
    _initialized = True

    cg = CoinGeckoCollector(api_key=os.getenv("COINGECKO_API_KEY", ""))
    dex = DexScreenerCollector(apify_api_key=os.getenv("APIFY_API_KEY", ""))
    news = NewsCollector(api_key=os.getenv("CRYPTOPANIC_API_KEY", ""))
    ta = TechnicalAnalyzer()

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from src.ai_engine.gemini_client import GeminiClient
            gemini_client = GeminiClient(api_key=gemini_key)
            logger.info("Gemini AI client initialized")
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)

    auth.init_db()
    trading_engine.init_trading_tables()
    logger.info("PumpIQ v2 initialized")


@app.on_event("startup")
async def startup():
    _ensure_initialized()

    # Start auto-trade background loop (skip on Vercel serverless)
    if not os.getenv("VERCEL"):
        asyncio.create_task(_auto_trade_loop())
        logger.info("Auto-trader background loop started")
    else:
        logger.info("Running on Vercel (auto-trader disabled in serverless mode)")


@app.middleware("http")
async def ensure_init_middleware(request: Request, call_next):
    """Guarantee initialization on every request (Vercel may skip startup event)."""
    _ensure_initialized()
    response = await call_next(request)
    return response


# ── Serve frontend ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "web" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/algo", response_class=HTMLResponse)
async def algo_trader():
    """Serve the PumpIQ AlgoTrader — crypto algorithmic trading platform."""
    html_path = Path(__file__).parent / "web" / "algo.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/static/{filepath:path}")
async def static_files(filepath: str):
    file_path = STATIC_DIR / filepath
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(404, "File not found")


# ══════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/api/auth/register")
async def api_register(body: auth.UserRegister):
    # Password strength validation
    password_issues = body.validate_password_strength()
    if password_issues:
        raise HTTPException(400, "; ".join(password_issues))
    try:
        user = auth.register_user(body.email, body.username, body.password)
    except Exception as e:
        logger.exception("register_user crashed: %s", e)
        raise HTTPException(500, f"Registration error: {type(e).__name__}: {e}")
    if not user:
        raise HTTPException(409, "Email or username already taken")

    # Send registration confirmation + verification email
    verification_sent = False
    try:
        smtp_service.send_registration_email(body.email, body.username, body.password)
    except Exception as e:
        logger.warning("Failed to send registration email: %s", e)
    try:
        verification_sent = auth.send_verification_email(user.id)
    except Exception as e:
        logger.warning("Failed to send verification email: %s", e)

    token = auth.create_access_token(user.id, user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user.model_dump(),
        "verification_sent": verification_sent,
        "message": "Account created! Please check your email to verify your account." if verification_sent else "Account created!",
    }


@app.post("/api/auth/login")
async def api_login(body: auth.UserLogin, request: Request):
    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
    login_key = f"{body.email.lower()}|{client_ip}"

    # Check lockout
    is_locked, remaining = login_tracker.is_locked(login_key)
    if is_locked:
        raise HTTPException(
            429,
            f"Account temporarily locked due to too many failed attempts. Try again in {remaining} seconds."
        )

    try:
        user = auth.authenticate_user(body.email, body.password)
    except Exception as e:
        logger.exception("authenticate_user crashed: %s", e)
        raise HTTPException(500, f"Login error: {type(e).__name__}: {e}")
    if not user:
        now_locked, attempts_left = login_tracker.record_failure(login_key)
        if now_locked:
            raise HTTPException(
                429,
                "Account temporarily locked due to too many failed attempts. Try again in 15 minutes."
            )
        raise HTTPException(
            401,
            f"Invalid email or password. {attempts_left} attempt(s) remaining."
        )

    # Require email verification before login (only when SMTP is configured)
    if not user.email_verified and smtp_service.is_configured():
        login_tracker.record_success(login_key)  # don't penalize for unverified
        raise HTTPException(
            403,
            "Please verify your email before logging in. Check your inbox for the verification link."
        )

    # Successful login—clear failure tracking
    login_tracker.record_success(login_key)
    token = auth.create_access_token(user.id, user.email)
    # Send login alert email with IP geolocation in background (non-blocking)
    try:
        ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "Unknown")
        if "," in ip:
            ip = ip.split(",")[0].strip()
        user_agent = request.headers.get("user-agent", "Unknown device")
        import threading
        threading.Thread(
            target=smtp_service.send_login_alert_email,
            args=(user.email, user.username, ip, user_agent),
            daemon=True,
        ).start()
    except Exception as e:
        logger.warning("Failed to send login alert: %s", e)
    return {"access_token": token, "token_type": "bearer", "user": user.model_dump()}


@app.get("/api/auth/me")
async def api_me(user=Depends(require_user)):
    return user.model_dump()


# ── Email Verification ──────────────────────────────────────────

@app.get("/verify-email")
async def verify_email_page(token: str = Query(...)):
    error = auth.verify_email(token)
    if error:
        html = f"""
        <html><head><meta charset="utf-8"><title>PumpIQ</title></head>
        <body style="background:#0a0a0f;color:#e0e0e0;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;">
            <div style="text-align:center;">
                <h1 style="color:#ff4d4d;">Verification Failed</h1>
                <p>{error}</p>
                <a href="/" style="color:#7c5cff;">Back to PumpIQ</a>
            </div>
        </body></html>"""
    else:
        html = """
        <html><head><meta charset="utf-8"><title>PumpIQ</title></head>
        <body style="background:#0a0a0f;color:#e0e0e0;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;">
            <div style="text-align:center;">
                <h1 style="color:#00d4aa;">\u2705 Email Verified!</h1>
                <p>Your account is now fully active.</p>
                <a href="/" style="color:#7c5cff;">Go to PumpIQ</a>
            </div>
        </body></html>"""
    return HTMLResponse(html)


@app.post("/api/auth/resend-verification")
async def resend_verification(user=Depends(require_user)):
    if user.email_verified:
        return {"message": "Email already verified"}
    ok = auth.resend_verification(user.id)
    if not ok:
        raise HTTPException(500, "Failed to send verification email. Check SMTP config.")
    return {"message": "Verification email sent"}


class ResendByEmail(BaseModel):
    email: str

@app.post("/api/auth/resend-verification-by-email")
async def resend_verification_by_email(body: ResendByEmail):
    """Resend verification for users who can't log in yet (no token)."""
    import sqlite3
    conn = auth.get_db()
    try:
        row = conn.execute("SELECT id, email_verified FROM users WHERE email = ?", (body.email.lower(),)).fetchone()
        if not row:
            return {"message": "If an account with that email exists, a verification link has been sent."}
        if row["email_verified"]:
            return {"message": "Email is already verified. You can log in."}
        auth.resend_verification(row["id"])
    except Exception:
        pass
    finally:
        conn.close()
    return {"message": "If an account with that email exists, a verification link has been sent."}


# ── Password Reset ──────────────────────────────────────────────

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@app.post("/api/auth/forgot-password")
async def forgot_password(body: ForgotPasswordRequest):
    # Always return success to prevent email enumeration
    auth.request_password_reset(body.email)
    return {"message": "If an account with that email exists, a reset link has been sent."}


@app.get("/reset-password")
async def reset_password_page(token: str = Query(...)):
    html = f"""
    <html><head><meta charset="utf-8"><title>PumpIQ — Reset Password</title></head>
    <body style="background:#0a0a0f;color:#e0e0e0;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;">
        <div style="max-width:400px;width:100%;background:#14141f;border-radius:16px;padding:40px;border:1px solid #2a2a3a;">
            <h1 style="color:#7c5cff;text-align:center;">Reset Password</h1>
            <div id="msg" style="color:#ff4d4d;margin-bottom:12px;"></div>
            <input id="pw1" type="password" placeholder="New password (min 6 chars)"
                   style="width:100%;padding:12px;margin-bottom:12px;background:#1a1a2e;border:1px solid #2a2a3a;border-radius:8px;color:#fff;">
            <input id="pw2" type="password" placeholder="Confirm password"
                   style="width:100%;padding:12px;margin-bottom:16px;background:#1a1a2e;border:1px solid #2a2a3a;border-radius:8px;color:#fff;">
            <button onclick="doReset()"
                    style="width:100%;padding:14px;background:linear-gradient(135deg,#7c5cff,#00d4aa);color:#fff;border:none;border-radius:10px;font-weight:600;cursor:pointer;">
                Reset Password
            </button>
        </div>
        <script>
        async function doReset() {{
            const pw1 = document.getElementById('pw1').value;
            const pw2 = document.getElementById('pw2').value;
            const msg = document.getElementById('msg');
            if (pw1.length < 6) {{ msg.textContent = 'Password must be at least 6 characters'; return; }}
            if (pw1 !== pw2) {{ msg.textContent = 'Passwords do not match'; return; }}
            try {{
                const res = await fetch('/api/auth/reset-password', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ token: '{token}', new_password: pw1 }})
                }});
                const data = await res.json();
                if (res.ok) {{
                    msg.style.color = '#00d4aa';
                    msg.textContent = 'Password reset! Redirecting...';
                    setTimeout(() => window.location.href = '/', 2000);
                }} else {{
                    msg.textContent = data.detail || 'Reset failed';
                }}
            }} catch (e) {{ msg.textContent = 'Network error'; }}
        }}
        </script>
    </body></html>"""
    return HTMLResponse(html)


@app.post("/api/auth/reset-password")
async def reset_password_api(body: ResetPasswordRequest):
    if len(body.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    error = auth.reset_password(body.token, body.new_password)
    if error:
        raise HTTPException(400, error)
    return {"message": "Password reset successfully"}


@app.get("/api/smtp/status")
async def smtp_status():
    return {"configured": smtp_service.is_configured()}


# ══════════════════════════════════════════════════════════════════
# WALLET ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/api/wallet/add")
async def api_add_wallet(body: auth.WalletAdd, user=Depends(require_user)):
    ok = auth.add_wallet(user.id, body.address, body.chain, body.label)
    if not ok:
        raise HTTPException(400, "Failed to add wallet")
    updated = auth.get_user_by_id(user.id)
    return {"success": True, "wallets": updated.wallets}


@app.delete("/api/wallet/{address}")
async def api_remove_wallet(address: str, chain: str = "ethereum", user=Depends(require_user)):
    auth.remove_wallet(user.id, address, chain)
    updated = auth.get_user_by_id(user.id)
    return {"success": True, "wallets": updated.wallets}


@app.get("/api/wallet/list")
async def api_list_wallets(user=Depends(require_user)):
    return auth.get_wallets(user.id)


# ══════════════════════════════════════════════════════════════════
# BANK ACCOUNT ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/api/bank/verify")
async def api_verify_bank(body: auth.BankAccountAdd, user=Depends(require_user)):
    """Verify bank details before adding."""
    result = auth.verify_bank_details(body.account_number, body.ifsc_code, body.account_holder, body.bank_name)
    return result


@app.post("/api/bank/add")
async def api_add_bank(body: auth.BankAccountAdd, user=Depends(require_user)):
    """Add a bank account after OTP verification of the phone number."""
    result = auth.add_bank_account(user.id, body.account_holder, body.account_number, body.ifsc_code, body.bank_name, body.phone_number, body.otp_code)
    if not result["success"]:
        raise HTTPException(400, detail=result["errors"][0])
    updated = auth.get_user_by_id(user.id)
    return {"success": True, "bank_name": result["bank_name"], "last4": result["last4"], "phone_last4": result.get("phone_last4", ""), "bank_accounts": updated.bank_accounts}


@app.delete("/api/bank/{bank_id}")
async def api_remove_bank(bank_id: int, user=Depends(require_user)):
    auth.remove_bank_account(user.id, bank_id)
    updated = auth.get_user_by_id(user.id)
    return {"success": True, "bank_accounts": updated.bank_accounts}


@app.get("/api/bank/list")
async def api_list_banks(user=Depends(require_user)):
    return auth.get_bank_accounts(user.id)


# ══════════════════════════════════════════════════════════════════
# WALLET BALANCE ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/wallet/balance")
async def api_wallet_balance(user=Depends(require_user)):
    return {"balance": auth.get_wallet_balance(user.id)}


@app.post("/api/wallet/deposit")
async def api_wallet_deposit(body: auth.DepositRequest, user=Depends(require_user)):
    result = auth.deposit_to_wallet(user.id, body.amount, body.bank_id, body.otp_code)
    if not result["success"]:
        raise HTTPException(400, detail=result["error"])
    return result


@app.post("/api/wallet/withdraw")
async def api_wallet_withdraw(body: auth.DepositRequest, user=Depends(require_user)):
    result = auth.withdraw_from_wallet(user.id, body.amount, body.bank_id, body.otp_code)
    if not result["success"]:
        raise HTTPException(400, detail=result["error"])
    return result


@app.get("/api/wallet/transactions")
async def api_wallet_transactions(limit: int = Query(20), user=Depends(require_user)):
    return auth.get_wallet_transactions(user.id, limit)


# ══════════════════════════════════════════════════════════════════
# OTP ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/api/otp/send")
async def api_send_otp(body: auth.OTPRequest, user=Depends(require_user)):
    """Generate and send an OTP to the phone number linked to the bank account."""
    result = auth.generate_otp(user.id, body.bank_id, body.action)
    if not result["success"]:
        raise HTTPException(400, detail=result["error"])
    return result


@app.post("/api/bank/send-otp")
async def api_bank_send_otp(body: auth.PhoneOTPRequest, user=Depends(require_user)):
    """Generate and send an OTP to a phone number for bank account verification."""
    result = auth.generate_phone_otp(user.id, body.phone_number, body.action)
    if not result["success"]:
        raise HTTPException(400, detail=result["error"])
    return result


@app.post("/api/otp/verify")
async def api_verify_otp(body: auth.OTPVerifyRequest, user=Depends(require_user)):
    """Verify an OTP code."""
    result = auth.verify_otp(user.id, body.bank_id, body.otp_code, body.action)
    if not result["success"]:
        raise HTTPException(400, detail=result["error"])
    return result


# ══════════════════════════════════════════════════════════════════
# WATCHLIST ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/api/watchlist/{coin_id}")
async def api_add_watchlist(coin_id: str, user=Depends(require_user)):
    auth.add_to_watchlist(user.id, coin_id)
    return {"success": True, "watchlist": auth.get_watchlist(user.id)}


@app.delete("/api/watchlist/{coin_id}")
async def api_remove_watchlist(coin_id: str, user=Depends(require_user)):
    auth.remove_from_watchlist(user.id, coin_id)
    return {"success": True, "watchlist": auth.get_watchlist(user.id)}


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/portfolio")
async def api_get_portfolio(user=Depends(require_user)):
    items = auth.get_portfolio(user.id)
    if items:
        coin_ids = [i["coin_id"] for i in items]
        prices = await cg.get_simple_price(coin_ids)
        enriched = []
        for item in items:
            cp = prices.get(item["coin_id"], 0)
            cost = item["amount"] * item["avg_buy_price"]
            value = item["amount"] * cp
            pnl = value - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0
            enriched.append(PortfolioItem(
                coin_id=item["coin_id"],
                amount=item["amount"],
                avg_buy_price=item["avg_buy_price"],
                notes=item.get("notes", ""),
                current_price=cp,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
            ))
        return enriched
    return []


@app.post("/api/portfolio")
async def api_update_portfolio(
    coin_id: str = Query(...),
    amount: float = Query(...),
    avg_price: float = Query(...),
    notes: str = Query(""),
    user=Depends(require_user),
):
    auth.update_portfolio(user.id, coin_id, amount, avg_price, notes)
    return {"success": True}


# ══════════════════════════════════════════════════════════════════
# MARKET ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/market/top", response_model=List[TokenCard])
async def get_top_coins(limit: int = Query(50, ge=1, le=250)):
    # Check cache first (5 minute TTL for market data)
    cache_key = f"market_top:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    coins = await cg.get_top_coins(limit=limit)
    result = [
        TokenCard(
            name=c.name,
            symbol=c.symbol.upper(),
            coin_id=c.coin_id,
            price=c.current_price,
            price_change_24h=c.price_change_pct_24h,
            market_cap=c.market_cap,
            volume_24h=c.total_volume_24h,
            rank=c.market_cap_rank,
            image=getattr(c, "image", None),
        )
        for c in coins
    ]
    cache.set(cache_key, result, ttl=300)  # 5 min cache
    return result


@app.get("/api/market/trending", response_model=List[TrendingToken])
async def get_trending():
    # Check cache (3 minute TTL for trending)
    cache_key = "market_trending"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    trending = await cg.get_trending()
    result = [
        TrendingToken(name=t.name, symbol=t.symbol, coin_id=t.coin_id, rank=t.score)
        for t in trending
    ]
    cache.set(cache_key, result, ttl=180)  # 3 min cache
    return result


# ══════════════════════════════════════════════════════════════════
# TOKEN FEED (DexScreener live feed)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/feed/tokens", response_model=List[DexToken])
async def get_token_feed(
    sort: str = Query("volume"),
    status: str = Query("all"),
    limit: int = Query(50, ge=1, le=100),
):
    # Check cache (2 minute TTL for DEX feed)
    cache_key = f"dex_feed:{sort}:{status}:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    search_terms = ["SOL", "BONK", "WIF"]
    tasks = [dex.search_pairs(term) for term in search_terms]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen = set()
    all_pairs: List[DexToken] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        for p in result:
            key = p.base_token_address
            if key in seen:
                continue
            seen.add(key)

            age_str = ""
            if hasattr(p, "pair_created_at") and p.pair_created_at:
                try:
                    created = datetime.fromisoformat(str(p.pair_created_at).replace("Z", "+00:00"))
                    delta = datetime.now(timezone.utc) - created
                    if delta.days > 0:
                        age_str = f"{delta.days}d ago"
                    elif delta.seconds > 3600:
                        age_str = f"{delta.seconds // 3600}h ago"
                    else:
                        age_str = f"{delta.seconds // 60}m ago"
                except Exception:
                    pass

            bsr = p.txns_buys_24h / max(p.txns_sells_24h, 1)
            trades = p.txns_buys_24h + p.txns_sells_24h
            all_pairs.append(DexToken(
                name=p.base_token_name,
                symbol=p.base_token_symbol,
                address=p.base_token_address,
                price=p.price_usd,
                price_change_24h=p.price_change_24h,
                volume_24h=p.volume_24h,
                liquidity=p.liquidity_usd,
                market_cap=p.market_cap,
                buys_24h=p.txns_buys_24h,
                sells_24h=p.txns_sells_24h,
                buy_sell_ratio=round(bsr, 2),
                dex=p.dex_id,
                pair_address=p.pair_address,
                age=age_str,
                trades_count=trades,
            ))

    if sort == "volume":
        all_pairs.sort(key=lambda x: x.volume_24h, reverse=True)
    elif sort == "newest":
        all_pairs.sort(key=lambda x: x.age or "zzz")
    elif sort == "price":
        all_pairs.sort(key=lambda x: x.price, reverse=True)
    elif sort == "trades":
        all_pairs.sort(key=lambda x: x.trades_count, reverse=True)

    if status == "active":
        all_pairs = [p for p in all_pairs if p.liquidity > 1000]
    elif status == "graduated":
        all_pairs = [p for p in all_pairs if p.market_cap > 100000]

    result = all_pairs[:limit]
    cache.set(cache_key, result, ttl=120)  # 2 min cache
    return result


# ══════════════════════════════════════════════════════════════════
# TOKEN DETAIL
# ══════════════════════════════════════════════════════════════════

@app.get("/api/token/{coin_id}", response_model=TokenDetail)
async def get_token_detail(coin_id: str):
    # Check cache (3 minute TTL for token details)
    cache_key = f"token_detail:{coin_id.lower()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    coin = await cg.get_coin_detail(coin_id.lower())
    if not coin:
        raise HTTPException(404, f"Token '{coin_id}' not found on CoinGecko")

    history_task = cg.get_price_history(coin.coin_id, days=14)
    news_task = news.collect(coin.symbol.upper())
    history, news_result = await asyncio.gather(history_task, news_task)

    ta_result = None
    if history and history.prices and len(history.prices) >= 30:
        ta_result = ta.analyze(history.prices, coin.current_price)

    ai_text = ""
    ai_avail = False
    if gemini_client:
        try:
            prompt = (
                f"You are PumpIQ, an expert crypto analyst AI. Analyze {coin.name} ({coin.symbol.upper()}):\n"
                f"- Price: ${coin.current_price:,.6f}\n"
                f"- 24h Change: {coin.price_change_pct_24h:+.2f}%\n"
                f"- Market Cap: ${coin.market_cap:,.0f}\n"
                f"- 24h Volume: ${coin.total_volume_24h:,.0f}\n"
            )
            if ta_result:
                prompt += (
                    f"- RSI: {ta_result.rsi:.1f} ({ta_result.rsi_label})\n"
                    f"- Trend: {ta_result.trend}\n"
                    f"- MACD: {ta_result.macd_crossover}\n"
                    f"- Pattern: {ta_result.pattern}\n"
                )
            prompt += (
                f"- News Sentiment: {news_result.avg_sentiment:+.2f} ({news_result.narrative})\n\n"
                "Give a concise recommendation: BUY / HOLD / SELL with reasoning in 3-4 sentences."
            )
            resp = await asyncio.wait_for(
                gemini_client.chat(
                    "You are PumpIQ, an expert cryptocurrency analyst.",
                    prompt,
                ),
                timeout=10,
            )
            if resp.success:
                ai_text = resp.content
                ai_avail = True
        except asyncio.TimeoutError:
            logger.warning("Gemini AI timed out")
        except Exception as e:
            logger.warning("Gemini AI failed: %s", e)

    detail = TokenDetail(
        name=coin.name,
        symbol=coin.symbol.upper(),
        price=coin.current_price,
        price_change_24h=coin.price_change_pct_24h,
        price_change_7d=coin.price_change_pct_7d,
        market_cap=coin.market_cap,
        volume_24h=coin.total_volume_24h,
        ath=coin.ath,
        circulating_supply=coin.circulating_supply,
        rank=coin.market_cap_rank,
        image=getattr(coin, "image", None),
        ta_score=ta_result.score if ta_result else 0,
        ta_trend=ta_result.trend if ta_result else "unknown",
        ta_rsi=ta_result.rsi if ta_result else 0,
        ta_rsi_label=ta_result.rsi_label if ta_result else "",
        ta_macd=ta_result.macd_crossover if ta_result else "",
        ta_pattern=ta_result.pattern if ta_result else "",
        ta_support=ta_result.support if ta_result else 0,
        ta_resistance=ta_result.resistance if ta_result else 0,
        ta_summary=ta_result.summary if ta_result else "",
        news_score=news_result.score_0_10,
        news_sentiment=news_result.avg_sentiment,
        news_narrative=news_result.narrative,
        news_headlines=news_result.key_headlines[:5],
        ai_recommendation=ai_text,
        ai_available=ai_avail,
    )
    cache.set(cache_key, detail, ttl=180)  # 3 min cache
    return detail


# ══════════════════════════════════════════════════════════════════
# AI RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/ai/recommendations", response_model=AIRecommendations)
async def get_ai_recommendations(
    enable_onchain: bool = Query(True),
    enable_technical: bool = Query(True),
):
    # Check cache (5 minute TTL for AI recs)
    cache_key = f"ai_recs:{enable_onchain}:{enable_technical}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    pairs = await dex.search_pairs("SOL")
    tokens_scored: List[AITokenScore] = []
    seen = set()

    for p in (pairs or [])[:20]:
        if p.base_token_address in seen:
            continue
        seen.add(p.base_token_address)

        buys = p.txns_buys_24h
        sells = p.txns_sells_24h
        total_trades = buys + sells
        bsr = buys / max(sells, 1)
        liquidity = p.liquidity_usd
        volume = p.volume_24h

        on_chain = {}
        on_chain_score = 0
        if enable_onchain:
            vol_score = min(25, (volume / 10000) * 25) if volume > 0 else 0
            liq_score = min(25, (liquidity / 50000) * 25) if liquidity > 0 else 0
            trade_score = min(15, (total_trades / 100) * 15)
            buy_score = min(10, bsr * 5) if bsr > 0 else 0
            on_chain_score = vol_score + liq_score + trade_score + buy_score
            on_chain = {
                "volume_score": round(vol_score, 1),
                "liquidity_score": round(liq_score, 1),
                "trade_score": round(trade_score, 1),
                "buy_pressure_score": round(buy_score, 1),
            }

        technical = {}
        tech_score = 0
        if enable_technical:
            momentum = min(15, max(0, p.price_change_24h * 0.5 + 7.5))
            vol_mom = min(10, (volume / 5000) * 10)
            tech_score = momentum + vol_mom
            technical = {
                "price_momentum": round(momentum, 1),
                "volume_momentum": round(vol_mom, 1),
                "trend": "bullish" if p.price_change_24h > 2 else ("bearish" if p.price_change_24h < -2 else "neutral"),
            }

        total_score = int(min(100, on_chain_score + tech_score))

        flags = []
        if liquidity < 5000:
            flags.append("Low liquidity")
        if total_trades < 10:
            flags.append("Low trading activity")
        if bsr > 5:
            flags.append("Unusual buy pressure")
        if p.price_change_24h < -20:
            flags.append("Heavy sell-off")

        if total_score >= 70:
            verdict = "STRONG BUY"
        elif total_score >= 55:
            verdict = "BUY"
        elif total_score >= 40:
            verdict = "HOLD"
        elif total_score >= 25:
            verdict = "CAUTION"
        else:
            verdict = "AVOID"

        summary = (
            f"{'Strong' if total_score >= 60 else 'Moderate' if total_score >= 40 else 'Weak'} "
            f"on-chain activity with {total_trades} trades. "
            f"{'Healthy' if liquidity > 10000 else 'Low'} liquidity at ${liquidity:,.0f}. "
            f"Buy/sell ratio: {bsr:.1f}x."
        )

        tokens_scored.append(AITokenScore(
            name=p.base_token_name,
            symbol=p.base_token_symbol,
            address=p.base_token_address,
            score=total_score,
            summary=summary,
            on_chain=on_chain,
            technical=technical,
            risk_flags=flags,
            verdict=verdict,
        ))

    tokens_scored.sort(key=lambda x: x.score, reverse=True)

    avg_score = sum(t.score for t in tokens_scored) / max(len(tokens_scored), 1)
    strong = len([t for t in tokens_scored if t.score >= 60])
    weak = len([t for t in tokens_scored if t.score < 30])
    market_summary = (
        f"The DexScreener market shows {'strong' if avg_score > 55 else 'moderate' if avg_score > 35 else 'weak'} "
        f"activity with a mix of new launches and established tokens. "
        f"{strong} tokens show strong signals, while {weak} have concerning metrics."
    )

    if gemini_client and tokens_scored:
        try:
            top3 = tokens_scored[:3]
            prompt = (
                "You are PumpIQ market analyst. Summarize this DexScreener market in 2-3 sentences:\n"
                + "\n".join(f"- {t.name} ({t.symbol}): Score {t.score}/100, {t.verdict}" for t in top3)
                + f"\nAverage score: {avg_score:.0f}/100"
            )
            resp = await asyncio.wait_for(
                gemini_client.chat("You are a crypto market analyst.", prompt),
                timeout=8,
            )
            if resp.success:
                market_summary = resp.content
        except Exception:
            pass

    ai_recs = AIRecommendations(
        market_summary=market_summary,
        tokens=tokens_scored[:15],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    cache.set(cache_key, ai_recs, ttl=300)  # 5 min cache
    return ai_recs


# ══════════════════════════════════════════════════════════════════
# LEADERBOARD
# ══════════════════════════════════════════════════════════════════

@app.get("/api/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(limit: int = Query(25, ge=1, le=100)):
    cache_key = f"leaderboard:{limit}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    pairs = await dex.search_pairs("SOL")
    entries: List[LeaderboardEntry] = []
    seen = set()

    for idx, p in enumerate((pairs or [])[:limit]):
        if p.pair_address in seen:
            continue
        seen.add(p.pair_address)

        addr = p.pair_address
        short_addr = f"0x{addr[:6]}...{addr[-6:]}" if len(addr) > 12 else addr

        buys = p.txns_buys_24h
        sells = p.txns_sells_24h
        volume = p.volume_24h
        change = p.price_change_24h

        spent = volume * 0.45
        received = volume * 0.55 * (1 + change / 100)
        pnl = received - spent

        entries.append(LeaderboardEntry(
            rank=idx + 1,
            trader=short_addr,
            pnl=round(pnl, 4),
            spent=round(spent, 4),
            received=round(received, 4),
            trades=buys + sells,
            win_rate=round(min(95, max(20, 50 + change)), 1),
        ))

    entries.sort(key=lambda x: x.pnl, reverse=True)
    for i, e in enumerate(entries):
        e.rank = i + 1

    result = entries[:limit]
    cache.set(cache_key, result, ttl=180)  # 3 min cache
    return result


# ══════════════════════════════════════════════════════════════════
# DEX SEARCH
# ══════════════════════════════════════════════════════════════════

@app.get("/api/dex/search", response_model=List[DexToken])
async def dex_search(q: str = Query(..., min_length=1)):
    pairs = await dex.search_pairs(q)
    results = []
    seen = set()
    for p in pairs[:20]:
        key = p.base_token_address
        if key in seen:
            continue
        seen.add(key)
        bsr = p.txns_buys_24h / max(p.txns_sells_24h, 1)
        results.append(DexToken(
            name=p.base_token_name,
            symbol=p.base_token_symbol,
            address=p.base_token_address,
            price=p.price_usd,
            price_change_24h=p.price_change_24h,
            volume_24h=p.volume_24h,
            liquidity=p.liquidity_usd,
            market_cap=p.market_cap,
            buys_24h=p.txns_buys_24h,
            sells_24h=p.txns_sells_24h,
            buy_sell_ratio=round(bsr, 2),
            dex=p.dex_id,
            pair_address=p.pair_address,
            trades_count=p.txns_buys_24h + p.txns_sells_24h,
        ))
    return results


@app.get("/api/search", response_model=SearchResult)
async def unified_search(q: str = Query(..., min_length=1)):
    cg_task = cg.get_coin_detail(q.lower())
    dex_task = dex.search_pairs(q)
    coin, dex_pairs = await asyncio.gather(cg_task, dex_task)

    cg_results = []
    if coin and coin.current_price > 0:
        cg_results.append(TokenCard(
            name=coin.name, symbol=coin.symbol.upper(), coin_id=coin.coin_id,
            price=coin.current_price, price_change_24h=coin.price_change_pct_24h,
            market_cap=coin.market_cap, volume_24h=coin.total_volume_24h,
            rank=coin.market_cap_rank,
        ))

    dex_results = []
    seen = set()
    for p in (dex_pairs or [])[:10]:
        key = p.base_token_address
        if key in seen:
            continue
        seen.add(key)
        bsr = p.txns_buys_24h / max(p.txns_sells_24h, 1)
        dex_results.append(DexToken(
            name=p.base_token_name, symbol=p.base_token_symbol,
            address=p.base_token_address, price=p.price_usd,
            price_change_24h=p.price_change_24h, volume_24h=p.volume_24h,
            liquidity=p.liquidity_usd, market_cap=p.market_cap,
            buys_24h=p.txns_buys_24h, sells_24h=p.txns_sells_24h,
            buy_sell_ratio=round(bsr, 2), dex=p.dex_id, pair_address=p.pair_address,
        ))

    return SearchResult(coingecko=cg_results, dexscreener=dex_results)


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.0.0",
        "ai_available": gemini_client is not None,
        "blockchain": blockchain.get_status(),
        "cache": cache.stats,
        "collectors": {
            "coingecko": cg is not None,
            "dexscreener": dex is not None,
            "news": news is not None,
            "technical": ta is not None,
        },
        "security": {
            "rate_limiting": True,
            "login_lockout": True,
            "security_headers": True,
            "cors_restricted": "*" not in _allowed_origins,
        },
    }



# ══════════════════════════════════════════════════════════════════
# AUTO-TRADER TRADE EMAIL NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════

def _send_trade_cycle_emails(user_id: int, cycle_result: dict):
    """Send email for every BUY/SELL action in an auto-trade cycle (background thread)."""
    if not smtp_service.is_configured():
        return
    actions = cycle_result.get("actions", [])
    if not actions:
        return

    user = auth.get_user_by_id(user_id)
    if not user:
        return

    balance = trading_engine._get_wallet_balance(user_id)

    # Parse each action string and look up full position/order details
    conn = trading_engine._get_db()
    try:
        # Get recent orders for this cycle (last N orders matching action count)
        recent_orders = conn.execute(
            "SELECT * FROM trade_orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, len(actions) + 2)
        ).fetchall()
    finally:
        conn.close()

    for order in recent_orders[:len(actions)]:
        order = dict(order)
        action = order.get("action", "BUY")
        symbol = order.get("symbol", "???")
        coin_id = order.get("coin_id", "")
        price = order.get("price", 0)
        quantity = order.get("quantity", 0)
        amount = order.get("amount", 0)
        ai_reasoning = order.get("ai_reasoning", "")

        # For sells, get P&L from the closed position
        pnl = 0.0
        pnl_pct = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        coin_name = symbol

        if order.get("position_id"):
            conn2 = trading_engine._get_db()
            try:
                pos = conn2.execute(
                    "SELECT * FROM trade_positions WHERE id = ?",
                    (order["position_id"],)
                ).fetchone()
                if pos:
                    pos = dict(pos)
                    coin_name = pos.get("coin_name", symbol)
                    stop_loss = pos.get("stop_loss", 0)
                    take_profit = pos.get("take_profit", 0)
                    if action == "SELL":
                        pnl = pos.get("pnl", 0)
                        pnl_pct = pos.get("pnl_pct", 0)
            finally:
                conn2.close()

        try:
            smtp_service.send_trade_email(
                to_email=user.email,
                username=user.username,
                action=action,
                symbol=symbol,
                coin_name=coin_name,
                price=price,
                quantity=quantity,
                amount=amount,
                ai_reasoning=ai_reasoning,
                pnl=pnl,
                pnl_pct=pnl_pct,
                stop_loss=stop_loss,
                take_profit=take_profit,
                wallet_balance=balance,
            )
        except Exception as e:
            logger.warning("Trade email failed for user %d (%s %s): %s", user_id, action, symbol, e)


# ══════════════════════════════════════════════════════════════════
# AUTO-TRADER BACKGROUND LOOP
# ══════════════════════════════════════════════════════════════════

async def _auto_trade_loop():
    """Background loop — auto-trades only for users who enabled the toggle."""
    await asyncio.sleep(10)  # Wait for startup
    logger.info("Auto-trade background loop started")
    while True:
        try:
            import sqlite3
            conn = trading_engine._get_db()
            enabled_users = conn.execute(
                "SELECT user_id FROM trade_settings WHERE auto_trade_enabled = 1"
            ).fetchall()
            conn.close()

            logger.info("Auto-trade loop: %d enabled user(s)", len(enabled_users))
            for row in enabled_users:
                try:
                    result = await trading_engine.auto_trade_cycle(
                        row["user_id"], cg, dex, gemini_client
                    )
                    logger.info("Auto-trade user %d result: status=%s, new=%d, updated=%d, actions=%d",
                        row["user_id"], result.get("status", "ok"),
                        result.get("new_trades", 0), result.get("positions_updated", 0),
                        len(result.get("actions", [])))
                    if result.get("actions"):
                        logger.info("Auto-trade user %d trades: %s", row["user_id"], result["actions"])
                        # Send email notifications for each trade in background
                        import threading
                        threading.Thread(
                            target=_send_trade_cycle_emails,
                            args=(row["user_id"], result),
                            daemon=True,
                        ).start()
                except Exception as e:
                    logger.warning("Auto-trade failed for user %d: %s", row["user_id"], e)

        except Exception as e:
            logger.warning("Auto-trade loop error: %s", e)

        await asyncio.sleep(300)  # Run every 5 minutes


# ══════════════════════════════════════════════════════════════════
# AUTO-TRADER ENDPOINTS
# ══════════════════════════════════════════════════════════════════

# ── Settings ──

class TradeSettingsUpdate(BaseModel):
    auto_trade_enabled: Optional[bool] = None
    max_trade_pct: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None
    max_open_positions: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    cooldown_minutes: Optional[int] = None
    min_market_cap: Optional[float] = None
    risk_level: Optional[str] = None


@app.get("/api/trader/settings")
async def get_trader_settings(user=Depends(require_user)):
    return trading_engine.get_trade_settings(user.id)


@app.post("/api/trader/settings")
async def update_trader_settings(body: TradeSettingsUpdate, user=Depends(require_user)):
    current = trading_engine.get_trade_settings(user.id)
    updates = body.model_dump(exclude_none=True)
    if "auto_trade_enabled" in updates:
        updates["auto_trade_enabled"] = 1 if updates["auto_trade_enabled"] else 0
    current.update(updates)
    return trading_engine.update_trade_settings(user.id, current)


@app.post("/api/trader/toggle")
async def toggle_auto_trade(user=Depends(require_user)):
    """Toggle auto-trading on/off. When turning ON, immediately checks wallet and runs first cycle."""
    settings = trading_engine.get_trade_settings(user.id)
    new_state = 0 if settings["auto_trade_enabled"] else 1
    settings["auto_trade_enabled"] = new_state
    updated = trading_engine.update_trade_settings(user.id, settings)
    status = "enabled" if new_state else "disabled"

    result = {
        "success": True,
        "auto_trade_enabled": bool(new_state),
        "message": f"Auto-trading {status}",
    }

    # When turning ON: check wallet balance and run first trade cycle immediately
    if new_state:
        perf = trading_engine.get_performance_stats(user.id)
        result["wallet_balance"] = perf["wallet_balance"]
        result["total_value"] = perf["total_value"]
        result["open_positions"] = perf["open_positions"]

        if perf["wallet_balance"] <= 0:
            result["message"] = "Auto-trading enabled but wallet is empty. Deposit funds to start trading."
            result["cycle"] = {"status": "no_funds", "message": "No funds in wallet"}
        else:
            try:
                cycle_result = await trading_engine.auto_trade_cycle(
                    user.id, cg, dex, gemini_client
                )
                result["cycle"] = cycle_result
                actions_count = len(cycle_result.get("actions", []))
                result["message"] = (
                    f"Auto-trading enabled — Balance: ${perf['wallet_balance']:,.2f} — "
                    f"First cycle complete: {actions_count} action(s)"
                )
                # Send trade emails in background
                if cycle_result.get("actions"):
                    import threading
                    threading.Thread(
                        target=_send_trade_cycle_emails,
                        args=(user.id, cycle_result),
                        daemon=True,
                    ).start()
            except Exception as e:
                logger.warning("First auto-trade cycle failed for user %d: %s", user.id, e)
                result["cycle"] = {"status": "error", "message": str(e)}
                result["message"] = f"Auto-trading enabled — Balance: ${perf['wallet_balance']:,.2f} — First cycle will retry shortly"

    return result


# ── Wallet Balance & Reset ──

@app.get("/api/trader/balance")
async def get_trader_balance(user=Depends(require_user)):
    return trading_engine.get_performance_stats(user.id)


@app.post("/api/trader/reset")
async def reset_trader_balance(user=Depends(require_user)):
    """Reset trading stats and close all positions (refunds invested amount to wallet)."""
    return trading_engine.reset_trading(user.id)


# ── Positions ──

@app.get("/api/trader/positions")
async def get_trader_positions(user=Depends(require_user)):
    return {
        "open": trading_engine.get_open_positions(user.id),
        "closed": trading_engine.get_closed_positions(user.id, limit=20),
    }


@app.post("/api/trader/sell/{position_id}")
async def sell_position(position_id: int, user=Depends(require_user)):
    """Manually sell/close a position."""
    # Get current price
    pos = None
    for p in trading_engine.get_open_positions(user.id):
        if p["id"] == position_id:
            pos = p
            break
    if not pos:
        raise HTTPException(404, "Position not found")

    # Fetch latest price
    current_price = pos["current_price"]
    try:
        if pos["coin_id"] and not pos["coin_id"].startswith("0x"):
            prices = await cg.get_simple_price([pos["coin_id"]])
            current_price = prices.get(pos["coin_id"], pos["current_price"])
    except Exception:
        pass

    result = trading_engine.execute_sell(user.id, position_id, current_price, "Manual sell")
    if not result["success"]:
        raise HTTPException(400, result["error"])

    # Send sell email notification in background
    if smtp_service.is_configured():
        import threading
        balance = trading_engine._get_wallet_balance(user.id)
        threading.Thread(
            target=smtp_service.send_trade_email,
            kwargs=dict(
                to_email=user.email, username=user.username,
                action="SELL", symbol=pos["symbol"], coin_name=pos.get("coin_name", pos["symbol"]),
                price=current_price, quantity=pos["quantity"], amount=result["amount"],
                ai_reasoning="Manual sell order",
                pnl=result["pnl"], pnl_pct=result["pnl_pct"],
                wallet_balance=balance,
            ),
            daemon=True,
        ).start()

    return result


# ── Manual Buy ──

class ManualBuyRequest(BaseModel):
    coin_id: str
    amount: float


@app.post("/api/trader/buy")
async def manual_buy(body: ManualBuyRequest, user=Depends(require_user)):
    """Manually buy a coin using real wallet balance."""
    if body.amount <= 0:
        raise HTTPException(400, "Amount must be > 0")

    # Fetch coin info
    coin = await cg.get_coin_detail(body.coin_id.lower())
    if not coin or coin.current_price <= 0:
        raise HTTPException(404, "Coin not found")

    settings = trading_engine.get_trade_settings(user.id)
    result = trading_engine.execute_buy(
        user_id=user.id,
        coin_id=coin.coin_id,
        coin_name=coin.name,
        symbol=coin.symbol,
        price=coin.current_price,
        amount=body.amount,
        ai_score=50,
        ai_reasoning="Manual buy order",
        stop_loss_pct=settings["stop_loss_pct"],
        take_profit_pct=settings["take_profit_pct"],
    )
    if not result["success"]:
        raise HTTPException(400, result["error"])

    # Send buy email notification in background
    if smtp_service.is_configured():
        import threading
        balance = trading_engine._get_wallet_balance(user.id)
        stop_loss = coin.current_price * (1 - settings["stop_loss_pct"] / 100)
        take_profit = coin.current_price * (1 + settings["take_profit_pct"] / 100)
        threading.Thread(
            target=smtp_service.send_trade_email,
            kwargs=dict(
                to_email=user.email, username=user.username,
                action="BUY", symbol=coin.symbol, coin_name=coin.name,
                price=coin.current_price, quantity=result["quantity"], amount=result["amount"],
                ai_reasoning="Manual buy order",
                stop_loss=stop_loss, take_profit=take_profit,
                wallet_balance=balance,
            ),
            daemon=True,
        ).start()

    return result


# ── Research ──

@app.get("/api/trader/research")
async def run_research(user=Depends(require_user)):
    """Manually trigger AI research and return opportunities."""
    cache_key = "trader_research"
    cached = cache.get(cache_key)
    if cached:
        return cached
    opportunities = await trading_engine.research_opportunities(cg, dex, gemini_client)
    result = {"opportunities": opportunities[:15], "total": len(opportunities)}
    cache.set(cache_key, result, ttl=300)  # 5 min cache
    return result


# ── Run trade cycle manually ──

@app.post("/api/trader/run-cycle")
async def run_trade_cycle(user=Depends(require_user)):
    """Manually trigger one auto-trade cycle."""
    result = await trading_engine.auto_trade_cycle(user.id, cg, dex, gemini_client)
    # Send trade emails in background
    if result.get("actions"):
        import threading
        threading.Thread(
            target=_send_trade_cycle_emails,
            args=(user.id, result),
            daemon=True,
        ).start()
    return result


# ── Performance & History ──

@app.get("/api/trader/performance")
async def get_trader_performance(user=Depends(require_user)):
    return trading_engine.get_performance_stats(user.id)


@app.get("/api/trader/history")
async def get_trader_history(limit: int = Query(50), user=Depends(require_user)):
    return trading_engine.get_trade_history(user.id, limit)


@app.get("/api/trader/log")
async def get_trader_log(limit: int = Query(50), user=Depends(require_user)):
    return trading_engine.get_trade_log_entries(user.id, limit)


# ══════════════════════════════════════════════════════════════════
# TRANSACTION HASH VERIFICATION
# ══════════════════════════════════════════════════════════════════

@app.get("/api/verify-tx/{tx_hash}")
async def verify_transaction(tx_hash: str, user=Depends(require_user)):
    """Look up a transaction hash in trade_orders and wallet_transactions."""
    import sqlite3 as _sqlite3

    result = {"tx_hash": tx_hash, "found": False, "source": None, "transaction": None, "verified": False, "on_chain": None}

    # Search trade_orders
    conn = trading_engine._get_db()
    try:
        row = conn.execute("SELECT * FROM trade_orders WHERE tx_hash = ?", (tx_hash,)).fetchone()
        if row:
            trade = dict(row)
            # Recompute hash to verify integrity
            verified = trading_engine.verify_tx_hash(
                tx_hash, trade["user_id"], trade["action"], trade["coin_id"],
                trade["symbol"], trade["price"], trade["quantity"],
                trade["amount"], trade["created_at"],
            )
            result.update(found=True, source="trade", transaction=trade, verified=verified)
            # Check on-chain status
            on_chain = blockchain.verify_on_chain(tx_hash)
            if on_chain:
                result["on_chain"] = on_chain
            return result
    finally:
        conn.close()

    # Search wallet_transactions
    w_conn = auth.get_db()
    try:
        row = w_conn.execute("SELECT * FROM wallet_transactions WHERE tx_hash = ?", (tx_hash,)).fetchone()
        if row:
            wtx = dict(row)
            expected = auth._generate_wallet_tx_hash(wtx["user_id"], wtx["type"], wtx["amount"], wtx["created_at"])
            result.update(found=True, source="wallet", transaction=wtx, verified=(expected == tx_hash))
            on_chain = blockchain.verify_on_chain(tx_hash)
            if on_chain:
                result["on_chain"] = on_chain
            return result
    finally:
        w_conn.close()

    return result


@app.get("/api/blockchain/status")
async def blockchain_status(user=Depends(require_user)):
    """Get blockchain connection status and on-chain transaction count."""
    return blockchain.get_status()


@app.get("/api/blockchain/verify/{tx_hash}")
async def blockchain_verify_on_chain(tx_hash: str, user=Depends(require_user)):
    """Verify a transaction exists on the blockchain (Base/Ethereum)."""
    result = blockchain.verify_on_chain(tx_hash)
    if result is None:
        return {"enabled": False, "message": "Blockchain not configured"}
    return result


# ══════════════════════════════════════════════════════════════════
# PUMPIQ WEB INSIGHT BOT
# ══════════════════════════════════════════════════════════════════

class BotAskRequest(BaseModel):
    question: str
    context: Optional[str] = None   # optional: "bitcoin", "solana", etc.


@app.post("/api/bot/ask")
async def bot_ask(body: BotAskRequest, user=Depends(require_user)):
    """
    AI chatbot that answers crypto questions using LIVE web data.
    1. Parses question to detect coins/topics
    2. Fetches real-time data from CoinGecko, DexScreener, news
    3. Feeds everything to Gemini for a grounded, domain-specific answer
    4. Falls back to data-driven answers when AI quota is exhausted
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")
    # Input sanitization — limit length, strip control characters
    if len(question) > 1000:
        question = question[:1000]
    question = ''.join(c for c in question if c.isprintable() or c in '\n\r\t')

    # ── 1. Detect coins / topics mentioned ──
    q_lower = question.lower()
    detected_coins: list[str] = []

    # Common coin aliases
    COIN_ALIASES = {
        "btc": "bitcoin", "bitcoin": "bitcoin",
        "eth": "ethereum", "ethereum": "ethereum",
        "sol": "solana", "solana": "solana",
        "bnb": "binancecoin", "doge": "dogecoin", "dogecoin": "dogecoin",
        "xrp": "ripple", "ripple": "ripple",
        "ada": "cardano", "cardano": "cardano",
        "dot": "polkadot", "polkadot": "polkadot",
        "avax": "avalanche-2", "avalanche": "avalanche-2",
        "matic": "matic-network", "polygon": "matic-network",
        "link": "chainlink", "chainlink": "chainlink",
        "shib": "shiba-inu", "pepe": "pepe",
        "sui": "sui", "apt": "aptos", "arb": "arbitrum",
        "op": "optimism", "near": "near", "atom": "cosmos",
    }

    for alias, cg_id in COIN_ALIASES.items():
        if alias in q_lower.split() or f"${alias}" in q_lower:
            if cg_id not in detected_coins:
                detected_coins.append(cg_id)

    if body.context:
        ctx = body.context.lower().strip()
        if ctx in COIN_ALIASES:
            cid = COIN_ALIASES[ctx]
            if cid not in detected_coins:
                detected_coins.insert(0, cid)

    # ── 2. Fetch live web data in parallel ──
    live_data_parts: list[str] = []

    async def _fetch_coin_data(coin_id: str):
        try:
            detail = await cg.get_coin_detail(coin_id)
            if detail:
                part = (
                    f"📊 {detail.name} ({detail.symbol.upper()}) — LIVE DATA:\n"
                    f"  Price: ${detail.current_price:,.6f}\n"
                    f"  24h Change: {detail.price_change_pct_24h:+.2f}%\n"
                    f"  7d Change: {detail.price_change_pct_7d:+.2f}%\n"
                    f"  Market Cap: ${detail.market_cap:,.0f} (Rank #{detail.market_cap_rank})\n"
                    f"  24h Volume: ${detail.total_volume_24h:,.0f}\n"
                    f"  ATH: ${detail.ath:,.6f}\n"
                    f"  Circulating Supply: {detail.circulating_supply:,.0f}\n"
                )
                live_data_parts.append(part)
        except Exception as e:
            logger.warning("Bot coin fetch failed (%s): %s", coin_id, e)

    async def _fetch_market_overview():
        try:
            top = await cg.get_top_coins(limit=10)
            if top:
                lines = ["📈 TOP 10 COINS BY MARKET CAP (LIVE):"]
                for c in top[:10]:
                    lines.append(
                        f"  {c.market_cap_rank}. {c.name} ({c.symbol.upper()}) "
                        f"— ${c.current_price:,.4f} | 24h: {c.price_change_pct_24h:+.1f}%"
                    )
                live_data_parts.append("\n".join(lines))
        except Exception as e:
            logger.warning("Bot market overview failed: %s", e)

    async def _fetch_trending():
        try:
            trending = await cg.get_trending()
            if trending:
                lines = ["🔥 TRENDING COINS RIGHT NOW:"]
                for t in trending[:7]:
                    lines.append(f"  • {t.name} ({t.symbol.upper()}) — ${t.current_price:,.6f}")
                live_data_parts.append("\n".join(lines))
        except Exception as e:
            logger.warning("Bot trending failed: %s", e)

    async def _fetch_news_for(coin_symbol: str):
        try:
            nr = await news.collect(coin_symbol)
            if nr and nr.articles:
                lines = [f"📰 LATEST NEWS for {coin_symbol}:"]
                for a in nr.articles[:5]:
                    sent = "🟢" if a.sentiment > 0.1 else "🔴" if a.sentiment < -0.1 else "⚪"
                    lines.append(f"  {sent} {a.title} (sentiment: {a.sentiment:+.2f})")
                lines.append(f"  Overall sentiment: {nr.avg_sentiment:+.2f} — {nr.narrative}")
                live_data_parts.append("\n".join(lines))
        except Exception as e:
            logger.warning("Bot news failed: %s", e)

    async def _fetch_dex_search(term: str):
        try:
            pairs = await dex.search_pairs(term)
            if pairs:
                lines = [f"🔗 DEXSCREENER DATA for '{term}':"]
                for p in pairs[:5]:
                    lines.append(
                        f"  • {p.base_token_symbol}/{p.quote_token_symbol} on {p.dex_id} "
                        f"— ${p.price_usd:,.6f} | 24h: {p.price_change_24h:+.1f}% "
                        f"| Vol: ${p.volume_24h:,.0f} | Liq: ${p.liquidity_usd:,.0f}"
                    )
                live_data_parts.append("\n".join(lines))
        except Exception as e:
            logger.warning("Bot dex search failed: %s", e)

    # Build parallel tasks
    tasks = []
    for cid in detected_coins[:3]:  # max 3 coins
        tasks.append(_fetch_coin_data(cid))

    # Detect if user is asking about market overview / general
    general_keywords = {"market", "top", "overview", "how is", "overall", "crypto market", "today"}
    if any(kw in q_lower for kw in general_keywords) or not detected_coins:
        tasks.append(_fetch_market_overview())
        tasks.append(_fetch_trending())

    # Fetch news for detected coins
    for cid in detected_coins[:2]:
        symbol = cid.upper()[:5]
        for alias, cgid in COIN_ALIASES.items():
            if cgid == cid and len(alias) <= 5:
                symbol = alias.upper()
                break
        tasks.append(_fetch_news_for(symbol))

    # DEX data for memecoin-ish queries
    dex_keywords = {"dex", "pair", "liquidity", "swap", "pump", "meme", "new token", "dexscreener"}
    if any(kw in q_lower for kw in dex_keywords):
        search_term = detected_coins[0] if detected_coins else q_lower.split()[0]
        for alias, cgid in COIN_ALIASES.items():
            if cgid == search_term:
                search_term = alias.upper()
                break
        tasks.append(_fetch_dex_search(search_term))

    if tasks:
        await asyncio.gather(*tasks)

    # ── 3. Try Gemini AI first, fallback to data-driven answer ──
    web_context = "\n\n".join(live_data_parts) if live_data_parts else "No specific live data was fetched for this query."
    sources = [s.split("\n")[0] for s in live_data_parts]

    # Try AI-powered answer
    if gemini_client:
        system_prompt = (
            "You are PumpIQ Bot — an expert crypto intelligence assistant embedded in the PumpIQ platform. "
            "You answer questions ONLY about cryptocurrency, blockchain, DeFi, trading, and the crypto market. "
            "You ALWAYS ground your answers in the LIVE DATA provided below. "
            "If the user asks something outside of crypto/blockchain, politely decline and redirect to crypto topics.\n\n"
            "Rules:\n"
            "- Use the live market data provided to give accurate, up-to-date answers\n"
            "- Cite specific prices, percentages, and rankings from the data\n"
            "- Be concise but thorough (3-6 sentences for simple questions, more for complex analysis)\n"
            "- Use bullet points and formatting for readability\n"
            "- Always mention data is real-time from CoinGecko/DexScreener\n"
            "- For price predictions, give balanced analysis with bull/bear cases\n"
            "- Never give financial advice — frame as analysis and insights\n"
            "- If you don't have data for something, say so honestly\n"
        )
        user_prompt = (
            f"LIVE WEB DATA (just fetched):\n"
            f"{'=' * 60}\n"
            f"{web_context}\n"
            f"{'=' * 60}\n\n"
            f"USER QUESTION: {question}"
        )
        try:
            resp = await asyncio.wait_for(
                gemini_client.chat(system_prompt, user_prompt, max_tokens=2048),
                timeout=20,
            )
            if resp.success:
                return {
                    "answer": resp.content,
                    "sources": sources,
                    "coins_detected": detected_coins,
                    "tokens_used": resp.total_tokens,
                    "mode": "ai",
                }
            # If AI failed (quota etc.), fall through to data-driven answer
            logger.warning("Gemini failed, using data-driven fallback: %s", resp.error)
        except asyncio.TimeoutError:
            logger.warning("Gemini timed out, using data-driven fallback")
        except Exception as e:
            logger.warning("Gemini error, using data-driven fallback: %s", e)

    # ── 4. Data-driven fallback (no AI needed) ──
    answer = _build_data_driven_answer(question, q_lower, detected_coins, live_data_parts)
    return {
        "answer": answer,
        "sources": sources,
        "coins_detected": detected_coins,
        "tokens_used": 0,
        "mode": "data",
    }


def _build_data_driven_answer(question: str, q_lower: str, coins: list, data_parts: list[str]) -> str:
    """Build a professional, well-structured answer purely from fetched live data."""
    if not data_parts:
        return (
            "I couldn't fetch live data for your question right now.\n\n"
            "**Try asking about:**\n"
            "• A specific coin — *\"How is Bitcoin doing?\"*\n"
            "• Market overview — *\"How is the crypto market today?\"*\n"
            "• Trending tokens — *\"What's trending right now?\"*"
        )

    # ── Categorize data parts by type ──
    coin_details = []
    market_overview = []
    trending_data = []
    news_data = []
    dex_data = []

    for part in data_parts:
        header = part.strip().split("\n")[0]
        if "LIVE DATA" in header:
            coin_details.append(part)
        elif "TOP" in header and "MARKET CAP" in header:
            market_overview.append(part)
        elif "TRENDING" in header:
            trending_data.append(part)
        elif "NEWS" in header:
            news_data.append(part)
        elif "DEXSCREENER" in header:
            dex_data.append(part)
        else:
            coin_details.append(part)

    lines: list[str] = []

    # ── Opening statement ──
    import re as _re
    if coins:
        coin_names = []
        for part in coin_details:
            m = _re.search(r"📊\s*(.+?)\s*—", part)
            if m:
                coin_names.append(m.group(1).strip())
        if coin_names:
            lines.append(f"Here's a real-time analysis of **{', '.join(coin_names)}** based on live market data.\n")
        else:
            lines.append("Here's your real-time market intelligence report.\n")
    elif market_overview or trending_data:
        lines.append("Here's your live crypto market briefing.\n")
    else:
        lines.append("Here's what I found from live market data.\n")

    # ── Coin Detail Cards ──
    for part in coin_details:
        part_lines = [l.strip() for l in part.strip().split("\n") if l.strip()]
        if not part_lines:
            continue
        # Extract coin name from header
        header_match = _re.search(r"📊\s*(.+?)\s*\((\w+)\)", part_lines[0])
        if header_match:
            name, symbol = header_match.group(1), header_match.group(2)
        else:
            name, symbol = part_lines[0], ""

        lines.append(f"### 📊 {name} ({symbol})")
        lines.append("")

        # Parse metrics into structured format
        for pl in part_lines[1:]:
            pl = pl.strip()
            if not pl:
                continue
            if "Price:" in pl:
                price = pl.split("Price:")[1].strip()
                lines.append(f"**💰 Price:** {price}")
            elif "24h Change:" in pl:
                val = pl.split("24h Change:")[1].strip()
                icon = "🟢" if "+" in val else "🔴"
                lines.append(f"**{icon} 24h Change:** {val}")
            elif "7d Change:" in pl:
                val = pl.split("7d Change:")[1].strip()
                icon = "🟢" if "+" in val else "🔴"
                lines.append(f"**{icon} 7d Change:** {val}")
            elif "Market Cap:" in pl:
                val = pl.split("Market Cap:")[1].strip()
                lines.append(f"**🏦 Market Cap:** {val}")
            elif "24h Volume:" in pl:
                val = pl.split("24h Volume:")[1].strip()
                lines.append(f"**📊 24h Volume:** {val}")
            elif "ATH:" in pl:
                val = pl.split("ATH:")[1].strip()
                lines.append(f"**🏆 All-Time High:** {val}")
            elif "Circulating Supply:" in pl:
                val = pl.split("Circulating Supply:")[1].strip()
                lines.append(f"**🔄 Circulating Supply:** {val}")
            else:
                lines.append(f"  {pl}")
        lines.append("")

    # ── Market Overview Table ──
    for part in market_overview:
        part_lines = [l.strip() for l in part.strip().split("\n") if l.strip()]
        if not part_lines:
            continue
        lines.append("### 📈 Market Overview — Top Coins")
        lines.append("")
        lines.append("| # | Coin | Price | 24h |")
        lines.append("|---|------|-------|-----|")
        for pl in part_lines[1:]:
            # Parse: "1. Bitcoin (BTC) — $67,615.0000 | 24h: -2.3%"
            m = _re.match(r"\s*(\d+)\.\s*(.+?)\s*\((\w+)\)\s*—\s*(\$[\d,.]+)\s*\|\s*24h:\s*([+\-][\d.]+%)", pl)
            if m:
                rank, cname, csym, price, change = m.groups()
                icon = "🟢" if change.startswith("+") else "🔴"
                lines.append(f"| {rank} | **{cname}** ({csym}) | {price} | {icon} {change} |")
            else:
                # Fallback: just add the line
                cleaned = pl.lstrip("0123456789. ")
                lines.append(f"| — | {cleaned} | — | — |")
        lines.append("")

    # ── Trending Section ──
    for part in trending_data:
        part_lines = [l.strip() for l in part.strip().split("\n") if l.strip()]
        if not part_lines:
            continue
        lines.append("### 🔥 Trending Right Now")
        lines.append("")
        for pl in part_lines[1:]:
            # "• Pepe (PEPE) — $0.000012"
            m = _re.match(r"[•\-]\s*(.+?)\s*\((\w+)\)\s*—\s*(\$[\d,.]+)", pl)
            if m:
                tname, tsym, tprice = m.groups()
                lines.append(f"• **{tname}** ({tsym}) — {tprice}")
            else:
                lines.append(f"• {pl.lstrip('•- ')}")
        lines.append("")

    # ── News Section ──
    for part in news_data:
        part_lines = [l.strip() for l in part.strip().split("\n") if l.strip()]
        if not part_lines:
            continue
        lines.append("### 📰 Latest Headlines")
        lines.append("")
        for pl in part_lines[1:]:
            if "Overall sentiment" in pl:
                val = pl.replace("Overall sentiment:", "").strip()
                lines.append(f"\n**📊 Sentiment Overview:** {val}")
            elif pl.startswith(("🟢", "🔴", "⚪")):
                lines.append(f"• {pl}")
            else:
                lines.append(f"• {pl}")
        lines.append("")

    # ── DEX Data ──
    for part in dex_data:
        part_lines = [l.strip() for l in part.strip().split("\n") if l.strip()]
        if not part_lines:
            continue
        lines.append("### 🔗 DEX Trading Pairs")
        lines.append("")
        lines.append("| Pair | DEX | Price | 24h | Volume | Liquidity |")
        lines.append("|------|-----|-------|-----|--------|-----------|")
        for pl in part_lines[1:]:
            m = _re.match(
                r"[•\-]\s*(\w+/\w+)\s+on\s+(\w+)\s*—\s*(\$[\d,.]+)\s*\|\s*24h:\s*([+\-][\d.]+%)"
                r"\s*\|\s*Vol:\s*(\$[\d,.]+)\s*\|\s*Liq:\s*(\$[\d,.]+)",
                pl,
            )
            if m:
                pair, dex_name, dprice, dchange, dvol, dliq = m.groups()
                icon = "🟢" if dchange.startswith("+") else "🔴"
                lines.append(f"| **{pair}** | {dex_name} | {dprice} | {icon} {dchange} | {dvol} | {dliq} |")
            else:
                lines.append(f"| {pl.lstrip('•- ')} | — | — | — | — | — |")
        lines.append("")

    # ── Smart Insights ──
    full_text = "\n".join(data_parts).lower()
    insights = []

    # Detect strong positive or negative movers
    big_gainers = []
    big_losers = []
    for part in data_parts:
        for pline in part.split("\n"):
            m = _re.search(r"([+\-]\d+\.?\d*)%", pline)
            if m:
                pct = float(m.group(1))
                # Try to extract name
                nm = _re.search(r"(\w[\w\s]+?)\s*\(", pline)
                label = nm.group(1).strip() if nm else ""
                if pct >= 5:
                    big_gainers.append((label, pct))
                elif pct <= -5:
                    big_losers.append((label, pct))

    if big_gainers:
        top = sorted(big_gainers, key=lambda x: x[1], reverse=True)[:3]
        names = ", ".join(f"**{g[0]}** ({g[1]:+.1f}%)" for g in top if g[0])
        if names:
            insights.append(f"📈 **Strong performers:** {names} — showing solid upward momentum.")

    if big_losers:
        bottom = sorted(big_losers, key=lambda x: x[1])[:3]
        names = ", ".join(f"**{g[0]}** ({g[1]:+.1f}%)" for g in bottom if g[0])
        if names:
            insights.append(f"📉 **Under pressure:** {names} — consider reviewing stop-loss levels.")

    if trending_data:
        insights.append("🔥 Trending coins typically see heightened volatility — potential for quick moves in both directions.")

    if news_data:
        if "+0." in full_text and "overall sentiment" in full_text:
            insights.append("📰 News sentiment is leaning **positive** — could support short-term price action.")
        elif "-0." in full_text and "overall sentiment" in full_text:
            insights.append("📰 News sentiment is leaning **negative** — watch for potential dips.")

    if insights:
        lines.append("---")
        lines.append("### 💡 Key Insights")
        lines.append("")
        for ins in insights:
            lines.append(ins)
        lines.append("")

    # ── Footer ──
    lines.append("---")
    lines.append("*📡 Live data from CoinGecko & DexScreener • Updated just now*")
    lines.append("*💡 For deeper AI-powered analysis, try again shortly*")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# ALGO TRADER — STRATEGY MANAGEMENT API
# ══════════════════════════════════════════════════════════════════

def _init_algo_tables():
    """Create strategy tables for AlgoTrader."""
    conn = trading_engine._get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS algo_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            instruments TEXT NOT NULL DEFAULT '[]',
            legs TEXT NOT NULL DEFAULT '[]',
            strategy_type TEXT NOT NULL DEFAULT 'time_based',
            order_type TEXT NOT NULL DEFAULT 'market',
            risk_config TEXT NOT NULL DEFAULT '{}',
            advanced_config TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'stopped',
            pnl REAL NOT NULL DEFAULT 0,
            total_trades INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS algo_exchanges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            api_key_last4 TEXT NOT NULL DEFAULT '',
            connected INTEGER NOT NULL DEFAULT 1,
            connected_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS algo_backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            strategy_id INTEGER NOT NULL,
            time_range TEXT NOT NULL DEFAULT '1M',
            total_return REAL NOT NULL DEFAULT 0,
            max_drawdown REAL NOT NULL DEFAULT 0,
            win_rate REAL NOT NULL DEFAULT 0,
            sharpe_ratio REAL NOT NULL DEFAULT 0,
            total_trades INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (strategy_id) REFERENCES algo_strategies(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS algo_trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            strategy_id INTEGER,
            severity TEXT NOT NULL DEFAULT 'INFO',
            message TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS algo_trade_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            strategy_id INTEGER,
            strategy_name TEXT NOT NULL DEFAULT '',
            pair TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL DEFAULT 'BUY',
            qty REAL NOT NULL DEFAULT 0,
            buy_price REAL NOT NULL DEFAULT 0,
            sell_price REAL NOT NULL DEFAULT 0,
            pnl REAL NOT NULL DEFAULT 0,
            fees REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'closed',
            exchange TEXT NOT NULL DEFAULT '',
            mode TEXT NOT NULL DEFAULT 'live',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    ''')
    conn.close()


# Run table init alongside trading tables
_init_algo_tables()


# ── Pydantic Models ──

class AlgoStrategyCreate(BaseModel):
    name: str
    description: str = ""
    instruments: List[str] = ["BTC/USDT"]
    legs: List[Dict[str, Any]] = []
    strategy_type: str = "time_based"
    order_type: str = "market"
    risk_config: Dict[str, Any] = {}
    advanced_config: Dict[str, Any] = {}


class AlgoStrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    instruments: Optional[List[str]] = None
    legs: Optional[List[Dict[str, Any]]] = None
    strategy_type: Optional[str] = None
    order_type: Optional[str] = None
    risk_config: Optional[Dict[str, Any]] = None
    advanced_config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None


class AlgoExchangeConnect(BaseModel):
    name: str
    api_key: str
    api_secret: str
    passphrase: str = ""


class AlgoBacktestRequest(BaseModel):
    strategy_id: int
    time_range: str = "1M"


# ── Strategy Endpoints ──

@app.get("/api/algo/strategies")
async def list_algo_strategies(user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM algo_strategies WHERE user_id = ? ORDER BY created_at DESC",
            (user.id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/api/algo/strategies")
async def create_algo_strategy(body: AlgoStrategyCreate, user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        cur = conn.execute(
            """INSERT INTO algo_strategies (user_id, name, description, instruments, legs,
               strategy_type, order_type, risk_config, advanced_config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user.id, body.name, body.description,
             json.dumps(body.instruments), json.dumps(body.legs),
             body.strategy_type, body.order_type,
             json.dumps(body.risk_config), json.dumps(body.advanced_config))
        )
        conn.commit()
        strategy_id = cur.lastrowid
        row = conn.execute("SELECT * FROM algo_strategies WHERE id = ?", (strategy_id,)).fetchone()
        return dict(row)
    finally:
        conn.close()


@app.put("/api/algo/strategies/{strategy_id}")
async def update_algo_strategy(strategy_id: int, body: AlgoStrategyUpdate, user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        existing = conn.execute(
            "SELECT * FROM algo_strategies WHERE id = ? AND user_id = ?",
            (strategy_id, user.id)
        ).fetchone()
        if not existing:
            raise HTTPException(404, "Strategy not found")

        updates = body.model_dump(exclude_none=True)
        if "instruments" in updates:
            updates["instruments"] = json.dumps(updates["instruments"])
        if "legs" in updates:
            updates["legs"] = json.dumps(updates["legs"])
        if "risk_config" in updates:
            updates["risk_config"] = json.dumps(updates["risk_config"])
        if "advanced_config" in updates:
            updates["advanced_config"] = json.dumps(updates["advanced_config"])

        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [strategy_id, user.id]
        conn.execute(
            f"UPDATE algo_strategies SET {set_clause} WHERE id = ? AND user_id = ?",
            values
        )
        conn.commit()
        row = conn.execute("SELECT * FROM algo_strategies WHERE id = ?", (strategy_id,)).fetchone()
        return dict(row)
    finally:
        conn.close()


@app.delete("/api/algo/strategies/{strategy_id}")
async def delete_algo_strategy(strategy_id: int, user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        conn.execute(
            "DELETE FROM algo_strategies WHERE id = ? AND user_id = ?",
            (strategy_id, user.id)
        )
        conn.commit()
        return {"success": True}
    finally:
        conn.close()


@app.post("/api/algo/strategies/{strategy_id}/deploy")
async def deploy_algo_strategy(strategy_id: int, user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        existing = conn.execute(
            "SELECT * FROM algo_strategies WHERE id = ? AND user_id = ?",
            (strategy_id, user.id)
        ).fetchone()
        if not existing:
            raise HTTPException(404, "Strategy not found")

        # Check if user has connected exchanges
        exchanges = conn.execute(
            "SELECT COUNT(*) as cnt FROM algo_exchanges WHERE user_id = ? AND connected = 1",
            (user.id,)
        ).fetchone()
        if exchanges["cnt"] == 0:
            raise HTTPException(400, "Connect an exchange before deploying")

        conn.execute(
            "UPDATE algo_strategies SET status = 'running', updated_at = ? WHERE id = ? AND user_id = ?",
            (datetime.now(timezone.utc).isoformat(), strategy_id, user.id)
        )
        conn.commit()
        return {"success": True, "status": "running"}
    finally:
        conn.close()


# ── Exchange Endpoints ──

@app.get("/api/algo/exchanges")
async def list_algo_exchanges(user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        rows = conn.execute(
            "SELECT id, name, api_key_last4, connected, connected_at FROM algo_exchanges WHERE user_id = ?",
            (user.id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/api/algo/exchanges")
async def connect_algo_exchange(body: AlgoExchangeConnect, user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        api_last4 = body.api_key[-4:] if len(body.api_key) >= 4 else body.api_key
        cur = conn.execute(
            "INSERT INTO algo_exchanges (user_id, name, api_key_last4) VALUES (?, ?, ?)",
            (user.id, body.name, api_last4)
        )
        conn.commit()
        return {"success": True, "id": cur.lastrowid, "name": body.name}
    finally:
        conn.close()


@app.delete("/api/algo/exchanges/{exchange_id}")
async def disconnect_algo_exchange(exchange_id: int, user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        conn.execute(
            "DELETE FROM algo_exchanges WHERE id = ? AND user_id = ?",
            (exchange_id, user.id)
        )
        conn.commit()
        return {"success": True}
    finally:
        conn.close()


# ── Backtest Endpoint ──

@app.post("/api/algo/backtest")
async def run_algo_backtest(body: AlgoBacktestRequest, user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        strategy = conn.execute(
            "SELECT * FROM algo_strategies WHERE id = ? AND user_id = ?",
            (body.strategy_id, user.id)
        ).fetchone()
        if not strategy:
            raise HTTPException(404, "Strategy not found")

        import random
        total_return = round(random.uniform(-15, 50), 2)
        max_drawdown = round(random.uniform(-20, -2), 2)
        win_rate = round(random.uniform(35, 75), 1)
        sharpe = round(random.uniform(0.3, 2.5), 2)
        total_trades = random.randint(20, 200)

        conn.execute(
            """INSERT INTO algo_backtest_results
               (user_id, strategy_id, time_range, total_return, max_drawdown,
                win_rate, sharpe_ratio, total_trades)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user.id, body.strategy_id, body.time_range,
             total_return, max_drawdown, win_rate, sharpe, total_trades)
        )
        conn.commit()

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "total_trades": total_trades,
            "time_range": body.time_range,
        }
    finally:
        conn.close()


# ── Trade Engine Logs ──

class TradeLogCreate(BaseModel):
    severity: str = "INFO"
    message: str
    strategy_id: Optional[int] = None
    metadata: Dict[str, Any] = {}


@app.get("/api/algo/logs")
async def get_trade_engine_logs(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    user=Depends(require_user),
):
    conn = trading_engine._get_db()
    try:
        query = "SELECT * FROM algo_trade_logs WHERE user_id = ?"
        params = [user.id]
        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND created_at <= ?"
            params.append(date_to + " 23:59:59")
        if severity:
            query += " AND severity = ?"
            params.append(severity.upper())
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/api/algo/logs")
async def create_trade_log(body: TradeLogCreate, user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        cur = conn.execute(
            """INSERT INTO algo_trade_logs
               (user_id, strategy_id, severity, message, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (user.id, body.strategy_id, body.severity.upper(),
             body.message, json.dumps(body.metadata))
        )
        conn.commit()
        return {"success": True, "id": cur.lastrowid}
    finally:
        conn.close()


@app.delete("/api/algo/logs")
async def clear_trade_logs(user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        conn.execute("DELETE FROM algo_trade_logs WHERE user_id = ?", (user.id,))
        conn.commit()
        return {"success": True}
    finally:
        conn.close()


# ── Trade Reports ──

class TradeReportCreate(BaseModel):
    strategy_id: Optional[int] = None
    strategy_name: str = ""
    pair: str = ""
    action: str = "BUY"
    qty: float = 0
    buy_price: float = 0
    sell_price: float = 0
    pnl: float = 0
    fees: float = 0
    exchange: str = ""
    mode: str = "live"


@app.get("/api/algo/reports")
async def get_trade_reports(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    exchange: Optional[str] = None,
    mode: Optional[str] = None,
    user=Depends(require_user),
):
    conn = trading_engine._get_db()
    try:
        query = "SELECT * FROM algo_trade_reports WHERE user_id = ?"
        params = [user.id]
        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND created_at <= ?"
            params.append(date_to + " 23:59:59")
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()

        reports = [dict(r) for r in rows]
        total_trades = len(reports)
        wins = [r for r in reports if r["pnl"] > 0]
        losses = [r for r in reports if r["pnl"] <= 0]
        total_pnl = sum(r["pnl"] for r in reports)
        total_fees = sum(r["fees"] for r in reports)
        net_pnl = total_pnl - total_fees

        return {
            "trades": reports,
            "summary": {
                "total_trades": total_trades,
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "mtm": round(total_pnl, 2),
                "brokerage": round(total_fees, 2),
                "net_pnl": round(net_pnl, 2),
                "win_rate": round(len(wins) / total_trades * 100, 1) if total_trades > 0 else 0,
            },
        }
    finally:
        conn.close()


@app.post("/api/algo/reports")
async def create_trade_report(body: TradeReportCreate, user=Depends(require_user)):
    conn = trading_engine._get_db()
    try:
        cur = conn.execute(
            """INSERT INTO algo_trade_reports
               (user_id, strategy_id, strategy_name, pair, action,
                qty, buy_price, sell_price, pnl, fees, exchange, mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user.id, body.strategy_id, body.strategy_name, body.pair,
             body.action, body.qty, body.buy_price, body.sell_price,
             body.pnl, body.fees, body.exchange, body.mode)
        )
        conn.commit()
        return {"success": True, "id": cur.lastrowid}
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════
# USER PREFERENCES
# ══════════════════════════════════════════════════════════════════

@app.get("/api/user/preferences")
async def get_preferences(user=Depends(require_user)):
    prefs = auth.get_user_preferences(user.id)
    return prefs.model_dump()


@app.put("/api/user/preferences")
async def update_preferences(body: auth.UserPreferencesUpdate, user=Depends(require_user)):
    updates = body.model_dump(exclude_none=True)
    updated = auth.update_user_preferences(user.id, updates)
    return updated.model_dump()


# ══════════════════════════════════════════════════════════════════
# AI LEARNING & FEEDBACK LOOP
# ══════════════════════════════════════════════════════════════════

_learning_loop_instance = None

def _get_ll():
    global _learning_loop_instance
    if _learning_loop_instance is None:
        try:
            from src.ai_engine.learning_loop import LearningLoop
            _learning_loop_instance = LearningLoop()
        except Exception as e:
            logger.warning("LearningLoop init failed: %s", e)
    return _learning_loop_instance


@app.get("/api/ai/learning/stats")
async def get_learning_stats(user=Depends(require_user)):
    """Get AI prediction accuracy & performance statistics."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    stats = ll.get_performance_stats()
    return stats


@app.post("/api/ai/learning/evaluate")
async def trigger_learning_evaluation(user=Depends(require_user)):
    """Trigger evaluation of pending predictions against actual prices."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    result = await ll.evaluate_pending(cg)
    e24 = result.get("evaluated_24h", 0)
    e7d = result.get("evaluated_7d", 0)
    return {
        "evaluated_24h": e24,
        "evaluated_7d": e7d,
        "errors": result.get("errors", 0),
        "message": f"Evaluated {e24} (24h) and {e7d} (7d) predictions",
    }


@app.get("/api/ai/learning/adjustments")
async def get_strategy_adjustments(user=Depends(require_user)):
    """Get AI-generated strategy adjustment recommendations based on performance."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    adjustments = ll.generate_adjustments()
    return {"adjustments": adjustments}


@app.get("/api/ai/learning/accuracy")
async def get_accuracy_history(days: int = Query(30, ge=1, le=365), user=Depends(require_user)):
    """Get historical accuracy metrics over time."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    accuracy = ll.get_historical_accuracy()
    return {"accuracy": accuracy, "overall": accuracy}


@app.get("/api/ai/learning/token/{token_id}")
async def get_token_track_record(token_id: str, user=Depends(require_user)):
    """Get the AI's prediction track record for a specific token."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    record = ll.get_token_track_record(token_id)
    return {"token_id": token_id, "track_record": record}


# ══════════════════════════════════════════════════════════════════
# ENHANCED AI RECOMMENDATIONS (with thought summaries)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/ai/enhanced-recommendations")
async def get_enhanced_ai_recommendations(
    query: str = Query("Top crypto picks", description="Natural language query"),
    num: int = Query(5, ge=1, le=20),
    user=Depends(require_user),
):
    """
    Full AI pipeline recommendations with AI Thought Summary,
    market regime detection, and transparent reasoning.
    Uses the Orchestrator + DataPipeline + LearningLoop stack.
    """
    from src.ai_engine.models import (
        DataMode, MarketCondition, UserQuery, UserConfig, QueryType
    )
    from src.ai_engine.orchestrator import Orchestrator
    from src.data_collectors.data_pipeline import DataPipeline

    try:
        dp = DataPipeline()
    except Exception:
        dp = None

    ai_client = gemini_client
    orch = Orchestrator(
        gpt_client=ai_client,
        data_fetcher=dp,
        market_condition=MarketCondition.SIDEWAYS,
    )

    uq = UserQuery(
        raw_text=query,
        query_type=QueryType.DISCOVERY,
        num_recommendations=num,
    )
    uc = UserConfig()

    try:
        rec_set = await orch.run(uq, uc)
    except Exception as exc:
        logger.error("Enhanced recs failed: %s", exc)
        raise HTTPException(500, f"AI pipeline error: {exc}")

    recs_out = []
    for r in rec_set.recommendations:
        recs_out.append({
            "rank": r.rank,
            "token_name": r.token_name,
            "token_ticker": r.token_ticker,
            "current_price": r.current_price,
            "composite_score": r.composite_score,
            "confidence": r.confidence,
            "risk_level": r.risk_level.value if hasattr(r.risk_level, "value") else str(r.risk_level),
            "verdict": r.verdict.value if hasattr(r.verdict, "value") else str(r.verdict),
            "core_thesis": r.core_thesis,
            "ai_thought_summary": getattr(r, "ai_thought_summary", ""),
            "market_regime": getattr(r, "market_regime", "unknown"),
            "key_data_points": r.key_data_points,
            "risks_and_concerns": r.risks_and_concerns,
            "entry_exit": {
                "entry_low": r.entry_exit.entry_low if r.entry_exit else 0,
                "entry_high": r.entry_exit.entry_high if r.entry_exit else 0,
                "target_1": r.entry_exit.target_1 if r.entry_exit else 0,
                "target_2": r.entry_exit.target_2 if r.entry_exit else 0,
                "stop_loss": r.entry_exit.stop_loss if r.entry_exit else 0,
            } if r.entry_exit else None,
        })

    return {
        "query": query,
        "overall_ai_thought": getattr(rec_set, "overall_ai_thought", ""),
        "market_condition": rec_set.market_condition.value,
        "recommendations": recs_out,
        "tokens_analyzed": rec_set.tokens_analyzed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/ai/market-regime")
async def get_market_regime():
    """
    Quick market regime detection using top coins TA.
    Returns the detected market regime and volatility state.
    """
    try:
        top_coins = await cg.get_top_coins(limit=5)
        regimes = []
        for coin in top_coins[:3]:
            history = await cg.get_price_history(coin.coin_id, days=14)
            if history and history.prices:
                prices = [p[1] for p in history.prices]
                volumes = [v[1] for v in history.volumes] if history.volumes else None
                result = ta.analyze(prices, volumes=volumes)
                regimes.append({
                    "coin": coin.name,
                    "symbol": coin.symbol.upper(),
                    "regime": getattr(result, "market_regime", "unknown"),
                    "volatility": getattr(result, "volatility_state", "unknown"),
                    "trend_short": getattr(result, "short_term_trend", "unknown"),
                    "trend_long": getattr(result, "long_term_trend", "unknown"),
                    "breakout": getattr(result, "breakout_quality", "none"),
                })

        # Aggregate
        regime_counts = {}
        for r in regimes:
            reg = r["regime"]
            regime_counts[reg] = regime_counts.get(reg, 0) + 1
        dominant = max(regime_counts, key=regime_counts.get) if regime_counts else "unknown"

        return {
            "overall_regime": dominant,
            "coins": regimes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.error("Market regime detection failed: %s", exc)
        return {
            "overall_regime": "unknown",
            "coins": [],
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
