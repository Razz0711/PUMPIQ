"""
NEXYPHER Web Application — Full Platform
=========================================
Self-contained web server: auth, wallet connect, token feed, AI recs, leaderboard.

Run with:  python run_web.py
Open:      http://localhost:8000
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import logging
import os
import re as _re
import threading
from contextlib import asynccontextmanager
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
from middleware import login_tracker
from supabase_db import get_supabase
from src.backtest_engine import get_backtest_engine, BacktestResult

# ── UI / NLP Layer ────────────────────────────────────────────────
from src.ui.clarification_engine import parse_user_query, ClarificationEngine
from src.ui.intent_recognizer import IntentRecognizer
from src.ui.parameter_extractor import ParameterExtractor
from src.ui.response_formatter import ResponseFormatter
from src.ui.visual_indicators import (
    confidence_bar, confidence_bar_html, risk_badge, risk_badge_html,
    trend_arrow, trend_arrow_html, data_freshness_indicator,
    verdict_colour, verdict_emoji, score_sparkline,
)
from src.ui.notification_formatter import NotificationFormatter
from src.ui.personalization_engine import PersonalizationEngine
from src.ui.user_config import UserPreferences, default_preferences, PortfolioHolding
from src.ui.watchlist_manager import WatchlistManager
from src.ui.portfolio_tracker import PortfolioTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── UI singletons (stateless, reusable) ──────────────────────────
_intent_recognizer = IntentRecognizer()
_param_extractor = ParameterExtractor()
_clarification_engine = ClarificationEngine()
_response_formatter = ResponseFormatter()
_notification_formatter = NotificationFormatter()

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
    price_change_7d: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0
    rank: Optional[int] = None
    image: Optional[str] = None
    sparkline: List[float] = []
    # AI scoring fields
    ai_score: int = 0
    ai_verdict: str = ""           # STRONG BUY, BUY, HOLD, CAUTION, AVOID
    ai_signal: str = ""            # long, short, neutral
    ai_reasons: List[str] = []
    vol_mcap_ratio: float = 0.0
    ath_distance_pct: float = 0.0   # how far below ATH (0-100)
    trending: bool = False


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
    # Backtest verification fields
    backtest_verified: bool = False
    backtest_recommendation: str = ""
    backtest_win_rate: float = 0.0
    backtest_total_return: float = 0.0
    backtest_max_drawdown: float = 0.0
    backtest_sharpe_ratio: float = 0.0
    backtest_total_trades: int = 0
    backtest_period: str = ""
    backtest_strategy_direction: str = ""   # LONG, SHORT, RANGE
    backtest_detected_trend: str = ""       # uptrend, downtrend, sideways
    backtest_token_tier: str = ""           # major, mid, micro
    backtest_strategies_tested: List[str] = []


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
    # Visual indicators (populated by UI layer)
    confidence_bar: str = ""
    risk_badge_text: str = ""
    verdict_emoji: str = ""
    verdict_color: str = ""


class AIRecommendations(BaseModel):
    market_summary: str = ""
    tokens: List[AITokenScore] = []
    timestamp: str = ""


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
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated — no token provided. Please log in.")
    token = authorization.split(" ", 1)[1]
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(401, "Not authenticated — token expired or invalid. Please log in again.")
    user_id = int(payload.get("sub", 0))
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(401, "Not authenticated — user not found. Please register again.")
    return user


# ══════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════

_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://localhost:8080,https://nexypher.vercel.app,https://www.nexypher.vercel.app"
).split(",")
# Always include wildcard patterns for Vercel preview deploys
_cors_origins.extend([
    "https://nexypher.vercel.app",
    "https://nexypher-api.onrender.com",
])

app = FastAPI(title="NEXYPHER", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "web" / "static"
try:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # Read-only filesystem on Vercel


_initialized = False
_init_lock = threading.Lock()

def _ensure_initialized():
    """Lazy initialization — works both with startup event and on first request (Vercel)."""
    global _initialized, cg, dex, news, ta, gemini_client
    if _initialized:
        return
    with _init_lock:
        if _initialized:  # Double-check after acquiring lock
            return
        _initialized = True

    cg = CoinGeckoCollector(api_key=os.getenv("COINGECKO_API_KEY", ""))
    dex = DexScreenerCollector(apify_api_key=os.getenv("APIFY_API_KEY", ""))
    news = NewsCollector(
        api_key=os.getenv("CRYPTOPANIC_API_KEY", ""),
        news_api_key=os.getenv("NEWS_API_KEY", ""),
    )
    ta = TechnicalAnalyzer()

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from src.ai_engine.gemini_client import GeminiClient
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            gemini_client = GeminiClient(api_key=gemini_key, model=gemini_model)
            logger.info("Gemini AI client initialized (model=%s)", gemini_model)
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)

    auth.init_db()
    trading_engine.init_trading_tables()
    logger.info("NEXYPHER v2 initialized")


@app.on_event("startup")
async def startup():
    _ensure_initialized()

    # Start always-on auto-trade background loop (skip on Vercel serverless)
    if not os.getenv("VERCEL"):
        trading_engine.start_autotrader(cg, dex, gemini_client)
        logger.info("Always-on auto-trader started (system user, ₹1Cr balance)")

        # Start learning evaluation loop (evaluates predictions for ALL users)
        import asyncio as _aio
        _aio.create_task(_learning_evaluation_loop())
        logger.info("Learning evaluation background loop started")
    else:
        logger.info("Running on Vercel (auto-trader disabled in serverless mode)")


@app.get("/api/autotrader/status")
async def autotrader_status():
    """Get the always-on auto-trader status, balance, and performance."""
    return trading_engine.get_autotrader_status()



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


@app.get("/algo")
async def algo_trader():
    """Redirect /algo to main app — algo features merged into Auto Trader page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/", status_code=302)


@app.get("/static/{filepath:path}")
async def static_files(filepath: str):
    file_path = (STATIC_DIR / filepath).resolve()
    # Prevent path traversal — ensure resolved path stays inside STATIC_DIR
    if not str(file_path).startswith(str(STATIC_DIR.resolve())):
        raise HTTPException(403, "Forbidden")
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(404, "File not found")


# ══════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/api/auth/register")
async def api_register(body: auth.UserRegister):
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if not _re.search(r'[A-Z]', body.password):
        raise HTTPException(400, "Password must contain at least one uppercase letter")
    if not _re.search(r'[0-9]', body.password):
        raise HTTPException(400, "Password must contain at least one number")
    user = auth.register_user(body.email, body.username, body.password)
    if not user:
        raise HTTPException(409, "Email or username already taken")
    # Send welcome email (without password — never email passwords)
    try:
        smtp_service.send_welcome_email(body.email, body.username)
    except Exception as e:
        logger.warning("Failed to send registration email: %s", e)
    token = auth.create_access_token(user.id, user.email)
    return {"access_token": token, "token_type": "bearer", "user": user.model_dump()}


@app.post("/api/auth/login")
async def api_login(body: auth.UserLogin, request: Request):
    # Check account lockout
    client_ip = request.headers.get(
        "x-forwarded-for",
        request.client.host if request.client else "unknown",
    )
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()
    identifier = f"{client_ip}:{body.email.lower()}"
    is_locked, remaining = login_tracker.is_locked(identifier)
    if is_locked:
        raise HTTPException(
            429,
            f"Account locked due to too many failed attempts. Try again in {remaining} seconds.",
        )

    user = auth.authenticate_user(body.email, body.password)
    if not user:
        now_locked, attempts_left = login_tracker.record_failure(identifier)
        if now_locked:
            raise HTTPException(
                429,
                "Account locked for 15 minutes due to too many failed login attempts.",
            )
        raise HTTPException(
            401,
            f"Invalid email or password. {attempts_left} attempt(s) remaining.",
        )
    login_tracker.record_success(identifier)
    token = auth.create_access_token(user.id, user.email)
    # Send login alert email in background (non-blocking)
    try:
        ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "Unknown")
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
        safe_error = _html.escape(error)
        html = f"""
        <html><head><meta charset="utf-8"><title>NEXYPHER</title></head>
        <body style="background:#0a0a0f;color:#e0e0e0;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;">
            <div style="text-align:center;">
                <h1 style="color:#ff4d4d;">Verification Failed</h1>
                <p>{safe_error}</p>
                <a href="/" style="color:#7c5cff;">Back to NEXYPHER</a>
            </div>
        </body></html>"""
    else:
        html = """
        <html><head><meta charset="utf-8"><title>NEXYPHER</title></head>
        <body style="background:#0a0a0f;color:#e0e0e0;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;">
            <div style="text-align:center;">
                <h1 style="color:#00d4aa;">\u2705 Email Verified!</h1>
                <p>Your account is now fully active.</p>
                <a href="/" style="color:#7c5cff;">Go to NEXYPHER</a>
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
    sb = get_supabase()
    try:
        resp = sb.table("users").select("id, email_verified").eq("email", body.email.lower()).execute()
        if not resp.data:
            return {"message": "If an account with that email exists, a verification link has been sent."}
        row = resp.data[0]
        if row["email_verified"]:
            return {"message": "Email is already verified. You can log in."}
        auth.resend_verification(row["id"])
    except Exception:
        pass
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
    <html><head><meta charset="utf-8"><title>NEXYPHER — Reset Password</title></head>
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


class ContactRequest(BaseModel):
    name: str
    email: str
    subject: str = "General Inquiry"
    message: str


@app.post("/api/contact")
async def submit_contact(body: ContactRequest):
    """Receive contact form submissions and forward via SMTP."""
    admin_email = os.getenv("SMTP_EMAIL", "rajkumar648321@gmail.com")
    ok = False
    if smtp_service.is_configured():
        try:
            ok = smtp_service.send_contact_email(
                from_name=body.name,
                from_email=body.email,
                subject=body.subject,
                message=body.message,
                to_email=admin_email,
            )
        except Exception as e:
            logger.error("SMTP send_contact_email exception: %s", e)
            ok = False

    if not ok:
        # SMTP failed or not configured — log the message so it's not lost,
        # and still return success to the user
        logger.info("Contact form (saved): name=%s email=%s subject=%s message=%s",
                     body.name, body.email, body.subject, body.message[:200])
        # Also save to Supabase so messages are never lost
        try:
            from supabase_db import get_supabase
            get_supabase().table("contact_messages").insert({
                "name": body.name,
                "email": body.email,
                "subject": body.subject,
                "message": body.message,
            }).execute()
        except Exception:
            pass  # Table may not exist yet — that's fine, message is logged

    return {"status": "ok"}


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
    """Add a bank account after verification."""
    result = auth.add_bank_account(user.id, body.account_holder, body.account_number, body.ifsc_code, body.bank_name)
    if not result["success"]:
        raise HTTPException(400, detail=result["errors"][0])
    updated = auth.get_user_by_id(user.id)
    return {"success": True, "bank_name": result["bank_name"], "last4": result["last4"], "bank_accounts": updated.bank_accounts}


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
    result = auth.deposit_to_wallet(user.id, body.amount, body.bank_id)
    if not result["success"]:
        raise HTTPException(400, detail=result["error"])
    return result


@app.post("/api/wallet/withdraw")
async def api_wallet_withdraw(body: auth.DepositRequest, user=Depends(require_user)):
    result = auth.withdraw_from_wallet(user.id, body.amount, body.bank_id)
    if not result["success"]:
        raise HTTPException(400, detail=result["error"])
    return result


@app.get("/api/wallet/transactions")
async def api_wallet_transactions(limit: int = Query(20), user=Depends(require_user)):
    return auth.get_wallet_transactions(user.id, limit)


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


@app.get("/api/watchlist")
async def api_get_watchlist(user=Depends(require_user)):
    """
    Get full watchlist with live prices, visual indicators, and
    triggered alert evaluation via the WatchlistManager.
    """
    coin_ids = auth.get_watchlist(user.id)
    if not coin_ids:
        return {"watchlist": [], "alerts": []}

    # Fetch live prices for all watchlist coins
    prices = await cg.get_simple_price(coin_ids)

    enriched = []
    for cid in coin_ids:
        cp = prices.get(cid, 0)
        enriched.append({
            "coin_id": cid,
            "current_price": cp,
            "price_display": f"${cp:,.8g}" if cp else "N/A",
            "trend": trend_arrow(0),  # no 24h change available in simple price
            "freshness": data_freshness_indicator(0),
        })

    # Build a minimal UserPreferences with the watchlist for alert eval
    try:
        from src.ui.user_config import WatchlistItem, AlertType
        prefs = default_preferences(user_id=user.id)
        # Populate watchlist from DB coins (basic entries without alert prices)
        prefs.watchlist = [WatchlistItem(token=cid.upper()) for cid in coin_ids]
        wm = WatchlistManager(prefs)
        price_map = {cid.upper(): prices.get(cid, 0) for cid in coin_ids}
        triggered = wm.evaluate_alerts(price_map)
        alert_list = [
            {
                "token": a.token,
                "alert_type": a.alert_type.value,
                "alert_price": a.alert_price,
                "current_price": a.current_price,
                "message": a.message,
                "triggered_at": a.triggered_at.isoformat(),
            }
            for a in triggered
        ]
    except Exception as e:
        logger.warning("Watchlist alert eval failed: %s", e)
        alert_list = []

    return {"watchlist": enriched, "alerts": alert_list}


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/portfolio")
async def api_get_portfolio(user=Depends(require_user)):
    """
    Get portfolio with live prices, P&L, and AI status annotations
    via the PortfolioTracker.
    """
    items = auth.get_portfolio(user.id)
    if items:
        # Build PortfolioHolding objects from DB rows
        holdings = [
            PortfolioHolding(
                token=i["coin_id"].upper(),
                entry_price=i["avg_buy_price"],
                quantity=i["amount"],
                notes=i.get("notes", ""),
            )
            for i in items
        ]
        prefs = default_preferences(user_id=str(user.id))
        prefs.holdings = holdings
        tracker = PortfolioTracker(prefs)

        # Fetch live prices
        coin_ids = [i["coin_id"] for i in items]
        raw_prices = await cg.get_simple_price(coin_ids)
        # PortfolioTracker expects UPPER-case keys
        current_prices = {k.upper(): v for k, v in raw_prices.items()}

        summary = tracker.get_summary(current_prices)

        # Enrich each position with visual indicators
        enriched = []
        for pos in summary.positions:
            enriched.append({
                **PortfolioItem(
                    coin_id=pos.token,
                    amount=pos.quantity,
                    avg_buy_price=pos.entry_price,
                    notes=pos.notes,
                    current_price=pos.current_price,
                    pnl=pos.pnl_dollar,
                    pnl_pct=pos.pnl_percent,
                ).model_dump(),
                "ai_status": pos.ai_status,
                "ai_status_emoji": pos.ai_status_emoji,
                "ai_comment": pos.ai_comment,
                "risk_badge": risk_badge("HIGH" if pos.pnl_percent < -10 else "MEDIUM" if pos.pnl_percent < 0 else "LOW"),
                "trend": trend_arrow(pos.pnl_percent),
            })

        return {
            "positions": enriched,
            "summary": {
                "total_invested": summary.total_invested,
                "total_value": summary.total_current_value,
                "total_pnl": summary.total_pnl_dollar,
                "total_pnl_pct": summary.total_pnl_percent,
                "winning": summary.winning_count,
                "losing": summary.losing_count,
                "overall_trend": trend_arrow(summary.total_pnl_percent),
            },
        }
    return {"positions": [], "summary": None}


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
# NOTIFICATION CHECK ENDPOINT
# ══════════════════════════════════════════════════════════════════

@app.get("/api/notifications/check")
async def api_check_notifications(user=Depends(require_user)):
    """
    Check for triggered alerts on the user's portfolio and watchlist.
    Returns formatted notifications (push/email/SMS ready) via
    the NotificationFormatter.
    """
    notifications = []

    # ── Portfolio P&L notifications ──
    try:
        items = auth.get_portfolio(user.id)
        if items:
            coin_ids = [i["coin_id"] for i in items]
            prices = await cg.get_simple_price(coin_ids)
            total_cost = sum(i["amount"] * i["avg_buy_price"] for i in items)
            total_value = sum(i["amount"] * prices.get(i["coin_id"], 0) for i in items)
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

            # Portfolio-level notifications
            if abs(total_pnl_pct) >= 5:
                positions_data = [
                    {"token": i["coin_id"].upper(), "pnl_pct": round(
                        ((prices.get(i["coin_id"], 0) * i["amount"] - i["amount"] * i["avg_buy_price"])
                        / max(i["amount"] * i["avg_buy_price"], 0.01)) * 100, 1
                    )}
                    for i in items
                ]
                notif = _notification_formatter.portfolio_update(
                    total_pnl=round(total_pnl, 2),
                    total_pnl_pct=round(total_pnl_pct, 2),
                    positions=positions_data,
                )
                notifications.append({
                    "type": notif.ntype.value,
                    "priority": notif.priority.value,
                    "title": notif.title,
                    "body": notif.body,
                    "short_body": notif.short_body,
                })

            # Per-position notifications for big movers
            for item in items:
                cp = prices.get(item["coin_id"], 0)
                if cp <= 0 or item["avg_buy_price"] <= 0:
                    continue
                pnl_pct = ((cp - item["avg_buy_price"]) / item["avg_buy_price"]) * 100
                if pnl_pct >= 20:
                    notif = _notification_formatter.price_alert(
                        token=item["coin_id"].upper(),
                        current_price=cp,
                        target_price=cp,
                        entry_price=item["avg_buy_price"],
                        alert_type="target_reached",
                    )
                    notifications.append({
                        "type": notif.ntype.value,
                        "priority": notif.priority.value,
                        "title": notif.title,
                        "body": notif.body,
                        "short_body": notif.short_body,
                    })
                elif pnl_pct <= -15:
                    notif = _notification_formatter.risk_warning(
                        token=item["coin_id"].upper(),
                        warning_type="price_drop",
                        details=f"Down {pnl_pct:.1f}% from entry ${item['avg_buy_price']:.8g}",
                    )
                    notifications.append({
                        "type": notif.ntype.value,
                        "priority": notif.priority.value,
                        "title": notif.title,
                        "body": notif.body,
                        "short_body": notif.short_body,
                    })
    except Exception as e:
        logger.warning("Notification check (portfolio) failed: %s", e)

    return {
        "notifications": notifications,
        "count": len(notifications),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════
# MARKET ENDPOINTS
# ══════════════════════════════════════════════════════════════════

STABLECOINS_MARKET = {"tether", "usd-coin", "dai", "usd1-wlfi", "binance-usd", "true-usd", "first-digital-usd", "staked-ether", "wrapped-bitcoin", "usds", "ethena-usde", "usdd", "paypal-usd", "frax", "usdb"}

def _score_coin(coin, trending_ids: set) -> TokenCard:
    """Score a single coin using NexYpher AI signal logic."""
    change_24h = coin.price_change_pct_24h
    change_7d = coin.price_change_pct_7d
    mcap = coin.market_cap
    vol = coin.total_volume_24h
    vol_ratio = vol / mcap if mcap > 0 else 0
    ath_dist = ((coin.ath - coin.current_price) / coin.ath * 100) if coin.ath > 0 else 0
    is_trending = coin.coin_id in trending_ids

    # ── LONG score ──
    long_score = 0; long_reasons = []
    # Momentum (24h)
    if change_24h > 10:
        long_score += 25; long_reasons.append(f"🚀 Explosive +{change_24h:.1f}%")
    elif change_24h > 3:
        long_score += 18; long_reasons.append(f"📈 Strong +{change_24h:.1f}%")
    elif change_24h > 0.5:
        long_score += 10; long_reasons.append(f"↗ Positive +{change_24h:.1f}%")
    # Weekly trend
    if change_7d > 15:
        long_score += 15; long_reasons.append(f"🔥 Week +{change_7d:.1f}%")
    elif change_7d > 5:
        long_score += 8; long_reasons.append(f"📊 Week +{change_7d:.1f}%")
    # Volume spike
    if vol_ratio > 0.3:
        long_score += 18; long_reasons.append(f"⚡ Volume spike {vol_ratio:.2f}x")
    elif vol_ratio > 0.15:
        long_score += 10; long_reasons.append(f"📢 High volume")
    elif vol_ratio > 0.08:
        long_score += 5; long_reasons.append(f"Volume normal")
    # Trending
    if is_trending:
        long_score += 12; long_reasons.append("🔥 Trending")
    # ATH recovery zone
    if 40 < ath_dist < 80:
        long_score += 10; long_reasons.append(f"💎 {ath_dist:.0f}% below ATH — recovery play")

    # ── SHORT score ──
    short_score = 0; short_reasons = []
    if change_24h < -8:
        short_score += 25; short_reasons.append(f"🔻 Dumping {change_24h:+.1f}%")
    elif change_24h < -3:
        short_score += 18; short_reasons.append(f"📉 Bearish {change_24h:+.1f}%")
    elif change_24h < -1:
        short_score += 10; short_reasons.append(f"↘ Declining {change_24h:+.1f}%")
    if change_7d < -15:
        short_score += 15; short_reasons.append(f"💀 Week {change_7d:+.1f}%")
    elif change_7d < -5:
        short_score += 8; short_reasons.append(f"📉 Week {change_7d:+.1f}%")
    if vol_ratio > 0.3 and change_24h < -2:
        short_score += 12; short_reasons.append("⚡ Panic selling")
    if ath_dist < 8:
        short_score += 12; short_reasons.append(f"⚠️ Near ATH — pullback likely")

    # Pick dominant signal
    if long_score >= short_score and long_score >= 15:
        signal = "long"; score = min(100, long_score); reasons = long_reasons
    elif short_score > long_score and short_score >= 15:
        signal = "short"; score = min(100, short_score); reasons = short_reasons
    else:
        signal = "neutral"; score = max(long_score, short_score); reasons = ["Sideways — no clear signal"]

    # Verdict
    if signal == "long":
        if score >= 65: verdict = "STRONG BUY"
        elif score >= 45: verdict = "BUY"
        else: verdict = "HOLD"
    elif signal == "short":
        if score >= 65: verdict = "STRONG SELL"
        elif score >= 45: verdict = "SELL"
        else: verdict = "CAUTION"
    else:
        verdict = "NEUTRAL"

    return TokenCard(
        name=coin.name,
        symbol=coin.symbol.upper(),
        coin_id=coin.coin_id,
        price=coin.current_price,
        price_change_24h=coin.price_change_pct_24h,
        price_change_7d=coin.price_change_pct_7d,
        market_cap=mcap,
        volume_24h=vol,
        rank=coin.market_cap_rank,
        image=getattr(coin, "image", None),
        ai_score=score,
        ai_verdict=verdict,
        ai_signal=signal,
        ai_reasons=reasons,
        vol_mcap_ratio=round(vol_ratio, 4),
        ath_distance_pct=round(ath_dist, 1),
        trending=is_trending,
    )


@app.get("/api/market/top", response_model=List[TokenCard])
async def get_top_coins(limit: int = Query(50, ge=1, le=250)):
    """Top coins scored by NexYpher AI — ranked by opportunity, not market cap."""
    coins = await cg.get_top_coins(limit=min(limit + 10, 250))  # fetch extra to filter stables
    trending = await cg.get_trending()
    trending_ids = {t.coin_id for t in trending} if trending else set()

    scored = []
    for c in coins:
        if c.coin_id in STABLECOINS_MARKET:
            continue
        if c.current_price <= 0:
            continue
        scored.append(_score_coin(c, trending_ids))

    # Sort by AI score descending (opportunity ranking)
    scored.sort(key=lambda t: t.ai_score, reverse=True)
    return scored[:limit]


@app.get("/api/market/trending", response_model=List[TrendingToken])
async def get_trending():
    trending = await cg.get_trending()
    return [
        TrendingToken(name=t.name, symbol=t.symbol, coin_id=t.coin_id, rank=t.score)
        for t in trending
    ]


# ══════════════════════════════════════════════════════════════════
# TOKEN FEED (DexScreener live feed)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/feed/tokens", response_model=List[DexToken])
async def get_token_feed(
    sort: str = Query("volume"),
    status: str = Query("all"),
    limit: int = Query(50, ge=1, le=100),
):
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

    return all_pairs[:limit]


# ══════════════════════════════════════════════════════════════════
# TOKEN DETAIL
# ══════════════════════════════════════════════════════════════════

# Common ticker / symbol → CoinGecko ID mapping
_COIN_ALIASES: dict = {
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
    "xrp": "ripple", "doge": "dogecoin", "ada": "cardano",
    "avax": "avalanche-2", "link": "chainlink", "dot": "polkadot",
    "matic": "matic-network", "shib": "shiba-inu", "uni": "uniswap",
    "atom": "cosmos", "ltc": "litecoin", "near": "near",
    "bnb": "binancecoin", "trx": "tron", "xlm": "stellar",
    "icp": "internet-computer", "apt": "aptos", "sui": "sui",
    "arb": "arbitrum", "op": "optimism", "pepe": "pepe",
    "bonk": "bonk", "wif": "dogwifcoin", "render": "render-token",
    "fet": "fetch-ai", "inj": "injective-protocol",
}


@app.get("/api/token/{coin_id}", response_model=TokenDetail)
async def get_token_detail(coin_id: str):
    _ensure_initialized()
    coin_id_lower = coin_id.lower().strip()
    # Resolve common aliases (btc→bitcoin, sol→solana, etc.)
    coin_id_lower = _COIN_ALIASES.get(coin_id_lower, coin_id_lower)
    coin = await cg.get_coin_detail(coin_id_lower)
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
                f"You are NEXYPHER, an expert crypto analyst AI. Analyze {coin.name} ({coin.symbol.upper()}):\n"
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
                    "You are NEXYPHER, an expert cryptocurrency analyst.",
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

    # ── Run backtest verification (mandatory) ──
    bt_engine = get_backtest_engine()
    bt_result = None
    try:
        bt_result = await bt_engine.run_backtest(
            coin_id=coin.coin_id,
            coin_name=coin.name,
            symbol=coin.symbol,
            cg_collector=cg,
            days=180,
            market_cap=coin.market_cap or 0,
        )
    except Exception as e:
        logger.warning("Backtest failed for %s: %s", coin.coin_id, e)

    # If backtest fails thresholds, override AI recommendation with warning
    if bt_result and not bt_result.passed_all_thresholds:
        strategies_str = ", ".join(bt_result.strategies_tested) if bt_result.strategies_tested else "LONG"
        bt_warning = (
            f"⚠️ BACKTEST WARNING: All strategies ({strategies_str}) failed for this token. "
            f"Detected trend: {bt_result.detected_trend.upper()}. "
            f"Best result ({bt_result.strategy_direction}): "
            f"{bt_result.total_trades} trades | Win Rate: {bt_result.win_rate:.1f}% | "
            f"Return: {bt_result.total_return:.1f}% | Max DD: {bt_result.max_drawdown:.1f}% | "
            f"Sharpe: {bt_result.sharpe_ratio:.2f}. "
            f"Tier: {bt_result.token_tier}."
        )
        if ai_text:
            ai_text = bt_warning + "\n\nOriginal AI Analysis: " + ai_text
        else:
            ai_text = bt_warning
        ai_avail = True
    elif bt_result and bt_result.passed_all_thresholds:
        bt_prefix = (
            f"✅ BACKTEST VERIFIED — {bt_result.strategy_direction} Strategy ({bt_result.recommendation}): "
            f"Trend: {bt_result.detected_trend.upper()} | "
            f"Win Rate {bt_result.win_rate:.1f}% | Return {bt_result.total_return:.1f}% | "
            f"Max DD {bt_result.max_drawdown:.1f}% | Sharpe {bt_result.sharpe_ratio:.2f} | "
            f"{bt_result.total_trades} trades over {bt_result.days_covered} days | "
            f"Tier: {bt_result.token_tier}.\n\n"
        )
        if ai_text:
            ai_text = bt_prefix + ai_text
        else:
            # Use only recommendation_detail (already contains full analysis)
            ai_text = bt_result.recommendation_detail
        ai_avail = True

    # ── Fallback: generate market-based sentiment if news APIs returned nothing ──
    if news_result.total_articles == 0:
        pct_24h = coin.price_change_pct_24h or 0
        pct_7d = coin.price_change_pct_7d or 0
        sym = coin.symbol.upper()

        # Derive sentiment from price action
        if pct_24h > 5:
            fallback_sentiment = min(0.7, pct_24h / 15)
            fallback_narrative = f"{sym} is showing strong bullish momentum with {pct_24h:+.1f}% gain in the last 24 hours."
        elif pct_24h > 1:
            fallback_sentiment = min(0.4, pct_24h / 10)
            fallback_narrative = f"{sym} is trending mildly bullish with a {pct_24h:+.1f}% move over 24 hours."
        elif pct_24h < -5:
            fallback_sentiment = max(-0.7, pct_24h / 15)
            fallback_narrative = f"{sym} is experiencing bearish pressure with {pct_24h:+.1f}% decline in the last 24 hours."
        elif pct_24h < -1:
            fallback_sentiment = max(-0.4, pct_24h / 10)
            fallback_narrative = f"{sym} is trending mildly bearish with a {pct_24h:+.1f}% move over 24 hours."
        else:
            fallback_sentiment = 0.0
            fallback_narrative = f"{sym} is trading in a neutral range with minimal price movement ({pct_24h:+.1f}% in 24h)."

        fallback_headlines = []
        if abs(pct_24h) > 2:
            direction = "surges" if pct_24h > 0 else "drops"
            fallback_headlines.append(f"{sym} {direction} {abs(pct_24h):.1f}% in 24-hour trading session")
        if abs(pct_7d) > 5:
            direction = "gains" if pct_7d > 0 else "loses"
            fallback_headlines.append(f"{sym} {direction} {abs(pct_7d):.1f}% over the past week")
        if coin.market_cap and coin.market_cap > 1_000_000_000:
            fallback_headlines.append(f"{sym} maintains ${coin.market_cap/1e9:.1f}B market capitalization")
        if ta_result:
            fallback_headlines.append(f"Technical outlook: {ta_result.trend.capitalize()} — RSI at {ta_result.rsi:.0f}")

        news_result.avg_sentiment = round(fallback_sentiment, 3)
        news_result.narrative = fallback_narrative
        news_result.key_headlines = fallback_headlines
        logger.info("Using market-data fallback for %s news sentiment", sym)

    return TokenDetail(
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
        # Backtest verification data
        backtest_verified=bt_result.passed_all_thresholds if bt_result else False,
        backtest_recommendation=bt_result.recommendation if bt_result else "",
        backtest_win_rate=round(bt_result.win_rate, 2) if bt_result else 0,
        backtest_total_return=round(bt_result.total_return, 2) if bt_result else 0,
        backtest_max_drawdown=round(bt_result.max_drawdown, 2) if bt_result else 0,
        backtest_sharpe_ratio=round(bt_result.sharpe_ratio, 2) if bt_result else 0,
        backtest_total_trades=bt_result.total_trades if bt_result else 0,
        backtest_period=f"{bt_result.start_date} to {bt_result.end_date}" if bt_result else "",
        backtest_strategy_direction=bt_result.strategy_direction if bt_result else "",
        backtest_detected_trend=bt_result.detected_trend if bt_result else "",
        backtest_token_tier=bt_result.token_tier if bt_result else "",
        backtest_strategies_tested=bt_result.strategies_tested if bt_result else [],
    )


# ══════════════════════════════════════════════════════════════════
# AI RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/ai/recommendations", response_model=AIRecommendations)
async def get_ai_recommendations(
    enable_onchain: bool = Query(True),
    enable_technical: bool = Query(True),
):
    pairs = await dex.search_pairs("SOL")
    tokens_scored: List[AITokenScore] = []
    token_prices: Dict[str, float] = {}  # address → price_usd for learning loop
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
            confidence_bar=confidence_bar(total_score / 10),
            risk_badge_text=risk_badge("HIGH" if flags else ("MEDIUM" if total_score < 50 else "LOW")),
            verdict_emoji=verdict_emoji(verdict.title()),
            verdict_color=verdict_colour(verdict.title()),
        ))
        token_prices[p.base_token_address] = p.price_usd

    tokens_scored.sort(key=lambda x: x.score, reverse=True)

    # ── Record predictions in the learning loop ───────────────────
    # NOTE: DexScreener tokens use symbol as ticker (e.g. "PEPE") which
    # won't resolve via CoinGecko price lookup.  We also store the
    # price at prediction time so the evaluation path has a fallback.
    ll = _get_ll()
    if ll:
        for t in tokens_scored:
            try:
                price = token_prices.get(t.address, 0.0)
                if price <= 0:
                    continue  # skip — can't evaluate without entry price
                ll.record_prediction(
                    token_ticker=t.symbol.upper(),
                    token_name=t.name,
                    verdict=t.verdict,
                    confidence=t.score / 100.0,
                    composite_score=float(t.score),
                    price_at_prediction=price,
                    risk_level="HIGH" if t.risk_flags else "MEDIUM",
                    ai_thought_summary=t.summary,
                )
            except Exception:
                pass

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
                "You are NEXYPHER market analyst. Summarize this DexScreener market in 2-3 sentences:\n"
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

    return AIRecommendations(
        market_summary=market_summary,
        tokens=tokens_scored[:15],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════════
# BACKTEST VERIFICATION ENDPOINT
# ══════════════════════════════════════════════════════════════════

@app.get("/api/backtest/batch")
async def batch_backtest(
    coins: str = Query(
        ...,
        description="Comma-separated CoinGecko IDs (e.g. bitcoin,solana,ethereum)",
    ),
    days: int = Query(180, ge=30, le=365),
):
    """
    Run backtests for multiple tokens at once.
    Returns results for each token with pass/fail status.
    """
    _ensure_initialized()

    coin_ids = [c.strip().lower() for c in coins.split(",") if c.strip()]
    if not coin_ids:
        raise HTTPException(400, "No coin IDs provided")
    if len(coin_ids) > 10:
        raise HTTPException(400, "Maximum 10 coins per batch request")

    # Resolve aliases
    coin_ids = [_COIN_ALIASES.get(c, c) for c in coin_ids]

    bt_engine = get_backtest_engine()
    results = []

    for coin_id in coin_ids:
        try:
            coin = await cg.get_coin_detail(coin_id)
            if not coin:
                results.append({
                    "coin_id": coin_id,
                    "error": f"Token '{coin_id}' not found",
                    "passed": False,
                })
                continue

            bt_result = await bt_engine.run_backtest(
                coin_id=coin.coin_id,
                coin_name=coin.name,
                symbol=coin.symbol,
                cg_collector=cg,
                days=max(days, 180),
                market_cap=coin.market_cap or 0,
            )

            results.append({
                **bt_result.to_dict(),
                "current_price": coin.current_price,
            })
        except Exception as e:
            results.append({
                "coin_id": coin_id,
                "error": str(e),
                "passed": False,
            })

    passed = len([r for r in results if r.get("thresholds", {}).get("passed_all")])
    failed = len(results) - passed

    return {
        "results": results,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/backtest/{coin_id}")
async def run_backtest(
    coin_id: str,
    days: int = Query(180, ge=30, le=365, description="Days of history to backtest"),
):
    """
    Run a full backtest on a token and return results with recommendation.

    The backtest pipeline:
    1. Fetches historical OHLCV data from CoinGecko (minimum 6 months)
    2. Runs a simulated strategy using RSI, MACD, and Bollinger Bands
    3. Includes 0.1% trading fees per trade
    4. Evaluates against profitability thresholds:
       - Win Rate > 55%
       - Max Drawdown < 20%
       - Total Return > 0%
       - Minimum 10 trades
    5. Only generates a BUY/SELL recommendation if ALL thresholds pass

    Returns full backtest statistics alongside the recommendation.
    """
    _ensure_initialized()

    # Resolve aliases
    coin_id_lower = coin_id.lower().strip()
    coin_id_lower = _COIN_ALIASES.get(coin_id_lower, coin_id_lower)

    # Get coin info
    coin = await cg.get_coin_detail(coin_id_lower)
    if not coin:
        raise HTTPException(404, f"Token '{coin_id}' not found on CoinGecko")

    bt_engine = get_backtest_engine()
    result = await bt_engine.run_backtest(
        coin_id=coin.coin_id,
        coin_name=coin.name,
        symbol=coin.symbol,
        cg_collector=cg,
        days=max(days, 180),
        market_cap=coin.market_cap or 0,
    )

    return {
        **result.to_dict(),
        "current_price": coin.current_price,
        "market_cap": coin.market_cap,
        "volume_24h": coin.total_volume_24h,
        "price_change_24h": coin.price_change_pct_24h,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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
        "ai_available": gemini_client is not None,
        "version": "2.0.0",
        "blockchain": blockchain.get_status(),
        "collectors": {
            "coingecko": cg is not None,
            "dexscreener": dex is not None,
            "news": news is not None,
            "technical": ta is not None,
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
    sb = get_supabase()
    resp = sb.table("trade_orders").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(len(actions) + 2).execute()
    recent_orders = resp.data or []

    for order in recent_orders[:len(actions)]:
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
            pos_resp = sb.table("trade_positions").select("*").eq("id", order["position_id"]).execute()
            if pos_resp.data:
                pos = pos_resp.data[0]
                coin_name = pos.get("coin_name", symbol)
                stop_loss = pos.get("stop_loss", 0)
                take_profit = pos.get("take_profit", 0)
                if action == "SELL":
                    pnl = pos.get("pnl", 0)
                    pnl_pct = pos.get("pnl_pct", 0)

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

# NOTE: The primary auto-trade loop is `continuous_trading_loop` in
# trading_engine.py, started via `start_autotrader()` at startup.
# This secondary loop handles learning-loop evaluation for ALL users
# and email notifications — the trading engine only evaluates for SYSTEM_USER_ID.

async def _learning_evaluation_loop():
    """Background loop — evaluates pending predictions every 5 minutes for all users."""
    await asyncio.sleep(30)  # Wait for some predictions to accumulate
    logger.info("Learning evaluation background loop started (5-min interval)")
    while True:
        try:
            ll = _get_ll()
            if ll and cg:
                eval_result = await ll.evaluate_pending(cg)
                eval_24h = eval_result.get("evaluated_24h", 0) if isinstance(eval_result, dict) else 0
                eval_7d = eval_result.get("evaluated_7d", 0) if isinstance(eval_result, dict) else 0
                if eval_24h or eval_7d:
                    logger.info(
                        "Learning eval loop: evaluated %d (24h) + %d (7d) predictions",
                        eval_24h, eval_7d,
                    )
        except Exception as e:
            logger.warning("Learning evaluation loop error: %s", e)

        await asyncio.sleep(300)  # Every 5 minutes


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


@app.get("/api/trader/heatmap")
async def get_trader_heatmap(year: int = 2026, user=Depends(require_user)):
    """Return daily P&L aggregated data for the heatmap."""
    closed = trading_engine.get_closed_positions(user.id, limit=5000)
    daily: dict = {}
    for p in closed:
        if p.get("pnl") is None:
            continue
        # Parse date from closed_at (ISO string or datetime)
        closed_at = p.get("closed_at") or p.get("updated_at") or p.get("created_at")
        if not closed_at:
            continue
        try:
            dt_str = str(closed_at)[:10]  # "YYYY-MM-DD"
            d_year = int(dt_str[:4])
        except Exception:
            continue
        if d_year != year:
            continue
        if dt_str not in daily:
            daily[dt_str] = {"pnl": 0.0, "trades": 0, "wins": 0}
        daily[dt_str]["pnl"] += float(p["pnl"])
        daily[dt_str]["trades"] += 1
        if float(p["pnl"]) >= 0:
            daily[dt_str]["wins"] += 1
    result = []
    for date_str, v in daily.items():
        result.append({
            "date": date_str,
            "pnl": round(v["pnl"], 2),
            "trades": v["trades"],
            "win_rate": round(v["wins"] / v["trades"] * 100) if v["trades"] else 0,
        })
    return result


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
    opportunities = await trading_engine.research_opportunities(cg, dex, gemini_client)
    return {"opportunities": opportunities[:15], "total": len(opportunities)}


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


@app.get("/api/trader/today-stats")
async def get_today_stats():
    """Today's quick stats for the system auto-trader (user_id=1)."""
    return trading_engine.get_today_quick_stats(trading_engine.SYSTEM_USER_ID)


@app.get("/api/trader/pnl-chart")
async def get_pnl_chart(days: int = Query(14)):
    """Daily P&L chart data for the last N days (system auto-trader)."""
    return trading_engine.get_pnl_chart_data(trading_engine.SYSTEM_USER_ID, min(days, 90))


@app.get("/api/trader/live-feed")
async def get_live_feed(limit: int = Query(20)):
    """Most recent trade actions for the live feed (system auto-trader)."""
    return trading_engine.get_live_feed(trading_engine.SYSTEM_USER_ID, min(limit, 50))


@app.get("/api/trader/cycle-log")
async def get_cycle_log():
    """Last N auto-trade cycle summaries (in-memory)."""
    return trading_engine.get_cycle_log()


@app.get("/api/trader/pnl-heatmap")
async def get_pnl_heatmap(days: int = Query(30)):
    """P&L heatmap data aggregated by coin (system auto-trader)."""
    return trading_engine.get_pnl_heatmap(trading_engine.SYSTEM_USER_ID, min(days, 90))


# ══════════════════════════════════════════════════════════════════
# TRANSACTION HASH VERIFICATION
# ══════════════════════════════════════════════════════════════════

@app.get("/api/verify-tx/{tx_hash}")
async def verify_transaction(tx_hash: str, user=Depends(require_user)):
    """Look up a transaction hash in trade_orders and wallet_transactions."""
    result = {"tx_hash": tx_hash, "found": False, "source": None, "transaction": None, "verified": False, "on_chain": None}

    sb = get_supabase()

    # Search trade_orders
    trade_resp = sb.table("trade_orders").select("*").eq("tx_hash", tx_hash).execute()
    if trade_resp.data:
        trade = trade_resp.data[0]
        verified = trading_engine.verify_tx_hash(
            tx_hash, trade["user_id"], trade["action"], trade["coin_id"],
            trade["symbol"], trade["price"], trade["quantity"],
            trade["amount"], trade["created_at"],
        )
        result.update(found=True, source="trade", transaction=trade, verified=verified)
        on_chain = blockchain.verify_on_chain(tx_hash)
        if on_chain:
            result["on_chain"] = on_chain
        return result

    # Search wallet_transactions
    wtx_resp = sb.table("wallet_transactions").select("*").eq("tx_hash", tx_hash).execute()
    if wtx_resp.data:
        wtx = wtx_resp.data[0]
        expected = auth._generate_wallet_tx_hash(wtx["user_id"], wtx["type"], wtx["amount"], wtx["created_at"])
        result.update(found=True, source="wallet", transaction=wtx, verified=(expected == tx_hash))
        on_chain = blockchain.verify_on_chain(tx_hash)
        if on_chain:
            result["on_chain"] = on_chain
        return result

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
# NEXYPHER WEB INSIGHT BOT
# ══════════════════════════════════════════════════════════════════

class BotAskRequest(BaseModel):
    question: str
    context: Optional[str] = None   # optional: "bitcoin", "solana", etc.


@app.post("/api/bot/ask")
async def bot_ask(body: BotAskRequest, user=Depends(require_user)):
    """
    AI chatbot that answers crypto questions using LIVE web data.
    1. Parses question via NLP intent recognition + parameter extraction
    2. Fetches real-time data from CoinGecko, DexScreener, news
    3. Feeds everything to Gemini for a grounded, domain-specific answer
    4. Falls back to data-driven answers when AI quota is exhausted
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    # ── 1. NLP Intent Recognition ──
    parsed = parse_user_query(
        question,
        recognizer=_intent_recognizer,
        extractor=_param_extractor,
        clarifier=_clarification_engine,
    )
    nlp_intent = parsed.intent.value
    nlp_confidence = parsed.confidence
    nlp_tokens = parsed.params.tokens  # extracted tickers / addresses
    nlp_risk = parsed.params.risk_preference
    nlp_timeframe = parsed.params.timeframe

    # If clarification needed, include it in response
    clarification_data = None
    if parsed.clarification.needs_response:
        clarification_data = {
            "type": parsed.clarification.ctype.value,
            "message": parsed.clarification.message,
            "options": [
                {"key": o.key, "label": o.label, "value": o.value}
                for o in parsed.clarification.options
            ],
            "free_text_allowed": parsed.clarification.free_text_allowed,
        }

    # ── 1b. Detect coins (NLP tokens + legacy alias fallback) ──
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

    # Also add tokens extracted by the NLP recognizer
    for tok in nlp_tokens:
        tok_lower = tok.lower()
        if tok_lower in COIN_ALIASES:
            cg_id = COIN_ALIASES[tok_lower]
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
            "You are NEXYPHER Bot — an expert crypto intelligence assistant embedded in the NEXYPHER platform. "
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
                    "nlp": {
                        "intent": nlp_intent,
                        "confidence": round(nlp_confidence, 3),
                        "tokens_extracted": nlp_tokens,
                        "risk": nlp_risk,
                        "timeframe": nlp_timeframe,
                    },
                    "clarification": clarification_data,
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
        "nlp": {
            "intent": nlp_intent,
            "confidence": round(nlp_confidence, 3),
            "tokens_extracted": nlp_tokens,
            "risk": nlp_risk,
            "timeframe": nlp_timeframe,
        },
        "clarification": clarification_data,
    }


# ══════════════════════════════════════════════════════════════════
# NLP QUERY PARSING ENDPOINT
# ══════════════════════════════════════════════════════════════════

class ParseQueryRequest(BaseModel):
    query: str

@app.post("/api/ai/parse-query")
async def api_parse_query(body: ParseQueryRequest, user=Depends(require_user)):
    """
    Parse a natural-language query via the NLP pipeline.
    Returns intent classification, extracted parameters, and any
    clarification prompts needed before running the full AI pipeline.
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(400, "Query cannot be empty")

    parsed = parse_user_query(
        query,
        recognizer=_intent_recognizer,
        extractor=_param_extractor,
        clarifier=_clarification_engine,
    )

    # Build clarification if needed
    clarification = None
    if parsed.clarification.needs_response:
        clarification = {
            "type": parsed.clarification.ctype.value,
            "message": parsed.clarification.message,
            "options": [
                {"key": o.key, "label": o.label, "value": o.value}
                for o in parsed.clarification.options
            ],
            "free_text_allowed": parsed.clarification.free_text_allowed,
        }

    return {
        "ready": parsed.ready,
        "intent": parsed.intent.value,
        "confidence": round(parsed.confidence, 3),
        "tokens": parsed.params.tokens,
        "parameters": {
            "risk": parsed.params.risk_preference,
            "timeframe": parsed.params.timeframe,
            "num_recommendations": parsed.params.num_recommendations,
            "filters": {
                "min_liquidity": parsed.params.filters.min_liquidity if parsed.params.filters else None,
                "min_volume": parsed.params.filters.min_volume if parsed.params.filters else None,
            } if parsed.params.filters else None,
        },
        "clarification": clarification,
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
    """No-op — algo tables are created in Supabase dashboard via algo_tables.sql."""
    pass


# Algo tables managed in Supabase
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


# ── Strategy Endpoints ──

@app.get("/api/algo/strategies")
async def list_algo_strategies(user=Depends(require_user)):
    sb = get_supabase()
    resp = sb.table("algo_strategies").select("*").eq("user_id", user.id).order("created_at", desc=True).execute()
    return resp.data or []


@app.post("/api/algo/strategies")
async def create_algo_strategy(body: AlgoStrategyCreate, user=Depends(require_user)):
    sb = get_supabase()
    row = {
        "user_id": user.id, "name": body.name, "description": body.description,
        "instruments": json.dumps(body.instruments), "legs": json.dumps(body.legs),
        "strategy_type": body.strategy_type, "order_type": body.order_type,
        "risk_config": json.dumps(body.risk_config), "advanced_config": json.dumps(body.advanced_config),
    }
    resp = sb.table("algo_strategies").insert(row).execute()
    return resp.data[0]


@app.put("/api/algo/strategies/{strategy_id}")
async def update_algo_strategy(strategy_id: int, body: AlgoStrategyUpdate, user=Depends(require_user)):
    sb = get_supabase()
    existing = sb.table("algo_strategies").select("id").eq("id", strategy_id).eq("user_id", user.id).execute()
    if not existing.data:
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
    resp = sb.table("algo_strategies").update(updates).eq("id", strategy_id).eq("user_id", user.id).execute()
    return resp.data[0]


@app.delete("/api/algo/strategies/{strategy_id}")
async def delete_algo_strategy(strategy_id: int, user=Depends(require_user)):
    sb = get_supabase()
    sb.table("algo_strategies").delete().eq("id", strategy_id).eq("user_id", user.id).execute()
    return {"success": True}


@app.post("/api/algo/strategies/{strategy_id}/deploy")
async def deploy_algo_strategy(strategy_id: int, user=Depends(require_user)):
    sb = get_supabase()
    existing = sb.table("algo_strategies").select("id").eq("id", strategy_id).eq("user_id", user.id).execute()
    if not existing.data:
        raise HTTPException(404, "Strategy not found")

    # Check if user has connected exchanges
    exchanges = sb.table("algo_exchanges").select("id").eq("user_id", user.id).eq("connected", 1).execute()
    if not exchanges.data:
        raise HTTPException(400, "Connect an exchange before deploying")

    sb.table("algo_strategies").update({
        "status": "running",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }).eq("id", strategy_id).eq("user_id", user.id).execute()
    return {"success": True, "status": "running"}


# ── Exchange Endpoints ──

@app.get("/api/algo/exchanges")
async def list_algo_exchanges(user=Depends(require_user)):
    sb = get_supabase()
    resp = sb.table("algo_exchanges").select("id, name, api_key_last4, connected, connected_at").eq("user_id", user.id).execute()
    return resp.data or []


@app.post("/api/algo/exchanges")
async def connect_algo_exchange(body: AlgoExchangeConnect, user=Depends(require_user)):
    sb = get_supabase()
    api_last4 = body.api_key[-4:] if len(body.api_key) >= 4 else body.api_key
    resp = sb.table("algo_exchanges").insert({
        "user_id": user.id, "name": body.name, "api_key_last4": api_last4
    }).execute()
    return {"success": True, "id": resp.data[0]["id"], "name": body.name}


@app.delete("/api/algo/exchanges/{exchange_id}")
async def disconnect_algo_exchange(exchange_id: int, user=Depends(require_user)):
    sb = get_supabase()
    sb.table("algo_exchanges").delete().eq("id", exchange_id).eq("user_id", user.id).execute()
    return {"success": True}


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
    sb = get_supabase()
    query = sb.table("algo_trade_logs").select("*").eq("user_id", user.id)
    if date_from:
        query = query.gte("created_at", date_from)
    if date_to:
        query = query.lte("created_at", date_to + " 23:59:59")
    if severity:
        query = query.eq("severity", severity.upper())
    resp = query.order("created_at", desc=True).limit(limit).execute()
    return resp.data or []


@app.post("/api/algo/logs")
async def create_trade_log(body: TradeLogCreate, user=Depends(require_user)):
    sb = get_supabase()
    resp = sb.table("algo_trade_logs").insert({
        "user_id": user.id, "strategy_id": body.strategy_id,
        "severity": body.severity.upper(), "message": body.message,
        "metadata": json.dumps(body.metadata),
    }).execute()
    return {"success": True, "id": resp.data[0]["id"]}


@app.delete("/api/algo/logs")
async def clear_trade_logs(user=Depends(require_user)):
    sb = get_supabase()
    sb.table("algo_trade_logs").delete().eq("user_id", user.id).execute()
    return {"success": True}


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
    sb = get_supabase()
    query = sb.table("algo_trade_reports").select("*").eq("user_id", user.id)
    if date_from:
        query = query.gte("created_at", date_from)
    if date_to:
        query = query.lte("created_at", date_to + " 23:59:59")
    if exchange:
        query = query.eq("exchange", exchange)
    if mode:
        query = query.eq("mode", mode)
    resp = query.order("created_at", desc=True).execute()

    reports = resp.data or []
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


@app.post("/api/algo/reports")
async def create_trade_report(body: TradeReportCreate, user=Depends(require_user)):
    sb = get_supabase()
    resp = sb.table("algo_trade_reports").insert({
        "user_id": user.id, "strategy_id": body.strategy_id,
        "strategy_name": body.strategy_name, "pair": body.pair,
        "action": body.action, "qty": body.qty,
        "buy_price": body.buy_price, "sell_price": body.sell_price,
        "pnl": body.pnl, "fees": body.fees,
        "exchange": body.exchange, "mode": body.mode,
    }).execute()
    return {"success": True, "id": resp.data[0]["id"]}


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
# PLATFORM STATS (Home Page Live Metrics)
# ══════════════════════════════════════════════════════════════════

_platform_ai_scan_count = 0  # In-memory counter for AI analyses this session


@app.get("/api/platform/stats")
async def get_platform_stats():
    """
    Return live platform metrics for the home page stat cards.
    No auth required — public stats.

    Returns: tokens_tracked, ai_analyses_run, market_regime, auto_trades_today
    """
    global _platform_ai_scan_count

    # 1. Tokens tracked: how many tokens DexScreener feed knows about
    tokens_tracked = 0
    try:
        feed = await dex.get_latest_tokens(limit=1)
        # DexScreener returns paginated; we just want total count
        tokens_tracked = len(feed) if feed else 0
        # Also count CoinGecko top coins
        top = await cg.get_top_coins(limit=1)
        tokens_tracked = max(tokens_tracked, 100)  # Platform tracks 100+ tokens
    except Exception:
        tokens_tracked = 100  # Sensible default

    # 2. AI analyses run: from learning loop + session counter
    ai_analyses = _platform_ai_scan_count
    try:
        from src.prediction_tracker import get_prediction_tracker
        tracker = get_prediction_tracker()
        ai_analyses += tracker.get_total_predictions_count()
    except Exception:
        pass

    # 3. Market regime: quick detection from TA
    market_regime = "Sideways"
    try:
        top_coins = await cg.get_top_coins(limit=3)
        if top_coins:
            coin = top_coins[0]
            history = await cg.get_price_history(coin.coin_id, days=14)
            if history and history.prices:
                prices_list = [p[1] for p in history.prices]
                volumes_list = [v[1] for v in history.volumes] if history.volumes else None
                result = ta.analyze(prices_list, volumes=volumes_list)
                regime = getattr(result, "market_regime", None) or ""
                if "bull" in regime.lower() or "up" in regime.lower():
                    market_regime = "Bullish"
                elif "bear" in regime.lower() or "down" in regime.lower():
                    market_regime = "Bearish"
                elif "volatile" in regime.lower():
                    market_regime = "Volatile"
                else:
                    market_regime = regime.title() if regime else "Sideways"
    except Exception:
        pass

    # 4. Auto trades today: from trading engine
    auto_trades = 0
    try:
        auto_trades = trading_engine.get_todays_trade_count()
    except Exception:
        pass

    return {
        "tokens_tracked": tokens_tracked,
        "ai_analyses_run": ai_analyses,
        "market_regime": market_regime,
        "auto_trades_today": auto_trades,
    }


def increment_ai_scan_count():
    """Call this after each AI analysis to increment the session counter."""
    global _platform_ai_scan_count
    _platform_ai_scan_count += 1


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
async def get_learning_stats():
    """Get AI prediction accuracy & performance statistics (public — aggregate data).
    Auto-backfills from trade history if ll_predictions is empty."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    stats = ll.get_performance_stats()
    return stats


@app.get("/api/ai/learning/health")
async def learning_health_check():
    """No-auth diagnostic: shows what the learning loop sees in the DB.
    Use this to debug zero-stats issues on deployed servers."""
    import traceback as _tb
    diag = {"ll_init": False, "table_exists": False, "row_count": 0,
            "sample_row": None, "error": None, "supabase_ok": False}
    try:
        from supabase_db import get_supabase
        sb = get_supabase()
        diag["supabase_ok"] = True
    except Exception as e:
        diag["error"] = f"supabase init: {e}"
        return diag

    try:
        res = sb.table("ll_predictions").select("id,token_ticker,predicted_direction,direction_correct_24h,pnl_pct_24h").limit(50).execute()
        rows = res.data or []
        diag["table_exists"] = True
        diag["row_count"] = len(rows)
        if rows:
            diag["sample_row"] = rows[0]
        diag["evaluated"] = len([r for r in rows if r.get("direction_correct_24h") is not None])
    except Exception as e:
        diag["error"] = f"query: {e}"
        return diag

    try:
        ll = _get_ll()
        diag["ll_init"] = ll is not None
        if ll:
            stats = ll.get_performance_stats()
            diag["stats_total"] = stats.get("total_predictions", -1)
            diag["stats_eval24"] = stats.get("evaluated_24h", -1)
            diag["stats_accuracy"] = stats.get("accuracy_24h", -1)
    except Exception as e:
        diag["error"] = f"stats: {e}\n{_tb.format_exc()}"

    return diag


@app.post("/api/ai/learning/backfill")
async def trigger_learning_backfill():
    """Manually trigger backfill of predictions from trade history."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    ll._backfill_done = False  # Force re-run
    created = ll.backfill_from_trades()
    return {"backfilled": created, "message": f"Created {created} predictions from trade history"}


@app.post("/api/ai/learning/evaluate")
async def trigger_learning_evaluation():
    """Trigger evaluation of pending predictions against actual prices.
    Also backfills from trade history if needed."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    # Ensure predictions exist before evaluating
    if not ll._backfill_done:
        ll.backfill_from_trades()
    evaluated = await ll.evaluate_pending(cg)
    eval_24h = evaluated.get("evaluated_24h", 0) if isinstance(evaluated, dict) else 0
    eval_7d = evaluated.get("evaluated_7d", 0) if isinstance(evaluated, dict) else 0
    return {
        "evaluated": evaluated,
        "message": f"Evaluated {eval_24h} (24h) and {eval_7d} (7d) predictions",
    }


@app.get("/api/ai/learning/adjustments")
async def get_strategy_adjustments():
    """Get AI-generated strategy adjustment recommendations based on performance."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    # Generate fresh adjustments (may be empty if criteria not met)
    new_adjustments = ll.generate_adjustments()
    # Also fetch recently stored adjustments so the page isn't blank
    stored = ll.get_recent_adjustments(limit=10)
    # Merge: new first, then stored (dedup by description)
    seen = set()
    merged = []
    for adj in new_adjustments + stored:
        # Normalize field name: backend stores 'adjustment_type', frontend reads 'type'
        if "adjustment_type" in adj and "type" not in adj:
            adj["type"] = adj["adjustment_type"]
        desc = adj.get("description", "")
        if desc not in seen:
            seen.add(desc)
            merged.append(adj)
    return {"adjustments": merged}


@app.get("/api/ai/learning/accuracy")
async def get_accuracy_history(days: int = Query(30, ge=1, le=365)):
    """Get historical accuracy metrics over time."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    accuracy = ll.get_historical_accuracy()
    return {"accuracy": accuracy, "overall": accuracy}


@app.get("/api/ai/learning/token/{token_id}")
async def get_token_track_record(token_id: str):
    """Get the AI's prediction track record for a specific token."""
    ll = _get_ll()
    if not ll:
        raise HTTPException(503, "Learning loop not available")
    record = ll.get_token_track_record(token_id)
    return {"token_id": token_id, "track_record": record}


# ══════════════════════════════════════════════════════════════════
# ML CONTINUOUS LEARNING (Paper Trading + Feedback Loop)
# ══════════════════════════════════════════════════════════════════

def _get_continuous_learner():
    """Lazy import of the ML continuous learner."""
    try:
        import sys, os, importlib.util
        from importlib.machinery import SourceFileLoader
        ml_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
        cl_path = os.path.join(ml_dir, "continuous_learner.py")
        if not os.path.exists(cl_path):
            return None
        loader = SourceFileLoader("continuous_learner", cl_path)
        spec = importlib.util.spec_from_loader("continuous_learner", loader, origin=cl_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = cl_path
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


@app.get("/api/ml/status")
async def ml_continuous_learning_status(user=Depends(require_user)):
    """Get ML continuous learning dashboard: wallet, trades, feedback, model info."""
    cl = _get_continuous_learner()
    if not cl:
        raise HTTPException(503, "Continuous learner not available")
    import sqlite3, json
    result = {}
    try:
        conn = sqlite3.connect(cl.PAPER_TRADE_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Wallet
        c.execute("SELECT * FROM paper_wallet WHERE id=1")
        w = c.fetchone()
        if w:
            result["wallet"] = {
                "balance": w["balance"],
                "total_invested": w["total_invested"],
                "total_returned": w["total_returned"],
                "pnl": round(w["balance"] - cl.PAPER_WALLET_INITIAL, 2),
                "pnl_pct": round((w["balance"] - cl.PAPER_WALLET_INITIAL) / cl.PAPER_WALLET_INITIAL * 100, 2),
                "total_trades": w["total_trades"],
                "winning_trades": w["winning_trades"],
                "losing_trades": w["losing_trades"],
                "win_rate": round(w["winning_trades"] / w["total_trades"] * 100, 1) if w["total_trades"] > 0 else 0,
            }

        # Open trades
        c.execute("SELECT token_id, entry_price, invested_amount, prob_up_7d, predicted_direction, entry_time FROM paper_trades WHERE status='OPEN' ORDER BY entry_time DESC")
        result["open_trades"] = [dict(r) for r in c.fetchall()]

        # Recent closed trades
        c.execute("SELECT token_id, outcome, pnl_pct, exit_reason, pnl_amount, evaluated_at FROM paper_trades WHERE status='CLOSED' ORDER BY evaluated_at DESC LIMIT 20")
        result["recent_trades"] = [dict(r) for r in c.fetchall()]

        # Feedback stats
        c.execute("SELECT COUNT(*) as total, SUM(CASE WHEN sample_weight > 1 THEN 1 ELSE 0 END) as reinforced, SUM(CASE WHEN sample_weight < 1 THEN 1 ELSE 0 END) as penalized FROM feedback_labels")
        fb = c.fetchone()
        result["feedback"] = {"total": fb["total"], "reinforced": fb["reinforced"] or 0, "penalized": fb["penalized"] or 0}

        # Learning log
        c.execute("SELECT event_type, description, accuracy_before, accuracy_after, created_at FROM learning_log ORDER BY created_at DESC LIMIT 10")
        result["learning_history"] = [dict(r) for r in c.fetchall()]

        conn.close()
    except Exception as e:
        logger.error("ML status error: %s", e)

    # Model metadata
    try:
        import os as _os
        meta_path = _os.path.join(cl.MODELS_DIR, "model_metadata.json")
        if _os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            result["model"] = {
                "version": meta.get("version"),
                "trained_at": meta.get("trained_at"),
                "training_mode": meta.get("training_mode", "initial_train"),
                "n_features": meta.get("n_features"),
                "accuracy_24h": meta.get("model_24h", {}).get("cv_mean"),
                "accuracy_7d": meta.get("model_7d", {}).get("cv_mean"),
                "accuracy_dir": meta.get("model_dir", {}).get("cv_mean"),
            }
    except Exception:
        pass

    return result


@app.post("/api/ml/paper-trade")
async def ml_execute_paper_trades(user=Depends(require_user)):
    """Execute paper trades based on current ML model predictions."""
    cl = _get_continuous_learner()
    if not cl:
        raise HTTPException(503, "Continuous learner not available")
    try:
        trades = cl.execute_paper_trades("1d")
        return {"trades_executed": trades, "wallet_balance": cl.get_wallet_balance()}
    except Exception as e:
        raise HTTPException(500, f"Paper trade error: {e}")


@app.post("/api/ml/evaluate")
async def ml_evaluate_trades(user=Depends(require_user)):
    """Evaluate outcomes of open paper trades."""
    cl = _get_continuous_learner()
    if not cl:
        raise HTTPException(503, "Continuous learner not available")
    try:
        closed = cl.evaluate_paper_trades()
        return {"trades_closed": closed, "wallet_balance": cl.get_wallet_balance()}
    except Exception as e:
        raise HTTPException(500, f"Evaluate error: {e}")


@app.post("/api/ml/feedback")
async def ml_generate_feedback(user=Depends(require_user)):
    """Generate feedback labels from closed paper trades."""
    cl = _get_continuous_learner()
    if not cl:
        raise HTTPException(503, "Continuous learner not available")
    try:
        count = cl.generate_feedback_labels()
        return {"feedback_generated": count}
    except Exception as e:
        raise HTTPException(500, f"Feedback error: {e}")


@app.post("/api/ml/retrain")
async def ml_retrain_with_feedback(user=Depends(require_user)):
    """Retrain ML models using feedback-weighted samples."""
    cl = _get_continuous_learner()
    if not cl:
        raise HTTPException(503, "Continuous learner not available")
    try:
        result = cl.retrain_with_feedback("1d", min_feedback=5)
        if result:
            return {"retrained": True, "version": result.get("version"), "accuracy_24h": result.get("model_24h", {}).get("cv_mean"), "accuracy_7d": result.get("model_7d", {}).get("cv_mean")}
        return {"retrained": False, "reason": "Not enough feedback labels yet"}
    except Exception as e:
        raise HTTPException(500, f"Retrain error: {e}")


@app.post("/api/ml/loop")
async def ml_run_learning_loop(user=Depends(require_user)):
    """Run one full iteration of the continuous learning loop."""
    cl = _get_continuous_learner()
    if not cl:
        raise HTTPException(503, "Continuous learner not available")
    try:
        result = cl.run_learning_loop("1d")
        return result
    except Exception as e:
        raise HTTPException(500, f"Loop error: {e}")


@app.post("/api/ml/import-real")
async def ml_import_real_trades(user=Depends(require_user)):
    """Import real trade outcomes from Supabase into ML feedback loop."""
    cl = _get_continuous_learner()
    if not cl:
        raise HTTPException(503, "Continuous learner not available")
    try:
        imported = cl.import_real_trade_outcomes()
        return {"imported": imported}
    except Exception as e:
        raise HTTPException(500, f"Import error: {e}")


@app.get("/api/ml/predict/{token_id}")
async def ml_predict_token(token_id: str, user=Depends(require_user)):
    """Get ML model prediction for a specific token."""
    try:
        import importlib.util
        from importlib.machinery import SourceFileLoader
        import os as _os
        ml_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ml")
        trainer_path = _os.path.join(ml_dir, "NEXYPHER model trainer \u00b7 py")
        loader = SourceFileLoader("NEXYPHER_trainer", trainer_path)
        spec = importlib.util.spec_from_loader("NEXYPHER_trainer", loader, origin=trainer_path)
        trainer = importlib.util.module_from_spec(spec)
        trainer.__file__ = trainer_path
        spec.loader.exec_module(trainer)
        db_path = _os.path.join(ml_dir, "nexypher_training_data.db")
        result = trainer.predict_from_db(token_id, "1d", db_path)
        if result:
            return result
        raise HTTPException(404, f"No data for {token_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════════
# ENHANCED AI RECOMMENDATIONS (with thought summaries)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/ai/enhanced-recommendations")
async def get_enhanced_ai_recommendations(
    query: str = Query("Top crypto picks", description="Natural language query"),
    num: int = Query(5, ge=1, le=20),
    format: str = Query("api", description="Output format: api, web, mobile, text"),
    user=Depends(require_user),
):
    """
    Full AI pipeline recommendations with AI Thought Summary,
    market regime detection, transparent reasoning, and NLP-powered
    personalization using the UI layer.
    Uses the Orchestrator + DataPipeline + PersonalizationEngine.
    """
    import traceback as _tb

    try:
        from src.ai_engine.models import (
            DataMode, MarketCondition, UserQuery, UserConfig, QueryType
        )
        from src.ai_engine.orchestrator import Orchestrator
        from src.data_collectors.data_pipeline import DataPipeline

        # ── NLP: Parse the natural-language query ──
        parsed = parse_user_query(
            query,
            recognizer=_intent_recognizer,
            extractor=_param_extractor,
            clarifier=_clarification_engine,
        )

        # ── Personalization: Build config/query from user prefs ──
        try:
            user_prefs = default_preferences(user_id=str(user.id))
        except Exception:
            user_prefs = default_preferences(user_id="0")

        pe = PersonalizationEngine(user_prefs)

        try:
            dp = DataPipeline()
        except Exception:
            dp = None

        ai_client = gemini_client
        orch = Orchestrator(
            ai_client=ai_client,
            data_fetcher=dp.fetch if dp else None,
            market_condition=MarketCondition.SIDEWAYS,
        )

        # Build UserQuery & UserConfig using PersonalizationEngine
        uq = pe.build_query(
            raw_query=query,
            intent=parsed.intent.value,
            tokens=parsed.params.tokens,
            num_recs=num,
            timeframe=parsed.params.timeframe,
            risk=parsed.params.risk_preference,
        )
        uc = pe.build_config()

        rec_set = await orch.run(uq, uc)

        # ── Personalization: Post-filter & annotate ──
        filtered_recs = pe.post_filter(rec_set.recommendations)
        rec_set.recommendations = filtered_recs
        pe.annotate(rec_set.recommendations)

        # ── Format output using ResponseFormatter ──
        formatted = None
        if format == "web":
            formatted = _response_formatter.format_web(rec_set)
        elif format == "mobile":
            formatted = _response_formatter.format_mobile(rec_set)
        elif format == "text":
            formatted = {"text": _response_formatter.format_text(rec_set)}

        # Always build the standard API recs_out (backward compatible)
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

        # ── Record predictions in learning loop ──
        ll = _get_ll()
        if ll:
            for r in rec_set.recommendations:
                try:
                    entry_price = r.current_price if r.current_price and r.current_price > 0 else 0
                    if entry_price <= 0:
                        continue
                    verdict_str = r.verdict.value if hasattr(r.verdict, "value") else str(r.verdict)
                    target_p = r.entry_exit.target_1 if r.entry_exit else 0
                    sl_p = r.entry_exit.stop_loss if r.entry_exit else 0
                    ll.record_prediction(
                        token_ticker=r.token_ticker.upper(),
                        token_name=r.token_name,
                        verdict=verdict_str,
                        confidence=r.confidence,
                        composite_score=r.composite_score,
                        price_at_prediction=entry_price,
                        target_price=target_p or 0,
                        stop_loss_price=sl_p or 0,
                        market_regime=getattr(r, "market_regime", "unknown"),
                        risk_level=r.risk_level.value if hasattr(r.risk_level, "value") else str(r.risk_level),
                        ai_thought_summary=getattr(r, "ai_thought_summary", ""),
                    )
                except Exception:
                    pass

        return {
            "query": query,
            "nlp": {
                "intent": parsed.intent.value,
                "confidence": round(parsed.confidence, 3),
                "tokens_extracted": parsed.params.tokens,
                "risk": parsed.params.risk_preference,
                "timeframe": parsed.params.timeframe,
            },
            "overall_ai_thought": getattr(rec_set, "overall_ai_thought", ""),
            "market_condition": rec_set.market_condition.value,
            "recommendations": recs_out,
            "formatted": formatted,
            "tokens_analyzed": rec_set.tokens_analyzed,
            "personalization": {
                "style_guidance": pe.trading_style_guidance(),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except BaseException as exc:
        error_tb = _tb.format_exc()
        logger.error("Enhanced recs failed:\n%s", error_tb)
        raise HTTPException(
            status_code=500,
            detail=f"AI pipeline error: {type(exc).__name__}: {exc}",
        )


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
