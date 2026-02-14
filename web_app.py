"""
PumpIQ Web Application — Full Platform
=========================================
Self-contained web server: auth, wallet connect, token feed, AI recs, leaderboard.

Run with:  python run_web.py
Open:      http://localhost:8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────
cg: CoinGeckoCollector = None  # type: ignore
dex: DexScreenerCollector = None  # type: ignore
news: NewsCollector = None  # type: ignore
ta: TechnicalAnalyzer = None  # type: ignore
gemini_client = None
_collectors_lock = threading.Lock()  # Thread-safe lazy initialization


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

app = FastAPI(title="PumpIQ", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "web" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup():
    """
    Lightweight startup for serverless compatibility.
    Data collectors are initialized lazily on first request.
    Database tables are created if they don't exist.
    
    Note: Background tasks (auto-trading) are disabled for serverless deployment.
    Use Vercel Cron Jobs or external schedulers for periodic tasks.
    """
    global cg, dex, news, ta, gemini_client
    
    # Initialize databases (idempotent)
    auth.init_db()
    trading_engine.init_trading_tables()
    
    logger.info("PumpIQ v2 ready for serverless deployment")
    
    # Background tasks disabled for serverless compatibility
    # To enable auto-trading in serverless:
    # - Use Vercel Cron Jobs (create vercel.json cron configuration)
    # - Or use external scheduler (GitHub Actions, etc.)
    # - Call /api/admin/run-auto-trade endpoint periodically
    
    # REMOVED for serverless: asyncio.create_task(_auto_trade_loop())


def _ensure_collectors_initialized():
    """Lazy initialization of data collectors (called on first request if needed).
    Thread-safe using a lock to prevent race conditions in concurrent requests."""
    global cg, dex, news, ta, gemini_client
    
    with _collectors_lock:
        if cg is None:
            cg = CoinGeckoCollector(api_key=os.getenv("COINGECKO_API_KEY", ""))
        if dex is None:
            dex = DexScreenerCollector(apify_api_key=os.getenv("APIFY_API_KEY", ""))
        if news is None:
            news = NewsCollector(api_key=os.getenv("CRYPTOPANIC_API_KEY", ""))
        if ta is None:
            ta = TechnicalAnalyzer()
        
        if gemini_client is None:
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            if gemini_key:
                try:
                    from src.ai_engine.gemini_client import GeminiClient
                    gemini_client = GeminiClient(api_key=gemini_key)
                    logger.info("Gemini AI client initialized (lazy)")
                except Exception as e:
                    logger.warning("Gemini init failed: %s", e)


# ── Serve frontend ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "web" / "index.html"
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
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    user = auth.register_user(body.email, body.username, body.password)
    if not user:
        raise HTTPException(409, "Email or username already taken")
    # Send welcome email with credentials
    try:
        smtp_service.send_registration_email(body.email, body.username, body.password)
    except Exception as e:
        logger.warning("Failed to send registration email: %s", e)
    token = auth.create_access_token(user.id, user.email)
    return {"access_token": token, "token_type": "bearer", "user": user.model_dump()}


@app.post("/api/auth/login")
async def api_login(body: auth.UserLogin, request: Request):
    user = auth.authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")
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


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/portfolio")
async def api_get_portfolio(user=Depends(require_user)):
    _ensure_collectors_initialized()
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
    _ensure_collectors_initialized()
    coins = await cg.get_top_coins(limit=limit)
    return [
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


@app.get("/api/market/trending", response_model=List[TrendingToken])
async def get_trending():
    _ensure_collectors_initialized()
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
    _ensure_collectors_initialized()
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

@app.get("/api/token/{coin_id}", response_model=TokenDetail)
async def get_token_detail(coin_id: str):
    _ensure_collectors_initialized()
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

    return AIRecommendations(
        market_summary=market_summary,
        tokens=tokens_scored[:15],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════════
# LEADERBOARD
# ══════════════════════════════════════════════════════════════════

@app.get("/api/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(limit: int = Query(25, ge=1, le=100)):
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

    return entries[:limit]


# ══════════════════════════════════════════════════════════════════
# DEX SEARCH
# ══════════════════════════════════════════════════════════════════

@app.get("/api/dex/search", response_model=List[DexToken])
async def dex_search(q: str = Query(..., min_length=1)):
    _ensure_collectors_initialized()
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
    _ensure_collectors_initialized()
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

            for row in enabled_users:
                try:
                    result = await trading_engine.auto_trade_cycle(
                        row["user_id"], cg, dex, gemini_client
                    )
                    if result.get("actions"):
                        logger.info("Auto-trade user %d: %s", row["user_id"], result["actions"])
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
    _ensure_collectors_initialized()
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
    _ensure_collectors_initialized()
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
    _ensure_collectors_initialized()
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
    _ensure_collectors_initialized()
    opportunities = await trading_engine.research_opportunities(cg, dex, gemini_client)
    return {"opportunities": opportunities[:15], "total": len(opportunities)}


# ── Run trade cycle manually ──

@app.post("/api/trader/run-cycle")
async def run_trade_cycle(user=Depends(require_user)):
    """Manually trigger one auto-trade cycle."""
    _ensure_collectors_initialized()
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
# PUMPIQ WEB INSIGHT BOT
# ══════════════════════════════════════════════════════════════════

class BotAskRequest(BaseModel):
    question: str
    context: Optional[str] = None   # optional: "bitcoin", "solana", etc.


@app.post("/api/bot/ask")
async def bot_ask(body: BotAskRequest, user=Depends(require_user)):
    _ensure_collectors_initialized()
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
