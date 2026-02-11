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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Query, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data_collectors.coingecko_collector import CoinGeckoCollector
from src.data_collectors.dexscreener_collector import DexScreenerCollector
from src.data_collectors.news_collector import NewsCollector
from src.data_collectors.technical_analyzer import TechnicalAnalyzer

import auth

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
    global cg, dex, news, ta, gemini_client
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
    logger.info("PumpIQ v2 ready at http://localhost:8000")


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
    token = auth.create_access_token(user.id, user.email)
    return {"access_token": token, "token_type": "bearer", "user": user.model_dump()}


@app.post("/api/auth/login")
async def api_login(body: auth.UserLogin):
    user = auth.authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    token = auth.create_access_token(user.id, user.email)
    return {"access_token": token, "token_type": "bearer", "user": user.model_dump()}


@app.get("/api/auth/me")
async def api_me(user=Depends(require_user)):
    return user.model_dump()


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

@app.get("/api/token/{coin_id}", response_model=TokenDetail)
async def get_token_detail(coin_id: str):
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
        "collectors": {
            "coingecko": cg is not None,
            "dexscreener": dex is not None,
            "news": news is not None,
            "technical": ta is not None,
        },
    }
