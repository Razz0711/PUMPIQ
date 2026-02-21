"""
CoinGecko Market-Data Collector
=================================
Pulls real-time and historical market data from the **free** CoinGecko API v3.

API base : https://api.coingecko.com/api/v3
Rate limit: ~30 calls / min (demo key not required)
Docs      : https://docs.coingecko.com/v3.0.1/reference/introduction

If you have a **Pro** key set ``COINGECKO_API_KEY`` in .env and pass it
to the constructor — requests will go through the Pro endpoint instead.

Endpoints used
--------------
1. ``/coins/markets``          – top coins by market cap
2. ``/coins/{id}``             – full coin detail
3. ``/coins/{id}/market_chart``– price + volume history (for TA)
4. ``/search/trending``        – CoinGecko's trending list
5. ``/simple/price``           – lightweight price lookup
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ───────────────────────────── Data classes ─────────────────────────────

@dataclass
class CoinMarketData:
    """Snapshot of a coin returned by /coins/markets or /coins/{id}."""
    coin_id: str                     # CoinGecko slug  e.g. "solana"
    symbol: str                      # e.g. "sol"
    name: str                        # e.g. "Solana"
    current_price: float = 0.0
    market_cap: float = 0.0
    market_cap_rank: Optional[int] = None
    total_volume_24h: float = 0.0
    price_change_24h: float = 0.0    # absolute
    price_change_pct_24h: float = 0.0
    price_change_pct_7d: float = 0.0
    price_change_pct_30d: float = 0.0
    ath: float = 0.0
    ath_change_pct: float = 0.0
    circulating_supply: float = 0.0
    total_supply: Optional[float] = None
    high_24h: float = 0.0
    low_24h: float = 0.0
    last_updated: Optional[datetime] = None


@dataclass
class CoinPriceHistory:
    """Time-series price + volume arrays for a coin."""
    coin_id: str
    prices: List[List[float]] = field(default_factory=list)       # [[ts_ms, price], …]
    volumes: List[List[float]] = field(default_factory=list)      # [[ts_ms, vol], …]
    market_caps: List[List[float]] = field(default_factory=list)  # [[ts_ms, mcap], …]
    currency: str = "usd"
    days: int = 7


@dataclass
class TrendingCoin:
    """A coin that's trending on CoinGecko right now."""
    coin_id: str
    symbol: str
    name: str
    market_cap_rank: Optional[int] = None
    thumb: str = ""
    score: int = 0  # position in trending list (0 = most trending)


# ───────────────────────────── Collector ────────────────────────────────

class CoinGeckoCollector:
    """
    Async collector that wraps the CoinGecko REST API.

    Usage::

        cg = CoinGeckoCollector()                       # free tier
        cg = CoinGeckoCollector(api_key="CG-xyz...")    # pro tier

        markets = await cg.get_top_coins(limit=50)
        detail  = await cg.get_coin_detail("solana")
        history = await cg.get_price_history("solana", days=30)
        trending = await cg.get_trending()
    """

    FREE_BASE = "https://api.coingecko.com/api/v3"
    PRO_BASE  = "https://pro-api.coingecko.com/api/v3"
    DEMO_BASE = "https://api.coingecko.com/api/v3"

    def __init__(
        self,
        api_key: Optional[str] = None,
        currency: str = "usd",
        rate_limit_sleep: float = 2.5,
    ):
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY", "")
        self.currency = currency
        self.rate_limit_sleep = rate_limit_sleep
        self._client: Optional["httpx.AsyncClient"] = None

        # Determine tier: Pro keys typically start with "CG-" but use
        # pro-api.coingecko.com.  Demo keys also start with "CG-" but
        # use the free base with x-cg-demo-api-key header.
        # We default to Demo tier unless env says otherwise.
        self._is_pro = os.getenv("COINGECKO_PRO", "").lower() in ("1", "true")
        if self._is_pro and self.api_key:
            self._base = self.PRO_BASE
        elif self.api_key:
            self._base = self.DEMO_BASE
        else:
            self._base = self.FREE_BASE

    # ── helpers ────────────────────────────────────────────────────

    def _get_client(self) -> "httpx.AsyncClient":
        """Return a long-lived httpx client (connection pooling)."""
        import httpx
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            if self._is_pro:
                h["x-cg-pro-api-key"] = self.api_key
            else:
                h["x-cg-demo-api-key"] = self.api_key
        return h

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET request with rate-limit sleep and connection pooling."""
        url = f"{self._base}{path}"
        client = self._get_client()
        resp = await client.get(url, params=params or {}, headers=self._headers())
        resp.raise_for_status()
        await asyncio.sleep(self.rate_limit_sleep)
        return resp.json()

    # ── public API ─────────────────────────────────────────────────

    async def get_top_coins(
        self,
        limit: int = 50,
        category: Optional[str] = None,
    ) -> List[CoinMarketData]:
        """
        Fetch top coins ranked by market cap.

        ``category`` can be e.g. ``"solana-ecosystem"`` to filter.
        """
        params: Dict[str, Any] = {
            "vs_currency": self.currency,
            "order": "market_cap_desc",
            "per_page": min(limit, 250),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h,7d,30d",
        }
        if category:
            params["category"] = category

        try:
            data = await self._get("/coins/markets", params)
        except Exception as exc:
            logger.error("CoinGecko /coins/markets error: %s", exc)
            return []

        coins: List[CoinMarketData] = []
        for item in data:
            coins.append(self._parse_market_item(item))
        return coins

    async def get_coin_detail(self, coin_id: str) -> Optional[CoinMarketData]:
        """Full detail for one coin (by CoinGecko id like ``solana``).

        Falls back to /coins/markets?ids=<id> if /coins/{id} is rate-limited.
        """
        # ── attempt 1: detailed endpoint ──
        try:
            data = await self._get(
                f"/coins/{coin_id}",
                {
                    "localization": "false",
                    "tickers": "false",
                    "community_data": "true",
                    "developer_data": "false",
                },
            )
            md = data.get("market_data", {})
            return CoinMarketData(
                coin_id=data.get("id", coin_id),
                symbol=data.get("symbol", ""),
                name=data.get("name", ""),
                current_price=md.get("current_price", {}).get(self.currency, 0),
                market_cap=md.get("market_cap", {}).get(self.currency, 0),
                market_cap_rank=data.get("market_cap_rank"),
                total_volume_24h=md.get("total_volume", {}).get(self.currency, 0),
                price_change_24h=md.get("price_change_24h", 0),
                price_change_pct_24h=md.get("price_change_percentage_24h", 0),
                price_change_pct_7d=md.get("price_change_percentage_7d", 0),
                price_change_pct_30d=md.get("price_change_percentage_30d", 0),
                ath=md.get("ath", {}).get(self.currency, 0),
                ath_change_pct=md.get("ath_change_percentage", {}).get(self.currency, 0),
                circulating_supply=md.get("circulating_supply", 0) or 0,
                total_supply=md.get("total_supply"),
                high_24h=md.get("high_24h", {}).get(self.currency, 0),
                low_24h=md.get("low_24h", {}).get(self.currency, 0),
                last_updated=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.warning("CoinGecko /coins/%s failed (%s), trying /coins/markets fallback", coin_id, exc)

        # ── attempt 2: fallback via /coins/markets?ids=<id> ──
        try:
            data = await self._get(
                "/coins/markets",
                {
                    "vs_currency": self.currency,
                    "ids": coin_id,
                    "order": "market_cap_desc",
                    "per_page": "1",
                    "page": "1",
                    "sparkline": "false",
                    "price_change_percentage": "24h,7d,30d",
                },
            )
            if data and len(data) > 0:
                logger.info("CoinGecko fallback succeeded for %s", coin_id)
                return self._parse_market_item(data[0])
        except Exception as exc2:
            logger.error("CoinGecko fallback /coins/markets?ids=%s also failed: %s", coin_id, exc2)

        return None

    async def get_price_history(
        self,
        coin_id: str,
        days: int = 7,
    ) -> Optional[CoinPriceHistory]:
        """
        Get OHLC-style price + volume arrays for the past *days*.

        Granularity is auto-selected by CoinGecko:
            ≤ 1 day → ~5-min candles
            ≤ 90 days → hourly
            > 90 days → daily
        """
        try:
            data = await self._get(
                f"/coins/{coin_id}/market_chart",
                {"vs_currency": self.currency, "days": str(days)},
            )
        except Exception as exc:
            logger.error("CoinGecko price_history error: %s", exc)
            return None

        return CoinPriceHistory(
            coin_id=coin_id,
            prices=data.get("prices", []),
            volumes=data.get("total_volumes", []),
            market_caps=data.get("market_caps", []),
            currency=self.currency,
            days=days,
        )

    async def get_trending(self) -> List[TrendingCoin]:
        """Return CoinGecko's top-7 trending coins."""
        try:
            data = await self._get("/search/trending")
        except Exception as exc:
            logger.error("CoinGecko /search/trending error: %s", exc)
            return []

        coins: List[TrendingCoin] = []
        for idx, entry in enumerate(data.get("coins", [])):
            item = entry.get("item", {})
            coins.append(
                TrendingCoin(
                    coin_id=item.get("id", ""),
                    symbol=item.get("symbol", ""),
                    name=item.get("name", ""),
                    market_cap_rank=item.get("market_cap_rank"),
                    thumb=item.get("thumb", ""),
                    score=idx,
                )
            )
        return coins

    async def get_simple_price(
        self,
        coin_ids: List[str],
    ) -> Dict[str, float]:
        """
        Quick price lookup for a list of CoinGecko IDs.

        Returns ``{"solana": 148.23, "bonk": 0.0000245, …}``.
        """
        try:
            data = await self._get(
                "/simple/price",
                {
                    "ids": ",".join(coin_ids),
                    "vs_currencies": self.currency,
                },
            )
        except Exception as exc:
            logger.error("CoinGecko simple_price error: %s", exc)
            return {}

        return {cid: info.get(self.currency, 0) for cid, info in data.items()}

    # ── parsing helper ─────────────────────────────────────────────

    def _parse_market_item(self, item: Dict[str, Any]) -> CoinMarketData:
        return CoinMarketData(
            coin_id=item.get("id", ""),
            symbol=item.get("symbol", ""),
            name=item.get("name", ""),
            current_price=item.get("current_price", 0) or 0,
            market_cap=item.get("market_cap", 0) or 0,
            market_cap_rank=item.get("market_cap_rank"),
            total_volume_24h=item.get("total_volume", 0) or 0,
            price_change_24h=item.get("price_change_24h", 0) or 0,
            price_change_pct_24h=item.get("price_change_percentage_24h", 0) or 0,
            price_change_pct_7d=(
                item.get("price_change_percentage_7d_in_currency", 0) or 0
            ),
            price_change_pct_30d=(
                item.get("price_change_percentage_30d_in_currency", 0) or 0
            ),
            ath=item.get("ath", 0) or 0,
            ath_change_pct=item.get("ath_change_percentage", 0) or 0,
            circulating_supply=item.get("circulating_supply", 0) or 0,
            total_supply=item.get("total_supply"),
            high_24h=item.get("high_24h", 0) or 0,
            low_24h=item.get("low_24h", 0) or 0,
            last_updated=datetime.now(timezone.utc),
        )
