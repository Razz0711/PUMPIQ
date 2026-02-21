"""
DexScreener On-Chain Data Collector
====================================
Fetches on-chain token data from **DexScreener** for any DEX-traded token
(primarily Solana / pump.fun-style launchpad tokens).

Data sources
------------
1. **DexScreener public API** (free, no key)  – real-time price, volume,
   liquidity, pair info for any DEX-traded token.
   Base: ``https://api.dexscreener.com``
2. **Apify DexScreener actor** (optional) – enriched scraping for extended
   data.  Requires ``APIFY_API_KEY`` in env.
3. **Solana JSON-RPC** – holder counts, token supply, account info.
   Default public endpoint: ``https://api.mainnet-beta.solana.com``

The collector outputs a ``DexScreenerToken`` dataclass that the pipeline
converts into the engine's ``OnchainScorePayload``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ───────────────────────── Data classes ─────────────────────────

@dataclass
class DexPairInfo:
    """One DEX trading pair for a token (from DexScreener)."""
    pair_address: str
    dex_id: str               # e.g. "raydium", "orca"
    base_token_address: str
    base_token_symbol: str
    base_token_name: str
    quote_token_symbol: str   # usually SOL or USDC
    price_usd: float = 0.0
    price_native: float = 0.0     # price in SOL
    volume_24h: float = 0.0
    volume_6h: float = 0.0
    volume_1h: float = 0.0
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    price_change_6h: float = 0.0
    price_change_24h: float = 0.0
    liquidity_usd: float = 0.0
    fdv: float = 0.0             # fully diluted valuation
    market_cap: float = 0.0
    pair_created_at: Optional[datetime] = None
    txns_buys_24h: int = 0
    txns_sells_24h: int = 0


@dataclass
class DexScreenerToken:
    """
    Aggregated on-chain profile of a single token from DexScreener.

    Fields are designed to map directly to ``OnchainScorePayload``.
    """
    token_address: str
    symbol: str
    name: str

    # Price & volume
    price_usd: float = 0.0
    volume_24h: float = 0.0
    volume_6h: float = 0.0
    volume_1h: float = 0.0
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    price_change_6h: float = 0.0
    price_change_24h: float = 0.0
    market_cap: float = 0.0
    fdv: float = 0.0

    # Liquidity
    liquidity_usd: float = 0.0
    liquidity_mcap_ratio: float = 0.0  # healthy > 0.05

    # Holders (from Solana RPC)
    holder_count: int = 0
    top_10_holder_pct: float = 0.0  # concentration %

    # Bonding curve (pump.fun-style launchpads)
    bonding_curve_pct: float = 0.0  # 0-100 – how far through the curve
    has_migrated: bool = False       # graduated from bonding curve to DEX

    # Trading activity
    buys_24h: int = 0
    sells_24h: int = 0
    buy_sell_ratio: float = 1.0

    # DEX info
    dex_id: str = ""
    pair_address: str = ""

    # Meta
    token_age_hours: float = 0.0
    collected_at: Optional[datetime] = None
    dex_pairs: List[DexPairInfo] = field(default_factory=list)


# ───────────────────────── Collector ────────────────────────────

class DexScreenerCollector:
    """
    Collects on-chain data from DexScreener + Solana RPC.

    Usage::

        dex = DexScreenerCollector()
        token = await dex.get_token("CONTRACT_ADDRESS_HERE")
        tokens = await dex.search_tokens("BONK")
    """

    DEXSCREENER_BASE = "https://api.dexscreener.com"
    APIFY_BASE = "https://api.apify.com/v2"
    SOLANA_RPC = "https://api.mainnet-beta.solana.com"

    def __init__(
        self,
        apify_api_key: Optional[str] = None,
        solana_rpc: Optional[str] = None,
        rate_limit_sleep: float = 1.0,
    ):
        self.apify_key = apify_api_key or os.getenv("APIFY_API_KEY", "")
        self.solana_rpc = solana_rpc or os.getenv(
            "SOLANA_RPC_URL", self.SOLANA_RPC
        )
        self.rate_limit_sleep = rate_limit_sleep
        self._client: Optional["httpx.AsyncClient"] = None

    # ── helpers ────────────────────────────────────────────────────

    def _get_client(self) -> "httpx.AsyncClient":
        """Return a long-lived httpx client (connection pooling)."""
        import httpx
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=20)
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _get_json(self, url: str, params: Optional[Dict] = None,
                        headers: Optional[Dict] = None) -> Any:
        client = self._get_client()
        resp = await client.get(url, params=params or {},
                                headers=headers or {})
        resp.raise_for_status()
        await asyncio.sleep(self.rate_limit_sleep)
        return resp.json()

    async def _rpc_call(self, method: str, params: list) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        client = self._get_client()
        resp = await client.post(self.solana_rpc, json=payload)
        resp.raise_for_status()
        return resp.json().get("result")

    # ── DexScreener endpoints ──────────────────────────────────────

    async def get_token_pairs(self, token_address: str) -> List[DexPairInfo]:
        """
        Get all DEX pairs for a Solana token address from DexScreener.

        DexScreener endpoint:
            GET /latest/dex/tokens/{tokenAddress}
        """
        url = f"{self.DEXSCREENER_BASE}/latest/dex/tokens/{token_address}"
        try:
            data = await self._get_json(url)
        except Exception as exc:
            logger.error("DexScreener token lookup error: %s", exc)
            return []

        pairs: List[DexPairInfo] = []
        for p in data.get("pairs", []) or []:
            if p.get("chainId") != "solana":
                continue
            pairs.append(self._parse_pair(p))

        return pairs

    async def search_pairs(self, query: str) -> List[DexPairInfo]:
        """
        Search DexScreener for a token by symbol / name.

        DexScreener endpoint:
            GET /latest/dex/search?q={query}
        """
        url = f"{self.DEXSCREENER_BASE}/latest/dex/search"
        try:
            data = await self._get_json(url, {"q": query})
        except Exception as exc:
            logger.error("DexScreener search error: %s", exc)
            return []

        pairs: List[DexPairInfo] = []
        for p in data.get("pairs", []) or []:
            if p.get("chainId") != "solana":
                continue
            pairs.append(self._parse_pair(p))
        return pairs

    # ── Apify DexScreener enrichment (optional) ────────────────────

    async def apify_search(self, query: str) -> List[Dict]:
        """
        Run DexScreener search via Apify actor for enriched data.

        Falls back silently to empty list if no Apify key is configured.
        """
        if not self.apify_key:
            return []

        try:
            # Run a DexScreener scraper actor (generic scraper pattern)
            run_url = (
                f"{self.APIFY_BASE}/acts/dexscreener~dexscreener-scraper"
                f"/run-sync-get-dataset-items"
            )
            headers = {"Authorization": f"Bearer {self.apify_key}"}
            body = {"searchQuery": query, "maxItems": 10}

            client = self._get_client()
            resp = await client.post(
                    run_url, json=body, headers=headers,
                    timeout=60,
                )
                if resp.status_code == 200:
                    items = resp.json()
                    logger.info("Apify returned %d items for '%s'", len(items), query)
                    return items if isinstance(items, list) else []
                else:
                    logger.warning(
                        "Apify DexScreener actor returned %d: %s",
                        resp.status_code, resp.text[:200],
                    )
        except Exception as exc:
            logger.warning("Apify DexScreener enrichment error: %s", exc)

        return []

    # ── Solana RPC ─────────────────────────────────────────────────

    async def get_holder_count(self, mint_address: str) -> int:
        """
        Estimate holder count via ``getTokenLargestAccounts`` (top 20).

        The public RPC is limited — for heavy usage switch to Helius/QuickNode.
        """
        try:
            result = await self._rpc_call(
                "getTokenLargestAccounts",
                [mint_address],
            )
            if result and "value" in result:
                return len(result["value"])
        except Exception as exc:
            logger.warning("Solana holder count error: %s", exc)
        return 0

    async def get_top_holder_concentration(self, mint_address: str) -> float:
        """
        Percentage of supply held by top 10 accounts.

        Uses ``getTokenLargestAccounts``.
        """
        try:
            supply_result = await self._rpc_call(
                "getTokenSupply", [mint_address]
            )
            total_supply = float(
                supply_result.get("value", {}).get("uiAmount", 0)
            ) if supply_result else 0

            if total_supply == 0:
                return 0.0

            largest = await self._rpc_call(
                "getTokenLargestAccounts", [mint_address]
            )
            if not largest or "value" not in largest:
                return 0.0

            top_10 = sorted(
                largest["value"],
                key=lambda x: float(x.get("uiAmount", 0) or 0),
                reverse=True,
            )[:10]
            top_10_total = sum(float(a.get("uiAmount", 0) or 0) for a in top_10)

            return round(top_10_total / total_supply * 100, 2)

        except Exception as exc:
            logger.warning("Top holder concentration error: %s", exc)
            return 0.0

    # ── high-level: full token profile ─────────────────────────────

    async def get_token(self, token_address: str) -> Optional[DexScreenerToken]:
        """
        Build a complete ``DexScreenerToken`` profile by combining
        DexScreener pair data + Solana RPC holder info.
        """
        pairs = await self.get_token_pairs(token_address)
        if not pairs:
            logger.warning("No DEX pairs found for %s", token_address)
            return None

        # Pick the highest-liquidity pair as primary
        primary = max(pairs, key=lambda p: p.liquidity_usd)

        # Holder info (async)
        holder_count = await self.get_holder_count(token_address)
        top_10_pct = await self.get_top_holder_concentration(token_address)

        # Bonding-curve estimation
        bonding_pct = self._estimate_bonding_curve(primary)
        has_migrated = bonding_pct >= 100.0

        age_hours = 0.0
        if primary.pair_created_at:
            delta = datetime.now(timezone.utc) - primary.pair_created_at
            age_hours = delta.total_seconds() / 3600

        buy_sell_ratio = (
            primary.txns_buys_24h / max(primary.txns_sells_24h, 1)
        )

        liq_mcap_ratio = (
            primary.liquidity_usd / max(primary.market_cap, 1)
        ) if primary.market_cap > 0 else 0.0

        return DexScreenerToken(
            token_address=token_address,
            symbol=primary.base_token_symbol,
            name=primary.base_token_name,
            price_usd=primary.price_usd,
            volume_24h=primary.volume_24h,
            volume_6h=primary.volume_6h,
            volume_1h=primary.volume_1h,
            price_change_5m=primary.price_change_5m,
            price_change_1h=primary.price_change_1h,
            price_change_6h=primary.price_change_6h,
            price_change_24h=primary.price_change_24h,
            market_cap=primary.market_cap,
            fdv=primary.fdv,
            liquidity_usd=primary.liquidity_usd,
            liquidity_mcap_ratio=round(liq_mcap_ratio, 4),
            holder_count=holder_count,
            top_10_holder_pct=top_10_pct,
            bonding_curve_pct=bonding_pct,
            has_migrated=has_migrated,
            buys_24h=primary.txns_buys_24h,
            sells_24h=primary.txns_sells_24h,
            buy_sell_ratio=round(buy_sell_ratio, 2),
            dex_id=primary.dex_id,
            pair_address=primary.pair_address,
            token_age_hours=round(age_hours, 1),
            collected_at=datetime.now(timezone.utc),
            dex_pairs=pairs,
        )

    async def search_tokens(self, query: str) -> List[DexScreenerToken]:
        """
        Search for tokens by symbol/name and return enriched profiles.
        """
        pairs = await self.search_pairs(query)
        if not pairs:
            return []

        # Group pairs by base token address
        by_address: Dict[str, List[DexPairInfo]] = {}
        for p in pairs:
            by_address.setdefault(p.base_token_address, []).append(p)

        results: List[DexScreenerToken] = []
        for addr in list(by_address.keys())[:5]:  # limit to 5 tokens
            token = await self.get_token(addr)
            if token:
                results.append(token)

        return results

    # ── internals ──────────────────────────────────────────────────

    def _parse_pair(self, p: Dict[str, Any]) -> DexPairInfo:
        base_token = p.get("baseToken", {})
        txns = p.get("txns", {})
        h24 = txns.get("h24", {})
        price_change = p.get("priceChange", {})

        created = None
        if p.get("pairCreatedAt"):
            try:
                created = datetime.fromtimestamp(
                    p["pairCreatedAt"] / 1000, tz=timezone.utc
                )
            except Exception:
                pass

        vol = p.get("volume", {})

        return DexPairInfo(
            pair_address=p.get("pairAddress", ""),
            dex_id=p.get("dexId", ""),
            base_token_address=base_token.get("address", ""),
            base_token_symbol=base_token.get("symbol", ""),
            base_token_name=base_token.get("name", ""),
            quote_token_symbol=p.get("quoteToken", {}).get("symbol", ""),
            price_usd=float(p.get("priceUsd", 0) or 0),
            price_native=float(p.get("priceNative", 0) or 0),
            volume_24h=float(vol.get("h24", 0) or 0),
            volume_6h=float(vol.get("h6", 0) or 0),
            volume_1h=float(vol.get("h1", 0) or 0),
            price_change_5m=float(price_change.get("m5", 0) or 0),
            price_change_1h=float(price_change.get("h1", 0) or 0),
            price_change_6h=float(price_change.get("h6", 0) or 0),
            price_change_24h=float(price_change.get("h24", 0) or 0),
            liquidity_usd=float(p.get("liquidity", {}).get("usd", 0) or 0),
            fdv=float(p.get("fdv", 0) or 0),
            market_cap=float(p.get("marketCap", 0) or p.get("fdv", 0) or 0),
            pair_created_at=created,
            txns_buys_24h=int(h24.get("buys", 0) or 0),
            txns_sells_24h=int(h24.get("sells", 0) or 0),
        )

    @staticmethod
    def _estimate_bonding_curve(pair: DexPairInfo) -> float:
        """
        Heuristic to estimate bonding-curve completion %.

        Pump.fun-style tokens start at ~$0 and migrate to a full DEX pair
        once the bonding curve reaches 100%.  We approximate this using
        market cap vs a typical graduation threshold (~$69K).
        """
        GRADUATION_MCAP = 69_000  # USD – typical bonding-curve cap
        if pair.market_cap <= 0:
            return 0.0
        pct = min(pair.market_cap / GRADUATION_MCAP * 100, 100.0)
        return round(pct, 1)
