"""
PumpIQ Data Pipeline
======================
The **data_fetcher** callback that the Orchestrator calls in Step 2.

This module wires together:

    CoinGeckoCollector    → market data (price, volume, market cap)
    DexScreenerCollector → on-chain data (holders, liquidity, volume)
    NewsCollector        → news headlines + sentiment
    TechnicalAnalyzer    → RSI, MACD, support/resistance, patterns

… and converts their outputs into the engine's ``TokenData`` model.

Usage::

    from src.data_collectors.data_pipeline import DataPipeline

    pipeline = DataPipeline()
    tokens = await pipeline.fetch(query, config)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from src.ai_engine.models import (
    DataMode,
    NewsScorePayload,
    OnchainScorePayload,
    RiskLevel,
    SocialScorePayload,
    TechnicalScorePayload,
    TokenData,
    UserConfig,
    UserQuery,
)
from .coingecko_collector import CoinGeckoCollector, CoinMarketData, CoinPriceHistory
from .dexscreener_collector import DexScreenerCollector, DexScreenerToken
from .news_collector import NewsCollector, NewsResult
from .technical_analyzer import TechnicalAnalyzer, TechnicalResult

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrator-compatible ``data_fetcher`` that merges all collectors.

    Call signature matches what the Orchestrator expects::

        tokens: List[TokenData] = await pipeline.fetch(query, config)
    """

    def __init__(
        self,
        coingecko: Optional[CoinGeckoCollector] = None,
        dexscreener: Optional[DexScreenerCollector] = None,
        news: Optional[NewsCollector] = None,
        technical: Optional[TechnicalAnalyzer] = None,
    ):
        self.cg = coingecko or CoinGeckoCollector()
        self.dex = dexscreener or DexScreenerCollector()
        self.news = news or NewsCollector()
        self.ta = technical or TechnicalAnalyzer()

    # ── main entry (compatible with Orchestrator.data_fetcher) ────

    async def fetch(
        self,
        query: UserQuery,
        config: UserConfig,
    ) -> List[TokenData]:
        """
        Fetch data from all enabled sources and return ``TokenData`` objects.

        If the user asked about specific tokens, only those are fetched.
        Otherwise the pipeline picks the top trending / top market-cap
        tokens from CoinGecko + DexScreener.
        """
        tickers = query.specific_tokens

        if tickers:
            return await self._fetch_specific(tickers, config)
        else:
            return await self._fetch_discovery(config, limit=query.num_recommendations * 3)

    # ── fetch specific tickers ─────────────────────────────────────

    async def _fetch_specific(
        self,
        tickers: List[str],
        config: UserConfig,
    ) -> List[TokenData]:
        results: List[TokenData] = []
        for ticker in tickers:
            token_data = await self._build_token_data(ticker, config)
            if token_data:
                results.append(token_data)
        return results

    # ── discovery mode (no specific token) ─────────────────────────

    async def _fetch_discovery(
        self,
        config: UserConfig,
        limit: int = 15,
    ) -> List[TokenData]:
        """
        Pull top coins from CoinGecko + CoinGecko trending and merge.
        """
        top_task = self.cg.get_top_coins(limit=limit, category="solana-ecosystem")
        trending_task = self.cg.get_trending()

        top_coins, trending = await asyncio.gather(top_task, trending_task)

        # Merge: all top coins + trending that aren't already there
        symbols_seen = {c.symbol.upper() for c in top_coins}
        merged: List[CoinMarketData] = list(top_coins)

        for tc in trending:
            if tc.symbol.upper() not in symbols_seen:
                detail = await self.cg.get_coin_detail(tc.coin_id)
                if detail:
                    merged.append(detail)
                    symbols_seen.add(detail.symbol.upper())

        results: List[TokenData] = []
        for coin in merged[:limit]:
            td = await self._coin_to_token_data(coin, config)
            results.append(td)

        return results

    # ── build TokenData for a single ticker ────────────────────────

    async def _build_token_data(
        self,
        ticker: str,
        config: UserConfig,
    ) -> Optional[TokenData]:
        """
        For a given ticker, try:
          1. CoinGecko detail (by id = ticker.lower())
          2. DexScreener search (for micro-cap / launchpad tokens)
        """
        ticker_lower = ticker.lower()

        # Try CoinGecko first
        coin = await self.cg.get_coin_detail(ticker_lower)
        if coin and coin.current_price > 0:
            return await self._coin_to_token_data(coin, config)

        # Fallback to DexScreener search
        dex_tokens = await self.dex.search_tokens(ticker)
        if dex_tokens:
            return await self._dex_to_token_data(dex_tokens[0], config)

        logger.warning("Could not find data for ticker: %s", ticker)
        return None

    # ── converters ─────────────────────────────────────────────────

    async def _coin_to_token_data(
        self,
        coin: CoinMarketData,
        config: UserConfig,
    ) -> TokenData:
        """Convert CoinGecko market data → engine TokenData."""
        now = datetime.now(timezone.utc)

        # Get price history for TA (parallel with news)
        history_task = self.cg.get_price_history(coin.coin_id, days=7)
        news_task = self.news.collect(coin.symbol.upper())

        history, news_result = await asyncio.gather(history_task, news_task)

        # Technical analysis (with volume data for advanced features)
        ta_result = TechnicalResult()
        if history and history.prices and len(history.prices) >= 30:
            # Extract volumes if available from the history object
            vol_data = getattr(history, "total_volumes", None)
            ta_result = self.ta.analyze(history.prices, coin.current_price, volumes=vol_data)

        # Build payloads based on enabled modes
        news_payload = None
        if DataMode.NEWS in config.enabled_modes:
            news_payload = self._news_to_payload(news_result)

        onchain_payload = None
        if DataMode.ONCHAIN in config.enabled_modes:
            onchain_payload = self._coingecko_to_onchain(coin)

        technical_payload = None
        if DataMode.TECHNICAL in config.enabled_modes:
            technical_payload = self._ta_to_payload(ta_result)

        # Social is not available without Twitter/Reddit — leave as None
        social_payload = None

        return TokenData(
            token_name=coin.name,
            token_ticker=coin.symbol.upper(),
            current_price=coin.current_price,
            market_cap=coin.market_cap,
            token_age_days=0,
            news=news_payload,
            onchain=onchain_payload,
            technical=technical_payload,
            social=social_payload,
            collected_at=now,
        )

    async def _dex_to_token_data(
        self,
        dex: DexScreenerToken,
        config: UserConfig,
    ) -> TokenData:
        """Convert DexScreener on-chain data → engine TokenData."""
        now = datetime.now(timezone.utc)

        # News
        news_result = await self.news.collect(dex.symbol)
        news_payload = None
        if DataMode.NEWS in config.enabled_modes:
            news_payload = self._news_to_payload(news_result)

        # On-chain from DexScreener data
        onchain_payload = None
        if DataMode.ONCHAIN in config.enabled_modes:
            risk = RiskLevel.HIGH
            if dex.liquidity_mcap_ratio > 0.1 and dex.top_10_holder_pct < 50:
                risk = RiskLevel.MEDIUM
            if dex.liquidity_mcap_ratio > 0.2 and dex.top_10_holder_pct < 30:
                risk = RiskLevel.LOW

            # Score heuristic (0-10)
            score = 5.0
            if dex.buy_sell_ratio > 1.5:
                score += 1.5
            if dex.liquidity_mcap_ratio > 0.1:
                score += 1.0
            if dex.top_10_holder_pct < 40:
                score += 1.0
            if dex.volume_24h > 50_000:
                score += 1.0
            if dex.bonding_curve_pct > 80:
                score -= 0.5
            score = max(0, min(10, score))

            vol_trend = "rising" if dex.volume_1h > dex.volume_6h / 6 else "flat"

            onchain_payload = OnchainScorePayload(
                score=round(score, 1),
                summary=(
                    f"DexScreener token with ${dex.liquidity_usd:,.0f} liquidity, "
                    f"{dex.holder_count} holders, "
                    f"bonding curve {dex.bonding_curve_pct:.0f}%."
                ),
                holder_count=dex.holder_count,
                holder_growth_24h=0,
                top_10_concentration=dex.top_10_holder_pct,
                volume_24h=dex.volume_24h,
                volume_trend=vol_trend,
                liquidity=dex.liquidity_usd,
                liquidity_mcap_ratio=dex.liquidity_mcap_ratio,
                smart_money_summary=f"Buy/sell ratio: {dex.buy_sell_ratio:.1f}",
                risk_level=risk,
            )

        # Technical: we don't have chart data from DexScreener (only snapshots)
        technical_payload = None

        return TokenData(
            token_name=dex.name,
            token_ticker=dex.symbol.upper(),
            current_price=dex.price_usd,
            market_cap=dex.market_cap,
            token_age_days=max(1, int(dex.token_age_hours / 24)),
            news=news_payload,
            onchain=onchain_payload,
            technical=technical_payload,
            social=None,
            collected_at=now,
        )

    # ── payload converters ─────────────────────────────────────────

    @staticmethod
    def _news_to_payload(nr: NewsResult) -> NewsScorePayload:
        risk = RiskLevel.MEDIUM
        if nr.avg_sentiment < -0.3:
            risk = RiskLevel.HIGH
        elif nr.avg_sentiment > 0.3:
            risk = RiskLevel.LOW

        return NewsScorePayload(
            score=nr.score_0_10,
            summary=nr.narrative,
            key_headlines=nr.key_headlines,
            narrative=nr.narrative,
            source_count=nr.source_count,
            freshness_minutes=0,
            risk_level=risk,
        )

    @staticmethod
    def _coingecko_to_onchain(coin: CoinMarketData) -> OnchainScorePayload:
        """
        CoinGecko doesn't give on-chain holder data, but we can infer
        a basic score from volume, market cap, and price action.
        """
        score = 5.0
        if coin.total_volume_24h > 1_000_000:
            score += 1.5
        elif coin.total_volume_24h > 100_000:
            score += 0.8
        if coin.price_change_pct_24h > 5:
            score += 1.0
        elif coin.price_change_pct_24h < -10:
            score -= 1.0
        if coin.market_cap > 100_000_000:
            score += 0.5

        liq = coin.total_volume_24h * 0.1  # rough proxy
        liq_ratio = liq / max(coin.market_cap, 1) if coin.market_cap > 0 else 0

        vol_trend = "rising" if coin.price_change_pct_24h > 0 else "declining"

        risk = RiskLevel.MEDIUM
        if coin.market_cap < 1_000_000:
            risk = RiskLevel.HIGH
        elif coin.market_cap > 500_000_000:
            risk = RiskLevel.LOW

        return OnchainScorePayload(
            score=round(max(0, min(10, score)), 1),
            summary=(
                f"{coin.name}: ${coin.total_volume_24h:,.0f} 24h vol, "
                f"${coin.market_cap:,.0f} mcap, "
                f"{coin.price_change_pct_24h:+.1f}% 24h."
            ),
            holder_count=0,
            holder_growth_24h=0,
            top_10_concentration=0,
            volume_24h=coin.total_volume_24h,
            volume_trend=vol_trend,
            liquidity=liq,
            liquidity_mcap_ratio=round(liq_ratio, 4),
            risk_level=risk,
        )

    @staticmethod
    def _ta_to_payload(ta: TechnicalResult) -> TechnicalScorePayload:
        risk = RiskLevel.MEDIUM
        if ta.rsi > 75 or ta.trend == "downtrend":
            risk = RiskLevel.HIGH
        elif ta.rsi < 35 and ta.trend == "uptrend":
            risk = RiskLevel.LOW

        return TechnicalScorePayload(
            score=ta.score,
            summary=ta.summary,
            trend=ta.trend,
            rsi=ta.rsi,
            rsi_label=ta.rsi_label,
            macd_signal=ta.macd_crossover,
            support=ta.support,
            resistance=ta.resistance,
            pattern=ta.pattern,
            risk_level=risk,
            # Advanced market analysis fields
            market_regime=getattr(ta, "market_regime", "unknown"),
            volatility_state=getattr(ta, "volatility_state", "normal"),
            breakout_quality=getattr(ta, "breakout_quality", "none"),
            abnormal_volume=getattr(ta, "abnormal_volume", False),
            volume_anomaly_score=getattr(ta, "volume_anomaly_score", 0.0),
            short_term_trend=getattr(ta, "short_term_trend", "sideways"),
            long_term_trend=getattr(ta, "long_term_trend", "sideways"),
            trend_consistency=getattr(ta, "trend_consistency", 0.0),
            liquidity_pressure=getattr(ta, "liquidity_pressure", "neutral"),
        )
