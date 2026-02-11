"""
Tests for the new data collectors and Gemini client.

Covers:
    - CoinGeckoCollector     (parsing, helpers)
    - DexScreenerCollector   (parsing, bonding curve estimation)
    - NewsCollector          (keyword sentiment, aggregation)
    - TechnicalAnalyzer      (RSI, MACD, trend, composite score)
    - GeminiClient           (response dataclass, retry structure)
    - DataPipeline           (payload converters)
"""

import math
import pytest
from datetime import datetime, timezone

# ── CoinGecko ──────────────────────────────────────────────────────────

from src.data_collectors.coingecko_collector import (
    CoinGeckoCollector,
    CoinMarketData,
    CoinPriceHistory,
    TrendingCoin,
)


class TestCoinGeckoCollector:

    def setup_method(self):
        self.cg = CoinGeckoCollector(api_key="", currency="usd")

    def test_headers_no_key(self):
        h = self.cg._headers()
        assert "x-cg-pro-api-key" not in h

    def test_headers_with_key(self):
        cg = CoinGeckoCollector(api_key="CG-test")
        h = cg._headers()
        assert h["x-cg-demo-api-key"] == "CG-test"

    def test_base_url_free(self):
        assert self.cg._base == CoinGeckoCollector.FREE_BASE

    def test_base_url_demo(self):
        cg = CoinGeckoCollector(api_key="CG-test")
        assert cg._base == CoinGeckoCollector.DEMO_BASE

    def test_parse_market_item(self):
        item = {
            "id": "solana",
            "symbol": "sol",
            "name": "Solana",
            "current_price": 150.0,
            "market_cap": 70_000_000_000,
            "market_cap_rank": 5,
            "total_volume": 3_000_000_000,
            "price_change_24h": 5.5,
            "price_change_percentage_24h": 3.8,
            "ath": 260.0,
            "ath_change_percentage": -42.0,
            "circulating_supply": 440_000_000,
            "total_supply": 570_000_000,
            "high_24h": 155.0,
            "low_24h": 142.0,
        }
        coin = self.cg._parse_market_item(item)
        assert coin.coin_id == "solana"
        assert coin.symbol == "sol"
        assert coin.current_price == 150.0
        assert coin.market_cap == 70_000_000_000
        assert coin.total_volume_24h == 3_000_000_000
        assert coin.market_cap_rank == 5

    def test_coin_market_data_defaults(self):
        c = CoinMarketData(coin_id="test", symbol="TST", name="Test")
        assert c.current_price == 0.0
        assert c.market_cap == 0.0

    def test_coin_price_history_default(self):
        h = CoinPriceHistory(coin_id="test")
        assert h.prices == []
        assert h.currency == "usd"
        assert h.days == 7

    def test_trending_coin(self):
        t = TrendingCoin(coin_id="bonk", symbol="BONK", name="Bonk", score=0)
        assert t.score == 0
        assert t.symbol == "BONK"


# ── DexScreener ────────────────────────────────────────────────────────

from src.data_collectors.dexscreener_collector import (
    DexPairInfo,
    DexScreenerCollector,
    DexScreenerToken,
)


class TestDexScreenerCollector:

    def setup_method(self):
        self.dex = DexScreenerCollector()

    def test_estimate_bonding_curve_zero(self):
        pair = DexPairInfo(
            pair_address="abc",
            dex_id="raydium",
            base_token_address="mint1",
            base_token_symbol="TEST",
            base_token_name="Test",
            quote_token_symbol="SOL",
            market_cap=0,
        )
        assert DexScreenerCollector._estimate_bonding_curve(pair) == 0.0

    def test_estimate_bonding_curve_mid(self):
        pair = DexPairInfo(
            pair_address="abc",
            dex_id="raydium",
            base_token_address="mint1",
            base_token_symbol="TEST",
            base_token_name="Test",
            quote_token_symbol="SOL",
            market_cap=34_500,
        )
        pct = DexScreenerCollector._estimate_bonding_curve(pair)
        assert 49 < pct < 51  # ~50%

    def test_estimate_bonding_curve_graduated(self):
        pair = DexPairInfo(
            pair_address="abc",
            dex_id="raydium",
            base_token_address="mint1",
            base_token_symbol="TEST",
            base_token_name="Test",
            quote_token_symbol="SOL",
            market_cap=500_000,
        )
        assert DexScreenerCollector._estimate_bonding_curve(pair) == 100.0

    def test_parse_pair(self):
        raw = {
            "pairAddress": "pair123",
            "dexId": "raydium",
            "baseToken": {
                "address": "mint_addr",
                "symbol": "BONK",
                "name": "Bonk",
            },
            "quoteToken": {"symbol": "SOL"},
            "priceUsd": "0.0000245",
            "priceNative": "0.0000001",
            "volume": {"h24": 50000, "h6": 12000, "h1": 3000},
            "priceChange": {"m5": 1.5, "h1": 3.2, "h6": -2.0, "h24": 10.0},
            "liquidity": {"usd": 80000},
            "fdv": 100000,
            "marketCap": 90000,
            "txns": {"h24": {"buys": 200, "sells": 100}},
            "pairCreatedAt": 1700000000000,
            "chainId": "solana",
        }
        pair = self.dex._parse_pair(raw)
        assert pair.pair_address == "pair123"
        assert pair.base_token_symbol == "BONK"
        assert pair.price_usd == 0.0000245
        assert pair.volume_24h == 50000
        assert pair.liquidity_usd == 80000
        assert pair.txns_buys_24h == 200
        assert pair.txns_sells_24h == 100

    def test_dexscreener_token_defaults(self):
        t = DexScreenerToken(token_address="abc", symbol="X", name="X Token")
        assert t.price_usd == 0.0
        assert t.bonding_curve_pct == 0.0
        assert t.has_migrated is False
        assert t.buy_sell_ratio == 1.0


# ── News Collector ─────────────────────────────────────────────────────

from src.data_collectors.news_collector import (
    NewsArticle,
    NewsCollector,
    NewsResult,
    _keyword_sentiment,
)


class TestKeywordSentiment:

    def test_bullish_text(self):
        score = _keyword_sentiment("SOL surge rally breakout bullish")
        assert score > 0.5

    def test_bearish_text(self):
        score = _keyword_sentiment("crash dump scam rug bearish")
        assert score < -0.5

    def test_neutral_text(self):
        score = _keyword_sentiment("the quick brown fox jumps over")
        assert score == 0.0

    def test_mixed_text(self):
        score = _keyword_sentiment("rally crash bullish bearish")
        assert -0.1 <= score <= 0.1  # balanced


class TestNewsCollector:

    def setup_method(self):
        self.nc = NewsCollector(api_key="")

    def test_aggregate_empty(self):
        result = self.nc._aggregate("BONK", [])
        assert result.total_articles == 0
        assert result.avg_sentiment == 0.0
        assert result.score_0_10 == 5.0  # neutral

    def test_aggregate_bullish(self):
        articles = [
            NewsArticle(title="SOL surge rally", sentiment=0.8),
            NewsArticle(title="bullish breakout", sentiment=0.6),
        ]
        result = self.nc._aggregate("SOL", articles)
        assert result.avg_sentiment > 0.5
        assert result.score_0_10 > 7.0
        assert result.bullish_count == 2

    def test_aggregate_bearish(self):
        articles = [
            NewsArticle(title="crash dump incoming", sentiment=-0.7),
            NewsArticle(title="scam warning", sentiment=-0.5),
        ]
        result = self.nc._aggregate("SCAM", articles)
        assert result.avg_sentiment < -0.3
        assert result.score_0_10 < 4.0
        assert result.bearish_count == 2

    def test_score_0_10_range(self):
        # -1 → 0, 0 → 5, +1 → 10
        r = NewsResult(avg_sentiment=-1.0)
        assert r.score_0_10 == 0.0
        r = NewsResult(avg_sentiment=0.0)
        assert r.score_0_10 == 5.0
        r = NewsResult(avg_sentiment=1.0)
        assert r.score_0_10 == 10.0

    def test_narrative_strongly_bullish(self):
        articles = [NewsArticle(title="moon", sentiment=0.8)]
        result = self.nc._aggregate("X", articles)
        assert "bullish" in result.narrative.lower()


# ── Technical Analyzer ─────────────────────────────────────────────────

from src.data_collectors.technical_analyzer import TechnicalAnalyzer, TechnicalResult


class TestTechnicalAnalyzer:

    def setup_method(self):
        self.ta = TechnicalAnalyzer()

    def _make_prices(self, values):
        """Convert a list of floats to [[timestamp, price], …]."""
        return [[i * 3600000, v] for i, v in enumerate(values)]

    def test_insufficient_data(self):
        prices = self._make_prices([1.0, 2.0, 3.0])
        result = self.ta.analyze(prices)
        assert "Insufficient" in result.summary

    def test_rsi_overbought(self):
        # Steadily rising prices → RSI should be high
        prices = self._make_prices([i * 1.0 for i in range(1, 51)])
        result = self.ta.analyze(prices)
        assert result.rsi > 60
        assert result.rsi_label in ("overbought", "neutral")

    def test_rsi_oversold(self):
        # Steadily falling prices → RSI should be low
        prices = self._make_prices([50 - i * 0.8 for i in range(50)])
        result = self.ta.analyze(prices)
        assert result.rsi < 40

    def test_uptrend_detection(self):
        # Strong uptrend
        prices = self._make_prices([10 + i * 0.5 for i in range(60)])
        result = self.ta.analyze(prices)
        assert result.trend == "uptrend"

    def test_downtrend_detection(self):
        # Strong downtrend
        prices = self._make_prices([100 - i * 1.2 for i in range(60)])
        result = self.ta.analyze(prices)
        assert result.trend == "downtrend"

    def test_score_range(self):
        prices = self._make_prices([10 + i * 0.3 for i in range(60)])
        result = self.ta.analyze(prices)
        assert 0 <= result.score <= 10

    def test_support_resistance(self):
        # Oscillating prices
        values = []
        for i in range(80):
            values.append(50 + 10 * math.sin(i / 5))
        prices = self._make_prices(values)
        result = self.ta.analyze(prices)
        assert result.support > 0
        assert result.resistance > result.support

    def test_macd_fields(self):
        prices = self._make_prices([10 + i * 0.2 for i in range(60)])
        result = self.ta.analyze(prices)
        assert isinstance(result.macd_value, float)
        assert result.macd_crossover in ("bullish_crossover", "bearish", "neutral")

    def test_ema_helper(self):
        vals = [10.0, 12.0, 11.0, 13.0, 14.0, 15.0, 16.0, 14.0, 13.0, 15.0]
        ema = TechnicalAnalyzer._ema(vals, 5)
        assert isinstance(ema, float)
        assert ema > 0

    def test_pattern_none_for_random(self):
        # Random-ish data, neutral RSI → likely "None" or "Consolidation"
        prices = self._make_prices([50 + (i % 3) for i in range(60)])
        result = self.ta.analyze(prices)
        assert result.pattern in ("None", "Consolidation", "Bullish Continuation",
                                  "Bearish Continuation", "Bullish Reversal",
                                  "Bearish Reversal")


# ── Gemini Client ──────────────────────────────────────────────────────

from src.ai_engine.gemini_client import GeminiClient, GeminiMessage, GeminiResponse


class TestGeminiClient:

    def test_response_defaults(self):
        r = GeminiResponse()
        assert r.success is True
        assert r.content == ""
        assert r.total_tokens == 0

    def test_response_error(self):
        r = GeminiResponse(success=False, error="timeout")
        assert r.success is False
        assert r.error == "timeout"

    def test_message_dataclass(self):
        m = GeminiMessage(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"

    def test_client_defaults(self):
        c = GeminiClient(api_key="test-key")
        assert c.model_name == "gemini-2.0-flash"
        assert c.temperature == 0.7
        assert c.max_retries == 3

    def test_no_key_raises(self):
        c = GeminiClient(api_key="")
        import os
        orig = os.environ.get("GEMINI_API_KEY", "")
        os.environ["GEMINI_API_KEY"] = ""
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            c._ensure_client()
        if orig:
            os.environ["GEMINI_API_KEY"] = orig


# ── DataPipeline converters ────────────────────────────────────────────

from src.data_collectors.data_pipeline import DataPipeline
from src.data_collectors.news_collector import NewsResult as NR


class TestDataPipelineConverters:

    def setup_method(self):
        self.pipe = DataPipeline()

    def test_news_to_payload_bullish(self):
        nr = NR(avg_sentiment=0.5, narrative="bullish", key_headlines=["a"], source_count=2)
        payload = DataPipeline._news_to_payload(nr)
        assert payload.score > 5
        assert payload.risk_level.value == "LOW"

    def test_news_to_payload_bearish(self):
        nr = NR(avg_sentiment=-0.5, narrative="bearish", key_headlines=[], source_count=1)
        payload = DataPipeline._news_to_payload(nr)
        assert payload.score < 5
        assert payload.risk_level.value == "HIGH"

    def test_coingecko_to_onchain(self):
        coin = CoinMarketData(
            coin_id="solana", symbol="sol", name="Solana",
            current_price=150, market_cap=70_000_000_000,
            total_volume_24h=3_000_000_000,
            price_change_pct_24h=5.5,
        )
        payload = DataPipeline._coingecko_to_onchain(coin)
        assert payload.score > 5
        assert payload.volume_24h == 3_000_000_000
        assert payload.risk_level.value == "LOW"

    def test_ta_to_payload(self):
        ta_result = TechnicalResult(
            score=7.5, trend="uptrend", rsi=45, rsi_label="neutral",
            macd_crossover="bullish_crossover", support=140, resistance=160,
            pattern="Bullish Continuation", summary="Trend: uptrend.",
        )
        payload = DataPipeline._ta_to_payload(ta_result)
        assert payload.score == 7.5
        assert payload.trend == "uptrend"
        assert payload.risk_level.value == "MEDIUM"  # 7.5 maps to MEDIUM threshold
