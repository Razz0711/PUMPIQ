"""
Unit Tests – Intent Recognizer & Parameter Extractor
======================================================
Step 5.2 – Validates query classification, token extraction,
and parameter parsing from natural-language input.
"""

from __future__ import annotations

import pytest

from src.ui.intent_recognizer import ClassificationResult, Intent, IntentRecognizer
from src.ui.parameter_extractor import (
    ExtractedParams,
    FilterParams,
    ParameterExtractor,
    PriceContext,
)


# ══════════════════════════════════════════════════════════════════
# Intent Recognizer
# ══════════════════════════════════════════════════════════════════

class TestIntentRecognizer:
    """Test intent classification for all six intent categories."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.recognizer = IntentRecognizer()

    # ── Discovery ─────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "What are the best coins to buy?",
        "Give me top tokens right now",
        "best coins to invest in",
        "recommend some altcoins",
        "what should I buy today?",
    ])
    def test_discovery_intent(self, query):
        result = self.recognizer.classify(query)
        assert result.intent == Intent.DISCOVERY
        assert result.confidence >= 0.3

    # ── Analysis ──────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "Analyze $BONK",
        "What do you think about SOL?",
        "Should I buy $WIF?",
        "Deep dive on $PEPE",
        "look into $AVAX for me",
    ])
    def test_analysis_intent(self, query):
        result = self.recognizer.classify(query)
        assert result.intent == Intent.ANALYSIS
        assert result.confidence >= 0.5

    # ── Comparison ────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "Compare SOL vs ETH",
        "Which is better: $BONK or $WIF?",
        "SOL versus AVAX",
        "Pick between PEPE and DOGE",
    ])
    def test_comparison_intent(self, query):
        result = self.recognizer.classify(query)
        assert result.intent == Intent.COMPARISON
        assert result.confidence >= 0.5

    # ── Portfolio ─────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "I bought $SOL at $100, what now?",
        "my portfolio is losing money",
        "should I hold or sell $BONK? I hold 5000 tokens",
        "p/l on my positions",
    ])
    def test_portfolio_intent(self, query):
        result = self.recognizer.classify(query)
        assert result.intent == Intent.PORTFOLIO
        assert result.confidence >= 0.5

    # ── Alert ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "alert me when $SOL hits $200",
        "notify me if BONK reaches $0.01",
        "set a price alert for ETH",
        "remind me when the price hits $50",
    ])
    def test_alert_intent(self, query):
        result = self.recognizer.classify(query)
        assert result.intent == Intent.ALERT
        assert result.confidence >= 0.5

    # ── Strategy ──────────────────────────────────────────────────

    @pytest.mark.parametrize("query", [
        "Find me low-risk tokens with high upside",
        "Show me undervalued gems",
        "Filter for tokens with strong on-chain metrics",
        "Find tokens with good on-chain but bad social",
    ])
    def test_strategy_intent(self, query):
        result = self.recognizer.classify(query)
        assert result.intent == Intent.STRATEGY
        assert result.confidence >= 0.5

    # ── Token Extraction ──────────────────────────────────────────

    def test_cashtag_extraction(self):
        result = self.recognizer.classify("Analyze $BONK right now")
        assert "BONK" in result.tokens

    def test_multi_token_extraction(self):
        result = self.recognizer.classify("Compare $SOL vs $ETH vs $AVAX")
        assert len(result.tokens) >= 2

    def test_uppercase_token(self):
        result = self.recognizer.classify("What about SOL?")
        assert "SOL" in result.tokens

    # ── Confidence range ──────────────────────────────────────────

    def test_confidence_range(self):
        result = self.recognizer.classify("best coins")
        assert 0 <= result.confidence <= 1.0

    # ── Unknown / ambiguous ───────────────────────────────────────

    def test_short_ambiguous_query(self):
        result = self.recognizer.classify("hi")
        assert result.confidence < 0.8  # low confidence for short queries


# ══════════════════════════════════════════════════════════════════
# Parameter Extractor
# ══════════════════════════════════════════════════════════════════

class TestParameterExtractor:
    """Test structured parameter extraction from classified queries."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.extractor = ParameterExtractor()

    # ── Risk Preference ───────────────────────────────────────────

    @pytest.mark.parametrize("query,expected", [
        ("show me safe low-risk tokens", "conservative"),
        ("I want aggressive high-risk plays", "aggressive"),
        ("moderate risk balanced approach", "moderate"),
    ])
    def test_risk_extraction(self, query, expected):
        params = self.extractor.extract(query)
        assert params.risk_preference == expected

    # ── Timeframe ─────────────────────────────────────────────────

    @pytest.mark.parametrize("query,expected", [
        ("best coins for intraday trading", "day"),
        ("swing trade recommendations", "swing"),
        ("long term holds", "long"),
        ("quick scalp plays", "scalp"),
    ])
    def test_timeframe_extraction(self, query, expected):
        params = self.extractor.extract(query)
        assert params.timeframe == expected

    # ── Number of Recommendations ─────────────────────────────────

    @pytest.mark.parametrize("query,expected", [
        ("give me top 5 coins", 5),
        ("show me 3 picks for today", 3),
        ("top 10 tokens", 10),
    ])
    def test_num_recommendations(self, query, expected):
        params = self.extractor.extract(query)
        assert params.num_recommendations == expected

    # ── Price Context ─────────────────────────────────────────────

    def test_entry_price_extraction(self):
        params = self.extractor.extract("I bought at $0.005, should I sell?")
        assert params.price_context is not None
        assert params.price_context.entry_price == 0.005

    def test_target_price_extraction(self):
        params = self.extractor.extract("Alert me when it hits $100")
        assert params.price_context is not None
        assert params.price_context.target_price == 100.0

    # ── Token Extraction in Params ────────────────────────────────

    def test_token_in_params(self):
        params = self.extractor.extract("$BONK analysis please", tokens=["BONK"])
        assert "BONK" in params.tokens

    # ── Filter Params ─────────────────────────────────────────────

    def test_filter_low_risk(self):
        params = self.extractor.extract("find low risk tokens")
        assert params.filters.max_risk is not None or params.risk_preference == "conservative"

    def test_no_meme_filter(self):
        params = self.extractor.extract("no meme coins please, serious tokens only")
        assert params.filters.exclude_meme is True

    # ── Edge Cases ────────────────────────────────────────────────

    def test_empty_query(self):
        params = self.extractor.extract("")
        assert isinstance(params, ExtractedParams)

    def test_no_special_params(self):
        params = self.extractor.extract("what are the best coins?")
        assert isinstance(params, ExtractedParams)
        assert params.tokens is not None  # should be a list (possibly empty)
