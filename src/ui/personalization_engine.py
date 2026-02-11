"""
Personalization Engine
========================
Step 4.3 – Apply user preferences to the recommendation pipeline.

This module bridges the user preferences model with the AI engine
orchestrator.  It translates UserPreferences into the UserConfig
and UserQuery objects expected by the orchestrator, and applies
post-pipeline filtering based on the user's advanced filters,
risk profile, and portfolio context.

Personalization Flow:
    1. Load UserPreferences
    2. Build UserConfig (modes, weights, risk, timeframe)
    3. Build UserQuery  (intent → query type, tokens, num recs)
    4. Run orchestrator pipeline
    5. Post-filter results (risk cap, exclusions, age, liquidity)
    6. Annotate results (watchlist flags, portfolio P&L, position sizing)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .user_config import (
    AdvancedFilters,
    RiskProfile,
    TokenAgeFilter,
    TradingStyle,
    UserPreferences,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# AI-Engine Model Adapters (lazy import to avoid circular deps)
# ══════════════════════════════════════════════════════════════════

def _build_user_config(prefs: UserPreferences):
    """
    Convert UI-layer UserPreferences → AI-engine UserConfig.
    """
    from src.ai_engine.models import DataMode, InvestmentTimeframe, RiskTolerance, UserConfig

    # Map enabled modes
    mode_map = {
        "news": DataMode.NEWS,
        "onchain": DataMode.ONCHAIN,
        "technical": DataMode.TECHNICAL,
        "social": DataMode.SOCIAL,
    }
    enabled = [mode_map[m] for m in prefs.data_sources.enabled_modes if m in mode_map]

    # Normalised weights
    nw = prefs.data_sources.normalised_weights
    weights = {mode_map[k]: v for k, v in nw.items() if k in mode_map}

    # Risk tolerance
    risk_map = {
        RiskProfile.CONSERVATIVE: RiskTolerance.CONSERVATIVE,
        RiskProfile.MODERATE: RiskTolerance.MODERATE,
        RiskProfile.AGGRESSIVE: RiskTolerance.AGGRESSIVE,
    }
    risk = risk_map.get(prefs.risk_profile, RiskTolerance.MODERATE)

    # Timeframe
    tf_map = {
        TradingStyle.SCALPING: InvestmentTimeframe.SCALPING,
        TradingStyle.DAY_TRADING: InvestmentTimeframe.DAY_TRADING,
        TradingStyle.SWING_TRADING: InvestmentTimeframe.SWING,
        TradingStyle.POSITION_TRADING: InvestmentTimeframe.LONG_TERM,
    }
    timeframe = tf_map.get(prefs.trading_style, InvestmentTimeframe.DAY_TRADING)

    return UserConfig(
        enabled_modes=enabled,
        mode_weights=weights,
        risk_tolerance=risk,
        timeframe=timeframe,
        min_confidence=prefs.filters.min_liquidity / 100_000 if prefs.filters.min_liquidity else 0.65,
        max_recommendations=prefs.portfolio.max_concurrent_positions or 10,
        min_liquidity=prefs.filters.min_liquidity,
    )


def _build_user_query(
    raw_query: str,
    intent: str,
    tokens: List[str],
    prefs: UserPreferences,
    num_recs: Optional[int] = None,
    timeframe_override: Optional[str] = None,
    risk_override: Optional[str] = None,
):
    """
    Convert parsed intent + params → AI-engine UserQuery.
    """
    from src.ai_engine.models import (
        InvestmentTimeframe,
        QueryType,
        RiskTolerance,
        UserQuery,
    )

    # Intent → QueryType
    qt_map = {
        "discovery": QueryType.BEST_COINS,
        "analysis": QueryType.ANALYZE_TOKEN,
        "comparison": QueryType.BEST_COINS,   # treated as multi-token best_coins
        "strategy": QueryType.BEST_COINS,
        "portfolio": QueryType.PORTFOLIO_ADVICE,
        "alert": QueryType.BEST_COINS,        # alerts don't need query type
    }
    query_type = qt_map.get(intent, QueryType.BEST_COINS)

    # Timeframe
    tf_map = {
        "scalp": InvestmentTimeframe.SCALPING,
        "day": InvestmentTimeframe.DAY_TRADING,
        "swing": InvestmentTimeframe.SWING,
        "long": InvestmentTimeframe.LONG_TERM,
    }
    tf_pref_map = {
        TradingStyle.SCALPING: InvestmentTimeframe.SCALPING,
        TradingStyle.DAY_TRADING: InvestmentTimeframe.DAY_TRADING,
        TradingStyle.SWING_TRADING: InvestmentTimeframe.SWING,
        TradingStyle.POSITION_TRADING: InvestmentTimeframe.LONG_TERM,
    }
    if timeframe_override:
        timeframe = tf_map.get(timeframe_override, InvestmentTimeframe.DAY_TRADING)
    else:
        timeframe = tf_pref_map.get(prefs.trading_style, InvestmentTimeframe.DAY_TRADING)

    # Risk tolerance
    risk_map_str = {
        "conservative": RiskTolerance.CONSERVATIVE,
        "moderate": RiskTolerance.MODERATE,
        "aggressive": RiskTolerance.AGGRESSIVE,
    }
    risk_pref_map = {
        RiskProfile.CONSERVATIVE: RiskTolerance.CONSERVATIVE,
        RiskProfile.MODERATE: RiskTolerance.MODERATE,
        RiskProfile.AGGRESSIVE: RiskTolerance.AGGRESSIVE,
    }
    if risk_override:
        risk = risk_map_str.get(risk_override, RiskTolerance.MODERATE)
    else:
        risk = risk_pref_map.get(prefs.risk_profile, RiskTolerance.MODERATE)

    return UserQuery(
        raw_query=raw_query,
        query_type=query_type,
        timeframe=timeframe,
        risk_tolerance=risk,
        specific_tokens=tokens,
        num_recommendations=num_recs or 3,
        held_tokens=prefs.held_tickers,
    )


# ══════════════════════════════════════════════════════════════════
# Personalization Engine
# ══════════════════════════════════════════════════════════════════

class PersonalizationEngine:
    """
    Apply user preferences to every stage of the recommendation pipeline.

    Usage::

        engine = PersonalizationEngine(prefs)
        user_config = engine.build_config()
        user_query  = engine.build_query(raw, intent, tokens)
        filtered    = engine.post_filter(recommendations)
        annotated   = engine.annotate(filtered, current_prices)
    """

    def __init__(self, prefs: UserPreferences):
        self.prefs = prefs

    # ── Step 2: Build AI engine inputs ────────────────────────────

    def build_config(self):
        """Build an AI-engine UserConfig from user preferences."""
        return _build_user_config(self.prefs)

    def build_query(
        self,
        raw_query: str,
        intent: str = "discovery",
        tokens: Optional[List[str]] = None,
        num_recs: Optional[int] = None,
        timeframe: Optional[str] = None,
        risk: Optional[str] = None,
    ):
        """Build an AI-engine UserQuery from parsed intent + prefs."""
        return _build_user_query(
            raw_query, intent, tokens or [], self.prefs,
            num_recs=num_recs, timeframe_override=timeframe, risk_override=risk,
        )

    # ── Step 5: Post-filter recommendations ───────────────────────

    def post_filter(self, recommendations: list) -> list:
        """
        Apply user's advanced filters and risk profile to the
        recommendation list returned by the orchestrator.
        """
        filtered = []
        filters = self.prefs.filters
        max_risk = self.prefs.max_risk_for_profile()
        risk_ord = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        max_risk_val = risk_ord.get(max_risk, 2)
        age_min, age_max = self.prefs.token_age_days_range()

        for rec in recommendations:
            # Risk cap based on profile
            rec_risk_val = risk_ord.get(getattr(rec, "risk_level", "MEDIUM"), 1)
            if hasattr(rec.risk_level, "value"):
                rec_risk_val = risk_ord.get(rec.risk_level.value, 1)
            if rec_risk_val > max_risk_val:
                logger.debug("Filtered %s: risk %s > max %s", rec.token_ticker, rec.risk_level, max_risk)
                continue

            # Exclude already-owned tokens
            if filters.exclude_already_owned:
                if rec.token_ticker.upper() in self.prefs.held_tickers:
                    logger.debug("Filtered %s: already held", rec.token_ticker)
                    continue

            # Exclude meme coins (heuristic: check if "meme" in name)
            if filters.exclude_meme_coins:
                if "meme" in rec.token_name.lower():
                    continue

            filtered.append(rec)

        return filtered

    # ── Step 6: Annotate results ──────────────────────────────────

    def annotate(
        self,
        recommendations: list,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> list:
        """
        Enrich recommendations with user-specific context:
          - Flag if token is on watchlist
          - Add suggested position size
          - Mark if already held
        """
        for rec in recommendations:
            ticker = rec.token_ticker.upper()

            # Watchlist flag
            rec_dict = getattr(rec, "__dict__", {})
            on_watchlist = ticker in self.prefs.watchlist_tickers
            already_held = ticker in self.prefs.held_tickers

            # Attach as extra attributes if possible
            if hasattr(rec, "key_data_points"):
                if on_watchlist:
                    rec.key_data_points.insert(0, "⭐ On your watchlist")
                if already_held:
                    rec.key_data_points.insert(0, "📦 Already in your portfolio")

            # Position sizing suggestion
            if hasattr(rec, "risks_and_concerns"):
                risk_str = rec.risk_level.value if hasattr(rec.risk_level, "value") else str(rec.risk_level)
                suggested = self.prefs.portfolio.suggested_size(risk_str)
                if suggested > 0:
                    rec.risks_and_concerns.append(
                        f"💰 Suggested position: ${suggested:,.2f} "
                        f"({self.prefs.portfolio.position_size_percent}% of portfolio)"
                    )

        return recommendations

    # ── Weight Adjustments by Trading Style ───────────────────────

    def trading_style_guidance(self) -> Dict[str, str]:
        """
        Return human-readable guidance about how the trading style
        affects the analysis weights.
        """
        return {
            TradingStyle.SCALPING: (
                "Focus: Social spikes & volume surges. "
                "Ignoring long-term fundamentals. "
                "Using 1m-5m chart data."
            ),
            TradingStyle.DAY_TRADING: (
                "Focus: Technical setups & news. "
                "Balanced short/medium signals. "
                "Using 15m-1h chart data."
            ),
            TradingStyle.SWING_TRADING: (
                "Focus: On-chain trends & patterns. "
                "Ignoring micro price movements. "
                "Using 4h-daily chart data."
            ),
            TradingStyle.POSITION_TRADING: (
                "Focus: Fundamentals & partnerships. "
                "Ignoring short-term volatility. "
                "Using daily-weekly chart data."
            ),
        }.get(self.prefs.trading_style, "Balanced analysis across all dimensions.")
