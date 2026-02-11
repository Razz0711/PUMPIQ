"""
Shared Test Fixtures & Configuration
=======================================
Step 5.2 – Pytest conftest with reusable fixtures for the entire test suite.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import List

import pytest
from fastapi.testclient import TestClient

# ── Ensure project root is on sys.path ────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════
# Event-loop fixture (for pytest-asyncio)
# ══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def event_loop():
    """Use a single event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ══════════════════════════════════════════════════════════════════
# FastAPI Test Client
# ══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def app():
    """Create the FastAPI app once for all tests."""
    from src.api.app import create_app
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Sync TestClient wrapping the FastAPI app."""
    with TestClient(app) as c:
        yield c


# ══════════════════════════════════════════════════════════════════
# AI-Engine Model Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture()
def sample_news_payload():
    from src.ai_engine.models import NewsScorePayload, RiskLevel
    return NewsScorePayload(
        score=7.5,
        summary="Strong bullish news coverage",
        key_headlines=["Major exchange listing announced", "Partnership with Visa"],
        narrative="Positive sentiment across news sources",
        source_count=12,
        freshness_minutes=8.0,
        risk_level=RiskLevel.LOW,
    )


@pytest.fixture()
def sample_onchain_payload():
    from src.ai_engine.models import OnchainScorePayload, RiskLevel
    return OnchainScorePayload(
        score=8.2,
        summary="Healthy on-chain metrics",
        holder_count=45000,
        holder_growth_24h=5.3,
        top_10_concentration=28.0,
        volume_24h=2_500_000,
        volume_trend="increasing",
        liquidity=1_200_000,
        liquidity_mcap_ratio=0.12,
        smart_money_summary="3 whale wallets accumulated",
        risk_level=RiskLevel.LOW,
    )


@pytest.fixture()
def sample_technical_payload():
    from src.ai_engine.models import TechnicalScorePayload, RiskLevel
    return TechnicalScorePayload(
        score=6.8,
        summary="Bullish chart with RSI room",
        trend="uptrend",
        rsi=58.0,
        rsi_label="neutral",
        macd_signal="bullish_crossover",
        support=0.0042,
        resistance=0.0065,
        pattern="Bull flag",
        risk_level=RiskLevel.MEDIUM,
    )


@pytest.fixture()
def sample_social_payload():
    from src.ai_engine.models import SocialScorePayload, RiskLevel
    return SocialScorePayload(
        score=9.0,
        score_max=12,
        summary="High community buzz",
        mention_count_24h=15000,
        mention_trend="rising_fast",
        influencer_count=8,
        telegram_members=32000,
        community_growth=12.5,
        trending_status="trending",
        red_flags=[],
        risk_level=RiskLevel.LOW,
    )


@pytest.fixture()
def sample_token_data(
    sample_news_payload,
    sample_onchain_payload,
    sample_technical_payload,
    sample_social_payload,
):
    from src.ai_engine.models import TokenData, RiskLevel
    return TokenData(
        token_name="SampleCoin",
        token_ticker="SAMP",
        current_price=0.0050,
        market_cap=5_000_000,
        token_age_days=45,
        news=sample_news_payload,
        onchain=sample_onchain_payload,
        technical=sample_technical_payload,
        social=sample_social_payload,
        composite_score=7.6,
        confidence=7.0,
        risk_level=RiskLevel.MEDIUM,
        collected_at=datetime.now(timezone.utc),
    )


@pytest.fixture()
def sample_token_data_bearish():
    """A bearish token for conflict / low-confidence tests."""
    from src.ai_engine.models import (
        NewsScorePayload, OnchainScorePayload,
        TechnicalScorePayload, SocialScorePayload,
        TokenData, RiskLevel,
    )
    return TokenData(
        token_name="RugPull",
        token_ticker="RUG",
        current_price=0.0001,
        market_cap=50_000,
        token_age_days=3,
        news=NewsScorePayload(score=3.0, risk_level=RiskLevel.HIGH),
        onchain=OnchainScorePayload(
            score=2.5, liquidity=5000, top_10_concentration=72,
            risk_level=RiskLevel.HIGH,
        ),
        technical=TechnicalScorePayload(
            score=2.0, trend="downtrend", rsi=85,
            risk_level=RiskLevel.HIGH,
        ),
        social=SocialScorePayload(
            score=10.0, score_max=12,
            red_flags=["Suspected bot activity", "Fake follower spike"],
            risk_level=RiskLevel.HIGH,
        ),
        composite_score=3.2,
        confidence=2.5,
        risk_level=RiskLevel.HIGH,
        collected_at=datetime.now(timezone.utc),
    )


@pytest.fixture()
def all_modes():
    from src.ai_engine.models import DataMode
    return list(DataMode)


@pytest.fixture()
def sample_user_config():
    from src.ai_engine.models import DataMode, UserConfig
    return UserConfig(
        enabled_modes=list(DataMode),
        mode_weights={
            DataMode.NEWS: 0.20,
            DataMode.ONCHAIN: 0.35,
            DataMode.TECHNICAL: 0.25,
            DataMode.SOCIAL: 0.20,
        },
    )


@pytest.fixture()
def sample_user_query():
    from src.ai_engine.models import UserQuery
    return UserQuery(
        raw_query="What are the best coins to buy right now?",
        num_recommendations=3,
    )


@pytest.fixture()
def sample_user_prefs():
    from src.ui.user_config import default_preferences
    return default_preferences()


@pytest.fixture()
def sample_recommendation():
    from src.ai_engine.models import (
        ConfidenceBreakdown, EntryExitPlan, RiskAssessment,
        RiskLevel, RecommendationVerdict, TokenRecommendation,
    )
    return TokenRecommendation(
        rank=1,
        token_name="SampleCoin",
        token_ticker="SAMP",
        current_price=0.005,
        verdict=RecommendationVerdict.MODERATE_BUY,
        confidence=7.5,
        confidence_breakdown=ConfidenceBreakdown(
            base=5.0,
            data_quality_modifier=2.0,
            signal_strength_modifier=1.0,
            conflict_penalty=0.0,
            data_freshness_modifier=0.5,
            raw_total=8.5,
            final_score=7.5,
            interpretation="High confidence",
        ),
        risk_level=RiskLevel.MEDIUM,
        risk_assessment=RiskAssessment(overall_risk=RiskLevel.MEDIUM),
        composite_score=7.6,
        entry_exit=EntryExitPlan(
            entry_low=0.0045,
            entry_high=0.0055,
            target_1=0.0070,
            target_1_pct=40.0,
            target_2=0.0100,
            target_2_pct=100.0,
            stop_loss=0.0035,
            stop_loss_pct=-30.0,
        ),
        core_thesis="Solid fundamentals with growing community",
        key_data_points=["Exchange listing announced", "Whale accumulation"],
        risks_and_concerns=["MEDIUM risk overall"],
    )
