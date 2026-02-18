"""
AI Engine – Data Models
========================
Pydantic models for the AI synthesis engine: token data, scores,
recommendations, and all intermediate structures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────

class InvestmentTimeframe(str, Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING = "swing"
    LONG_TERM = "long_term"


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class PositionSizePreference(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class MarketCondition(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class RecommendationVerdict(str, Enum):
    STRONG_BUY = "Strong Buy"
    MODERATE_BUY = "Moderate Buy"
    HOLD = "Hold"
    CAUTIOUS_BUY = "Cautious Buy"
    WATCH = "Watch"
    AVOID = "Avoid"
    SELL = "Sell"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DataMode(str, Enum):
    NEWS = "news"
    ONCHAIN = "onchain"
    TECHNICAL = "technical"
    SOCIAL = "social"


class ConflictSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"


class QueryType(str, Enum):
    BEST_COINS = "best_coins"          # "What are the best coins to buy now?"
    ANALYZE_TOKEN = "analyze_token"    # "Analyze $BONK"
    PORTFOLIO_ADVICE = "portfolio"     # "How is my portfolio looking?"
    MARKET_OVERVIEW = "market_overview" # "How's the market?"


# ─────────────────────────────────────────────────────────────────
# User Context
# ─────────────────────────────────────────────────────────────────

class UserQuery(BaseModel):
    """Parsed representation of a user's request."""
    raw_query: str
    query_type: QueryType = QueryType.BEST_COINS
    timeframe: InvestmentTimeframe = InvestmentTimeframe.SWING
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    position_size: PositionSizePreference = PositionSizePreference.MEDIUM
    specific_tokens: List[str] = Field(default_factory=list)
    num_recommendations: int = Field(default=3, ge=1, le=10)
    held_tokens: List[str] = Field(default_factory=list)


class UserConfig(BaseModel):
    """User's active configuration for the analysis session."""
    enabled_modes: List[DataMode] = Field(default_factory=lambda: list(DataMode))
    mode_weights: Dict[DataMode, float] = Field(default_factory=lambda: {
        DataMode.TECHNICAL: 0.35,
        DataMode.ONCHAIN: 0.30,
        DataMode.NEWS: 0.20,
        DataMode.SOCIAL: 0.10,
    })
    ml_signal_weight: float = Field(default=0.05, description="Weight for ML/LSTM signal blended into composite")
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    timeframe: InvestmentTimeframe = InvestmentTimeframe.SWING
    min_confidence: float = Field(default=0.65, ge=0, le=1)
    max_recommendations: int = 10
    min_liquidity: float = 10_000

    @field_validator("mode_weights")
    @classmethod
    def weights_must_sum_to_one(cls, v: Dict[DataMode, float]) -> Dict[DataMode, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            # Auto-normalise
            return {k: round(val / total, 4) for k, val in v.items()}
        return v


# ─────────────────────────────────────────────────────────────────
# Per-Module Score Payloads
# ─────────────────────────────────────────────────────────────────

class NewsScorePayload(BaseModel):
    """Output of the news sentiment module."""
    score: float = Field(0, ge=0, le=10, description="News sentiment 0-10")
    summary: str = ""
    key_headlines: List[str] = Field(default_factory=list)
    narrative: str = ""
    source_count: int = 0
    freshness_minutes: float = 0
    risk_level: RiskLevel = RiskLevel.MEDIUM


class OnchainScorePayload(BaseModel):
    """Output of the on-chain analysis module."""
    score: float = Field(0, ge=0, le=10, description="On-chain health 0-10")
    summary: str = ""
    holder_count: int = 0
    holder_growth_24h: float = 0  # percent
    top_10_concentration: float = 0  # percent
    volume_24h: float = 0
    volume_trend: str = ""
    liquidity: float = 0
    liquidity_mcap_ratio: float = 0
    smart_money_summary: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM


class TechnicalScorePayload(BaseModel):
    """Output of the technical analysis module."""
    score: float = Field(0, ge=0, le=10, description="Technical score 0-10")
    summary: str = ""
    trend: str = "sideways"  # uptrend / downtrend / sideways
    rsi: float = 50
    rsi_label: str = "neutral"  # overbought / neutral / oversold
    macd_signal: str = "neutral"  # bullish_crossover / bearish / neutral
    support: float = 0
    resistance: float = 0
    pattern: str = "None"
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # Advanced market analysis fields
    market_regime: str = "unknown"  # trending / ranging / unstable
    volatility_state: str = "normal"  # expanding / contracting / normal
    breakout_quality: str = "none"  # confirmed / weak / none
    abnormal_volume: bool = False  # possible whale activity
    volume_anomaly_score: float = 0.0  # 0-10 how anomalous the volume is
    short_term_trend: str = "sideways"  # uptrend / downtrend / sideways
    long_term_trend: str = "sideways"  # uptrend / downtrend / sideways
    trend_consistency: float = 0.0  # 0-1, how consistent the trend is
    liquidity_pressure: str = "neutral"  # buying / selling / neutral


class SocialScorePayload(BaseModel):
    """Output of the social sentiment module."""
    score: float = Field(0, ge=0, le=12, description="Social score 0-12")
    score_max: float = 12
    summary: str = ""
    mention_count_24h: int = 0
    mention_trend: str = ""
    influencer_count: int = 0
    telegram_members: int = 0
    community_growth: float = 0  # percent
    trending_status: str = ""
    red_flags: List[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM


# ─────────────────────────────────────────────────────────────────
# Aggregated Token Data (Step 3.1 – Step 2)
# ─────────────────────────────────────────────────────────────────

class TokenData(BaseModel):
    """
    Composite data object for a single token, carrying scores
    from every enabled module plus metadata.
    """
    token_name: str
    token_ticker: str
    current_price: float = 0
    market_cap: float = 0
    token_age_days: int = 0

    # Per-module payloads (None when mode disabled)
    news: Optional[NewsScorePayload] = None
    onchain: Optional[OnchainScorePayload] = None
    technical: Optional[TechnicalScorePayload] = None
    social: Optional[SocialScorePayload] = None

    # Computed fields (filled by orchestrator)
    composite_score: float = 0      # 0-10
    confidence: float = 0           # 0-10
    risk_level: RiskLevel = RiskLevel.MEDIUM
    conflicts: List["ConflictFlag"] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)

    collected_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────
# Conflict Flag (Step 3.1 – Step 4)
# ─────────────────────────────────────────────────────────────────

class ConflictFlag(BaseModel):
    """Represents a disagreement between two data modules."""
    severity: ConflictSeverity = ConflictSeverity.MINOR
    module_a: DataMode
    module_b: DataMode
    description: str
    confidence_penalty: float = 0  # How much to subtract from confidence


# ─────────────────────────────────────────────────────────────────
# Risk Assessment Details (Step 3.3)
# ─────────────────────────────────────────────────────────────────

class RiskAssessment(BaseModel):
    """Granular risk breakdown."""
    onchain_risk: RiskLevel = RiskLevel.MEDIUM
    technical_risk: RiskLevel = RiskLevel.MEDIUM
    social_risk: RiskLevel = RiskLevel.MEDIUM
    news_risk: RiskLevel = RiskLevel.MEDIUM
    volatility_risk: RiskLevel = RiskLevel.MEDIUM
    overall_risk: RiskLevel = RiskLevel.MEDIUM

    position_size_guidance: str = ""
    risk_factors: List[str] = Field(default_factory=list)
    risk_multipliers: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# Confidence Details (Step 3.3)
# ─────────────────────────────────────────────────────────────────

class ConfidenceBreakdown(BaseModel):
    """How the confidence score was computed."""
    base: float = 5.0
    data_quality_modifier: float = 0.0
    signal_strength_modifier: float = 0.0
    conflict_penalty: float = 0.0
    data_freshness_modifier: float = 0.0
    historical_accuracy_modifier: float = 0.0
    raw_total: float = 0.0
    final_score: float = Field(0, ge=1, le=10)
    interpretation: str = ""


# ─────────────────────────────────────────────────────────────────
# Entry / Exit Targets
# ─────────────────────────────────────────────────────────────────

class EntryExitPlan(BaseModel):
    """Computed trading plan for a recommendation."""
    entry_low: float = 0
    entry_high: float = 0
    target_1: float = 0
    target_1_pct: float = 0
    target_2: float = 0
    target_2_pct: float = 0
    stop_loss: float = 0
    stop_loss_pct: float = 0
    timeframe_estimate: str = ""
    rationale: str = ""


# ─────────────────────────────────────────────────────────────────
# Final Recommendation (output of the entire engine)
# ─────────────────────────────────────────────────────────────────

class TokenRecommendation(BaseModel):
    """A single actionable recommendation for a token."""
    rank: int = 0
    token_name: str
    token_ticker: str
    current_price: float = 0

    verdict: RecommendationVerdict = RecommendationVerdict.WATCH
    confidence: float = Field(0, ge=0, le=10)
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_assessment: Optional[RiskAssessment] = None
    composite_score: float = 0

    entry_exit: Optional[EntryExitPlan] = None
    core_thesis: str = ""
    key_data_points: List[str] = Field(default_factory=list)
    risks_and_concerns: List[str] = Field(default_factory=list)
    conflicts: List[ConflictFlag] = Field(default_factory=list)

    # AI Thought Summary — explains WHY the AI made this decision
    ai_thought_summary: str = ""

    # Per-module summaries
    news_analysis: str = ""
    onchain_analysis: str = ""
    technical_analysis: str = ""
    social_analysis: str = ""

    # Market regime classification
    market_regime: str = ""  # trending / ranging / unstable

    generated_at: Optional[datetime] = None


class PredictionRecord(BaseModel):
    """Tracks a prediction for the learning/feedback loop."""
    prediction_id: str = ""
    token_ticker: str
    verdict: RecommendationVerdict = RecommendationVerdict.WATCH
    confidence: float = 0
    predicted_direction: str = ""  # up / down / flat
    predicted_target: float = 0  # predicted price target
    price_at_prediction: float = 0
    timestamp: Optional[datetime] = None
    # Outcome fields (filled later)
    actual_price_24h: float = 0
    actual_price_7d: float = 0
    outcome_correct: Optional[bool] = None
    outcome_pnl_pct: float = 0
    evaluated_at: Optional[datetime] = None


class StrategyPerformance(BaseModel):
    """Aggregate performance metrics for the learning loop."""
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy_pct: float = 0
    avg_confidence_correct: float = 0
    avg_confidence_incorrect: float = 0
    market_regime_accuracy: Dict[str, float] = Field(default_factory=dict)
    best_performing_mode: str = ""
    worst_performing_mode: str = ""
    strategy_adjustments: List[str] = Field(default_factory=list)


class RecommendationSet(BaseModel):
    """Collection of recommendations returned to the user."""
    query: UserQuery
    market_condition: MarketCondition = MarketCondition.SIDEWAYS
    recommendations: List[TokenRecommendation] = Field(default_factory=list)
    final_thoughts: str = ""
    tokens_analyzed: int = 0
    tokens_filtered_out: int = 0
    enabled_modes: List[DataMode] = Field(default_factory=list)
    generated_at: Optional[datetime] = None

    # AI Thought Summary for overall analysis
    overall_ai_thought: str = ""

    # Learning loop metadata
    prediction_records: List[PredictionRecord] = Field(default_factory=list)
    strategy_performance: Optional[StrategyPerformance] = None

    # Raw GPT output (for debugging / logging)
    raw_gpt_response: str = ""


# Forward references
TokenData.model_rebuild()
