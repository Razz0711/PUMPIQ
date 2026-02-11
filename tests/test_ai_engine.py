"""
Unit Tests – Confidence Scorer, Risk Rater & Conflict Detector
================================================================
Step 5.2 – Validates the core AI-engine scoring and conflict logic.
"""

from __future__ import annotations

import pytest

from src.ai_engine.models import (
    ConflictFlag,
    ConflictSeverity,
    DataMode,
    InvestmentTimeframe,
    NewsScorePayload,
    OnchainScorePayload,
    RecommendationVerdict,
    RiskLevel,
    RiskTolerance,
    SocialScorePayload,
    TechnicalScorePayload,
    TokenData,
)
from src.ai_engine.confidence_scorer import (
    ConfidenceScorer,
    EntryExitCalculator,
    RiskRater,
    confidence_risk_verdict,
)
from src.ai_engine.conflict_detector import ConflictDetector


# ══════════════════════════════════════════════════════════════════
# Confidence Scorer
# ══════════════════════════════════════════════════════════════════

class TestConfidenceScorer:
    """Test the 5-modifier confidence scoring formula."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.scorer = ConfidenceScorer()

    # ── Basic computation ─────────────────────────────────────────

    def test_base_score_is_five(self, sample_token_data, all_modes):
        """Base starts at 5; final must be ≥ 1 and ≤ 10."""
        result = self.scorer.compute(sample_token_data, all_modes, [])
        assert result.base == 5.0
        assert 1 <= result.final_score <= 10

    def test_no_conflicts_no_penalty(self, sample_token_data, all_modes):
        result = self.scorer.compute(sample_token_data, all_modes, [])
        assert result.conflict_penalty == 0.0

    def test_major_conflict_applies_penalty(self, sample_token_data, all_modes):
        conflicts = [
            ConflictFlag(
                severity=ConflictSeverity.MAJOR,
                module_a=DataMode.NEWS,
                module_b=DataMode.ONCHAIN,
                description="test",
                confidence_penalty=2.0,
            ),
        ]
        result = self.scorer.compute(sample_token_data, all_modes, conflicts)
        assert result.conflict_penalty < 0

    def test_minor_conflict_applies_penalty(self, sample_token_data, all_modes):
        conflicts = [
            ConflictFlag(
                severity=ConflictSeverity.MINOR,
                module_a=DataMode.NEWS,
                module_b=DataMode.TECHNICAL,
                description="test",
                confidence_penalty=1.0,
            ),
        ]
        result = self.scorer.compute(sample_token_data, all_modes, conflicts)
        assert result.conflict_penalty < 0

    # ── Single-mode cap at 6 ──────────────────────────────────────

    def test_single_mode_capped_at_six(self, sample_token_data):
        """If only 1 mode enabled, confidence is capped at 6."""
        result = self.scorer.compute(
            sample_token_data, [DataMode.ONCHAIN], [],
        )
        assert result.final_score <= 6.0

    # ── Data quality ──────────────────────────────────────────────

    def test_four_bullish_modes_high_quality(self, sample_token_data, all_modes):
        """All 4 modes > 5/10 → +3.0 data quality modifier."""
        result = self.scorer.compute(sample_token_data, all_modes, [])
        assert result.data_quality_modifier >= 2.0  # at least 3 agree

    # ── Signal strength ───────────────────────────────────────────

    def test_bearish_token_low_signal(self, sample_token_data_bearish, all_modes):
        result = self.scorer.compute(sample_token_data_bearish, all_modes, [])
        assert result.signal_strength_modifier <= 0

    # ── Interpretation ────────────────────────────────────────────

    def test_high_confidence_interpretation(self, sample_token_data, all_modes):
        result = self.scorer.compute(sample_token_data, all_modes, [])
        assert result.interpretation != ""
        assert isinstance(result.interpretation, str)

    # ── Clamping ──────────────────────────────────────────────────

    def test_final_score_clamped(self, sample_token_data_bearish, all_modes):
        """Even with heavy penalties, score stays ≥ 1."""
        many_conflicts = [
            ConflictFlag(
                severity=ConflictSeverity.MAJOR,
                module_a=DataMode.NEWS,
                module_b=DataMode.ONCHAIN,
                description=f"conflict-{i}",
                confidence_penalty=2.0,
            )
            for i in range(5)
        ]
        result = self.scorer.compute(
            sample_token_data_bearish, all_modes, many_conflicts,
        )
        assert result.final_score >= 1.0


# ══════════════════════════════════════════════════════════════════
# Risk Rater
# ══════════════════════════════════════════════════════════════════

class TestRiskRater:
    """Test multi-dimensional risk assessment."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rater = RiskRater()

    def test_healthy_token_moderate_risk(self, sample_token_data, all_modes):
        assessment = self.rater.assess(sample_token_data, all_modes)
        assert assessment.overall_risk in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_bearish_token_high_risk(self, sample_token_data_bearish, all_modes):
        assessment = self.rater.assess(sample_token_data_bearish, all_modes)
        assert assessment.overall_risk == RiskLevel.HIGH

    def test_low_liquidity_raises_risk(self, all_modes):
        token = TokenData(
            token_name="LowLiq", token_ticker="LLIQ",
            current_price=0.001, token_age_days=30,
            onchain=OnchainScorePayload(score=6, liquidity=5000),
        )
        assessment = self.rater.assess(token, all_modes)
        assert assessment.overall_risk == RiskLevel.HIGH

    def test_new_token_at_least_medium(self, all_modes):
        token = TokenData(
            token_name="NewToken", token_ticker="NEW",
            current_price=0.01, token_age_days=2,
            onchain=OnchainScorePayload(score=7, liquidity=100_000),
        )
        assessment = self.rater.assess(token, all_modes)
        assert assessment.overall_risk in (RiskLevel.MEDIUM, RiskLevel.HIGH)

    def test_overbought_rsi_high_volatility(self, all_modes):
        token = TokenData(
            token_name="Overbought", token_ticker="OVB",
            current_price=1.0, token_age_days=90,
            technical=TechnicalScorePayload(score=5, rsi=90),
        )
        assessment = self.rater.assess(token, all_modes)
        assert assessment.volatility_risk == RiskLevel.HIGH

    def test_position_guidance_present(self, sample_token_data, all_modes):
        assessment = self.rater.assess(sample_token_data, all_modes)
        assert assessment.position_size_guidance != ""


# ══════════════════════════════════════════════════════════════════
# Verdict Matrix
# ══════════════════════════════════════════════════════════════════

class TestVerdictMatrix:
    """Test the confidence × risk → verdict mapping."""

    def test_high_conf_low_risk_strong_buy(self):
        assert confidence_risk_verdict(9.0, RiskLevel.LOW) == RecommendationVerdict.STRONG_BUY

    def test_high_conf_medium_risk_moderate_buy(self):
        assert confidence_risk_verdict(8.5, RiskLevel.MEDIUM) == RecommendationVerdict.MODERATE_BUY

    def test_high_conf_high_risk_cautious_buy(self):
        assert confidence_risk_verdict(8.0, RiskLevel.HIGH) == RecommendationVerdict.CAUTIOUS_BUY

    def test_medium_conf_low_risk_moderate_buy(self):
        assert confidence_risk_verdict(6.0, RiskLevel.LOW) == RecommendationVerdict.MODERATE_BUY

    def test_medium_conf_medium_risk_watch(self):
        assert confidence_risk_verdict(6.0, RiskLevel.MEDIUM) == RecommendationVerdict.WATCH

    def test_medium_conf_high_risk_avoid(self):
        assert confidence_risk_verdict(6.0, RiskLevel.HIGH) == RecommendationVerdict.AVOID

    def test_low_conf_any_risk_avoid(self):
        for risk in RiskLevel:
            assert confidence_risk_verdict(3.0, risk) == RecommendationVerdict.AVOID


# ══════════════════════════════════════════════════════════════════
# Conflict Detector
# ══════════════════════════════════════════════════════════════════

class TestConflictDetector:
    """Test the 8-rule conflict detection engine."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.detector = ConflictDetector()

    # ── Rule 1: News hype but weak fundamentals ───────────────────

    def test_news_hype_weak_onchain(self):
        token = TokenData(
            token_name="HypeCoin", token_ticker="HYPE",
            news=NewsScorePayload(score=8.5),
            onchain=OnchainScorePayload(score=3.0),
        )
        conflicts = self.detector.detect(token)
        majors = [c for c in conflicts if c.severity == ConflictSeverity.MAJOR]
        assert len(majors) >= 1
        assert any("hype" in c.description.lower() or "fundamentals" in c.description.lower() for c in majors)

    # ── Rule 2: Strong chart but no community ─────────────────────

    def test_strong_chart_no_community(self):
        token = TokenData(
            token_name="ChartCoin", token_ticker="CHTC",
            technical=TechnicalScorePayload(score=9.0),
            social=SocialScorePayload(score=3.0, score_max=12),  # norm ≈ 2.5
        )
        conflicts = self.detector.detect(token)
        majors = [c for c in conflicts if c.severity == ConflictSeverity.MAJOR]
        assert len(majors) >= 1

    # ── Rule 3: Solid fundamentals but poor chart ─────────────────

    def test_solid_onchain_poor_chart(self):
        token = TokenData(
            token_name="FundaCoin", token_ticker="FNDA",
            onchain=OnchainScorePayload(score=9.0),
            technical=TechnicalScorePayload(score=3.0),
        )
        conflicts = self.detector.detect(token)
        assert any(c.severity == ConflictSeverity.MAJOR for c in conflicts)

    # ── Rule 4: Social hype, weak on-chain (pump-and-dump) ───────

    def test_social_hype_weak_onchain(self):
        token = TokenData(
            token_name="PumpCoin", token_ticker="PUMP",
            social=SocialScorePayload(score=10.0, score_max=12),  # norm ≈ 8.3
            onchain=OnchainScorePayload(score=3.0),
        )
        conflicts = self.detector.detect(token)
        assert any("pump" in c.description.lower() for c in conflicts)

    # ── Rule 5: Bearish news, bullish chart ───────────────────────

    def test_bearish_news_bullish_chart(self):
        token = TokenData(
            token_name="ChartGood", token_ticker="CHGD",
            news=NewsScorePayload(score=3.0),
            technical=TechnicalScorePayload(score=8.0),
        )
        conflicts = self.detector.detect(token)
        minors = [c for c in conflicts if c.severity == ConflictSeverity.MINOR]
        assert len(minors) >= 1

    # ── Rule 8: Social red flags ──────────────────────────────────

    def test_social_red_flags_flagged(self):
        token = TokenData(
            token_name="BotCoin", token_ticker="BOT",
            news=NewsScorePayload(score=7.0),
            onchain=OnchainScorePayload(score=7.0),
            social=SocialScorePayload(
                score=8.0, score_max=12,
                red_flags=["Suspected bot activity"],
            ),
        )
        conflicts = self.detector.detect(token)
        assert any("red flag" in c.description.lower() for c in conflicts)

    # ── No conflicts when aligned ─────────────────────────────────

    def test_no_conflicts_when_aligned(self, sample_token_data):
        conflicts = self.detector.detect(sample_token_data)
        # May have rule 7 (all moderate) but no MAJOR
        majors = [c for c in conflicts if c.severity == ConflictSeverity.MAJOR]
        assert len(majors) == 0


# ══════════════════════════════════════════════════════════════════
# Entry / Exit Calculator
# ══════════════════════════════════════════════════════════════════

class TestEntryExitCalculator:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.calc = EntryExitCalculator()

    def test_basic_plan(self, sample_token_data):
        plan = self.calc.compute(
            sample_token_data,
            RiskLevel.MEDIUM,
            InvestmentTimeframe.SWING,
            RiskTolerance.MODERATE,
        )
        assert plan.entry_low > 0
        assert plan.entry_high > plan.entry_low
        assert plan.target_1 > plan.entry_high
        assert plan.target_2 > plan.target_1
        assert plan.stop_loss < plan.entry_low

    def test_stop_loss_is_negative_pct(self, sample_token_data):
        plan = self.calc.compute(
            sample_token_data, RiskLevel.LOW,
            InvestmentTimeframe.DAY_TRADING, RiskTolerance.MODERATE,
        )
        assert plan.stop_loss_pct < 0

    def test_scalp_smaller_targets(self, sample_token_data):
        scalp_plan = self.calc.compute(
            sample_token_data, RiskLevel.MEDIUM,
            InvestmentTimeframe.SCALPING, RiskTolerance.MODERATE,
        )
        swing_plan = self.calc.compute(
            sample_token_data, RiskLevel.MEDIUM,
            InvestmentTimeframe.SWING, RiskTolerance.MODERATE,
        )
        assert scalp_plan.target_1_pct < swing_plan.target_1_pct

    def test_zero_price_returns_empty(self):
        token = TokenData(
            token_name="Zero", token_ticker="ZRO", current_price=0,
        )
        plan = self.calc.compute(
            token, RiskLevel.LOW,
            InvestmentTimeframe.SWING, RiskTolerance.MODERATE,
        )
        assert plan.entry_low == 0
        assert plan.target_1 == 0

    def test_timeframe_estimate(self, sample_token_data):
        plan = self.calc.compute(
            sample_token_data, RiskLevel.MEDIUM,
            InvestmentTimeframe.LONG_TERM, RiskTolerance.MODERATE,
        )
        assert "week" in plan.timeframe_estimate.lower()
