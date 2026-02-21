"""
Confidence Scoring & Risk Rating Engine
==========================================
Step 3.3 – Systematic confidence (0-10) and risk (Low/Medium/High) assignment.

Confidence Formula:
    Final = clamp(Base(5) + DataQuality + SignalStrength + ConflictPenalty
                       + Freshness + HistoricalAccuracy, 1, 10)

Risk Determination:
    Overall = HIGHEST risk from any enabled category.
    Then apply risk multipliers for token age, liquidity, volatility.

Confidence × Risk Matrix:
    ┌──────────────┬────────┬───────────────┬─────────────────────────┐
    │ Confidence    │ Risk   │ Verdict       │ Strength                │
    ├──────────────┼────────┼───────────────┼─────────────────────────┤
    │ High (8+)    │ Low    │ Strong Buy    │ Best setup              │
    │ High (8+)    │ Medium │ Moderate Buy  │ Good but watch risk     │
    │ High (8+)    │ High   │ Cautious Buy  │ Strong signals, risky   │
    │ Medium (5-7) │ Low    │ Moderate Buy  │ Safe, less conviction   │
    │ Medium (5-7) │ Medium │ Watch         │ Needs confirmation      │
    │ Medium (5-7) │ High   │ Avoid         │ Too risky               │
    │ Low (<5)     │ Any    │ Do Not Rec.   │ Insufficient confidence │
    └──────────────┴────────┴───────────────┴─────────────────────────┘
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from .models import (
    ConfidenceBreakdown,
    ConflictFlag,
    ConflictSeverity,
    DataMode,
    EntryExitPlan,
    InvestmentTimeframe,
    RecommendationVerdict,
    RiskAssessment,
    RiskLevel,
    RISK_ORD,
    max_risk,
    RiskTolerance,
    TokenData,
    UserConfig,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Confidence Scorer
# ══════════════════════════════════════════════════════════════════

class ConfidenceScorer:
    """
    Computes a 1-10 confidence score and an interpretation string.

    Usage::

        scorer = ConfidenceScorer()
        breakdown = scorer.compute(token_data, enabled_modes, conflicts)
    """

    # ── Public ────────────────────────────────────────────────────

    def compute(
        self,
        token: TokenData,
        enabled_modes: List[DataMode],
        conflicts: List[ConflictFlag],
        historical_accuracy: Optional[float] = None,
    ) -> ConfidenceBreakdown:
        base = 5.0

        dq = self._data_quality_modifier(token, enabled_modes)
        ss = self._signal_strength_modifier(token, enabled_modes)
        cp = self._conflict_penalty(conflicts)
        df = self._data_freshness_modifier(token)
        ha = self._historical_accuracy_modifier(historical_accuracy)

        raw = base + dq + ss + cp + df + ha
        final = max(1.0, min(10.0, round(raw, 1)))

        # Cap at 6 if only 1 mode enabled
        if len(enabled_modes) == 1:
            final = min(6.0, final)

        interpretation = self._interpret(final)

        return ConfidenceBreakdown(
            base=base,
            data_quality_modifier=dq,
            signal_strength_modifier=ss,
            conflict_penalty=cp,
            data_freshness_modifier=df,
            historical_accuracy_modifier=ha,
            raw_total=round(raw, 2),
            final_score=final,
            interpretation=interpretation,
        )

    # ── Data Quality ──────────────────────────────────────────────

    def _data_quality_modifier(
        self, token: TokenData, modes: List[DataMode]
    ) -> float:
        """
        +3 if 4 modes agree, +2 if 3, +1 if 2, +0 if 1.
        "Agreeing" means each module's normalised score > 5/10.
        """
        bullish_count = 0
        for mode in modes:
            norm = self._normalised_score(token, mode)
            if norm is not None and norm > 5:
                bullish_count += 1

        n_modes = len(modes)
        if n_modes >= 4 and bullish_count >= 4:
            return 3.0
        if n_modes >= 3 and bullish_count >= 3:
            return 2.0
        if n_modes >= 2 and bullish_count >= 2:
            return 1.0
        return 0.0

    # ── Signal Strength ───────────────────────────────────────────

    def _signal_strength_modifier(
        self, token: TokenData, modes: List[DataMode]
    ) -> float:
        """
        +2 if all scores > 8, +1 if all > 7, -1 if any < 5, else 0.
        """
        scores = [
            s for mode in modes
            if (s := self._normalised_score(token, mode)) is not None
        ]
        if not scores:
            return 0.0

        avg = sum(scores) / len(scores)
        mn = min(scores)

        if mn > 8:
            return 2.0
        if mn > 7:
            return 1.0
        if avg < 5:
            return -1.0
        return 0.0

    # ── Conflict Penalty ──────────────────────────────────────────

    @staticmethod
    def _conflict_penalty(conflicts: List[ConflictFlag]) -> float:
        penalty = 0.0
        for c in conflicts:
            if c.severity == ConflictSeverity.MAJOR:
                penalty -= 2.0
            elif c.severity == ConflictSeverity.MINOR:
                penalty -= 1.0
        return max(-4.0, penalty)  # floor

    # ── Data Freshness ────────────────────────────────────────────

    @staticmethod
    def _data_freshness_modifier(token: TokenData) -> float:
        """
        +0.5 all data < 15 min, 0 if < 60 min, -0.5 if > 60 min.
        Uses collected_at on the token or defaults to now.
        """
        if token.collected_at is None:
            return 0.0
        now = datetime.now()
        collected = token.collected_at
        # Strip tzinfo to avoid naive/aware mismatch
        if collected.tzinfo is not None:
            collected = collected.replace(tzinfo=None)
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        age_minutes = (now - collected).total_seconds() / 60
        if age_minutes < 15:
            return 0.5
        if age_minutes <= 60:
            return 0.0
        return -0.5

    # ── Historical Accuracy ───────────────────────────────────────

    @staticmethod
    def _historical_accuracy_modifier(accuracy: Optional[float]) -> float:
        """
        Historical success rate for similar recommendations.
        >70 % → +1, 50-70 % → 0, <50 % → -1.
        """
        if accuracy is None:
            return 0.0
        if accuracy > 0.70:
            return 1.0
        if accuracy < 0.50:
            return -1.0
        return 0.0

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _normalised_score(token: TokenData, mode: DataMode) -> Optional[float]:
        """Return a 0-10 normalised score for a given mode."""
        if mode == DataMode.NEWS and token.news:
            return token.news.score
        if mode == DataMode.ONCHAIN and token.onchain:
            return token.onchain.score
        if mode == DataMode.TECHNICAL and token.technical:
            return token.technical.score
        if mode == DataMode.SOCIAL and token.social:
            # Social is 0-12 → normalise to 0-10
            return token.social.score / token.social.score_max * 10
        return None

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 9:
            return "Extremely high confidence – near-perfect alignment across all signals."
        if score >= 7:
            return "High confidence – strong case with minimal concerns."
        if score >= 5:
            return "Moderate confidence – decent setup but some uncertainty remains."
        if score >= 3:
            return "Low confidence – conflicting signals or weak data."
        return "Very low confidence – insufficient evidence to recommend."


# ══════════════════════════════════════════════════════════════════
# Risk Rater
# ══════════════════════════════════════════════════════════════════

class RiskRater:
    """
    Computes a multi-dimensional risk assessment and an overall risk level.

    Usage::

        rater = RiskRater()
        assessment = rater.assess(token_data, enabled_modes)
    """

    # ── Public ────────────────────────────────────────────────────

    def assess(
        self,
        token: TokenData,
        enabled_modes: List[DataMode],
    ) -> RiskAssessment:
        assessment = RiskAssessment()
        factors: List[str] = []
        multipliers: List[str] = []

        # Per-module risk (taken from the module payload or inferred)
        if DataMode.ONCHAIN in enabled_modes and token.onchain:
            assessment.onchain_risk = self._onchain_risk(token.onchain, factors)
        if DataMode.TECHNICAL in enabled_modes and token.technical:
            assessment.technical_risk = self._technical_risk(token.technical, factors)
        if DataMode.SOCIAL in enabled_modes and token.social:
            assessment.social_risk = self._social_risk(token.social, factors)
        if DataMode.NEWS in enabled_modes and token.news:
            assessment.news_risk = self._news_risk(token.news, factors)

        # Volatility risk (estimated from technical data if available)
        assessment.volatility_risk = self._volatility_risk(token, factors)

        # Overall = highest
        all_risks = [
            assessment.onchain_risk,
            assessment.technical_risk,
            assessment.social_risk,
            assessment.news_risk,
            assessment.volatility_risk,
        ]
        assessment.overall_risk = max_risk(*all_risks)

        # Risk multipliers
        if token.token_age_days < 7:
            multipliers.append(f"New token ({token.token_age_days}d old) → +1 risk")
            assessment.overall_risk = max_risk(
                assessment.overall_risk, RiskLevel.MEDIUM
            )
        if token.onchain and token.onchain.liquidity < 20_000:
            multipliers.append(
                f"Low liquidity (${token.onchain.liquidity:,.0f}) → +1 risk"
            )
            assessment.overall_risk = max_risk(
                assessment.overall_risk, RiskLevel.HIGH
            )

        # Position size guidance
        assessment.position_size_guidance = self._position_guidance(
            assessment.overall_risk
        )
        assessment.risk_factors = factors
        assessment.risk_multipliers = multipliers
        return assessment

    # ── Per-module risk ───────────────────────────────────────────

    def _onchain_risk(self, oc, factors: List[str]) -> RiskLevel:
        level = oc.risk_level
        if oc.liquidity < 20_000:
            factors.append(f"Liquidity critically low (${oc.liquidity:,.0f})")
            level = max_risk(level, RiskLevel.HIGH)
        elif oc.liquidity < 50_000:
            factors.append(f"Liquidity moderate (${oc.liquidity:,.0f})")
            level = max_risk(level, RiskLevel.MEDIUM)
        if oc.top_10_concentration > 55:
            factors.append(f"Top 10 hold {oc.top_10_concentration:.0f}% – centralised")
            level = max_risk(level, RiskLevel.HIGH)
        elif oc.top_10_concentration > 40:
            level = max_risk(level, RiskLevel.MEDIUM)
        return level

    def _technical_risk(self, tech, factors: List[str]) -> RiskLevel:
        level = tech.risk_level
        if tech.rsi > 80:
            factors.append(f"RSI overbought ({tech.rsi:.0f})")
            level = max_risk(level, RiskLevel.HIGH)
        if tech.trend == "downtrend":
            factors.append("Active downtrend")
            level = max_risk(level, RiskLevel.MEDIUM)
        return level

    def _social_risk(self, soc, factors: List[str]) -> RiskLevel:
        level = soc.risk_level
        if soc.red_flags:
            factors.extend(soc.red_flags)
            level = max_risk(level, RiskLevel.HIGH)
        return level

    def _news_risk(self, news, factors: List[str]) -> RiskLevel:
        return news.risk_level

    def _volatility_risk(self, token: TokenData, factors: List[str]) -> RiskLevel:
        """Coarse estimate from technical data or default MEDIUM."""
        if token.technical and token.technical.rsi:
            rsi = token.technical.rsi
            if rsi > 80 or rsi < 20:
                factors.append(f"Extreme RSI ({rsi:.0f}) → high volatility likely")
                return RiskLevel.HIGH
        return RiskLevel.MEDIUM

    @staticmethod
    def _position_guidance(risk: RiskLevel) -> str:
        if risk == RiskLevel.LOW:
            return "Consider standard position size (2-5% of portfolio)."
        if risk == RiskLevel.MEDIUM:
            return "Reduce position size (1-3% of portfolio)."
        return "Small speculative position only (0.5-1% of portfolio)."


# ══════════════════════════════════════════════════════════════════
# Verdict Matrix
# ══════════════════════════════════════════════════════════════════

def confidence_risk_verdict(
    confidence: float, risk: RiskLevel
) -> RecommendationVerdict:
    """
    Map (confidence, risk) → verdict using the Step 3.3 matrix.
    """
    if confidence < 5:
        return RecommendationVerdict.AVOID  # Do not recommend

    if confidence >= 8:
        if risk == RiskLevel.LOW:
            return RecommendationVerdict.STRONG_BUY
        if risk == RiskLevel.MEDIUM:
            return RecommendationVerdict.MODERATE_BUY
        return RecommendationVerdict.CAUTIOUS_BUY  # High risk

    # Medium confidence 5-7
    if risk == RiskLevel.LOW:
        return RecommendationVerdict.MODERATE_BUY
    if risk == RiskLevel.MEDIUM:
        return RecommendationVerdict.WATCH
    return RecommendationVerdict.AVOID


# ══════════════════════════════════════════════════════════════════
# Entry / Exit Calculator
# ══════════════════════════════════════════════════════════════════

class EntryExitCalculator:
    """
    Compute entry zones, targets, and stop-loss levels from token data.
    Uses technical support/resistance when available, else percentage-based.
    """

    def compute(
        self,
        token: TokenData,
        risk: RiskLevel,
        timeframe: InvestmentTimeframe,
        risk_tolerance: RiskTolerance,
    ) -> EntryExitPlan:
        price = token.current_price
        if price <= 0:
            return EntryExitPlan()

        # Support / resistance from technical module
        support = token.technical.support if token.technical and token.technical.support > 0 else price * 0.92
        resistance = token.technical.resistance if token.technical and token.technical.resistance > 0 else price * 1.40

        # Entry zone
        entry_low = round(max(support, price * 0.95), 8)
        entry_high = round(price * 1.02, 8)

        # Targets – scale by timeframe & risk tolerance
        t1_pct, t2_pct, sl_pct = self._target_percentages(
            risk, timeframe, risk_tolerance
        )
        target_1 = round(price * (1 + t1_pct / 100), 8)
        target_2 = round(price * (1 + t2_pct / 100), 8)
        stop_loss = round(price * (1 + sl_pct / 100), 8)

        tf_est = self._timeframe_estimate(timeframe)

        return EntryExitPlan(
            entry_low=entry_low,
            entry_high=entry_high,
            target_1=target_1,
            target_1_pct=t1_pct,
            target_2=target_2,
            target_2_pct=t2_pct,
            stop_loss=stop_loss,
            stop_loss_pct=sl_pct,
            timeframe_estimate=tf_est,
        )

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _target_percentages(
        risk: RiskLevel,
        tf: InvestmentTimeframe,
        tol: RiskTolerance,
    ) -> tuple[float, float, float]:
        """
        Return (target_1_pct, target_2_pct, stop_loss_pct).
        Positive values for targets, negative for stop-loss.
        """
        # Base targets per timeframe
        base = {
            InvestmentTimeframe.SCALPING:    (5, 12, -3),
            InvestmentTimeframe.DAY_TRADING: (12, 30, -7),
            InvestmentTimeframe.SWING:       (25, 60, -15),
            InvestmentTimeframe.LONG_TERM:   (50, 120, -25),
        }
        t1, t2, sl = base.get(tf, (25, 60, -15))

        # Adjust by risk tolerance
        if tol == RiskTolerance.CONSERVATIVE:
            t1 *= 0.8; t2 *= 0.7; sl *= 0.8   # tighter
        elif tol == RiskTolerance.AGGRESSIVE:
            t1 *= 1.3; t2 *= 1.5; sl *= 1.2   # wider

        # Adjust by risk level
        if risk == RiskLevel.HIGH:
            sl *= 1.2  # wider stop
            t1 *= 1.2  # need bigger reward
            t2 *= 1.3

        return (round(t1, 1), round(t2, 1), round(sl, 1))

    @staticmethod
    def _timeframe_estimate(tf: InvestmentTimeframe) -> str:
        return {
            InvestmentTimeframe.SCALPING:    "Minutes to 1 hour",
            InvestmentTimeframe.DAY_TRADING: "1–24 hours",
            InvestmentTimeframe.SWING:       "2–14 days",
            InvestmentTimeframe.LONG_TERM:   "2–12 weeks",
        }.get(tf, "2–14 days")
