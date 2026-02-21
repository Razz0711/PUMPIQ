"""
Social-Only Mode Recommendation Engine
=========================================
Generates trading recommendations based exclusively on social signals.

Recommendation Thresholds:
╔════════════════════╦═══════════════╦═════════════════════════════════╗
║ Recommendation     ║ Score Range   ║ Condition                       ║
╠════════════════════╬═══════════════╬═════════════════════════════════╣
║ 🟢 STRONG BUY     ║ > 9/12        ║ No red flags                    ║
║ 🟡 MODERATE BUY   ║ 6–9 / 12      ║ ≤1 minor red flag               ║
║ 🟠 WATCH          ║ 4–6 / 12      ║ Neutral / developing            ║
║ 🔴 AVOID / SELL   ║ < 4 / 12      ║ OR any major red flag present   ║
╚════════════════════╩═══════════════╩═════════════════════════════════╝

Entry / Exit Logic:
- ENTRY when social score crosses 7/12 with positive momentum
- Target 1: +25–40%   (partial take-profit)
- Target 2: +60–100%  (full take-profit)
- Stop Loss: -15–20%  (exit on loss)
- EXIT when social score drops below 6/12

Risk Assessment:
- LOW:    Score 9+, 0 red flags, high organic, strong consensus
- MEDIUM: Score 6-9, ≤1 red flag, moderate organic
- HIGH:   Score 4-6 or 1+ red flags or low organic

Output matches the user's exact template format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.ai_engine.models import RiskLevel
from .social_aggregator import (
    OverallTone,
    SocialScoreReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Recommendation(str, Enum):
    STRONG_BUY = "STRONG BUY"
    MODERATE_BUY = "MODERATE BUY"
    WATCH = "WATCH"
    AVOID = "AVOID"
    SELL = "SELL"


class SignalAction(str, Enum):
    ENTRY = "ENTRY"
    HOLD = "HOLD"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    FULL_EXIT = "FULL_EXIT"
    STOP_LOSS = "STOP_LOSS"
    NO_ACTION = "NO_ACTION"


# ---------------------------------------------------------------------------
# Position Tracking
# ---------------------------------------------------------------------------

@dataclass
class EntryExitTargets:
    """Computed entry/exit targets for a position."""
    entry_price: Optional[float] = None
    target_1_pct: float = 30.0       # +25–40%, default 30%
    target_2_pct: float = 80.0       # +60–100%, default 80%
    stop_loss_pct: float = -17.5     # -15–20%, default -17.5%

    target_1_price: Optional[float] = None
    target_2_price: Optional[float] = None
    stop_loss_price: Optional[float] = None

    def compute_prices(self, entry_price: float) -> None:
        """Calculate target prices from entry."""
        self.entry_price = entry_price
        self.target_1_price = round(entry_price * (1 + self.target_1_pct / 100), 8)
        self.target_2_price = round(entry_price * (1 + self.target_2_pct / 100), 8)
        self.stop_loss_price = round(entry_price * (1 + self.stop_loss_pct / 100), 8)


@dataclass
class PositionState:
    """Tracks the current state of a position for a token."""
    token_ticker: str
    is_active: bool = False
    entry_score: float = 0.0         # Score when position was entered
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    current_score: float = 0.0
    targets: Optional[EntryExitTargets] = None
    target_1_hit: bool = False
    target_2_hit: bool = False
    stop_loss_hit: bool = False


# ---------------------------------------------------------------------------
# Recommendation Output
# ---------------------------------------------------------------------------

@dataclass
class SocialRecommendation:
    """
    Complete recommendation output for SOCIAL_ONLY mode.

    Matches the exact output template from the design spec.
    """
    # Token identification
    token_ticker: str
    token_name: str = ""

    # Core recommendation
    recommendation: Recommendation = Recommendation.WATCH
    risk_level: RiskLevel = RiskLevel.MEDIUM
    signal_action: SignalAction = SignalAction.NO_ACTION

    # Social Score
    social_score: float = 0.0
    social_score_max: float = 12.0

    # Score breakdown
    mention_volume_score: float = 0.0
    mention_volume_max: float = 3.0
    sentiment_quality_score: float = 0.0
    sentiment_quality_max: float = 3.0
    influencer_signal_score: float = 0.0
    influencer_signal_max: float = 2.0
    organic_activity_score: float = 0.0
    organic_activity_max: float = 2.0
    trend_momentum_score: float = 0.0
    trend_momentum_max: float = 2.0

    # Red flags
    red_flags: List[str] = field(default_factory=list)
    red_flag_penalty: float = 0.0

    # Entry / Exit targets
    entry_exit: Optional[EntryExitTargets] = None

    # Platform summary
    platform_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Confidence
    confidence: float = 0.0
    overall_tone: str = "neutral"

    # Reasoning
    reasoning: str = ""
    key_signals: List[str] = field(default_factory=list)

    # Metadata
    generated_at: Optional[datetime] = None
    data_points: int = 0
    analysis_window: str = "24h"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SocialRecommendationEngine:
    """
    Generates trading recommendations from social scores.

    Usage::

        engine = SocialRecommendationEngine()
        recommendation = engine.generate_recommendation(
            score_report=social_score_report,
            token_ticker="BONK",
            current_price=0.00002,
        )
        print(recommendation.recommendation)  # "STRONG BUY"
        print(engine.format_output(recommendation))
    """

    def __init__(self):
        # Track active positions for entry/exit logic
        self._positions: Dict[str, PositionState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_recommendation(
        self,
        score_report: SocialScoreReport,
        token_ticker: str,
        token_name: str = "",
        current_price: Optional[float] = None,
        previous_score: Optional[float] = None,
    ) -> SocialRecommendation:
        """
        Generate a complete social-only recommendation.

        Args:
            score_report: Output from SocialAggregator
            token_ticker: Token symbol
            token_name: Full token name
            current_price: Current token price (for target calculation)
            previous_score: Previous social score (for signal detection)

        Returns:
            SocialRecommendation with all fields populated
        """
        score = score_report.final_score
        red_flags = score_report.red_flags
        has_major_red_flag = self._has_major_red_flag(score_report)

        # 1. Determine recommendation
        recommendation = self._classify_recommendation(
            score, red_flags, has_major_red_flag
        )

        # 2. Determine risk level
        risk = self._assess_risk(score, score_report)

        # 3. Determine signal action (entry / hold / exit)
        action = self._determine_action(
            token_ticker, score, previous_score, has_major_red_flag
        )

        # 4. Calculate entry/exit targets
        targets = None
        if action in (SignalAction.ENTRY,) and current_price:
            targets = self._compute_targets(score, risk)
            targets.compute_prices(current_price)

            # Record position
            self._positions[token_ticker] = PositionState(
                token_ticker=token_ticker,
                is_active=True,
                entry_score=score,
                entry_price=current_price,
                entry_time=datetime.now(timezone.utc),
                current_score=score,
                targets=targets,
            )
        elif token_ticker in self._positions:
            pos = self._positions[token_ticker]
            pos.current_score = score
            targets = pos.targets

        # 5. Extract score breakdown
        cat_map = {cs.category.value: cs for cs in score_report.category_scores}

        # 6. Generate reasoning
        reasoning = self._generate_reasoning(recommendation, score_report, action)
        key_signals = self._extract_key_signals(score_report)

        # 7. Build platform breakdown
        platform_breakdown: Dict[str, Dict[str, Any]] = {}
        for ps in score_report.platform_summaries:
            platform_breakdown[ps.platform] = {
                "mentions": ps.mentions,
                "avg_sentiment": ps.avg_sentiment,
                "weighted_sentiment": ps.weighted_sentiment,
                "influencer_mentions": ps.influencer_mentions,
                "engagement": ps.engagement_total,
            }

        return SocialRecommendation(
            token_ticker=token_ticker,
            token_name=token_name,
            recommendation=recommendation,
            risk_level=risk,
            signal_action=action,
            social_score=score,
            mention_volume_score=cat_map.get("mention_volume", type("", (), {"score": 0})).score,
            sentiment_quality_score=cat_map.get("sentiment_quality", type("", (), {"score": 0})).score,
            influencer_signal_score=cat_map.get("influencer_signal", type("", (), {"score": 0})).score,
            organic_activity_score=cat_map.get("organic_activity", type("", (), {"score": 0})).score,
            trend_momentum_score=cat_map.get("trend_momentum", type("", (), {"score": 0})).score,
            red_flags=red_flags,
            red_flag_penalty=score_report.penalty_total,
            entry_exit=targets,
            platform_breakdown=platform_breakdown,
            confidence=score_report.confidence,
            overall_tone=score_report.overall_tone.value,
            reasoning=reasoning,
            key_signals=key_signals,
            generated_at=datetime.now(timezone.utc),
            data_points=score_report.total_data_points,
        )

    def update_position(
        self,
        token_ticker: str,
        current_price: float,
        current_score: float,
    ) -> SignalAction:
        """
        Update an active position and determine if any exit condition is met.

        Returns:
            SignalAction indicating what to do
        """
        if token_ticker not in self._positions:
            return SignalAction.NO_ACTION

        pos = self._positions[token_ticker]
        if not pos.is_active or not pos.targets or not pos.entry_price:
            return SignalAction.NO_ACTION

        pos.current_score = current_score

        # Check stop loss
        if current_price <= pos.targets.stop_loss_price:
            pos.stop_loss_hit = True
            pos.is_active = False
            return SignalAction.STOP_LOSS

        # Check social score exit
        if current_score < 6.0:
            pos.is_active = False
            return SignalAction.FULL_EXIT

        # Check target 2
        if not pos.target_2_hit and current_price >= pos.targets.target_2_price:
            pos.target_2_hit = True
            pos.is_active = False
            return SignalAction.FULL_EXIT

        # Check target 1
        if not pos.target_1_hit and current_price >= pos.targets.target_1_price:
            pos.target_1_hit = True
            return SignalAction.PARTIAL_EXIT

        return SignalAction.HOLD

    def format_output(self, rec: SocialRecommendation) -> str:
        """
        Format recommendation into the exact output template.

        Returns a human-readable report string.
        """
        # Emoji mapping
        rec_emoji = {
            Recommendation.STRONG_BUY: "🟢",
            Recommendation.MODERATE_BUY: "🟡",
            Recommendation.WATCH: "🟠",
            Recommendation.AVOID: "🔴",
            Recommendation.SELL: "🔴",
        }
        risk_emoji = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🔴",
        }

        emoji = rec_emoji.get(rec.recommendation, "⚪")
        r_emoji = risk_emoji.get(rec.risk_level, "⚪")

        lines = [
            "═" * 60,
            f"  NEXYPHER SOCIAL-ONLY ANALYSIS",
            f"  Token: ${rec.token_ticker}" + (f" ({rec.token_name})" if rec.token_name else ""),
            f"  Generated: {rec.generated_at.strftime('%Y-%m-%d %H:%M UTC') if rec.generated_at else 'N/A'}",
            "═" * 60,
            "",
            f"  {emoji}  RECOMMENDATION: {rec.recommendation.value}",
            f"  {r_emoji}  RISK LEVEL: {rec.risk_level.value}",
            f"  📊 CONFIDENCE: {rec.confidence:.0%}",
            "",
            "─" * 60,
            "  SOCIAL SCORE BREAKDOWN",
            "─" * 60,
            f"  Overall Score:     {rec.social_score:.1f} / {rec.social_score_max:.0f}",
            "",
            f"  Mention Volume:    {rec.mention_volume_score:.1f} / {rec.mention_volume_max:.0f}",
            f"  Sentiment Quality: {rec.sentiment_quality_score:.1f} / {rec.sentiment_quality_max:.0f}",
            f"  Influencer Signal: {rec.influencer_signal_score:.1f} / {rec.influencer_signal_max:.0f}",
            f"  Organic Activity:  {rec.organic_activity_score:.1f} / {rec.organic_activity_max:.0f}",
            f"  Trend Momentum:    {rec.trend_momentum_score:+.1f} / {rec.trend_momentum_max:.0f}",
        ]

        if rec.red_flags:
            lines.extend([
                "",
                f"  Red Flag Penalty:  -{rec.red_flag_penalty:.1f} pts",
            ])

        # Platform breakdown
        if rec.platform_breakdown:
            lines.extend([
                "",
                "─" * 60,
                "  PLATFORM BREAKDOWN",
                "─" * 60,
            ])
            for plat, data in rec.platform_breakdown.items():
                lines.append(
                    f"  {plat}: {data.get('mentions', 0)} mentions, "
                    f"sentiment {data.get('avg_sentiment', 0):+.1f}, "
                    f"engagement {data.get('engagement', 0):,}"
                )

        # Entry/exit targets
        if rec.entry_exit and rec.entry_exit.entry_price:
            lines.extend([
                "",
                "─" * 60,
                "  ENTRY / EXIT TARGETS",
                "─" * 60,
                f"  Entry Price:   ${rec.entry_exit.entry_price:.8f}",
                f"  Target 1:      ${rec.entry_exit.target_1_price:.8f} (+{rec.entry_exit.target_1_pct:.0f}%)",
                f"  Target 2:      ${rec.entry_exit.target_2_price:.8f} (+{rec.entry_exit.target_2_pct:.0f}%)",
                f"  Stop Loss:     ${rec.entry_exit.stop_loss_price:.8f} ({rec.entry_exit.stop_loss_pct:.0f}%)",
            ])

        # Red flags
        if rec.red_flags:
            lines.extend([
                "",
                "─" * 60,
                "  ⚠️  RED FLAGS",
                "─" * 60,
            ])
            for flag in rec.red_flags:
                lines.append(f"  • {flag}")

        # Key signals
        if rec.key_signals:
            lines.extend([
                "",
                "─" * 60,
                "  KEY SIGNALS",
                "─" * 60,
            ])
            for sig in rec.key_signals:
                lines.append(f"  ✦ {sig}")

        # Reasoning
        if rec.reasoning:
            lines.extend([
                "",
                "─" * 60,
                "  ANALYSIS",
                "─" * 60,
                f"  {rec.reasoning}",
            ])

        lines.extend([
            "",
            "─" * 60,
            f"  Overall Tone: {rec.overall_tone.upper()}",
            f"  Data Points: {rec.data_points:,}",
            f"  Signal Action: {rec.signal_action.value}",
            "═" * 60,
            "  ⚠️  DISCLAIMER: Social sentiment is volatile.",
            "  This is NOT financial advice. DYOR.",
            "═" * 60,
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal – Classification
    # ------------------------------------------------------------------

    def _classify_recommendation(
        self,
        score: float,
        red_flags: List[str],
        has_major_red_flag: bool,
    ) -> Recommendation:
        """Map score + flags to recommendation tier."""
        # Any major red flag → AVOID regardless of score
        if has_major_red_flag:
            return Recommendation.AVOID

        if score > 9:
            return Recommendation.STRONG_BUY
        elif score >= 6:
            # Moderate buy if ≤1 minor red flag
            if len(red_flags) <= 1:
                return Recommendation.MODERATE_BUY
            else:
                return Recommendation.WATCH
        elif score >= 4:
            return Recommendation.WATCH
        else:
            return Recommendation.AVOID

    def _has_major_red_flag(self, report: SocialScoreReport) -> bool:
        """Check if any major red flag is present."""
        major_keywords = [
            "pump group", "coordinated shilling",
            "fake engagement", "honeypot", "rug",
        ]
        for flag in report.red_flags:
            if any(kw in flag.lower() for kw in major_keywords):
                return True
        return report.penalty_total >= 3.0

    # ------------------------------------------------------------------
    # Internal – Risk Assessment
    # ------------------------------------------------------------------

    def _assess_risk(
        self, score: float, report: SocialScoreReport
    ) -> RiskLevel:
        """Determine risk level."""
        # LOW: Score 9+, 0 red flags, high confidence
        if score >= 9 and report.red_flag_count == 0 and report.confidence >= 0.6:
            return RiskLevel.LOW

        # HIGH: Score < 6 or ≥2 red flags or low confidence
        if score < 6 or report.red_flag_count >= 2 or report.confidence < 0.3:
            return RiskLevel.HIGH

        return RiskLevel.MEDIUM

    # ------------------------------------------------------------------
    # Internal – Entry / Exit Logic
    # ------------------------------------------------------------------

    def _determine_action(
        self,
        token_ticker: str,
        score: float,
        previous_score: Optional[float],
        has_major_red_flag: bool,
    ) -> SignalAction:
        """Determine what action to take based on score changes."""
        # Any major red flag → no entry / exit if active
        if has_major_red_flag:
            if token_ticker in self._positions and self._positions[token_ticker].is_active:
                return SignalAction.FULL_EXIT
            return SignalAction.NO_ACTION

        # Check if we have an active position
        if token_ticker in self._positions:
            pos = self._positions[token_ticker]
            if pos.is_active:
                # Score dropped below 6 → exit
                if score < 6.0:
                    pos.is_active = False
                    return SignalAction.FULL_EXIT
                return SignalAction.HOLD

        # Entry condition: score crosses 7/12 with upward momentum
        if score >= 7.0:
            if previous_score is not None:
                if score > previous_score:
                    return SignalAction.ENTRY
                else:
                    return SignalAction.NO_ACTION  # Score declining; don't enter
            else:
                return SignalAction.ENTRY  # First observation at 7+ → entry

        return SignalAction.NO_ACTION

    def _compute_targets(
        self, score: float, risk: RiskLevel
    ) -> EntryExitTargets:
        """
        Compute entry/exit target percentages based on score and risk.

        Higher score / lower risk = more aggressive targets.
        """
        if score >= 10 and risk == RiskLevel.LOW:
            return EntryExitTargets(
                target_1_pct=40.0,
                target_2_pct=100.0,
                stop_loss_pct=-15.0,
            )
        elif score >= 8:
            return EntryExitTargets(
                target_1_pct=35.0,
                target_2_pct=80.0,
                stop_loss_pct=-17.5,
            )
        elif score >= 7:
            return EntryExitTargets(
                target_1_pct=25.0,
                target_2_pct=60.0,
                stop_loss_pct=-20.0,
            )
        else:
            return EntryExitTargets(
                target_1_pct=20.0,
                target_2_pct=40.0,
                stop_loss_pct=-20.0,
            )

    # ------------------------------------------------------------------
    # Internal – Reasoning
    # ------------------------------------------------------------------

    def _generate_reasoning(
        self,
        rec: Recommendation,
        report: SocialScoreReport,
        action: SignalAction,
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        score = report.final_score
        parts: List[str] = []

        if rec == Recommendation.STRONG_BUY:
            parts.append(
                f"Social score of {score:.1f}/12 indicates exceptionally strong "
                f"community interest with positive sentiment across platforms."
            )
        elif rec == Recommendation.MODERATE_BUY:
            parts.append(
                f"Social score of {score:.1f}/12 shows solid community interest. "
                f"Sentiment is predominantly positive."
            )
        elif rec == Recommendation.WATCH:
            parts.append(
                f"Social score of {score:.1f}/12 indicates developing interest. "
                f"Monitor for score improvement above 7."
            )
        elif rec in (Recommendation.AVOID, Recommendation.SELL):
            parts.append(
                f"Social score of {score:.1f}/12 is below threshold. "
            )
            if report.red_flag_count > 0:
                parts.append(
                    f"{report.red_flag_count} red flag(s) detected, "
                    f"suggesting artificial or unhealthy activity."
                )

        if action == SignalAction.ENTRY:
            parts.append("Score crossed entry threshold of 7/12 with positive momentum.")
        elif action == SignalAction.FULL_EXIT:
            parts.append("Exit triggered — score dropped below 6/12 or major red flag.")
        elif action == SignalAction.HOLD:
            parts.append("Holding existing position — score remains above 6/12.")

        return " ".join(parts)

    def _extract_key_signals(self, report: SocialScoreReport) -> List[str]:
        """Extract the most important signals for the summary."""
        signals: List[str] = []

        for cs in report.category_scores:
            pct = (cs.score / cs.max_score * 100) if cs.max_score > 0 else 0
            if pct >= 80:
                signals.append(f"Strong {cs.category.value.replace('_', ' ')}: {cs.details}")
            elif pct <= 25 and cs.max_score > 0:
                signals.append(f"Weak {cs.category.value.replace('_', ' ')}: {cs.details}")

        if report.red_flag_count > 0:
            signals.append(f"{report.red_flag_count} red flag(s) detected")

        if report.confidence >= 0.7:
            signals.append(f"High confidence ({report.confidence:.0%}) based on {report.total_data_points:,} data points")
        elif report.confidence < 0.3:
            signals.append(f"Low confidence ({report.confidence:.0%}) — limited data")

        return signals[:5]  # Top 5 signals
