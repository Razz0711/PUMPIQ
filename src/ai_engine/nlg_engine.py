"""
Natural Language Generation Engine
=====================================
Step 3.4 – Templates and guidelines for generating clear, compelling
explanations for recommendations.  Used as a fallback when GPT-4o is
unavailable AND as structured input for GPT-4o to improve output quality.

Every recommendation answers five questions:
  1. WHY should I buy this token?   (core thesis)
  2. WHAT data supports this?       (evidence)
  3. WHEN should I enter and exit?  (timing)
  4. HOW MUCH risk am I taking?     (risk disclosure)
  5. WHAT could go wrong?           (worst-case)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import (
    ConflictFlag,
    ConfidenceBreakdown,
    DataMode,
    EntryExitPlan,
    InvestmentTimeframe,
    RecommendationVerdict,
    RiskAssessment,
    RiskLevel,
    RISK_ORD,
    RiskTolerance,
    TokenData,
    TokenRecommendation,
    RecommendationSet,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Phrase Banks (Step 3.4 – do / don't lists)
# ══════════════════════════════════════════════════════════════════

PHRASES_USE = {
    RiskLevel.LOW: [
        "Strong setup", "Solid fundamentals", "Well-positioned",
        "Textbook low-risk entry", "Multiple confirming signals",
    ],
    RiskLevel.MEDIUM: [
        "Promising but watch for", "Good setup with caveats",
        "Opportunistic", "Requires active monitoring",
        "The opportunity is real, but manage your risk",
    ],
    RiskLevel.HIGH: [
        "Speculative play", "High-risk/high-reward",
        "Gamble-sized position only", "Risk what you can afford to lose",
        "Pure speculation – tread carefully",
    ],
}

PHRASES_AVOID = [
    "guaranteed moon", "100% winner", "can't lose", "free money",
    "no risk", "easy profit", "sure thing",
]


def _sanitize_text(text: str) -> str:
    """Remove any prohibited phrases from generated text (case-insensitive)."""
    for phrase in PHRASES_AVOID:
        # Case-insensitive replacement
        import re
        text = re.sub(re.escape(phrase), "[redacted]", text, flags=re.IGNORECASE)
    return text


# ══════════════════════════════════════════════════════════════════
# Template Renderer
# ══════════════════════════════════════════════════════════════════

class NLGEngine:
    """
    Generates human-readable recommendation reports from structured data.

    Used in two modes:
    1. **Fallback** – when GPT-4o is unavailable, produces a complete
       deterministic report from templates.
    2. **Pre-processing** – builds structured summaries that are fed
       into GPT-4o as context.

    Usage::

        nlg = NLGEngine()
        text = nlg.render_recommendation_set(rec_set)
    """

    # ── Full recommendation set ───────────────────────────────────

    def render_recommendation_set(self, rec_set: RecommendationSet) -> str:
        """Render the entire recommendation set as formatted text."""
        parts: List[str] = []

        parts.append(self._header(rec_set))

        for rec in rec_set.recommendations:
            parts.append(self._render_single(rec))

        if rec_set.final_thoughts:
            parts.append(self._section("FINAL THOUGHTS", rec_set.final_thoughts))

        parts.append(self._footer(rec_set))
        return _sanitize_text("\n".join(parts))

    # ── Single recommendation ─────────────────────────────────────

    def render_single(self, rec: TokenRecommendation) -> str:
        return _sanitize_text(self._render_single(rec))

    # ── Core thesis builder ───────────────────────────────────────

    def build_core_thesis(self, token: TokenData, risk: RiskLevel) -> str:
        """
        Template: "I'm recommending {TOKEN} because {primary reason}.
        The case rests on {key 1}, {key 2}, and {key 3}.
        However, {main risk}, so I rate this as {risk} with {conf} confidence."
        """
        reasons: List[str] = []
        concerns: List[str] = []

        if token.news and token.news.score > 6:
            reasons.append(
                f"positive news sentiment ({token.news.score:.1f}/10)"
            )
        if token.onchain and token.onchain.score > 6:
            reasons.append(
                f"healthy on-chain fundamentals ({token.onchain.score:.1f}/10)"
            )
        if token.technical and token.technical.score > 6:
            reasons.append(
                f"bullish technical setup ({token.technical.score:.1f}/10)"
            )
        if token.social and (token.social.score / token.social.score_max * 10) > 6:
            reasons.append(
                f"strong social buzz ({token.social.score:.1f}/{token.social.score_max:.0f})"
            )

        if token.onchain and token.onchain.liquidity < 30_000:
            concerns.append("limited liquidity")
        if token.social and token.social.red_flags:
            concerns.append("social red flags")
        for c in token.conflicts:
            concerns.append(c.description[:60])

        if not reasons:
            reasons.append("mixed signals across available data")

        primary = reasons[0]
        supporting = ", ".join(reasons[1:]) if len(reasons) > 1 else "additional confirming data"
        concern_text = concerns[0] if concerns else "standard market volatility"

        tone = PHRASES_USE.get(risk, PHRASES_USE[RiskLevel.MEDIUM])[0]

        return (
            f"I'm recommending {token.token_ticker} because {primary}. "
            f"The case rests on {supporting}. "
            f"However, {concern_text}, so I rate this as "
            f"{risk.value} risk. {tone}."
        )

    # ── Evidence bullets ──────────────────────────────────────────

    def build_evidence_bullets(
        self, token: TokenData, modes: List[DataMode]
    ) -> List[str]:
        """Build key data-point bullets for each enabled module."""
        bullets: List[str] = []

        if DataMode.NEWS in modes and token.news:
            n = token.news
            if n.key_headlines:
                bullets.append(f"📰 News: {n.key_headlines[0]}")
            bullets.append(f"📰 News sentiment: {n.score:.1f}/10")

        if DataMode.ONCHAIN in modes and token.onchain:
            o = token.onchain
            bullets.append(
                f"⛓️ {o.holder_count:,} holders "
                f"({o.holder_growth_24h:+.1f}% in 24h)"
            )
            bullets.append(
                f"⛓️ Liquidity: ${o.liquidity:,.0f} | "
                f"Top 10 hold {o.top_10_concentration:.0f}%"
            )

        if DataMode.TECHNICAL in modes and token.technical:
            t = token.technical
            bullets.append(
                f"📊 {t.trend.title()} trend, RSI {t.rsi:.0f} ({t.rsi_label})"
            )
            if t.pattern != "None":
                bullets.append(f"📊 Pattern: {t.pattern}")

        if DataMode.SOCIAL in modes and token.social:
            s = token.social
            bullets.append(
                f"💬 {s.mention_count_24h:,} mentions, "
                f"{s.influencer_count} influencers"
            )
            if s.trending_status:
                bullets.append(f"💬 Trending: {s.trending_status}")

        return bullets

    # ── Risk disclosure ───────────────────────────────────────────

    def build_risk_disclosure(
        self, token: TokenData, assessment: Optional[RiskAssessment]
    ) -> List[str]:
        """
        Template: "⚠️ WATCH OUT FOR: {risk1}, {risk2}, {risk3}"
        """
        warnings: List[str] = []

        if assessment:
            for f in assessment.risk_factors[:3]:
                warnings.append(f)
            for m in assessment.risk_multipliers[:2]:
                warnings.append(m)

        if token.social and token.social.red_flags:
            for rf in token.social.red_flags[:2]:
                warnings.append(rf)

        for c in token.conflicts[:2]:
            warnings.append(c.description[:100])

        if not warnings:
            warnings.append("Standard crypto market volatility")

        return warnings

    # ── Entry/exit explanation ────────────────────────────────────

    def explain_entry_exit(self, plan: EntryExitPlan, token: TokenData) -> str:
        """
        Don't just give numbers – explain the logic.
        """
        lines: List[str] = []

        # Entry
        entry_reason = "near current support level" if (
            token.technical and token.technical.support > 0
        ) else "close to current price with a slight pullback buffer"
        lines.append(
            f"Entry: ${plan.entry_low:.8g} – ${plan.entry_high:.8g}\n"
            f"  Why: {entry_reason}"
        )

        # Target 1
        lines.append(
            f"Target 1: ${plan.target_1:.8g} (+{plan.target_1_pct:.0f}%)\n"
            f"  Why: First resistance zone / typical momentum gain"
        )
        # Target 2
        lines.append(
            f"Target 2: ${plan.target_2:.8g} (+{plan.target_2_pct:.0f}%)\n"
            f"  Why: Extended target if strong momentum continues"
        )
        # Stop loss
        lines.append(
            f"Stop Loss: ${plan.stop_loss:.8g} ({plan.stop_loss_pct:.0f}%)\n"
            f"  Why: Below key support – invalidates the thesis"
        )
        # Timeframe
        lines.append(f"Timeframe: {plan.timeframe_estimate}")

        return "\n".join(lines)

    # ── Comparative summary ───────────────────────────────────────

    def comparative_summary(self, recs: List[TokenRecommendation]) -> str:
        """
        "Comparing my top N picks: #1 is SAFEST, #2 is BEST R/R, #3 is HIGHEST UPSIDE"
        """
        if not recs:
            return ""

        lines = ["Comparing my top picks:"]
        labels = ["SAFEST play", "BEST RISK/REWARD", "HIGHEST UPSIDE"]

        sorted_recs = sorted(recs, key=lambda r: RISK_ORD.get(r.risk_level, 1))

        for i, rec in enumerate(sorted_recs[:3]):
            label = labels[i] if i < len(labels) else f"Pick #{i+1}"
            lines.append(
                f"  #{i+1} {rec.token_ticker} is the {label} "
                f"– {rec.verdict.value}, confidence {rec.confidence:.1f}/10."
            )

        return "\n".join(lines)

    # ── Conflict explanation ──────────────────────────────────────

    def explain_conflicts(self, conflicts: List[ConflictFlag]) -> str:
        if not conflicts:
            return ""

        parts = ["⚠️ MIXED SIGNALS DETECTED:"]
        for c in conflicts:
            parts.append(
                f"  • [{c.module_a.value.title()} vs {c.module_b.value.title()}] "
                f"{c.description}"
            )
        parts.append(
            "My assessment: confidence has been adjusted downward to "
            "reflect these disagreements."
        )
        return "\n".join(parts)

    # ══════════════════════════════════════════════════════════════
    # Internal renderers
    # ══════════════════════════════════════════════════════════════

    def _render_single(self, rec: TokenRecommendation) -> str:
        verdict_emoji = {
            RecommendationVerdict.STRONG_BUY: "🟢",
            RecommendationVerdict.MODERATE_BUY: "🟡",
            RecommendationVerdict.CAUTIOUS_BUY: "🟠",
            RecommendationVerdict.HOLD: "🔵",
            RecommendationVerdict.WATCH: "🟠",
            RecommendationVerdict.AVOID: "🔴",
            RecommendationVerdict.SELL: "🔴",
        }
        risk_emoji = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🔴",
        }

        v_em = verdict_emoji.get(rec.verdict, "⚪")
        r_em = risk_emoji.get(rec.risk_level, "⚪")

        lines: List[str] = [
            "",
            "═" * 62,
            f"🏆 RECOMMENDATION #{rec.rank}: {rec.token_name} ({rec.token_ticker})",
            f"   {v_em} {rec.verdict.value} | Confidence: {rec.confidence:.1f}/10 "
            f"| {r_em} Risk: {rec.risk_level.value}",
            f"   Current Price: ${rec.current_price:.8g}",
        ]

        # Entry / exit
        if rec.entry_exit:
            e = rec.entry_exit
            lines.extend([
                f"   Entry Zone: ${e.entry_low:.8g} – ${e.entry_high:.8g}",
                f"   Target 1:   ${e.target_1:.8g} (+{e.target_1_pct:.0f}%)",
                f"   Target 2:   ${e.target_2:.8g} (+{e.target_2_pct:.0f}%)",
                f"   Stop Loss:  ${e.stop_loss:.8g} ({e.stop_loss_pct:.0f}%)",
                f"   Timeframe:  {e.timeframe_estimate}",
            ])

        # Core thesis
        if rec.core_thesis:
            lines.extend(["", "🎯 THE CASE FOR THIS PICK:", f"   {rec.core_thesis}"])

        # Key data points
        if rec.key_data_points:
            lines.append("\n📊 KEY DATA POINTS:")
            for dp in rec.key_data_points:
                lines.append(f"   • {dp}")

        # Module analyses
        for label, text in [
            ("📰 News", rec.news_analysis),
            ("⛓️ On-Chain", rec.onchain_analysis),
            ("📊 Technical", rec.technical_analysis),
            ("💬 Social", rec.social_analysis),
        ]:
            if text:
                lines.append(f"\n{label}:\n   {text}")

        # Conflicts
        if rec.conflicts:
            lines.append("\n⚡ CONFLICTING SIGNALS:")
            for c in rec.conflicts:
                lines.append(f"   • {c.description}")

        # Risks
        if rec.risks_and_concerns:
            lines.append("\n⚠️ WHAT COULD GO WRONG:")
            for r in rec.risks_and_concerns:
                lines.append(f"   • {r}")

        # Position sizing
        if rec.risk_assessment and rec.risk_assessment.position_size_guidance:
            lines.append(f"\n💰 POSITION SIZE: {rec.risk_assessment.position_size_guidance}")

        lines.append("═" * 62)
        return "\n".join(lines)

    # ── Header / footer ───────────────────────────────────────────

    @staticmethod
    def _header(rs: RecommendationSet) -> str:
        modes_str = ", ".join(m.value.title() for m in rs.enabled_modes)
        return (
            f"{'═' * 62}\n"
            f"  NEXYPHER ANALYSIS REPORT\n"
            f"  Generated: {rs.generated_at.strftime('%Y-%m-%d %H:%M UTC') if rs.generated_at else 'N/A'}\n"
            f"  Market: {rs.market_condition.value.title()}\n"
            f"  Active Modes: {modes_str}\n"
            f"  Query: \"{rs.query.raw_query}\"\n"
            f"  Tokens Analyzed: {rs.tokens_analyzed} | "
            f"Filtered Out: {rs.tokens_filtered_out}\n"
            f"{'═' * 62}"
        )

    @staticmethod
    def _footer(rs: RecommendationSet) -> str:
        return (
            f"\n{'═' * 62}\n"
            f"⚠️ DISCLAIMER: Crypto markets are highly volatile.\n"
            f"This analysis is algorithmic and NOT financial advice.\n"
            f"Always do your own research (DYOR) before investing.\n"
            f"{'═' * 62}"
        )

    @staticmethod
    def _section(title: str, body: str) -> str:
        return f"\n{'─' * 62}\n{title}\n{'─' * 62}\n{body}"
