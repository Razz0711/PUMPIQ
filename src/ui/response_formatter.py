"""
Response Formatter
====================
Step 4.2 – Presentation layer that transforms AI engine output into
structured responses for web, mobile, and API consumers.

Formats:
    - Web:    Full recommendation cards with expandable details
    - Mobile: Condensed swipeable cards
    - API:    JSON response schema
    - Text:   Plain-text (terminal / IRC / Telegram bot)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .visual_indicators import (
    confidence_bar,
    confidence_bar_html,
    data_freshness_indicator,
    risk_badge,
    risk_badge_html,
    trend_arrow,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Format Enum
# ══════════════════════════════════════════════════════════════════

class OutputFormat:
    WEB = "web"
    MOBILE = "mobile"
    API = "api"
    TEXT = "text"


# ══════════════════════════════════════════════════════════════════
# Response Formatter
# ══════════════════════════════════════════════════════════════════

class ResponseFormatter:
    """
    Transform a RecommendationSet into display-ready output.

    Usage::

        formatter = ResponseFormatter()
        web_cards = formatter.format_web(rec_set)
        api_json  = formatter.format_api(rec_set)
        mobile    = formatter.format_mobile(rec_set)
    """

    # ── Web format (full cards) ───────────────────────────────────

    def format_web(self, rec_set) -> Dict[str, Any]:
        """
        Return a dict structure suitable for rendering HTML/React cards.

        Structure mirrors the recommendation card wireframe from spec.
        """
        cards = []
        for rec in rec_set.recommendations:
            card = self._build_web_card(rec)
            cards.append(card)

        return {
            "type": "web",
            "generated_at": _iso(rec_set.generated_at),
            "market_condition": rec_set.market_condition.value,
            "enabled_modes": [m.value for m in rec_set.enabled_modes],
            "tokens_analyzed": rec_set.tokens_analyzed,
            "tokens_filtered_out": rec_set.tokens_filtered_out,
            "cards": cards,
            "final_thoughts": rec_set.final_thoughts,
            "disclaimer": _DISCLAIMER,
        }

    def _build_web_card(self, rec) -> Dict[str, Any]:
        """Build a single recommendation card for web display."""
        return {
            "rank": rec.rank,
            "token": {
                "name": rec.token_name,
                "ticker": rec.token_ticker,
                "current_price": rec.current_price,
                "logo_url": f"/api/tokens/{rec.token_ticker.lower()}/logo",
            },
            "verdict": rec.verdict.value,
            "confidence": {
                "score": rec.confidence,
                "bar_html": confidence_bar_html(rec.confidence),
                "bar_text": confidence_bar(rec.confidence),
            },
            "risk": {
                "level": rec.risk_level.value,
                "badge_html": risk_badge_html(rec.risk_level.value),
                "badge_text": risk_badge(rec.risk_level.value),
            },
            "entry_exit": self._format_entry_exit(rec.entry_exit),
            "thesis": rec.core_thesis,
            "key_data_points": rec.key_data_points,
            "risks_and_concerns": rec.risks_and_concerns,
            "detailed_analysis": {
                "news": rec.news_analysis or None,
                "onchain": rec.onchain_analysis or None,
                "technical": rec.technical_analysis or None,
                "social": rec.social_analysis or None,
            },
            "conflicts": [
                {
                    "severity": c.severity.value,
                    "modules": f"{c.module_a.value} vs {c.module_b.value}",
                    "description": c.description,
                }
                for c in rec.conflicts
            ],
            "position_sizing": (
                rec.risk_assessment.position_size_guidance
                if rec.risk_assessment else ""
            ),
            "actions": ["add_watchlist", "set_alert", "view_chart"],
        }

    # ── Mobile format (condensed) ─────────────────────────────────

    def format_mobile(self, rec_set) -> Dict[str, Any]:
        """
        Condensed card data optimised for swipeable mobile views.
        """
        cards = []
        for rec in rec_set.recommendations:
            cards.append(self._build_mobile_card(rec))

        return {
            "type": "mobile",
            "generated_at": _iso(rec_set.generated_at),
            "market": rec_set.market_condition.value,
            "count": len(cards),
            "cards": cards,
        }

    def _build_mobile_card(self, rec) -> Dict[str, Any]:
        entry = rec.entry_exit
        return {
            "rank": rec.rank,
            "ticker": rec.token_ticker,
            "name": rec.token_name,
            "price": rec.current_price,
            "verdict": rec.verdict.value,
            "confidence": rec.confidence,
            "risk": rec.risk_level.value,
            "entry": entry.entry_low if entry else 0,
            "target_1": entry.target_1 if entry else 0,
            "target_1_pct": entry.target_1_pct if entry else 0,
            "stop_loss": entry.stop_loss if entry else 0,
            "stop_loss_pct": entry.stop_loss_pct if entry else 0,
            "thesis_short": (rec.core_thesis[:120] + "...") if len(rec.core_thesis) > 120 else rec.core_thesis,
            "quick_actions": ["watchlist", "alert"],
        }

    # ── API JSON format (full structured) ─────────────────────────

    def format_api(self, rec_set) -> Dict[str, Any]:
        """
        Canonical JSON response for the REST API.
        Matches the schema defined in Step 4.2.
        """
        recommendations = []
        for rec in rec_set.recommendations:
            recommendations.append(self._build_api_rec(rec))

        return {
            "query_timestamp": _iso(rec_set.generated_at),
            "recommendations": recommendations,
            "market_context": {
                "condition": rec_set.market_condition.value,
            },
            "metadata": {
                "modes_enabled": [m.value for m in rec_set.enabled_modes],
                "tokens_analyzed": rec_set.tokens_analyzed,
                "tokens_filtered_out": rec_set.tokens_filtered_out,
            },
        }

    def _build_api_rec(self, rec) -> Dict[str, Any]:
        entry = rec.entry_exit
        return {
            "rank": rec.rank,
            "token": {
                "name": rec.token_name,
                "ticker": rec.token_ticker,
                "current_price": rec.current_price,
            },
            "scores": {
                "overall": rec.composite_score,
                "confidence": rec.confidence,
                "risk": rec.risk_level.value,
            },
            "entry_exit": {
                "entry_min": entry.entry_low if entry else 0,
                "entry_max": entry.entry_high if entry else 0,
                "target_1": entry.target_1 if entry else 0,
                "target_1_percent": entry.target_1_pct if entry else 0,
                "target_2": entry.target_2 if entry else 0,
                "target_2_percent": entry.target_2_pct if entry else 0,
                "stop_loss": entry.stop_loss if entry else 0,
                "stop_loss_percent": entry.stop_loss_pct if entry else 0,
                "timeframe": entry.timeframe_estimate if entry else "",
            },
            "thesis": rec.core_thesis,
            "key_data_points": rec.key_data_points,
            "risks": rec.risks_and_concerns,
            "verdict": rec.verdict.value,
            "detailed_analysis": {
                "news": rec.news_analysis or None,
                "onchain": rec.onchain_analysis or None,
                "technical": rec.technical_analysis or None,
                "social": rec.social_analysis or None,
            },
        }

    # ── Text format (plain terminal) ──────────────────────────────

    def format_text(self, rec_set) -> str:
        """Plain-text rendering for CLI / Telegram / Discord bots."""
        lines: List[str] = []
        lines.append("═" * 50)
        lines.append(f"  NEXYPHER RECOMMENDATIONS")
        lines.append(f"  Market: {rec_set.market_condition.value.title()}")
        lines.append("═" * 50)

        for rec in rec_set.recommendations:
            lines.append("")
            lines.append(f"#{rec.rank} {rec.token_name} ({rec.token_ticker})")
            lines.append(f"   {rec.verdict.value} | Confidence: {confidence_bar(rec.confidence)} {rec.confidence}/10")
            lines.append(f"   Risk: {risk_badge(rec.risk_level.value)} | Price: ${rec.current_price:.8g}")

            if rec.entry_exit:
                e = rec.entry_exit
                lines.append(f"   Entry: ${e.entry_low:.8g} - ${e.entry_high:.8g}")
                lines.append(f"   T1: ${e.target_1:.8g} (+{e.target_1_pct:.0f}%) | T2: ${e.target_2:.8g} (+{e.target_2_pct:.0f}%)")
                lines.append(f"   SL: ${e.stop_loss:.8g} ({e.stop_loss_pct:.0f}%)")

            if rec.core_thesis:
                lines.append(f"   {rec.core_thesis[:150]}")

        if rec_set.final_thoughts:
            lines.append("")
            lines.append("─" * 50)
            # Truncate GPT raw response for text format
            thoughts = rec_set.final_thoughts
            if len(thoughts) > 300:
                thoughts = thoughts[:300] + "..."
            lines.append(thoughts)

        lines.append("")
        lines.append("⚠️ DYOR – This is not financial advice.")
        return "\n".join(lines)

    # ── Comparison format ─────────────────────────────────────────

    def format_comparison(self, rec_set) -> Dict[str, Any]:
        """
        Side-by-side comparison view for 2-4 tokens.
        Highlights where each token is stronger/weaker.
        """
        tokens = []
        for rec in rec_set.recommendations:
            tokens.append({
                "ticker": rec.token_ticker,
                "name": rec.token_name,
                "price": rec.current_price,
                "verdict": rec.verdict.value,
                "confidence": rec.confidence,
                "risk": rec.risk_level.value,
                "composite": rec.composite_score,
                "entry": rec.entry_exit.entry_low if rec.entry_exit else 0,
                "target_1_pct": rec.entry_exit.target_1_pct if rec.entry_exit else 0,
                "stop_loss_pct": rec.entry_exit.stop_loss_pct if rec.entry_exit else 0,
            })

        # Determine "winner" per dimension
        if len(tokens) >= 2:
            dimensions = ["confidence", "composite"]
            highlights = {}
            for dim in dimensions:
                best = max(tokens, key=lambda t: t.get(dim, 0))
                highlights[dim] = best["ticker"]
            # Lowest risk wins
            risk_ord = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            safest = min(tokens, key=lambda t: risk_ord.get(t["risk"], 1))
            highlights["safest"] = safest["ticker"]
        else:
            highlights = {}

        return {
            "type": "comparison",
            "tokens": tokens,
            "highlights": highlights,
        }

    # ── Entry/exit helper ─────────────────────────────────────────

    @staticmethod
    def _format_entry_exit(entry_exit) -> Optional[Dict[str, Any]]:
        if not entry_exit:
            return None
        return {
            "entry_low": entry_exit.entry_low,
            "entry_high": entry_exit.entry_high,
            "target_1": entry_exit.target_1,
            "target_1_pct": entry_exit.target_1_pct,
            "target_2": entry_exit.target_2,
            "target_2_pct": entry_exit.target_2_pct,
            "stop_loss": entry_exit.stop_loss,
            "stop_loss_pct": entry_exit.stop_loss_pct,
            "timeframe": entry_exit.timeframe_estimate,
        }


# ══════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════

_DISCLAIMER = (
    "⚠️ DISCLAIMER: Crypto markets are highly volatile. "
    "This analysis is algorithmic and NOT financial advice. "
    "Always do your own research (DYOR) before investing."
)


def _iso(dt: Optional[datetime]) -> str:
    return dt.isoformat() + "Z" if dt else datetime.now(timezone.utc).isoformat() + "Z"
