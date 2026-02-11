"""
Visual Indicators System
==========================
Step 4.2 – Confidence bars, risk badges, trend arrows, and
data-freshness indicators in text, HTML, and emoji formats.

Specification:
    Confidence  9-10  ██████████  green
                7-8   ████████░░  light green
                5-6   ██████░░░░  yellow
                3-4   ████░░░░░░  orange
                1-2   ██░░░░░░░░  red

    Risk        LOW   🟢  green circle
                MEDIUM 🟡 yellow triangle
                HIGH  🔴  red octagon

    Trend       >+20%  ↑   strong increase
                +5-20% ↗   moderate increase
                ±5%    →   stable
                -5--20% ↘  moderate decrease
                <-20%  ↓   strong decrease

    Freshness   <5m    🟢  real-time
                5-30m  🟡  recent
                30-60m 🟠  aging
                >60m   🔴  stale
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════
# Confidence Visualization
# ══════════════════════════════════════════════════════════════════

def confidence_bar(score: float, width: int = 10) -> str:
    """
    Return a text progress bar for a 0-10 confidence score.

    >>> confidence_bar(8.5)
    '████████░░'
    """
    filled = round(score / 10 * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def confidence_bar_html(score: float) -> str:
    """Return an HTML snippet for a colour-coded confidence bar."""
    pct = max(0, min(100, score / 10 * 100))
    colour = _confidence_colour(score)
    return (
        f'<div class="confidence-bar" style="width:100%;background:#e0e0e0;'
        f'border-radius:4px;height:12px">'
        f'<div style="width:{pct:.0f}%;background:{colour};height:100%;'
        f'border-radius:4px"></div></div>'
    )


def confidence_label(score: float) -> str:
    """Short human-readable label for confidence."""
    if score >= 9:
        return "Very High"
    if score >= 7:
        return "High"
    if score >= 5:
        return "Moderate"
    if score >= 3:
        return "Low"
    return "Very Low"


def _confidence_colour(score: float) -> str:
    if score >= 9:
        return "#22c55e"   # green-500
    if score >= 7:
        return "#86efac"   # green-300
    if score >= 5:
        return "#facc15"   # yellow-400
    if score >= 3:
        return "#fb923c"   # orange-400
    return "#ef4444"       # red-500


# ══════════════════════════════════════════════════════════════════
# Risk Badges
# ══════════════════════════════════════════════════════════════════

def risk_badge(level: str) -> str:
    """Return an emoji risk badge."""
    return {
        "LOW": "🟢 LOW",
        "MEDIUM": "🟡 MEDIUM",
        "HIGH": "🔴 HIGH",
    }.get(level.upper(), "⚪ UNKNOWN")


def risk_badge_html(level: str) -> str:
    """Return an HTML span for a risk badge."""
    colours = {
        "LOW": ("#22c55e", "#f0fdf4"),
        "MEDIUM": ("#eab308", "#fefce8"),
        "HIGH": ("#ef4444", "#fef2f2"),
    }
    fg, bg = colours.get(level.upper(), ("#6b7280", "#f3f4f6"))
    shape = _risk_shape(level)
    return (
        f'<span class="risk-badge" style="background:{bg};color:{fg};'
        f'padding:2px 8px;border-radius:4px;font-weight:600">'
        f'{shape} {level.upper()}</span>'
    )


def _risk_shape(level: str) -> str:
    return {"LOW": "●", "MEDIUM": "▲", "HIGH": "⬣"}.get(level.upper(), "?")


# ══════════════════════════════════════════════════════════════════
# Trend Arrows
# ══════════════════════════════════════════════════════════════════

def trend_arrow(pct_change: float) -> str:
    """
    Return an arrow character and label for a percentage change.

    >>> trend_arrow(25)
    '↑ +25.0%'
    """
    if pct_change > 20:
        return f"↑ +{pct_change:.1f}%"
    if pct_change > 5:
        return f"↗ +{pct_change:.1f}%"
    if pct_change > -5:
        sign = "+" if pct_change >= 0 else ""
        return f"→ {sign}{pct_change:.1f}%"
    if pct_change > -20:
        return f"↘ {pct_change:.1f}%"
    return f"↓ {pct_change:.1f}%"


def trend_arrow_html(pct_change: float) -> str:
    """Coloured HTML span for a trend arrow."""
    arrow = trend_arrow(pct_change).split()[0]
    colour = "#22c55e" if pct_change > 5 else "#ef4444" if pct_change < -5 else "#6b7280"
    sign = "+" if pct_change >= 0 else ""
    return (
        f'<span style="color:{colour};font-weight:600">'
        f'{arrow} {sign}{pct_change:.1f}%</span>'
    )


# ══════════════════════════════════════════════════════════════════
# Data Freshness
# ══════════════════════════════════════════════════════════════════

def data_freshness_indicator(minutes: float) -> str:
    """
    Emoji + label for data age.

    >>> data_freshness_indicator(3)
    '🟢 Real-time'
    """
    if minutes < 5:
        return "🟢 Real-time"
    if minutes < 30:
        return "🟡 Recent"
    if minutes < 60:
        return "🟠 Aging"
    return "🔴 Stale"


def data_freshness_html(minutes: float) -> str:
    """HTML badge for data freshness."""
    if minutes < 5:
        colour, label = "#22c55e", "Real-time"
    elif minutes < 30:
        colour, label = "#eab308", "Recent"
    elif minutes < 60:
        colour, label = "#f97316", "Aging"
    else:
        colour, label = "#ef4444", "Stale"
    return (
        f'<span class="freshness" style="color:{colour}">'
        f'● {label} ({minutes:.0f}m ago)</span>'
    )


# ══════════════════════════════════════════════════════════════════
# Verdict Colours
# ══════════════════════════════════════════════════════════════════

def verdict_colour(verdict: str) -> str:
    """Return hex colour for a RecommendationVerdict value string."""
    return {
        "Strong Buy": "#22c55e",
        "Moderate Buy": "#86efac",
        "Cautious Buy": "#facc15",
        "Hold": "#60a5fa",
        "Watch": "#fb923c",
        "Avoid": "#ef4444",
        "Sell": "#dc2626",
    }.get(verdict, "#6b7280")


def verdict_emoji(verdict: str) -> str:
    """Return emoji for a verdict."""
    return {
        "Strong Buy": "🟢",
        "Moderate Buy": "🟡",
        "Cautious Buy": "🟠",
        "Hold": "🔵",
        "Watch": "🟠",
        "Avoid": "🔴",
        "Sell": "🔴",
    }.get(verdict, "⚪")


# ══════════════════════════════════════════════════════════════════
# Score Breakdown Sparkline
# ══════════════════════════════════════════════════════════════════

def score_sparkline(scores: dict) -> str:
    """
    Compact inline display of per-module scores.

    >>> score_sparkline({"news": 7.2, "onchain": 8.5, "technical": 6.8, "social": 7.9})
    'N:7.2 O:8.5 T:6.8 S:7.9'
    """
    abbrev = {"news": "N", "onchain": "O", "technical": "T", "social": "S"}
    parts = []
    for key, score in scores.items():
        ab = abbrev.get(key, key[0].upper())
        if score is not None:
            parts.append(f"{ab}:{score:.1f}")
    return " ".join(parts)
