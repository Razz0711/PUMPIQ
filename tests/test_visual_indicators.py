"""
Unit Tests – Visual Indicators
================================
Step 5.2 – Validates text, HTML, and emoji formatting for
confidence, risk, trend, and freshness indicators.
"""

from __future__ import annotations

import pytest

from src.ui.visual_indicators import (
    confidence_bar,
    confidence_bar_html,
    confidence_label,
    data_freshness_indicator,
    risk_badge,
    risk_badge_html,
    score_sparkline,
    trend_arrow,
    verdict_colour,
    verdict_emoji,
)


# ══════════════════════════════════════════════════════════════════
# Confidence Bar
# ══════════════════════════════════════════════════════════════════

class TestConfidenceBar:

    def test_full_bar(self):
        bar = confidence_bar(10.0)
        assert bar == "██████████"

    def test_empty_bar(self):
        bar = confidence_bar(0.0)
        assert bar == "░░░░░░░░░░"

    def test_half_bar(self):
        bar = confidence_bar(5.0)
        assert len(bar) == 10
        assert bar.count("█") == 5
        assert bar.count("░") == 5

    def test_custom_width(self):
        bar = confidence_bar(5.0, width=20)
        assert len(bar) == 20

    def test_clamps_above_10(self):
        bar = confidence_bar(15.0)
        assert bar == "██████████"

    def test_clamps_below_0(self):
        bar = confidence_bar(-5.0)
        assert bar == "░░░░░░░░░░"


# ══════════════════════════════════════════════════════════════════
# Confidence HTML
# ══════════════════════════════════════════════════════════════════

class TestConfidenceBarHtml:

    def test_returns_html(self):
        html = confidence_bar_html(7.0)
        assert "<div" in html
        assert "confidence-bar" in html

    def test_contains_percentage(self):
        html = confidence_bar_html(10.0)
        assert "100%" in html


# ══════════════════════════════════════════════════════════════════
# Confidence Label
# ══════════════════════════════════════════════════════════════════

class TestConfidenceLabel:

    @pytest.mark.parametrize("score,expected", [
        (9.5, "Very High"),
        (7.5, "High"),
        (5.5, "Moderate"),
        (3.5, "Low"),
        (1.5, "Very Low"),
    ])
    def test_labels(self, score, expected):
        assert confidence_label(score) == expected


# ══════════════════════════════════════════════════════════════════
# Risk Badge
# ══════════════════════════════════════════════════════════════════

class TestRiskBadge:

    @pytest.mark.parametrize("level,contains", [
        ("LOW", "🟢"),
        ("MEDIUM", "🟡"),
        ("HIGH", "🔴"),
    ])
    def test_emoji_badge(self, level, contains):
        badge = risk_badge(level)
        assert contains in badge
        assert level in badge

    def test_unknown_level(self):
        badge = risk_badge("EXTREME")
        assert "UNKNOWN" in badge

    def test_html_badge(self):
        html = risk_badge_html("HIGH")
        assert "risk-badge" in html
        assert "HIGH" in html


# ══════════════════════════════════════════════════════════════════
# Trend Arrow
# ══════════════════════════════════════════════════════════════════

class TestTrendArrow:

    @pytest.mark.parametrize("change,expected_arrow", [
        (25.0, "↑"),
        (10.0, "↗"),
        (0.0, "→"),
        (-10.0, "↘"),
        (-25.0, "↓"),
    ])
    def test_arrows(self, change, expected_arrow):
        arrow = trend_arrow(change)
        assert expected_arrow in arrow


# ══════════════════════════════════════════════════════════════════
# Data Freshness
# ══════════════════════════════════════════════════════════════════

class TestDataFreshness:

    @pytest.mark.parametrize("minutes,label", [
        (2, "real-time"),
        (15, "recent"),
        (45, "aging"),
        (90, "stale"),
    ])
    def test_freshness_labels(self, minutes, label):
        indicator = data_freshness_indicator(minutes)
        assert label.lower() in indicator.lower()


# ══════════════════════════════════════════════════════════════════
# Verdict Helpers
# ══════════════════════════════════════════════════════════════════

class TestVerdictHelpers:

    def test_verdict_colour_strong_buy(self):
        colour = verdict_colour("Strong Buy")
        assert isinstance(colour, str)
        assert len(colour) > 0

    def test_verdict_emoji(self):
        emoji = verdict_emoji("Strong Buy")
        assert isinstance(emoji, str)
        assert len(emoji) > 0


# ══════════════════════════════════════════════════════════════════
# Sparkline
# ══════════════════════════════════════════════════════════════════

class TestSparkline:

    def test_sparkline_basic(self):
        line = score_sparkline({"news": 3.0, "onchain": 5.0, "technical": 7.0, "social": 9.0})
        assert isinstance(line, str)
        assert len(line) > 0

    def test_sparkline_empty(self):
        line = score_sparkline({})
        assert isinstance(line, str)
