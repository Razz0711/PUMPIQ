"""
Parameter Extractor
=====================
Step 4.1 – Extract structured parameters from a classified user query.

Extracts: tokens, risk preference, timeframe, filters, price context,
alert conditions, and numeric thresholds mentioned in natural language.

The extractor runs *after* intent classification and enriches the
ClassificationResult with actionable parameters.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Extracted Parameters Model
# ══════════════════════════════════════════════════════════════════

@dataclass
class PriceContext:
    """Price information mentioned by the user."""
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    quantity: Optional[float] = None


@dataclass
class FilterParams:
    """Strategy / advanced filters parsed from the query."""
    min_score: Optional[float] = None
    max_risk: Optional[str] = None       # "low" | "medium" | "high"
    required_modes: List[str] = field(default_factory=list)
    token_age: Optional[str] = None      # "new" | "established"
    min_liquidity: Optional[float] = None
    min_holders: Optional[int] = None
    rsi_range: Optional[tuple] = None    # (min, max)
    exclude_meme: bool = False


@dataclass
class AlertCondition:
    """Parsed alert setup parameters."""
    token: Optional[str] = None
    trigger_type: str = "price_above"    # price_above | price_below | condition
    trigger_value: Optional[float] = None
    condition_text: Optional[str] = None


@dataclass
class ExtractedParams:
    """Full parameter extraction result for a user query."""
    tokens: List[str] = field(default_factory=list)
    risk_preference: Optional[str] = None     # conservative | moderate | aggressive
    timeframe: Optional[str] = None           # scalp | day | swing | long
    num_recommendations: Optional[int] = None
    filters: FilterParams = field(default_factory=FilterParams)
    price_context: Optional[PriceContext] = None
    alert_condition: Optional[AlertCondition] = None


# ══════════════════════════════════════════════════════════════════
# Regex Patterns for Parameter Extraction
# ══════════════════════════════════════════════════════════════════

# Prices:  $0.0042, 0.05, $123.45
_PRICE = re.compile(r"\$?\s*(\d+\.?\d*)")

# "bought at $X", "entry at $X", "entry price $X"
_ENTRY_PRICE = re.compile(
    r"(?:bought|entry|entered|purchased)\s*(?:at|price[: ])\s*\$?\s*(\d+\.?\d*)", re.I
)

# "currently at $X", "now at $X", "price is $X"
_CURRENT_PRICE = re.compile(
    r"(?:currently|now|current\s+price|price\s+is)\s*(?:at)?\s*\$?\s*(\d+\.?\d*)", re.I
)

# "target $X", "hits $X", "reaches $X"
_TARGET_PRICE = re.compile(
    r"(?:target|hits?|reaches?|gets?\s+to)\s*\$?\s*(\d+\.?\d*)", re.I
)

# Quantity: "10000 tokens", "I hold 5000"
_QUANTITY = re.compile(
    r"(?:hold|have|bought|own)\s*(\d[\d,]*)\s*(?:tokens?|coins?)?", re.I
)

# Risk preference
_RISK_CONSERVATIVE = re.compile(
    r"\b(conservative|safe|low[\s-]?risk|safest|minimal\s+risk)\b", re.I
)
_RISK_MODERATE = re.compile(
    r"\b(moderate|balanced|medium[\s-]?risk|middle)\b", re.I
)
_RISK_AGGRESSIVE = re.compile(
    r"\b(aggressive|risky|high[\s-]?risk|degen|yolo|riskiest|gamble)\b", re.I
)

# Timeframe
_TF_SCALP = re.compile(r"\b(scalp|scalping|quick\s+flip|minutes?)\b", re.I)
_TF_DAY = re.compile(r"\b(day[\s-]?trad\w*|intraday|today|24\s*h|hours?)\b", re.I)
_TF_SWING = re.compile(r"\b(swing|few\s+days|days?|weekly|1[\s-]?7\s*d)\b", re.I)
_TF_LONG = re.compile(r"\b(long[\s-]?term|weeks?|months?|hold\s+long|position)\b", re.I)

# Number of results: "top 5", "give me 3", "best 10"
_NUM_RECS = re.compile(r"\btop\s+(\d+)\b|\b(\d+)\s+(?:picks?|recs?|recommendations?|coins?|tokens?)\b", re.I)

# Strategy filters
_LOW_RISK_HIGH_REWARD = re.compile(r"\blow[\s-]risk\b.*\b(high|upside|reward)\b", re.I)
_OVERSOLD = re.compile(r"\boversold\b", re.I)
_NEW_TOKEN = re.compile(r"\bnew\s+tokens?\b|\b<\s*7\s*d|\bfresh\b", re.I)
_ESTABLISHED = re.compile(r"\bestablished\b|\b>\s*30\s*d|\bmature\b", re.I)
_MIN_LIQUIDITY = re.compile(r"\bmin(?:imum)?\s+liquidity\b.*?\$?([\d,]+)", re.I)
_MIN_HOLDERS = re.compile(r"\bmin(?:imum)?\s+(\d[\d,]*)\s*holders?\b", re.I)
_MODE_REQUIRE = re.compile(
    r"\b(on[\s-]?chain|onchain|technical|social|news)\b", re.I
)
_EXCLUDE_MEME = re.compile(r"\bexclude\s+meme\b|\bno\s+meme\b", re.I)

# Alert trigger
_ALERT_PRICE_ABOVE = re.compile(
    r"(?:hits?|reaches?|above|over)\s*\$?\s*(\d+\.?\d*)", re.I
)
_ALERT_PRICE_BELOW = re.compile(
    r"(?:drops?\s+(?:below|under)|below|under)\s*\$?\s*(\d+\.?\d*)", re.I
)


# ══════════════════════════════════════════════════════════════════
# Extractor
# ══════════════════════════════════════════════════════════════════

class ParameterExtractor:
    """
    Extracts structured parameters from a raw user query.

    Usage::

        extractor = ParameterExtractor()
        params = extractor.extract("Show me top 5 low-risk tokens", tokens=["PEPE"])
    """

    def extract(
        self,
        query: str,
        intent: str = "discovery",
        tokens: Optional[List[str]] = None,
    ) -> ExtractedParams:
        """
        Extract all parameters from *query*.

        Parameters
        ----------
        query   : Raw user input
        intent  : Already-classified intent string
        tokens  : Already-extracted token list (from IntentRecognizer)
        """
        params = ExtractedParams(tokens=tokens or [])

        params.risk_preference = self._extract_risk(query)
        params.timeframe = self._extract_timeframe(query)
        params.num_recommendations = self._extract_num_recs(query)
        params.filters = self._extract_filters(query)
        params.price_context = self._extract_price_context(query)

        if intent == "alert":
            params.alert_condition = self._extract_alert(query, params.tokens)

        return params

    # ── Risk preference ───────────────────────────────────────────

    @staticmethod
    def _extract_risk(q: str) -> Optional[str]:
        if _RISK_CONSERVATIVE.search(q):
            return "conservative"
        if _RISK_AGGRESSIVE.search(q):
            return "aggressive"
        if _RISK_MODERATE.search(q):
            return "moderate"
        # No explicit mention → None (caller uses default)
        return None

    # ── Timeframe ─────────────────────────────────────────────────

    @staticmethod
    def _extract_timeframe(q: str) -> Optional[str]:
        if _TF_SCALP.search(q):
            return "scalp"
        if _TF_DAY.search(q):
            return "day"
        if _TF_SWING.search(q):
            return "swing"
        if _TF_LONG.search(q):
            return "long"
        return None

    # ── Number of recommendations ─────────────────────────────────

    @staticmethod
    def _extract_num_recs(q: str) -> Optional[int]:
        m = _NUM_RECS.search(q)
        if m:
            val = m.group(1) or m.group(2)
            n = int(val)
            return max(1, min(n, 10))
        return None

    # ── Strategy filters ──────────────────────────────────────────

    @staticmethod
    def _extract_filters(q: str) -> FilterParams:
        f = FilterParams()

        if _LOW_RISK_HIGH_REWARD.search(q):
            f.max_risk = "low"

        if _OVERSOLD.search(q):
            f.rsi_range = (0, 35)

        if _NEW_TOKEN.search(q):
            f.token_age = "new"
        elif _ESTABLISHED.search(q):
            f.token_age = "established"

        m = _MIN_LIQUIDITY.search(q)
        if m:
            f.min_liquidity = float(m.group(1).replace(",", ""))

        m = _MIN_HOLDERS.search(q)
        if m:
            f.min_holders = int(m.group(1).replace(",", ""))

        # Required data modes mentioned
        modes = set()
        for m in _MODE_REQUIRE.finditer(q):
            mode = m.group(1).lower().replace("-", "").replace(" ", "")
            if mode == "onchain":
                modes.add("onchain")
            elif mode in ("technical", "social", "news"):
                modes.add(mode)
        f.required_modes = sorted(modes)

        if _EXCLUDE_MEME.search(q):
            f.exclude_meme = True

        return f

    # ── Price context ─────────────────────────────────────────────

    @staticmethod
    def _extract_price_context(q: str) -> Optional[PriceContext]:
        ctx = PriceContext()
        found = False

        m = _ENTRY_PRICE.search(q)
        if m:
            ctx.entry_price = float(m.group(1))
            found = True

        m = _CURRENT_PRICE.search(q)
        if m:
            ctx.current_price = float(m.group(1))
            found = True

        m = _TARGET_PRICE.search(q)
        if m:
            ctx.target_price = float(m.group(1))
            found = True

        m = _QUANTITY.search(q)
        if m:
            ctx.quantity = float(m.group(1).replace(",", ""))
            found = True

        return ctx if found else None

    # ── Alert condition ───────────────────────────────────────────

    @staticmethod
    def _extract_alert(q: str, tokens: List[str]) -> AlertCondition:
        alert = AlertCondition()
        alert.token = tokens[0] if tokens else None

        m = _ALERT_PRICE_BELOW.search(q)
        if m:
            alert.trigger_type = "price_below"
            alert.trigger_value = float(m.group(1))
            return alert

        m = _ALERT_PRICE_ABOVE.search(q)
        if m:
            alert.trigger_type = "price_above"
            alert.trigger_value = float(m.group(1))
            return alert

        # Condition-based (not a simple price)
        alert.trigger_type = "condition"
        alert.condition_text = q
        return alert
