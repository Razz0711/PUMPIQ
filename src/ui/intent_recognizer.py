"""
Query Intent Recognizer
=========================
Step 4.1 – Natural language understanding for user queries.

Classifies user queries into one of six intent categories and routes
them to the appropriate pipeline handler.

Intent Categories:
    1. discovery   – General "best coins" recommendations
    2. analysis    – Deep dive on a specific token
    3. comparison  – Side-by-side comparison of 2-4 tokens
    4. strategy    – Filtered recommendations (specific criteria)
    5. portfolio   – Position management advice
    6. alert       – Price/condition alert setup
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Intent Enum (extends the AI engine QueryType with UI-layer intents)
# ══════════════════════════════════════════════════════════════════

class Intent(str, Enum):
    DISCOVERY = "discovery"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"
    ALERT = "alert"
    UNKNOWN = "unknown"


# ══════════════════════════════════════════════════════════════════
# Keyword / Pattern Banks
# ══════════════════════════════════════════════════════════════════

# Ordered from most specific → least specific so first match wins.

_ALERT_PATTERNS: List[re.Pattern] = [
    re.compile(r"\balert\b.*\bwhen\b", re.I),
    re.compile(r"\bnotify\b.*\b(if|when)\b", re.I),
    re.compile(r"\bwatch\b.*\btell\s*me\b", re.I),
    re.compile(r"\bset\s*(a\s+)?(price\s+)?alert\b", re.I),
    re.compile(r"\bping\s*me\b", re.I),
    re.compile(r"\bremind\s*me\b.*\b(price|hits?|reaches?)\b", re.I),
]

_PORTFOLIO_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(hold|sell)\b.*\btoken\b|\bshould\s+i\s+(hold|sell)\b", re.I),
    re.compile(r"\bi\s+(bought|hold|own|have)\b.*\$?\d", re.I),
    re.compile(r"\bmy\s+portfolio\b", re.I),
    re.compile(r"\btake\s+profit", re.I),
    re.compile(r"\bentry\s+(price|was)\b", re.I),
    re.compile(r"\bwhat\s+now\b.*\bbought\b|\bbought\b.*\bwhat\s+now\b", re.I),
    re.compile(r"\bposition\b.*\b(exit|close|cut)\b", re.I),
    re.compile(r"\bstop\s*loss\b.*\bshould\b", re.I),
    re.compile(r"\bp/?l\b|\bprofit\s+loss\b", re.I),
]

_COMPARISON_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bcompare\b", re.I),
    re.compile(r"\bvs\.?\b|\bversus\b", re.I),
    re.compile(r"\bwhich\s+is\s+better\b", re.I),
    re.compile(r"\b(pick|choose)\s+between\b", re.I),
    re.compile(r"\bor\b.*\bwhich\b|\bwhich\b.*\bor\b", re.I),
]

_ANALYSIS_PATTERNS: List[re.Pattern] = [
    re.compile(r"\banalyze\b|\banalysis\b|\banalyse\b", re.I),
    re.compile(r"\bshould\s+i\s+buy\b", re.I),
    re.compile(r"\bwhat\s+(do\s+you\s+think|about)\b", re.I),
    re.compile(r"\bis\s+\w+\s+(a\s+)?good\b", re.I),
    re.compile(r"\btell\s+me\s+about\b", re.I),
    re.compile(r"\bdeep\s*dive\b", re.I),
    re.compile(r"\bbreakdown\b.*token\b|\btoken\b.*\bbreakdown\b", re.I),
    re.compile(r"\blook\s+into\b", re.I),
]

_STRATEGY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\blow[\s-]risk\b.*\b(high|upside|reward)\b", re.I),
    re.compile(r"\bhigh[\s-]risk\b.*\b(high|reward)\b", re.I),
    re.compile(r"\boversold\b.*\bstrong\b|\bstrong\b.*\boversold\b", re.I),
    re.compile(r"\bgood\s+on[\s-]chain\b.*\bbad\s+social\b", re.I),
    re.compile(r"\bnew\s+tokens?\b.*\bcommunity\b|\bcommunity\b.*\bnew\s+tokens?\b", re.I),
    re.compile(r"\bwith\b.*\b(strong|good|high)\b.*\b(onchain|on-chain|social|technical|news)\b", re.I),
    re.compile(r"\bfilter\b|\bcriteria\b|\bcondition", re.I),
    re.compile(r"\bfind\s+(me\s+)?tokens?\b.*\bwith\b", re.I),
    re.compile(r"\bunder\s*valued\b|\bunder[\s-]?rated\b", re.I),
    re.compile(r"\bsafest\b|\briskiest\b", re.I),
    re.compile(r"\bgem\b|\bhidden\b", re.I),
]

_DISCOVERY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bbest\s+(coins?|tokens?)\b", re.I),
    re.compile(r"\btop\s+(picks?|opportunities|coins?|tokens?)\b", re.I),
    re.compile(r"\bwhat('?s|\s+is)\s+looking\s+good\b", re.I),
    re.compile(r"\bgive\s+me\b.*\b(picks?|recs?|recommendations?)\b", re.I),
    re.compile(r"\bwhat\s+should\s+i\s+buy\b", re.I),
    re.compile(r"\bshow\s+me\b.*\b(opportunities|tokens?|coins?)\b", re.I),
    re.compile(r"\brecommend\b", re.I),
    re.compile(r"\bsuggest\b.*\b(token|coin|buy)\b", re.I),
    re.compile(r"\bany\s+good\b.*\b(buys?|picks?)\b", re.I),
    re.compile(r"\bwhat\s+to\s+buy\b", re.I),
]


# ══════════════════════════════════════════════════════════════════
# Token Detection Patterns
# ══════════════════════════════════════════════════════════════════

# $TICKER, cashtag style
_CASHTAG = re.compile(r"\$([A-Za-z][A-Za-z0-9]{1,11})\b")

# Explicit "token X" or "coin X"
_TOKEN_COIN = re.compile(
    r"\b(?:token|coin)\s+([A-Za-z][A-Za-z0-9]{1,11})\b", re.I
)

# ALL-CAPS word ≥ 2 chars that looks like a ticker
_CAPS_TICKER = re.compile(r"\b([A-Z][A-Z0-9]{1,10})\b")

# Solana-style address (base58, 32-44 chars)
_SOL_ADDRESS = re.compile(r"\b([1-9A-HJ-NP-Za-km-z]{32,44})\b")

# Ethereum-style address (0x + 40 hex)
_ETH_ADDRESS = re.compile(r"\b(0x[0-9a-fA-F]{40})\b")

# Common English words to exclude from ticker detection
_STOPWORDS = frozenset({
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "HAD", "HAS", "HIS",
    "HOW", "ITS", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID",
    "GET", "LET", "SAY", "SHE", "TOO", "USE", "BUY", "TOP", "LOW",
    "HIGH", "ANY", "SET", "HIT", "BIG", "TRY", "ASK", "WHY", "HOW",
    "YES", "MAY", "RUN", "OWN", "ADD", "END", "PUT", "AGO", "GOT",
    "TWO", "FEW", "FAR", "RED", "BAD", "RSI", "MACD", "ATH", "ATL",
    "WITH", "WHAT", "WHEN", "FROM", "HAVE", "THIS", "THAT", "BEEN",
    "SOME", "WILL", "MORE", "VERY", "JUST", "THAN", "THEM", "MUCH",
    "BEST", "GOOD", "SHOW", "GIVE", "TELL", "HOLD", "SELL", "FIND",
    "PICK", "COIN", "TOKEN", "ALERT", "WATCH", "ABOUT", "WHICH",
    "SHOULD", "COULD", "WOULD", "THEIR", "THERE", "WHERE", "THESE",
    "RIGHT", "THINK", "PRICE", "TODAY", "ENTRY", "COMPARE", "ANALYZE",
    "ANALYSIS", "LOOKING", "OPPORTUNITIES", "RECOMMENDATIONS", "PROFIT",
    "PORTFOLIO", "NOTIFY", "BOUGHT", "POSITION", "MARKET",
})


# ══════════════════════════════════════════════════════════════════
# Intent Recognizer
# ══════════════════════════════════════════════════════════════════

class IntentRecognizer:
    """
    Classify user queries using rule-based pattern matching.

    The classifier checks patterns in order from most specific intent
    (alert, portfolio) → most general (discovery).  This prevents
    broad patterns from stealing queries that carry specific intent
    signals.

    Usage::

        recognizer = IntentRecognizer()
        result = recognizer.classify("Analyze PEPE")
        print(result.intent)   # Intent.ANALYSIS
        print(result.tokens)   # ["PEPE"]
    """

    def classify(self, query: str) -> "ClassificationResult":
        """
        Classify *query* and return a full ClassificationResult
        including intent, confidence, and extracted tokens.
        """
        query = query.strip()
        if not query:
            return ClassificationResult(
                raw_query=query,
                intent=Intent.UNKNOWN,
                confidence=0.0,
            )

        # Extract tokens first (used by several intents)
        tokens = self._extract_tokens(query)

        # Score each intent
        scores: Dict[Intent, float] = {}
        scores[Intent.ALERT] = self._pattern_score(query, _ALERT_PATTERNS)
        scores[Intent.PORTFOLIO] = self._pattern_score(query, _PORTFOLIO_PATTERNS)
        scores[Intent.COMPARISON] = self._pattern_score(query, _COMPARISON_PATTERNS)
        scores[Intent.STRATEGY] = self._pattern_score(query, _STRATEGY_PATTERNS)
        scores[Intent.ANALYSIS] = self._pattern_score(query, _ANALYSIS_PATTERNS)
        scores[Intent.DISCOVERY] = self._pattern_score(query, _DISCOVERY_PATTERNS)

        # Heuristic boosts
        if len(tokens) >= 2 and scores[Intent.COMPARISON] > 0:
            scores[Intent.COMPARISON] += 0.3
        if len(tokens) == 1 and scores[Intent.ANALYSIS] > 0:
            scores[Intent.ANALYSIS] += 0.2
        if len(tokens) == 0 and scores[Intent.DISCOVERY] > 0:
            scores[Intent.DISCOVERY] += 0.2

        # Pick highest-scoring intent
        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]

        if best_score == 0:
            # No patterns matched – fall back based on token count
            if len(tokens) >= 2:
                best_intent = Intent.COMPARISON
            elif len(tokens) == 1:
                best_intent = Intent.ANALYSIS
            else:
                best_intent = Intent.DISCOVERY
            best_score = 0.3  # low confidence

        # Normalise confidence to 0-1 range
        confidence = min(1.0, best_score)

        return ClassificationResult(
            raw_query=query,
            intent=best_intent,
            confidence=round(confidence, 2),
            tokens=tokens,
            all_scores=scores,
        )

    # ── Token extraction ──────────────────────────────────────────

    def _extract_tokens(self, query: str) -> List[str]:
        """
        Pull token tickers, names, or contract addresses from the query.
        Priority: $CASHTAG > contract address > "token X" > ALL-CAPS word.
        Deduplicates and preserves order.
        """
        found: List[str] = []
        seen: set = set()

        def _add(t: str) -> None:
            key = t.upper()
            if key not in seen:
                seen.add(key)
                found.append(t)

        # 1. $CASHTAG
        for m in _CASHTAG.finditer(query):
            _add(m.group(1).upper())

        # 2. Contract addresses
        for m in _ETH_ADDRESS.finditer(query):
            _add(m.group(1))
        for m in _SOL_ADDRESS.finditer(query):
            addr = m.group(1)
            # Avoid false positives on normal words
            if len(addr) >= 32 and not addr.isalpha():
                _add(addr)

        # 3. "token X" / "coin X"
        for m in _TOKEN_COIN.finditer(query):
            _add(m.group(1).upper())

        # 4. ALL-CAPS tickers (≥ 2 chars, not stopword)
        for m in _CAPS_TICKER.finditer(query):
            word = m.group(1)
            if len(word) >= 2 and word not in _STOPWORDS:
                _add(word)

        return found

    # ── Pattern scoring ───────────────────────────────────────────

    @staticmethod
    def _pattern_score(query: str, patterns: List[re.Pattern]) -> float:
        """
        Return a 0-1 score based on how many patterns match.
        First match gives 0.5, each additional adds 0.15 (capped at 1.0).
        """
        hits = sum(1 for p in patterns if p.search(query))
        if hits == 0:
            return 0.0
        return min(1.0, 0.5 + (hits - 1) * 0.15)


# ══════════════════════════════════════════════════════════════════
# Classification Result
# ══════════════════════════════════════════════════════════════════

@dataclass
class ClassificationResult:
    """Output of the intent recognizer."""
    raw_query: str
    intent: Intent
    confidence: float = 0.0
    tokens: List[str] = field(default_factory=list)
    all_scores: Dict[Intent, float] = field(default_factory=dict)

    @property
    def is_ambiguous(self) -> bool:
        """True when confidence is too low to act without clarification."""
        return self.confidence < 0.5

    @property
    def needs_token_disambiguation(self) -> bool:
        """True when a token was detected but may be ambiguous."""
        return len(self.tokens) == 1 and self.confidence < 0.6

    @property
    def top_two_intents(self) -> List[Tuple[Intent, float]]:
        """Return the two highest-scoring intents (for clarification)."""
        sorted_scores = sorted(
            self.all_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_scores[:2]
