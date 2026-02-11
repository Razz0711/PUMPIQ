"""
Crypto-Specific NLP Sentiment Analyzer
=======================================
Core sentiment analysis engine tuned for cryptocurrency discussions.

Scoring Method:
- Each text is scored from -10 (extremely bearish) to +10 (extremely bullish)
- Uses a crypto-specific lexicon layered on top of general NLP sentiment
- Detects emotional tone: Bullish, Bearish, Neutral, FUD, FOMO
- Handles emojis, slang, and crypto-native terminology
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Constants & Lexicon
# ---------------------------------------------------------------------------

class EmotionalTone(str, Enum):
    EXTREMELY_BULLISH = "extremely_bullish"
    BULLISH = "bullish"
    SLIGHTLY_BULLISH = "slightly_bullish"
    NEUTRAL = "neutral"
    SLIGHTLY_BEARISH = "slightly_bearish"
    BEARISH = "bearish"
    EXTREMELY_BEARISH = "extremely_bearish"
    FUD = "fud"
    FOMO = "fomo"


# Crypto-specific bullish terms and their sentiment weights
BULLISH_LEXICON: Dict[str, float] = {
    # Strong bullish
    "moon": 3.0, "mooning": 3.5, "to the moon": 4.0,
    "gem": 2.5, "hidden gem": 3.0, "100x": 4.0, "1000x": 4.5,
    "bullish": 2.5, "very bullish": 3.5, "super bullish": 4.0,
    "buy the dip": 2.0, "btd": 2.0, "accumulate": 2.0,
    "ath": 2.5, "all time high": 2.5, "new ath": 3.0,
    "breakout": 2.5, "breaking out": 3.0,
    "lambo": 2.0, "wen lambo": 1.5,
    "diamond hands": 2.0, "hodl": 1.5, "hold": 1.0,
    "undervalued": 2.5, "sleeping giant": 3.0,
    "massive potential": 3.0, "huge potential": 3.0,
    "easy money": 2.0, "free money": 2.0,
    "next big thing": 2.5, "early": 2.0, "still early": 2.5,
    "generational wealth": 3.0, "life changing": 3.0,
    "pump": 2.0, "pumping": 2.5, "sending": 2.0, "sent": 2.0,
    "rocket": 2.5, "fire": 1.5, "lit": 1.5,
    "based": 1.5, "alpha": 2.0, "alpha call": 2.5,
    "conviction": 2.0, "high conviction": 3.0,
    "green candle": 1.5, "green dildo": 2.0,
    "parabolic": 3.0, "vertical": 2.5,
    "whale buy": 2.5, "whale accumulating": 3.0,
    "institutional": 2.0, "smart money": 2.5,
    "gm": 0.5, "wagmi": 1.5, "lfg": 2.0, "let's go": 1.5,

    # Moderate bullish
    "promising": 1.5, "solid project": 2.0, "good fundamentals": 2.0,
    "interesting": 1.0, "worth watching": 1.5, "keep an eye": 1.0,
    "strong team": 2.0, "doxxed team": 2.0,
    "audit": 1.5, "audited": 1.5, "renounced": 1.5,
    "locked liquidity": 2.0, "lp locked": 2.0,
    "partnership": 2.0, "collab": 1.5, "listing": 2.0,
}

# Crypto-specific bearish terms and their sentiment weights (negative)
BEARISH_LEXICON: Dict[str, float] = {
    # Strong bearish
    "rug": -4.0, "rug pull": -5.0, "rugged": -5.0, "rugpull": -5.0,
    "scam": -4.5, "scammer": -4.5, "ponzi": -4.0,
    "dump": -3.0, "dumping": -3.5, "dumped": -3.5,
    "dead": -3.5, "dead coin": -4.0, "dead project": -4.0,
    "rekt": -3.0, "rip": -2.5, "paper hands": -1.5,
    "bearish": -2.5, "very bearish": -3.5, "super bearish": -4.0,
    "crash": -3.5, "crashing": -4.0, "crashed": -4.0,
    "sell": -1.5, "selling": -2.0, "exit": -2.0, "exit scam": -5.0,
    "fud": -1.0,  # FUD can be used defensively too
    "honeypot": -5.0, "can't sell": -5.0, "cant sell": -5.0,
    "red flag": -3.0, "warning": -2.0, "be careful": -2.0,
    "avoid": -3.0, "stay away": -3.5, "do not buy": -4.0,
    "fake": -3.0, "fake project": -4.0, "copycat": -2.5,
    "no audit": -2.0, "not audited": -2.0, "unaudited": -2.0,
    "unlocked liquidity": -3.0, "lp not locked": -3.0,
    "whale dump": -3.5, "whale selling": -3.0, "dev selling": -4.0,
    "insider": -2.5, "insider selling": -3.5,
    "overvalued": -2.0, "overbought": -1.5,
    "bubble": -2.5, "top signal": -2.0,
    "bag holder": -2.0, "bagholding": -2.0,
    "ngmi": -2.0, "down bad": -2.0, "capitulation": -3.0,
    "red candle": -1.5, "blood": -2.0, "bloodbath": -3.0,

    # Moderate bearish
    "risky": -1.5, "high risk": -2.0, "concerned": -1.5,
    "disappointed": -2.0, "disappointing": -2.0,
    "delay": -1.5, "delayed": -1.5, "no updates": -2.0,
    "abandoned": -3.5, "inactive": -2.5, "ghost chain": -3.0,
    "lawsuit": -3.0, "sec": -2.0, "regulation": -1.5,
}

# Emoji sentiment weights
EMOJI_SENTIMENT: Dict[str, float] = {
    "🚀": 2.5, "🌙": 2.0, "💎": 2.0, "🙌": 1.5,
    "📈": 2.0, "💰": 1.5, "🔥": 1.5, "💪": 1.0,
    "🎯": 1.5, "✅": 1.0, "💚": 1.0, "🟢": 1.0,
    "⬆️": 1.0, "🤑": 1.5, "👀": 0.5, "👑": 1.5,
    "🏆": 1.5, "⚡": 1.0, "🌕": 2.0, "🐂": 2.0,
    "📉": -2.0, "🔴": -1.5, "🐻": -2.0, "💀": -2.5,
    "⬇️": -1.0, "😱": -2.0, "😭": -1.5, "🤡": -2.0,
    "🚩": -3.0, "⚠️": -2.0, "❌": -2.0, "🗑️": -2.5,
    "🪦": -3.0, "💩": -2.5, "🐀": -3.0, "🤮": -2.0,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Result of sentiment analysis on a single text"""
    raw_score: float                    # -10 to +10
    normalized_score: float             # -1.0 to +1.0
    emotional_tone: EmotionalTone
    confidence: float                   # 0.0 to 1.0
    bullish_signals: List[str] = field(default_factory=list)
    bearish_signals: List[str] = field(default_factory=list)
    emoji_sentiment: float = 0.0
    lexicon_sentiment: float = 0.0
    context_adjustment: float = 0.0
    is_sarcastic: bool = False
    has_question: bool = False
    word_count: int = 0


@dataclass
class WeightedSentiment:
    """Sentiment with author influence weight applied"""
    sentiment: SentimentResult
    author_weight: float = 1.0          # Multiplier based on influence
    platform: str = "unknown"
    weighted_score: float = 0.0         # raw_score * author_weight

    def __post_init__(self):
        self.weighted_score = self.sentiment.raw_score * self.author_weight


# ---------------------------------------------------------------------------
# Sentiment Analyzer
# ---------------------------------------------------------------------------

class CryptoSentimentAnalyzer:
    """
    NLP sentiment analyzer specialized for cryptocurrency discussions.

    Pipeline:
    1. Text preprocessing (lowercase, emoji extraction, URL removal)
    2. Crypto lexicon matching (bullish/bearish term detection)
    3. Emoji sentiment scoring
    4. Context modifiers (negation, sarcasm, questions)
    5. Score normalization to [-10, +10]
    6. Emotional tone classification
    7. Confidence estimation

    Weighting by Author Influence:
    - Influencer (10K+ followers): 10x weight
    - Mid-tier (1K-10K followers): 3x weight
    - Regular user: 1x weight
    - New/suspicious account: 0.3x weight
    """

    # Negation words that flip sentiment
    NEGATION_WORDS = {
        "not", "no", "never", "don't", "dont", "doesn't", "doesnt",
        "isn't", "isnt", "wasn't", "wasnt", "won't", "wont",
        "can't", "cant", "shouldn't", "shouldnt", "wouldn't", "wouldnt",
        "none", "neither", "nor", "hardly", "barely", "rarely",
    }

    # Sarcasm / irony indicators
    SARCASM_INDICATORS = {
        "/s", "sure thing", "yeah right", "totally", "definitely not",
        "oh great", "wow amazing", "lmao sure", "copium", "hopium",
    }

    # Intensifiers
    INTENSIFIERS = {
        "very": 1.5, "extremely": 2.0, "super": 1.8, "incredibly": 2.0,
        "absolutely": 1.8, "definitely": 1.5, "totally": 1.5,
        "massively": 2.0, "insanely": 2.0, "ridiculously": 1.8,
    }

    # Question patterns (reduce certainty)
    QUESTION_PATTERNS = [
        r"\?$", r"^(is|are|will|can|should|would|does|do|has|have|what|when|why|how)\b",
    ]

    def __init__(self):
        """Initialize analyzer with compiled regex patterns."""
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._mention_pattern = re.compile(r"@\w+")
        self._hashtag_pattern = re.compile(r"#(\w+)")
        self._ticker_pattern = re.compile(r"\$([A-Za-z]{2,10})")
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        self._question_patterns = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze a single piece of text for crypto sentiment.

        Args:
            text: Raw text (tweet, post, message, etc.)

        Returns:
            SentimentResult with score in [-10, +10]
        """
        if not text or not text.strip():
            return SentimentResult(
                raw_score=0, normalized_score=0,
                emotional_tone=EmotionalTone.NEUTRAL,
                confidence=0.0, word_count=0,
            )

        # Step 1: Preprocess
        cleaned, emojis_found = self._preprocess(text)
        word_count = len(cleaned.split())

        # Step 2: Lexicon scoring
        lexicon_score, bullish_signals, bearish_signals = self._score_lexicon(cleaned)

        # Step 3: Emoji scoring
        emoji_score = self._score_emojis(emojis_found, text)

        # Step 4: Context modifiers
        negation_multiplier = self._detect_negation(cleaned)
        sarcasm_detected = self._detect_sarcasm(cleaned)
        is_question = self._detect_question(text)
        intensifier_multiplier = self._detect_intensifiers(cleaned)

        # Step 5: Combine scores
        raw_combined = (lexicon_score * intensifier_multiplier * negation_multiplier) + emoji_score

        # Sarcasm flips sentiment
        if sarcasm_detected:
            raw_combined *= -0.7  # Flip and slightly dampen

        # Questions reduce intensity
        question_dampening = 0.6 if is_question else 1.0
        raw_combined *= question_dampening

        # Clamp to [-10, +10]
        raw_score = max(-10.0, min(10.0, raw_combined))
        normalized_score = raw_score / 10.0

        # Step 6: Emotional tone
        emotional_tone = self._classify_tone(raw_score, bullish_signals, bearish_signals)

        # Step 7: Confidence
        confidence = self._estimate_confidence(
            word_count, len(bullish_signals), len(bearish_signals),
            abs(raw_score), is_question, sarcasm_detected,
        )

        context_adj = 0.0
        if sarcasm_detected:
            context_adj -= 0.3
        if is_question:
            context_adj -= 0.2

        return SentimentResult(
            raw_score=round(raw_score, 2),
            normalized_score=round(normalized_score, 4),
            emotional_tone=emotional_tone,
            confidence=round(confidence, 4),
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals,
            emoji_sentiment=round(emoji_score, 2),
            lexicon_sentiment=round(lexicon_score, 2),
            context_adjustment=round(context_adj, 2),
            is_sarcastic=sarcasm_detected,
            has_question=is_question,
            word_count=word_count,
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze a batch of texts."""
        return [self.analyze(t) for t in texts]

    def compute_weighted_sentiment(
        self,
        text: str,
        follower_count: int = 0,
        account_age_days: int = 365,
        is_verified: bool = False,
        onchain_reputation: Optional[float] = None,
        platform: str = "twitter",
    ) -> WeightedSentiment:
        """
        Analyze text and apply author-influence weighting.

        Weight tiers:
        - Influencer (10K+ followers):  10x
        - Mid-tier   (1K-10K):           3x
        - Regular    (100-1K):            1x
        - New/small  (<100):              0.5x
        - Suspicious (<30 days, <50):     0.3x

        Additional multipliers:
        - Verified badge: 1.5x
        - On-chain reputation (Farcaster): up to 2x
        """
        sentiment = self.analyze(text)
        weight = self._compute_author_weight(
            follower_count, account_age_days, is_verified,
            onchain_reputation, platform,
        )

        return WeightedSentiment(
            sentiment=sentiment,
            author_weight=round(weight, 2),
            platform=platform,
        )

    # ------------------------------------------------------------------
    # Internal – Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> Tuple[str, List[str]]:
        """Clean text and extract emojis."""
        # Extract emojis before removal
        emojis_found = self._emoji_pattern.findall(text)

        cleaned = text.lower()
        cleaned = self._url_pattern.sub("", cleaned)
        cleaned = self._mention_pattern.sub("", cleaned)
        # Keep hashtag text, remove #
        cleaned = self._hashtag_pattern.sub(r"\1", cleaned)
        # Remove emojis from text for lexicon analysis
        cleaned = self._emoji_pattern.sub("", cleaned)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned, emojis_found

    # ------------------------------------------------------------------
    # Internal – Lexicon Scoring
    # ------------------------------------------------------------------

    def _score_lexicon(self, text: str) -> Tuple[float, List[str], List[str]]:
        """Match crypto lexicon terms and accumulate score."""
        score = 0.0
        bullish: List[str] = []
        bearish: List[str] = []

        text_lower = f" {text} "  # pad for whole-word matching

        # Check multi-word phrases first (longer = higher priority)
        all_terms = list(BULLISH_LEXICON.items()) + list(BEARISH_LEXICON.items())
        # Sort by phrase length descending so longer phrases match first
        all_terms.sort(key=lambda x: len(x[0]), reverse=True)

        matched_spans: List[Tuple[int, int]] = []

        for term, weight in all_terms:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text_lower):
                span = (match.start(), match.end())
                # Skip if overlapping with already-matched span
                if any(s <= span[0] < e or s < span[1] <= e for s, e in matched_spans):
                    continue
                matched_spans.append(span)
                score += weight
                if weight > 0:
                    bullish.append(term)
                else:
                    bearish.append(term)

        return score, bullish, bearish

    # ------------------------------------------------------------------
    # Internal – Emoji Scoring
    # ------------------------------------------------------------------

    def _score_emojis(self, emoji_groups: List[str], original_text: str) -> float:
        """Score emojis found in text."""
        score = 0.0
        for char in original_text:
            if char in EMOJI_SENTIMENT:
                score += EMOJI_SENTIMENT[char]
        return score

    # ------------------------------------------------------------------
    # Internal – Context Modifiers
    # ------------------------------------------------------------------

    def _detect_negation(self, text: str) -> float:
        """
        Detect negation words.  If found near sentiment terms, flip polarity.
        Returns a multiplier: 1.0 (no negation) or -0.8 (negation present).
        """
        words = text.split()
        for i, word in enumerate(words):
            if word in self.NEGATION_WORDS:
                # Check if the next 1-3 words contain a sentiment term
                window = " ".join(words[i + 1: i + 4])
                for term in list(BULLISH_LEXICON) + list(BEARISH_LEXICON):
                    if term in window:
                        return -0.8
        return 1.0

    def _detect_sarcasm(self, text: str) -> bool:
        """Detect common sarcasm/irony indicators."""
        text_lower = text.lower()
        return any(ind in text_lower for ind in self.SARCASM_INDICATORS)

    def _detect_question(self, text: str) -> bool:
        """Detect if text is a question (reduces sentiment certainty)."""
        return any(p.search(text) for p in self._question_patterns)

    def _detect_intensifiers(self, text: str) -> float:
        """Detect intensifier words and return a multiplier."""
        words = text.split()
        max_intensifier = 1.0
        for word in words:
            if word in self.INTENSIFIERS:
                max_intensifier = max(max_intensifier, self.INTENSIFIERS[word])
        return max_intensifier

    # ------------------------------------------------------------------
    # Internal – Classification
    # ------------------------------------------------------------------

    def _classify_tone(
        self, score: float,
        bullish_signals: List[str],
        bearish_signals: List[str],
    ) -> EmotionalTone:
        """Classify the emotional tone based on score and signals."""
        # Check for FOMO / FUD patterns
        fud_terms = {"fud", "fear", "panic", "crash", "collapse"}
        fomo_terms = {"fomo", "missing out", "last chance", "don't miss", "hurry"}

        all_signals_lower = {s.lower() for s in bullish_signals + bearish_signals}

        if fud_terms & all_signals_lower and score < -2:
            return EmotionalTone.FUD
        if fomo_terms & all_signals_lower and score > 2:
            return EmotionalTone.FOMO

        if score >= 7:
            return EmotionalTone.EXTREMELY_BULLISH
        elif score >= 4:
            return EmotionalTone.BULLISH
        elif score >= 1.5:
            return EmotionalTone.SLIGHTLY_BULLISH
        elif score > -1.5:
            return EmotionalTone.NEUTRAL
        elif score > -4:
            return EmotionalTone.SLIGHTLY_BEARISH
        elif score > -7:
            return EmotionalTone.BEARISH
        else:
            return EmotionalTone.EXTREMELY_BEARISH

    def _estimate_confidence(
        self,
        word_count: int,
        n_bullish: int,
        n_bearish: int,
        abs_score: float,
        is_question: bool,
        is_sarcastic: bool,
    ) -> float:
        """
        Estimate confidence in the sentiment assessment.

        Factors:
        - More text → higher confidence
        - More signal terms → higher confidence
        - Strong score → higher confidence
        - Questions → lower confidence
        - Sarcasm → lower confidence
        - Mixed signals → lower confidence
        """
        confidence = 0.5  # baseline

        # Text length factor
        if word_count < 5:
            confidence -= 0.15
        elif word_count > 20:
            confidence += 0.1
        elif word_count > 50:
            confidence += 0.15

        # Signal density
        total_signals = n_bullish + n_bearish
        if total_signals >= 3:
            confidence += 0.15
        elif total_signals >= 1:
            confidence += 0.05

        # Mixed signals reduce confidence
        if n_bullish > 0 and n_bearish > 0:
            confidence -= 0.1 * min(n_bullish, n_bearish)

        # Score strength
        if abs_score > 6:
            confidence += 0.15
        elif abs_score > 3:
            confidence += 0.05

        # Penalties
        if is_question:
            confidence -= 0.15
        if is_sarcastic:
            confidence -= 0.2

        return max(0.05, min(0.99, confidence))

    # ------------------------------------------------------------------
    # Internal – Author Weighting
    # ------------------------------------------------------------------

    def _compute_author_weight(
        self,
        follower_count: int,
        account_age_days: int,
        is_verified: bool,
        onchain_reputation: Optional[float],
        platform: str,
    ) -> float:
        """
        Compute author influence weight.

        Follower tiers:
        - 10K+:   10x (major influencer)
        - 5K-10K:  5x (growing influencer)
        - 1K-5K:   3x (micro-influencer)
        - 100-1K:  1x (regular active user)
        - <100:    0.5x (small account)

        Modifiers:
        - Verified: 1.5x
        - Account < 30 days old: 0.3x penalty
        - On-chain reputation (Farcaster): up to +2x
        """
        # Base weight from followers
        if follower_count >= 10_000:
            weight = 10.0
        elif follower_count >= 5_000:
            weight = 5.0
        elif follower_count >= 1_000:
            weight = 3.0
        elif follower_count >= 100:
            weight = 1.0
        else:
            weight = 0.5

        # Verified badge
        if is_verified:
            weight *= 1.5

        # New account penalty
        if account_age_days < 30:
            weight *= 0.3
        elif account_age_days < 90:
            weight *= 0.6

        # On-chain reputation (Farcaster specific)
        if onchain_reputation is not None and platform == "farcaster":
            # onchain_reputation expected 0-1, maps to 1x-2x multiplier
            weight *= 1.0 + min(onchain_reputation, 1.0)

        return weight

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def score_to_label(score: float) -> str:
        """Convert numeric score to human-readable label."""
        if score >= 7:
            return "Extremely Bullish"
        elif score >= 4:
            return "Bullish"
        elif score >= 1.5:
            return "Slightly Bullish"
        elif score > -1.5:
            return "Neutral"
        elif score > -4:
            return "Slightly Bearish"
        elif score > -7:
            return "Bearish"
        else:
            return "Extremely Bearish"
