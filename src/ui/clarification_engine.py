"""
Clarification Engine
======================
Step 4.1 – Decision tree for handling ambiguous user queries.

When the IntentRecognizer returns low confidence or missing critical
parameters, this engine generates the appropriate clarification prompt
to present to the user.

Clarification triggers:
  - Intent unclear (confidence < 0.5)
  - Token name ambiguous (multiple matches)
  - Missing critical info for the detected intent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .intent_recognizer import ClassificationResult, Intent
from .parameter_extractor import ExtractedParams

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Clarification Types
# ══════════════════════════════════════════════════════════════════

class ClarificationType(str, Enum):
    INTENT = "intent"
    TOKEN_DISAMBIGUATION = "token_disambiguation"
    MISSING_PARAM = "missing_param"
    COMPARISON_INCOMPLETE = "comparison_incomplete"
    NONE = "none"


@dataclass
class ClarificationOption:
    """A single selectable option in a clarification prompt."""
    key: str           # "A", "B", "C"
    label: str         # Display text
    value: str         # Machine-readable value


@dataclass
class ClarificationPrompt:
    """Prompt returned to the UI when clarification is needed."""
    ctype: ClarificationType
    message: str
    options: List[ClarificationOption] = field(default_factory=list)
    free_text_allowed: bool = False

    @property
    def needs_response(self) -> bool:
        return self.ctype != ClarificationType.NONE


# ══════════════════════════════════════════════════════════════════
# Known Token Database (stub – would be backed by real DB)
# ══════════════════════════════════════════════════════════════════

# Map of ambiguous ticker → list of (full_name, ticker, description)
_KNOWN_AMBIGUOUS: Dict[str, List[Tuple[str, str, str]]] = {
    "PEPE": [
        ("Pepe", "PEPE", "Original Pepe meme coin on Ethereum"),
        ("Pepe 2.0", "PEPE2.0", "Pepe Second Edition"),
        ("PepeCoin", "PEPECOIN", "Community Pepe fork"),
    ],
    "DOGE": [
        ("Dogecoin", "DOGE", "Original Dogecoin"),
        ("Baby Doge", "BABYDOGE", "Baby Doge Coin"),
    ],
    "SHIB": [
        ("Shiba Inu", "SHIB", "Shiba Inu token on Ethereum"),
        ("ShibaDoge", "SHIBDOGE", "ShibaDoge hybrid token"),
    ],
}


# ══════════════════════════════════════════════════════════════════
# Clarification Engine
# ══════════════════════════════════════════════════════════════════

class ClarificationEngine:
    """
    Evaluate a classification result and decide if clarification is needed.

    Usage::

        engine = ClarificationEngine()
        prompt = engine.check(classification, params)
        if prompt.needs_response:
            # Present prompt to user
            ...
    """

    def __init__(
        self,
        ambiguous_tokens: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
    ):
        self._ambiguous = ambiguous_tokens or _KNOWN_AMBIGUOUS

    def check(
        self,
        classification: ClassificationResult,
        params: ExtractedParams,
    ) -> ClarificationPrompt:
        """
        Run the clarification decision tree.

        Returns ``ClarificationPrompt`` with ``ctype=NONE`` if
        no clarification is needed.
        """
        # ── 1. Intent unclear ─────────────────────────────────────
        if classification.is_ambiguous:
            return self._ask_intent(classification)

        # ── 2. Token disambiguation ───────────────────────────────
        for tok in params.tokens:
            if tok.upper() in self._ambiguous:
                candidates = self._ambiguous[tok.upper()]
                if len(candidates) > 1:
                    return self._ask_token(tok, candidates)

        # ── 3. Missing critical parameters per intent ─────────────
        intent = classification.intent

        if intent == Intent.PORTFOLIO:
            if params.price_context is None or params.price_context.entry_price is None:
                return self._ask_missing_param(
                    "entry_price",
                    "What was your entry price?",
                )

        if intent == Intent.COMPARISON:
            if len(params.tokens) < 2:
                first = params.tokens[0] if params.tokens else "the token"
                return self._ask_comparison_peer(first)

        if intent == Intent.ALERT:
            if params.alert_condition and params.alert_condition.trigger_value is None:
                if params.alert_condition.trigger_type != "condition":
                    return self._ask_missing_param(
                        "alert_price",
                        "At what price should I alert you?",
                    )

        if intent == Intent.ANALYSIS and not params.tokens:
            return self._ask_missing_param(
                "token",
                "Which token would you like me to analyze?",
            )

        # ── No clarification needed ───────────────────────────────
        return ClarificationPrompt(ctype=ClarificationType.NONE, message="")

    # ── Builders ──────────────────────────────────────────────────

    @staticmethod
    def _ask_intent(cls_result: ClassificationResult) -> ClarificationPrompt:
        return ClarificationPrompt(
            ctype=ClarificationType.INTENT,
            message="I'm not sure what you're looking for. Would you like me to:",
            options=[
                ClarificationOption("A", "Recommend top tokens to buy", "discovery"),
                ClarificationOption("B", "Analyze a specific token", "analysis"),
                ClarificationOption("C", "Compare multiple tokens", "comparison"),
                ClarificationOption("D", "Something else", "other"),
            ],
            free_text_allowed=True,
        )

    @staticmethod
    def _ask_token(
        ticker: str,
        candidates: List[Tuple[str, str, str]],
    ) -> ClarificationPrompt:
        options = []
        for idx, (name, sym, desc) in enumerate(candidates):
            key = chr(65 + idx)  # A, B, C...
            options.append(ClarificationOption(
                key=key,
                label=f"{name} ({sym}) – {desc}",
                value=sym,
            ))
        return ClarificationPrompt(
            ctype=ClarificationType.TOKEN_DISAMBIGUATION,
            message=f'Did you mean "{ticker}"? I found multiple matches:',
            options=options,
        )

    @staticmethod
    def _ask_missing_param(
        param_name: str,
        question: str,
    ) -> ClarificationPrompt:
        return ClarificationPrompt(
            ctype=ClarificationType.MISSING_PARAM,
            message=question,
            free_text_allowed=True,
        )

    @staticmethod
    def _ask_comparison_peer(first_token: str) -> ClarificationPrompt:
        return ClarificationPrompt(
            ctype=ClarificationType.COMPARISON_INCOMPLETE,
            message=f"What would you like to compare {first_token} against?",
            free_text_allowed=True,
        )


# ══════════════════════════════════════════════════════════════════
# Convenience: Full Query Pipeline
# ══════════════════════════════════════════════════════════════════

@dataclass
class ParsedQuery:
    """
    Final parsed representation ready for the orchestrator.

    If ``clarification.needs_response`` is True, the UI must ask
    the user before proceeding.
    """
    intent: Intent
    confidence: float
    params: ExtractedParams
    clarification: ClarificationPrompt
    raw_query: str

    @property
    def ready(self) -> bool:
        """True when no clarification is needed and pipeline can proceed."""
        return not self.clarification.needs_response


def parse_user_query(
    query: str,
    recognizer=None,
    extractor=None,
    clarifier=None,
) -> ParsedQuery:
    """
    One-shot convenience: classify → extract → clarify.

    Returns a ParsedQuery that the UI or API layer can act on.
    """
    from .intent_recognizer import IntentRecognizer
    from .parameter_extractor import ParameterExtractor

    recognizer = recognizer or IntentRecognizer()
    extractor = extractor or ParameterExtractor()
    clarifier = clarifier or ClarificationEngine()

    cls_result = recognizer.classify(query)
    params = extractor.extract(
        query,
        intent=cls_result.intent.value,
        tokens=cls_result.tokens,
    )
    clarification = clarifier.check(cls_result, params)

    return ParsedQuery(
        intent=cls_result.intent,
        confidence=cls_result.confidence,
        params=params,
        clarification=clarification,
        raw_query=query,
    )
