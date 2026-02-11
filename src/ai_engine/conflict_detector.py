"""
Conflict Detector
===================
Step 3.1 – Step 4: Identifies when different data modules disagree.

Rules (from spec):
  news > 7  AND onchain < 5  → "News hype but weak fundamentals"
  tech > 8  AND social < 4   → "Strong chart but no community interest"
  onchain > 8 AND tech < 4   → "Solid fundamentals but poor chart setup"

Additional heuristic rules are included for deeper conflict detection.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .models import (
    ConflictFlag,
    ConflictSeverity,
    DataMode,
    TokenData,
)

logger = logging.getLogger(__name__)


def _norm_social(score: float, score_max: float = 12) -> float:
    """Normalise social score from 0-12 to 0-10."""
    return score / score_max * 10


class ConflictDetector:
    """
    Scans a TokenData object for disagreements between modules.

    Usage::

        detector = ConflictDetector()
        conflicts = detector.detect(token)
    """

    def detect(self, token: TokenData) -> List[ConflictFlag]:
        conflicts: List[ConflictFlag] = []

        ns = token.news.score if token.news else None
        oc = token.onchain.score if token.onchain else None
        tc = token.technical.score if token.technical else None
        sc = _norm_social(token.social.score, token.social.score_max) if token.social else None

        # ── Spec-defined conflicts ──────────────────────────────

        # 1. News hype but weak fundamentals
        if ns is not None and oc is not None:
            if ns > 7 and oc < 5:
                conflicts.append(ConflictFlag(
                    severity=ConflictSeverity.MAJOR,
                    module_a=DataMode.NEWS,
                    module_b=DataMode.ONCHAIN,
                    description=(
                        f"News hype but weak fundamentals – news sentiment "
                        f"({ns:.1f}/10) is bullish while on-chain health "
                        f"({oc:.1f}/10) raises concerns."
                    ),
                    confidence_penalty=2.0,
                ))

        # 2. Strong chart but no community interest
        if tc is not None and sc is not None:
            if tc > 8 and sc < 4:
                conflicts.append(ConflictFlag(
                    severity=ConflictSeverity.MAJOR,
                    module_a=DataMode.TECHNICAL,
                    module_b=DataMode.SOCIAL,
                    description=(
                        f"Strong chart but no community interest – technical "
                        f"score ({tc:.1f}/10) is strong but social buzz "
                        f"({sc:.1f}/10) is very low."
                    ),
                    confidence_penalty=2.0,
                ))

        # 3. Solid fundamentals but poor chart setup
        if oc is not None and tc is not None:
            if oc > 8 and tc < 4:
                conflicts.append(ConflictFlag(
                    severity=ConflictSeverity.MAJOR,
                    module_a=DataMode.ONCHAIN,
                    module_b=DataMode.TECHNICAL,
                    description=(
                        f"Solid fundamentals but poor chart setup – on-chain "
                        f"health ({oc:.1f}/10) is excellent but the chart "
                        f"({tc:.1f}/10) looks weak."
                    ),
                    confidence_penalty=2.0,
                ))

        # ── Extended heuristic conflicts ────────────────────────

        # 4. Social hype but poor on-chain (pump-and-dump signal)
        if sc is not None and oc is not None:
            if sc > 7 and oc < 4:
                conflicts.append(ConflictFlag(
                    severity=ConflictSeverity.MAJOR,
                    module_a=DataMode.SOCIAL,
                    module_b=DataMode.ONCHAIN,
                    description=(
                        f"High social buzz ({sc:.1f}/10) paired with weak "
                        f"on-chain ({oc:.1f}/10) may indicate pump-and-dump."
                    ),
                    confidence_penalty=2.0,
                ))

        # 5. News bearish but technical bullish
        if ns is not None and tc is not None:
            if ns < 4 and tc > 7:
                conflicts.append(ConflictFlag(
                    severity=ConflictSeverity.MINOR,
                    module_a=DataMode.NEWS,
                    module_b=DataMode.TECHNICAL,
                    description=(
                        f"Negative news ({ns:.1f}/10) contrasts with bullish "
                        f"chart ({tc:.1f}/10) – market may not have priced in yet."
                    ),
                    confidence_penalty=1.0,
                ))

        # 6. On-chain strong, social bearish (hidden gem?)
        if oc is not None and sc is not None:
            if oc > 7 and sc < 4:
                conflicts.append(ConflictFlag(
                    severity=ConflictSeverity.MINOR,
                    module_a=DataMode.ONCHAIN,
                    module_b=DataMode.SOCIAL,
                    description=(
                        f"On-chain is strong ({oc:.1f}/10) but community "
                        f"sentiment is low ({sc:.1f}/10) – could be undiscovered."
                    ),
                    confidence_penalty=1.0,
                ))

        # 7. All modules moderately bullish but none strongly so
        scores = [s for s in [ns, oc, tc, sc] if s is not None]
        if len(scores) >= 3:
            if all(5 <= s <= 7 for s in scores):
                conflicts.append(ConflictFlag(
                    severity=ConflictSeverity.MINOR,
                    module_a=DataMode.NEWS,
                    module_b=DataMode.ONCHAIN,
                    description=(
                        "All signals are moderately positive but none are "
                        "strongly bullish – conviction is limited."
                    ),
                    confidence_penalty=0.5,
                ))

        # 8. Social red flags present despite high composite scores
        if token.social and token.social.red_flags:
            avg = sum(scores) / len(scores) if scores else 0
            if avg > 6:
                for flag in token.social.red_flags[:2]:
                    conflicts.append(ConflictFlag(
                        severity=ConflictSeverity.MAJOR,
                        module_a=DataMode.SOCIAL,
                        module_b=DataMode.SOCIAL,
                        description=f"Social red flag despite positive scores: {flag}",
                        confidence_penalty=1.5,
                    ))

        return conflicts
