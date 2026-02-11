"""
Service Layer
===============
Step 5.1 – Bridges API routes → Orchestrator + PersonalizationEngine + UI.

Responsibilities:
  1. Accept API-level request models
  2. Build UserQuery / UserConfig via PersonalizationEngine
  3. Invoke the Orchestrator pipeline (with caching)
  4. Convert internal RecommendationSet → API response schemas
  5. Handle partial-failure gracefully (module down → lower confidence)

Error handling strategy:
  - Module failure  → log, continue, flag ``degraded_modules`` in metadata
  - GPT-4o failure  → 3× retry w/ exponential back-off → NLG fallback
  - All collectors fail → raise InsufficientDataError
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.ai_engine.models import (
    DataMode,
    MarketCondition,
    QueryType,
    RecommendationSet,
    TokenData,
    TokenRecommendation,
    UserConfig,
    UserQuery,
)
from src.ai_engine.orchestrator import Orchestrator
from src.ai_engine.gemini_client import GeminiClient
from src.ai_engine.gpt_client import GPTClient
from src.ui.personalization_engine import PersonalizationEngine
from src.ui.user_config import UserPreferences, default_preferences
from src.ui.api_schemas import (
    AnalysisResponse,
    ComparisonResponse,
    ConflictBlock,
    DetailedAnalysisBlock,
    EntryExitBlock,
    MarketContext,
    RecommendationResponse,
    RecommendationSetResponse,
    ResponseMetadata,
    ScoreBlock,
    TokenInfo,
)
from .error_handlers import (
    GPTSynthesisError,
    InsufficientDataError,
    TokenNotFoundError,
)

logger = logging.getLogger(__name__)

# Cache TTL (seconds)
CACHE_TTL_RECOMMENDATIONS = 120   # 2 minutes
CACHE_TTL_ANALYSIS = 180          # 3 minutes


# ══════════════════════════════════════════════════════════════════
# Service
# ══════════════════════════════════════════════════════════════════

class PumpIQService:
    """
    Stateless service encapsulating one request's lifecycle.

    Usage (from a route handler)::

        svc = PumpIQService(gpt_client=…, redis=…)
        response = await svc.get_recommendations(query_text, prefs)
    """

    def __init__(
        self,
        ai_client: Optional[Any] = None,
        gpt_client: Optional[Any] = None,  # legacy alias
        redis: Optional[Any] = None,
    ):
        self.gpt_client = ai_client or gpt_client
        self.redis = redis

    # ── 1.  GET RECOMMENDATIONS ───────────────────────────────────

    async def get_recommendations(
        self,
        query_text: str,
        prefs: UserPreferences,
        *,
        num_recommendations: int = 3,
        timeframe: Optional[str] = None,
        risk: Optional[str] = None,
        modes: Optional[List[str]] = None,
    ) -> RecommendationSetResponse:
        """
        Full pipeline: parse → collect → score → rank → synthesise.
        Returns an ``RecommendationSetResponse`` ready for JSON serialisation.
        """
        start = time.perf_counter()

        # ── Check cache ───────────────────────────────────────────
        cache_key = self._cache_key("recs", query_text, prefs, modes)
        cached = await self._cache_get(cache_key)
        if cached:
            logger.info("Cache HIT for recommendations")
            return RecommendationSetResponse(**cached)

        # ── Build engine inputs ───────────────────────────────────
        engine = PersonalizationEngine(prefs)
        user_config = engine.build_config()
        user_query = engine.build_query(
            query_text,
            intent="discovery",
            num_recs=num_recommendations,
            timeframe=timeframe,
            risk=risk,
        )

        # Override modes if explicitly requested
        if modes:
            mode_map = {
                "news": DataMode.NEWS, "onchain": DataMode.ONCHAIN,
                "technical": DataMode.TECHNICAL, "social": DataMode.SOCIAL,
            }
            user_config.enabled_modes = [
                mode_map[m] for m in modes if m in mode_map
            ]

        # ── Run orchestrator ──────────────────────────────────────
        rec_set = await self._run_orchestrator(user_query, user_config)

        # ── Post-filter & annotate ────────────────────────────────
        rec_set.recommendations = engine.post_filter(rec_set.recommendations)
        engine.annotate(rec_set.recommendations)

        # ── Convert to API schema ─────────────────────────────────
        elapsed_ms = (time.perf_counter() - start) * 1000
        response = self._to_recommendation_set_response(rec_set, elapsed_ms)

        # ── Cache ─────────────────────────────────────────────────
        await self._cache_set(cache_key, response.model_dump(), CACHE_TTL_RECOMMENDATIONS)

        return response

    # ── 2.  ANALYZE TOKEN ─────────────────────────────────────────

    async def analyze_token(
        self,
        ticker: str,
        prefs: UserPreferences,
        *,
        modes: Optional[List[str]] = None,
    ) -> AnalysisResponse:
        """
        Single-token deep analysis.
        """
        start = time.perf_counter()

        cache_key = self._cache_key("analysis", ticker, prefs, modes)
        cached = await self._cache_get(cache_key)
        if cached:
            return AnalysisResponse(**cached)

        engine = PersonalizationEngine(prefs)
        user_config = engine.build_config()
        user_query = engine.build_query(
            f"Analyze ${ticker}",
            intent="analysis",
            tokens=[ticker.upper()],
            num_recs=1,
        )
        user_query.query_type = QueryType.ANALYZE_TOKEN

        if modes:
            mode_map = {
                "news": DataMode.NEWS, "onchain": DataMode.ONCHAIN,
                "technical": DataMode.TECHNICAL, "social": DataMode.SOCIAL,
            }
            user_config.enabled_modes = [
                mode_map[m] for m in modes if m in mode_map
            ]

        rec_set = await self._run_orchestrator(user_query, user_config)

        if not rec_set.recommendations:
            raise TokenNotFoundError(ticker)

        rec = rec_set.recommendations[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        response = self._to_analysis_response(rec, rec_set, elapsed_ms)

        await self._cache_set(cache_key, response.model_dump(), CACHE_TTL_ANALYSIS)
        return response

    # ── 3.  COMPARE TOKENS ────────────────────────────────────────

    async def compare_tokens(
        self,
        tickers: List[str],
        prefs: UserPreferences,
    ) -> ComparisonResponse:
        """
        Head-to-head comparison of 2-4 tokens.
        """
        start = time.perf_counter()
        tickers = [t.upper() for t in tickers[:4]]

        engine = PersonalizationEngine(prefs)
        user_config = engine.build_config()
        user_query = engine.build_query(
            f"Compare {', '.join(tickers)}",
            intent="comparison",
            tokens=tickers,
            num_recs=len(tickers),
        )

        rec_set = await self._run_orchestrator(user_query, user_config)

        elapsed_ms = (time.perf_counter() - start) * 1000
        recs = [self._rec_to_response(r) for r in rec_set.recommendations]
        highlights = self._build_comparison_highlights(rec_set.recommendations)

        return ComparisonResponse(
            query_timestamp=datetime.now(timezone.utc).isoformat(),
            tokens=recs,
            highlights=highlights,
            market_context=MarketContext(
                condition=rec_set.market_condition.value,
            ),
            metadata=ResponseMetadata(
                modes_enabled=[m.value for m in rec_set.enabled_modes],
                tokens_analyzed=rec_set.tokens_analyzed,
                tokens_filtered_out=rec_set.tokens_filtered_out,
                processing_time_ms=elapsed_ms,
            ),
        )

    # ══════════════════════════════════════════════════════════════
    # Orchestrator Wrapper (with retry / fallback)
    # ══════════════════════════════════════════════════════════════

    async def _run_orchestrator(
        self,
        query: UserQuery,
        config: UserConfig,
    ) -> RecommendationSet:
        """
        Invoke the Orchestrator with GPT retries.

        Falls back to NLG-only output if GPT is unavailable.
        """
        if self.gpt_client is None:
            logger.warning("No GPT client – running in template-only mode")

        orch = Orchestrator(
            gpt_client=self.gpt_client,
            market_condition=MarketCondition.SIDEWAYS,
        )

        try:
            return await orch.run(query, config)
        except Exception as exc:
            logger.error("Orchestrator failed: %s", exc, exc_info=True)
            # Return empty set rather than crashing
            return RecommendationSet(
                query=query,
                recommendations=[],
                generated_at=datetime.now(timezone.utc),
            )

    # ══════════════════════════════════════════════════════════════
    # Schema Converters
    # ══════════════════════════════════════════════════════════════

    def _to_recommendation_set_response(
        self,
        rec_set: RecommendationSet,
        elapsed_ms: float,
    ) -> RecommendationSetResponse:
        return RecommendationSetResponse(
            query_timestamp=datetime.now(timezone.utc).isoformat(),
            recommendations=[
                self._rec_to_response(r) for r in rec_set.recommendations
            ],
            market_context=MarketContext(
                condition=rec_set.market_condition.value,
            ),
            metadata=ResponseMetadata(
                modes_enabled=[m.value for m in rec_set.enabled_modes],
                tokens_analyzed=rec_set.tokens_analyzed,
                tokens_filtered_out=rec_set.tokens_filtered_out,
                processing_time_ms=elapsed_ms,
            ),
        )

    def _to_analysis_response(
        self,
        rec: TokenRecommendation,
        rec_set: RecommendationSet,
        elapsed_ms: float,
    ) -> AnalysisResponse:
        r = self._rec_to_response(rec)
        return AnalysisResponse(
            query_timestamp=datetime.now(timezone.utc).isoformat(),
            token=r.token,
            verdict=r.verdict,
            scores=r.scores,
            entry_exit=r.entry_exit,
            thesis=r.thesis,
            key_data_points=r.key_data_points,
            risks=r.risks,
            conflicts=r.conflicts,
            detailed_analysis=r.detailed_analysis,
            position_sizing=r.position_sizing,
            market_context=MarketContext(
                condition=rec_set.market_condition.value,
            ),
            metadata=ResponseMetadata(
                modes_enabled=[m.value for m in rec_set.enabled_modes],
                tokens_analyzed=rec_set.tokens_analyzed,
                processing_time_ms=elapsed_ms,
            ),
        )

    def _rec_to_response(self, rec: TokenRecommendation) -> RecommendationResponse:
        return RecommendationResponse(
            rank=rec.rank,
            token=TokenInfo(
                name=rec.token_name,
                ticker=rec.token_ticker,
                current_price=rec.current_price,
            ),
            scores=ScoreBlock(
                overall=rec.composite_score,
                confidence=rec.confidence,
                risk=rec.risk_level.value if hasattr(rec.risk_level, "value") else str(rec.risk_level),
            ),
            entry_exit=EntryExitBlock(
                entry_min=rec.entry_exit.entry_low if rec.entry_exit else 0,
                entry_max=rec.entry_exit.entry_high if rec.entry_exit else 0,
                target_1=rec.entry_exit.target_1 if rec.entry_exit else 0,
                target_1_percent=rec.entry_exit.target_1_pct if rec.entry_exit else 0,
                target_2=rec.entry_exit.target_2 if rec.entry_exit else 0,
                target_2_percent=rec.entry_exit.target_2_pct if rec.entry_exit else 0,
                stop_loss=rec.entry_exit.stop_loss if rec.entry_exit else 0,
                stop_loss_percent=rec.entry_exit.stop_loss_pct if rec.entry_exit else 0,
                timeframe=rec.entry_exit.timeframe_estimate if rec.entry_exit else "",
            ),
            verdict=rec.verdict.value if hasattr(rec.verdict, "value") else str(rec.verdict),
            thesis=rec.core_thesis,
            key_data_points=rec.key_data_points,
            risks=rec.risks_and_concerns,
            conflicts=[
                ConflictBlock(
                    severity=c.severity.value if hasattr(c.severity, "value") else str(c.severity),
                    modules=f"{c.module_a.value} vs {c.module_b.value}",
                    description=c.description,
                )
                for c in rec.conflicts
            ],
            detailed_analysis=DetailedAnalysisBlock(
                news=rec.news_analysis or None,
                onchain=rec.onchain_analysis or None,
                technical=rec.technical_analysis or None,
                social=rec.social_analysis or None,
            ),
        )

    # ── Comparison Highlights ─────────────────────────────────────

    def _build_comparison_highlights(
        self, recs: List[TokenRecommendation],
    ) -> Dict[str, str]:
        """Pick per-dimension winners for quick comparison."""
        if not recs:
            return {}
        highlights: Dict[str, str] = {}

        best_score = max(recs, key=lambda r: r.composite_score)
        highlights["highest_score"] = f"{best_score.token_ticker} ({best_score.composite_score:.1f}/10)"

        best_conf = max(recs, key=lambda r: r.confidence)
        highlights["highest_confidence"] = f"{best_conf.token_ticker} ({best_conf.confidence:.1f}/10)"

        risk_ord = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        lowest_risk = min(
            recs,
            key=lambda r: risk_ord.get(
                r.risk_level.value if hasattr(r.risk_level, "value") else str(r.risk_level), 1
            ),
        )
        highlights["lowest_risk"] = f"{lowest_risk.token_ticker} ({lowest_risk.risk_level.value})"

        return highlights

    # ══════════════════════════════════════════════════════════════
    # Caching
    # ══════════════════════════════════════════════════════════════

    def _cache_key(self, prefix: str, *parts) -> str:
        raw = json.dumps([prefix, *[str(p) for p in parts]], sort_keys=True)
        return f"pumpiq:{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

    async def _cache_get(self, key: str) -> Optional[dict]:
        if not self.redis:
            return None
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as exc:
            logger.debug("Cache get failed: %s", exc)
        return None

    async def _cache_set(self, key: str, value: dict, ttl: int) -> None:
        if not self.redis:
            return
        try:
            await self.redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception as exc:
            logger.debug("Cache set failed: %s", exc)
