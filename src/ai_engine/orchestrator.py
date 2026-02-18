"""
Master Orchestrator
=====================
Step 3.1 – The central coordinator for the PumpIQ AI Synthesis Engine.

The orchestrator implements the 11-step pipeline:

  ┌───────────┐   ┌──────────────┐   ┌───────────────┐
  │ 1. Parse  │──▶│ 2. Collect   │──▶│ 3. Optimize   │
  │   Query   │   │   Data       │   │   Params      │
  └───────────┘   └──────────────┘   └───────┬───────┘
  ┌───────────┐   ┌──────────────┐   ┌───────▼───────┐
  │ 6. LSTM   │◀──│ 5. XGBoost   │◀──│ 4. MTF Gate   │
  │  Patterns │   │   Gate       │   │  Confluence   │
  └─────┬─────┘   └──────────────┘   └───────────────┘
  ┌─────▼─────┐   ┌──────────────┐   ┌───────────────┐
  │ 7. Score  │──▶│ 8. Conflict  │──▶│ 9. Rank &     │
  │ Composite │   │  Detection   │   │   Filter      │
  └───────────┘   └──────────────┘   └───────┬───────┘
  ┌───────────┐   ┌──────────────┐   ┌───────▼───────┐
  │11. Record │◀──│ 11. Record   │◀──│ 10. AI Synth  │
  │ Extended  │   │ (Learn Loop) │   │   Gemini/GPT  │
  └───────────┘   └──────────────┘   └───────────────┘

AI backend: Google Gemini (default) or OpenAI GPT-4o (legacy).
Both expose the same ``chat()`` interface.

Composite Scoring Formula:
    Overall = (news × 0.20 + onchain × 0.30 + technical × 0.35
              + (social / 12 × 10) × 0.10) × (1 - 0.05)
              + ml_signal × 10 × 0.05

Filtering (Step 9):
    - Composite score > 6/10
    - No critical red flags (social red_flags list empty OR score still > threshold)
    - Minimum liquidity met
    - Diversity: max 1 token per project family

Weight Adjustments:
    Bear market  → onchain weight +10%, social −5%
    Bull market  → social weight +5%, technical −5%
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    ConfidenceBreakdown,
    ConflictFlag,
    DataMode,
    EntryExitPlan,
    InvestmentTimeframe,
    MarketCondition,
    QueryType,
    RecommendationSet,
    RecommendationVerdict,
    RiskAssessment,
    RiskLevel,
    RiskTolerance,
    TokenData,
    TokenRecommendation,
    UserConfig,
    UserQuery,
)
from .confidence_scorer import (
    ConfidenceScorer,
    EntryExitCalculator,
    RiskRater,
    confidence_risk_verdict,
)
from .conflict_detector import ConflictDetector
from .gpt_client import GPTClient, GPTResponse
from .gemini_client import GeminiClient, GeminiResponse
from .nlg_engine import NLGEngine
from .prompt_templates import PromptBuilder
from .learning_loop import LearningLoop

logger = logging.getLogger(__name__)

# ── Lazy imports for ML engines (graceful degradation) ────────────
_ml_backtester = None
_lstm_engine = None
_mtf_analyzer = None
_param_optimizer = None
_prediction_tracker = None


def _get_ml_backtester():
    global _ml_backtester
    if _ml_backtester is None:
        try:
            from ..ml_backtester import get_ml_backtester
            _ml_backtester = get_ml_backtester()
        except Exception:
            pass
    return _ml_backtester


def _get_lstm_engine():
    global _lstm_engine
    if _lstm_engine is None:
        try:
            from ..lstm_pattern_engine import get_lstm_engine
            _lstm_engine = get_lstm_engine()
        except Exception:
            pass
    return _lstm_engine


def _get_mtf_analyzer():
    global _mtf_analyzer
    if _mtf_analyzer is None:
        try:
            from ..mtf_analyzer import get_mtf_analyzer
            _mtf_analyzer = get_mtf_analyzer()
        except Exception:
            pass
    return _mtf_analyzer


def _get_param_optimizer():
    global _param_optimizer
    if _param_optimizer is None:
        try:
            from ..parameter_optimizer import get_param_optimizer
            _param_optimizer = get_param_optimizer()
        except Exception:
            pass
    return _param_optimizer


def _get_prediction_tracker():
    global _prediction_tracker
    if _prediction_tracker is None:
        try:
            from ..prediction_tracker import get_prediction_tracker
            _prediction_tracker = get_prediction_tracker()
        except Exception:
            pass
    return _prediction_tracker


# ══════════════════════════════════════════════════════════════════
# Public Interface
# ══════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Central coordinator for the PumpIQ AI synthesis pipeline.

    Accepts either a ``GeminiClient`` (default) or ``GPTClient`` (legacy)
    as the AI backend — both expose the same ``chat()`` interface.

    Usage::

        orch = Orchestrator(
            ai_client=GeminiClient(api_key="AIza..."),
            data_fetcher=my_data_fetcher,
        )
        result = await orch.run(query, config)
    """

    def __init__(
        self,
        ai_client: Any = None,                              # GeminiClient or GPTClient
        gpt_client: Any = None,                              # legacy alias
        data_fetcher: Optional[Callable[..., Any]] = None,
        market_condition: MarketCondition = MarketCondition.SIDEWAYS,
    ):
        # Accept either keyword — ai_client takes priority
        self.gpt = ai_client or gpt_client
        self.data_fetcher = data_fetcher
        self.market = market_condition

        # Sub-engines
        self._scorer = ConfidenceScorer()
        self._rater = RiskRater()
        self._detector = ConflictDetector()
        self._prompt = PromptBuilder()
        self._nlg = NLGEngine()
        self._entry_exit = EntryExitCalculator()
        self._learning = LearningLoop()

        # ML engines (lazy — None until first use, graceful degradation)
        self._ml_backtester = _get_ml_backtester()
        self._lstm_engine = _get_lstm_engine()
        self._mtf_analyzer = _get_mtf_analyzer()
        self._param_optimizer = _get_param_optimizer()
        self._prediction_tracker = _get_prediction_tracker()

    # ── Main entry point ──────────────────────────────────────────

    async def run(
        self,
        query: UserQuery,
        config: UserConfig,
        tokens: Optional[List[TokenData]] = None,
    ) -> RecommendationSet:
        """
        Execute the full 6-step pipeline and return a RecommendationSet.

        Parameters
        ----------
        query : UserQuery
            Parsed user request.
        config : UserConfig
            Active analysis configuration.
        tokens : list[TokenData], optional
            Pre-fetched token data.  If *None*, the orchestrator uses
            ``self.data_fetcher`` to collect data.
        """
        logger.info(
            "Orchestrator.run  query_type=%s  modes=%s  tokens_provided=%s",
            query.query_type.value,
            [m.value for m in config.enabled_modes],
            tokens is not None,
        )

        # ── Step 1: Parse query (already done – UserQuery is pre-parsed) ──
        # Adjust weights based on market conditions & token context
        adjusted_weights = self._adjust_weights(config, self.market)

        # ── Step 2: Collect data ──────────────────────────────────
        if tokens is None:
            tokens = await self._collect_data(query, config)

        if not tokens:
            return self._empty_result(query, config)

        # ── Step 3: Optimize parameters per token (Optuna) ────────
        await self._optimize_parameters(tokens)

        # ── Step 4: Multi-timeframe confluence gate ───────────────
        await self._apply_mtf_confluence(tokens)

        # ── Step 5: ML backtest gate (XGBoost) ────────────────────
        await self._apply_ml_backtest(tokens)

        # ── Step 6: LSTM pattern recognition ──────────────────────
        await self._apply_lstm_patterns(tokens)

        # ── Step 7-9: Score → Detect Conflicts → Rank & Filter ───
        enriched = self._enrich_all(tokens, config, adjusted_weights)
        ranked = self._rank_and_filter(enriched, query, config)

        # ── Step 10: AI synthesis (Gemini or GPT) ─────────────────
        rec_set = await self._synthesize(query, config, ranked)

        # ── Step 11: Record predictions for learning loop ─────────
        self._record_predictions(rec_set)
        self._record_extended_predictions(rec_set)

        return rec_set

    # ══════════════════════════════════════════════════════════════
    # Step 2 – Data Collection (via plug-in or fallback)
    # ══════════════════════════════════════════════════════════════

    async def _collect_data(
        self, query: UserQuery, config: UserConfig,
    ) -> List[TokenData]:
        """
        Collect token data from all enabled modules in parallel.
        Delegates to the injected ``data_fetcher`` callback.
        """
        if self.data_fetcher is None:
            logger.warning("No data_fetcher configured – returning empty list")
            return []

        try:
            tokens = await self.data_fetcher(query, config)
            logger.info("Collected %d tokens from data_fetcher", len(tokens))
            return tokens if tokens else []
        except Exception as exc:
            logger.error("Data collection failed: %s", exc, exc_info=True)
            return []

    # ══════════════════════════════════════════════════════════════
    # Step 3 – Parameter Optimization (Optuna)
    # ══════════════════════════════════════════════════════════════

    async def _optimize_parameters(self, tokens: List[TokenData]):
        """Run per-token parameter optimization using Optuna (if available)."""
        optimizer = self._param_optimizer
        if optimizer is None:
            return

        for token in tokens:
            try:
                coin_id = getattr(token, "coin_id", "") or token.token_ticker.lower()
                result = await optimizer.optimize(coin_id, token.token_ticker)
                if result:
                    # Store optimal params on the token for downstream use
                    if not hasattr(token, "_optimal_params"):
                        object.__setattr__(token, "_optimal_params", None)
                    token._optimal_params = result
                    logger.debug("Optimized params for %s: RSI=%d, BB=%d",
                                 token.token_ticker, result.rsi_period, result.bb_period)
            except Exception as e:
                logger.debug("Param optimization failed for %s: %s", token.token_ticker, e)

    # ══════════════════════════════════════════════════════════════
    # Step 4 – Multi-Timeframe Confluence Gate
    # ══════════════════════════════════════════════════════════════

    async def _apply_mtf_confluence(self, tokens: List[TokenData]):
        """
        Run MTF analysis on each token. Tokens that fail the confluence
        gate (score < 0.5 or daily bearish) get a penalty flag.
        """
        mtf = self._mtf_analyzer
        if mtf is None:
            return

        for token in tokens:
            try:
                coin_id = getattr(token, "coin_id", "") or token.token_ticker.lower()
                result = await mtf.analyze(coin_id, token.token_ticker)
                if result:
                    # Store confluence result on token
                    if not hasattr(token, "_confluence"):
                        object.__setattr__(token, "_confluence", None)
                    token._confluence = result

                    # Apply confluence as a score multiplier
                    if result.blocked:
                        # Hard block: daily bearish or insufficient confluence
                        token.composite_score = max(0, token.composite_score * 0.3)
                        logger.info("MTF BLOCKED %s: %s (confluence=%.2f)",
                                    token.token_ticker, result.dominant_bias, result.confluence_score)
                    else:
                        # Boost/attenuate based on confluence
                        token._confluence_multiplier = result.confluence_score
            except Exception as e:
                logger.debug("MTF analysis failed for %s: %s", token.token_ticker, e)

    # ══════════════════════════════════════════════════════════════
    # Step 5 – ML Backtest Gate (XGBoost)
    # ══════════════════════════════════════════════════════════════

    async def _apply_ml_backtest(self, tokens: List[TokenData]):
        """
        Run XGBoost ML backtest validation on each token.
        Tokens failing ML validation get their score dampened.
        """
        ml = self._ml_backtester
        if ml is None:
            return

        for token in tokens:
            try:
                coin_id = getattr(token, "coin_id", "") or token.token_ticker.lower()
                result = await ml.run(coin_id, token.token_ticker)
                if result:
                    if not hasattr(token, "_ml_result"):
                        object.__setattr__(token, "_ml_result", None)
                    token._ml_result = result

                    # Store ML signal for composite blending
                    if not hasattr(token, "_ml_signal"):
                        object.__setattr__(token, "_ml_signal", 0.5)
                    token._ml_signal = result.latest_prediction

                    if result.accuracy < 0.50:
                        logger.debug("ML backtest low accuracy for %s: %.2f", token.token_ticker, result.accuracy)
            except Exception as e:
                logger.debug("ML backtest failed for %s: %s", token.token_ticker, e)

    # ══════════════════════════════════════════════════════════════
    # Step 6 – LSTM Pattern Recognition
    # ══════════════════════════════════════════════════════════════

    async def _apply_lstm_patterns(self, tokens: List[TokenData]):
        """Run LSTM pattern recognition on each token."""
        lstm = self._lstm_engine
        if lstm is None:
            return

        for token in tokens:
            try:
                coin_id = getattr(token, "coin_id", "") or token.token_ticker.lower()
                prediction = await lstm.predict(coin_id, token.token_ticker)
                if prediction:
                    if not hasattr(token, "_lstm_prediction"):
                        object.__setattr__(token, "_lstm_prediction", None)
                    token._lstm_prediction = prediction
                    logger.debug("LSTM prediction for %s: direction=%s prob=%.2f pattern=%s",
                                 token.token_ticker, prediction.direction,
                                 prediction.probability, prediction.pattern_detected)
            except Exception as e:
                logger.debug("LSTM prediction failed for %s: %s", token.token_ticker, e)

    # ══════════════════════════════════════════════════════════════
    # Composite Scoring
    # ══════════════════════════════════════════════════════════════

    def _compute_composite(
        self,
        token: TokenData,
        config: UserConfig,
        weights: Dict[DataMode, float],
    ) -> float:
        """
        Composite = Σ mode_score × mode_weight  (all scores normalised 0-10).

        Extended formula:
            Overall = (news×w_n + onchain×w_o + technical×w_t
                       + (social/12×10)×w_s) × (1 - ml_w)
                      + ml_signal × 10 × ml_w

        Where ml_w = config.ml_signal_weight (default 0.05).
        Confluence score is applied as a multiplier if available.
        """
        total = 0.0
        weight_sum = 0.0

        for mode in config.enabled_modes:
            score = self._normalised_score(token, mode)
            if score is not None:
                w = weights.get(mode, 0.0)
                total += score * w
                weight_sum += w

        # Re-normalise if not all modes provided data
        if weight_sum > 0 and abs(weight_sum - 1.0) > 0.01:
            total = total / weight_sum

        # Blend ML signal (0-1 → 0-10 scale)
        ml_w = getattr(config, "ml_signal_weight", 0.05)
        ml_signal = getattr(token, "_ml_signal", None)
        if ml_signal is not None and ml_w > 0:
            ml_score = ml_signal * 10.0  # normalise to 0-10
            total = total * (1.0 - ml_w) + ml_score * ml_w

        # Apply confluence multiplier if available
        confluence_mult = getattr(token, "_confluence_multiplier", None)
        if confluence_mult is not None:
            # Scale: 1.0 = full alignment, 0.5 = partial, 0.25 = weak
            total = total * (0.5 + 0.5 * confluence_mult)

        return round(total, 2)

    @staticmethod
    def _normalised_score(token: TokenData, mode: DataMode) -> Optional[float]:
        """Return a 0-10 normalised score for a mode, or None."""
        if mode == DataMode.NEWS and token.news:
            return token.news.score
        if mode == DataMode.ONCHAIN and token.onchain:
            return token.onchain.score
        if mode == DataMode.TECHNICAL and token.technical:
            return token.technical.score
        if mode == DataMode.SOCIAL and token.social:
            return token.social.score / token.social.score_max * 10
        return None

    # ══════════════════════════════════════════════════════════════
    # Steps 3-4-5 combined: enrich each token
    # ══════════════════════════════════════════════════════════════

    def _enrich_all(
        self,
        tokens: List[TokenData],
        config: UserConfig,
        weights: Dict[DataMode, float],
    ) -> List[TokenData]:
        """
        For every token:
          1. Composite score
          2. Conflict detection
          3. Confidence scoring
          4. Risk assessment
        """
        # Get historical accuracy for confidence adjustment
        historical_accuracy = self._learning.get_historical_accuracy()

        for token in tokens:
            # Composite
            token.composite_score = self._compute_composite(token, config, weights)

            # Conflicts
            token.conflicts = self._detector.detect(token)

            # Confidence (with historical accuracy feedback)
            breakdown = self._scorer.compute(
                token, config.enabled_modes, token.conflicts,
                historical_accuracy=historical_accuracy,
            )
            token.confidence = breakdown.final_score

            # Risk
            assessment = self._rater.assess(token, config.enabled_modes)
            token.risk_level = assessment.overall_risk
            token.risk_factors = assessment.risk_factors + assessment.risk_multipliers

        return tokens

    # ══════════════════════════════════════════════════════════════
    # Step 5 – Rank & Filter
    # ══════════════════════════════════════════════════════════════

    def _rank_and_filter(
        self,
        tokens: List[TokenData],
        query: UserQuery,
        config: UserConfig,
    ) -> List[TokenData]:
        """
        Filtering rules (from spec):
          1. Composite score > 6/10
          2. No critical red flags  (social red_flags count ≤ 1, or score > threshold)
          3. Minimum liquidity (from config, default $10K)
          4. Diversity: max 1 token per "project family" (by ticker prefix)

        Then sort by composite_score * confidence (weighted rank) descending
        and return top N.
        """
        candidates: List[TokenData] = []

        for token in tokens:
            # Filter: composite score threshold
            if token.composite_score < 6.0:
                logger.debug(
                    "Filtered %s: composite %.1f < 6.0",
                    token.token_ticker, token.composite_score,
                )
                continue

            # Filter: critical social red flags
            if token.social and len(token.social.red_flags) >= 3:
                logger.debug(
                    "Filtered %s: %d social red flags",
                    token.token_ticker, len(token.social.red_flags),
                )
                continue

            # Filter: minimum liquidity
            if token.onchain and token.onchain.liquidity < config.min_liquidity:
                logger.debug(
                    "Filtered %s: liquidity $%.0f < $%.0f",
                    token.token_ticker,
                    token.onchain.liquidity,
                    config.min_liquidity,
                )
                continue

            candidates.append(token)

        # Sort: composite_score × confidence (higher = better)
        candidates.sort(
            key=lambda t: t.composite_score * t.confidence,
            reverse=True,
        )

        # Diversity filter: unique by first 4 chars of ticker (simple dedup)
        seen_families: set[str] = set()
        diverse: List[TokenData] = []
        for token in candidates:
            family = token.token_ticker[:4].upper()
            if family not in seen_families:
                seen_families.add(family)
                diverse.append(token)

        # Top N
        n = query.num_recommendations
        selected = diverse[:n]
        logger.info(
            "Rank & filter: %d → %d candidates → %d selected (of %d requested)",
            len(tokens), len(candidates), len(selected), n,
        )
        return selected

    # ══════════════════════════════════════════════════════════════
    # Step 6 – AI Synthesis (Gemini or GPT-4o)
    # ══════════════════════════════════════════════════════════════

    async def _synthesize(
        self,
        query: UserQuery,
        config: UserConfig,
        tokens: List[TokenData],
    ) -> RecommendationSet:
        """
        Build prompts, call AI (Gemini / GPT), then construct the RecommendationSet.

        If the AI call fails, falls back to the NLG template engine.
        """
        now = datetime.utcnow()

        # Build prompts
        system_msg = self._prompt.system_prompt(config)

        if not tokens:
            user_msg = self._prompt.no_tokens_prompt()
        elif query.query_type == QueryType.ANALYZE_TOKEN and len(tokens) == 1:
            user_msg = self._prompt.analyze_token_prompt(
                query, config, tokens[0], self.market,
            )
        else:
            user_msg = self._prompt.best_coins_prompt(
                query, config, tokens, self.market,
            )

        # Call AI backend
        gpt_resp = await self.gpt.chat(
            system_prompt=system_msg,
            user_prompt=user_msg,
        )

        # Build recommendations from our own scoring (authoritative)
        recommendations: List[TokenRecommendation] = []
        for rank, token in enumerate(tokens, 1):
            rec = self._build_recommendation(
                rank, token, config, query,
            )
            recommendations.append(rec)

        # Comparative summary from NLG
        final_thoughts = ""
        if len(recommendations) > 1:
            final_thoughts = self._nlg.comparative_summary(recommendations)

        # If AI responded successfully, use its content for final_thoughts
        if gpt_resp.success and gpt_resp.content:
            final_thoughts = gpt_resp.content

        tokens_total = len(tokens)

        return RecommendationSet(
            query=query,
            market_condition=self.market,
            recommendations=recommendations,
            final_thoughts=final_thoughts,
            tokens_analyzed=tokens_total,
            tokens_filtered_out=0,  # computed at higher level if needed
            enabled_modes=config.enabled_modes,
            generated_at=now,
            raw_gpt_response=gpt_resp.content if gpt_resp.success else gpt_resp.error,
        )

    # ── Build a single TokenRecommendation ────────────────────────

    def _build_recommendation(
        self,
        rank: int,
        token: TokenData,
        config: UserConfig,
        query: UserQuery,
    ) -> TokenRecommendation:
        """Construct a fully populated TokenRecommendation with AI Thought Summary."""
        # Confidence
        historical_accuracy = self._learning.get_historical_accuracy()
        breakdown = self._scorer.compute(
            token, config.enabled_modes, token.conflicts,
            historical_accuracy=historical_accuracy,
        )
        # Risk
        assessment = self._rater.assess(token, config.enabled_modes)

        # Verdict via matrix
        verdict = confidence_risk_verdict(
            breakdown.final_score, assessment.overall_risk,
        )

        # Entry / exit plan
        entry_exit = self._entry_exit.compute(
            token,
            assessment.overall_risk,
            query.timeframe,
            query.risk_tolerance,
        )

        # NLG helpers
        thesis = self._nlg.build_core_thesis(token, assessment.overall_risk)
        bullets = self._nlg.build_evidence_bullets(token, config.enabled_modes)
        risks = self._nlg.build_risk_disclosure(token, assessment)

        # ── AI Thought Summary (unique feature) ───────────────────
        ai_thought = self._build_ai_thought_summary(
            token, verdict, breakdown, assessment, config,
        )

        # Determine market regime from technical data
        market_regime = ""
        if token.technical:
            market_regime = getattr(token.technical, "market_regime", "") or ""
            if not market_regime and hasattr(token.technical, "trend"):
                market_regime = "trending" if token.technical.trend != "sideways" else "ranging"

        return TokenRecommendation(
            rank=rank,
            token_name=token.token_name,
            token_ticker=token.token_ticker,
            current_price=token.current_price,
            verdict=verdict,
            confidence=breakdown.final_score,
            confidence_breakdown=breakdown,
            risk_level=assessment.overall_risk,
            risk_assessment=assessment,
            composite_score=token.composite_score,
            entry_exit=entry_exit,
            core_thesis=thesis,
            key_data_points=bullets,
            risks_and_concerns=risks,
            conflicts=token.conflicts,
            ai_thought_summary=ai_thought,
            news_analysis=token.news.summary if token.news else "",
            onchain_analysis=token.onchain.summary if token.onchain else "",
            technical_analysis=token.technical.summary if token.technical else "",
            social_analysis=token.social.summary if token.social else "",
            market_regime=market_regime,
            generated_at=datetime.utcnow(),
        )

    # ── AI Thought Summary Generator ─────────────────────────────

    def _build_ai_thought_summary(
        self,
        token: TokenData,
        verdict: RecommendationVerdict,
        breakdown: ConfidenceBreakdown,
        assessment: RiskAssessment,
        config: UserConfig,
    ) -> str:
        """
        Generate a short AI Thought Summary explaining WHY the decision was made.

        This is the unique differentiator — each recommendation includes a
        transparent explanation of the AI's internal reasoning process.
        """
        thoughts: List[str] = []

        # 1. Trend consistency evaluation
        if token.technical:
            trend = token.technical.trend
            consistency = getattr(token.technical, "trend_consistency", 0)
            if consistency > 0.7:
                thoughts.append(
                    f"The {trend} trend is highly consistent ({consistency:.0%} of candles align)"
                )
            elif consistency > 0.5:
                thoughts.append(
                    f"I see a {trend} trend but with moderate consistency ({consistency:.0%})"
                )
            else:
                thoughts.append(
                    f"The trend direction is unclear — only {consistency:.0%} alignment"
                )

        # 2. Volatility assessment
        if token.technical:
            vol_state = getattr(token.technical, "volatility_state", "normal")
            if vol_state == "expanding":
                thoughts.append(
                    "Volatility is expanding — large moves likely, which increases both opportunity and risk"
                )
            elif vol_state == "contracting":
                thoughts.append(
                    "Volatility is contracting — a breakout in either direction is forming"
                )

        # 3. Liquidity pressure
        if token.technical:
            pressure = getattr(token.technical, "liquidity_pressure", "neutral")
            if pressure == "buying":
                thoughts.append("I detect buying pressure — volume confirms price moves up")
            elif pressure == "selling":
                thoughts.append("There's selling pressure — volume is rising as price falls")

        # 4. Volume anomaly (whale activity)
        if token.technical:
            abnormal = getattr(token.technical, "abnormal_volume", False)
            if abnormal:
                vol_score = getattr(token.technical, "volume_anomaly_score", 0)
                thoughts.append(
                    f"ALERT: Abnormal volume detected (anomaly score {vol_score:.1f}/10) — possible whale activity"
                )

        # 5. Historical pattern similarity
        if token.technical and token.technical.pattern != "None":
            thoughts.append(
                f"I recognize a {token.technical.pattern} pattern — historically this has follow-through"
            )

        # 6. Data confidence note
        modes_count = len(config.enabled_modes)
        if modes_count >= 4:
            thoughts.append("All 4 data modes are active, giving me a comprehensive view")
        elif modes_count == 1:
            thoughts.append(
                "Only 1 data mode is enabled — my confidence is capped due to limited perspective"
            )

        # 7. Conflict summary
        if token.conflicts:
            major = sum(1 for c in token.conflicts if c.severity.value == "major")
            if major > 0:
                thoughts.append(
                    f"I found {major} major signal conflict(s) between modules, which lowered my confidence"
                )

        # 8. Final verdict reasoning
        if verdict in (RecommendationVerdict.STRONG_BUY, RecommendationVerdict.MODERATE_BUY):
            thoughts.append(
                f"DECISION: {verdict.value} — the data aligns positively across key dimensions"
            )
        elif verdict == RecommendationVerdict.AVOID:
            thoughts.append(
                "DECISION: Avoid — insufficient evidence or too many red flags to recommend"
            )
        elif verdict == RecommendationVerdict.WATCH:
            thoughts.append(
                "DECISION: Watch — promising signals but needs more confirmation before acting"
            )
        elif verdict == RecommendationVerdict.SELL:
            thoughts.append(
                "DECISION: Sell — negative indicators outweigh positives"
            )

        return " | ".join(thoughts) if thoughts else (
            f"{verdict.value} with {breakdown.final_score:.1f}/10 confidence "
            f"based on composite analysis."
        )

    # ══════════════════════════════════════════════════════════════
    # Weight Adjustment (market condition + token context)
    # ══════════════════════════════════════════════════════════════

    def _adjust_weights(
        self,
        config: UserConfig,
        market: MarketCondition,
    ) -> Dict[DataMode, float]:
        """
        Return adjusted weights based on market conditions.

        New defaults: Technical 0.35, On-chain 0.30, News 0.20, Social 0.10
        ML signal weight (0.05) is applied separately during composite scoring.

        Bear  → onchain +10 pp, social −5 pp
        Bull  → social  +5 pp,  technical −5 pp
        """
        w = dict(config.mode_weights)

        if market == MarketCondition.BEAR:
            w[DataMode.ONCHAIN] = w.get(DataMode.ONCHAIN, 0.30) + 0.10
            w[DataMode.SOCIAL] = max(0.05, w.get(DataMode.SOCIAL, 0.10) - 0.05)
        elif market == MarketCondition.BULL:
            w[DataMode.SOCIAL] = w.get(DataMode.SOCIAL, 0.10) + 0.05
            w[DataMode.TECHNICAL] = max(0.05, w.get(DataMode.TECHNICAL, 0.35) - 0.05)

        # Re-normalise to 1.0
        total = sum(w.values())
        if total > 0:
            w = {k: round(v / total, 4) for k, v in w.items()}

        return w

    # ══════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════

    def _empty_result(
        self, query: UserQuery, config: UserConfig,
    ) -> RecommendationSet:
        """Return a RecommendationSet with zero recommendations."""
        return RecommendationSet(
            query=query,
            market_condition=self.market,
            recommendations=[],
            final_thoughts=(
                "No tokens met the minimum criteria (composite > 6/10, "
                "liquidity met, no critical red flags). "
                "Sometimes the best trade is no trade."
            ),
            tokens_analyzed=0,
            tokens_filtered_out=0,
            enabled_modes=config.enabled_modes,
            generated_at=datetime.now(),
        )

    # ══════════════════════════════════════════════════════════════
    # Step 7 – Record Predictions for Learning Loop
    # ══════════════════════════════════════════════════════════════

    def _record_predictions(self, rec_set: RecommendationSet):
        """Record each recommendation as a trackable prediction for the learning loop."""
        for rec in rec_set.recommendations:
            try:
                target_price = rec.entry_exit.target_1 if rec.entry_exit else 0
                stop_loss = rec.entry_exit.stop_loss if rec.entry_exit else 0

                self._learning.record_prediction(
                    token_ticker=rec.token_ticker,
                    token_name=rec.token_name,
                    verdict=rec.verdict.value if hasattr(rec.verdict, "value") else str(rec.verdict),
                    confidence=rec.confidence,
                    composite_score=rec.composite_score,
                    price_at_prediction=rec.current_price,
                    target_price=target_price,
                    stop_loss_price=stop_loss,
                    market_condition=rec_set.market_condition.value if hasattr(rec_set.market_condition, "value") else str(rec_set.market_condition),
                    market_regime=rec.market_regime,
                    risk_level=rec.risk_level.value if hasattr(rec.risk_level, "value") else str(rec.risk_level),
                    enabled_modes=[m.value for m in rec_set.enabled_modes],
                    ai_thought_summary=rec.ai_thought_summary,
                )
            except Exception as exc:
                logger.debug("Failed to record prediction for %s: %s", rec.token_ticker, exc)

    def _record_extended_predictions(self, rec_set: RecommendationSet):
        """Record predictions in the extended PredictionTracker for per-token/regime accuracy."""
        tracker = self._prediction_tracker
        if tracker is None:
            return

        for rec in rec_set.recommendations:
            try:
                import hashlib
                pred_id = hashlib.sha256(
                    f"{rec.token_ticker}|{rec.current_price}|{datetime.utcnow().isoformat()}".encode()
                ).hexdigest()[:16]

                target_price = rec.entry_exit.target_1 if rec.entry_exit else 0
                stop_loss = rec.entry_exit.stop_loss if rec.entry_exit else 0

                # Determine predicted direction
                verdict_str = rec.verdict.value if hasattr(rec.verdict, "value") else str(rec.verdict)
                if verdict_str in ("Strong Buy", "Moderate Buy", "Cautious Buy"):
                    direction = "up"
                elif verdict_str in ("Sell", "Avoid"):
                    direction = "down"
                else:
                    direction = "flat"

                # Collect indicator combo
                indicators = []
                if rec.confidence_breakdown:
                    if getattr(rec.confidence_breakdown, "signal_strength_detail", ""):
                        indicators.append(rec.confidence_breakdown.signal_strength_detail)
                if hasattr(rec, "_ml_signal") and rec._ml_signal is not None:
                    indicators.append(f"ML_{rec._ml_signal:.2f}")

                # Get ML/LSTM signals from token data if available
                ml_signal = getattr(rec, "_ml_signal", None)
                lstm_signal = None
                confluence_score = None

                tracker.record(
                    prediction_id=pred_id,
                    token_ticker=rec.token_ticker,
                    entry_price=rec.current_price,
                    target_price=target_price,
                    stop_loss_price=stop_loss,
                    predicted_direction=direction,
                    confidence_score=rec.confidence,
                    composite_score=rec.composite_score,
                    market_regime=rec.market_regime or "unknown",
                    indicator_combo=indicators,
                    ml_signal=ml_signal,
                    lstm_signal=lstm_signal,
                    confluence_score=confluence_score,
                )
            except Exception as exc:
                logger.debug("Failed to record extended prediction for %s: %s", rec.token_ticker, exc)
