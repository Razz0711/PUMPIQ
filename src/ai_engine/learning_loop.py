"""
Continuous Learning & Feedback Loop  (Supabase Edition)
=======================================================
Tracks prediction accuracy, compares AI decisions with actual outcomes,
and adapts strategies based on market regime changes.

All data is persisted in **Supabase (PostgreSQL)** so it survives
Render/Vercel deploys and service restarts.

Tables (run database/supabase_schema.sql once):
  - ll_predictions
  - ll_strategy_adjustments
  - ll_accuracy_snapshots

Features:
  - Records every recommendation as a trackable prediction
  - Evaluates predictions after 1 h (short-term) and 7 d against actual prices
  - Computes accuracy metrics per market regime, direction, and timeframe
  - Generates strategy adjustments based on performance
  - Adapts confidence modifiers based on historical accuracy

The learning loop makes the AI **evolving, not static**.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supabase helper
# ---------------------------------------------------------------------------

def _sb():
    """Return the Supabase client singleton (lazy import)."""
    from supabase_db import get_supabase
    return get_supabase()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LearningLoop:
    """
    Continuous learning engine that tracks and evaluates predictions.
    Uses Supabase (PostgreSQL) for persistent storage.

    Usage::

        loop = LearningLoop()
        loop.record_prediction(token_ticker="BTC", verdict="Strong Buy", ...)
        await loop.evaluate_pending(cg_collector)
        stats = loop.get_performance_stats()
        adjustments = loop.generate_adjustments()
    """

    def __init__(self):
        # Quick probe - warns if migration SQL hasn't been run yet
        try:
            _sb().table("ll_predictions").select("id", count="exact").limit(0).execute()
        except Exception as e:
            logger.warning(
                "LearningLoop: ll_predictions table not found - run "
                "database/supabase_schema.sql in your Supabase SQL Editor. (%s)",
                e,
            )
        self._backfill_done = False

    # ==================================================================
    # Backfill — Seed ll_predictions from existing trade history
    # ==================================================================

    def backfill_from_trades(self) -> int:
        """
        Populate ll_predictions from trade_orders + trade_positions.
        Creates one prediction per BUY/SHORT order and immediately evaluates
        it if the position is already closed (has an exit price + P&L).

        Safe to call multiple times — uses prediction_id dedup (SHA-256 of
        coin_id + entry_price + timestamp).

        Returns number of predictions created.
        """
        if self._backfill_done:
            return 0

        sb = _sb()
        created = 0

        try:
            # 1. Get ALL buy/short orders (the entry trades)
            res = (
                sb.table("trade_orders")
                .select("*")
                .in_("action", ["BUY", "SHORT"])
                .order("created_at", desc=True)
                .limit(500)
                .execute()
            )
            orders = res.data or []
            if not orders:
                self._backfill_done = True
                return 0

            # 2. Get ALL positions (both open and closed) for matching
            pos_res = (
                sb.table("trade_positions")
                .select("*")
                .order("opened_at", desc=True)
                .limit(500)
                .execute()
            )
            positions = {p["id"]: p for p in (pos_res.data or [])}

            # 3. Check which prediction_ids already exist to avoid duplicates
            existing_ids = set()
            try:
                ex_res = sb.table("ll_predictions").select("prediction_id").limit(5000).execute()
                existing_ids = {r["prediction_id"] for r in (ex_res.data or [])}
            except Exception:
                pass

            for order in orders:
                coin_id = order.get("coin_id", "")
                price = float(order.get("price", 0))
                ts = order.get("created_at", "")
                action = order.get("action", "BUY")
                ai_score = int(order.get("ai_score", 0))
                symbol = order.get("symbol", "")
                user_id = order.get("user_id", 0)
                position_id = order.get("position_id")

                if price <= 0 or not coin_id:
                    continue

                # Generate deterministic prediction_id from order data
                prediction_id = hashlib.sha256(
                    f"{coin_id}|{price}|{ts}".encode()
                ).hexdigest()[:16]

                if prediction_id in existing_ids:
                    continue  # already backfilled

                # Determine verdict/direction from action
                if action == "SHORT":
                    verdict = "bearish"
                    predicted_direction = "down"
                else:
                    verdict = "bullish"
                    predicted_direction = "up"

                # Resolve a readable ticker name:
                # Prefer position's coin_name/symbol over raw coin_id
                pos_for_name = positions.get(position_id) if position_id else None
                display_ticker = (
                    (pos_for_name.get("symbol") or pos_for_name.get("coin_name") or coin_id)
                    if pos_for_name else (symbol or coin_id)
                )
                # Shorten contract addresses
                if len(display_ticker) > 20:
                    display_ticker = display_ticker[:6] + "..." + display_ticker[-4:]

                row = {
                    "prediction_id": prediction_id,
                    "user_id": user_id,
                    "token_ticker": display_ticker,
                    "token_name": symbol or (pos_for_name.get("coin_name", "") if pos_for_name else ""),
                    "verdict": verdict,
                    "confidence": ai_score / 10.0 if ai_score > 0 else 5.0,
                    "composite_score": float(ai_score),
                    "predicted_direction": predicted_direction,
                    "price_at_prediction": price,
                    "target_price": 0,
                    "stop_loss_price": 0,
                    "market_condition": "unknown",
                    "market_regime": "unknown",
                    "risk_level": "MEDIUM",
                    "enabled_modes": json.dumps([]),
                    "ai_thought_summary": order.get("ai_reasoning", "")[:500],
                    "created_at": ts,
                }

                # If the position is closed, we can immediately evaluate
                pos = positions.get(position_id) if position_id else None
                if pos and pos.get("status") == "closed":
                    exit_price = float(pos.get("current_price", 0))
                    pnl_pct = float(pos.get("pnl_pct", 0))
                    closed_at = pos.get("closed_at", datetime.now(timezone.utc).isoformat())

                    if exit_price > 0:
                        if predicted_direction == "up":
                            correct = exit_price > price
                        elif predicted_direction == "down":
                            correct = exit_price < price
                        else:
                            correct = abs(pnl_pct) < 2

                        row["actual_price_24h"] = exit_price
                        row["direction_correct_24h"] = correct
                        row["pnl_pct_24h"] = round(pnl_pct, 2)
                        row["evaluated_24h_at"] = closed_at

                try:
                    sb.table("ll_predictions").insert(row).execute()
                    created += 1
                    existing_ids.add(prediction_id)
                except Exception as e:
                    err = str(e).lower()
                    if "duplicate" not in err and "23505" not in err:
                        logger.warning("Backfill insert failed: %s", e)

            self._backfill_done = True
            if created:
                logger.info(
                    "Learning loop backfill: created %d predictions from trade history",
                    created,
                )

        except Exception as e:
            logger.error("backfill_from_trades failed: %s", e)

        return created

    # ==================================================================
    # Record Predictions
    # ==================================================================

    def record_prediction(
        self,
        token_ticker: str,
        token_name: str,
        verdict: str,
        confidence: float,
        composite_score: float,
        price_at_prediction: float,
        target_price: float = 0,
        stop_loss_price: float = 0,
        market_condition: str = "sideways",
        market_regime: str = "unknown",
        risk_level: str = "MEDIUM",
        enabled_modes: List[str] | None = None,
        ai_thought_summary: str = "",
        user_id: int = 0,
    ) -> str:
        """Record a recommendation as a trackable prediction. Returns prediction_id."""
        prediction_id = hashlib.sha256(
            f"{token_ticker}|{price_at_prediction}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        # Normalize verdict to direction — handle ALL sources:
        #   trading_engine: "bullish", "bearish"
        #   AI recs (web_app): "STRONG_BUY", "BUY", "HOLD", "CAUTION", "AVOID"
        #   original format: "Strong Buy", "Moderate Buy", "Cautious Buy", "Sell", "Avoid"
        v = verdict.lower().replace("_", " ").strip()
        UP_VERDICTS = {
            "strong buy", "moderate buy", "cautious buy",
            "buy", "bullish", "strong bullish", "long",
        }
        DOWN_VERDICTS = {
            "sell", "avoid", "bearish", "strong bearish",
            "short", "caution",
        }
        if v in UP_VERDICTS:
            predicted_direction = "up"
        elif v in DOWN_VERDICTS:
            predicted_direction = "down"
        else:
            predicted_direction = "flat"

        row = {
            "prediction_id": prediction_id,
            "user_id": user_id,
            "token_ticker": token_ticker if len(token_ticker) <= 20 else token_ticker[:6] + "..." + token_ticker[-4:],
            "token_name": token_name,
            "verdict": verdict,
            "confidence": confidence,
            "composite_score": composite_score,
            "predicted_direction": predicted_direction,
            "price_at_prediction": price_at_prediction,
            "target_price": target_price,
            "stop_loss_price": stop_loss_price,
            "market_condition": market_condition,
            "market_regime": market_regime,
            "risk_level": risk_level,
            "enabled_modes": json.dumps(enabled_modes or []),
            "ai_thought_summary": ai_thought_summary,
        }

        try:
            _sb().table("ll_predictions").insert(row).execute()
            logger.info(
                "Recorded prediction %s for %s: %s @ $%.6f (conf=%.2f)",
                prediction_id, token_ticker, verdict,
                price_at_prediction, confidence,
            )
        except Exception as e:
            err = str(e).lower()
            if "duplicate" in err or "23505" in err:
                logger.debug("Duplicate prediction_id %s (skipped)", prediction_id)
            else:
                logger.warning("Failed to record prediction: %s", e)

        return prediction_id

    # ==================================================================
    # Evaluate Predictions Against Actual Outcomes
    # ==================================================================

    async def evaluate_pending(self, cg_collector) -> Dict[str, Any]:
        """
        Evaluate predictions that haven't been checked yet.
        Compares predicted direction against actual price movement.
        """
        sb = _sb()
        results = {"evaluated_24h": 0, "evaluated_7d": 0, "errors": 0}

        try:
            cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            res_24h = (
                sb.table("ll_predictions")
                .select("*")
                .is_("evaluated_24h_at", "null")
                .lt("created_at", cutoff_24h)
                .order("created_at", desc=True)
                .limit(50)
                .execute()
            )
            pending_24h = res_24h.data or []

            cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            res_7d = (
                sb.table("ll_predictions")
                .select("*")
                .not_.is_("evaluated_24h_at", "null")
                .is_("evaluated_7d_at", "null")
                .lt("created_at", cutoff_7d)
                .order("created_at", desc=True)
                .limit(50)
                .execute()
            )
            pending_7d = res_7d.data or []

            tickers_24h = list({row["token_ticker"].lower() for row in pending_24h})
            tickers_7d = list({row["token_ticker"].lower() for row in pending_7d})
            all_tickers = list(set(tickers_24h + tickers_7d))

            if not all_tickers:
                return results

            # Try fetching prices — use both coin_id and symbol lookups
            try:
                prices = await cg_collector.get_simple_price(all_tickers)
            except Exception as e:
                logger.warning("Price fetch for evaluation failed: %s", e)
                prices = {}

            # If some tickers returned no price, they might be symbols — skip them
            # rather than aborting the entire evaluation
            if not prices:
                logger.warning("No prices returned for %d tickers — skipping eval", len(all_tickers))
                return results

            for row in pending_24h:
                ticker = row["token_ticker"].lower()
                current_price = prices.get(ticker, 0)
                if current_price <= 0:
                    continue
                try:
                    self._evaluate_24h(sb, row, current_price)
                    results["evaluated_24h"] += 1
                except Exception as e:
                    logger.warning("24h eval failed for %s: %s", ticker, e)
                    results["errors"] += 1

            for row in pending_7d:
                ticker = row["token_ticker"].lower()
                current_price = prices.get(ticker, 0)
                if current_price <= 0:
                    continue
                try:
                    self._evaluate_7d(sb, row, current_price)
                    results["evaluated_7d"] += 1
                except Exception as e:
                    logger.warning("7d eval failed for %s: %s", ticker, e)
                    results["errors"] += 1

        except Exception as e:
            logger.error("evaluate_pending failed: %s", e)

        return results

    def evaluate_trade_close(
        self, token_ticker: str, exit_price: float, pnl_pct: float,
        hold_duration_minutes: float = 0, exit_reason: str = "unknown",
    ) -> int:
        """
        Immediately evaluate open predictions for a token when a trade closes.
        Called by the trading engine on every sell. Returns number of records updated.

        Args:
            token_ticker: The coin_id / ticker of the token
            exit_price: Price at which the position was closed
            pnl_pct: Realized P&L percentage
            hold_duration_minutes: How long the position was held (minutes)
            exit_reason: What triggered the exit — "stop_loss", "take_profit", "auto_exit", or "manual"
        """
        sb = _sb()
        updated = 0

        try:
            res = (
                sb.table("ll_predictions")
                .select("*")
                .ilike("token_ticker", token_ticker)
                .is_("evaluated_24h_at", "null")
                .order("created_at", desc=True)
                .limit(10)
                .execute()
            )
            rows = res.data or []

            for pred in rows:
                entry = pred["price_at_prediction"]
                if entry <= 0:
                    continue

                direction = pred["predicted_direction"]
                if direction == "up":
                    correct = exit_price > entry
                elif direction == "down":
                    correct = exit_price < entry
                else:
                    correct = abs(pnl_pct) < 2

                update_data = {
                    "actual_price_24h": exit_price,
                    "direction_correct_24h": correct,
                    "pnl_pct_24h": round(pnl_pct, 2),
                    "evaluated_24h_at": datetime.now(timezone.utc).isoformat(),
                    "hold_duration_minutes": round(hold_duration_minutes, 1),
                    "exit_reason": exit_reason,
                }

                sb.table("ll_predictions").update(update_data).eq(
                    "prediction_id", pred["prediction_id"]
                ).execute()
                updated += 1

            if updated:
                logger.info(
                    "Learning loop: evaluated %d predictions for %s "
                    "(exit=$%.4f, P&L=%.2f%%, held=%.0fmin, reason=%s)",
                    updated, token_ticker, exit_price, pnl_pct,
                    hold_duration_minutes, exit_reason,
                )
        except Exception as e:
            logger.warning("evaluate_trade_close failed: %s", e)

        return updated

    # Alias for backward compatibility with trading_engine.py
    def evaluate_predictions_with_price(self, token_ticker: str, actual_price: float, **kwargs) -> int:
        return self.evaluate_trade_close(token_ticker, actual_price, pnl_pct=0.0)

    # ------------------------------------------------------------------

    def _evaluate_24h(self, sb, pred: dict, actual_price: float):
        entry = pred["price_at_prediction"]
        pnl_pct = ((actual_price - entry) / entry) * 100

        direction = pred["predicted_direction"]
        if direction == "up":
            correct = actual_price > entry
        elif direction == "down":
            correct = actual_price < entry
        else:
            correct = abs(pnl_pct) < 2

        sb.table("ll_predictions").update({
            "actual_price_24h": actual_price,
            "direction_correct_24h": correct,
            "pnl_pct_24h": round(pnl_pct, 2),
            "evaluated_24h_at": datetime.now(timezone.utc).isoformat(),
        }).eq("prediction_id", pred["prediction_id"]).execute()

    def _evaluate_7d(self, sb, pred: dict, actual_price: float):
        entry = pred["price_at_prediction"]
        pnl_pct = ((actual_price - entry) / entry) * 100

        direction = pred["predicted_direction"]
        if direction == "up":
            correct = actual_price > entry
        elif direction == "down":
            correct = actual_price < entry
        else:
            correct = abs(pnl_pct) < 5

        target_price = pred.get("target_price") or 0
        stop_loss_price = pred.get("stop_loss_price") or 0
        target_hit = actual_price >= target_price if target_price > 0 else None
        stop_hit = actual_price <= stop_loss_price if stop_loss_price > 0 else None

        sb.table("ll_predictions").update({
            "actual_price_7d": actual_price,
            "direction_correct_7d": correct,
            "pnl_pct_7d": round(pnl_pct, 2),
            "target_hit": target_hit,
            "stop_loss_hit": stop_hit,
            "evaluated_7d_at": datetime.now(timezone.utc).isoformat(),
        }).eq("prediction_id", pred["prediction_id"]).execute()

    # ==================================================================
    # Performance Metrics
    # ==================================================================

    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get aggregate performance statistics (computed in Python).
        Auto-backfills from trade history on first call if table is empty."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        try:
            res = (
                _sb().table("ll_predictions")
                .select("*")
                .gte("created_at", cutoff)
                .order("created_at", desc=True)
                .limit(5000)
                .execute()
            )
            rows = res.data or []
        except Exception as e:
            logger.warning("get_performance_stats query failed: %s", e)
            rows = []

        # Auto-backfill: if table is empty, seed from trade history
        if not rows and not self._backfill_done:
            backfilled = self.backfill_from_trades()
            if backfilled > 0:
                try:
                    res = (
                        _sb().table("ll_predictions")
                        .select("*")
                        .gte("created_at", cutoff)
                        .order("created_at", desc=True)
                        .limit(5000)
                        .execute()
                    )
                    rows = res.data or []
                except Exception:
                    pass

        total = len(rows)

        eval_24h = [r for r in rows if r.get("evaluated_24h_at")]
        eval_7d = [r for r in rows if r.get("evaluated_7d_at")]
        correct_24h = [r for r in eval_24h if r.get("direction_correct_24h") is True]
        correct_7d = [r for r in eval_7d if r.get("direction_correct_7d") is True]

        evaluated_24h_cnt = len(eval_24h)
        evaluated_7d_cnt = len(eval_7d)

        conf_correct = [r["confidence"] for r in correct_24h if r.get("confidence") is not None]
        incorrect_24h = [r for r in eval_24h if r.get("direction_correct_24h") is False]
        conf_incorrect = [r["confidence"] for r in incorrect_24h if r.get("confidence") is not None]

        avg_conf_correct = sum(conf_correct) / len(conf_correct) if conf_correct else 0
        avg_conf_incorrect = sum(conf_incorrect) / len(conf_incorrect) if conf_incorrect else 0

        regime_stats = {}
        regimes = {r.get("market_regime", "unknown") for r in rows}
        for regime in regimes:
            r_eval = [r for r in eval_24h if r.get("market_regime") == regime]
            r_correct = [r for r in r_eval if r.get("direction_correct_24h") is True]
            r_total = len(r_eval)
            regime_stats[regime] = {
                "total": r_total,
                "correct": len(r_correct),
                "accuracy": round(len(r_correct) / max(r_total, 1) * 100, 1),
            }

        pnl_vals = [r["pnl_pct_24h"] for r in rows if r.get("pnl_pct_24h") is not None]
        avg_pnl = sum(pnl_vals) / len(pnl_vals) if pnl_vals else 0

        direction_map = {"up": "LONG", "down": "SHORT", "flat": "RANGE"}
        direction_stats = {}
        for db_dir, label in direction_map.items():
            d_eval = [r for r in eval_24h if r.get("predicted_direction") == db_dir]
            d_correct = [r for r in d_eval if r.get("direction_correct_24h") is True]
            d_pnl = [r["pnl_pct_24h"] for r in d_eval if r.get("pnl_pct_24h") is not None]
            d_total = len(d_eval)
            direction_stats[label] = {
                "total": d_total,
                "correct": len(d_correct),
                "accuracy": round(len(d_correct) / max(d_total, 1) * 100, 1),
                "avg_pnl": round(sum(d_pnl) / len(d_pnl), 2) if d_pnl else 0,
            }

        with_7d_pnl = sorted(
            [r for r in rows if r.get("pnl_pct_7d") is not None],
            key=lambda r: r["pnl_pct_7d"],
            reverse=True,
        )
        best = [
            {"token_ticker": r["token_ticker"], "verdict": r["verdict"], "pnl_pct_7d": r["pnl_pct_7d"]}
            for r in with_7d_pnl[:3]
        ]
        worst = [
            {"token_ticker": r["token_ticker"], "verdict": r["verdict"], "pnl_pct_7d": r["pnl_pct_7d"]}
            for r in with_7d_pnl[-3:]
        ]

        return {
            "period_days": days,
            "total_predictions": total,
            "evaluated_24h": evaluated_24h_cnt,
            "evaluated_7d": evaluated_7d_cnt,
            "accuracy_24h": round(len(correct_24h) / max(evaluated_24h_cnt, 1) * 100, 1),
            "accuracy_7d": round(len(correct_7d) / max(evaluated_7d_cnt, 1) * 100, 1),
            "avg_confidence_correct": round(avg_conf_correct, 1),
            "avg_confidence_incorrect": round(avg_conf_incorrect, 1),
            "avg_pnl_24h": round(avg_pnl, 2),
            "regime_accuracy": regime_stats,
            "direction_accuracy": direction_stats,
            "best_predictions": best,
            "worst_predictions": worst,
        }

    def get_historical_accuracy(self) -> float:
        """
        Get the overall historical accuracy rate (0-1) for use by
        the ConfidenceScorer's historical_accuracy_modifier.
        """
        try:
            res = (
                _sb().table("ll_predictions")
                .select("direction_correct_24h")
                .not_.is_("evaluated_24h_at", "null")
                .limit(5000)
                .execute()
            )
            rows = res.data or []
            total = len(rows)
            if total < 10:
                return 0.5
            correct = sum(1 for r in rows if r.get("direction_correct_24h") is True)
            return correct / total
        except Exception as e:
            logger.warning("get_historical_accuracy failed: %s", e)
            return 0.5

    def get_token_accuracy(self, ticker: str) -> Optional[float]:
        """Per-token accuracy (0-1). Returns None if <3 evaluated predictions."""
        try:
            res = (
                _sb().table("ll_predictions")
                .select("direction_correct_24h")
                .eq("token_ticker", ticker)
                .not_.is_("evaluated_24h_at", "null")
                .limit(200)
                .execute()
            )
            rows = res.data or []
            if len(rows) < 3:
                return None
            correct = sum(1 for r in rows if r.get("direction_correct_24h") is True)
            return correct / len(rows)
        except Exception:
            return None

    def get_regime_accuracy(self, regime: str) -> Optional[float]:
        """Per-regime accuracy (0-1). Returns None if <5 evaluated predictions."""
        try:
            res = (
                _sb().table("ll_predictions")
                .select("direction_correct_24h")
                .eq("market_regime", regime)
                .not_.is_("evaluated_24h_at", "null")
                .limit(500)
                .execute()
            )
            rows = res.data or []
            if len(rows) < 5:
                return None
            correct = sum(1 for r in rows if r.get("direction_correct_24h") is True)
            return correct / len(rows)
        except Exception:
            return None

    def get_accuracy_context(self, ticker: str = "", regime: str = "") -> Dict[str, Optional[float]]:
        """Get global + per-token + per-regime accuracy for confidence adjustment."""
        return {
            "global": self.get_historical_accuracy(),
            "token": self.get_token_accuracy(ticker) if ticker else None,
            "regime": self.get_regime_accuracy(regime) if regime else None,
        }

    # ==================================================================
    # Strategy Adjustments (Adaptive Learning)
    # ==================================================================

    def generate_adjustments(self) -> List[Dict[str, Any]]:
        """
        Analyze performance and generate strategy adjustment recommendations.
        Checks: regime accuracy, confidence gap, global accuracy, per-token
        accuracy, direction bias, PnL trends, and win-streak patterns.
        """
        stats = self.get_performance_stats(days=14)
        adjustments: List[Dict[str, Any]] = []
        sb = _sb()

        def _save(adj: Dict[str, Any]):
            adjustments.append(adj)
            try:
                sb.table("ll_strategy_adjustments").insert(adj).execute()
            except Exception:
                pass  # DB write failure shouldn't block

        try:
            # ── 1. Regime-specific accuracy ──
            for regime, rdata in stats.get("regime_accuracy", {}).items():
                if rdata["total"] >= 5 and rdata["accuracy"] < 40:
                    _save({
                        "adjustment_type": "regime_weight_reduction",
                        "description": f"Low accuracy ({rdata['accuracy']}%) in {regime} — reduce confidence",
                        "old_value": "standard_weights",
                        "new_value": "conservative_weights",
                        "reason": f"Only {rdata['correct']}/{rdata['total']} correct in {regime} regime",
                        "market_regime": regime,
                    })
                elif rdata["total"] >= 5 and rdata["accuracy"] >= 70:
                    _save({
                        "adjustment_type": "regime_weight_increase",
                        "description": f"Strong accuracy ({rdata['accuracy']}%) in {regime} — boost confidence",
                        "old_value": "standard_weights",
                        "new_value": "aggressive_weights",
                        "reason": f"{rdata['correct']}/{rdata['total']} correct in {regime} regime — performing well",
                        "market_regime": regime,
                    })

            # ── 2. Confidence calibration gap ──
            if stats["avg_confidence_correct"] > 0 and stats["avg_confidence_incorrect"] > 0:
                gap = stats["avg_confidence_correct"] - stats["avg_confidence_incorrect"]
                if gap < 1.0 and stats["evaluated_24h"] >= 10:
                    _save({
                        "adjustment_type": "confidence_calibration",
                        "description": f"Confidence gap too small ({gap:.1f})",
                        "old_value": f"gap={gap:.1f}",
                        "new_value": "widen_confidence_spread",
                        "reason": "Model doesn't distinguish high/low confidence well enough",
                        "market_regime": "",
                    })
                elif gap >= 3.0 and stats["evaluated_24h"] >= 15:
                    _save({
                        "adjustment_type": "confidence_calibration_good",
                        "description": f"Confidence gap healthy ({gap:.1f})",
                        "old_value": f"gap={gap:.1f}",
                        "new_value": "maintain_spread",
                        "reason": "Model is good at assigning high confidence to winning trades",
                        "market_regime": "",
                    })

            # ── 3. Global accuracy threshold ──
            if stats["evaluated_24h"] >= 15 and stats["accuracy_24h"] < 45:
                _save({
                    "adjustment_type": "global_confidence_reduction",
                    "description": f"Overall accuracy ({stats['accuracy_24h']}%) below threshold",
                    "old_value": "base_confidence=5.0",
                    "new_value": "base_confidence=4.5",
                    "reason": "Below 45% accuracy suggests systematic overconfidence",
                    "market_regime": "",
                })
            elif stats["evaluated_24h"] >= 15 and stats["accuracy_24h"] >= 70:
                _save({
                    "adjustment_type": "global_confidence_increase",
                    "description": f"Excellent accuracy ({stats['accuracy_24h']}%) — AI is well-calibrated",
                    "old_value": "base_confidence=5.0",
                    "new_value": "base_confidence=5.5",
                    "reason": "Above 70% accuracy — model can be more assertive",
                    "market_regime": "",
                })

            # ── 4. Direction bias detection ──
            dir_stats = stats.get("direction_accuracy", {})
            long_acc = dir_stats.get("LONG", {}).get("accuracy", 50)
            short_acc = dir_stats.get("SHORT", {}).get("accuracy", 50)
            long_total = dir_stats.get("LONG", {}).get("total", 0)
            short_total = dir_stats.get("SHORT", {}).get("total", 0)
            if long_total >= 5 and short_total >= 5:
                bias = abs(long_acc - short_acc)
                if bias > 25:
                    weaker = "SHORT" if short_acc < long_acc else "LONG"
                    stronger = "LONG" if weaker == "SHORT" else "SHORT"
                    _save({
                        "adjustment_type": "direction_bias_correction",
                        "description": f"Direction imbalance: {stronger} {max(long_acc, short_acc):.0f}% vs {weaker} {min(long_acc, short_acc):.0f}%",
                        "old_value": f"LONG={long_acc:.0f}%, SHORT={short_acc:.0f}%",
                        "new_value": f"reduce_{weaker.lower()}_confidence",
                        "reason": f"AI is significantly worse at {weaker} predictions — reduce {weaker} confidence",
                        "market_regime": "",
                    })

            # ── 5. Per-token underperformers ──
            best = stats.get("best_predictions", [])
            worst = stats.get("worst_predictions", [])
            for w in worst:
                ticker = w.get("token_ticker", "")
                pnl = w.get("pnl_pct_7d", 0) or 0
                if ticker and pnl < -3:
                    _save({
                        "adjustment_type": "token_confidence_reduction",
                        "description": f"{ticker} consistently underperforming ({pnl:+.1f}% PnL)",
                        "old_value": "standard_confidence",
                        "new_value": "reduced_confidence",
                        "reason": f"{ticker} in worst predictions — reduce future confidence for this token",
                        "market_regime": "",
                    })

            # ── 6. PnL trend (positive/negative drift) ──
            avg_pnl = stats.get("avg_pnl_24h", 0)
            if stats["evaluated_24h"] >= 10:
                if avg_pnl < -1.0:
                    _save({
                        "adjustment_type": "pnl_drift_alert",
                        "description": f"Average PnL negative ({avg_pnl:+.2f}%) — tighten risk controls",
                        "old_value": f"avg_pnl={avg_pnl:+.2f}%",
                        "new_value": "tighten_stop_loss",
                        "reason": "Negative PnL drift suggests entries are too aggressive or exits too late",
                        "market_regime": "",
                    })
                elif avg_pnl > 2.0:
                    _save({
                        "adjustment_type": "pnl_momentum",
                        "description": f"Strong avg PnL ({avg_pnl:+.2f}%) — strategy is working",
                        "old_value": f"avg_pnl={avg_pnl:+.2f}%",
                        "new_value": "maintain_strategy",
                        "reason": "Current strategy is generating consistent positive returns",
                        "market_regime": "",
                    })

        except Exception as e:
            logger.warning("generate_adjustments failed: %s", e)

        return adjustments

    def get_recent_adjustments(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently generated strategy adjustments."""
        try:
            res = (
                _sb().table("ll_strategy_adjustments")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data or []
        except Exception as e:
            logger.warning("get_recent_adjustments failed: %s", e)
            return []

    def get_token_track_record(self, ticker: str) -> Dict[str, Any]:
        """Get the track record for a specific token."""
        try:
            res = (
                _sb().table("ll_predictions")
                .select("*")
                .eq("token_ticker", ticker.upper())
                .order("created_at", desc=True)
                .limit(20)
                .execute()
            )
            preds = res.data or []
        except Exception as e:
            logger.warning("get_token_track_record failed: %s", e)
            preds = []

        if not preds:
            return {"ticker": ticker, "predictions": 0, "accuracy": None}

        total = len(preds)
        evaluated = [p for p in preds if p.get("evaluated_24h_at")]
        correct = sum(1 for p in evaluated if p.get("direction_correct_24h") is True)
        pnl_vals = [p["pnl_pct_24h"] for p in evaluated if p.get("pnl_pct_24h") is not None]

        return {
            "ticker": ticker.upper(),
            "predictions": total,
            "evaluated": len(evaluated),
            "correct": correct,
            "accuracy": round(correct / max(len(evaluated), 1) * 100, 1) if evaluated else None,
            "avg_pnl_24h": round(sum(pnl_vals) / max(len(pnl_vals), 1), 2) if pnl_vals else 0,
            "recent": preds[:5],
        }

    def snapshot_accuracy(self) -> Dict[str, Any]:
        """Take a point-in-time snapshot of accuracy for historical tracking."""
        stats = self.get_performance_stats(days=7)
        try:
            _sb().table("ll_accuracy_snapshots").insert({
                "period": "7d",
                "total_predictions": stats["total_predictions"],
                "correct_24h": (
                    int(stats["accuracy_24h"] * stats["evaluated_24h"] / 100)
                    if stats["evaluated_24h"] > 0 else 0
                ),
                "correct_7d": (
                    int(stats["accuracy_7d"] * stats["evaluated_7d"] / 100)
                    if stats["evaluated_7d"] > 0 else 0
                ),
                "accuracy_24h": stats["accuracy_24h"],
                "accuracy_7d": stats["accuracy_7d"],
                "avg_confidence_correct": stats["avg_confidence_correct"],
                "avg_confidence_incorrect": stats["avg_confidence_incorrect"],
            }).execute()
        except Exception as e:
            logger.warning("snapshot_accuracy insert failed: %s", e)

        return stats
