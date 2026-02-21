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

        predicted_direction = (
            "up" if verdict in ("Strong Buy", "Moderate Buy", "Cautious Buy")
            else "down" if verdict in ("Sell", "Avoid")
            else "flat"
        )

        row = {
            "prediction_id": prediction_id,
            "user_id": user_id,
            "token_ticker": token_ticker,
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
            cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
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

            try:
                prices = await cg_collector.get_simple_price(all_tickers)
            except Exception as e:
                logger.warning("Price fetch for evaluation failed: %s", e)
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
        self, token_ticker: str, exit_price: float, pnl_pct: float
    ) -> int:
        """
        Immediately evaluate open predictions for a token when a trade closes.
        Called by the trading engine on every sell. Returns number of records updated.
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

                sb.table("ll_predictions").update({
                    "actual_price_24h": exit_price,
                    "direction_correct_24h": correct,
                    "pnl_pct_24h": round(pnl_pct, 2),
                    "evaluated_24h_at": datetime.now(timezone.utc).isoformat(),
                }).eq("prediction_id", pred["prediction_id"]).execute()
                updated += 1

            if updated:
                logger.info(
                    "Learning loop: evaluated %d predictions for %s "
                    "(exit=$%.4f, P&L=%.2f%%)",
                    updated, token_ticker, exit_price, pnl_pct,
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
        """Get aggregate performance statistics (computed in Python)."""
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

    # ==================================================================
    # Strategy Adjustments (Adaptive Learning)
    # ==================================================================

    def generate_adjustments(self) -> List[Dict[str, Any]]:
        """
        Analyze performance and generate strategy adjustment recommendations.
        """
        stats = self.get_performance_stats(days=14)
        adjustments: List[Dict[str, Any]] = []
        sb = _sb()

        try:
            for regime, rdata in stats.get("regime_accuracy", {}).items():
                if rdata["total"] >= 5 and rdata["accuracy"] < 40:
                    adj = {
                        "adjustment_type": "regime_weight_adjustment",
                        "description": f"Low accuracy ({rdata['accuracy']}%) in {regime} - reduce confidence",
                        "old_value": "standard_weights",
                        "new_value": "conservative_weights",
                        "reason": f"Only {rdata['correct']}/{rdata['total']} correct in {regime} regime",
                        "market_regime": regime,
                    }
                    adjustments.append(adj)
                    sb.table("ll_strategy_adjustments").insert(adj).execute()

            if stats["avg_confidence_correct"] > 0 and stats["avg_confidence_incorrect"] > 0:
                gap = stats["avg_confidence_correct"] - stats["avg_confidence_incorrect"]
                if gap < 1.0 and stats["evaluated_24h"] >= 10:
                    adj = {
                        "adjustment_type": "confidence_calibration",
                        "description": f"Confidence gap too small ({gap:.1f})",
                        "old_value": f"gap={gap:.1f}",
                        "new_value": "widen_confidence_spread",
                        "reason": "Model doesn't distinguish high/low confidence well enough",
                        "market_regime": "",
                    }
                    adjustments.append(adj)
                    sb.table("ll_strategy_adjustments").insert(adj).execute()

            if stats["evaluated_24h"] >= 20 and stats["accuracy_24h"] < 45:
                adj = {
                    "adjustment_type": "global_confidence_reduction",
                    "description": f"Overall accuracy ({stats['accuracy_24h']}%) below threshold",
                    "old_value": "base_confidence=5.0",
                    "new_value": "base_confidence=4.5",
                    "reason": "Below 45% accuracy suggests systematic overconfidence",
                    "market_regime": "",
                }
                adjustments.append(adj)
                sb.table("ll_strategy_adjustments").insert(adj).execute()

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
