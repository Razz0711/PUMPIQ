"""
Continuous Learning & Feedback Loop
======================================
Tracks prediction accuracy, compares AI decisions with actual outcomes,
and adapts strategies based on market regime changes.

Features:
  - Records every recommendation as a trackable prediction
  - Evaluates predictions after 24h and 7d against actual prices
  - Computes accuracy metrics per market regime, mode, and timeframe
  - Generates strategy adjustments based on performance
  - Adapts confidence modifiers based on historical accuracy

The learning loop makes the AI **evolving, not static**.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Database path
IS_VERCEL = bool(os.getenv("VERCEL"))
DB_PATH = "/tmp/pumpiq_learning.db" if IS_VERCEL else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pumpiq_learning.db"
)


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    return conn


def init_learning_tables():
    """Create tables for the continuous learning loop."""
    conn = _get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT UNIQUE NOT NULL,
            user_id INTEGER DEFAULT 0,
            token_ticker TEXT NOT NULL,
            token_name TEXT NOT NULL DEFAULT '',
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            composite_score REAL NOT NULL DEFAULT 0,
            predicted_direction TEXT NOT NULL DEFAULT 'up',
            price_at_prediction REAL NOT NULL,
            target_price REAL NOT NULL DEFAULT 0,
            stop_loss_price REAL NOT NULL DEFAULT 0,
            market_condition TEXT NOT NULL DEFAULT 'sideways',
            market_regime TEXT NOT NULL DEFAULT 'unknown',
            risk_level TEXT NOT NULL DEFAULT 'MEDIUM',
            enabled_modes TEXT NOT NULL DEFAULT '[]',
            ai_thought_summary TEXT NOT NULL DEFAULT '',
            -- Outcome fields (filled by evaluation)
            actual_price_24h REAL DEFAULT NULL,
            actual_price_7d REAL DEFAULT NULL,
            direction_correct_24h INTEGER DEFAULT NULL,
            direction_correct_7d INTEGER DEFAULT NULL,
            pnl_pct_24h REAL DEFAULT NULL,
            pnl_pct_7d REAL DEFAULT NULL,
            target_hit INTEGER DEFAULT NULL,
            stop_loss_hit INTEGER DEFAULT NULL,
            -- Metadata
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            evaluated_24h_at TEXT DEFAULT NULL,
            evaluated_7d_at TEXT DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS strategy_adjustments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            adjustment_type TEXT NOT NULL,
            description TEXT NOT NULL,
            old_value TEXT NOT NULL DEFAULT '',
            new_value TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL DEFAULT '',
            market_regime TEXT NOT NULL DEFAULT '',
            applied INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS accuracy_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT NOT NULL,
            total_predictions INTEGER NOT NULL DEFAULT 0,
            correct_24h INTEGER NOT NULL DEFAULT 0,
            correct_7d INTEGER NOT NULL DEFAULT 0,
            accuracy_24h REAL NOT NULL DEFAULT 0,
            accuracy_7d REAL NOT NULL DEFAULT 0,
            avg_confidence_correct REAL NOT NULL DEFAULT 0,
            avg_confidence_incorrect REAL NOT NULL DEFAULT 0,
            best_mode TEXT NOT NULL DEFAULT '',
            worst_mode TEXT NOT NULL DEFAULT '',
            market_regime TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(token_ticker);
        CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
        CREATE INDEX IF NOT EXISTS idx_predictions_evaluated ON predictions(evaluated_24h_at);
    ''')
    conn.commit()
    conn.close()


# Initialize tables
init_learning_tables()


class LearningLoop:
    """
    Continuous learning engine that tracks and evaluates predictions.

    Usage::

        loop = LearningLoop()
        # Record a new prediction
        loop.record_prediction(token_ticker="BTC", verdict="Strong Buy", ...)
        # Later: evaluate against actual prices
        await loop.evaluate_pending(cg_collector)
        # Get performance metrics
        stats = loop.get_performance_stats()
        # Get strategy adjustments
        adjustments = loop.generate_adjustments()
    """

    def __init__(self):
        init_learning_tables()

    # ══════════════════════════════════════════════════════════════
    # Record Predictions
    # ══════════════════════════════════════════════════════════════

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
        enabled_modes: List[str] = None,
        ai_thought_summary: str = "",
        user_id: int = 0,
    ) -> str:
        """Record a recommendation as a trackable prediction. Returns prediction_id."""
        prediction_id = hashlib.sha256(
            f"{token_ticker}|{price_at_prediction}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        conn = _get_db()
        try:
            predicted_direction = "up" if verdict in (
                "Strong Buy", "Moderate Buy", "Cautious Buy"
            ) else "down" if verdict in ("Sell", "Avoid") else "flat"

            conn.execute('''
                INSERT INTO predictions (
                    prediction_id, user_id, token_ticker, token_name, verdict,
                    confidence, composite_score, predicted_direction,
                    price_at_prediction, target_price, stop_loss_price,
                    market_condition, market_regime, risk_level,
                    enabled_modes, ai_thought_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id, user_id, token_ticker, token_name, verdict,
                confidence, composite_score, predicted_direction,
                price_at_prediction, target_price, stop_loss_price,
                market_condition, market_regime, risk_level,
                json.dumps(enabled_modes or []), ai_thought_summary,
            ))
            conn.commit()
            logger.info(
                "Recorded prediction %s for %s: %s @ $%.6f (conf=%.1f)",
                prediction_id, token_ticker, verdict, price_at_prediction, confidence,
            )
            return prediction_id
        except sqlite3.IntegrityError:
            logger.debug("Duplicate prediction_id %s (skipped)", prediction_id)
            return prediction_id
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════
    # Evaluate Predictions Against Actual Outcomes
    # ══════════════════════════════════════════════════════════════

    async def evaluate_pending(self, cg_collector) -> Dict[str, Any]:
        """
        Evaluate predictions that haven't been checked yet.
        Compares predicted direction against actual price movement.
        """
        conn = _get_db()
        results = {"evaluated_24h": 0, "evaluated_7d": 0, "errors": 0}

        try:
            # 24h evaluations: predictions older than 24h not yet evaluated
            cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            pending_24h = conn.execute('''
                SELECT * FROM predictions
                WHERE evaluated_24h_at IS NULL AND created_at < ?
                ORDER BY created_at DESC LIMIT 50
            ''', (cutoff_24h,)).fetchall()

            # 7d evaluations
            cutoff_7d = (datetime.utcnow() - timedelta(days=7)).isoformat()
            pending_7d = conn.execute('''
                SELECT * FROM predictions
                WHERE evaluated_7d_at IS NULL AND evaluated_24h_at IS NOT NULL
                AND created_at < ?
                ORDER BY created_at DESC LIMIT 50
            ''', (cutoff_7d,)).fetchall()

            # Batch: collect all unique tickers
            tickers_24h = list({row["token_ticker"].lower() for row in pending_24h})
            tickers_7d = list({row["token_ticker"].lower() for row in pending_7d})
            all_tickers = list(set(tickers_24h + tickers_7d))

            if not all_tickers:
                return results

            # Fetch current prices
            prices = {}
            try:
                price_data = await cg_collector.get_simple_price(all_tickers)
                prices = price_data
            except Exception as e:
                logger.warning("Price fetch for evaluation failed: %s", e)
                return results

            # Evaluate 24h predictions
            for row in pending_24h:
                ticker = row["token_ticker"].lower()
                current_price = prices.get(ticker, 0)
                if current_price <= 0:
                    continue
                try:
                    self._evaluate_24h(conn, dict(row), current_price)
                    results["evaluated_24h"] += 1
                except Exception as e:
                    logger.warning("24h eval failed for %s: %s", ticker, e)
                    results["errors"] += 1

            # Evaluate 7d predictions
            for row in pending_7d:
                ticker = row["token_ticker"].lower()
                current_price = prices.get(ticker, 0)
                if current_price <= 0:
                    continue
                try:
                    self._evaluate_7d(conn, dict(row), current_price)
                    results["evaluated_7d"] += 1
                except Exception as e:
                    logger.warning("7d eval failed for %s: %s", ticker, e)
                    results["errors"] += 1

            conn.commit()
        finally:
            conn.close()

        return results

    def _evaluate_24h(self, conn, pred: dict, actual_price: float):
        """Evaluate a prediction after 24 hours."""
        entry = pred["price_at_prediction"]
        pnl_pct = ((actual_price - entry) / entry) * 100

        direction = pred["predicted_direction"]
        if direction == "up":
            correct = actual_price > entry
        elif direction == "down":
            correct = actual_price < entry
        else:
            correct = abs(pnl_pct) < 2  # within 2% = "flat" is correct

        conn.execute('''
            UPDATE predictions SET
                actual_price_24h = ?,
                direction_correct_24h = ?,
                pnl_pct_24h = ?,
                evaluated_24h_at = datetime('now')
            WHERE prediction_id = ?
        ''', (actual_price, 1 if correct else 0, round(pnl_pct, 2), pred["prediction_id"]))

    def _evaluate_7d(self, conn, pred: dict, actual_price: float):
        """Evaluate a prediction after 7 days."""
        entry = pred["price_at_prediction"]
        pnl_pct = ((actual_price - entry) / entry) * 100

        direction = pred["predicted_direction"]
        if direction == "up":
            correct = actual_price > entry
        elif direction == "down":
            correct = actual_price < entry
        else:
            correct = abs(pnl_pct) < 5

        # Check target/stop-loss hit
        target_hit = actual_price >= pred["target_price"] if pred["target_price"] > 0 else None
        stop_hit = actual_price <= pred["stop_loss_price"] if pred["stop_loss_price"] > 0 else None

        conn.execute('''
            UPDATE predictions SET
                actual_price_7d = ?,
                direction_correct_7d = ?,
                pnl_pct_7d = ?,
                target_hit = ?,
                stop_loss_hit = ?,
                evaluated_7d_at = datetime('now')
            WHERE prediction_id = ?
        ''', (
            actual_price, 1 if correct else 0, round(pnl_pct, 2),
            1 if target_hit else 0 if target_hit is not None else None,
            1 if stop_hit else 0 if stop_hit is not None else None,
            pred["prediction_id"],
        ))

    # ══════════════════════════════════════════════════════════════
    # Performance Metrics
    # ══════════════════════════════════════════════════════════════

    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get aggregate performance statistics."""
        conn = _get_db()
        try:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

            # Overall accuracy
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE created_at > ?", (cutoff,)
            ).fetchone()["cnt"]

            evaluated_24h = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE evaluated_24h_at IS NOT NULL AND created_at > ?",
                (cutoff,)
            ).fetchone()["cnt"]

            correct_24h = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE direction_correct_24h = 1 AND created_at > ?",
                (cutoff,)
            ).fetchone()["cnt"]

            evaluated_7d = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE evaluated_7d_at IS NOT NULL AND created_at > ?",
                (cutoff,)
            ).fetchone()["cnt"]

            correct_7d = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE direction_correct_7d = 1 AND created_at > ?",
                (cutoff,)
            ).fetchone()["cnt"]

            # Confidence analysis
            avg_conf_correct = conn.execute(
                "SELECT AVG(confidence) as avg FROM predictions WHERE direction_correct_24h = 1 AND created_at > ?",
                (cutoff,)
            ).fetchone()["avg"] or 0

            avg_conf_incorrect = conn.execute(
                "SELECT AVG(confidence) as avg FROM predictions WHERE direction_correct_24h = 0 AND created_at > ?",
                (cutoff,)
            ).fetchone()["avg"] or 0

            # Per-regime accuracy
            regime_stats = {}
            regimes = conn.execute(
                "SELECT DISTINCT market_regime FROM predictions WHERE created_at > ?", (cutoff,)
            ).fetchall()
            for r in regimes:
                regime = r["market_regime"]
                r_total = conn.execute(
                    "SELECT COUNT(*) as cnt FROM predictions WHERE market_regime = ? AND evaluated_24h_at IS NOT NULL AND created_at > ?",
                    (regime, cutoff)
                ).fetchone()["cnt"]
                r_correct = conn.execute(
                    "SELECT COUNT(*) as cnt FROM predictions WHERE market_regime = ? AND direction_correct_24h = 1 AND created_at > ?",
                    (regime, cutoff)
                ).fetchone()["cnt"]
                regime_stats[regime] = {
                    "total": r_total,
                    "correct": r_correct,
                    "accuracy": round(r_correct / max(r_total, 1) * 100, 1),
                }

            # Average P&L
            avg_pnl = conn.execute(
                "SELECT AVG(pnl_pct_24h) as avg FROM predictions WHERE pnl_pct_24h IS NOT NULL AND created_at > ?",
                (cutoff,)
            ).fetchone()["avg"] or 0

            # Best / worst predictions
            best = conn.execute(
                "SELECT token_ticker, verdict, pnl_pct_7d FROM predictions WHERE pnl_pct_7d IS NOT NULL AND created_at > ? ORDER BY pnl_pct_7d DESC LIMIT 3",
                (cutoff,)
            ).fetchall()

            worst = conn.execute(
                "SELECT token_ticker, verdict, pnl_pct_7d FROM predictions WHERE pnl_pct_7d IS NOT NULL AND created_at > ? ORDER BY pnl_pct_7d ASC LIMIT 3",
                (cutoff,)
            ).fetchall()

            return {
                "period_days": days,
                "total_predictions": total,
                "evaluated_24h": evaluated_24h,
                "evaluated_7d": evaluated_7d,
                "accuracy_24h": round(correct_24h / max(evaluated_24h, 1) * 100, 1),
                "accuracy_7d": round(correct_7d / max(evaluated_7d, 1) * 100, 1),
                "avg_confidence_correct": round(avg_conf_correct, 1),
                "avg_confidence_incorrect": round(avg_conf_incorrect, 1),
                "avg_pnl_24h": round(avg_pnl, 2),
                "regime_accuracy": regime_stats,
                "best_predictions": [dict(b) for b in best],
                "worst_predictions": [dict(w) for w in worst],
            }
        finally:
            conn.close()

    def get_historical_accuracy(self) -> float:
        """
        Get the overall historical accuracy rate (0-1) for use by
        the ConfidenceScorer's historical_accuracy_modifier.
        """
        conn = _get_db()
        try:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE evaluated_24h_at IS NOT NULL"
            ).fetchone()["cnt"]
            correct = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE direction_correct_24h = 1"
            ).fetchone()["cnt"]
            if total < 10:
                return 0.5  # Not enough data to adjust
            return correct / total
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════
    # Strategy Adjustments (Adaptive Learning)
    # ══════════════════════════════════════════════════════════════

    def generate_adjustments(self) -> List[Dict[str, Any]]:
        """
        Analyze performance and generate strategy adjustment recommendations.
        These can be applied to modify confidence scoring, weight allocation, etc.
        """
        stats = self.get_performance_stats(days=14)
        adjustments: List[Dict[str, Any]] = []
        conn = _get_db()

        try:
            # Adjustment 1: If accuracy is low in a regime, suggest weight changes
            for regime, regime_data in stats.get("regime_accuracy", {}).items():
                if regime_data["total"] >= 5 and regime_data["accuracy"] < 40:
                    adj = {
                        "type": "regime_weight_adjustment",
                        "description": f"Low accuracy ({regime_data['accuracy']}%) in {regime} market — reduce confidence in this regime",
                        "old_value": "standard_weights",
                        "new_value": "conservative_weights",
                        "reason": f"Only {regime_data['correct']}/{regime_data['total']} correct in {regime} regime",
                        "market_regime": regime,
                    }
                    adjustments.append(adj)
                    conn.execute('''
                        INSERT INTO strategy_adjustments (adjustment_type, description, old_value, new_value, reason, market_regime)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (adj["type"], adj["description"], adj["old_value"], adj["new_value"], adj["reason"], adj["market_regime"]))

            # Adjustment 2: If confidence is poorly calibrated
            if stats["avg_confidence_correct"] > 0 and stats["avg_confidence_incorrect"] > 0:
                gap = stats["avg_confidence_correct"] - stats["avg_confidence_incorrect"]
                if gap < 1.0 and stats["evaluated_24h"] >= 10:
                    adj = {
                        "type": "confidence_calibration",
                        "description": f"Confidence gap too small ({gap:.1f}) — correct and incorrect predictions have similar confidence",
                        "old_value": f"gap={gap:.1f}",
                        "new_value": "widen_confidence_spread",
                        "reason": "The model doesn't distinguish high-confidence from low-confidence well enough",
                        "market_regime": "",
                    }
                    adjustments.append(adj)
                    conn.execute('''
                        INSERT INTO strategy_adjustments (adjustment_type, description, old_value, new_value, reason, market_regime)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (adj["type"], adj["description"], adj["old_value"], adj["new_value"], adj["reason"], adj["market_regime"]))

            # Adjustment 3: If overall accuracy drops below threshold
            if stats["evaluated_24h"] >= 20 and stats["accuracy_24h"] < 45:
                adj = {
                    "type": "global_confidence_reduction",
                    "description": f"Overall accuracy ({stats['accuracy_24h']}%) is below acceptable threshold — reducing base confidence",
                    "old_value": "base_confidence=5.0",
                    "new_value": "base_confidence=4.5",
                    "reason": "Below 45% accuracy suggests systematic overconfidence",
                    "market_regime": "",
                }
                adjustments.append(adj)
                conn.execute('''
                    INSERT INTO strategy_adjustments (adjustment_type, description, old_value, new_value, reason, market_regime)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (adj["type"], adj["description"], adj["old_value"], adj["new_value"], adj["reason"], adj["market_regime"]))

            conn.commit()
        finally:
            conn.close()

        return adjustments

    def get_recent_adjustments(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently generated strategy adjustments."""
        conn = _get_db()
        try:
            rows = conn.execute(
                "SELECT * FROM strategy_adjustments ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_token_track_record(self, ticker: str) -> Dict[str, Any]:
        """Get the track record for a specific token."""
        conn = _get_db()
        try:
            preds = conn.execute(
                "SELECT * FROM predictions WHERE token_ticker = ? ORDER BY created_at DESC LIMIT 20",
                (ticker.upper(),)
            ).fetchall()

            if not preds:
                return {"ticker": ticker, "predictions": 0, "accuracy": None}

            total = len(preds)
            evaluated = [p for p in preds if p["evaluated_24h_at"]]
            correct = sum(1 for p in evaluated if p["direction_correct_24h"])

            return {
                "ticker": ticker.upper(),
                "predictions": total,
                "evaluated": len(evaluated),
                "correct": correct,
                "accuracy": round(correct / max(len(evaluated), 1) * 100, 1) if evaluated else None,
                "avg_pnl_24h": round(
                    sum(p["pnl_pct_24h"] for p in evaluated if p["pnl_pct_24h"] is not None) /
                    max(len([p for p in evaluated if p["pnl_pct_24h"] is not None]), 1), 2
                ),
                "recent": [dict(p) for p in preds[:5]],
            }
        finally:
            conn.close()

    def snapshot_accuracy(self) -> Dict[str, Any]:
        """Take a point-in-time snapshot of accuracy for historical tracking."""
        stats = self.get_performance_stats(days=7)
        conn = _get_db()
        try:
            conn.execute('''
                INSERT INTO accuracy_snapshots (
                    period, total_predictions, correct_24h, correct_7d,
                    accuracy_24h, accuracy_7d,
                    avg_confidence_correct, avg_confidence_incorrect
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                "7d", stats["total_predictions"],
                int(stats["accuracy_24h"] * stats["evaluated_24h"] / 100) if stats["evaluated_24h"] > 0 else 0,
                int(stats["accuracy_7d"] * stats["evaluated_7d"] / 100) if stats["evaluated_7d"] > 0 else 0,
                stats["accuracy_24h"], stats["accuracy_7d"],
                stats["avg_confidence_correct"], stats["avg_confidence_incorrect"],
            ))
            conn.commit()
        finally:
            conn.close()
        return stats
