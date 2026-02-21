"""
Prediction Tracker & Feedback Loop
=====================================
Algorithm 5 – Extended prediction tracking with per-token rolling accuracy,
per-market-regime accuracy, per-indicator-combination accuracy, and feedback
into the ConfidenceScorer's historical_accuracy modifier.

Extends the existing LearningLoop (src/ai_engine/learning_loop.py) with:
  - Per-token rolling accuracy (last 30 predictions)
  - Per-market-regime accuracy breakdown
  - Per-indicator-combination win rate tracking
  - Automatic HistoricalAccuracy updates fed back to ConfidenceScorer
  - Dashboard-ready performance summary

Storage: SQLite (same DB as LearningLoop for co-location).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Database ──────────────────────────────────────────────────────

IS_VERCEL = bool(os.getenv("VERCEL"))
DB_PATH = "/tmp/nexypher_learning.db" if IS_VERCEL else os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "nexypher_learning.db"
)


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    return conn


def _init_tracker_tables():
    """Create extended tracking tables (additive to LearningLoop tables)."""
    conn = _get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS token_accuracy_cache (
            token_ticker TEXT PRIMARY KEY,
            rolling_accuracy REAL NOT NULL DEFAULT 0.5,
            total_predictions INTEGER NOT NULL DEFAULT 0,
            correct_predictions INTEGER NOT NULL DEFAULT 0,
            avg_pnl_24h REAL NOT NULL DEFAULT 0,
            avg_pnl_7d REAL NOT NULL DEFAULT 0,
            best_pnl REAL NOT NULL DEFAULT 0,
            worst_pnl REAL NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS regime_accuracy_cache (
            market_regime TEXT PRIMARY KEY,
            accuracy REAL NOT NULL DEFAULT 0.5,
            total_predictions INTEGER NOT NULL DEFAULT 0,
            correct_predictions INTEGER NOT NULL DEFAULT 0,
            avg_confidence REAL NOT NULL DEFAULT 5.0,
            last_updated TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS indicator_combo_accuracy (
            combo_key TEXT PRIMARY KEY,
            description TEXT NOT NULL DEFAULT '',
            accuracy REAL NOT NULL DEFAULT 0.5,
            total_predictions INTEGER NOT NULL DEFAULT 0,
            correct_predictions INTEGER NOT NULL DEFAULT 0,
            avg_pnl REAL NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS prediction_extended (
            prediction_id TEXT PRIMARY KEY,
            token_ticker TEXT NOT NULL,
            entry_price REAL NOT NULL DEFAULT 0,
            target_price REAL NOT NULL DEFAULT 0,
            stop_loss_price REAL NOT NULL DEFAULT 0,
            predicted_direction TEXT NOT NULL DEFAULT 'up',
            confidence_score REAL NOT NULL DEFAULT 0,
            composite_score REAL NOT NULL DEFAULT 0,
            market_regime TEXT NOT NULL DEFAULT 'unknown',
            indicator_combo TEXT NOT NULL DEFAULT '[]',
            ml_signal REAL DEFAULT NULL,
            lstm_signal REAL DEFAULT NULL,
            confluence_score REAL DEFAULT NULL,
            backtest_passed INTEGER DEFAULT NULL,
            timeframe TEXT NOT NULL DEFAULT 'swing',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            evaluated_24h INTEGER NOT NULL DEFAULT 0,
            evaluated_7d INTEGER NOT NULL DEFAULT 0,
            actual_price_24h REAL DEFAULT NULL,
            actual_price_7d REAL DEFAULT NULL,
            pnl_24h REAL DEFAULT NULL,
            pnl_7d REAL DEFAULT NULL,
            direction_correct_24h INTEGER DEFAULT NULL,
            direction_correct_7d INTEGER DEFAULT NULL,
            target_hit INTEGER DEFAULT NULL,
            stop_hit INTEGER DEFAULT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ext_ticker ON prediction_extended(token_ticker);
        CREATE INDEX IF NOT EXISTS idx_ext_regime ON prediction_extended(market_regime);
        CREATE INDEX IF NOT EXISTS idx_ext_created ON prediction_extended(created_at);
    ''')
    conn.commit()
    conn.close()


# Lazy initialization — tables are created on first use, not on import
_tracker_tables_initialized = False


def _ensure_tracker_tables():
    """Initialize tracker tables on first use."""
    global _tracker_tables_initialized
    if not _tracker_tables_initialized:
        _init_tracker_tables()
        _tracker_tables_initialized = True


# ── Data Classes ──────────────────────────────────────────────────

@dataclass
class TokenAccuracy:
    """Rolling accuracy stats for a single token."""
    token_ticker: str
    rolling_accuracy: float = 0.5     # 0-1
    total_predictions: int = 0
    correct_predictions: int = 0
    avg_pnl_24h: float = 0.0
    avg_pnl_7d: float = 0.0
    best_pnl: float = 0.0
    worst_pnl: float = 0.0


@dataclass
class RegimeAccuracy:
    """Accuracy stats per market regime."""
    market_regime: str
    accuracy: float = 0.5
    total_predictions: int = 0
    correct_predictions: int = 0
    avg_confidence: float = 5.0


@dataclass
class PredictionFeedback:
    """Complete feedback summary for the system."""
    overall_accuracy_24h: float = 0.5
    overall_accuracy_7d: float = 0.5
    total_predictions: int = 0
    total_evaluated: int = 0
    token_accuracies: Dict[str, TokenAccuracy] = field(default_factory=dict)
    regime_accuracies: Dict[str, RegimeAccuracy] = field(default_factory=dict)
    best_indicator_combos: List[Dict[str, Any]] = field(default_factory=list)
    worst_indicator_combos: List[Dict[str, Any]] = field(default_factory=list)
    confidence_calibration: float = 0.0  # gap between correct/incorrect avg confidence


# ── Prediction Tracker ────────────────────────────────────────────

class PredictionTracker:
    """
    Extended prediction tracking with per-token, per-regime, and
    per-indicator accuracy.  Feeds results back into the AI pipeline.

    Usage::

        tracker = PredictionTracker()

        # Record every recommendation
        tracker.record(
            prediction_id="abc123",
            token_ticker="SOL",
            entry_price=150.0,
            target_price=172.5,
            stop_loss_price=135.0,
            confidence_score=7.8,
            composite_score=7.5,
            market_regime="trending_up",
            indicator_combo=["RSI_oversold", "MACD_bullish", "BB_lower"],
            ml_signal=0.72,
            lstm_signal=0.68,
            confluence_score=0.85,
            backtest_passed=True,
        )

        # Evaluate pending predictions (call periodically)
        await tracker.evaluate_pending(cg_collector)

        # Get feedback for pipeline tuning
        feedback = tracker.get_feedback()

        # Get per-token accuracy for confidence modifier
        acc = tracker.get_token_accuracy("SOL")
    """

    def __init__(self):
        _ensure_tracker_tables()

    # ══════════════════════════════════════════════════════════════
    # Record Prediction
    # ══════════════════════════════════════════════════════════════

    def record(
        self,
        prediction_id: str,
        token_ticker: str,
        entry_price: float,
        target_price: float = 0.0,
        stop_loss_price: float = 0.0,
        predicted_direction: str = "up",
        confidence_score: float = 5.0,
        composite_score: float = 5.0,
        market_regime: str = "unknown",
        indicator_combo: Optional[List[str]] = None,
        ml_signal: Optional[float] = None,
        lstm_signal: Optional[float] = None,
        confluence_score: Optional[float] = None,
        backtest_passed: Optional[bool] = None,
        timeframe: str = "swing",
    ):
        """Record an extended prediction for tracking."""
        conn = _get_db()
        try:
            conn.execute('''
                INSERT OR REPLACE INTO prediction_extended (
                    prediction_id, token_ticker, entry_price, target_price,
                    stop_loss_price, predicted_direction, confidence_score,
                    composite_score, market_regime, indicator_combo,
                    ml_signal, lstm_signal, confluence_score,
                    backtest_passed, timeframe
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id, token_ticker.upper(), entry_price, target_price,
                stop_loss_price, predicted_direction, confidence_score,
                composite_score, market_regime,
                json.dumps(indicator_combo or []),
                ml_signal, lstm_signal, confluence_score,
                1 if backtest_passed else 0 if backtest_passed is not None else None,
                timeframe,
            ))
            conn.commit()
            logger.info(
                "PredictionTracker: recorded %s for %s @ $%.4f (conf=%.1f, regime=%s)",
                prediction_id, token_ticker, entry_price, confidence_score, market_regime,
            )
        except Exception as e:
            logger.warning("PredictionTracker record failed: %s", e)
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════
    # Evaluate Pending Predictions
    # ══════════════════════════════════════════════════════════════

    async def evaluate_pending(self, cg_collector) -> Dict[str, int]:
        """
        Evaluate predictions that haven't been assessed yet.
        Checks 24h and 7d outcomes, updates accuracy caches.
        """
        results = {"evaluated_24h": 0, "evaluated_7d": 0, "errors": 0}
        conn = _get_db()

        try:
            # 24h evaluations
            cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            pending_24h = conn.execute('''
                SELECT * FROM prediction_extended
                WHERE evaluated_24h = 0 AND created_at < ?
                ORDER BY created_at DESC LIMIT 100
            ''', (cutoff_24h,)).fetchall()

            # 7d evaluations
            cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            pending_7d = conn.execute('''
                SELECT * FROM prediction_extended
                WHERE evaluated_7d = 0 AND evaluated_24h = 1
                AND created_at < ?
                ORDER BY created_at DESC LIMIT 100
            ''', (cutoff_7d,)).fetchall()

            # Collect unique tickers
            tickers = list({
                row["token_ticker"].lower()
                for rows in [pending_24h, pending_7d]
                for row in rows
            })

            if not tickers:
                return results

            # Fetch current prices
            prices = {}
            try:
                prices = await cg_collector.get_simple_price(tickers)
            except Exception as e:
                logger.warning("Price fetch for tracker evaluation failed: %s", e)
                return results

            # Evaluate 24h
            for row in pending_24h:
                ticker = row["token_ticker"].lower()
                current_price = prices.get(ticker, 0)
                if current_price <= 0:
                    continue
                try:
                    self._eval_24h(conn, dict(row), current_price)
                    results["evaluated_24h"] += 1
                except Exception as e:
                    logger.warning("Tracker 24h eval failed for %s: %s", ticker, e)
                    results["errors"] += 1

            # Evaluate 7d
            for row in pending_7d:
                ticker = row["token_ticker"].lower()
                current_price = prices.get(ticker, 0)
                if current_price <= 0:
                    continue
                try:
                    self._eval_7d(conn, dict(row), current_price)
                    results["evaluated_7d"] += 1
                except Exception as e:
                    logger.warning("Tracker 7d eval failed for %s: %s", ticker, e)
                    results["errors"] += 1

            conn.commit()

            # Refresh accuracy caches
            self._refresh_token_accuracy_cache(conn)
            self._refresh_regime_accuracy_cache(conn)
            self._refresh_indicator_combo_cache(conn)
            conn.commit()

        except Exception as e:
            logger.error("PredictionTracker evaluation error: %s", e)
            results["errors"] += 1
        finally:
            conn.close()

        return results

    def _eval_24h(self, conn, pred: dict, actual_price: float):
        """Evaluate a prediction after 24 hours."""
        entry = pred["entry_price"]
        if entry <= 0:
            return
        pnl = ((actual_price - entry) / entry) * 100

        direction = pred["predicted_direction"]
        if direction == "up":
            correct = actual_price > entry
        elif direction == "down":
            correct = actual_price < entry
        else:
            correct = abs(pnl) < 2

        conn.execute('''
            UPDATE prediction_extended SET
                evaluated_24h = 1,
                actual_price_24h = ?,
                pnl_24h = ?,
                direction_correct_24h = ?
            WHERE prediction_id = ?
        ''', (actual_price, round(pnl, 2), 1 if correct else 0, pred["prediction_id"]))

    def _eval_7d(self, conn, pred: dict, actual_price: float):
        """Evaluate a prediction after 7 days."""
        entry = pred["entry_price"]
        if entry <= 0:
            return
        pnl = ((actual_price - entry) / entry) * 100

        direction = pred["predicted_direction"]
        if direction == "up":
            correct = actual_price > entry
        elif direction == "down":
            correct = actual_price < entry
        else:
            correct = abs(pnl) < 5

        target_hit = actual_price >= pred["target_price"] if pred["target_price"] > 0 else None
        stop_hit = actual_price <= pred["stop_loss_price"] if pred["stop_loss_price"] > 0 else None

        conn.execute('''
            UPDATE prediction_extended SET
                evaluated_7d = 1,
                actual_price_7d = ?,
                pnl_7d = ?,
                direction_correct_7d = ?,
                target_hit = ?,
                stop_hit = ?
            WHERE prediction_id = ?
        ''', (
            actual_price, round(pnl, 2), 1 if correct else 0,
            1 if target_hit else 0 if target_hit is not None else None,
            1 if stop_hit else 0 if stop_hit is not None else None,
            pred["prediction_id"],
        ))

    # ══════════════════════════════════════════════════════════════
    # Accuracy Cache Refresh
    # ══════════════════════════════════════════════════════════════

    def _refresh_token_accuracy_cache(self, conn):
        """Recompute per-token rolling accuracy (last 30 predictions)."""
        tickers = conn.execute(
            "SELECT DISTINCT token_ticker FROM prediction_extended WHERE evaluated_24h = 1"
        ).fetchall()

        for row in tickers:
            ticker = row["token_ticker"]
            # Rolling window: last 30 evaluated predictions
            preds = conn.execute('''
                SELECT direction_correct_24h, pnl_24h, pnl_7d
                FROM prediction_extended
                WHERE token_ticker = ? AND evaluated_24h = 1
                ORDER BY created_at DESC LIMIT 30
            ''', (ticker,)).fetchall()

            total = len(preds)
            correct = sum(1 for p in preds if p["direction_correct_24h"] == 1)
            accuracy = correct / max(total, 1)

            pnl_24h_vals = [p["pnl_24h"] for p in preds if p["pnl_24h"] is not None]
            pnl_7d_vals = [p["pnl_7d"] for p in preds if p["pnl_7d"] is not None]

            avg_pnl_24h = sum(pnl_24h_vals) / max(len(pnl_24h_vals), 1) if pnl_24h_vals else 0
            avg_pnl_7d = sum(pnl_7d_vals) / max(len(pnl_7d_vals), 1) if pnl_7d_vals else 0
            best_pnl = max(pnl_24h_vals) if pnl_24h_vals else 0
            worst_pnl = min(pnl_24h_vals) if pnl_24h_vals else 0

            conn.execute('''
                INSERT OR REPLACE INTO token_accuracy_cache
                (token_ticker, rolling_accuracy, total_predictions, correct_predictions,
                 avg_pnl_24h, avg_pnl_7d, best_pnl, worst_pnl, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (ticker, round(accuracy, 4), total, correct,
                  round(avg_pnl_24h, 2), round(avg_pnl_7d, 2),
                  round(best_pnl, 2), round(worst_pnl, 2)))

    def _refresh_regime_accuracy_cache(self, conn):
        """Recompute per-regime accuracy."""
        regimes = conn.execute(
            "SELECT DISTINCT market_regime FROM prediction_extended WHERE evaluated_24h = 1"
        ).fetchall()

        for row in regimes:
            regime = row["market_regime"]
            r_total = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE market_regime = ? AND evaluated_24h = 1",
                (regime,)
            ).fetchone()["cnt"]
            r_correct = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE market_regime = ? AND direction_correct_24h = 1",
                (regime,)
            ).fetchone()["cnt"]
            r_avg_conf = conn.execute(
                "SELECT AVG(confidence_score) as avg FROM prediction_extended WHERE market_regime = ? AND evaluated_24h = 1",
                (regime,)
            ).fetchone()["avg"] or 5.0

            conn.execute('''
                INSERT OR REPLACE INTO regime_accuracy_cache
                (market_regime, accuracy, total_predictions, correct_predictions, avg_confidence, last_updated)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            ''', (regime, round(r_correct / max(r_total, 1), 4), r_total, r_correct, round(r_avg_conf, 1)))

    def _refresh_indicator_combo_cache(self, conn):
        """Recompute per-indicator-combination accuracy."""
        combos = conn.execute(
            "SELECT DISTINCT indicator_combo FROM prediction_extended WHERE evaluated_24h = 1 AND indicator_combo != '[]'"
        ).fetchall()

        for row in combos:
            combo_str = row["indicator_combo"]
            try:
                combo_list = json.loads(combo_str)
                combo_key = "|".join(sorted(combo_list))
            except (json.JSONDecodeError, TypeError):
                combo_key = combo_str

            if not combo_key:
                continue

            c_total = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE indicator_combo = ? AND evaluated_24h = 1",
                (combo_str,)
            ).fetchone()["cnt"]
            c_correct = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE indicator_combo = ? AND direction_correct_24h = 1",
                (combo_str,)
            ).fetchone()["cnt"]
            c_avg_pnl = conn.execute(
                "SELECT AVG(pnl_24h) as avg FROM prediction_extended WHERE indicator_combo = ? AND pnl_24h IS NOT NULL",
                (combo_str,)
            ).fetchone()["avg"] or 0

            conn.execute('''
                INSERT OR REPLACE INTO indicator_combo_accuracy
                (combo_key, description, accuracy, total_predictions, correct_predictions, avg_pnl, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (
                combo_key, combo_str,
                round(c_correct / max(c_total, 1), 4),
                c_total, c_correct, round(c_avg_pnl, 2),
            ))

    # ══════════════════════════════════════════════════════════════
    # Query Methods
    # ══════════════════════════════════════════════════════════════

    def get_token_accuracy(self, ticker: str) -> TokenAccuracy:
        """Get rolling accuracy for a specific token."""
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM token_accuracy_cache WHERE token_ticker = ?",
                (ticker.upper(),)
            ).fetchone()
            if row:
                return TokenAccuracy(
                    token_ticker=ticker.upper(),
                    rolling_accuracy=row["rolling_accuracy"],
                    total_predictions=row["total_predictions"],
                    correct_predictions=row["correct_predictions"],
                    avg_pnl_24h=row["avg_pnl_24h"],
                    avg_pnl_7d=row["avg_pnl_7d"],
                    best_pnl=row["best_pnl"],
                    worst_pnl=row["worst_pnl"],
                )
            return TokenAccuracy(token_ticker=ticker.upper())
        finally:
            conn.close()

    def get_regime_accuracy(self, regime: str) -> RegimeAccuracy:
        """Get accuracy stats for a specific market regime."""
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM regime_accuracy_cache WHERE market_regime = ?",
                (regime,)
            ).fetchone()
            if row:
                return RegimeAccuracy(
                    market_regime=regime,
                    accuracy=row["accuracy"],
                    total_predictions=row["total_predictions"],
                    correct_predictions=row["correct_predictions"],
                    avg_confidence=row["avg_confidence"],
                )
            return RegimeAccuracy(market_regime=regime)
        finally:
            conn.close()

    def get_historical_accuracy_for_confidence(self) -> float:
        """
        Get the overall historical accuracy (0-1) for ConfidenceScorer feedback.
        Uses the extended prediction table for a more comprehensive view.
        """
        conn = _get_db()
        try:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE evaluated_24h = 1"
            ).fetchone()["cnt"]
            correct = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE direction_correct_24h = 1"
            ).fetchone()["cnt"]

            if total < 10:
                return 0.5  # Not enough data — neutral modifier
            return round(correct / total, 4)
        finally:
            conn.close()

    def get_feedback(self, days: int = 30) -> PredictionFeedback:
        """
        Get a comprehensive feedback summary for the AI pipeline.
        Includes per-token, per-regime, and per-indicator accuracy.
        """
        conn = _get_db()
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Overall
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE created_at > ?",
                (cutoff,)
            ).fetchone()["cnt"]
            evaluated = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE evaluated_24h = 1 AND created_at > ?",
                (cutoff,)
            ).fetchone()["cnt"]
            correct_24h = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE direction_correct_24h = 1 AND created_at > ?",
                (cutoff,)
            ).fetchone()["cnt"]
            correct_7d = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE direction_correct_7d = 1 AND created_at > ?",
                (cutoff,)
            ).fetchone()["cnt"]
            eval_7d = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_extended WHERE evaluated_7d = 1 AND created_at > ?",
                (cutoff,)
            ).fetchone()["cnt"]

            # Confidence calibration
            avg_conf_correct = conn.execute(
                "SELECT AVG(confidence_score) as avg FROM prediction_extended WHERE direction_correct_24h = 1 AND created_at > ?",
                (cutoff,)
            ).fetchone()["avg"] or 0
            avg_conf_incorrect = conn.execute(
                "SELECT AVG(confidence_score) as avg FROM prediction_extended WHERE direction_correct_24h = 0 AND created_at > ?",
                (cutoff,)
            ).fetchone()["avg"] or 0

            # Token accuracies
            token_rows = conn.execute(
                "SELECT * FROM token_accuracy_cache ORDER BY total_predictions DESC LIMIT 50"
            ).fetchall()
            token_accs = {}
            for r in token_rows:
                token_accs[r["token_ticker"]] = TokenAccuracy(
                    token_ticker=r["token_ticker"],
                    rolling_accuracy=r["rolling_accuracy"],
                    total_predictions=r["total_predictions"],
                    correct_predictions=r["correct_predictions"],
                    avg_pnl_24h=r["avg_pnl_24h"],
                    avg_pnl_7d=r["avg_pnl_7d"],
                    best_pnl=r["best_pnl"],
                    worst_pnl=r["worst_pnl"],
                )

            # Regime accuracies
            regime_rows = conn.execute(
                "SELECT * FROM regime_accuracy_cache"
            ).fetchall()
            regime_accs = {}
            for r in regime_rows:
                regime_accs[r["market_regime"]] = RegimeAccuracy(
                    market_regime=r["market_regime"],
                    accuracy=r["accuracy"],
                    total_predictions=r["total_predictions"],
                    correct_predictions=r["correct_predictions"],
                    avg_confidence=r["avg_confidence"],
                )

            # Best and worst indicator combos
            best_combos = conn.execute(
                "SELECT * FROM indicator_combo_accuracy WHERE total_predictions >= 3 ORDER BY accuracy DESC LIMIT 5"
            ).fetchall()
            worst_combos = conn.execute(
                "SELECT * FROM indicator_combo_accuracy WHERE total_predictions >= 3 ORDER BY accuracy ASC LIMIT 5"
            ).fetchall()

            return PredictionFeedback(
                overall_accuracy_24h=round(correct_24h / max(evaluated, 1), 4),
                overall_accuracy_7d=round(correct_7d / max(eval_7d, 1), 4),
                total_predictions=total,
                total_evaluated=evaluated,
                token_accuracies=token_accs,
                regime_accuracies=regime_accs,
                best_indicator_combos=[dict(r) for r in best_combos],
                worst_indicator_combos=[dict(r) for r in worst_combos],
                confidence_calibration=round(avg_conf_correct - avg_conf_incorrect, 2),
            )
        finally:
            conn.close()

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get a dashboard-ready summary of prediction performance."""
        feedback = self.get_feedback(days=30)
        return {
            "overall_accuracy_24h": round(feedback.overall_accuracy_24h * 100, 1),
            "overall_accuracy_7d": round(feedback.overall_accuracy_7d * 100, 1),
            "total_predictions": feedback.total_predictions,
            "total_evaluated": feedback.total_evaluated,
            "confidence_calibration_gap": feedback.confidence_calibration,
            "top_tokens": [
                {
                    "ticker": ta.token_ticker,
                    "accuracy": round(ta.rolling_accuracy * 100, 1),
                    "predictions": ta.total_predictions,
                    "avg_pnl": ta.avg_pnl_24h,
                }
                for ta in sorted(
                    feedback.token_accuracies.values(),
                    key=lambda x: x.rolling_accuracy,
                    reverse=True,
                )[:10]
            ],
            "regime_performance": {
                regime: {
                    "accuracy": round(ra.accuracy * 100, 1),
                    "predictions": ra.total_predictions,
                }
                for regime, ra in feedback.regime_accuracies.items()
            },
            "best_indicator_combos": [
                {"combo": c["combo_key"], "accuracy": round(c["accuracy"] * 100, 1), "trades": c["total_predictions"]}
                for c in feedback.best_indicator_combos
            ],
            "worst_indicator_combos": [
                {"combo": c["combo_key"], "accuracy": round(c["accuracy"] * 100, 1), "trades": c["total_predictions"]}
                for c in feedback.worst_indicator_combos
            ],
        }

    def get_total_predictions_count(self) -> int:
        """Get total number of predictions recorded (for platform stats)."""
        conn = _get_db()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM prediction_extended").fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()


# ── Singleton ─────────────────────────────────────────────────────

_tracker_instance: Optional[PredictionTracker] = None


def get_prediction_tracker() -> PredictionTracker:
    """Get or create the singleton PredictionTracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PredictionTracker()
    return _tracker_instance
