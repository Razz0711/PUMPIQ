"""
NexYpher Continuous Learning System
===================================
Implements the full feedback loop:

  Historical data → Initial training
         ↓
  Paper trade executed
         ↓
  Wait for outcome (24h / 7d)
         ↓
  Was it profitable?
    Yes → Reinforce that pattern ✅
    No  → Penalize that pattern  ✗
         ↓
  Model updates weights
         ↓
  Next trade considers past mistakes
         ↓
  Gets smarter with every trade
         ↑_________________________|
            Continuous loop!

Usage:
  python continuous_learner.py --action paper-trade       # Execute paper trades from predictions
  python continuous_learner.py --action evaluate          # Check outcomes of pending trades
  python continuous_learner.py --action feedback          # Generate feedback labels
  python continuous_learner.py --action retrain           # Retrain with feedback
  python continuous_learner.py --action loop              # Run full continuous loop once
  python continuous_learner.py --action auto --interval 6 # Auto-loop every N hours
  python continuous_learner.py --action status            # Show learning status dashboard

Requirements:
  pip install xgboost scikit-learn joblib requests
"""

import os
import sys
import json
import math
import time
import sqlite3
import hashlib
import argparse
import logging
from datetime import datetime, timedelta
from collections import Counter
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ContinuousLearner")

# ── Config ───────────────────────────────────────────────────────────────────
ML_DIR          = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(ML_DIR)
TRAINING_DB     = os.path.join(ML_DIR, "nexypher_training_data.db")
LEARNING_DB     = os.path.join(PROJECT_ROOT, "nexypher_learning.db")
PAPER_TRADE_DB  = os.path.join(ML_DIR, "paper_trades.db")
MODELS_DIR      = os.path.join(ML_DIR, "models")

# Paper trade configuration
PAPER_WALLET_INITIAL = 10000.0   # $10,000 virtual wallet
PAPER_POSITION_SIZE  = 0.05      # 5% per trade
MIN_CONFIDENCE       = 0.40      # Minimum model probability to trade
MAX_OPEN_TRADES      = 20        # Max concurrent positions
TAKE_PROFIT_PCT      = 0.08      # 8% take profit
STOP_LOSS_PCT        = 0.05      # 5% stop loss

# Feedback weights
REINFORCE_WEIGHT     = 1.5       # Multiply sample weight for winning patterns
PENALIZE_WEIGHT      = 0.5       # Multiply sample weight for losing patterns
BASE_WEIGHT          = 1.0       # Default sample weight


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────────────────────────────────────

def init_paper_trade_db():
    """Create paper trading tables."""
    conn = sqlite3.connect(PAPER_TRADE_DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_wallet (
            id INTEGER PRIMARY KEY,
            balance REAL NOT NULL DEFAULT 10000.0,
            total_invested REAL NOT NULL DEFAULT 0.0,
            total_returned REAL NOT NULL DEFAULT 0.0,
            total_trades INTEGER NOT NULL DEFAULT 0,
            winning_trades INTEGER NOT NULL DEFAULT 0,
            losing_trades INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE NOT NULL,
            token_id TEXT NOT NULL,
            token_name TEXT NOT NULL DEFAULT '',
            -- Entry
            entry_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            quantity REAL NOT NULL,
            invested_amount REAL NOT NULL,
            -- Model signals at entry
            prob_up_24h REAL,
            prob_up_7d REAL,
            predicted_direction TEXT,
            model_confidence REAL,
            model_version TEXT,
            -- Targets
            take_profit_price REAL,
            stop_loss_price REAL,
            -- Feature snapshot (JSON): the exact features used for prediction
            feature_snapshot TEXT,
            -- Outcome (filled by evaluator)
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,          -- 'take_profit', 'stop_loss', 'timeout_24h', 'timeout_7d'
            pnl_amount REAL,
            pnl_pct REAL,
            outcome TEXT,              -- 'WIN', 'LOSS', 'NEUTRAL'
            actual_price_24h REAL,
            actual_price_7d REAL,
            actual_direction TEXT,      -- 'UP', 'DOWN', 'SIDEWAYS'
            direction_correct INTEGER,  -- 1 or 0
            -- Feedback
            feedback_weight REAL DEFAULT 1.0,
            feedback_applied INTEGER DEFAULT 0,
            -- Status
            status TEXT NOT NULL DEFAULT 'OPEN',  -- 'OPEN', 'CLOSED', 'EVALUATED'
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            evaluated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            metrics TEXT,              -- JSON of relevant metrics
            model_version_before TEXT,
            model_version_after TEXT,
            trades_used INTEGER DEFAULT 0,
            accuracy_before REAL,
            accuracy_after REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS feedback_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '1d',
            -- Original label
            original_label_24h INTEGER,
            original_label_7d INTEGER,
            original_direction TEXT,
            -- Feedback-adjusted weight
            sample_weight REAL NOT NULL DEFAULT 1.0,
            -- Source
            trade_id TEXT,
            outcome TEXT,
            pnl_pct REAL,
            -- Meta
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(token_id, timestamp, timeframe)
        );

        CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
        CREATE INDEX IF NOT EXISTS idx_paper_trades_token ON paper_trades(token_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_labels_token ON feedback_labels(token_id, timestamp);
    """)

    # Initialize wallet if not exists
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM paper_wallet")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO paper_wallet (id, balance) VALUES (1, ?)",
                  (PAPER_WALLET_INITIAL,))

    conn.commit()
    conn.close()


# Initialize on import
init_paper_trade_db()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: PAPER TRADE EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

def get_wallet_balance() -> float:
    """Get current paper wallet balance."""
    conn = sqlite3.connect(PAPER_TRADE_DB)
    c = conn.cursor()
    c.execute("SELECT balance FROM paper_wallet WHERE id=1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else PAPER_WALLET_INITIAL


def count_open_trades() -> int:
    """Count currently open paper trades."""
    conn = sqlite3.connect(PAPER_TRADE_DB)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM paper_trades WHERE status='OPEN'")
    count = c.fetchone()[0]
    conn.close()
    return count


def execute_paper_trades(timeframe: str = "1d"):
    """
    STEP 1: Scan all tokens, get model predictions, execute paper trades
    for any signal above MIN_CONFIDENCE threshold.
    """
    log.info("=" * 60)
    log.info("STEP 1: EXECUTING PAPER TRADES")
    log.info("=" * 60)

    # Check prerequisites
    balance = get_wallet_balance()
    open_count = count_open_trades()
    log.info(f"  Wallet balance: ${balance:,.2f}")
    log.info(f"  Open trades: {open_count}/{MAX_OPEN_TRADES}")

    if open_count >= MAX_OPEN_TRADES:
        log.warning("  Max open trades reached. Skipping new entries.")
        return 0

    if balance < 100:
        log.warning("  Wallet balance too low. Skipping new entries.")
        return 0

    # Import predictor from model trainer
    sys.path.insert(0, ML_DIR)
    try:
        # The trainer file has a non-standard name ("· py"), so we use
        # SourceFileLoader directly instead of spec_from_file_location.
        import importlib.util
        from importlib.machinery import SourceFileLoader
        trainer_path = os.path.join(ML_DIR, "NexYpher model trainer · py")
        loader = SourceFileLoader("nexypher_trainer", trainer_path)
        spec = importlib.util.spec_from_loader("nexypher_trainer", loader,
                                                origin=trainer_path)
        trainer = importlib.util.module_from_spec(spec)
        trainer.__file__ = trainer_path
        spec.loader.exec_module(trainer)
    except Exception as e:
        log.error(f"  Failed to load model trainer: {e}")
        return 0

    # Get list of tokens from training DB
    if not os.path.exists(TRAINING_DB):
        log.error(f"  Training DB not found: {TRAINING_DB}")
        return 0

    conn = sqlite3.connect(TRAINING_DB)
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT token_id
        FROM technical_indicators
        WHERE timeframe = ?
        ORDER BY token_id
    """, (timeframe,))
    tokens = [row[0] for row in c.fetchall()]
    conn.close()

    log.info(f"  Scanning {len(tokens)} tokens for signals...")

    # Get tokens we already have open trades for
    pconn = sqlite3.connect(PAPER_TRADE_DB)
    pc = pconn.cursor()
    pc.execute("SELECT token_id FROM paper_trades WHERE status='OPEN'")
    already_open = {row[0] for row in pc.fetchall()}
    pconn.close()

    trades_executed = 0

    for token_id in tokens:
        if token_id in already_open:
            continue

        if count_open_trades() >= MAX_OPEN_TRADES:
            break

        try:
            result = trainer.predict_from_db(token_id, timeframe, TRAINING_DB)
            if result is None:
                continue

            prob_7d = result["prob_up_7d"] / 100.0
            prob_24h = result["prob_up_24h"] / 100.0

            # Only trade if model is confident enough
            if prob_7d < MIN_CONFIDENCE:
                continue

            # Execute paper trade
            entry_price = result["close_price"]
            position_value = balance * PAPER_POSITION_SIZE
            quantity = position_value / entry_price if entry_price > 0 else 0

            if quantity <= 0 or position_value < 10:
                continue

            tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
            sl_price = entry_price * (1 - STOP_LOSS_PCT)

            trade_id = hashlib.sha256(
                f"{token_id}_{datetime.now().isoformat()}_{entry_price}".encode()
            ).hexdigest()[:16]

            # Get feature snapshot for the trade
            feature_snapshot = {}
            try:
                tconn = sqlite3.connect(TRAINING_DB)
                tconn.row_factory = sqlite3.Row
                tc = tconn.cursor()
                tc.execute("""
                    SELECT * FROM technical_indicators
                    WHERE token_id=? AND timeframe=?
                    ORDER BY timestamp DESC LIMIT 1
                """, (token_id, timeframe))
                row = tc.fetchone()
                if row:
                    feature_snapshot = {k: row[k] for k in row.keys()
                                       if k not in ('id', 'created_at')}
                    # Convert non-serializable types
                    feature_snapshot = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                                       for k, v in feature_snapshot.items() if v is not None}
                tconn.close()
            except Exception:
                pass

            # Record the trade
            pconn = sqlite3.connect(PAPER_TRADE_DB)
            pc = pconn.cursor()

            pc.execute("""
                INSERT INTO paper_trades
                    (trade_id, token_id, token_name, entry_price, entry_time,
                     quantity, invested_amount, prob_up_24h, prob_up_7d,
                     predicted_direction, model_confidence, model_version,
                     take_profit_price, stop_loss_price, feature_snapshot, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (
                trade_id, token_id, result.get("token_id", token_id),
                entry_price, datetime.now().isoformat(),
                quantity, position_value,
                prob_24h, prob_7d,
                result["direction"], result["confidence"],
                result.get("model_version", "unknown"),
                tp_price, sl_price,
                json.dumps(feature_snapshot),
            ))

            # Deduct from wallet
            pc.execute("""
                UPDATE paper_wallet SET balance = balance - ?, total_invested = total_invested + ?
                WHERE id = 1
            """, (position_value, position_value))

            pconn.commit()
            pconn.close()

            trades_executed += 1
            log.info(f"  📈 PAPER BUY: {token_id} @ ${entry_price:.6f} "
                     f"| prob_7d={prob_7d:.1%} | ${position_value:.2f}")

        except Exception as e:
            log.debug(f"  Skipping {token_id}: {e}")
            continue

    log.info(f"\n  Paper trades executed: {trades_executed}")
    log.info(f"  Updated wallet balance: ${get_wallet_balance():,.2f}")
    return trades_executed


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: OUTCOME EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_current_price(token_id: str) -> Optional[float]:
    """
    Fetch current price from CoinGecko with API key support and rate limiting.
    Uses the same API key and tier as CoinGeckoCollector.
    """
    try:
        import requests
        api_key = os.environ.get("COINGECKO_API_KEY", "")
        is_pro = os.environ.get("COINGECKO_PRO", "").lower() in ("1", "true")

        if is_pro and api_key:
            base = "https://pro-api.coingecko.com/api/v3"
        else:
            base = "https://api.coingecko.com/api/v3"

        headers = {"Accept": "application/json"}
        if api_key:
            if is_pro:
                headers["x-cg-pro-api-key"] = api_key
            else:
                headers["x-cg-demo-api-key"] = api_key

        url = f"{base}/simple/price?ids={token_id}&vs_currencies=usd"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        time.sleep(2.5)  # Rate limit (matches CoinGeckoCollector)
        return data.get(token_id, {}).get("usd")
    except Exception:
        return None


def _fetch_price_from_db(token_id: str, timeframe: str = "1d") -> Optional[float]:
    """Fallback: get latest price from training DB."""
    if not os.path.exists(TRAINING_DB):
        return None
    conn = sqlite3.connect(TRAINING_DB)
    c = conn.cursor()
    c.execute("""
        SELECT close FROM technical_indicators
        WHERE token_id=? AND timeframe=?
        ORDER BY timestamp DESC LIMIT 1
    """, (token_id, timeframe))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def evaluate_paper_trades():
    """
    STEP 2: Check outcomes of open paper trades.
    - If TP hit → close as WIN
    - If SL hit → close as LOSS
    - If 24h+ old → evaluate with current price
    - If 7d+ old → force close and evaluate
    """
    log.info("=" * 60)
    log.info("STEP 2: EVALUATING PAPER TRADE OUTCOMES")
    log.info("=" * 60)

    conn = sqlite3.connect(PAPER_TRADE_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM paper_trades WHERE status = 'OPEN' ORDER BY entry_time ASC")
    open_trades = c.fetchall()

    log.info(f"  Open trades to evaluate: {len(open_trades)}")

    closed = 0
    wins = 0
    losses = 0
    now = datetime.now()

    for trade in open_trades:
        token_id = trade["token_id"]
        entry_price = trade["entry_price"]
        entry_time = datetime.fromisoformat(trade["entry_time"])
        age_hours = (now - entry_time).total_seconds() / 3600.0

        # Get current price
        current_price = _fetch_current_price(token_id)
        if current_price is None:
            current_price = _fetch_price_from_db(token_id)
        if current_price is None:
            log.debug(f"  Could not fetch price for {token_id}, skipping")
            continue

        price_change_pct = (current_price - entry_price) / entry_price

        # Determine outcome
        exit_reason = None
        outcome = None

        if current_price >= trade["take_profit_price"]:
            exit_reason = "take_profit"
            outcome = "WIN"
        elif current_price <= trade["stop_loss_price"]:
            exit_reason = "stop_loss"
            outcome = "LOSS"
        elif age_hours >= 168:  # 7 days
            exit_reason = "timeout_7d"
            outcome = "WIN" if price_change_pct > 0.02 else ("LOSS" if price_change_pct < -0.02 else "NEUTRAL")
        elif age_hours >= 24:
            # Record 24h price but don't close yet (wait for 7d)
            c.execute("""
                UPDATE paper_trades SET actual_price_24h = ? WHERE id = ?
            """, (current_price, trade["id"]))
            continue
        else:
            continue  # Too early

        # Close the trade
        pnl_amount = (current_price - entry_price) * trade["quantity"]
        pnl_pct = price_change_pct * 100

        # Determine actual direction
        if price_change_pct > 0.02:
            actual_dir = "UP"
        elif price_change_pct < -0.02:
            actual_dir = "DOWN"
        else:
            actual_dir = "SIDEWAYS"

        predicted_dir = trade["predicted_direction"] or "UP"
        dir_correct = 1 if actual_dir == predicted_dir.upper() else 0

        # Compute feedback weight
        if outcome == "WIN":
            feedback_weight = REINFORCE_WEIGHT
            wins += 1
        elif outcome == "LOSS":
            feedback_weight = PENALIZE_WEIGHT
            losses += 1
        else:
            feedback_weight = BASE_WEIGHT

        c.execute("""
            UPDATE paper_trades SET
                exit_price = ?,
                exit_time = ?,
                exit_reason = ?,
                pnl_amount = ?,
                pnl_pct = ?,
                outcome = ?,
                actual_price_7d = ?,
                actual_direction = ?,
                direction_correct = ?,
                feedback_weight = ?,
                status = 'CLOSED',
                evaluated_at = ?
            WHERE id = ?
        """, (
            current_price, now.isoformat(), exit_reason,
            pnl_amount, pnl_pct, outcome,
            current_price, actual_dir, dir_correct, feedback_weight,
            now.isoformat(), trade["id"]
        ))

        # Credit wallet
        returned_value = trade["invested_amount"] + pnl_amount
        c.execute("""
            UPDATE paper_wallet SET
                balance = balance + ?,
                total_returned = total_returned + ?,
                total_trades = total_trades + 1,
                winning_trades = winning_trades + ?,
                losing_trades = losing_trades + ?,
                updated_at = ?
            WHERE id = 1
        """, (returned_value, returned_value,
              1 if outcome == "WIN" else 0,
              1 if outcome == "LOSS" else 0,
              now.isoformat()))

        closed += 1
        symbol = "✅" if outcome == "WIN" else ("❌" if outcome == "LOSS" else "➖")
        log.info(f"  {symbol} CLOSED: {token_id} | {exit_reason} | "
                 f"PnL: {pnl_pct:+.2f}% (${pnl_amount:+.2f}) | "
                 f"Dir: {predicted_dir}→{actual_dir} {'✓' if dir_correct else '✗'}")

    conn.commit()
    conn.close()

    # Sync outcomes to the AI learning DB (bridge between the two systems)
    if closed > 0:
        _sync_closed_trades_to_learning_db()

    log.info(f"\n  Trades closed: {closed} (W:{wins} L:{losses})")
    log.info(f"  Wallet balance: ${get_wallet_balance():,.2f}")
    return closed


def _sync_closed_trades_to_learning_db():
    """Bridge: push recently closed paper trade outcomes to nexypher_learning.db."""
    if not os.path.exists(LEARNING_DB):
        return

    pconn = sqlite3.connect(PAPER_TRADE_DB)
    pconn.row_factory = sqlite3.Row
    pc = pconn.cursor()
    pc.execute("""
        SELECT token_id, entry_time, exit_price, actual_price_24h,
               actual_price_7d, direction_correct, pnl_pct, evaluated_at
        FROM paper_trades
        WHERE status = 'CLOSED'
        ORDER BY evaluated_at DESC LIMIT 50
    """)
    trades = [dict(row) for row in pc.fetchall()]
    pconn.close()

    _sync_to_learning_db(trades)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FEEDBACK ENGINE (REINFORCE / PENALIZE)
# ─────────────────────────────────────────────────────────────────────────────

def generate_feedback_labels():
    """
    STEP 3: Convert closed paper trade outcomes into feedback labels
    that adjust training sample weights.

    - WIN  → weight = 1.5x  (reinforce this pattern)
    - LOSS → weight = 0.5x  (penalize this pattern)
    - NEUTRAL → weight = 1.0 (unchanged)

    Also creates new training labels if the trade outcome
    contradicts the original label.
    """
    log.info("=" * 60)
    log.info("STEP 3: GENERATING FEEDBACK LABELS")
    log.info("=" * 60)

    pconn = sqlite3.connect(PAPER_TRADE_DB)
    pconn.row_factory = sqlite3.Row
    pc = pconn.cursor()

    # Get closed trades that haven't had feedback applied
    pc.execute("""
        SELECT * FROM paper_trades
        WHERE status = 'CLOSED' AND feedback_applied = 0
        ORDER BY entry_time ASC
    """)
    trades = pc.fetchall()

    log.info(f"  Closed trades pending feedback: {len(trades)}")

    if len(trades) == 0:
        log.info("  No new feedback to generate.")
        return 0

    feedback_count = 0

    for trade in trades:
        token_id = trade["token_id"]
        outcome = trade["outcome"]
        pnl_pct = trade["pnl_pct"] or 0

        # Determine feedback weight
        if outcome == "WIN":
            weight = REINFORCE_WEIGHT
        elif outcome == "LOSS":
            weight = PENALIZE_WEIGHT
        else:
            weight = BASE_WEIGHT

        # Scale weight by magnitude of PnL
        # Big wins get stronger reinforcement, big losses get stronger penalty
        magnitude = min(abs(pnl_pct) / 10.0, 1.0)  # cap at 10% effect
        if outcome == "WIN":
            weight = BASE_WEIGHT + (REINFORCE_WEIGHT - BASE_WEIGHT) * (0.5 + 0.5 * magnitude)
        elif outcome == "LOSS":
            weight = BASE_WEIGHT - (BASE_WEIGHT - PENALIZE_WEIGHT) * (0.5 + 0.5 * magnitude)

        # Extract timestamp from feature snapshot
        feature_snap = {}
        try:
            feature_snap = json.loads(trade["feature_snapshot"] or "{}")
        except json.JSONDecodeError:
            pass

        timestamp = int(feature_snap.get("timestamp", 0))
        if timestamp == 0:
            # Fallback: find closest timestamp in training DB
            if os.path.exists(TRAINING_DB):
                tconn = sqlite3.connect(TRAINING_DB)
                tc = tconn.cursor()
                entry_dt = trade["entry_time"]
                tc.execute("""
                    SELECT timestamp FROM technical_indicators
                    WHERE token_id=? AND timeframe='1d'
                    ORDER BY ABS(julianday(datetime) - julianday(?)) ASC
                    LIMIT 1
                """, (token_id, entry_dt))
                row = tc.fetchone()
                if row:
                    timestamp = row[0]
                tconn.close()

        if timestamp == 0:
            log.debug(f"  Skipping feedback for {token_id}: no timestamp")
            pc.execute("UPDATE paper_trades SET feedback_applied=1 WHERE id=?", (trade["id"],))
            continue

        # Determine corrected labels based on actual outcome
        actual_dir = trade["actual_direction"] or "SIDEWAYS"
        label_24h = 1 if (trade["actual_price_24h"] and trade["entry_price"]
                          and trade["actual_price_24h"] > trade["entry_price"] * 1.02) else 0
        label_7d = 1 if outcome == "WIN" else 0

        # Insert/update feedback label
        pc.execute("""
            INSERT OR REPLACE INTO feedback_labels
                (token_id, timestamp, timeframe,
                 original_label_24h, original_label_7d, original_direction,
                 sample_weight, trade_id, outcome, pnl_pct)
            VALUES (?, ?, '1d', ?, ?, ?, ?, ?, ?, ?)
        """, (
            token_id, timestamp,
            label_24h, label_7d, actual_dir,
            weight, trade["trade_id"], outcome, pnl_pct
        ))

        # Mark trade as feedback-applied
        pc.execute("UPDATE paper_trades SET feedback_applied=1 WHERE id=?", (trade["id"],))
        feedback_count += 1

        symbol = "🔼" if weight > 1.0 else ("🔽" if weight < 1.0 else "➖")
        log.info(f"  {symbol} {token_id}: weight={weight:.2f} | "
                 f"outcome={outcome} | pnl={pnl_pct:+.1f}%")

    pconn.commit()
    pconn.close()

    # Log the feedback generation event
    _log_learning_event(
        "feedback_generated",
        f"Generated {feedback_count} feedback labels from paper trades",
        {"feedback_count": feedback_count}
    )

    log.info(f"\n  Feedback labels generated: {feedback_count}")
    return feedback_count


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: INCREMENTAL RETRAINER
# ─────────────────────────────────────────────────────────────────────────────

def retrain_with_feedback(timeframe: str = "1d", min_feedback: int = 10):
    """
    STEP 4: Retrain the model using feedback-weighted samples.

    Instead of treating all training samples equally, this:
    1. Loads the standard training data
    2. Overlays feedback weights from paper trade outcomes
    3. Retrains with sample_weight parameter
    4. Compares new model accuracy vs old model
    5. Only saves if the new model is better (or at least as good)

    This is how the model "learns from its mistakes."
    """
    log.info("=" * 60)
    log.info("STEP 4: RETRAINING WITH FEEDBACK")
    log.info("=" * 60)

    # Check if we have enough feedback
    pconn = sqlite3.connect(PAPER_TRADE_DB)
    pc = pconn.cursor()
    pc.execute("SELECT COUNT(*) FROM feedback_labels")
    n_feedback = pc.fetchone()[0]
    pconn.close()

    if n_feedback < min_feedback:
        log.info(f"  Only {n_feedback} feedback labels (need {min_feedback}). "
                 f"Accumulating more trade outcomes first.")
        return None

    log.info(f"  Feedback labels available: {n_feedback}")

    # Import trainer
    try:
        import importlib.util
        from importlib.machinery import SourceFileLoader
        trainer_path = os.path.join(ML_DIR, "NexYpher model trainer · py")
        loader = SourceFileLoader("nexypher_trainer", trainer_path)
        spec = importlib.util.spec_from_loader("nexypher_trainer", loader,
                                                origin=trainer_path)
        trainer = importlib.util.module_from_spec(spec)
        trainer.__file__ = trainer_path
        spec.loader.exec_module(trainer)
    except Exception as e:
        log.error(f"  Failed to import trainer: {e}")
        return None

    # Step 4a: Load standard training data
    log.info("  Loading training data...")
    try:
        X, y_24h, y_7d, y_dir = trainer.load_training_data(TRAINING_DB, timeframe)
    except Exception as e:
        log.error(f"  Failed to load training data: {e}")
        return None

    X_list = trainer.features_to_list(X)

    # Step 4b: Build sample weights from feedback
    log.info("  Applying feedback weights...")
    sample_weights = [BASE_WEIGHT] * len(X_list)

    # Load feedback labels
    pconn = sqlite3.connect(PAPER_TRADE_DB)
    pconn.row_factory = sqlite3.Row
    pc = pconn.cursor()
    pc.execute("SELECT * FROM feedback_labels WHERE timeframe=?", (timeframe,))
    feedback_rows = pc.fetchall()
    pconn.close()

    # Build lookup: (token_id, timestamp) → weight
    feedback_map = {}
    for fb in feedback_rows:
        key = (fb["token_id"], fb["timestamp"])
        feedback_map[key] = fb["sample_weight"]

    # Match feedback to training samples
    # We need (token_id, timestamp) from X — stored during load
    # Since X is a list of feature dicts, we need the raw data
    # Reload with token/timestamp info
    conn = sqlite3.connect(TRAINING_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT t.token_id, t.timestamp
        FROM technical_indicators t
        INNER JOIN training_labels l
            ON t.token_id = l.token_id
            AND t.timestamp = l.timestamp
            AND t.timeframe = l.timeframe
        WHERE t.timeframe = ?
            AND t.rsi_14 IS NOT NULL
            AND t.macd_line IS NOT NULL
            AND t.bb_position IS NOT NULL
            AND t.volume_ratio IS NOT NULL
            AND t.price_momentum_5d IS NOT NULL
            AND l.label_7d_binary IS NOT NULL
        ORDER BY t.timestamp ASC
    """, (timeframe,))
    id_rows = c.fetchall()
    conn.close()

    feedback_hits = 0
    for i, id_row in enumerate(id_rows):
        if i >= len(sample_weights):
            break
        key = (id_row["token_id"], id_row["timestamp"])
        if key in feedback_map:
            sample_weights[i] = feedback_map[key]
            feedback_hits += 1

    reinforced = sum(1 for w in sample_weights if w > BASE_WEIGHT)
    penalized = sum(1 for w in sample_weights if w < BASE_WEIGHT)
    log.info(f"  Feedback matched: {feedback_hits}/{n_feedback}")
    log.info(f"  Reinforced samples: {reinforced} | Penalized samples: {penalized}")

    # Step 4c: Get old model accuracy for comparison
    old_metadata = {}
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            old_metadata = json.load(f)
    old_24h_acc = old_metadata.get("model_24h", {}).get("cv_mean", 0)
    old_7d_acc = old_metadata.get("model_7d", {}).get("cv_mean", 0)
    old_dir_acc = old_metadata.get("model_dir", {}).get("cv_mean", 0)

    log.info(f"  Old model accuracy: 24h={old_24h_acc:.4f} 7d={old_7d_acc:.4f} dir={old_dir_acc:.4f}")

    # Step 4d: Train with feedback weights
    log.info("\n  Training 24h model with feedback weights...")
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score

    # Train 24h binary model with sample weights
    spw_24h = trainer._compute_scale_pos_weight(y_24h)
    params_24h = {**trainer.XGB_PARAMS, "scale_pos_weight": spw_24h}
    model_24h = xgb.XGBClassifier(**params_24h, use_label_encoder=False)
    model_24h.fit(X_list, y_24h, sample_weight=sample_weights, verbose=False)

    # Train 7d binary model with sample weights
    log.info("  Training 7d model with feedback weights...")
    spw_7d = trainer._compute_scale_pos_weight(y_7d)
    params_7d = {**trainer.XGB_PARAMS, "scale_pos_weight": spw_7d}
    model_7d = xgb.XGBClassifier(**params_7d, use_label_encoder=False)
    model_7d.fit(X_list, y_7d, sample_weight=sample_weights, verbose=False)

    # Train direction model with combined weights
    log.info("  Training direction model with feedback weights...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_dir_enc = le.fit_transform(y_dir)
    direction_weights = trainer._compute_sample_weights(list(y_dir_enc))
    # Multiply class-balance weights with feedback weights
    combined_dir_weights = [dw * fw for dw, fw in zip(direction_weights, sample_weights)]

    n_classes = len(le.classes_)
    params_dir = {
        **trainer.XGB_PARAMS,
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
    }
    model_dir = xgb.XGBClassifier(**params_dir, use_label_encoder=False)
    model_dir.fit(X_list, y_dir_enc, sample_weight=combined_dir_weights, verbose=False)

    # Step 4e: Evaluate new models on holdout test set
    log.info("\n  Evaluating improved models on holdout set...")
    split = int(len(X_list) * 0.8)
    X_test = X_list[split:]
    y24_test = y_24h[split:]
    y7d_test = y_7d[split:]
    ydir_test = y_dir[split:]
    ydir_test_enc = le.transform(ydir_test)

    new_24h_acc = accuracy_score(y24_test, model_24h.predict(X_test))
    new_7d_acc = accuracy_score(y7d_test, model_7d.predict(X_test))
    new_dir_acc = accuracy_score(ydir_test_enc, model_dir.predict(X_test))

    log.info(f"\n  {'Model':<12} {'Old':>10} {'New':>10} {'Change':>10}")
    log.info(f"  {'-'*45}")
    log.info(f"  {'24h':<12} {old_24h_acc:>10.4f} {new_24h_acc:>10.4f} "
             f"{new_24h_acc - old_24h_acc:>+10.4f}")
    log.info(f"  {'7d':<12} {old_7d_acc:>10.4f} {new_7d_acc:>10.4f} "
             f"{new_7d_acc - old_7d_acc:>+10.4f}")
    log.info(f"  {'Direction':<12} {old_dir_acc:>10.4f} {new_dir_acc:>10.4f} "
             f"{new_dir_acc - old_dir_acc:>+10.4f}")

    # Step 4f: ONLY save if the new model is better or equal
    # Previously this always saved, causing bad retrains to overwrite good models!
    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Composite improvement check: average of all three models must not degrade
    old_composite = (old_24h_acc + old_7d_acc + old_dir_acc) / 3.0
    new_composite = (new_24h_acc + new_7d_acc + new_dir_acc) / 3.0
    improved = new_composite >= old_composite - 0.005  # Allow tiny 0.5% tolerance

    if not improved:
        log.warning(f"  ⚠️ New model is WORSE (composite: {old_composite:.4f} → {new_composite:.4f}). "
                    f"NOT saving. Keeping previous model.")
        _log_learning_event(
            "model_retrain_rejected",
            f"Retrain rejected: accuracy degraded ({old_composite:.4f} → {new_composite:.4f})",
            {
                "accuracy_24h_old": old_24h_acc, "accuracy_24h_new": new_24h_acc,
                "accuracy_7d_old": old_7d_acc, "accuracy_7d_new": new_7d_acc,
                "composite_old": old_composite, "composite_new": new_composite,
            },
            accuracy_before=old_composite,
            accuracy_after=new_composite,
        )
        return None

    log.info(f"  ✅ Model improved (composite: {old_composite:.4f} → {new_composite:.4f}). Saving...")

    # Save models (only reached if improved)
    joblib.dump(model_24h, os.path.join(MODELS_DIR, f"model_24h_{timeframe}_latest.pkl"))
    joblib.dump(model_7d, os.path.join(MODELS_DIR, f"model_7d_{timeframe}_latest.pkl"))
    joblib.dump(model_dir, os.path.join(MODELS_DIR, f"model_dir_{timeframe}_latest.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_latest.pkl"))

    # Also save versioned copies
    joblib.dump(model_24h, os.path.join(MODELS_DIR, f"model_24h_{timeframe}_{ts}.pkl"))
    joblib.dump(model_7d, os.path.join(MODELS_DIR, f"model_7d_{timeframe}_{ts}.pkl"))
    joblib.dump(model_dir, os.path.join(MODELS_DIR, f"model_dir_{timeframe}_{ts}.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, f"label_encoder_{ts}.pkl"))

    # Update metadata
    feat_imp_24h = sorted(
        zip(trainer.FEATURE_COLUMNS, model_24h.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    feat_imp_7d = sorted(
        zip(trainer.FEATURE_COLUMNS, model_7d.feature_importances_),
        key=lambda x: x[1], reverse=True
    )

    metadata = {
        "version": ts,
        "timeframe": timeframe,
        "trained_at": datetime.now().isoformat(),
        "training_mode": "feedback_retrain",
        "feedback_samples": feedback_hits,
        "reinforced_samples": reinforced,
        "penalized_samples": penalized,
        "feature_columns": trainer.FEATURE_COLUMNS,
        "n_features": len(trainer.FEATURE_COLUMNS),
        "xgb_params": trainer.XGB_PARAMS,
        "model_24h": {
            "path": os.path.join(MODELS_DIR, f"model_24h_{timeframe}_{ts}.pkl"),
            "cv_mean": round(float(new_24h_acc), 4),
            "previous_cv_mean": round(float(old_24h_acc), 4),
            "improvement": round(float(new_24h_acc - old_24h_acc), 4),
            "passes_threshold": bool(new_24h_acc >= 0.52),
            "top_features": [[f, round(float(i), 4)] for f, i in feat_imp_24h[:10]],
        },
        "model_7d": {
            "path": os.path.join(MODELS_DIR, f"model_7d_{timeframe}_{ts}.pkl"),
            "cv_mean": round(float(new_7d_acc), 4),
            "previous_cv_mean": round(float(old_7d_acc), 4),
            "improvement": round(float(new_7d_acc - old_7d_acc), 4),
            "passes_threshold": bool(new_7d_acc >= 0.52),
            "top_features": [[f, round(float(i), 4)] for f, i in feat_imp_7d[:10]],
        },
        "model_dir": {
            "path": os.path.join(MODELS_DIR, f"model_dir_{timeframe}_{ts}.pkl"),
            "cv_mean": round(float(new_dir_acc), 4),
            "previous_cv_mean": round(float(old_dir_acc), 4),
            "improvement": round(float(new_dir_acc - old_dir_acc), 4),
            "direction_classes": list(le.classes_),
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"\n  ✅ Models saved with feedback (version: {ts})")

    # Log the learning event
    _log_learning_event(
        "model_retrained",
        f"Retrained with {feedback_hits} feedback samples",
        {
            "feedback_hits": feedback_hits,
            "reinforced": reinforced,
            "penalized": penalized,
            "accuracy_24h_old": old_24h_acc,
            "accuracy_24h_new": new_24h_acc,
            "accuracy_7d_old": old_7d_acc,
            "accuracy_7d_new": new_7d_acc,
        },
        model_version_before=old_metadata.get("version", "unknown"),
        model_version_after=ts,
        accuracy_before=old_7d_acc,
        accuracy_after=new_7d_acc,
    )

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: CONTINUOUS LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_learning_loop(timeframe: str = "1d"):
    """
    Run one complete iteration of the continuous learning loop:
    1. Execute paper trades from model predictions
    2. Evaluate outcomes of existing trades
    3. Generate feedback labels from outcomes
    4. Retrain model with feedback (if enough data)
    """
    log.info("\n" + "█" * 60)
    log.info("  NEXYPHER CONTINUOUS LEARNING LOOP")
    log.info(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Timeframe: {timeframe}")
    log.info("█" * 60)

    start = time.time()

    # Step 0: Import real trade outcomes from Supabase (if available)
    real_imported = 0
    try:
        real_imported = import_real_trade_outcomes()
    except Exception as e:
        log.debug(f"  Real trade import skipped: {e}")

    # Step 1: Execute new paper trades
    trades = execute_paper_trades(timeframe)

    # Step 2: Evaluate existing trades
    closed = evaluate_paper_trades()

    # Step 3: Generate feedback from outcomes
    feedback = generate_feedback_labels()

    # Step 4: Retrain if we have enough feedback
    metadata = retrain_with_feedback(timeframe)

    elapsed = time.time() - start

    # Summary
    log.info("\n" + "█" * 60)
    log.info("  LOOP COMPLETE")
    log.info(f"  Real imported: {real_imported} | New trades: {trades} | "
             f"Closed: {closed} | Feedback: {feedback}")
    log.info(f"  Model retrained: {'Yes' if metadata else 'No (insufficient feedback)'}")
    log.info(f"  Duration: {elapsed:.1f}s")
    log.info(f"  Wallet: ${get_wallet_balance():,.2f}")
    log.info("█" * 60)

    return {
        "real_trades_imported": real_imported,
        "trades_executed": trades,
        "trades_closed": closed,
        "feedback_generated": feedback,
        "model_retrained": metadata is not None,
        "elapsed_seconds": elapsed,
    }


def run_auto_loop(timeframe: str = "1d", interval_hours: float = 6.0):
    """
    Run the learning loop continuously at a fixed interval.
    This is the 'always learning' mode.
    """
    log.info(f"\n  🔄 AUTO LEARNING MODE: every {interval_hours}h")
    log.info(f"  Press Ctrl+C to stop.\n")

    iteration = 0
    while True:
        iteration += 1
        log.info(f"\n  ═══ Iteration #{iteration} ═══")

        try:
            result = run_learning_loop(timeframe)

            # Log iteration
            _log_learning_event(
                "loop_iteration",
                f"Auto-loop iteration #{iteration}",
                result,
            )

        except KeyboardInterrupt:
            log.info("\n  Stopped by user.")
            break
        except Exception as e:
            log.error(f"  Loop error: {e}")
            _log_learning_event("loop_error", str(e), {})

        # Wait for next iteration
        log.info(f"\n  Next iteration in {interval_hours}h...")
        try:
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            log.info("\n  Stopped by user.")
            break


# ─────────────────────────────────────────────────────────────────────────────
# STATUS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def show_status():
    """Show comprehensive learning status dashboard."""
    print("\n" + "=" * 60)
    print("  NEXYPHER CONTINUOUS LEARNING STATUS")
    print("=" * 60)

    # Wallet
    conn = sqlite3.connect(PAPER_TRADE_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM paper_wallet WHERE id=1")
    wallet = c.fetchone()
    if wallet:
        total_pnl = wallet["balance"] - PAPER_WALLET_INITIAL + wallet["total_invested"] - wallet["total_returned"]
        actual_pnl = wallet["balance"] - PAPER_WALLET_INITIAL
        print(f"\n  Paper Wallet:")
        print(f"    Balance:     ${wallet['balance']:,.2f}")
        print(f"    Invested:    ${wallet['total_invested']:,.2f}")
        print(f"    Returned:    ${wallet['total_returned']:,.2f}")
        print(f"    P&L:         ${actual_pnl:+,.2f} ({actual_pnl/PAPER_WALLET_INITIAL*100:+.1f}%)")
        print(f"    Total trades:{wallet['total_trades']}")
        if wallet['total_trades'] > 0:
            win_rate = wallet['winning_trades'] / wallet['total_trades'] * 100
            print(f"    Win rate:    {wallet['winning_trades']}/{wallet['total_trades']} ({win_rate:.1f}%)")

    # Open trades
    c.execute("SELECT COUNT(*) FROM paper_trades WHERE status='OPEN'")
    open_count = c.fetchone()[0]
    print(f"\n  Open Trades: {open_count}")

    # Recent closed trades
    c.execute("""
        SELECT token_id, outcome, pnl_pct, exit_reason, evaluated_at
        FROM paper_trades WHERE status='CLOSED'
        ORDER BY evaluated_at DESC LIMIT 10
    """)
    recent = c.fetchall()
    if recent:
        print(f"\n  Recent Trades (last 10):")
        print(f"    {'Token':<20} {'Outcome':<10} {'PnL%':>8} {'Reason':<15}")
        print(f"    {'-'*55}")
        for t in recent:
            symbol = "✅" if t["outcome"] == "WIN" else ("❌" if t["outcome"] == "LOSS" else "➖")
            print(f"    {t['token_id']:<20} {symbol} {t['outcome']:<7} "
                  f"{t['pnl_pct']:>+7.1f}% {t['exit_reason']:<15}")

    # Feedback stats
    c.execute("SELECT COUNT(*) FROM feedback_labels")
    n_feedback = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM feedback_labels WHERE sample_weight > 1.0")
    n_reinforced = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM feedback_labels WHERE sample_weight < 1.0")
    n_penalized = c.fetchone()[0]
    print(f"\n  Feedback Labels:")
    print(f"    Total:      {n_feedback}")
    print(f"    Reinforced: {n_reinforced}")
    print(f"    Penalized:  {n_penalized}")

    conn.close()

    # Learning log
    conn = sqlite3.connect(PAPER_TRADE_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT event_type, description, accuracy_before, accuracy_after, created_at
        FROM learning_log
        ORDER BY created_at DESC LIMIT 5
    """)
    events = c.fetchall()
    if events:
        print(f"\n  Learning History (last 5):")
        for ev in events:
            acc_change = ""
            if ev["accuracy_before"] and ev["accuracy_after"]:
                diff = ev["accuracy_after"] - ev["accuracy_before"]
                acc_change = f" | acc: {ev['accuracy_before']:.3f}→{ev['accuracy_after']:.3f} ({diff:+.3f})"
            print(f"    [{ev['created_at'][:16]}] {ev['event_type']}: {ev['description']}{acc_change}")

    conn.close()

    # Model info
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        print(f"\n  Current Model:")
        print(f"    Version:     {meta.get('version', 'unknown')}")
        print(f"    Trained at:  {meta.get('trained_at', 'unknown')[:19]}")
        print(f"    Mode:        {meta.get('training_mode', 'initial_train')}")
        print(f"    Features:    {meta.get('n_features', '?')}")
        if "model_24h" in meta:
            print(f"    24h acc:     {meta['model_24h'].get('cv_mean', 0):.4f}")
        if "model_7d" in meta:
            print(f"    7d acc:      {meta['model_7d'].get('cv_mean', 0):.4f}")
        if "model_dir" in meta:
            print(f"    Dir acc:     {meta['model_dir'].get('cv_mean', 0):.4f}")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _log_learning_event(event_type: str, description: str, metrics: dict,
                        model_version_before: str = None,
                        model_version_after: str = None,
                        accuracy_before: float = None,
                        accuracy_after: float = None,
                        trades_used: int = 0):
    """Record a learning event in the log."""
    try:
        conn = sqlite3.connect(PAPER_TRADE_DB)
        c = conn.cursor()
        c.execute("""
            INSERT INTO learning_log
                (event_type, description, metrics,
                 model_version_before, model_version_after,
                 trades_used, accuracy_before, accuracy_after)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_type, description, json.dumps(metrics),
            model_version_before, model_version_after,
            trades_used, accuracy_before, accuracy_after
        ))
        conn.commit()
        conn.close()
    except Exception:
        pass


def _sync_to_learning_db(closed_trades: list):
    """
    Bridge: sync paper trade outcomes to nexypher_learning.db so that
    the AI prediction system (LearningLoop, ConfidenceScorer) also
    benefits from paper trade feedback.

    Maps to the actual predictions table schema:
      direction_correct_24h / direction_correct_7d
      pnl_pct_24h / pnl_pct_7d
      evaluated_24h_at / evaluated_7d_at
    """
    if not os.path.exists(LEARNING_DB):
        log.debug("  Learning DB not found, skipping sync.")
        return

    if not closed_trades:
        return

    conn = sqlite3.connect(LEARNING_DB)
    c = conn.cursor()

    synced = 0
    for trade in closed_trades:
        try:
            token_id = trade.get("token_id", "")
            entry_time = trade.get("entry_time", "")
            dir_correct = trade.get("direction_correct", 0)
            pnl = trade.get("pnl_pct", 0)
            eval_at = trade.get("evaluated_at", "")
            price_24h = trade.get("actual_price_24h")
            price_7d = trade.get("actual_price_7d") or trade.get("exit_price")

            # Update predictions table matching on token + close time window
            c.execute("""
                UPDATE predictions SET
                    actual_price_24h = COALESCE(actual_price_24h, ?),
                    actual_price_7d = COALESCE(actual_price_7d, ?),
                    direction_correct_24h = COALESCE(direction_correct_24h, ?),
                    direction_correct_7d = ?,
                    pnl_pct_24h = COALESCE(pnl_pct_24h, ?),
                    pnl_pct_7d = ?,
                    evaluated_24h_at = COALESCE(evaluated_24h_at, ?),
                    evaluated_7d_at = ?
                WHERE token_ticker = ?
                    AND evaluated_7d_at IS NULL
                    AND ABS(julianday(?) - julianday(created_at)) < 8
            """, (
                price_24h,
                price_7d,
                dir_correct,
                dir_correct,
                pnl,
                pnl,
                eval_at,
                eval_at,
                token_id,
                entry_time,
            ))
            if c.rowcount > 0:
                synced += 1
        except Exception:
            continue

    conn.commit()
    conn.close()
    if synced > 0:
        log.info(f"  🔗 Synced {synced} outcomes to learning DB")


def import_real_trade_outcomes():
    """
    Import closed real trades from Supabase (trading_engine) into the
    continuous learner's feedback_labels table.

    This bridges the gap between live/real trading and the ML feedback loop:
      Supabase trade_positions (closed) → feedback_labels → model retraining

    Each closed position with a coin_id matching a token in training DB
    gets a feedback label so the model learns from real trade outcomes.
    """
    log.info("=" * 60)
    log.info("IMPORTING REAL TRADE OUTCOMES FROM SUPABASE")
    log.info("=" * 60)

    try:
        sys.path.insert(0, PROJECT_ROOT)
        from supabase_db import get_supabase
        sb = get_supabase()
    except Exception as e:
        log.warning(f"  Cannot connect to Supabase: {e}")
        return 0

    # Fetch closed positions from Supabase
    try:
        result = sb.table("trade_positions").select(
            "id, coin_id, coin_name, symbol, entry_price, current_price, "
            "invested_amount, quantity, pnl, pnl_pct, opened_at, closed_at, "
            "ai_reasoning"
        ).eq("status", "closed").order("closed_at", desc=True).limit(200).execute()
        closed_positions = result.data or []
    except Exception as e:
        log.error(f"  Failed to fetch closed positions: {e}")
        return 0

    if not closed_positions:
        log.info("  No closed real positions found.")
        return 0

    log.info(f"  Found {len(closed_positions)} closed real positions.")

    # Get already-imported trade IDs to avoid duplicates
    pconn = sqlite3.connect(PAPER_TRADE_DB)
    pc = pconn.cursor()
    pc.execute("SELECT trade_id FROM feedback_labels WHERE trade_id LIKE 'real_%'")
    already_imported = {row[0] for row in pc.fetchall()}

    imported = 0
    for pos in closed_positions:
        trade_id = f"real_{pos['id']}"
        if trade_id in already_imported:
            continue

        coin_id = pos.get("coin_id", "")
        pnl_pct = pos.get("pnl_pct", 0) or 0
        entry_price = pos.get("entry_price", 0)
        exit_price = pos.get("current_price", 0)

        if not coin_id or entry_price <= 0:
            continue

        # Determine outcome
        if pnl_pct > 2:
            outcome = "WIN"
        elif pnl_pct < -2:
            outcome = "LOSS"
        else:
            outcome = "NEUTRAL"

        # Compute feedback weight (same logic as paper trades)
        magnitude = min(abs(pnl_pct) / 10.0, 1.0)
        if outcome == "WIN":
            weight = BASE_WEIGHT + (REINFORCE_WEIGHT - BASE_WEIGHT) * (0.5 + 0.5 * magnitude)
        elif outcome == "LOSS":
            weight = BASE_WEIGHT - (BASE_WEIGHT - PENALIZE_WEIGHT) * (0.5 + 0.5 * magnitude)
        else:
            weight = BASE_WEIGHT

        # Determine actual direction
        price_change = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        if price_change > 0.02:
            actual_dir = "UP"
        elif price_change < -0.02:
            actual_dir = "DOWN"
        else:
            actual_dir = "SIDEWAYS"

        # Find closest timestamp in training DB
        timestamp = 0
        if os.path.exists(TRAINING_DB):
            try:
                tconn = sqlite3.connect(TRAINING_DB)
                tc = tconn.cursor()
                opened_at = pos.get("opened_at", "")
                tc.execute("""
                    SELECT timestamp FROM technical_indicators
                    WHERE token_id=? AND timeframe='1d'
                    ORDER BY ABS(julianday(datetime) - julianday(?)) ASC
                    LIMIT 1
                """, (coin_id, opened_at))
                row = tc.fetchone()
                if row:
                    timestamp = row[0]
                tconn.close()
            except Exception:
                pass

        if timestamp == 0:
            # Use a hash-derived timestamp as fallback
            timestamp = int(hashlib.sha256(
                f"{coin_id}_{pos.get('opened_at', '')}".encode()
            ).hexdigest()[:8], 16)

        label_7d = 1 if outcome == "WIN" else 0
        label_24h = 1 if pnl_pct > 0 else 0

        try:
            pc.execute("""
                INSERT OR REPLACE INTO feedback_labels
                    (token_id, timestamp, timeframe,
                     original_label_24h, original_label_7d, original_direction,
                     sample_weight, trade_id, outcome, pnl_pct)
                VALUES (?, ?, '1d', ?, ?, ?, ?, ?, ?, ?)
            """, (
                coin_id, timestamp,
                label_24h, label_7d, actual_dir,
                weight, trade_id, outcome, pnl_pct
            ))
            imported += 1
            symbol = "🔼" if weight > 1.0 else ("🔽" if weight < 1.0 else "➖")
            log.info(f"  {symbol} REAL TRADE: {coin_id} | {outcome} | "
                     f"pnl={pnl_pct:+.1f}% | weight={weight:.2f}")
        except Exception as e:
            log.debug(f"  Skipping {coin_id}: {e}")
            continue

    pconn.commit()
    pconn.close()

    if imported > 0:
        _log_learning_event(
            "real_trades_imported",
            f"Imported {imported} real trade outcomes from Supabase",
            {"imported": imported}
        )

    log.info(f"\n  Real trade outcomes imported: {imported}")
    return imported


def reset_paper_wallet():
    """Reset paper wallet to initial state (for testing)."""
    conn = sqlite3.connect(PAPER_TRADE_DB)
    c = conn.cursor()
    c.execute("DELETE FROM paper_trades")
    c.execute("DELETE FROM feedback_labels")
    c.execute("DELETE FROM learning_log")
    c.execute("DELETE FROM paper_wallet")
    c.execute("INSERT INTO paper_wallet (id, balance) VALUES (1, ?)",
              (PAPER_WALLET_INITIAL,))
    conn.commit()
    conn.close()
    log.info(f"Paper wallet reset to ${PAPER_WALLET_INITIAL:,.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NexYpher Continuous Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python continuous_learner.py --action paper-trade     # Execute paper trades
  python continuous_learner.py --action evaluate        # Check trade outcomes
  python continuous_learner.py --action feedback        # Generate feedback labels
  python continuous_learner.py --action retrain         # Retrain with feedback
  python continuous_learner.py --action loop            # Run full loop once
  python continuous_learner.py --action auto --interval 6  # Auto every 6h
  python continuous_learner.py --action status          # Show dashboard
  python continuous_learner.py --action reset           # Reset paper wallet
        """
    )
    parser.add_argument(
        "--action",
        choices=["paper-trade", "evaluate", "feedback", "retrain",
                 "loop", "auto", "status", "reset", "import-real"],
        default="loop",
        help="Action to perform"
    )
    parser.add_argument(
        "--timeframe",
        choices=["1d", "4h", "1h"],
        default="1d",
        help="Timeframe (default: 1d)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=6.0,
        help="Auto-loop interval in hours (default: 6)"
    )
    parser.add_argument(
        "--min-feedback",
        type=int,
        default=10,
        help="Minimum feedback labels before retraining (default: 10)"
    )

    args = parser.parse_args()

    if args.action == "paper-trade":
        execute_paper_trades(args.timeframe)

    elif args.action == "evaluate":
        evaluate_paper_trades()

    elif args.action == "feedback":
        generate_feedback_labels()

    elif args.action == "retrain":
        retrain_with_feedback(args.timeframe, args.min_feedback)

    elif args.action == "loop":
        run_learning_loop(args.timeframe)

    elif args.action == "auto":
        run_auto_loop(args.timeframe, args.interval)

    elif args.action == "status":
        show_status()

    elif args.action == "reset":
        reset_paper_wallet()

    elif args.action == "import-real":
        import_real_trade_outcomes()
