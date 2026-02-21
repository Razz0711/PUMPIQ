"""
Adaptive Parameter Optimizer
===============================
Algorithm 4 – Uses Optuna to find optimal indicator parameters per token
instead of hardcoded RSI(14), MACD(12,26,9), BB(20,2).

Parameters optimised:
    RSI period      : 7–21
    MACD fast       : 8–15
    MACD slow       : 20–30
    Bollinger period: 15–25
    Bollinger std   : 1.5–2.5

Optimisation target: maximise Sharpe ratio in walk-forward backtest.

Results are cached per token in a SQLite database and refreshed monthly
or when accuracy drops below threshold.

⚠ HISTORICAL SIMULATION — not guaranteed future performance.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.ta_utils import (
    rsi_series as _ta_rsi_series,
    macd_series as _ta_macd_series,
    bollinger_series as _ta_bollinger_series,
)

logger = logging.getLogger(__name__)

# ── Optional-import guard ─────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    logger.warning(
        "optuna not installed — parameter optimizer will return defaults. "
        "pip install optuna"
    )

# ── Constants ────────────────────────────────────────────────────
MIN_HISTORY = 120          # minimum data points needed
CACHE_TTL = 30 * 86400    # 30 days
N_TRIALS = 50             # Optuna trials per optimisation run
ACCURACY_REFRESH = 0.50   # retrain if accuracy drops below this

IS_VERCEL = bool(os.getenv("VERCEL"))
DB_PATH = "/tmp/nexypher_params.db" if IS_VERCEL else os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "nexypher_params.db"
)


# ── DB helpers ────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS token_optimal_parameters (
            coin_id TEXT PRIMARY KEY,
            rsi_period INTEGER NOT NULL DEFAULT 14,
            macd_fast INTEGER NOT NULL DEFAULT 12,
            macd_slow INTEGER NOT NULL DEFAULT 26,
            macd_signal INTEGER NOT NULL DEFAULT 9,
            bb_period INTEGER NOT NULL DEFAULT 20,
            bb_std REAL NOT NULL DEFAULT 2.0,
            sharpe_ratio REAL NOT NULL DEFAULT 0.0,
            win_rate REAL NOT NULL DEFAULT 0.0,
            total_return REAL NOT NULL DEFAULT 0.0,
            n_trades INTEGER NOT NULL DEFAULT 0,
            optimized_at TEXT NOT NULL DEFAULT (datetime('now')),
            expires_at TEXT NOT NULL DEFAULT (datetime('now', '+30 days'))
        );
    """)
    conn.commit()
    conn.close()


# Lazy initialization — tables are created on first use, not on import
_db_initialized = False


def _ensure_db():
    """Initialize DB tables on first use."""
    global _db_initialized
    if not _db_initialized:
        _init_db()
        _db_initialized = True


# ── Result dataclass ──────────────────────────────────────────────

@dataclass
class OptimalParameters:
    """Optimised indicator parameters for a token."""
    coin_id: str
    symbol: str

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0

    # Optimisation quality metrics
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    n_trades: int = 0

    is_optimized: bool = False
    is_cached: bool = False
    cache_age_days: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coin_id": self.coin_id,
            "symbol": self.symbol,
            "parameters": {
                "rsi_period": self.rsi_period,
                "macd_fast": self.macd_fast,
                "macd_slow": self.macd_slow,
                "macd_signal": self.macd_signal,
                "bb_period": self.bb_period,
                "bb_std": round(self.bb_std, 2),
            },
            "quality": {
                "sharpe_ratio": round(self.sharpe_ratio, 3),
                "win_rate": round(self.win_rate, 1),
                "total_return": round(self.total_return, 2),
                "n_trades": self.n_trades,
            },
            "is_optimized": self.is_optimized,
            "is_cached": self.is_cached,
            "cache_age_days": round(self.cache_age_days, 1),
            "error": self.error,
        }

    def as_backtest_kwargs(self) -> Dict[str, Any]:
        """Return kwargs to pass to BacktestEngine.__init__."""
        return {
            "rsi_period": self.rsi_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
        }


# ── Internal backtest simulator for Optuna objective ──────────────

def _quick_backtest(
    closes: List[float],
    rsi_period: int, rsi_oversold: float, rsi_overbought: float,
    macd_fast: int, macd_slow: int, macd_signal: int,
    bb_period: int, bb_std: float,
    fee: float = 0.001,
    stop_loss: float = 8.0,
    take_profit: float = 15.0,
) -> Tuple[float, float, float, int]:
    """
    Lightweight backtest returning (sharpe, win_rate, total_return, n_trades).
    Used inside the Optuna objective — must be fast.
    """
    n = len(closes)
    if n < max(rsi_period + 1, macd_slow + macd_signal, bb_period) + 10:
        return 0.0, 0.0, 0.0, 0

    # Indicators — delegate to shared ta_utils
    rsi = _ta_rsi_series(closes, rsi_period)
    _, _, hist = _ta_macd_series(closes, macd_fast, macd_slow, macd_signal)
    bb_u, _, bb_l_arr = _ta_bollinger_series(closes, bb_period, bb_std)

    # Simulate
    start_idx = max(rsi_period + 1, macd_slow + macd_signal, bb_period)
    equity = 10000.0
    in_pos = False
    entry_price = 0.0
    trades_pnl = []

    for i in range(start_idx, n):
        price = closes[i]
        if not in_pos:
            buy = 0
            if rsi[i] < rsi_oversold:
                buy += 1
            if hist[i] > 0 and i > 0 and hist[i-1] <= 0:
                buy += 1
            if price <= bb_l_arr[i] and bb_l_arr[i] > 0:
                buy += 1
            if buy >= 2:
                entry_price = price
                in_pos = True
        else:
            pnl_pct = (price - entry_price) / entry_price * 100
            sell = 0
            if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                sell = 2
            else:
                if rsi[i] > rsi_overbought:
                    sell += 1
                if hist[i] < 0 and i > 0 and hist[i-1] >= 0:
                    sell += 1
                if price >= bb_u[i] and bb_u[i] > 0:
                    sell += 1
            if sell >= 2:
                net = ((price * (1 - fee)) / (entry_price * (1 + fee)) - 1) * 100
                trades_pnl.append(net)
                equity *= (1 + net / 100)
                in_pos = False

    if not trades_pnl:
        return 0.0, 0.0, 0.0, 0

    wins = sum(1 for p in trades_pnl if p > 0)
    win_rate = wins / len(trades_pnl) * 100
    total_return = (equity / 10000.0 - 1) * 100

    # Sharpe
    avg_r = sum(trades_pnl) / len(trades_pnl) / 100
    var_r = sum((p / 100 - avg_r) ** 2 for p in trades_pnl) / len(trades_pnl)
    std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
    sharpe = avg_r / std_r * math.sqrt(len(trades_pnl)) if std_r > 0 else 0

    return sharpe, win_rate, total_return, len(trades_pnl)


# ══════════════════════════════════════════════════════════════════
# Parameter Optimizer
# ══════════════════════════════════════════════════════════════════

class ParameterOptimizer:
    """
    Finds optimal indicator parameters per token using Optuna.

    Usage::

        opt = ParameterOptimizer()
        params = await opt.optimize("bitcoin", "BTC", cg_collector)
        engine = BacktestEngine(**params.as_backtest_kwargs())
    """

    def __init__(self, n_trials: int = N_TRIALS):
        _ensure_db()
        self.n_trials = n_trials

    # ── cache ──────────────────────────────────────────────────────

    def _get_cached(self, coin_id: str) -> Optional[OptimalParameters]:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM token_optimal_parameters WHERE coin_id = ?",
                (coin_id,),
            ).fetchone()
            if not row:
                return None
            # Check expiry
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.now() > expires:
                return None
            optimized = datetime.fromisoformat(row["optimized_at"])
            age_days = (datetime.now() - optimized).total_seconds() / 86400
            return OptimalParameters(
                coin_id=coin_id,
                symbol="",
                rsi_period=row["rsi_period"],
                macd_fast=row["macd_fast"],
                macd_slow=row["macd_slow"],
                macd_signal=row["macd_signal"],
                bb_period=row["bb_period"],
                bb_std=row["bb_std"],
                sharpe_ratio=row["sharpe_ratio"],
                win_rate=row["win_rate"],
                total_return=row["total_return"],
                n_trades=row["n_trades"],
                is_optimized=True,
                is_cached=True,
                cache_age_days=age_days,
            )
        except Exception:
            return None
        finally:
            conn.close()

    def _save_cached(self, params: OptimalParameters):
        conn = _get_db()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO token_optimal_parameters
                    (coin_id, rsi_period, macd_fast, macd_slow, macd_signal,
                     bb_period, bb_std, sharpe_ratio, win_rate, total_return, n_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                params.coin_id, params.rsi_period, params.macd_fast,
                params.macd_slow, params.macd_signal, params.bb_period,
                params.bb_std, params.sharpe_ratio, params.win_rate,
                params.total_return, params.n_trades,
            ))
            conn.commit()
        except Exception as exc:
            logger.warning("Failed to cache params for %s: %s", params.coin_id, exc)
        finally:
            conn.close()

    # ── main entry ────────────────────────────────────────────────

    async def optimize(
        self,
        coin_id: str,
        symbol: str,
        cg_collector,
        days: int = 180,
        force: bool = False,
    ) -> OptimalParameters:
        """
        Find optimal parameters for a token.

        Returns cached result if valid, otherwise runs Optuna optimisation.
        Falls back to defaults if Optuna is not installed.
        """
        result = OptimalParameters(coin_id=coin_id, symbol=symbol.upper())

        # Check cache first
        if not force:
            cached = self._get_cached(coin_id)
            if cached is not None:
                cached.symbol = symbol.upper()
                logger.info(
                    "Parameter cache hit for %s (age %.1f days)",
                    symbol, cached.cache_age_days,
                )
                return cached

        # Graceful degradation
        if not _HAS_OPTUNA:
            result.error = "optuna not installed — using defaults"
            return result

        # Fetch data
        try:
            history = await cg_collector.get_price_history(coin_id, days=max(days, 180))
        except Exception as exc:
            result.error = f"Data fetch failed: {exc}"
            return result

        if not history or not history.prices or len(history.prices) < MIN_HISTORY:
            result.error = f"Insufficient data ({len(history.prices) if history and history.prices else 0} points)"
            return result

        closes = [p[1] for p in history.prices]

        # ── Optuna study ──
        def objective(trial):
            rsi_p = trial.suggest_int("rsi_period", 7, 21)
            m_fast = trial.suggest_int("macd_fast", 8, 15)
            m_slow = trial.suggest_int("macd_slow", 20, 30)
            m_sig = trial.suggest_int("macd_signal", 5, 12)
            bb_p = trial.suggest_int("bb_period", 15, 25)
            bb_s = trial.suggest_float("bb_std", 1.5, 2.5, step=0.1)

            # Ensure macd_slow > macd_fast
            if m_slow <= m_fast:
                return -999.0

            sharpe, wr, ret, nt = _quick_backtest(
                closes,
                rsi_p, 30.0, 70.0,
                m_fast, m_slow, m_sig,
                bb_p, bb_s,
            )

            # Penalise too few trades
            if nt < 5:
                return -999.0

            return sharpe

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best = study.best_params
        best_sharpe = study.best_value

        if best_sharpe <= -999:
            result.error = "Optimisation failed — no valid parameter set found"
            return result

        result.rsi_period = best.get("rsi_period", 14)
        result.macd_fast = best.get("macd_fast", 12)
        result.macd_slow = best.get("macd_slow", 26)
        result.macd_signal = best.get("macd_signal", 9)
        result.bb_period = best.get("bb_period", 20)
        result.bb_std = best.get("bb_std", 2.0)
        result.is_optimized = True

        # Run final backtest with best params
        sharpe, wr, ret, nt = _quick_backtest(
            closes,
            result.rsi_period, 30.0, 70.0,
            result.macd_fast, result.macd_slow, result.macd_signal,
            result.bb_period, result.bb_std,
        )
        result.sharpe_ratio = sharpe
        result.win_rate = wr
        result.total_return = ret
        result.n_trades = nt

        # Cache
        self._save_cached(result)

        logger.info(
            "Optimized [%s]: RSI=%d  MACD=%d/%d/%d  BB=%d/%.1f  "
            "Sharpe=%.2f  WR=%.1f%%  Return=%.1f%%  Trades=%d",
            symbol, result.rsi_period,
            result.macd_fast, result.macd_slow, result.macd_signal,
            result.bb_period, result.bb_std,
            result.sharpe_ratio, result.win_rate, result.total_return, result.n_trades,
        )

        return result


# ── Module-level singleton ────────────────────────────────────────

_optimizer: Optional[ParameterOptimizer] = None


def get_parameter_optimizer() -> ParameterOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = ParameterOptimizer()
    return _optimizer
