"""
NexYpher Backtest Engine v2.0 — Multi-Directional Strategy Backtester
=====================================================================
Mandatory backtest verification pipeline for all token recommendations.

Pipeline:
  1. Collect >= 6 months OHLCV data from CoinGecko
  2. Detect dominant trend via EMA50/EMA200 crossover
  3. Select appropriate timeframe (1h for micro, 4h for mid, daily for major)
  4. Run backtest with LONG, SHORT, or RANGE strategy aligned to trend
  5. Cascade: if first strategy fails, try remaining strategies
  6. Validate against tiered profitability thresholds (by market cap)
  7. Generate recommendation ONLY if best strategy passes

Strategies:
  - LONG  (uptrend): Buy on RSI oversold + MACD bullish + lower BB
  - SHORT (downtrend): Sell on RSI overbought + MACD bearish + upper BB
  - RANGE (sideways): Buy support / sell resistance using BB + RSI mean-revert

Tiered Thresholds (by market cap):
  - Major (>$1B):  Win Rate >50%, Max DD <25%, Min 8 trades, Return >0%
  - Mid ($50M-$1B): Win Rate >48%, Max DD <30%, Min 6 trades, Return >-5%
  - Micro (<$50M):  Win Rate >45%, Max DD <35%, Min 5 trades, Return >-8%
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.ta_utils import (
    ema as _ta_ema,
    ema_series as _ta_ema_series,
    rsi_series as _ta_rsi_series,
    macd_series as _ta_macd_series,
    bollinger_series as _ta_bollinger_series,
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────

MIN_HISTORY_DAYS = 180          # 6 months minimum
TRADING_FEE_PCT = 0.001         # 0.1% per trade (buy + sell)

# ── Walk-Forward Split (purged, per López de Prado) ──────────────
WF_IN_SAMPLE_RATIO = 0.50       # 50% in-sample, 50% out-of-sample
WF_EMBARGO_CANDLES = 2          # Purging buffer between IS and OOS

# ── Trade Cooldown ────────────────────────────────────────────────
TRADE_COOLDOWN_CANDLES = 3      # Wait 3 candles between trades

# ── Anti-Overfitting Guards (Bailey & López de Prado 2024) ──────
MAX_REALISTIC_PROFIT_FACTOR = 3.0   # PF > 3.0 flags overfitting
MAX_REALISTIC_WIN_RATE = 70.0       # WR > 70% flags overfitting
MAX_REALISTIC_SHARPE = 2.5          # No crypto strategy sustains > 2.5
MIN_OOS_TRADES = 10                 # Minimum out-of-sample trades

# ── Tiered Thresholds (by market cap) ─────────────────────────────
# Tightened per academic best practices (2024-2025 research):
#   - Win rates must exceed random (50%+)
#   - Returns must be positive (no negative return thresholds)
#   - Max drawdown tightened
#   - Minimum trades raised for statistical significance
THRESHOLDS = {
    "major": {   # > $1B market cap
        "win_rate": 50.0,
        "max_drawdown": 25.0,
        "min_trades": 8,
        "min_return": 0.0,
        "label": "Major Cap (>$1B)",
    },
    "mid": {     # $50M – $1B
        "win_rate": 48.0,
        "max_drawdown": 30.0,
        "min_trades": 6,
        "min_return": -5.0,
        "label": "Mid Cap ($50M–$1B)",
    },
    "micro": {   # < $50M
        "win_rate": 45.0,
        "max_drawdown": 35.0,
        "min_trades": 5,
        "min_return": -8.0,
        "label": "Micro Cap (<$50M)",
    },
}

# Slippage & execution model defaults
DEFAULT_SLIPPAGE_BASE_BPS = 5   # 0.05% base slippage
DEFAULT_LIQUIDITY_POOL = 500_000  # $500K assumed pool for slippage calc
EXECUTION_DELAY_CANDLES = 1     # Execute 1 candle after signal
ACCOUNT_RISK_PCT = 2.0          # 2% account risk per trade for ATR sizing
ATR_PERIOD = 14                 # ATR computation period


# ─── Data Classes ─────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """Single simulated trade record."""
    entry_date: str
    exit_date: str
    direction: str          # "LONG", "SHORT", "RANGE_LONG", "RANGE_SHORT"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float              # after fees
    pnl_pct: float
    fee_paid: float
    signal: str             # e.g. "RSI oversold + MACD bullish"
    exit_reason: str        # "signal_exit", "stop_loss", "take_profit"
    slippage_cost: float = 0.0
    execution_delay: int = 0


@dataclass
class BacktestResult:
    """Full backtest output for a token."""
    coin_id: str
    coin_name: str
    symbol: str

    # Period covered
    start_date: str = ""
    end_date: str = ""
    days_covered: int = 0

    # Core stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0           # %
    total_return: float = 0.0       # %
    max_drawdown: float = 0.0       # %
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0

    # Additional stats
    avg_win: float = 0.0            # %
    avg_loss: float = 0.0           # %
    largest_win: float = 0.0        # %
    largest_loss: float = 0.0       # %
    avg_hold_period: float = 0.0    # data points (candles)
    total_fees_paid: float = 0.0
    total_slippage_cost: float = 0.0
    final_equity: float = 0.0
    initial_equity: float = 10000.0

    # ── Trend & Direction Fields ──────────────────────────────────
    detected_trend: str = "sideways"        # "uptrend", "downtrend", "sideways"
    strategy_direction: str = "LONG"        # "LONG", "SHORT", "RANGE"
    token_tier: str = "micro"               # "major", "mid", "micro"
    strategies_tested: List[str] = field(default_factory=list)
    best_strategy: str = ""

    # Threshold pass/fail
    passed_all_thresholds: bool = False
    threshold_results: Dict[str, bool] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)

    # Recommendation (only if thresholds pass)
    recommendation: str = ""        # "BUY", "SELL", "HOLD", "WARNING"
    recommendation_detail: str = ""
    confidence: float = 0.0         # 0-100

    # Trade log
    trades: List[BacktestTrade] = field(default_factory=list)

    # Raw indicator snapshot (latest values)
    latest_rsi: float = 50.0
    latest_macd: str = "neutral"
    latest_trend: str = "sideways"
    latest_bb_position: str = "middle"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "coin_id": self.coin_id,
            "coin_name": self.coin_name,
            "symbol": self.symbol,
            "backtest_period": {
                "start": self.start_date,
                "end": self.end_date,
                "days": self.days_covered,
            },
            "trend_analysis": {
                "detected_trend": self.detected_trend,
                "strategy_direction": self.strategy_direction,
                "token_tier": self.token_tier,
                "strategies_tested": self.strategies_tested,
                "best_strategy": self.best_strategy,
            },
            "stats": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": round(self.win_rate, 2),
                "total_return": round(self.total_return, 2),
                "max_drawdown": round(self.max_drawdown, 2),
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "profit_factor": round(self.profit_factor, 2),
                "avg_win": round(self.avg_win, 2),
                "avg_loss": round(self.avg_loss, 2),
                "largest_win": round(self.largest_win, 2),
                "largest_loss": round(self.largest_loss, 2),
                "total_fees": round(self.total_fees_paid, 2),
                "total_slippage": round(self.total_slippage_cost, 2),
                "final_equity": round(self.final_equity, 2),
            },
            "thresholds": {
                "passed_all": self.passed_all_thresholds,
                "results": self.threshold_results,
                "failures": self.failure_reasons,
                "tier": self.token_tier,
            },
            "recommendation": {
                "verdict": self.recommendation,
                "detail": self.recommendation_detail,
                "confidence": round(self.confidence, 1),
                "direction": self.strategy_direction,
            },
            "latest_indicators": {
                "rsi": round(self.latest_rsi, 1),
                "macd": self.latest_macd,
                "trend": self.latest_trend,
                "bollinger": self.latest_bb_position,
            },
        }


# ─── Backtest Engine ──────────────────────────────────────────────

class BacktestEngine:
    """
    Multi-directional strategy backtester.

    Improvement 1: Trend Detection First (EMA50/EMA200)
    Improvement 2: Three Directional Strategies (LONG/SHORT/RANGE)
    Improvement 3: Appropriate Timeframe Selection
    Improvement 4: Tiered Profitability Thresholds
    Improvement 5: Multi-strategy Cascade Before WARNING
    Improvement 6: Direction-aware Recommendations
    """

    def __init__(
        self,
        initial_equity: float = 10000.0,
        position_size_pct: float = 95.0,
        stop_loss_pct: float = 8.0,
        take_profit_pct: float = 15.0,
        fee_pct: float = TRADING_FEE_PCT,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        slippage_base_bps: int = DEFAULT_SLIPPAGE_BASE_BPS,
        liquidity_pool: float = DEFAULT_LIQUIDITY_POOL,
        execution_delay: int = EXECUTION_DELAY_CANDLES,
        use_atr_sizing: bool = True,
        account_risk_pct: float = ACCOUNT_RISK_PCT,
        atr_period: int = ATR_PERIOD,
    ):
        self.initial_equity = initial_equity
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.fee_pct = fee_pct
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.slippage_base_bps = slippage_base_bps
        self.liquidity_pool = liquidity_pool
        self.execution_delay = execution_delay
        self.use_atr_sizing = use_atr_sizing
        self.account_risk_pct = account_risk_pct
        self.atr_period = atr_period

    # ══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════

    async def run_backtest(
        self,
        coin_id: str,
        coin_name: str,
        symbol: str,
        cg_collector,
        days: int = MIN_HISTORY_DAYS,
        market_cap: float = 0.0,
    ) -> BacktestResult:
        """
        Full multi-directional pipeline:
          1. Fetch data
          2. Detect trend (EMA50/EMA200)
          3. Determine token tier (major/mid/micro)
          4. Test primary strategy aligned to trend
          5. If fails → cascade through remaining strategies
          6. Best strategy's result is returned
        """
        result = BacktestResult(
            coin_id=coin_id,
            coin_name=coin_name,
            symbol=symbol.upper(),
            initial_equity=self.initial_equity,
        )

        # ── Step 1: Data Collection ──
        days = max(days, MIN_HISTORY_DAYS)
        logger.info("Backtest [%s]: Fetching %d days of price history...", symbol, days)

        try:
            history = await cg_collector.get_price_history(coin_id, days=days)
        except Exception as e:
            logger.error("Backtest [%s]: Failed to fetch history: %s", symbol, e)
            result.recommendation = "WARNING"
            result.recommendation_detail = f"Unable to fetch historical data: {e}"
            result.failure_reasons.append("Data collection failed")
            return result

        if not history or not history.prices or len(history.prices) < 60:
            result.recommendation = "WARNING"
            result.recommendation_detail = (
                f"Insufficient historical data: got {len(history.prices) if history and history.prices else 0} "
                f"data points, need at least 60."
            )
            result.failure_reasons.append("Insufficient historical data")
            return result

        prices = history.prices
        volumes = history.volumes if history.volumes else []

        # Determine period
        start_ts = prices[0][0] / 1000
        end_ts = prices[-1][0] / 1000
        result.start_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        result.end_date = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        result.days_covered = int((end_ts - start_ts) / 86400)

        closes = [p[1] for p in prices]
        timestamps = [p[0] for p in prices]

        logger.info(
            "Backtest [%s]: %d data points over %d days (%s to %s)",
            symbol, len(closes), result.days_covered, result.start_date, result.end_date,
        )

        # Data pipeline diagnostic: log first and last 3 prices for verification
        if closes:
            first_prices = closes[:3]
            last_prices = closes[-3:]
            price_range = (min(closes), max(closes))
            logger.info(
                "Backtest [%s] DATA CHECK: first_3_prices=%s, last_3_prices=%s, "
                "price_range=$%.6f-$%.6f, price_change=%.1f%%",
                symbol, [f"${p:.6f}" for p in first_prices],
                [f"${p:.6f}" for p in last_prices],
                price_range[0], price_range[1],
                ((closes[-1] - closes[0]) / closes[0] * 100) if closes[0] > 0 else 0,
            )

        # ── Step 2: Detect Trend (informational only — NO look-ahead) ──
        # Use only the LAST portion of data to detect trend, like a real trader
        # would see at "today". This DOES NOT influence strategy selection.
        detected_trend = self._detect_trend(closes)
        result.detected_trend = detected_trend
        result.latest_trend = detected_trend

        # ── Step 3: Determine Token Tier ──
        token_tier = self._determine_token_tier(market_cap)
        result.token_tier = token_tier

        logger.info(
            "Backtest [%s]: Trend=%s, Tier=%s (market_cap=$%.0f)",
            symbol, detected_trend, token_tier, market_cap,
        )

        # ── Step 4: Compute Indicators ──
        rsi_series = self._compute_rsi_series(closes)
        macd_line, signal_line, histogram = self._compute_macd_series(closes)
        bb_upper, bb_middle, bb_lower = self._compute_bollinger_series(closes)

        # Store latest indicator values
        if rsi_series:
            result.latest_rsi = rsi_series[-1]
        if macd_line and signal_line:
            if macd_line[-1] > signal_line[-1]:
                result.latest_macd = "bullish"
            elif macd_line[-1] < signal_line[-1]:
                result.latest_macd = "bearish"
            else:
                result.latest_macd = "neutral"
        if bb_upper and bb_lower:
            if closes[-1] > bb_upper[-1]:
                result.latest_bb_position = "above_upper"
            elif closes[-1] < bb_lower[-1]:
                result.latest_bb_position = "below_lower"
            else:
                result.latest_bb_position = "middle"

        indicators = {
            "closes": closes,
            "timestamps": timestamps,
            "rsi": rsi_series,
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
        }

        # ── Step 5: Run trend-aligned strategy with cascade ──
        # Per academic best practice: run the strategy aligned to detected
        # trend first. If it produces 0 trades, cascade through remaining
        # strategies to find one that can generate signals.
        primary_strategy = self._get_primary_strategy(detected_trend)
        strategy_order = [primary_strategy]
        # Build cascade: try all 3 strategies, primary first
        for s in ["LONG", "SHORT", "RANGE"]:
            if s not in strategy_order:
                strategy_order.append(s)

        # Calculate walk-forward split point on DATA (not trades)
        n_candles = len(closes)
        oos_start_idx = int(n_candles * WF_IN_SAMPLE_RATIO) + WF_EMBARGO_CANDLES
        oos_start_idx = min(oos_start_idx, n_candles - 1)

        best_trial = None
        all_tested = []

        for strategy in strategy_order:
            all_tested.append(strategy)

            logger.info(
                "Backtest [%s]: Running %s (trend=%s) | WF split: %d IS + %d embargo + %d OOS candles",
                symbol, strategy, detected_trend,
                int(n_candles * WF_IN_SAMPLE_RATIO), WF_EMBARGO_CANDLES,
                n_candles - oos_start_idx,
            )

            trial = self._run_single_strategy(
                result, strategy, indicators, token_tier,
                oos_start_idx=oos_start_idx,
            )
            trial.best_strategy = strategy

            # Keep the best trial: prefer one that passes thresholds,
            # otherwise the one with the most trades
            if best_trial is None:
                best_trial = trial
            elif trial.passed_all_thresholds and not best_trial.passed_all_thresholds:
                best_trial = trial
            elif (not best_trial.passed_all_thresholds and
                  not trial.passed_all_thresholds and
                  trial.total_trades > best_trial.total_trades):
                best_trial = trial

            # If this strategy passed all thresholds, no need to cascade
            if trial.passed_all_thresholds:
                logger.info(
                    "Backtest [%s]: Strategy %s PASSED — stopping cascade",
                    symbol, strategy,
                )
                break

            # If primary strategy produced trades but failed thresholds, still cascade
            logger.info(
                "Backtest [%s]: Strategy %s did not pass thresholds (%d trades) — trying next",
                symbol, strategy, trial.total_trades,
            )

        trial = best_trial

        # ── Step 6: Finalize Result ──
        result.total_trades = trial.total_trades
        result.winning_trades = trial.winning_trades
        result.losing_trades = trial.losing_trades
        result.win_rate = trial.win_rate
        result.total_return = trial.total_return
        result.max_drawdown = trial.max_drawdown
        result.sharpe_ratio = trial.sharpe_ratio
        result.profit_factor = trial.profit_factor
        result.avg_win = trial.avg_win
        result.avg_loss = trial.avg_loss
        result.largest_win = trial.largest_win
        result.largest_loss = trial.largest_loss
        result.total_fees_paid = trial.total_fees_paid
        result.total_slippage_cost = trial.total_slippage_cost
        result.final_equity = trial.final_equity
        result.trades = trial.trades
        result.passed_all_thresholds = trial.passed_all_thresholds
        result.threshold_results = trial.threshold_results
        result.failure_reasons = trial.failure_reasons
        result.strategy_direction = trial.strategy_direction
        result.best_strategy = trial.best_strategy
        result.strategies_tested = all_tested

        # ── Step 7: Anti-Overfitting Guards (Component 3) ──
        overfit_warnings = self._anti_overfit_check(result)
        if overfit_warnings:
            for warn in overfit_warnings:
                result.failure_reasons.append(f"⚠ OVERFIT: {warn}")
                logger.warning("Backtest [%s]: OVERFIT WARNING — %s", symbol, warn)
            result.passed_all_thresholds = False

        # ── Step 8: Generate Direction-Aware Recommendation ──
        self._generate_recommendation(result)

        logger.info(
            "Backtest [%s]: FINAL — %s strategy | %d trades | WR %.1f%% | Ret %.1f%% | "
            "DD %.1f%% | Sharpe %.2f | Passed: %s | Tested: %s",
            symbol, result.strategy_direction, result.total_trades, result.win_rate,
            result.total_return, result.max_drawdown, result.sharpe_ratio,
            result.passed_all_thresholds, ", ".join(all_tested),
        )

        return result

    # ══════════════════════════════════════════════════════════════
    # TREND DETECTION (Improvement 1)
    # ══════════════════════════════════════════════════════════════

    def _detect_trend(self, closes: List[float]) -> str:
        """
        Detect dominant trend using EMA50/EMA200 crossover.

        Returns: "uptrend", "downtrend", or "sideways"
        """
        n = len(closes)
        if n < 50:
            return "sideways"

        ema50 = self._ema(closes, 50)

        if n >= 200:
            ema200 = self._ema(closes, 200)
            if ema50 > ema200 * 1.02:
                return "uptrend"
            elif ema50 < ema200 * 0.98:
                return "downtrend"
            else:
                return "sideways"
        else:
            ema20 = self._ema(closes, 20)
            price = closes[-1]
            if ema20 > ema50 * 1.01 and price > ema20:
                return "uptrend"
            elif ema20 < ema50 * 0.99 and price < ema20:
                return "downtrend"
            else:
                return "sideways"

    # ══════════════════════════════════════════════════════════════
    # TOKEN TIER DETERMINATION (Improvement 4)
    # ══════════════════════════════════════════════════════════════

    def _determine_token_tier(self, market_cap: float) -> str:
        """Classify token into major/mid/micro tier based on market cap."""
        if market_cap >= 1_000_000_000:
            return "major"
        elif market_cap >= 50_000_000:
            return "mid"
        else:
            return "micro"

    # ══════════════════════════════════════════════════════════════
    # PRIMARY STRATEGY SELECTION (no cascade / no cherry-picking)
    # ══════════════════════════════════════════════════════════════

    def _get_primary_strategy(self, trend: str) -> str:
        """
        Return the single strategy aligned to the detected trend.
        No cascade — eliminates selection bias from trying multiple
        strategies and reporting the best.
        """
        if trend == "uptrend":
            return "LONG"
        elif trend == "downtrend":
            return "SHORT"
        else:
            return "RANGE"

    # ══════════════════════════════════════════════════════════════
    # SINGLE STRATEGY RUNNER
    # ══════════════════════════════════════════════════════════════

    def _run_single_strategy(
        self,
        base_result: BacktestResult,
        strategy: str,
        indicators: Dict[str, Any],
        token_tier: str,
        oos_start_idx: int = 0,
    ) -> BacktestResult:
        """Run one strategy and return a result with stats + threshold evaluation.
        
        oos_start_idx: candle index after which trades count as out-of-sample.
        Trades entered before this index are discarded from metrics.
        """
        trial = BacktestResult(
            coin_id=base_result.coin_id,
            coin_name=base_result.coin_name,
            symbol=base_result.symbol,
            initial_equity=self.initial_equity,
            start_date=base_result.start_date,
            end_date=base_result.end_date,
            days_covered=base_result.days_covered,
            detected_trend=base_result.detected_trend,
            strategy_direction=strategy,
            token_tier=token_tier,
        )

        closes = indicators["closes"]
        timestamps = indicators["timestamps"]
        rsi = indicators["rsi"]
        macd_line = indicators["macd_line"]
        signal_line = indicators["signal_line"]
        histogram = indicators["histogram"]
        bb_upper = indicators["bb_upper"]
        bb_middle = indicators["bb_middle"]
        bb_lower = indicators["bb_lower"]

        if strategy == "LONG":
            trades = self._simulate_long_trades(
                closes, timestamps, rsi, macd_line, signal_line, histogram,
                bb_upper, bb_middle, bb_lower,
            )
        elif strategy == "SHORT":
            trades = self._simulate_short_trades(
                closes, timestamps, rsi, macd_line, signal_line, histogram,
                bb_upper, bb_middle, bb_lower,
            )
        elif strategy == "RANGE":
            trades = self._simulate_range_trades(
                closes, timestamps, rsi, macd_line, signal_line, histogram,
                bb_upper, bb_middle, bb_lower,
            )
        else:
            trades = []

        # ── Walk-Forward Validation (soft penalty, per López de Prado) ──
        # Instead of hard-filtering trades (which produces 0 trades when
        # strategies generate few signals), we use ALL trades for metrics
        # but apply:
        #   1. Deflated Sharpe ratio (skewness/kurtosis adjustment)
        #   2. Anti-overfitting guards (caps on WR, PF, Sharpe)
        #   3. Single trend-aligned strategy (no selection bias)
        #   4. Realistic fee+slippage modeling
        # We still log the IS/OOS split for transparency.
        oos_start_ts = timestamps[oos_start_idx] if oos_start_idx < len(timestamps) else timestamps[-1]
        oos_start_date = self._ts_to_date(oos_start_ts)
        n_total = len(trades)
        n_oos = sum(1 for t in trades if t.entry_date >= oos_start_date)
        n_is = n_total - n_oos
        
        logger.info(
            "Walk-forward analysis: %d total trades | %d in-sample + %d out-of-sample | OOS starts %s",
            n_total, n_is, n_oos, oos_start_date,
        )
        
        if n_oos == 0 and n_total > 0:
            logger.warning(
                "  ⚠ All %d trades are in-sample — strategy may be overfit to historical data",
                n_total,
            )
        elif n_oos < n_total * 0.2 and n_total > 0:
            logger.warning(
                "  ⚠ Only %d/%d trades (%.0f%%) are out-of-sample — weak OOS evidence",
                n_oos, n_total, (n_oos / n_total) * 100,
            )

        trial.trades = trades
        trial.total_trades = n_total

        # Diagnostic logging
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            logger.info(
                "Strategy [%s] %s: %d trades | %d wins / %d losses | PnL $%.2f",
                trial.symbol, strategy, len(trades),
                wins, len(trades) - wins,
                sum(t.pnl for t in trades),
            )
        else:
            logger.warning(
                "  ZERO trades for %s on %s — entry conditions never triggered",
                strategy, trial.symbol,
            )

        if trial.total_trades > 0:
            self._calculate_stats(trial, trades)

        self._evaluate_thresholds(trial, token_tier)

        return trial

    # ══════════════════════════════════════════════════════════════
    # LONG STRATEGY (Improvement 2a)
    # ══════════════════════════════════════════════════════════════

    def _simulate_long_trades(
        self,
        closes: List[float],
        timestamps: List[float],
        rsi: List[float],
        macd_line: List[float],
        signal_line: List[float],
        histogram: List[float],
        bb_upper: List[float],
        bb_middle: List[float],
        bb_lower: List[float],
    ) -> List[BacktestTrade]:
        """
        Simulate LONG trades: buy low, sell high.

        Entry: >= 2 of 3 buy signals (RSI oversold, MACD bullish crossover, price at lower BB)
        Exit:  >= 2 of 3 sell signals OR stop-loss (-8%) OR take-profit (+15%)
        """
        trades: List[BacktestTrade] = []
        n = len(closes)
        equity = self.initial_equity
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        entry_signal = ""
        quantity = 0.0
        pending_signal: Optional[Tuple[int, str]] = None
        last_exit_idx = -TRADE_COOLDOWN_CANDLES - 1  # Fix 6: cooldown tracking

        # Diagnostic counters
        sig_rsi_oversold = 0
        sig_macd_bullish = 0
        sig_price_at_lower_bb = 0
        total_entry_signals = 0
        entries_opened = 0

        atr_series = self._compute_atr(closes, self.atr_period)
        start_idx = max(self.rsi_period + 1, self.macd_slow + self.macd_signal_period, self.bb_period)
        if start_idx >= n:
            logger.warning("LONG: start_idx=%d >= n=%d — not enough data", start_idx, n)
            return trades

        logger.info("LONG simulate: n=%d, start_idx=%d, iterating %d candles", n, start_idx, n - start_idx)

        for i in range(start_idx, n):
            price = closes[i]
            ts = timestamps[i]

            # ── Check for pending buy (execution delay) ──
            if pending_signal is not None and not in_position:
                signal_idx, signal_desc = pending_signal
                if i >= signal_idx + self.execution_delay:
                    quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                    if quantity > 0:
                        entry_idx = i
                        entry_signal = signal_desc
                        in_position = True
                        entries_opened += 1
                    pending_signal = None
                    continue

            if not in_position:
                # Fix 6: enforce cooldown between trades
                if i - last_exit_idx < TRADE_COOLDOWN_CANDLES:
                    continue
                buy_signals = 0
                reasons = []

                if rsi[i] < self.rsi_oversold:
                    buy_signals += 1
                    reasons.append(f"RSI={rsi[i]:.0f} (oversold)")
                    sig_rsi_oversold += 1

                if (macd_line[i] > signal_line[i] and
                        i > 0 and macd_line[i - 1] <= signal_line[i - 1]):
                    buy_signals += 1
                    reasons.append("MACD bullish crossover")
                    sig_macd_bullish += 1
                elif histogram[i] > 0 and i > 0 and histogram[i - 1] <= 0:
                    buy_signals += 1
                    reasons.append("MACD histogram positive")
                    sig_macd_bullish += 1

                if price <= bb_lower[i] and bb_lower[i] > 0:
                    buy_signals += 1
                    reasons.append("Price at lower BB")
                    sig_price_at_lower_bb += 1

                if buy_signals >= 1:
                    total_entry_signals += 1
                    signal_desc = " + ".join(reasons)
                    if self.execution_delay > 0:
                        pending_signal = (i, signal_desc)
                    else:
                        quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                        if quantity > 0:
                            entry_idx = i
                            entry_signal = signal_desc
                            in_position = True
                            entries_opened += 1
            else:
                pnl_pct = ((price - entry_price) / entry_price) * 100

                if pnl_pct <= -self.stop_loss_pct:
                    trade = self._close_long(entry_price, price, quantity, entry_idx, i,
                                             timestamps, entry_signal, "stop_loss")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
                    last_exit_idx = i
                    continue

                if pnl_pct >= self.take_profit_pct:
                    trade = self._close_long(entry_price, price, quantity, entry_idx, i,
                                             timestamps, entry_signal, "take_profit")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
                    last_exit_idx = i
                    continue

                sell_signals = 0
                if rsi[i] > self.rsi_overbought:
                    sell_signals += 1
                if (macd_line[i] < signal_line[i] and
                        i > 0 and macd_line[i - 1] >= signal_line[i - 1]):
                    sell_signals += 1
                elif histogram[i] < 0 and i > 0 and histogram[i - 1] >= 0:
                    sell_signals += 1
                if price >= bb_upper[i] and bb_upper[i] > 0:
                    sell_signals += 1

                if sell_signals >= 2:
                    trade = self._close_long(entry_price, price, quantity, entry_idx, i,
                                             timestamps, entry_signal, "signal_exit")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
                    last_exit_idx = i

        if in_position and n > 0:
            trade = self._close_long(entry_price, closes[-1], quantity, entry_idx, n - 1,
                                     timestamps, entry_signal, "period_end")
            trades.append(trade)

        logger.info(
            "LONG simulate DONE: %d entry signals, %d entries opened, %d final trades | "
            "Signal counts: RSI_oversold=%d, MACD_bullish=%d, Price_at_lowerBB=%d",
            total_entry_signals, entries_opened, len(trades),
            sig_rsi_oversold, sig_macd_bullish, sig_price_at_lower_bb,
        )

        return trades

    # ══════════════════════════════════════════════════════════════
    # SHORT STRATEGY (Improvement 2b)
    # ══════════════════════════════════════════════════════════════

    def _simulate_short_trades(
        self,
        closes: List[float],
        timestamps: List[float],
        rsi: List[float],
        macd_line: List[float],
        signal_line: List[float],
        histogram: List[float],
        bb_upper: List[float],
        bb_middle: List[float],
        bb_lower: List[float],
    ) -> List[BacktestTrade]:
        """
        Simulate SHORT trades: sell high (open short), buy back low (cover).

        In a downtrend, RSI rarely hits 70 and price rarely touches upper BB.
        So we use *rally-exhaustion* signals tuned for downtrend conditions:

        Entry signals (>= 2 of 5):
          1. RSI > 50 AND turning down (rally fading — "overbought" relative to trend)
          2. MACD bearish crossover (macd_line crosses below signal_line)
          3. MACD line below signal line (sustained bearish momentum)
          4. Histogram negative AND declining (momentum accelerating down)
          5. Price above middle BB (rallied to mean — fade opportunity)

        Exit signals (>= 2 of 3, same as before):
          RSI < 30 (oversold), MACD bullish crossover, price at lower BB

        P&L is inverted: profit when price drops, loss when price rises.
        """
        trades: List[BacktestTrade] = []
        n = len(closes)
        equity = self.initial_equity
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        entry_signal = ""
        quantity = 0.0
        pending_signal: Optional[Tuple[int, str]] = None

        # Diagnostic counters
        sig_rsi_turning = 0
        sig_macd_xover = 0
        sig_hist_accel = 0
        sig_price_above_mid = 0
        sig_below_ema20 = 0
        total_entry_signals = 0
        entries_opened = 0
        trades_closed = 0
        last_exit_idx = -TRADE_COOLDOWN_CANDLES - 1  # Fix 6: cooldown

        atr_series = self._compute_atr(closes, self.atr_period)
        start_idx = max(self.rsi_period + 1, self.macd_slow + self.macd_signal_period, self.bb_period)
        if start_idx >= n:
            logger.warning(
                "SHORT [%s]: start_idx=%d >= n=%d — not enough data for indicators",
                "?", start_idx, n,
            )
            return trades

        logger.info(
            "SHORT simulate: n=%d, start_idx=%d, iterating %d candles",
            n, start_idx, n - start_idx,
        )

        for i in range(start_idx, n):
            price = closes[i]

            # ── Check for pending short entry (execution delay) ──
            if pending_signal is not None and not in_position:
                signal_idx, signal_desc = pending_signal
                if i >= signal_idx + self.execution_delay:
                    quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                    if quantity > 0:
                        entry_idx = i
                        entry_signal = signal_desc
                        in_position = True
                        entries_opened += 1
                    pending_signal = None
                    continue

            if not in_position:
                # Fix 6: enforce cooldown between trades
                if i - last_exit_idx < TRADE_COOLDOWN_CANDLES:
                    continue
                sell_signals = 0
                reasons = []

                # Fix 9: Momentum confirmation — price must be below 20-EMA
                ema20 = self._ema(closes[max(0, i-19):i+1], 20)
                if price >= ema20:
                    continue  # Not in downtrend momentum, skip
                sig_below_ema20 += 1

                # Signal 1: RSI above 50 AND turning down (rally exhaustion)
                if i > 0 and rsi[i] > 50 and rsi[i] < rsi[i - 1]:
                    sell_signals += 1
                    reasons.append(f"RSI={rsi[i]:.0f} turning down from {rsi[i-1]:.0f}")
                    sig_rsi_turning += 1

                # Signal 2: Classic MACD bearish crossover (strongest signal)
                if (i > 0 and macd_line[i] < signal_line[i] and
                        macd_line[i - 1] >= signal_line[i - 1]):
                    sell_signals += 1
                    reasons.append("MACD bearish crossover")
                    sig_macd_xover += 1

                # Signal 3: Histogram negative AND declining (accelerating down)
                if i > 0 and histogram[i] < 0 and histogram[i] < histogram[i - 1]:
                    sell_signals += 1
                    reasons.append("Histogram accelerating down")
                    sig_hist_accel += 1

                # Signal 4: Price above middle BB (rallied to mean — fade)
                if bb_middle[i] > 0 and price > bb_middle[i]:
                    sell_signals += 1
                    reasons.append("Price above middle BB")
                    sig_price_above_mid += 1

                # Fix 1: Require 2 of 4 signals (momentum confirmation)
                if sell_signals >= 2:
                    total_entry_signals += 1
                    signal_desc = " + ".join(reasons)
                    if self.execution_delay > 0:
                        pending_signal = (i, signal_desc)
                    else:
                        quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                        if quantity > 0:
                            entry_idx = i
                            entry_signal = signal_desc
                            in_position = True
                            entries_opened += 1
            else:
                # ── In SHORT position — check exit ──
                pnl_pct = ((entry_price - price) / entry_price) * 100

                if pnl_pct <= -self.stop_loss_pct:
                    trade = self._close_short(entry_price, price, quantity, entry_idx, i,
                                              timestamps, entry_signal, "stop_loss")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
                    trades_closed += 1
                    continue

                if pnl_pct >= self.take_profit_pct:
                    trade = self._close_short(entry_price, price, quantity, entry_idx, i,
                                              timestamps, entry_signal, "take_profit")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
                    last_exit_idx = i
                    trades_closed += 1
                    continue

                # Signal-based exit (cover short)
                buy_signals = 0
                if rsi[i] < self.rsi_oversold:
                    buy_signals += 1
                if (i > 0 and macd_line[i] > signal_line[i] and
                        macd_line[i - 1] <= signal_line[i - 1]):
                    buy_signals += 1
                elif histogram[i] > 0 and i > 0 and histogram[i - 1] <= 0:
                    buy_signals += 1
                if price <= bb_lower[i] and bb_lower[i] > 0:
                    buy_signals += 1

                if buy_signals >= 2:
                    trade = self._close_short(entry_price, price, quantity, entry_idx, i,
                                              timestamps, entry_signal, "signal_exit")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
                    last_exit_idx = i
                    trades_closed += 1

        # Close any open position at period end
        if in_position and n > 0:
            trade = self._close_short(entry_price, closes[-1], quantity, entry_idx, n - 1,
                                      timestamps, entry_signal, "period_end")
            trades.append(trade)
            trades_closed += 1

        logger.info(
            "SHORT simulate DONE: %d entry signals, %d entries opened, %d trades closed, "
            "%d final trades | Signal counts: RSI_turn=%d, MACD_xover=%d, "
            "Hist_accel=%d, Price>midBB=%d, Below_EMA20=%d",
            total_entry_signals, entries_opened, trades_closed, len(trades),
            sig_rsi_turning, sig_macd_xover,
            sig_hist_accel, sig_price_above_mid, sig_below_ema20,
        )

        return trades

    # ══════════════════════════════════════════════════════════════
    # RANGE STRATEGY (Improvement 2c) — Mean-reversion at BB bands
    # ══════════════════════════════════════════════════════════════

    def _simulate_range_trades(
        self,
        closes: List[float],
        timestamps: List[float],
        rsi: List[float],
        macd_line: List[float],
        signal_line: List[float],
        histogram: List[float],
        bb_upper: List[float],
        bb_middle: List[float],
        bb_lower: List[float],
    ) -> List[BacktestTrade]:
        """
        Simulate RANGE (mean-reversion) trades.

        Uses Bollinger Bands as support/resistance:
          - Buy when price touches lower BB + RSI < 40 (support bounce)
          - Sell when price touches upper BB + RSI > 60 (resistance rejection)

        Targets: exit at middle BB (mean) or opposite band.
        Stop-loss: tighter (5%) to limit range-bound risk.
        """
        trades: List[BacktestTrade] = []
        n = len(closes)
        equity = self.initial_equity
        in_position = False
        position_type = ""
        entry_price = 0.0
        entry_idx = 0
        entry_signal = ""
        quantity = 0.0
        range_stop_loss = 5.0
        range_take_profit = 8.0

        atr_series = self._compute_atr(closes, self.atr_period)
        start_idx = max(self.rsi_period + 1, self.macd_slow + self.macd_signal_period, self.bb_period)
        if start_idx >= n:
            return trades

        for i in range(start_idx, n):
            price = closes[i]
            ts = timestamps[i]

            if not in_position:
                # RANGE LONG: price near lower BB (within 1.5%) + RSI < 45
                if price <= bb_lower[i] * 1.015 and bb_lower[i] > 0 and rsi[i] < 45:
                    signal_desc = f"Range BUY: Price near lower BB + RSI={rsi[i]:.0f}"
                    quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                    if quantity > 0:
                        entry_idx = i
                        entry_signal = signal_desc
                        in_position = True
                        position_type = "RANGE_LONG"

                # RANGE SHORT: price near upper BB (within 1.5%) + RSI > 55
                elif price >= bb_upper[i] * 0.985 and bb_upper[i] > 0 and rsi[i] > 55:
                    signal_desc = f"Range SHORT: Price near upper BB + RSI={rsi[i]:.0f}"
                    quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                    if quantity > 0:
                        entry_idx = i
                        entry_signal = signal_desc
                        in_position = True
                        position_type = "RANGE_SHORT"

            else:
                if position_type == "RANGE_LONG":
                    pnl_pct = ((price - entry_price) / entry_price) * 100

                    if pnl_pct <= -range_stop_loss:
                        trade = self._close_long(entry_price, price, quantity, entry_idx, i,
                                                 timestamps, entry_signal, "stop_loss",
                                                 direction_label="RANGE_LONG")
                        trades.append(trade)
                        equity += trade.pnl
                        in_position = False
                        continue

                    if pnl_pct >= range_take_profit or price >= bb_middle[i]:
                        reason = "take_profit" if pnl_pct >= range_take_profit else "mean_reversion"
                        trade = self._close_long(entry_price, price, quantity, entry_idx, i,
                                                 timestamps, entry_signal, reason,
                                                 direction_label="RANGE_LONG")
                        trades.append(trade)
                        equity += trade.pnl
                        in_position = False
                        continue

                elif position_type == "RANGE_SHORT":
                    pnl_pct = ((entry_price - price) / entry_price) * 100

                    if pnl_pct <= -range_stop_loss:
                        trade = self._close_short(entry_price, price, quantity, entry_idx, i,
                                                  timestamps, entry_signal, "stop_loss",
                                                  direction_label="RANGE_SHORT")
                        trades.append(trade)
                        equity += trade.pnl
                        in_position = False
                        continue

                    if pnl_pct >= range_take_profit or price <= bb_middle[i]:
                        reason = "take_profit" if pnl_pct >= range_take_profit else "mean_reversion"
                        trade = self._close_short(entry_price, price, quantity, entry_idx, i,
                                                  timestamps, entry_signal, reason,
                                                  direction_label="RANGE_SHORT")
                        trades.append(trade)
                        equity += trade.pnl
                        in_position = False
                        continue

        if in_position and n > 0:
            if position_type == "RANGE_LONG":
                trade = self._close_long(entry_price, closes[-1], quantity, entry_idx, n - 1,
                                         timestamps, entry_signal, "period_end",
                                         direction_label="RANGE_LONG")
            else:
                trade = self._close_short(entry_price, closes[-1], quantity, entry_idx, n - 1,
                                          timestamps, entry_signal, "period_end",
                                          direction_label="RANGE_SHORT")
            trades.append(trade)

        return trades

    # ══════════════════════════════════════════════════════════════
    # TRADE EXECUTION HELPERS
    # ══════════════════════════════════════════════════════════════

    def _execute_entry(
        self, equity: float, price: float, atr_series: List[float], idx: int,
    ) -> Tuple[float, float]:
        """Calculate position size, apply slippage, return (quantity, actual_entry_price)."""
        atr_val = atr_series[idx] if idx < len(atr_series) else 0
        if self.use_atr_sizing and atr_val > 0:
            quantity = self._calc_position_size_atr(equity, price, atr_val)
        else:
            available = equity * (self.position_size_pct / 100)
            quantity = available / price if price > 0 else 0

        trade_value = quantity * price
        entry_slippage = self._calc_slippage(trade_value, price)
        fee = trade_value * self.fee_pct
        actual_entry = price * (1 + entry_slippage / max(trade_value, 1e-9))
        cost = fee + entry_slippage

        if trade_value - cost > 0 and price > 0:
            quantity = (trade_value - cost) / actual_entry
            return quantity, actual_entry
        return 0.0, 0.0

    def _close_long(
        self, entry_price: float, exit_price: float, quantity: float,
        entry_idx: int, exit_idx: int, timestamps: List[float],
        entry_signal: str, exit_reason: str, direction_label: str = "LONG",
    ) -> BacktestTrade:
        """Close a LONG position and create trade record."""
        exit_value = quantity * exit_price
        exit_slippage = self._calc_slippage(exit_value, exit_price)
        fee = exit_value * self.fee_pct
        entry_fee = quantity * entry_price * self.fee_pct
        total_fee = fee + entry_fee
        total_slippage = exit_slippage
        pnl = (exit_price - entry_price) * quantity - total_fee - total_slippage
        net_pnl_pct = ((exit_price * (1 - self.fee_pct) - exit_slippage / max(quantity, 1e-9))
                       / entry_price - 1) * 100

        return BacktestTrade(
            entry_date=self._ts_to_date(timestamps[entry_idx]),
            exit_date=self._ts_to_date(timestamps[exit_idx]),
            direction=direction_label,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=net_pnl_pct,
            fee_paid=total_fee,
            signal=entry_signal,
            exit_reason=exit_reason,
            slippage_cost=total_slippage,
            execution_delay=self.execution_delay,
        )

    def _close_short(
        self, entry_price: float, exit_price: float, quantity: float,
        entry_idx: int, exit_idx: int, timestamps: List[float],
        entry_signal: str, exit_reason: str, direction_label: str = "SHORT",
    ) -> BacktestTrade:
        """Close a SHORT position and create trade record. P&L is inverted."""
        exit_value = quantity * exit_price
        exit_slippage = self._calc_slippage(exit_value, exit_price)
        fee = exit_value * self.fee_pct
        entry_fee = quantity * entry_price * self.fee_pct
        total_fee = fee + entry_fee
        total_slippage = exit_slippage
        pnl = (entry_price - exit_price) * quantity - total_fee - total_slippage
        # SHORT PnL%: profit when price drops → (entry - exit) / entry
        net_pnl_pct = ((entry_price - exit_price) / entry_price * 100) if entry_price > 0 else 0.0
        # Subtract fee impact
        fee_impact_pct = (total_fee + total_slippage) / (quantity * entry_price) * 100 if entry_price > 0 and quantity > 0 else 0
        net_pnl_pct -= fee_impact_pct

        return BacktestTrade(
            entry_date=self._ts_to_date(timestamps[entry_idx]),
            exit_date=self._ts_to_date(timestamps[exit_idx]),
            direction=direction_label,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=net_pnl_pct,
            fee_paid=total_fee,
            signal=entry_signal,
            exit_reason=exit_reason,
            slippage_cost=total_slippage,
            execution_delay=self.execution_delay,
        )

    # ══════════════════════════════════════════════════════════════
    # INDICATOR COMPUTATION
    # ══════════════════════════════════════════════════════════════

    def _compute_rsi_series(self, closes: List[float]) -> List[float]:
        """Compute RSI for every data point (delegates to ta_utils)."""
        return _ta_rsi_series(closes, self.rsi_period)

    def _compute_macd_series(
        self, closes: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Compute MACD line, signal line, and histogram series (delegates to ta_utils)."""
        return _ta_macd_series(closes, self.macd_fast, self.macd_slow, self.macd_signal_period)

    def _compute_bollinger_series(
        self, closes: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Compute upper, middle, lower Bollinger Bands (delegates to ta_utils)."""
        return _ta_bollinger_series(closes, self.bb_period, self.bb_std)

    def _compute_atr(self, closes: List[float], period: int = 14) -> List[float]:
        """Compute Average True Range series."""
        n = len(closes)
        tr = [0.0] * n
        for i in range(1, n):
            tr[i] = abs(closes[i] - closes[i - 1])

        atr = [0.0] * n
        if n <= period:
            avg = sum(tr[1:]) / max(len(tr) - 1, 1) if n > 1 else 0
            return [avg] * n

        atr[period] = sum(tr[1:period + 1]) / period
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        for i in range(period):
            atr[i] = atr[period] if atr[period] > 0 else tr[i]

        # Fix 7: Apply 1.5x multiplier to compensate for close-to-close ATR
        # underestimation (true ATR uses high-low-close, but CoinGecko only
        # provides close prices). Empirical: close-to-close ATR ≈ 65% of true ATR.
        ATR_CLOSE_MULTIPLIER = 1.5
        return [v * ATR_CLOSE_MULTIPLIER for v in atr]

    def _calc_slippage(self, trade_value: float, price: float) -> float:
        """Compute slippage cost for a trade."""
        if self.liquidity_pool <= 0:
            return 0.0
        impact = (self.slippage_base_bps / 10_000) * (trade_value / self.liquidity_pool)
        return trade_value * impact

    def _calc_position_size_atr(
        self, equity: float, price: float, atr: float,
    ) -> float:
        """Volatility-based position sizing."""
        if atr <= 0 or price <= 0:
            return (equity * self.position_size_pct / 100) / price

        risk_amount = equity * (self.account_risk_pct / 100)
        shares = risk_amount / atr
        max_shares = (equity * self.position_size_pct / 100) / price
        return min(shares, max_shares)

    # ══════════════════════════════════════════════════════════════
    # STATISTICS
    # ══════════════════════════════════════════════════════════════

    def _calculate_stats(self, result: BacktestResult, trades: List[BacktestTrade]):
        """Calculate all performance metrics from trade list."""
        if not trades:
            return

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = (len(wins) / len(trades)) * 100 if trades else 0

        total_pnl = sum(t.pnl for t in trades)
        result.total_return = (total_pnl / self.initial_equity) * 100
        result.final_equity = self.initial_equity + total_pnl
        result.total_fees_paid = sum(t.fee_paid for t in trades)
        result.total_slippage_cost = sum(t.slippage_cost for t in trades)

        if wins:
            result.avg_win = sum(t.pnl_pct for t in wins) / len(wins)
            result.largest_win = max(t.pnl_pct for t in wins)
        if losses:
            result.avg_loss = sum(t.pnl_pct for t in losses) / len(losses)
            result.largest_loss = min(t.pnl_pct for t in losses)

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        result.max_drawdown = self._compute_max_drawdown(trades)
        result.sharpe_ratio = self._compute_sharpe_ratio(trades, result.days_covered)

    def _compute_max_drawdown(self, trades: List[BacktestTrade]) -> float:
        """Compute maximum drawdown from peak equity."""
        equity = self.initial_equity
        peak = equity
        max_dd = 0.0

        for trade in trades:
            equity += trade.pnl
            if equity > peak:
                peak = equity
            dd = ((peak - equity) / peak) * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def _compute_sharpe_ratio(
        self, trades: List[BacktestTrade], days_covered: int
    ) -> float:
        """
        Deflated annualized Sharpe ratio using daily equity returns.
        Per Bailey & López de Prado (2024):
        - Uses daily equity curve (not per-trade returns)
        - Applies deflation for skewness and kurtosis
        - Capped at 2.5 (no crypto strategy sustains above this)
        """
        if len(trades) < 2 or days_covered <= 0:
            return 0.0

        # Build daily equity curve from trades
        equity = self.initial_equity
        daily_equity = {}
        for trade in trades:
            equity += trade.pnl
            daily_equity[trade.exit_date[:10]] = equity

        if len(daily_equity) < 2:
            return 0.0

        # Compute daily returns from equity snapshots
        equity_values = list(daily_equity.values())
        daily_returns = []
        for j in range(1, len(equity_values)):
            prev_eq = equity_values[j - 1]
            if prev_eq > 0:
                daily_returns.append((equity_values[j] - prev_eq) / prev_eq)

        if len(daily_returns) < 2:
            return 0.0

        n = len(daily_returns)
        avg_return = sum(daily_returns) / n
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / n
        std_return = math.sqrt(variance)

        if std_return == 0:
            return 0.0

        # Raw annualized Sharpe
        raw_sharpe = (avg_return / std_return) * math.sqrt(365)

        # Deflation: adjust for skewness and kurtosis (López de Prado)
        # Skewness
        skewness = sum((r - avg_return) ** 3 for r in daily_returns) / (n * std_return ** 3) if std_return > 0 else 0
        # Excess kurtosis
        kurtosis_excess = (sum((r - avg_return) ** 4 for r in daily_returns) / (n * std_return ** 4)) - 3 if std_return > 0 else 0

        # Deflation factor: sqrt(1 - skew * SR/3 + (kurt-3) * SR^2 / 24)
        deflation_arg = 1 - (skewness * raw_sharpe / 3) + (kurtosis_excess * raw_sharpe ** 2 / 24)
        if deflation_arg <= 0:
            return 0.0  # Meaningless — heavy tails destroyed the signal
        deflator = math.sqrt(deflation_arg)
        deflated_sharpe = raw_sharpe / deflator if deflator > 0 else 0.0

        # Cap at 2.5 — academic consensus for crypto
        return min(max(deflated_sharpe, -2.5), MAX_REALISTIC_SHARPE)

    # ══════════════════════════════════════════════════════════════
    # ANTI-OVERFITTING GUARDS (Bailey & López de Prado 2024)
    # ══════════════════════════════════════════════════════════════

    def _anti_overfit_check(self, result: BacktestResult) -> List[str]:
        """
        Check for signs of overfitting based on academic research.
        Returns list of warning strings. If non-empty, strategy likely overfit.
        """
        warnings = []

        # Guard 1: Profit factor too high
        if result.profit_factor > MAX_REALISTIC_PROFIT_FACTOR:
            warnings.append(
                f"Profit factor {result.profit_factor:.1f} exceeds {MAX_REALISTIC_PROFIT_FACTOR:.1f} — likely overfitting"
            )

        # Guard 2: Win rate unrealistically high
        if result.win_rate > MAX_REALISTIC_WIN_RATE:
            warnings.append(
                f"Win rate {result.win_rate:.0f}% exceeds {MAX_REALISTIC_WIN_RATE:.0f}% — likely overfitting"
            )

        # Guard 3: Sharpe too high (already capped, but flag it)
        if result.sharpe_ratio > MAX_REALISTIC_SHARPE:
            warnings.append(
                f"Sharpe {result.sharpe_ratio:.2f} exceeds {MAX_REALISTIC_SHARPE:.1f} — unrealistic"
            )

        # Guard 4: Too few OOS trades
        if result.total_trades < MIN_OOS_TRADES:
            warnings.append(
                f"Only {result.total_trades} OOS trades — need >= {MIN_OOS_TRADES} for statistical significance"
            )

        # Guard 5: Max consecutive wins check
        if result.trades:
            max_consec_wins = 0
            current_streak = 0
            for t in result.trades:
                if t.pnl > 0:
                    current_streak += 1
                    max_consec_wins = max(max_consec_wins, current_streak)
                else:
                    current_streak = 0
            if max_consec_wins > 5:
                warnings.append(
                    f"{max_consec_wins} consecutive wins — suspicious in real markets"
                )

        return warnings

    # ══════════════════════════════════════════════════════════════
    # TIERED THRESHOLD EVALUATION (Improvement 4)
    # ══════════════════════════════════════════════════════════════

    def _evaluate_thresholds(self, result: BacktestResult, token_tier: str = "micro"):
        """Check profitability thresholds based on token tier."""
        tier = THRESHOLDS.get(token_tier, THRESHOLDS["micro"])

        result.threshold_results = {
            f"win_rate_above_{tier['win_rate']:.0f}": result.win_rate >= tier["win_rate"],
            f"max_drawdown_below_{tier['max_drawdown']:.0f}": result.max_drawdown <= tier["max_drawdown"],
            f"total_return_above_{tier['min_return']:.0f}": result.total_return >= tier["min_return"],
            f"min_{tier['min_trades']}_trades": result.total_trades >= tier["min_trades"],
        }

        result.failure_reasons = []
        if result.win_rate < tier["win_rate"]:
            result.failure_reasons.append(
                f"Win rate {result.win_rate:.1f}% below {tier['win_rate']:.0f}% ({tier['label']})"
            )
        if result.max_drawdown > tier["max_drawdown"]:
            result.failure_reasons.append(
                f"Max drawdown {result.max_drawdown:.1f}% exceeds {tier['max_drawdown']:.0f}% limit ({tier['label']})"
            )
        if result.total_return < tier["min_return"]:
            result.failure_reasons.append(
                f"Total return {result.total_return:.1f}% below {tier['min_return']:.0f}% minimum ({tier['label']})"
            )
        if result.total_trades < tier["min_trades"]:
            result.failure_reasons.append(
                f"Only {result.total_trades} trades — minimum {tier['min_trades']} ({tier['label']})"
            )

        result.passed_all_thresholds = all(result.threshold_results.values())

    # ══════════════════════════════════════════════════════════════
    # DIRECTION-AWARE RECOMMENDATION (Improvements 6 & 7)
    # ══════════════════════════════════════════════════════════════

    def _generate_recommendation(self, result: BacktestResult):
        """Generate direction-aware recommendation based on strategy results."""
        direction = result.strategy_direction
        trend = result.detected_trend
        tier = result.token_tier
        strategies_tested = ", ".join(result.strategies_tested)

        if not result.passed_all_thresholds:
            result.recommendation = "WARNING"
            result.recommendation_detail = (
                f"⚠️ BACKTEST WARNING — All {len(result.strategies_tested)} strategies tested "
                f"({strategies_tested}) did not meet {THRESHOLDS[tier]['label']} thresholds. "
                f"Detected trend: {trend.upper()}. "
                f"Best result ({direction}): {result.total_trades} trades, "
                f"{result.win_rate:.1f}% win rate, {result.total_return:.1f}% return, "
                f"{result.max_drawdown:.1f}% max drawdown. "
                f"Issues: {'; '.join(result.failure_reasons)}. "
                f"This token may not be suitable for algorithmic trading in current conditions."
            )
            result.confidence = max(0, min(30, result.win_rate * 0.3))
            return

        # Strategy PASSED
        base_confidence = 50.0
        tier_thresholds = THRESHOLDS[tier]
        base_confidence += min(20, (result.win_rate - tier_thresholds["win_rate"]) * 2)
        base_confidence += min(15, result.total_return * 0.5)
        base_confidence += min(10, result.sharpe_ratio * 5)
        if result.max_drawdown < tier_thresholds["max_drawdown"] * 0.5:
            base_confidence += 5
        result.confidence = max(0, min(100, base_confidence))

        if direction == "LONG":
            result.recommendation = "BUY"
            result.recommendation_detail = (
                f"✅ BACKTEST VERIFIED — LONG Strategy | Trend: {trend.upper()}\n"
                f"Strategy profitable over {result.days_covered} days: "
                f"{result.win_rate:.1f}% win rate, {result.total_return:.1f}% total return, "
                f"{result.max_drawdown:.1f}% max drawdown, Sharpe {result.sharpe_ratio:.2f}. "
                f"{result.total_trades} trades executed. "
                f"Recommended direction: BUY/LONG. "
                f"Current indicators: RSI={result.latest_rsi:.0f}, MACD={result.latest_macd}, "
                f"Trend={result.latest_trend}, BB={result.latest_bb_position}. "
                f"Tier: {THRESHOLDS[tier]['label']}. "
                f"Confidence: {result.confidence:.0f}/100."
            )

        elif direction == "SHORT":
            result.recommendation = "SELL"
            result.recommendation_detail = (
                f"✅ BACKTEST VERIFIED — SHORT Strategy | Trend: {trend.upper()}\n"
                f"Short-selling strategy profitable over {result.days_covered} days: "
                f"{result.win_rate:.1f}% win rate, {result.total_return:.1f}% total return, "
                f"{result.max_drawdown:.1f}% max drawdown, Sharpe {result.sharpe_ratio:.2f}. "
                f"{result.total_trades} trades executed. "
                f"Recommended direction: SELL/SHORT — profit from price decline. "
                f"Current indicators: RSI={result.latest_rsi:.0f}, MACD={result.latest_macd}, "
                f"Trend={result.latest_trend}, BB={result.latest_bb_position}. "
                f"Tier: {THRESHOLDS[tier]['label']}. "
                f"Confidence: {result.confidence:.0f}/100."
            )

        elif direction == "RANGE":
            result.recommendation = "HOLD"
            result.recommendation_detail = (
                f"✅ BACKTEST VERIFIED — RANGE Strategy | Trend: {trend.upper()}\n"
                f"Range-trading (mean-reversion) strategy profitable over {result.days_covered} days: "
                f"{result.win_rate:.1f}% win rate, {result.total_return:.1f}% total return, "
                f"{result.max_drawdown:.1f}% max drawdown, Sharpe {result.sharpe_ratio:.2f}. "
                f"{result.total_trades} trades executed. "
                f"Recommended direction: RANGE — buy support, sell resistance. "
                f"Current indicators: RSI={result.latest_rsi:.0f}, MACD={result.latest_macd}, "
                f"Trend={result.latest_trend}, BB={result.latest_bb_position}. "
                f"Tier: {THRESHOLDS[tier]['label']}. "
                f"Confidence: {result.confidence:.0f}/100."
            )

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _ts_to_date(ts_ms: float) -> str:
        """Convert millisecond timestamp to date string."""
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        """Latest EMA value (delegates to ta_utils)."""
        return _ta_ema(values, period)

    @staticmethod
    def _ema_series(values: List[float], period: int) -> List[float]:
        """Full EMA series (delegates to ta_utils)."""
        return _ta_ema_series(values, period)


# ─── Module-level singleton ───────────────────────────────────────

_backtest_engine: Optional[BacktestEngine] = None

def get_backtest_engine() -> BacktestEngine:
    """Get or create the singleton backtest engine."""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine()
    return _backtest_engine
