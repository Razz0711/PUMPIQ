"""
PumpIQ Backtest Engine v2.0 — Multi-Directional Strategy Backtester
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
  - Major (>$1B):  Win Rate >50%, Max DD <25%, Min 8 trades, Return >-5%
  - Mid ($50M-$1B): Win Rate >48%, Max DD <30%, Min 6 trades, Return >-8%
  - Micro (<$50M):  Win Rate >45%, Max DD <35%, Min 5 trades, Return >-10%
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────

MIN_HISTORY_DAYS = 180          # 6 months minimum
TRADING_FEE_PCT = 0.001         # 0.1% per trade (buy + sell)

# ── Tiered Thresholds (by market cap) ─────────────────────────────
THRESHOLDS = {
    "major": {   # > $1B market cap
        "win_rate": 50.0,
        "max_drawdown": 25.0,
        "min_trades": 8,
        "min_return": -5.0,
        "label": "Major Cap (>$1B)",
    },
    "mid": {     # $50M – $1B
        "win_rate": 48.0,
        "max_drawdown": 30.0,
        "min_trades": 6,
        "min_return": -8.0,
        "label": "Mid Cap ($50M–$1B)",
    },
    "micro": {   # < $50M
        "win_rate": 45.0,
        "max_drawdown": 35.0,
        "min_trades": 5,
        "min_return": -10.0,
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
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
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

        # ── Step 2: Detect Trend (EMA50/EMA200) ──
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

        # ── Step 5: Multi-Strategy Cascade ──
        strategy_order = self._get_strategy_order(detected_trend)

        best_result: Optional[BacktestResult] = None
        all_tested: List[str] = []

        for strategy in strategy_order:
            all_tested.append(strategy)
            logger.info("Backtest [%s]: Testing %s strategy...", symbol, strategy)

            trial = self._run_single_strategy(
                result, strategy, indicators, token_tier,
            )

            if trial.passed_all_thresholds:
                best_result = trial
                best_result.best_strategy = strategy
                logger.info(
                    "Backtest [%s]: %s strategy PASSED — WR=%.1f%% Ret=%.1f%% DD=%.1f%%",
                    symbol, strategy, trial.win_rate, trial.total_return, trial.max_drawdown,
                )
                break
            else:
                logger.info(
                    "Backtest [%s]: %s strategy failed — %s",
                    symbol, strategy, "; ".join(trial.failure_reasons),
                )
                if best_result is None or trial.total_return > best_result.total_return:
                    best_result = trial
                    best_result.best_strategy = strategy

        # ── Step 6: Finalize Result ──
        if best_result is None:
            result.recommendation = "WARNING"
            result.recommendation_detail = "No strategies could be tested."
            result.failure_reasons.append("No strategies tested")
            result.strategies_tested = all_tested
            return result

        # Copy best result fields into the main result
        result.total_trades = best_result.total_trades
        result.winning_trades = best_result.winning_trades
        result.losing_trades = best_result.losing_trades
        result.win_rate = best_result.win_rate
        result.total_return = best_result.total_return
        result.max_drawdown = best_result.max_drawdown
        result.sharpe_ratio = best_result.sharpe_ratio
        result.profit_factor = best_result.profit_factor
        result.avg_win = best_result.avg_win
        result.avg_loss = best_result.avg_loss
        result.largest_win = best_result.largest_win
        result.largest_loss = best_result.largest_loss
        result.total_fees_paid = best_result.total_fees_paid
        result.total_slippage_cost = best_result.total_slippage_cost
        result.final_equity = best_result.final_equity
        result.trades = best_result.trades
        result.passed_all_thresholds = best_result.passed_all_thresholds
        result.threshold_results = best_result.threshold_results
        result.failure_reasons = best_result.failure_reasons
        result.strategy_direction = best_result.strategy_direction
        result.best_strategy = best_result.best_strategy
        result.strategies_tested = all_tested

        # ── Step 7: Generate Direction-Aware Recommendation ──
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
    # STRATEGY ORDER (Improvement 5)
    # ══════════════════════════════════════════════════════════════

    def _get_strategy_order(self, trend: str) -> List[str]:
        """
        Determine strategy test order based on detected trend.
        Natural direction first, then alternates.
        """
        if trend == "uptrend":
            return ["LONG", "RANGE", "SHORT"]
        elif trend == "downtrend":
            return ["SHORT", "RANGE", "LONG"]
        else:
            return ["RANGE", "LONG", "SHORT"]

    # ══════════════════════════════════════════════════════════════
    # SINGLE STRATEGY RUNNER
    # ══════════════════════════════════════════════════════════════

    def _run_single_strategy(
        self,
        base_result: BacktestResult,
        strategy: str,
        indicators: Dict[str, Any],
        token_tier: str,
    ) -> BacktestResult:
        """Run one strategy and return a result with stats + threshold evaluation."""
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

        trial.trades = trades
        trial.total_trades = len(trades)

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

        atr_series = self._compute_atr(closes, self.atr_period)
        start_idx = max(self.rsi_period + 1, self.macd_slow + self.macd_signal_period, self.bb_period)
        if start_idx >= n:
            return trades

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
                    pending_signal = None
                    continue

            if not in_position:
                buy_signals = 0
                reasons = []

                if rsi[i] < self.rsi_oversold:
                    buy_signals += 1
                    reasons.append(f"RSI={rsi[i]:.0f} (oversold)")

                if (macd_line[i] > signal_line[i] and
                        i > 0 and macd_line[i - 1] <= signal_line[i - 1]):
                    buy_signals += 1
                    reasons.append("MACD bullish crossover")
                elif histogram[i] > 0 and i > 0 and histogram[i - 1] <= 0:
                    buy_signals += 1
                    reasons.append("MACD histogram positive")

                if price <= bb_lower[i] and bb_lower[i] > 0:
                    buy_signals += 1
                    reasons.append("Price at lower BB")

                if buy_signals >= 2:
                    signal_desc = " + ".join(reasons)
                    if self.execution_delay > 0:
                        pending_signal = (i, signal_desc)
                    else:
                        quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                        if quantity > 0:
                            entry_idx = i
                            entry_signal = signal_desc
                            in_position = True
            else:
                pnl_pct = ((price - entry_price) / entry_price) * 100

                if pnl_pct <= -self.stop_loss_pct:
                    trade = self._close_long(entry_price, price, quantity, entry_idx, i,
                                             timestamps, entry_signal, "stop_loss")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
                    continue

                if pnl_pct >= self.take_profit_pct:
                    trade = self._close_long(entry_price, price, quantity, entry_idx, i,
                                             timestamps, entry_signal, "take_profit")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
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

        if in_position and n > 0:
            trade = self._close_long(entry_price, closes[-1], quantity, entry_idx, n - 1,
                                     timestamps, entry_signal, "period_end")
            trades.append(trade)

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

        Entry: >= 2 of 3 sell signals (RSI overbought, MACD bearish crossover, price at upper BB)
        Exit:  >= 2 of 3 buy signals OR stop-loss (+8%) OR take-profit (-15%)

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

        atr_series = self._compute_atr(closes, self.atr_period)
        start_idx = max(self.rsi_period + 1, self.macd_slow + self.macd_signal_period, self.bb_period)
        if start_idx >= n:
            return trades

        for i in range(start_idx, n):
            price = closes[i]
            ts = timestamps[i]

            if pending_signal is not None and not in_position:
                signal_idx, signal_desc = pending_signal
                if i >= signal_idx + self.execution_delay:
                    quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                    if quantity > 0:
                        entry_idx = i
                        entry_signal = signal_desc
                        in_position = True
                    pending_signal = None
                    continue

            if not in_position:
                sell_signals = 0
                reasons = []

                if rsi[i] > self.rsi_overbought:
                    sell_signals += 1
                    reasons.append(f"RSI={rsi[i]:.0f} (overbought)")

                if (macd_line[i] < signal_line[i] and
                        i > 0 and macd_line[i - 1] >= signal_line[i - 1]):
                    sell_signals += 1
                    reasons.append("MACD bearish crossover")
                elif histogram[i] < 0 and i > 0 and histogram[i - 1] >= 0:
                    sell_signals += 1
                    reasons.append("MACD histogram negative")

                if price >= bb_upper[i] and bb_upper[i] > 0:
                    sell_signals += 1
                    reasons.append("Price at upper BB")

                if sell_signals >= 2:
                    signal_desc = " + ".join(reasons)
                    if self.execution_delay > 0:
                        pending_signal = (i, signal_desc)
                    else:
                        quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                        if quantity > 0:
                            entry_idx = i
                            entry_signal = signal_desc
                            in_position = True
            else:
                pnl_pct = ((entry_price - price) / entry_price) * 100

                if pnl_pct <= -self.stop_loss_pct:
                    trade = self._close_short(entry_price, price, quantity, entry_idx, i,
                                              timestamps, entry_signal, "stop_loss")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
                    continue

                if pnl_pct >= self.take_profit_pct:
                    trade = self._close_short(entry_price, price, quantity, entry_idx, i,
                                              timestamps, entry_signal, "take_profit")
                    trades.append(trade)
                    equity += trade.pnl
                    in_position = False
                    continue

                buy_signals = 0
                if rsi[i] < self.rsi_oversold:
                    buy_signals += 1
                if (macd_line[i] > signal_line[i] and
                        i > 0 and macd_line[i - 1] <= signal_line[i - 1]):
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

        if in_position and n > 0:
            trade = self._close_short(entry_price, closes[-1], quantity, entry_idx, n - 1,
                                      timestamps, entry_signal, "period_end")
            trades.append(trade)

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
                if price <= bb_lower[i] * 1.005 and bb_lower[i] > 0 and rsi[i] < 40:
                    signal_desc = f"Range BUY: Price at lower BB + RSI={rsi[i]:.0f}"
                    quantity, entry_price = self._execute_entry(equity, price, atr_series, i)
                    if quantity > 0:
                        entry_idx = i
                        entry_signal = signal_desc
                        in_position = True
                        position_type = "RANGE_LONG"

                elif price >= bb_upper[i] * 0.995 and bb_upper[i] > 0 and rsi[i] > 60:
                    signal_desc = f"Range SHORT: Price at upper BB + RSI={rsi[i]:.0f}"
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
        net_pnl_pct = ((entry_price * (1 - self.fee_pct) - exit_slippage / max(quantity, 1e-9))
                       / exit_price - 1) * 100

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
        """Compute RSI for every data point."""
        period = self.rsi_period
        n = len(closes)
        if n < period + 1:
            return [50.0] * n

        rsi_values = [50.0] * (period + 1)
        deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
        gains = [max(d, 0) for d in deltas]
        losses = [max(-d, 0) for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        if avg_loss == 0:
            rsi_values[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

        while len(rsi_values) < n:
            rsi_values.append(rsi_values[-1] if rsi_values else 50.0)

        return rsi_values[:n]

    def _compute_macd_series(
        self, closes: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Compute MACD line, signal line, and histogram series."""
        n = len(closes)
        if n < self.macd_slow + self.macd_signal_period:
            return [0.0] * n, [0.0] * n, [0.0] * n

        ema_fast = self._ema_series(closes, self.macd_fast)
        ema_slow = self._ema_series(closes, self.macd_slow)

        macd_full = [0.0] * n
        slow_start = self.macd_slow - 1
        fast_start = self.macd_fast - 1
        for i in range(slow_start, n):
            ei_fast = i - fast_start
            ei_slow = i - slow_start
            if 0 <= ei_fast < len(ema_fast) and 0 <= ei_slow < len(ema_slow):
                macd_full[i] = ema_fast[ei_fast] - ema_slow[ei_slow]

        valid_macd = macd_full[slow_start:]
        signal_raw = self._ema_series(valid_macd, self.macd_signal_period)

        signal_full = [0.0] * n
        sig_start = slow_start + self.macd_signal_period - 1
        for i, v in enumerate(signal_raw):
            idx = sig_start + i
            if idx < n:
                signal_full[idx] = v

        histogram = [macd_full[i] - signal_full[i] for i in range(n)]

        return macd_full, signal_full, histogram

    def _compute_bollinger_series(
        self, closes: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Compute upper, middle, lower Bollinger Bands."""
        period = self.bb_period
        n = len(closes)
        upper = [0.0] * n
        middle = [0.0] * n
        lower = [0.0] * n

        for i in range(period - 1, n):
            window = closes[i - period + 1: i + 1]
            sma = sum(window) / period
            variance = sum((c - sma) ** 2 for c in window) / period
            std = math.sqrt(variance)
            middle[i] = sma
            upper[i] = sma + self.bb_std * std
            lower[i] = sma - self.bb_std * std

        for i in range(period - 1):
            middle[i] = closes[i]
            upper[i] = closes[i]
            lower[i] = closes[i]

        return upper, middle, lower

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

        return atr

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
        """Annualized Sharpe ratio."""
        if len(trades) < 2 or days_covered <= 0:
            return 0.0

        returns = [t.pnl_pct / 100 for t in trades]
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_return = math.sqrt(variance)

        if std_return == 0:
            return 0.0

        trades_per_year = len(trades) / (days_covered / 365) if days_covered > 0 else len(trades)
        annualized_return = avg_return * trades_per_year
        annualized_std = std_return * math.sqrt(trades_per_year)

        return annualized_return / annualized_std if annualized_std > 0 else 0.0

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
        """Latest EMA value."""
        if not values:
            return 0.0
        if len(values) < period:
            return sum(values) / len(values)
        k = 2 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = v * k + ema * (1 - k)
        return ema

    @staticmethod
    def _ema_series(values: List[float], period: int) -> List[float]:
        """Full EMA series."""
        if len(values) < period:
            return values[:]
        k = 2 / (period + 1)
        ema = sum(values[:period]) / period
        result = [ema]
        for v in values[period:]:
            ema = v * k + ema * (1 - k)
            result.append(ema)
        return result


# ─── Module-level singleton ───────────────────────────────────────

_backtest_engine: Optional[BacktestEngine] = None

def get_backtest_engine() -> BacktestEngine:
    """Get or create the singleton backtest engine."""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine()
    return _backtest_engine
