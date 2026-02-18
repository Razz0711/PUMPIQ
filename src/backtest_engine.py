"""
PumpIQ Backtest Engine
========================
Mandatory backtest verification pipeline for all token recommendations.

Pipeline:
  1. Collect >= 6 months OHLCV data from CoinGecko
  2. Run backtest with RSI, MACD, Bollinger Bands (0.1% fees)
  3. Validate against profitability thresholds
  4. Generate recommendation ONLY if all thresholds pass

Thresholds:
  - Win Rate > 55%
  - Max Drawdown < 20%
  - Total Return > 0%
  - Minimum 10 trades
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

# Profitability thresholds
THRESHOLD_WIN_RATE = 55.0       # %
THRESHOLD_MAX_DRAWDOWN = 20.0   # %
THRESHOLD_MIN_TRADES = 10
THRESHOLD_MIN_RETURN = 0.0      # % (must be positive)

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
    direction: str          # "LONG"
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
    slippage_cost: float = 0.0   # total slippage $ both sides
    execution_delay: int = 0     # candles delayed before execution


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
    latest_bb_position: str = "middle"  # "above_upper", "below_lower", "middle"

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
            },
            "recommendation": {
                "verdict": self.recommendation,
                "detail": self.recommendation_detail,
                "confidence": round(self.confidence, 1),
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
    Runs strategy backtests on historical OHLCV data.

    Strategy uses combined signals from:
      - RSI (14): oversold < 30 → buy, overbought > 70 → sell
      - MACD (12/26/9): bullish crossover → buy, bearish → sell
      - Bollinger Bands (20, 2σ): price below lower → buy, above upper → sell

    Entry: At least 2 of 3 indicators agree on direction
    Exit:  Opposite signal, stop-loss (-8%), or take-profit (+15%)
    """

    def __init__(
        self,
        initial_equity: float = 10000.0,
        position_size_pct: float = 95.0,   # % of equity per trade
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
        # Slippage & execution model
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
        # Slippage & execution parameters
        self.slippage_base_bps = slippage_base_bps
        self.liquidity_pool = liquidity_pool
        self.execution_delay = execution_delay
        self.use_atr_sizing = use_atr_sizing
        self.account_risk_pct = account_risk_pct
        self.atr_period = atr_period

    # ── Public API ─────────────────────────────────────────────────

    async def run_backtest(
        self,
        coin_id: str,
        coin_name: str,
        symbol: str,
        cg_collector,
        days: int = MIN_HISTORY_DAYS,
    ) -> BacktestResult:
        """
        Full pipeline: fetch data → compute indicators → simulate trades → evaluate.

        Parameters
        ----------
        coin_id : str
            CoinGecko coin ID (e.g. "bitcoin", "solana")
        coin_name : str
            Display name
        symbol : str
            Ticker symbol
        cg_collector : CoinGeckoCollector
            Initialized collector instance
        days : int
            Number of days of history to fetch (min 180)

        Returns
        -------
        BacktestResult with all stats, thresholds, and conditional recommendation.
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

        prices = history.prices           # [[ts_ms, price], ...]
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

        # ── Step 2: Compute Indicators ──
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

        # Detect trend from EMAs
        if len(closes) >= 50:
            ema20 = self._ema(closes, 20)
            ema50 = self._ema(closes, 50)
            if ema20 > ema50 and closes[-1] > ema20:
                result.latest_trend = "uptrend"
            elif ema20 < ema50 and closes[-1] < ema20:
                result.latest_trend = "downtrend"
            else:
                result.latest_trend = "sideways"

        # ── Step 3: Simulate Trades ──
        trades = self._simulate_trades(
            closes, timestamps, rsi_series, macd_line, signal_line, histogram,
            bb_upper, bb_middle, bb_lower,
        )
        result.trades = trades
        result.total_trades = len(trades)

        if result.total_trades == 0:
            result.recommendation = "WARNING"
            result.recommendation_detail = (
                "No trade signals generated during the backtest period. "
                "The strategy found no clear entry/exit points."
            )
            result.failure_reasons.append("No trades executed")
            return result

        # ── Step 4: Calculate Statistics ──
        self._calculate_stats(result, trades)

        # ── Step 5: Evaluate Thresholds ──
        self._evaluate_thresholds(result)

        # ── Step 6: Generate Recommendation ──
        self._generate_recommendation(result)

        logger.info(
            "Backtest [%s]: %d trades | Win Rate %.1f%% | Return %.1f%% | "
            "Max DD %.1f%% | Sharpe %.2f | Passed: %s",
            symbol, result.total_trades, result.win_rate, result.total_return,
            result.max_drawdown, result.sharpe_ratio, result.passed_all_thresholds,
        )

        return result

    # ── Indicator Computation ──────────────────────────────────────

    def _compute_rsi_series(self, closes: List[float]) -> List[float]:
        """Compute RSI for every data point (returns list same length as closes)."""
        period = self.rsi_period
        n = len(closes)
        if n < period + 1:
            return [50.0] * n

        rsi_values = [50.0] * (period + 1)  # pad initial values (indices 0..period)
        deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
        gains = [max(d, 0) for d in deltas]
        losses = [max(-d, 0) for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # RSI at index=period (first valid RSI)
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

        # Ensure exact length match
        while len(rsi_values) < n:
            rsi_values.append(rsi_values[-1] if rsi_values else 50.0)

        return rsi_values[:n]

    def _compute_macd_series(
        self, closes: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Compute MACD line, signal line, and histogram series (each len = len(closes))."""
        n = len(closes)
        if n < self.macd_slow + self.macd_signal_period:
            return [0.0] * n, [0.0] * n, [0.0] * n

        ema_fast = self._ema_series(closes, self.macd_fast)
        ema_slow = self._ema_series(closes, self.macd_slow)

        # Build MACD line aligned to closes
        macd_full = [0.0] * n
        # ema_fast starts at index (macd_fast - 1), ema_slow at (macd_slow - 1)
        slow_start = self.macd_slow - 1
        fast_start = self.macd_fast - 1
        for i in range(slow_start, n):
            ei_fast = i - fast_start
            ei_slow = i - slow_start
            if 0 <= ei_fast < len(ema_fast) and 0 <= ei_slow < len(ema_slow):
                macd_full[i] = ema_fast[ei_fast] - ema_slow[ei_slow]

        # Signal line from the valid MACD portion
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

        # Fill initial values
        for i in range(period - 1):
            middle[i] = closes[i]
            upper[i] = closes[i]
            lower[i] = closes[i]

        return upper, middle, lower

    # ── Trade Simulation ───────────────────────────────────────────

    def _compute_atr(self, closes: List[float], period: int = 14) -> List[float]:
        """Compute Average True Range series (simplified for daily closes only)."""
        n = len(closes)
        tr = [0.0] * n
        for i in range(1, n):
            tr[i] = abs(closes[i] - closes[i - 1])

        atr = [0.0] * n
        if n <= period:
            avg = sum(tr[1:]) / max(len(tr) - 1, 1) if n > 1 else 0
            return [avg] * n

        # Initial ATR = simple average of first `period` TRs
        atr[period] = sum(tr[1:period + 1]) / period
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        # Fill initial values
        for i in range(period):
            atr[i] = atr[period] if atr[period] > 0 else tr[i]

        return atr

    def _calc_slippage(self, trade_value: float, price: float) -> float:
        """
        Compute slippage cost for a trade.

        Formula: slippage = base_fee_bps * (trade_size / liquidity_pool)
        Applied as a price impact in $ terms.
        """
        if self.liquidity_pool <= 0:
            return 0.0
        impact = (self.slippage_base_bps / 10_000) * (trade_value / self.liquidity_pool)
        return trade_value * impact

    def _calc_position_size_atr(
        self, equity: float, price: float, atr: float,
    ) -> float:
        """
        Volatility-based position sizing.

        Formula: position_size = (account_risk% × equity) / ATR
        This sizes positions inversely to volatility — smaller in volatile
        markets, larger in calm markets.
        """
        if atr <= 0 or price <= 0:
            # Fall back to percentage-based sizing
            return (equity * self.position_size_pct / 100) / price

        risk_amount = equity * (self.account_risk_pct / 100)
        shares = risk_amount / atr
        # Cap at position_size_pct of equity
        max_shares = (equity * self.position_size_pct / 100) / price
        return min(shares, max_shares)

    def _simulate_trades(
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
        Simulate long-only trades using combined indicator signals.

        Features:
          - Execution delay: signal at candle i → execute at candle i+delay
          - Slippage model: slippage = base_bps × (trade_size / liquidity_pool)
          - ATR-based position sizing: position = (account_risk% × equity) / ATR

        Entry: >= 2 of 3 buy signals active
        Exit: >= 2 of 3 sell signals OR stop-loss OR take-profit
        """
        trades: List[BacktestTrade] = []
        n = len(closes)
        equity = self.initial_equity
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        entry_signal = ""
        quantity = 0.0
        pending_buy_signal: Optional[Tuple[int, str]] = None  # (signal_candle_idx, signal_desc)

        # Compute ATR for position sizing
        atr_series = self._compute_atr(closes, self.atr_period)

        # Verify all indicator arrays match closes length
        assert len(rsi) == n, f"RSI len {len(rsi)} != closes len {n}"
        assert len(macd_line) == n, f"MACD len {len(macd_line)} != closes len {n}"
        assert len(signal_line) == n, f"Signal len {len(signal_line)} != closes len {n}"
        assert len(histogram) == n, f"Histogram len {len(histogram)} != closes len {n}"
        assert len(bb_upper) == n, f"BB upper len {len(bb_upper)} != closes len {n}"
        assert len(bb_lower) == n, f"BB lower len {len(bb_lower)} != closes len {n}"

        # Start after all indicators are initialized
        start_idx = max(self.rsi_period + 1, self.macd_slow + self.macd_signal_period, self.bb_period)
        if start_idx >= n:
            return trades

        for i in range(start_idx, n):
            price = closes[i]
            ts = timestamps[i]

            # ── Check for pending buy (execution delay) ──
            if pending_buy_signal is not None and not in_position:
                signal_idx, signal_desc = pending_buy_signal
                if i >= signal_idx + self.execution_delay:
                    # Execute the delayed buy
                    atr_val = atr_series[i] if i < len(atr_series) else 0
                    if self.use_atr_sizing and atr_val > 0:
                        quantity = self._calc_position_size_atr(equity, price, atr_val)
                    else:
                        available = equity * (self.position_size_pct / 100)
                        quantity = available / price if price > 0 else 0

                    trade_value = quantity * price
                    entry_slippage = self._calc_slippage(trade_value, price)
                    fee = trade_value * self.fee_pct

                    # Adjust entry price for slippage (buy at worse price)
                    actual_entry = price * (1 + entry_slippage / max(trade_value, 1e-9))
                    # Deduct fee + slippage from available
                    cost = fee + entry_slippage
                    if trade_value - cost > 0 and price > 0:
                        quantity = (trade_value - cost) / actual_entry
                        entry_price = actual_entry
                        entry_idx = i
                        entry_signal = signal_desc
                        in_position = True
                    pending_buy_signal = None
                    continue

            if not in_position:
                # ── Check BUY signals ──
                buy_signals = 0
                reasons = []

                # RSI oversold
                if rsi[i] < self.rsi_oversold:
                    buy_signals += 1
                    reasons.append(f"RSI={rsi[i]:.0f} (oversold)")

                # MACD bullish crossover
                if (macd_line[i] > signal_line[i] and
                        i > 0 and macd_line[i - 1] <= signal_line[i - 1]):
                    buy_signals += 1
                    reasons.append("MACD bullish crossover")
                elif histogram[i] > 0 and i > 0 and histogram[i - 1] <= 0:
                    buy_signals += 1
                    reasons.append("MACD histogram turned positive")

                # Bollinger: price near or below lower band
                if price <= bb_lower[i] and bb_lower[i] > 0:
                    buy_signals += 1
                    reasons.append("Price at lower Bollinger Band")

                # Need at least 2 confirming signals
                if buy_signals >= 2:
                    signal_desc = " + ".join(reasons)
                    if self.execution_delay > 0:
                        # Defer execution by N candles
                        pending_buy_signal = (i, signal_desc)
                    else:
                        # Immediate execution (delay=0)
                        atr_val = atr_series[i] if i < len(atr_series) else 0
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
                            entry_price = actual_entry
                            entry_idx = i
                            entry_signal = signal_desc
                            in_position = True

            else:
                # ── Check EXIT conditions ──
                pnl_pct = ((price - entry_price) / entry_price) * 100

                # Stop-loss
                if pnl_pct <= -self.stop_loss_pct:
                    exit_reason = "stop_loss"
                    exit_value = quantity * price
                    exit_slippage = self._calc_slippage(exit_value, price)
                    fee = exit_value * self.fee_pct
                    entry_fee = quantity * entry_price * self.fee_pct
                    total_fee = fee + entry_fee
                    total_slippage = exit_slippage  # entry slippage already in entry_price
                    pnl = (price - entry_price) * quantity - total_fee - total_slippage
                    net_pnl_pct = ((price * (1 - self.fee_pct) - exit_slippage / max(quantity, 1e-9)) / (entry_price) - 1) * 100
                    trades.append(BacktestTrade(
                        entry_date=self._ts_to_date(timestamps[entry_idx]),
                        exit_date=self._ts_to_date(ts),
                        direction="LONG",
                        entry_price=entry_price,
                        exit_price=price,
                        quantity=quantity,
                        pnl=pnl,
                        pnl_pct=net_pnl_pct,
                        fee_paid=total_fee,
                        signal=entry_signal,
                        exit_reason=exit_reason,
                        slippage_cost=total_slippage,
                        execution_delay=self.execution_delay,
                    ))
                    equity += pnl
                    in_position = False
                    continue

                # Take-profit
                if pnl_pct >= self.take_profit_pct:
                    exit_reason = "take_profit"
                    exit_value = quantity * price
                    exit_slippage = self._calc_slippage(exit_value, price)
                    fee = exit_value * self.fee_pct
                    entry_fee = quantity * entry_price * self.fee_pct
                    total_fee = fee + entry_fee
                    total_slippage = exit_slippage
                    pnl = (price - entry_price) * quantity - total_fee - total_slippage
                    net_pnl_pct = ((price * (1 - self.fee_pct) - exit_slippage / max(quantity, 1e-9)) / (entry_price) - 1) * 100
                    trades.append(BacktestTrade(
                        entry_date=self._ts_to_date(timestamps[entry_idx]),
                        exit_date=self._ts_to_date(ts),
                        direction="LONG",
                        entry_price=entry_price,
                        exit_price=price,
                        quantity=quantity,
                        pnl=pnl,
                        pnl_pct=net_pnl_pct,
                        fee_paid=total_fee,
                        signal=entry_signal,
                        exit_reason=exit_reason,
                        slippage_cost=total_slippage,
                        execution_delay=self.execution_delay,
                    ))
                    equity += pnl
                    in_position = False
                    continue

                # Signal-based exit: >= 2 sell signals
                sell_signals = 0

                # RSI overbought
                if rsi[i] > self.rsi_overbought:
                    sell_signals += 1

                # MACD bearish crossover
                if (macd_line[i] < signal_line[i] and
                        i > 0 and macd_line[i - 1] >= signal_line[i - 1]):
                    sell_signals += 1
                elif histogram[i] < 0 and i > 0 and histogram[i - 1] >= 0:
                    sell_signals += 1

                # Price above upper Bollinger Band
                if price >= bb_upper[i] and bb_upper[i] > 0:
                    sell_signals += 1

                if sell_signals >= 2:
                    exit_reason = "signal_exit"
                    exit_value = quantity * price
                    exit_slippage = self._calc_slippage(exit_value, price)
                    fee = exit_value * self.fee_pct
                    entry_fee = quantity * entry_price * self.fee_pct
                    total_fee = fee + entry_fee
                    total_slippage = exit_slippage
                    pnl = (price - entry_price) * quantity - total_fee - total_slippage
                    net_pnl_pct = ((price * (1 - self.fee_pct) - exit_slippage / max(quantity, 1e-9)) / (entry_price) - 1) * 100
                    trades.append(BacktestTrade(
                        entry_date=self._ts_to_date(timestamps[entry_idx]),
                        exit_date=self._ts_to_date(ts),
                        direction="LONG",
                        entry_price=entry_price,
                        exit_price=price,
                        quantity=quantity,
                        pnl=pnl,
                        pnl_pct=net_pnl_pct,
                        fee_paid=total_fee,
                        signal=entry_signal,
                        exit_reason=exit_reason,
                        slippage_cost=total_slippage,
                        execution_delay=self.execution_delay,
                    ))
                    equity += pnl
                    in_position = False

        # Close any remaining open position at last price
        if in_position and n > 0:
            price = closes[-1]
            exit_value = quantity * price
            exit_slippage = self._calc_slippage(exit_value, price)
            fee = exit_value * self.fee_pct
            entry_fee = quantity * entry_price * self.fee_pct
            total_fee = fee + entry_fee
            total_slippage = exit_slippage
            pnl = (price - entry_price) * quantity - total_fee - total_slippage
            net_pnl_pct = ((price * (1 - self.fee_pct) - exit_slippage / max(quantity, 1e-9)) / (entry_price) - 1) * 100
            trades.append(BacktestTrade(
                entry_date=self._ts_to_date(timestamps[entry_idx]),
                exit_date=self._ts_to_date(timestamps[-1]),
                direction="LONG",
                entry_price=entry_price,
                exit_price=price,
                quantity=quantity,
                pnl=pnl,
                pnl_pct=net_pnl_pct,
                fee_paid=total_fee,
                signal=entry_signal,
                exit_reason="period_end",
                slippage_cost=total_slippage,
                execution_delay=self.execution_delay,
            ))

        return trades

    # ── Statistics Calculation ─────────────────────────────────────

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

        # Win/loss averages
        if wins:
            result.avg_win = sum(t.pnl_pct for t in wins) / len(wins)
            result.largest_win = max(t.pnl_pct for t in wins)
        if losses:
            result.avg_loss = sum(t.pnl_pct for t in losses) / len(losses)
            result.largest_loss = min(t.pnl_pct for t in losses)

        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        result.max_drawdown = self._compute_max_drawdown(trades)

        # Sharpe ratio (annualized, assuming daily returns)
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
        Annualized Sharpe ratio.
        Risk-free rate assumed 0 for crypto.
        """
        if len(trades) < 2 or days_covered <= 0:
            return 0.0

        returns = [t.pnl_pct / 100 for t in trades]
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_return = math.sqrt(variance)

        if std_return == 0:
            return 0.0

        # Annualize: trades_per_year = total_trades / (days / 365)
        trades_per_year = len(trades) / (days_covered / 365) if days_covered > 0 else len(trades)
        annualized_return = avg_return * trades_per_year
        annualized_std = std_return * math.sqrt(trades_per_year)

        return annualized_return / annualized_std if annualized_std > 0 else 0.0

    # ── Threshold Evaluation ───────────────────────────────────────

    def _evaluate_thresholds(self, result: BacktestResult):
        """Check all profitability thresholds."""
        result.threshold_results = {
            "win_rate_above_55": result.win_rate > THRESHOLD_WIN_RATE,
            "max_drawdown_below_20": result.max_drawdown < THRESHOLD_MAX_DRAWDOWN,
            "total_return_positive": result.total_return > THRESHOLD_MIN_RETURN,
            "min_10_trades": result.total_trades >= THRESHOLD_MIN_TRADES,
        }

        result.failure_reasons = []
        if not result.threshold_results["win_rate_above_55"]:
            result.failure_reasons.append(
                f"Win rate {result.win_rate:.1f}% is below {THRESHOLD_WIN_RATE}% threshold"
            )
        if not result.threshold_results["max_drawdown_below_20"]:
            result.failure_reasons.append(
                f"Max drawdown {result.max_drawdown:.1f}% exceeds {THRESHOLD_MAX_DRAWDOWN}% limit"
            )
        if not result.threshold_results["total_return_positive"]:
            result.failure_reasons.append(
                f"Total return {result.total_return:.1f}% is not positive"
            )
        if not result.threshold_results["min_10_trades"]:
            result.failure_reasons.append(
                f"Only {result.total_trades} trades — minimum {THRESHOLD_MIN_TRADES} required"
            )

        result.passed_all_thresholds = all(result.threshold_results.values())

    # ── Recommendation Generation ──────────────────────────────────

    def _generate_recommendation(self, result: BacktestResult):
        """Generate recommendation ONLY if all thresholds pass."""
        if not result.passed_all_thresholds:
            result.recommendation = "WARNING"
            result.recommendation_detail = (
                f"BACKTEST FAILED — This token did not pass profitability verification. "
                f"Issues: {'; '.join(result.failure_reasons)}. "
                f"Stats: {result.total_trades} trades, {result.win_rate:.1f}% win rate, "
                f"{result.total_return:.1f}% return, {result.max_drawdown:.1f}% max drawdown. "
                f"DO NOT trade this token based on current strategy signals."
            )
            result.confidence = max(0, min(30, result.win_rate * 0.3))
            return

        # ── All thresholds passed → generate directional recommendation ──
        # Combine current indicator states for direction
        bullish_signals = 0
        bearish_signals = 0

        if result.latest_rsi < 40:
            bullish_signals += 1
        elif result.latest_rsi > 60:
            bearish_signals += 1

        if result.latest_macd == "bullish":
            bullish_signals += 1
        elif result.latest_macd == "bearish":
            bearish_signals += 1

        if result.latest_trend == "uptrend":
            bullish_signals += 1
        elif result.latest_trend == "downtrend":
            bearish_signals += 1

        if result.latest_bb_position == "below_lower":
            bullish_signals += 1
        elif result.latest_bb_position == "above_upper":
            bearish_signals += 1

        # Calculate confidence from backtest quality
        base_confidence = 50.0
        base_confidence += min(20, (result.win_rate - THRESHOLD_WIN_RATE) * 2)
        base_confidence += min(15, result.total_return * 0.5)
        base_confidence += min(10, result.sharpe_ratio * 5)
        if result.max_drawdown < 10:
            base_confidence += 5
        result.confidence = max(0, min(100, base_confidence))

        if bullish_signals >= 2 and bearish_signals == 0:
            result.recommendation = "BUY"
            result.recommendation_detail = (
                f"BACKTEST VERIFIED BUY — Strategy profitable over {result.days_covered} days: "
                f"{result.win_rate:.1f}% win rate, {result.total_return:.1f}% total return, "
                f"{result.max_drawdown:.1f}% max drawdown, Sharpe {result.sharpe_ratio:.2f}. "
                f"Current signals: RSI={result.latest_rsi:.0f}, MACD={result.latest_macd}, "
                f"Trend={result.latest_trend}, BB={result.latest_bb_position}. "
                f"Confidence: {result.confidence:.0f}/100."
            )
        elif bearish_signals >= 2 and bullish_signals == 0:
            result.recommendation = "SELL"
            result.recommendation_detail = (
                f"BACKTEST VERIFIED SELL — Strategy profitable but current indicators bearish: "
                f"RSI={result.latest_rsi:.0f}, MACD={result.latest_macd}, "
                f"Trend={result.latest_trend}. "
                f"Wait for better entry. Backtest: {result.win_rate:.1f}% win rate, "
                f"{result.total_return:.1f}% return over {result.days_covered} days."
            )
            result.confidence = max(0, result.confidence - 15)
        else:
            result.recommendation = "HOLD"
            result.recommendation_detail = (
                f"BACKTEST VERIFIED HOLD — Strategy is profitable ({result.win_rate:.1f}% "
                f"win rate, {result.total_return:.1f}% return) but current signals are mixed. "
                f"RSI={result.latest_rsi:.0f}, MACD={result.latest_macd}, "
                f"Trend={result.latest_trend}. Wait for clearer entry signal."
            )
            result.confidence = max(0, result.confidence - 10)

    # ── Helpers ────────────────────────────────────────────────────

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
