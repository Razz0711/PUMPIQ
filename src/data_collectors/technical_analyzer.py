"""
Technical Analyzer
====================
Computes classic technical-analysis indicators from price history arrays
(supplied by CoinGecko ``/market_chart`` or any OHLC source).

Indicators implemented
----------------------
- **RSI** (14-period Relative Strength Index)
- **MACD** (12/26/9 EMA crossover)
- **Support / Resistance** (local min/max of price series)
- **Trend detection** (EMA-20 vs EMA-50 crossover)
- **Momentum score** (combined RSI + MACD + trend)

No external TA library needed (pure Python).  Uses only the ``prices``
list-of-lists ``[[timestamp_ms, price], …]`` from CoinGecko.

Output: ``TechnicalResult`` dataclass that maps to ``TechnicalScorePayload``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ───────────────────────── Data class ──────────────────────────

@dataclass
class TechnicalResult:
    """
    Technical-analysis summary for one token.

    Maps to the engine's ``TechnicalScorePayload``.
    """
    score: float = 5.0             # 0-10 composite technical score
    trend: str = "sideways"        # "uptrend" | "downtrend" | "sideways"
    rsi: float = 50.0              # 0-100
    rsi_label: str = "neutral"     # "overbought" | "neutral" | "oversold"
    macd_value: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_crossover: str = "neutral"  # "bullish_crossover" | "bearish" | "neutral"
    support: float = 0.0
    resistance: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    pattern: str = "None"
    summary: str = ""


# ───────────────────────── Analyzer ────────────────────────────

class TechnicalAnalyzer:
    """
    Pure-Python technical analysis engine.

    Usage::

        ta = TechnicalAnalyzer()
        # prices from CoinGecko: [[ts_ms, price], …]
        result = ta.analyze(prices)
        print(result.score, result.trend, result.rsi)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        ema_short: int = 12,
        ema_long: int = 26,
        signal_period: int = 9,
    ):
        self.rsi_period = rsi_period
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.signal_period = signal_period

    # ── public API ─────────────────────────────────────────────────

    def analyze(
        self,
        prices: List[List[float]],
        current_price: float = 0.0,
    ) -> TechnicalResult:
        """
        Run full TA suite on a price series.

        Parameters
        ----------
        prices : list of [timestamp_ms, price]
            As returned by CoinGecko ``/coins/{id}/market_chart``.
        current_price : float
            Latest price (for support/resistance context).
        """
        if len(prices) < 30:
            logger.warning("Too few data points (%d) for reliable TA", len(prices))
            return TechnicalResult(summary="Insufficient price data for analysis")

        closes = [p[1] for p in prices]
        if current_price <= 0:
            current_price = closes[-1]

        # --- Individual indicators ---
        rsi = self._compute_rsi(closes)
        rsi_label = self._rsi_label(rsi)

        macd_val, signal_val, hist = self._compute_macd(closes)
        macd_cross = self._macd_crossover(macd_val, signal_val, hist)

        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50) if len(closes) >= 50 else ema20
        trend = self._detect_trend(closes, ema20, ema50)

        support, resistance = self._support_resistance(closes)
        pattern = self._detect_pattern(closes, rsi, macd_cross, trend)

        # --- Composite score (0-10) ---
        score = self._composite_score(rsi, macd_cross, trend, current_price, support, resistance)

        summary = self._build_summary(
            trend, rsi, rsi_label, macd_cross, support, resistance, current_price, pattern,
        )

        return TechnicalResult(
            score=score,
            trend=trend,
            rsi=round(rsi, 1),
            rsi_label=rsi_label,
            macd_value=round(macd_val, 6),
            macd_signal=round(signal_val, 6),
            macd_histogram=round(hist, 6),
            macd_crossover=macd_cross,
            support=round(support, 6),
            resistance=round(resistance, 6),
            ema_20=round(ema20, 6),
            ema_50=round(ema50, 6),
            pattern=pattern,
            summary=summary,
        )

    # ── RSI ────────────────────────────────────────────────────────

    def _compute_rsi(self, closes: List[float]) -> float:
        if len(closes) < self.rsi_period + 1:
            return 50.0

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[: self.rsi_period]) / self.rsi_period
        avg_loss = sum(losses[: self.rsi_period]) / self.rsi_period

        for i in range(self.rsi_period, len(deltas)):
            avg_gain = (avg_gain * (self.rsi_period - 1) + gains[i]) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + losses[i]) / self.rsi_period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _rsi_label(rsi: float) -> str:
        if rsi >= 70:
            return "overbought"
        if rsi <= 30:
            return "oversold"
        return "neutral"

    # ── MACD ───────────────────────────────────────────────────────

    def _compute_macd(self, closes: List[float]) -> Tuple[float, float, float]:
        """Return (macd_line, signal_line, histogram)."""
        if len(closes) < self.ema_long + self.signal_period:
            return 0.0, 0.0, 0.0

        ema_short_vals = self._ema_series(closes, self.ema_short)
        ema_long_vals = self._ema_series(closes, self.ema_long)

        # Align lengths
        min_len = min(len(ema_short_vals), len(ema_long_vals))
        macd_line = [
            ema_short_vals[-(min_len - i)] - ema_long_vals[-(min_len - i)]
            for i in range(min_len)
        ]

        signal_line = self._ema_series(macd_line, self.signal_period)

        macd_val = macd_line[-1] if macd_line else 0
        signal_val = signal_line[-1] if signal_line else 0
        histogram = macd_val - signal_val

        return macd_val, signal_val, histogram

    @staticmethod
    def _macd_crossover(macd: float, signal: float, hist: float) -> str:
        if macd > signal and hist > 0:
            return "bullish_crossover"
        if macd < signal and hist < 0:
            return "bearish"
        return "neutral"

    # ── EMA helpers ────────────────────────────────────────────────

    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        """Return the latest EMA value."""
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
        """Return the full EMA series."""
        if len(values) < period:
            return values[:]
        k = 2 / (period + 1)
        ema = sum(values[:period]) / period
        result = [ema]
        for v in values[period:]:
            ema = v * k + ema * (1 - k)
            result.append(ema)
        return result

    # ── Trend detection ────────────────────────────────────────────

    @staticmethod
    def _detect_trend(closes: List[float], ema20: float, ema50: float) -> str:
        last = closes[-1]
        if ema20 > ema50 and last > ema20:
            return "uptrend"
        if ema20 < ema50 and last < ema20:
            return "downtrend"
        return "sideways"

    # ── Support / Resistance ───────────────────────────────────────

    @staticmethod
    def _support_resistance(
        closes: List[float], window: int = 10,
    ) -> Tuple[float, float]:
        """
        Simple local-min / local-max over a rolling window.

        Returns (support, resistance) closest to current price.
        """
        if len(closes) < window * 2:
            return min(closes), max(closes)

        local_mins: List[float] = []
        local_maxs: List[float] = []

        for i in range(window, len(closes) - window):
            segment = closes[i - window: i + window + 1]
            if closes[i] == min(segment):
                local_mins.append(closes[i])
            if closes[i] == max(segment):
                local_maxs.append(closes[i])

        current = closes[-1]
        support = max((p for p in local_mins if p < current), default=min(closes))
        resistance = min((p for p in local_maxs if p > current), default=max(closes))

        return support, resistance

    # ── Pattern detection ──────────────────────────────────────────

    @staticmethod
    def _detect_pattern(
        closes: List[float],
        rsi: float,
        macd_cross: str,
        trend: str,
    ) -> str:
        """
        Simple heuristic pattern detector.

        Returns one of:
          - "Bullish Reversal"
          - "Bearish Reversal"
          - "Bullish Continuation"
          - "Bearish Continuation"
          - "Consolidation"
          - "None"
        """
        # Oversold + bullish MACD crossover → reversal
        if rsi < 35 and macd_cross == "bullish_crossover":
            return "Bullish Reversal"

        # Overbought + bearish MACD → reversal
        if rsi > 65 and macd_cross == "bearish":
            return "Bearish Reversal"

        # Uptrend + healthy RSI → continuation
        if trend == "uptrend" and 40 < rsi < 70:
            return "Bullish Continuation"

        # Downtrend + bearish MACD → continuation
        if trend == "downtrend" and macd_cross == "bearish":
            return "Bearish Continuation"

        # Tight range → consolidation
        recent = closes[-20:] if len(closes) >= 20 else closes
        if recent:
            pct_range = (max(recent) - min(recent)) / max(max(recent), 1e-9) * 100
            if pct_range < 5:
                return "Consolidation"

        return "None"

    # ── Composite score ────────────────────────────────────────────

    def _composite_score(
        self,
        rsi: float,
        macd_cross: str,
        trend: str,
        price: float,
        support: float,
        resistance: float,
    ) -> float:
        """
        0-10 technical score combining all indicators.

        Breakdown:
          - RSI component    (0-3 pts)
          - MACD component   (0-3 pts)
          - Trend component  (0-2 pts)
          - S/R component    (0-2 pts)
        """
        score = 0.0

        # RSI (0-3): Oversold = 3, neutral = 1.5, overbought = 0
        if rsi <= 30:
            score += 3.0
        elif rsi <= 45:
            score += 2.5
        elif rsi <= 55:
            score += 1.5
        elif rsi <= 70:
            score += 0.8
        else:
            score += 0.0

        # MACD (0-3)
        if macd_cross == "bullish_crossover":
            score += 3.0
        elif macd_cross == "neutral":
            score += 1.5
        else:
            score += 0.5

        # Trend (0-2)
        if trend == "uptrend":
            score += 2.0
        elif trend == "sideways":
            score += 1.0
        else:
            score += 0.0

        # Support/resistance proximity (0-2)
        if support > 0 and resistance > 0 and price > 0:
            range_total = resistance - support
            if range_total > 0:
                pos_in_range = (price - support) / range_total
                # Closer to support = more upside = higher score
                sr_score = (1 - pos_in_range) * 2
                score += max(0, min(2, sr_score))
            else:
                score += 1.0
        else:
            score += 1.0

        return round(min(10.0, max(0.0, score)), 1)

    # ── Summary builder ────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        trend: str,
        rsi: float,
        rsi_label: str,
        macd_cross: str,
        support: float,
        resistance: float,
        price: float,
        pattern: str,
    ) -> str:
        parts: List[str] = []
        parts.append(f"Trend: {trend}.")
        parts.append(f"RSI {rsi:.0f} ({rsi_label}).")

        if macd_cross == "bullish_crossover":
            parts.append("MACD bullish crossover detected.")
        elif macd_cross == "bearish":
            parts.append("MACD bearish signal.")
        else:
            parts.append("MACD neutral.")

        if support > 0:
            parts.append(f"Support ~${support:.6g}, Resistance ~${resistance:.6g}.")

        if pattern != "None":
            parts.append(f"Pattern: {pattern}.")

        return " ".join(parts)
