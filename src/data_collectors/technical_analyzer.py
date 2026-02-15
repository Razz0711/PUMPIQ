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

    # ── Advanced Market Analysis Fields ──
    market_regime: str = "unknown"        # "trending" | "ranging" | "unstable"
    volatility_state: str = "normal"      # "expanding" | "contracting" | "normal"
    breakout_quality: str = "none"        # "confirmed" | "weak" | "none"
    abnormal_volume: bool = False         # possible whale activity
    volume_anomaly_score: float = 0.0     # 0-10 how anomalous the volume is
    short_term_trend: str = "sideways"    # EMA-10 vs EMA-20
    long_term_trend: str = "sideways"     # EMA-50 vs EMA-200 (or best available)
    trend_consistency: float = 0.0        # 0-1, fraction of candles aligned with trend
    liquidity_pressure: str = "neutral"   # "buying" | "selling" | "neutral"
    bollinger_width: float = 0.0          # normalized Bollinger Band width
    atr_ratio: float = 0.0               # ATR / price ratio (volatility measure)


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
        volumes: Optional[List[List[float]]] = None,
    ) -> TechnicalResult:
        """
        Run full TA suite on a price series.

        Parameters
        ----------
        prices : list of [timestamp_ms, price]
            As returned by CoinGecko ``/coins/{id}/market_chart``.
        current_price : float
            Latest price (for support/resistance context).
        volumes : list of [timestamp_ms, volume], optional
            Volume data for whale/anomaly detection.
        """
        if len(prices) < 30:
            logger.warning("Too few data points (%d) for reliable TA", len(prices))
            return TechnicalResult(summary="Insufficient price data for analysis")

        closes = [p[1] for p in prices]
        if current_price <= 0:
            current_price = closes[-1]

        # Extract volume data if available
        vol_data = [v[1] for v in volumes] if volumes and len(volumes) >= 30 else None

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

        # --- Advanced Analysis ---
        short_term_trend = self._detect_short_term_trend(closes)
        long_term_trend = self._detect_long_term_trend(closes)
        trend_consistency = self._compute_trend_consistency(closes, trend)
        market_regime = self._classify_market_regime(closes, rsi, trend, trend_consistency)
        volatility_state, bollinger_width = self._classify_volatility(closes)
        atr_ratio = self._compute_atr_ratio(closes)
        breakout_quality = self._detect_breakout_quality(
            closes, current_price, support, resistance, vol_data
        )
        abnormal_volume, volume_anomaly_score = self._detect_volume_anomaly(vol_data)
        liquidity_pressure = self._detect_liquidity_pressure(closes, vol_data)

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
            # Advanced fields
            market_regime=market_regime,
            volatility_state=volatility_state,
            breakout_quality=breakout_quality,
            abnormal_volume=abnormal_volume,
            volume_anomaly_score=round(volume_anomaly_score, 1),
            short_term_trend=short_term_trend,
            long_term_trend=long_term_trend,
            trend_consistency=round(trend_consistency, 2),
            liquidity_pressure=liquidity_pressure,
            bollinger_width=round(bollinger_width, 4),
            atr_ratio=round(atr_ratio, 4),
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

    # ══════════════════════════════════════════════════════════════
    # Advanced Market Analysis Methods
    # ══════════════════════════════════════════════════════════════

    def _detect_short_term_trend(self, closes: List[float]) -> str:
        """Short-term trend using EMA-10 vs EMA-20."""
        if len(closes) < 20:
            return "sideways"
        ema10 = self._ema(closes, 10)
        ema20 = self._ema(closes, 20)
        last = closes[-1]
        if ema10 > ema20 and last > ema10:
            return "uptrend"
        if ema10 < ema20 and last < ema10:
            return "downtrend"
        return "sideways"

    def _detect_long_term_trend(self, closes: List[float]) -> str:
        """Long-term trend using EMA-50 vs EMA-100 (or best available)."""
        if len(closes) < 50:
            return "sideways"
        ema50 = self._ema(closes, 50)
        period_long = min(100, len(closes))
        ema_long = self._ema(closes, period_long)
        last = closes[-1]
        if ema50 > ema_long and last > ema50:
            return "uptrend"
        if ema50 < ema_long and last < ema50:
            return "downtrend"
        return "sideways"

    @staticmethod
    def _compute_trend_consistency(closes: List[float], trend: str) -> float:
        """
        Measure how consistently candles align with the detected trend.
        Returns 0-1 (fraction of recent candles moving in trend direction).
        """
        if len(closes) < 10:
            return 0.5
        recent = closes[-20:] if len(closes) >= 20 else closes
        total_moves = len(recent) - 1
        if total_moves == 0:
            return 0.5
        aligned = 0
        for i in range(1, len(recent)):
            diff = recent[i] - recent[i - 1]
            if trend == "uptrend" and diff > 0:
                aligned += 1
            elif trend == "downtrend" and diff < 0:
                aligned += 1
            elif trend == "sideways" and abs(diff / max(recent[i - 1], 1e-9)) < 0.005:
                aligned += 1
        return aligned / total_moves

    @staticmethod
    def _classify_market_regime(
        closes: List[float], rsi: float, trend: str, consistency: float
    ) -> str:
        """
        Classify market as trending / ranging / unstable.

        - Trending: clear direction + high consistency (>0.6)
        - Ranging: sideways trend + moderate RSI (40-60)
        - Unstable: wild swings, low consistency, extreme RSI
        """
        if consistency < 0.35:
            return "unstable"
        if trend in ("uptrend", "downtrend") and consistency > 0.55:
            return "trending"
        if trend == "sideways" and 35 < rsi < 65:
            return "ranging"
        # Check recent price range vs average
        if len(closes) >= 20:
            recent = closes[-20:]
            pct_range = (max(recent) - min(recent)) / max(max(recent), 1e-9) * 100
            if pct_range < 8:
                return "ranging"
            if pct_range > 25:
                return "unstable"
        return "trending" if consistency > 0.5 else "ranging"

    def _classify_volatility(self, closes: List[float]) -> Tuple[str, float]:
        """
        Classify volatility as expanding / contracting / normal
        using Bollinger Band width (20-period, 2 std dev).

        Returns (state, normalized_width).
        """
        if len(closes) < 20:
            return "normal", 0.0

        period = 20
        sma = sum(closes[-period:]) / period
        variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
        std = math.sqrt(variance)
        upper = sma + 2 * std
        lower = sma - 2 * std
        width = (upper - lower) / max(sma, 1e-9)

        # Compare current width to historical width (last 50 candles)
        if len(closes) >= 50:
            hist_widths = []
            for i in range(max(20, len(closes) - 50), len(closes) - period + 1):
                segment = closes[i - period:i]
                if len(segment) < period:
                    continue
                s = sum(segment) / period
                v = sum((c - s) ** 2 for c in segment) / period
                sd = math.sqrt(v)
                hist_widths.append((s + 2 * sd - (s - 2 * sd)) / max(s, 1e-9))
            if hist_widths:
                avg_width = sum(hist_widths) / len(hist_widths)
                if width > avg_width * 1.3:
                    return "expanding", width
                if width < avg_width * 0.7:
                    return "contracting", width

        return "normal", width

    @staticmethod
    def _compute_atr_ratio(closes: List[float], period: int = 14) -> float:
        """Average True Range as ratio of current price (volatility measure)."""
        if len(closes) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            high_low = abs(closes[i] - closes[i - 1])  # simplified TR
            trs.append(high_low)
        recent_trs = trs[-period:]
        atr = sum(recent_trs) / len(recent_trs)
        return atr / max(closes[-1], 1e-9)

    def _detect_breakout_quality(
        self,
        closes: List[float],
        current_price: float,
        support: float,
        resistance: float,
        vol_data: Optional[List[float]] = None,
    ) -> str:
        """
        Classify breakout as confirmed / weak / none.

        Confirmed: price closes above resistance with >1.5x avg volume
        Weak: price near resistance but volume is normal or below average
        """
        if resistance <= 0 or support <= 0:
            return "none"

        pct_above_resistance = (current_price - resistance) / max(resistance, 1e-9) * 100

        # Check if price is near or above resistance
        if pct_above_resistance > 1.0:
            # Price above resistance — check volume confirmation
            if vol_data and len(vol_data) >= 20:
                avg_vol = sum(vol_data[-20:]) / 20
                recent_vol = sum(vol_data[-3:]) / 3 if len(vol_data) >= 3 else vol_data[-1]
                if recent_vol > avg_vol * 1.5:
                    return "confirmed"
                return "weak"
            return "weak"  # No volume data → can't fully confirm

        pct_below_support = (support - current_price) / max(support, 1e-9) * 100
        if pct_below_support > 1.0:
            # Breakdown below support
            if vol_data and len(vol_data) >= 20:
                avg_vol = sum(vol_data[-20:]) / 20
                recent_vol = sum(vol_data[-3:]) / 3 if len(vol_data) >= 3 else vol_data[-1]
                if recent_vol > avg_vol * 1.5:
                    return "confirmed"
                return "weak"
            return "weak"

        return "none"

    @staticmethod
    def _detect_volume_anomaly(
        vol_data: Optional[List[float]],
    ) -> Tuple[bool, float]:
        """
        Detect abnormal volume (possible whale activity).

        Returns (is_abnormal, anomaly_score 0-10).
        Uses z-score of recent volume vs historical average.
        """
        if not vol_data or len(vol_data) < 20:
            return False, 0.0

        avg_vol = sum(vol_data[-20:]) / 20
        std_vol = math.sqrt(
            sum((v - avg_vol) ** 2 for v in vol_data[-20:]) / 20
        )
        if std_vol == 0:
            return False, 0.0

        # Check last 3 candles for spikes
        recent_avg = sum(vol_data[-3:]) / 3
        z_score = (recent_avg - avg_vol) / std_vol

        # Map z-score to 0-10 anomaly score
        anomaly_score = min(10.0, max(0.0, z_score * 2.5))
        is_abnormal = z_score > 2.0  # >2 sigma = abnormal

        return is_abnormal, anomaly_score

    @staticmethod
    def _detect_liquidity_pressure(
        closes: List[float],
        vol_data: Optional[List[float]] = None,
    ) -> str:
        """
        Infer buying vs selling pressure from price action and volume.

        If price rising + volume rising → buying pressure
        If price falling + volume rising → selling pressure
        Otherwise → neutral
        """
        if len(closes) < 5:
            return "neutral"

        recent_close = closes[-5:]
        price_trend = recent_close[-1] - recent_close[0]
        price_pct = price_trend / max(abs(recent_close[0]), 1e-9) * 100

        if vol_data and len(vol_data) >= 10:
            vol_recent = sum(vol_data[-5:]) / 5
            vol_earlier = sum(vol_data[-10:-5]) / 5 if len(vol_data) >= 10 else vol_recent
            vol_rising = vol_recent > vol_earlier * 1.2

            if price_pct > 1 and vol_rising:
                return "buying"
            if price_pct < -1 and vol_rising:
                return "selling"

        # Fallback: just price action
        if price_pct > 2:
            return "buying"
        if price_pct < -2:
            return "selling"
        return "neutral"
