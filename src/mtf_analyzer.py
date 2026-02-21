"""
Multi-Timeframe Confluence Engine
====================================
Algorithm 3 – Analyses 4 timeframes before issuing any recommendation.

Timeframes:
    1D  → Trend direction  (EMA50 vs EMA200 – "Golden/Death Cross")
    4H  → Setup confirmation  (MACD + Bollinger squeeze detection)
    1H  → Entry signal  (RSI divergence + EMA9/21 crossover)
    15M → Timing  (CVD proxy + support/resistance)

Confluence Scoring:
    4/4 agree → 1.00
    3/4 agree → 0.75
    2/4 agree → 0.50
    1 or fewer → 0.25 → BLOCK recommendation

HARD RULE: Never recommend if 1D trend is bearish.

⚠ HISTORICAL SIMULATION — not guaranteed future performance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.ta_utils import (
    ema_series as _ta_ema_series,
    rsi_series as _ta_rsi_series,
    macd_series as _ta_macd_series,
    bollinger_series as _ta_bollinger_series,
)

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


# ── Result dataclass ──────────────────────────────────────────────

@dataclass
class TimeframeSignal:
    """Signal from a single timeframe analysis."""
    timeframe: str          # "1D", "4H", "1H", "15M"
    direction: str          # "bullish", "bearish", "neutral"
    confidence: float       # 0-1
    signals: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfluenceResult:
    """Full multi-timeframe confluence analysis."""
    coin_id: str
    symbol: str

    confluence_score: float = 0.25
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    daily_trend: str = "neutral"      # overall 1D bias
    is_blocked: bool = False          # True if confluence < 0.50 or 1D bearish
    block_reason: str = ""

    recommended_entry_zone: Dict[str, float] = field(default_factory=dict)
    invalidation_level: float = 0.0

    timeframe_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timeframe_signals: List[TimeframeSignal] = field(default_factory=list)

    error: str = ""
    is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coin_id": self.coin_id,
            "symbol": self.symbol,
            "confluence_score": round(self.confluence_score, 2),
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "daily_trend": self.daily_trend,
            "is_blocked": self.is_blocked,
            "block_reason": self.block_reason,
            "recommended_entry_zone": {
                k: round(v, 8) for k, v in self.recommended_entry_zone.items()
            },
            "invalidation_level": round(self.invalidation_level, 8),
            "timeframe_breakdown": self.timeframe_breakdown,
            "error": self.error,
            "is_fallback": self.is_fallback,
            "label": "HISTORICAL SIMULATION — not guaranteed future performance",
        }


# ── Indicator helpers (delegates to shared ta_utils) ────────────────────

def _ema_series(values: List[float], period: int) -> List[float]:
    return _ta_ema_series(values, period)


def _sma(values: List[float], period: int) -> List[float]:
    n = len(values)
    out = [0.0] * n
    for i in range(period - 1, n):
        out[i] = sum(values[i - period + 1: i + 1]) / period
    return out


def _rsi(closes: List[float], period: int = 14) -> List[float]:
    return _ta_rsi_series(closes, period)


def _bollinger(closes: List[float], period: int = 20,
               std_mult: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    return _ta_bollinger_series(closes, period, std_mult)


def _macd(closes: List[float], fast: int = 12, slow: int = 26,
          sig: int = 9) -> Tuple[List[float], List[float], List[float]]:
    return _ta_macd_series(closes, fast, slow, sig)


def _atr(highs: List[float], lows: List[float], closes: List[float],
         period: int = 14) -> List[float]:
    n = len(closes)
    tr = [0.0] * n
    atr_out = [0.0] * n
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    if n > period:
        atr_out[period] = sum(tr[1:period + 1]) / period
        for i in range(period + 1, n):
            atr_out[i] = (atr_out[i - 1] * (period - 1) + tr[i]) / period
    return atr_out


def _support_resistance(closes: List[float], window: int = 20) -> Tuple[float, float]:
    """Very simple: recent low = support, recent high = resistance."""
    if len(closes) < window:
        w = closes
    else:
        w = closes[-window:]
    return min(w), max(w)


# ── Simulated multi-timeframe from daily data ─────────────────────
# CoinGecko free API only gives daily data for 180-day requests.
# We *simulate* lower timeframes from daily candles using different
# lookback windows (this is an approximation, not real 4H/1H/15M data).

def _resample_daily_as_timeframe(
    closes: List[float], volumes: List[float], timeframe: str,
) -> Tuple[List[float], List[float]]:
    """
    Partition daily data to approximate a lower timeframe's perspective.
    
    - 1D: full data
    - 4H: last 60 days   (simulates shorter-term setup)
    - 1H: last 30 days   (simulates entry window)
    - 15M: last 14 days  (simulates timing window)
    """
    slices = {
        "1D": (0, len(closes)),
        "4H": (max(0, len(closes) - 60), len(closes)),
        "1H": (max(0, len(closes) - 30), len(closes)),
        "15M": (max(0, len(closes) - 14), len(closes)),
    }
    start, end = slices.get(timeframe, (0, len(closes)))
    return closes[start:end], volumes[start:end]


# ══════════════════════════════════════════════════════════════════
# Multi-Timeframe Analyzer
# ══════════════════════════════════════════════════════════════════

class MTFAnalyzer:
    """
    Analyses 4 simulated timeframes and produces a confluence score.

    Usage::

        mtf = MTFAnalyzer()
        result = await mtf.analyze("bitcoin", "Bitcoin", "BTC", cg_collector)
    """

    async def analyze(
        self,
        coin_id: str,
        coin_name: str,
        symbol: str,
        cg_collector,
        days: int = 180,
    ) -> ConfluenceResult:
        result = ConfluenceResult(coin_id=coin_id, symbol=symbol.upper())

        # ── fetch data ──
        try:
            history = await cg_collector.get_price_history(coin_id, days=max(days, 180))
        except Exception as exc:
            result.error = f"Data fetch failed: {exc}"
            result.is_fallback = True
            result.is_blocked = True
            result.block_reason = "Data unavailable"
            return result

        if not history or not history.prices or len(history.prices) < 60:
            result.error = "Insufficient data"
            result.is_fallback = True
            result.is_blocked = True
            result.block_reason = "Insufficient historical data"
            return result

        closes = [p[1] for p in history.prices]
        volumes = [v[1] for v in history.volumes] if history.volumes else [0.0] * len(closes)
        if len(volumes) < len(closes):
            volumes.extend([0.0] * (len(closes) - len(volumes)))

        # ── Analyze each timeframe ──
        signals: List[TimeframeSignal] = []

        # 1D Trend
        sig_1d = self._analyze_1d(closes, volumes)
        signals.append(sig_1d)

        # 4H Setup
        c_4h, v_4h = _resample_daily_as_timeframe(closes, volumes, "4H")
        sig_4h = self._analyze_4h(c_4h, v_4h)
        signals.append(sig_4h)

        # 1H Entry
        c_1h, v_1h = _resample_daily_as_timeframe(closes, volumes, "1H")
        sig_1h = self._analyze_1h(c_1h, v_1h)
        signals.append(sig_1h)

        # 15M Timing
        c_15m, v_15m = _resample_daily_as_timeframe(closes, volumes, "15M")
        sig_15m = self._analyze_15m(c_15m, v_15m, closes)
        signals.append(sig_15m)

        result.timeframe_signals = signals

        # ── Compute confluence ──
        bullish = sum(1 for s in signals if s.direction == "bullish")
        bearish = sum(1 for s in signals if s.direction == "bearish")
        neutral = sum(1 for s in signals if s.direction == "neutral")

        result.bullish_count = bullish
        result.bearish_count = bearish
        result.neutral_count = neutral
        result.daily_trend = sig_1d.direction

        if bullish == 4:
            result.confluence_score = 1.0
        elif bullish >= 3:
            result.confluence_score = 0.75
        elif bullish >= 2:
            result.confluence_score = 0.50
        else:
            result.confluence_score = 0.25

        # ── HARD RULE: block if 1D bearish ──
        if sig_1d.direction == "bearish":
            result.is_blocked = True
            result.block_reason = "1D trend is bearish — HARD BLOCK regardless of lower timeframes"
            result.confluence_score = min(result.confluence_score, 0.25)

        # Block if confluence too low
        if result.confluence_score < 0.50:
            result.is_blocked = True
            if not result.block_reason:
                result.block_reason = (
                    f"Confluence {result.confluence_score:.2f} < 0.50 — "
                    f"insufficient timeframe agreement ({bullish}/4 bullish)"
                )

        # ── Entry zone & invalidation ──
        support, resistance = _support_resistance(closes)
        current = closes[-1]
        result.recommended_entry_zone = {
            "entry_low": round(support, 8),
            "entry_high": round(current * 1.01, 8),
        }
        atr_vals = _atr(closes, closes, closes, 14)  # approximate (no true high/low)
        latest_atr = atr_vals[-1] if atr_vals else current * 0.02
        result.invalidation_level = round(support - latest_atr, 8)

        # Timeframe breakdown for API
        result.timeframe_breakdown = {
            s.timeframe: {
                "direction": s.direction,
                "confidence": round(s.confidence, 2),
                "signals": s.signals,
                "details": s.details,
            }
            for s in signals
        }

        logger.info(
            "MTF [%s]: confluence=%.2f  1D=%s  4H=%s  1H=%s  15M=%s  blocked=%s",
            symbol, result.confluence_score,
            sig_1d.direction, sig_4h.direction,
            sig_1h.direction, sig_15m.direction,
            result.is_blocked,
        )

        return result

    # ── 1D: Trend Direction ──────────────────────────────────────

    def _analyze_1d(self, closes: List[float],
                    volumes: List[float]) -> TimeframeSignal:
        """EMA50 vs EMA200 — Golden Cross / Death Cross."""
        sig = TimeframeSignal(timeframe="1D", direction="neutral", confidence=0.5)

        if len(closes) < 200:
            # Not enough data for EMA200 — use EMA20 vs EMA50
            if len(closes) < 50:
                sig.signals.append("Insufficient data for 1D trend")
                return sig
            ema_short = _ema_series(closes, 20)
            ema_long = _ema_series(closes, 50)
            short_label, long_label = "EMA20", "EMA50"
        else:
            ema_short = _ema_series(closes, 50)
            ema_long = _ema_series(closes, 200)
            short_label, long_label = "EMA50", "EMA200"

        current = closes[-1]
        ema_s = ema_short[-1]
        ema_l = ema_long[-1]

        if ema_s > ema_l and current > ema_s:
            sig.direction = "bullish"
            sig.confidence = 0.8
            sig.signals.append(f"{short_label} > {long_label} (Golden Cross zone)")
            sig.signals.append(f"Price above {short_label}")
        elif ema_s < ema_l and current < ema_s:
            sig.direction = "bearish"
            sig.confidence = 0.8
            sig.signals.append(f"{short_label} < {long_label} (Death Cross zone)")
            sig.signals.append(f"Price below {short_label}")
        elif ema_s > ema_l:
            sig.direction = "bullish"
            sig.confidence = 0.6
            sig.signals.append(f"{short_label} > {long_label} but price pulling back")
        elif current > ema_l:
            sig.direction = "neutral"
            sig.confidence = 0.4
            sig.signals.append("Price above long EMA but short EMA still below")
        else:
            sig.direction = "bearish"
            sig.confidence = 0.7
            sig.signals.append(f"Price below both {short_label} and {long_label}")

        sig.details = {
            f"{short_label}": round(ema_s, 8),
            f"{long_label}": round(ema_l, 8),
            "price": round(current, 8),
        }
        return sig

    # ── 4H: Setup Confirmation ────────────────────────────────────

    def _analyze_4h(self, closes: List[float],
                    volumes: List[float]) -> TimeframeSignal:
        """MACD + Bollinger squeeze → setup confirmation."""
        sig = TimeframeSignal(timeframe="4H", direction="neutral", confidence=0.5)

        if len(closes) < 30:
            sig.signals.append("Insufficient data for 4H analysis")
            return sig

        macd_l, signal_l, hist = _macd(closes)
        bb_u, bb_m, bb_l = _bollinger(closes)

        # Bollinger squeeze detection
        if bb_u[-1] > 0 and bb_l[-1] > 0:
            bandwidth = (bb_u[-1] - bb_l[-1]) / bb_m[-1] if bb_m[-1] > 0 else 0
            avg_bw = 0
            count = 0
            for i in range(max(0, len(bb_u) - 20), len(bb_u)):
                if bb_m[i] > 0:
                    avg_bw += (bb_u[i] - bb_l[i]) / bb_m[i]
                    count += 1
            if count > 0:
                avg_bw /= count
            squeeze = bandwidth < avg_bw * 0.75
            sig.details["bollinger_squeeze"] = squeeze
            sig.details["bandwidth"] = round(bandwidth, 4)
            if squeeze:
                sig.signals.append("Bollinger Band squeeze — breakout imminent")

        # MACD direction
        if len(hist) > 1:
            if hist[-1] > 0 and hist[-1] > hist[-2]:
                sig.signals.append("MACD histogram expanding bullish")
                sig.direction = "bullish"
                sig.confidence = 0.7
            elif hist[-1] > 0:
                sig.signals.append("MACD histogram positive but contracting")
                sig.direction = "bullish"
                sig.confidence = 0.55
            elif hist[-1] < 0 and hist[-1] < hist[-2]:
                sig.signals.append("MACD histogram expanding bearish")
                sig.direction = "bearish"
                sig.confidence = 0.7
            elif hist[-1] < 0:
                sig.signals.append("MACD histogram negative but contracting")
                sig.direction = "neutral"
                sig.confidence = 0.5

        # Check for MACD crossover
        if len(macd_l) > 1 and len(signal_l) > 1:
            if macd_l[-1] > signal_l[-1] and macd_l[-2] <= signal_l[-2]:
                sig.signals.append("MACD bullish crossover (confirming)")
                sig.direction = "bullish"
                sig.confidence = max(sig.confidence, 0.75)
            elif macd_l[-1] < signal_l[-1] and macd_l[-2] >= signal_l[-2]:
                sig.signals.append("MACD bearish crossover")
                sig.direction = "bearish"
                sig.confidence = max(sig.confidence, 0.7)

        sig.details["macd"] = round(macd_l[-1], 8) if macd_l else 0
        sig.details["macd_signal"] = round(signal_l[-1], 8) if signal_l else 0
        return sig

    # ── 1H: Entry Signal ──────────────────────────────────────────

    def _analyze_1h(self, closes: List[float],
                    volumes: List[float]) -> TimeframeSignal:
        """RSI divergence + EMA9/21 crossover → entry trigger."""
        sig = TimeframeSignal(timeframe="1H", direction="neutral", confidence=0.5)

        if len(closes) < 25:
            sig.signals.append("Insufficient data for 1H analysis")
            return sig

        rsi_arr = _rsi(closes)
        ema9 = _ema_series(closes, 9)
        ema21 = _ema_series(closes, 21)

        # EMA9/21 crossover
        if len(ema9) > 1 and len(ema21) > 1:
            if ema9[-1] > ema21[-1] and ema9[-2] <= ema21[-2]:
                sig.signals.append("EMA9/21 bullish crossover — entry trigger")
                sig.direction = "bullish"
                sig.confidence = 0.75
            elif ema9[-1] < ema21[-1] and ema9[-2] >= ema21[-2]:
                sig.signals.append("EMA9/21 bearish crossover")
                sig.direction = "bearish"
                sig.confidence = 0.7
            elif ema9[-1] > ema21[-1]:
                sig.signals.append("EMA9 > EMA21 — bullish bias")
                sig.direction = "bullish"
                sig.confidence = 0.6
            else:
                sig.signals.append("EMA9 < EMA21 — bearish bias")
                sig.direction = "bearish"
                sig.confidence = 0.55

        # RSI analysis
        current_rsi = rsi_arr[-1]
        sig.details["rsi"] = round(current_rsi, 1)

        if current_rsi < 30:
            sig.signals.append(f"RSI oversold ({current_rsi:.0f}) — potential reversal")
            if sig.direction != "bearish":
                sig.direction = "bullish"
                sig.confidence = max(sig.confidence, 0.7)
        elif current_rsi > 70:
            sig.signals.append(f"RSI overbought ({current_rsi:.0f}) — caution")
            sig.confidence = min(sig.confidence, 0.4)
        elif 40 <= current_rsi <= 60:
            sig.signals.append(f"RSI neutral ({current_rsi:.0f})")

        # Simple RSI divergence check (price lower low, RSI higher low)
        if len(closes) >= 10 and len(rsi_arr) >= 10:
            price_low_recent = min(closes[-5:])
            price_low_prev = min(closes[-10:-5])
            rsi_low_recent = min(rsi_arr[-5:])
            rsi_low_prev = min(rsi_arr[-10:-5])

            if price_low_recent < price_low_prev and rsi_low_recent > rsi_low_prev:
                sig.signals.append("Bullish RSI divergence detected")
                sig.direction = "bullish"
                sig.confidence = max(sig.confidence, 0.75)
            elif price_low_recent > price_low_prev and rsi_low_recent < rsi_low_prev:
                sig.signals.append("Bearish RSI divergence detected")
                sig.direction = "bearish"
                sig.confidence = max(sig.confidence, 0.7)

        sig.details["ema9"] = round(ema9[-1], 8) if ema9 else 0
        sig.details["ema21"] = round(ema21[-1], 8) if ema21 else 0
        return sig

    # ── 15M: Timing ───────────────────────────────────────────────

    def _analyze_15m(self, closes: List[float], volumes: List[float],
                     full_closes: List[float]) -> TimeframeSignal:
        """CVD proxy + support/resistance → timing."""
        sig = TimeframeSignal(timeframe="15M", direction="neutral", confidence=0.5)

        if len(closes) < 5:
            sig.signals.append("Insufficient data for 15M timing")
            return sig

        # CVD (Cumulative Volume Delta) proxy
        # Positive CVD = buying pressure, Negative = selling pressure
        cvd = 0.0
        for i in range(1, len(closes)):
            direction = 1.0 if closes[i] >= closes[i-1] else -1.0
            cvd += direction * (volumes[i] if i < len(volumes) else 0)

        if cvd > 0:
            sig.signals.append(f"CVD positive — net buying pressure")
            sig.direction = "bullish"
            sig.confidence = 0.6
        elif cvd < 0:
            sig.signals.append(f"CVD negative — net selling pressure")
            sig.direction = "bearish"
            sig.confidence = 0.6

        # Support / resistance from full dataset
        support, resistance = _support_resistance(full_closes, window=30)
        current = closes[-1]
        range_pct = (resistance - support) / support * 100 if support > 0 else 0

        sig.details["support"] = round(support, 8)
        sig.details["resistance"] = round(resistance, 8)
        sig.details["range_pct"] = round(range_pct, 2)
        sig.details["cvd_direction"] = "positive" if cvd > 0 else "negative"

        # Near support = potential bounce (bullish timing)
        dist_to_support = (current - support) / support * 100 if support > 0 else 50
        dist_to_resist = (resistance - current) / current * 100 if current > 0 else 50

        if dist_to_support < 2:
            sig.signals.append(f"Price near support ({dist_to_support:.1f}% above)")
            if sig.direction != "bearish":
                sig.direction = "bullish"
                sig.confidence = max(sig.confidence, 0.65)
        elif dist_to_resist < 2:
            sig.signals.append(f"Price near resistance ({dist_to_resist:.1f}% below)")
            sig.confidence = min(sig.confidence, 0.4)

        return sig


# ── Module-level singleton ────────────────────────────────────────

_mtf_analyzer: Optional[MTFAnalyzer] = None


def get_mtf_analyzer() -> MTFAnalyzer:
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MTFAnalyzer()
    return _mtf_analyzer
