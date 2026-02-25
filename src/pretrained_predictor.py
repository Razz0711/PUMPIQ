"""
Pretrained Model Predictor Bridge
====================================
Computes the 38 features expected by the pretrained XGBoost models
(trained in ``ml/NexYpher model trainer · py``) from live CoinGecko
price-history + market-data, then returns predictions.

This bridges the gap between the pretrained 38-feature models (stored in
``ml/models/``) and the runtime trading engine, which previously only used
the 8-feature MLBacktester models.

Features: 26 DB-equivalent + 5 market + 7 engineered = 38 total
Source:   CoinGecko ``/coins/{id}/market_chart`` (closes + volumes)
          + ``CoinMarketData`` snapshot for price changes & ATH
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ta_utils import (
    ema_series,
    rsi_series as _ta_rsi_series,
    macd_series as _ta_macd_series,
    bollinger_series as _ta_bollinger_series,
    bollinger_position as _ta_bollinger_position,
)

logger = logging.getLogger(__name__)

# ── Cache settings ──────────────────────────────────────────────
_PREDICTION_CACHE_TTL = 900  # 15 minutes
_prediction_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# ── Models (lazy-loaded singleton) ──────────────────────────────
_models_loaded = False
_model_24h = None
_model_7d = None
_model_dir = None
_label_encoder = None
_metadata: Dict[str, Any] = {}
_FEATURE_COLUMNS: List[str] = []


def _load_models():
    """Lazy-load the pretrained 38-feature models from ml/models/."""
    global _models_loaded, _model_24h, _model_7d, _model_dir
    global _label_encoder, _metadata, _FEATURE_COLUMNS

    if _models_loaded:
        return _model_24h is not None

    _models_loaded = True

    try:
        import joblib
        import json

        models_dir = Path(__file__).parent.parent / "ml" / "models"
        meta_path = models_dir / "model_metadata.json"

        if not meta_path.exists():
            logger.warning("Pretrained model metadata not found at %s", meta_path)
            return False

        with open(meta_path) as f:
            _metadata = json.load(f)

        paths = {
            "24h": models_dir / "model_24h_1d_latest.pkl",
            "7d":  models_dir / "model_7d_1d_latest.pkl",
            "dir": models_dir / "model_dir_1d_latest.pkl",
            "le":  models_dir / "label_encoder_latest.pkl",
        }

        for name, path in paths.items():
            if not path.exists():
                logger.warning("Pretrained model file missing: %s", path)
                return False

        _model_24h = joblib.load(paths["24h"])
        _model_7d = joblib.load(paths["7d"])
        _model_dir = joblib.load(paths["dir"])
        _label_encoder = joblib.load(paths["le"])

        _FEATURE_COLUMNS = _metadata.get("feature_columns", [])
        if len(_FEATURE_COLUMNS) != _metadata.get("n_features", 38):
            logger.warning(
                "Feature column count mismatch: %d vs expected %d",
                len(_FEATURE_COLUMNS), _metadata.get("n_features", 38),
            )
            return False

        logger.info(
            "Pretrained 38-feature models loaded (v%s): 24h=%.1f%% 7d=%.1f%%",
            _metadata.get("version", "?"),
            _metadata.get("model_24h", {}).get("cv_mean", 0) * 100,
            _metadata.get("model_7d", {}).get("cv_mean", 0) * 100,
        )
        return True

    except Exception as e:
        logger.warning("Failed to load pretrained models: %s", e)
        return False


# ── Feature computation from CoinGecko data ────────────────────

def _sma_series(values: List[float], period: int) -> List[float]:
    """Simple moving average series."""
    n = len(values)
    result = list(values)
    for i in range(period - 1, n):
        result[i] = sum(values[i - period + 1: i + 1]) / period
    return result


def _atr_series(closes: List[float], period: int = 14) -> List[float]:
    """Average True Range computed from close prices only (no high/low).
    Uses |close[i] - close[i-1]| as the true range proxy."""
    n = len(closes)
    tr = [0.0] * n
    for i in range(1, n):
        tr[i] = abs(closes[i] - closes[i - 1])

    atr = [0.0] * n
    if n > period:
        atr[period] = sum(tr[1:period + 1]) / period
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _volatility_series(closes: List[float], period: int) -> List[float]:
    """Rolling standard deviation of returns."""
    n = len(closes)
    vol = [0.0] * n
    returns = [0.0] * n
    for i in range(1, n):
        if closes[i - 1] > 0:
            returns[i] = (closes[i] - closes[i - 1]) / closes[i - 1]
    for i in range(period, n):
        window = returns[i - period + 1: i + 1]
        mean = sum(window) / len(window)
        var = sum((r - mean) ** 2 for r in window) / len(window)
        vol[i] = math.sqrt(var)
    return vol


def _momentum_series(closes: List[float], period: int) -> List[float]:
    """Percentage change over N periods."""
    n = len(closes)
    mom = [0.0] * n
    for i in range(period, n):
        if closes[i - period] > 0:
            mom[i] = (closes[i] / closes[i - period] - 1.0) * 100.0
    return mom


def _rate_of_change(closes: List[float], period: int = 14) -> List[float]:
    """Rate of change: (close / close_N_ago - 1) * 100."""
    return _momentum_series(closes, period)


def _volume_ratio_series(volumes: List[float], period: int = 20) -> List[float]:
    """Volume / 20-day MA of volume."""
    n = len(volumes)
    ratio = [1.0] * n
    for i in range(period, n):
        ma = sum(volumes[i - period: i]) / period
        ratio[i] = volumes[i] / ma if ma > 0 else 1.0
    return ratio


def _volume_spike(volumes: List[float], period: int = 20, threshold: float = 2.0) -> List[float]:
    """1 if volume > threshold * MA, else 0."""
    vr = _volume_ratio_series(volumes, period)
    return [1.0 if v > threshold else 0.0 for v in vr]


def _support_resistance(closes: List[float], window: int = 20):
    """Estimate support/resistance as rolling min/max."""
    n = len(closes)
    support = list(closes)
    resist = list(closes)
    for i in range(window, n):
        w = closes[i - window: i]
        support[i] = min(w)
        resist[i] = max(w)
    return support, resist


def compute_38_features(
    closes: List[float],
    volumes: List[float],
    coin_data: Any = None,
    fear_greed_value: float = 50.0,
) -> Dict[str, float]:
    """
    Compute all 38 features from CoinGecko price/volume history.

    Parameters
    ----------
    closes : list of daily close prices (oldest → newest, ≥100 points)
    volumes : list of daily volumes (same length)
    coin_data : CoinMarketData snapshot (for price changes, ATH, vol/mcap)
    fear_greed_value : current Fear & Greed index (0-100), default 50

    Returns
    -------
    dict with keys matching FEATURE_COLUMNS from the trainer
    """
    n = len(closes)
    i = n - 1  # latest index

    # ── RSI at multiple periods ──
    rsi_7 = _ta_rsi_series(closes, 7)
    rsi_14 = _ta_rsi_series(closes, 14)
    rsi_21 = _ta_rsi_series(closes, 21)

    # ── MACD ──
    macd_line, macd_signal, macd_hist = _ta_macd_series(closes, 12, 26, 9)
    # MACD crossover: 1 if line > signal (bullish cross), -1 if below, 0 at start
    macd_cross = 0.0
    if i >= 1:
        prev_diff = macd_line[i - 1] - macd_signal[i - 1]
        curr_diff = macd_line[i] - macd_signal[i]
        if prev_diff <= 0 and curr_diff > 0:
            macd_cross = 1.0
        elif prev_diff >= 0 and curr_diff < 0:
            macd_cross = -1.0

    # ── Bollinger Bands ──
    bb_upper, bb_mid, bb_lower = _ta_bollinger_series(closes, 20, 2.0)
    bb_width_val = (bb_upper[i] - bb_lower[i]) / bb_mid[i] if bb_mid[i] > 0 else 0.0
    bb_pos = _ta_bollinger_position(closes, 20, 2.0)

    # ── Moving averages ──
    ema_9 = ema_series(closes, 9)
    ema_21 = ema_series(closes, 21)
    ema_50 = ema_series(closes, 50)
    ema_200 = ema_series(closes, 200)
    sma_20 = _sma_series(closes, 20)

    # EMA crossovers (binary: 1 if fast > slow)
    ema_9_21_cross = 1.0 if ema_9[i] > ema_21[i] else 0.0
    ema_50_200_cross = 1.0 if ema_50[i] > ema_200[i] else 0.0
    price_above_ema200 = 1.0 if closes[i] > ema_200[i] else 0.0

    # ── Volume ──
    vol_ratio = _volume_ratio_series(volumes, 20)
    vol_spike = _volume_spike(volumes, 20, 2.0)

    # ── Momentum ──
    mom_5 = _momentum_series(closes, 5)
    mom_10 = _momentum_series(closes, 10)
    mom_30 = _momentum_series(closes, 30)
    roc_14 = _rate_of_change(closes, 14)

    # ── Volatility & ATR ──
    atr_14 = _atr_series(closes, 14)
    vol_10 = _volatility_series(closes, 10)
    vol_30 = _volatility_series(closes, 30)

    # ── Support / Resistance ──
    support, resist = _support_resistance(closes, 20)
    dist_support = (closes[i] - support[i]) / support[i] * 100 if support[i] > 0 else 0.0
    dist_resist = (resist[i] - closes[i]) / closes[i] * 100 if closes[i] > 0 else 0.0

    # ── Market-data features (from CoinMarketData snapshot) ──
    if coin_data:
        price_change_24h = getattr(coin_data, "price_change_pct_24h", 0.0) or 0.0
        price_change_7d = getattr(coin_data, "price_change_pct_7d", 0.0) or 0.0
        price_change_30d = getattr(coin_data, "price_change_pct_30d", 0.0) or 0.0
        mcap = getattr(coin_data, "market_cap", 0.0) or 1.0
        vol_24h = getattr(coin_data, "total_volume_24h", 0.0) or 0.0
        vol_mcap_ratio = vol_24h / mcap if mcap > 0 else 0.0
        ath_chg = getattr(coin_data, "ath_change_pct", -50.0) or -50.0
    else:
        # Fallback: compute from price array
        price_change_24h = mom_5[i] / 5.0 if i >= 5 else 0.0  # rough proxy
        price_change_7d = _momentum_series(closes, 7)[i] if n > 7 else 0.0
        price_change_30d = mom_30[i]
        vol_mcap_ratio = vol_ratio[i] * 0.02  # proxy (same as trainer)
        ath_chg = -50.0

    # ── Engineered features ──
    rsi_avg = (rsi_7[i] + rsi_14[i] + rsi_21[i]) / 3.0
    rsi_14_ma_diff = rsi_14[i] - rsi_avg
    rsi_oversold = 1.0 if rsi_14[i] < 30 else 0.0
    rsi_overbought = 1.0 if rsi_14[i] > 70 else 0.0

    close = closes[i] if closes[i] > 0 else 1.0
    price_vs_sma20 = (close - sma_20[i]) / sma_20[i] * 100 if sma_20[i] > 0 else 0.0
    price_vs_ema50 = (close - ema_50[i]) / ema_50[i] * 100 if ema_50[i] > 0 else 0.0
    price_vs_ema200 = (close - ema_200[i]) / ema_200[i] * 100 if ema_200[i] > 0 else 0.0

    v10 = vol_10[i] if vol_10[i] > 0 else 1e-9
    v30 = vol_30[i] if vol_30[i] > 0 else 1e-9
    vol_ratio_10_30 = v10 / v30

    mom_accel = mom_5[i] - mom_10[i]

    # Trend encoded: UP if price > EMA50 and EMA50 > EMA200, DOWN if opposite
    if ema_50[i] > ema_200[i] and closes[i] > ema_50[i]:
        trend_enc = 1.0
    elif ema_50[i] < ema_200[i] and closes[i] < ema_50[i]:
        trend_enc = -1.0
    else:
        trend_enc = 0.0

    # ── Assemble in EXACT trainer column order ──
    features = {
        # DB_FEATURE_COLUMNS (26)
        "rsi_7": rsi_7[i],
        "rsi_14": rsi_14[i],
        "rsi_21": rsi_21[i],
        "macd_line": macd_line[i],
        "macd_signal": macd_signal[i],
        "macd_histogram": macd_hist[i],
        "macd_crossover": macd_cross,
        "bb_width": bb_width_val,
        "bb_position": bb_pos[i],
        "ema_9_21_cross": ema_9_21_cross,
        "ema_50_200_cross": ema_50_200_cross,
        "price_above_ema200": price_above_ema200,
        "volume_ratio": vol_ratio[i],
        "volume_spike": vol_spike[i],
        "price_momentum_5d": mom_5[i],
        "price_momentum_10d": mom_10[i],
        "price_momentum_30d": mom_30[i],
        "rate_of_change_14": roc_14[i],
        "atr_14": atr_14[i],
        "volatility_10d": vol_10[i],
        "volatility_30d": vol_30[i],
        "dist_to_support_pct": dist_support,
        "dist_to_resist_pct": dist_resist,
        # MARKET_FEATURE_COLUMNS (5)
        "price_change_24h": price_change_24h,
        "price_change_7d": price_change_7d,
        "price_change_30d": price_change_30d,
        "volume_mcap_ratio": vol_mcap_ratio,
        "ath_change_pct": ath_chg,
        # ENGINEERED_FEATURE_COLUMNS (7)
        "rsi_14_ma_diff": rsi_14_ma_diff,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "price_vs_sma20_pct": price_vs_sma20,
        "price_vs_ema50_pct": price_vs_ema50,
        "price_vs_ema200_pct": price_vs_ema200,
        "vol_ratio_10_30": vol_ratio_10_30,
        "momentum_accel": mom_accel,
        "fear_greed_value": fear_greed_value,
        "trend_encoded": trend_enc,
    }
    return features


# ── Prediction API ──────────────────────────────────────────────

async def predict_pretrained(
    coin_id: str,
    cg_collector,
    coin_data=None,
    days: int = 200,
) -> Optional[Dict[str, Any]]:
    """
    Predict using the pretrained 38-feature models.

    Returns dict with:
        verdict:        STRONG BUY | BUY | NEUTRAL | AVOID | SELL
        direction:      UP | DOWN | SIDEWAYS
        prob_up_24h:    float (0-100)
        prob_up_7d:     float (0-100)
        confidence:     float (1-10)
        direction_probs: {UP: %, DOWN: %, SIDEWAYS: %}

    Returns None if models can't load or insufficient data.
    """
    # ── Check prediction cache ──
    if coin_id in _prediction_cache:
        cached_time, cached_result = _prediction_cache[coin_id]
        if time.time() - cached_time < _PREDICTION_CACHE_TTL:
            return cached_result

    # ── Load models ──
    if not _load_models():
        return None

    # ── Fetch price history ──
    try:
        history = await cg_collector.get_price_history(coin_id, days=days)
    except Exception:
        return None

    if not history or not history.prices:
        return None

    closes = [p[1] for p in history.prices]
    volumes = [v[1] for v in history.volumes] if history.volumes else [0.0] * len(closes)
    if len(volumes) < len(closes):
        volumes.extend([0.0] * (len(closes) - len(volumes)))

    if len(closes) < 50:
        logger.debug("Insufficient price history for %s: %d points", coin_id, len(closes))
        return None

    # ── Fetch Fear & Greed index ──
    fear_greed = 50.0
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=1")
            if resp.status_code == 200:
                data = resp.json()
                fear_greed = float(data.get("data", [{}])[0].get("value", 50))
    except Exception:
        pass  # Default to 50 (neutral)

    # ── Compute features ──
    features = compute_38_features(closes, volumes, coin_data, fear_greed)

    # ── Build feature vector in model's column order ──
    X = [[features.get(col, 0.0) for col in _FEATURE_COLUMNS]]

    try:
        prob_24h = float(_model_24h.predict_proba(X)[0][1])
        prob_7d = float(_model_7d.predict_proba(X)[0][1])
        dir_probs = _model_dir.predict_proba(X)[0]
        dir_pred = str(_label_encoder.classes_[dir_probs.argmax()])
    except Exception as e:
        logger.warning("Pretrained model prediction failed for %s: %s", coin_id, e)
        return None

    # ── Confidence: directional agreement + direction model confirmation ──
    # Confidence is HIGH only when:
    # 1. Both 24h and 7d models agree on direction
    # 2. The direction classifier confirms via its top-class probability
    # 3. The probabilities are well away from 50% (not just noise)
    both_bullish = min(prob_24h, prob_7d)
    both_bearish = min(1 - prob_24h, 1 - prob_7d)
    directional_strength = max(both_bullish, both_bearish)
    dir_confirmation = float(max(dir_probs))  # How confident is the direction model?
    # Penalize when direction model disagrees with prob models
    if dir_pred == "UP" and both_bearish > both_bullish:
        directional_strength *= 0.6  # Direction model says UP but prob models say DOWN
    elif dir_pred == "DOWN" and both_bullish > both_bearish:
        directional_strength *= 0.6  # Disagreement penalty
    # Scale: 0.5 = random → 1, 0.75 = strong → 7, 0.9+ = very strong → 10
    raw_conf = (directional_strength - 0.5) * 16 * dir_confirmation
    confidence = round(max(1.0, min(10.0, raw_conf)), 1)

    # ── Verdict ──
    # Thresholds raised: model CV accuracy is ~59%, so 50% = random noise.
    # Require clear edge (>65%) before acting, and strong agreement for STRONG BUY.
    if prob_7d >= 0.70 and prob_24h >= 0.65:
        verdict = "STRONG BUY"
    elif prob_7d >= 0.62 and prob_24h >= 0.55:
        verdict = "BUY"
    elif prob_7d <= 0.30 and prob_24h <= 0.35:
        verdict = "SELL"
    elif prob_7d <= 0.38 and prob_24h <= 0.40:
        verdict = "AVOID"
    else:
        verdict = "NEUTRAL"

    result = {
        "verdict": verdict,
        "direction": dir_pred,
        "prob_up_24h": round(prob_24h * 100, 1),
        "prob_up_7d": round(prob_7d * 100, 1),
        "confidence": confidence,
        "direction_probs": {
            str(cls): round(float(prob) * 100, 1)
            for cls, prob in zip(_label_encoder.classes_, dir_probs)
        },
        "model_version": _metadata.get("version", "unknown"),
        "model_24h_acc": _metadata.get("model_24h", {}).get("cv_mean", 0),
        "model_7d_acc": _metadata.get("model_7d", {}).get("cv_mean", 0),
    }

    # ── Cache it ──
    _prediction_cache[coin_id] = (time.time(), result)
    return result


def get_pretrained_model_info() -> Dict[str, Any]:
    """Return metadata about the pretrained models (for health checks)."""
    _load_models()
    return {
        "loaded": _model_24h is not None,
        "version": _metadata.get("version"),
        "n_features": _metadata.get("n_features"),
        "model_24h_acc": _metadata.get("model_24h", {}).get("cv_mean"),
        "model_7d_acc": _metadata.get("model_7d", {}).get("cv_mean"),
        "model_dir_acc": _metadata.get("model_dir", {}).get("cv_mean"),
    }
