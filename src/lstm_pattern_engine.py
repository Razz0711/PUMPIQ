"""
LSTM Price Pattern Recognizer
===============================
Algorithm 2 – Deep learning pattern recognition using LSTM neural network.

Architecture:
    LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1, sigmoid)

Input shape: (30, 8) — 30-day lookback, 8 normalised features:
    close, volume, RSI, MACD, BB_upper, BB_lower, EMA20, volatility

Output:
    buy_probability (0-1), pattern_detected (string),
    predicted_direction (up/down/sideways)

Models are cached per token and retrained weekly.

⚠ HISTORICAL SIMULATION — not guaranteed future performance.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Optional-import guard (graceful degradation) ──────────────
_HAS_TF = False
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import numpy as np

    # Attempt TensorFlow import
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler

    _HAS_TF = True
except ImportError:
    try:
        import numpy as np
        _has_numpy = True
    except ImportError:
        _has_numpy = False
    logger.warning(
        "TensorFlow / scikit-learn not installed — LSTM engine will "
        "degrade to heuristic mode.  pip install tensorflow scikit-learn"
    )

# ── Constants ────────────────────────────────────────────────────
LOOKBACK = 30          # 30-day sliding window
N_FEATURES = 8
MODEL_DIR = Path(os.path.dirname(__file__)) / ".." / "lstm_models"
MODEL_TTL = 7 * 86400  # retrain weekly
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.15

# Pattern detection thresholds
BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40


# ── Result dataclass ──────────────────────────────────────────────

@dataclass
class LSTMPrediction:
    """Output of the LSTM pattern recognizer."""
    coin_id: str
    symbol: str

    buy_probability: float = 0.5
    predicted_direction: str = "sideways"   # up / down / sideways
    pattern_detected: str = "none"

    model_trained: bool = False
    model_age_hours: float = 0.0
    training_loss: float = 0.0
    training_accuracy: float = 0.0

    error: str = ""
    is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coin_id": self.coin_id,
            "symbol": self.symbol,
            "buy_probability": round(self.buy_probability, 4),
            "predicted_direction": self.predicted_direction,
            "pattern_detected": self.pattern_detected,
            "model_trained": self.model_trained,
            "model_age_hours": round(self.model_age_hours, 1),
            "training_loss": round(self.training_loss, 4),
            "training_accuracy": round(self.training_accuracy, 4),
            "error": self.error,
            "is_fallback": self.is_fallback,
            "label": "HISTORICAL SIMULATION — not guaranteed future performance",
        }


# ── Feature engineering ──────────────────────────────────────────

def _compute_features(closes: List[float],
                      volumes: List[float]) -> Optional[Any]:
    """Compute 8-feature array: [close, volume, RSI, MACD, BB_upper, BB_lower, EMA20, volatility]."""
    if not _HAS_TF and not globals().get("_has_numpy", False):
        return None

    n = len(closes)
    if n < LOOKBACK + 20:
        return None

    # RSI(14)
    rsi = [50.0] * n
    period = 14
    if n > period:
        deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
        gains = [max(d, 0) for d in deltas]
        losses_l = [max(-d, 0) for d in deltas]
        avg_g = sum(gains[:period]) / period
        avg_l = sum(losses_l[:period]) / period
        for idx in range(period, n):
            if idx - 1 < len(gains):
                avg_g = (avg_g * (period - 1) + gains[idx - 1]) / period
                avg_l = (avg_l * (period - 1) + losses_l[idx - 1]) / period
            rsi[idx] = 100.0 - 100.0 / (1 + avg_g / avg_l) if avg_l > 0 else 100.0

    # EMA helper
    def ema_series(vals, p):
        out = list(vals)
        if len(vals) < p:
            return out
        k = 2.0 / (p + 1)
        out[p - 1] = sum(vals[:p]) / p
        for j in range(p, len(vals)):
            out[j] = vals[j] * k + out[j - 1] * (1 - k)
        return out

    # MACD(12,26,9)
    ema12 = ema_series(closes, 12)
    ema26 = ema_series(closes, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(n)]
    signal_line = ema_series(macd_line, 9)
    macd_hist = [macd_line[i] - signal_line[i] for i in range(n)]

    # Bollinger Bands(20, 2)
    bb_upper = [0.0] * n
    bb_lower = [0.0] * n
    bb_period = 20
    for i in range(bb_period - 1, n):
        w = closes[i - bb_period + 1: i + 1]
        sma = sum(w) / bb_period
        std = math.sqrt(sum((c - sma) ** 2 for c in w) / bb_period)
        bb_upper[i] = sma + 2.0 * std
        bb_lower[i] = sma - 2.0 * std

    # EMA20
    ema20 = ema_series(closes, 20)

    # Volatility (10-day rolling std of returns)
    volat = [0.0] * n
    returns = [0.0] * n
    for i in range(1, n):
        returns[i] = (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] > 0 else 0
    for i in range(10, n):
        w = returns[i - 9: i + 1]
        mean = sum(w) / len(w)
        var = sum((r - mean) ** 2 for r in w) / len(w)
        volat[i] = math.sqrt(var)

    # Stack into array [n, 8]
    features = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i in range(n):
        features[i] = [
            closes[i], volumes[i], rsi[i], macd_hist[i],
            bb_upper[i], bb_lower[i], ema20[i], volat[i],
        ]

    return features


def _create_sequences(features: Any, target: Any,
                      lookback: int = LOOKBACK) -> Tuple[Any, Any]:
    """Create sliding-window sequences for LSTM."""
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback: i])
        y.append(target[i])
    return np.array(X), np.array(y)


def _detect_pattern(closes: List[float], buy_prob: float) -> str:
    """Simple pattern detection from recent price action."""
    if len(closes) < 30:
        return "insufficient_data"

    recent = closes[-30:]
    # Higher lows check
    lows = [min(recent[i:i+5]) for i in range(0, 25, 5)]
    highs = [max(recent[i:i+5]) for i in range(0, 25, 5)]

    higher_lows = all(lows[i] <= lows[i+1] for i in range(len(lows)-1))
    higher_highs = all(highs[i] <= highs[i+1] for i in range(len(highs)-1))
    lower_lows = all(lows[i] >= lows[i+1] for i in range(len(lows)-1))
    lower_highs = all(highs[i] >= highs[i+1] for i in range(len(highs)-1))
    flat_highs = max(highs) - min(highs) < (max(highs) * 0.02)

    if higher_lows and flat_highs and buy_prob > 0.55:
        return "ascending_triangle"
    if higher_lows and higher_highs:
        return "uptrend_channel"
    if lower_lows and lower_highs:
        return "downtrend_channel"
    if higher_lows and lower_highs:
        return "symmetrical_triangle"
    if flat_highs and lower_lows:
        return "descending_triangle"

    # Volatility squeeze
    vol_ratio = (max(recent[-5:]) - min(recent[-5:])) / (max(recent) - min(recent) + 1e-10)
    if vol_ratio < 0.15:
        return "consolidation_squeeze"

    # Recent momentum
    if buy_prob > 0.7:
        return "strong_bullish_momentum"
    if buy_prob < 0.3:
        return "strong_bearish_momentum"

    return "none"


# ══════════════════════════════════════════════════════════════════
# LSTM Engine
# ══════════════════════════════════════════════════════════════════

class LSTMPatternEngine:
    """
    LSTM-based price pattern recognizer with per-token model caching.

    Usage::

        engine = LSTMPatternEngine()
        pred = await engine.predict("bitcoin", "Bitcoin", "BTC", cg_collector)
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._scalers: Dict[str, Any] = {}

    # ── cache helpers ─────────────────────────────────────────────

    def _model_path(self, coin_id: str) -> Path:
        return self.model_dir / f"lstm_{coin_id}.keras"

    def _scaler_path(self, coin_id: str) -> Path:
        return self.model_dir / f"scaler_{coin_id}.pkl"

    def _is_cache_valid(self, coin_id: str) -> bool:
        path = self._model_path(coin_id)
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < MODEL_TTL

    def _load_model(self, coin_id: str) -> Optional[Any]:
        if not _HAS_TF:
            return None
        path = self._model_path(coin_id)
        if not path.exists():
            return None
        try:
            return load_model(str(path))
        except Exception as exc:
            logger.warning("Failed to load LSTM model for %s: %s", coin_id, exc)
            return None

    def _load_scaler(self, coin_id: str) -> Optional[Any]:
        if not _HAS_TF:
            return None
        import pickle
        path = self._scaler_path(coin_id)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_scaler(self, coin_id: str, scaler: Any):
        import pickle
        try:
            with open(self._scaler_path(coin_id), "wb") as f:
                pickle.dump(scaler, f)
        except Exception as exc:
            logger.warning("Failed to save scaler for %s: %s", coin_id, exc)

    # ── build & train ─────────────────────────────────────────────

    def _build_model(self) -> Any:
        """Build LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1, sigmoid)."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(LOOKBACK, N_FEATURES)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    async def _train_model(
        self, coin_id: str, closes: List[float], volumes: List[float],
    ) -> Tuple[Optional[Any], Optional[Any], float, float]:
        """Train a new LSTM model for the given token. Returns (model, scaler, loss, accuracy)."""
        features = _compute_features(closes, volumes)
        if features is None or len(features) < LOOKBACK + 20:
            return None, None, 0, 0

        # Target: price up > 2% within next 5 candles
        target = np.zeros(len(closes))
        for i in range(len(closes) - 5):
            future_max = max(closes[i+1: i+6])
            if (future_max - closes[i]) / closes[i] > 0.02:
                target[i] = 1.0

        # Normalise features
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)

        X, y = _create_sequences(scaled, target)
        if len(X) < 50:
            return None, None, 0, 0

        # Build & train
        model = self._build_model()

        # Suppress TF output
        history = model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=0,
        )

        final_loss = float(history.history["loss"][-1])
        final_acc = float(history.history.get("accuracy", history.history.get("acc", [0]))[-1])

        # Save model & scaler
        try:
            model.save(str(self._model_path(coin_id)))
            self._save_scaler(coin_id, scaler)
        except Exception as exc:
            logger.warning("Failed to save LSTM model for %s: %s", coin_id, exc)

        return model, scaler, final_loss, final_acc

    # ── main prediction ───────────────────────────────────────────

    async def predict(
        self,
        coin_id: str,
        coin_name: str,
        symbol: str,
        cg_collector,
        days: int = 180,
        force_retrain: bool = False,
    ) -> LSTMPrediction:
        """Run LSTM prediction for a token."""
        result = LSTMPrediction(coin_id=coin_id, symbol=symbol.upper())

        # ── graceful degradation ──
        if not _HAS_TF:
            result.is_fallback = True
            result.error = "TensorFlow not installed"
            # Heuristic fallback: use simple momentum
            try:
                history = await cg_collector.get_price_history(coin_id, days=30)
                if history and history.prices and len(history.prices) > 10:
                    closes = [p[1] for p in history.prices]
                    pct = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0
                    result.buy_probability = min(1.0, max(0.0, 0.5 + pct * 2))
                    result.predicted_direction = (
                        "up" if result.buy_probability > 0.6 else
                        "down" if result.buy_probability < 0.4 else "sideways"
                    )
                    result.pattern_detected = _detect_pattern(closes, result.buy_probability)
            except Exception:
                pass
            return result

        # ── fetch data ──
        try:
            history = await cg_collector.get_price_history(coin_id, days=max(days, 180))
        except Exception as exc:
            result.error = f"Data fetch failed: {exc}"
            result.is_fallback = True
            return result

        if not history or not history.prices or len(history.prices) < LOOKBACK + 20:
            result.error = "Insufficient historical data"
            result.is_fallback = True
            return result

        closes = [p[1] for p in history.prices]
        volumes = [v[1] for v in history.volumes] if history.volumes else [0.0] * len(closes)
        if len(volumes) < len(closes):
            volumes.extend([0.0] * (len(closes) - len(volumes)))

        # ── check cache or train ──
        model = None
        scaler = None

        if not force_retrain and self._is_cache_valid(coin_id):
            model = self._load_model(coin_id)
            scaler = self._load_scaler(coin_id)
            if model is not None:
                age = time.time() - self._model_path(coin_id).stat().st_mtime
                result.model_age_hours = age / 3600

        if model is None or scaler is None:
            logger.info("Training new LSTM model for %s ...", symbol)
            model, scaler, loss, acc = await self._train_model(coin_id, closes, volumes)
            if model is None:
                result.error = "Training failed — insufficient data"
                result.is_fallback = True
                return result
            result.training_loss = loss
            result.training_accuracy = acc
            result.model_age_hours = 0.0

        result.model_trained = True

        # ── predict on latest window ──
        features = _compute_features(closes, volumes)
        if features is None:
            result.error = "Feature computation failed"
            result.is_fallback = True
            return result

        scaled = scaler.transform(features)
        # Take the last LOOKBACK rows
        X_pred = scaled[-LOOKBACK:].reshape(1, LOOKBACK, N_FEATURES)

        prob = float(model.predict(X_pred, verbose=0)[0][0])
        result.buy_probability = round(prob, 4)

        if prob >= BULLISH_THRESHOLD:
            result.predicted_direction = "up"
        elif prob <= BEARISH_THRESHOLD:
            result.predicted_direction = "down"
        else:
            result.predicted_direction = "sideways"

        result.pattern_detected = _detect_pattern(closes, prob)

        logger.info(
            "LSTM [%s]: buy_prob=%.3f  direction=%s  pattern=%s",
            symbol, prob, result.predicted_direction, result.pattern_detected,
        )

        return result


# ── Module-level singleton ────────────────────────────────────────

_lstm_engine: Optional[LSTMPatternEngine] = None


def get_lstm_engine() -> LSTMPatternEngine:
    global _lstm_engine
    if _lstm_engine is None:
        _lstm_engine = LSTMPatternEngine()
    return _lstm_engine
