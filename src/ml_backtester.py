"""
XGBoost ML Backtester
=======================
Algorithm 1 – Machine learning–based backtest validation using XGBClassifier.

Replaces static threshold checks with a trained classifier that learns
which indicator configurations actually predict profitable trades.

Features (8):
  1. RSI(14)
  2. MACD histogram
  3. Bollinger Band position  (price − mid) / (upper − lower)
  4. Volume ratio  (current / 20-day MA)
  5. Price momentum  (5-day % change)
  6. Volatility  (10-day rolling std of returns)
  7. EMA crossover signal  (EMA9 − EMA21 normalised)
  8. Buy/sell ratio proxy  (volume delta heuristic)

Target: price increases > 5 % within next 7 data points.

Validation: **TimeSeriesSplit(n_splits=5)** — never random split.

⚠ HISTORICAL SIMULATION — not guaranteed future performance.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ta_utils import (
    ema_series as _ta_ema_series,
    rsi_series as _ta_rsi_series,
    macd_series as _ta_macd_series,
    bollinger_position as _ta_bollinger_position,
)

logger = logging.getLogger(__name__)

# ── Optional-import guard  (graceful degradation) ──────────────
try:
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBClassifier

    _HAS_ML = True
except ImportError:
    _HAS_ML = False
    logger.warning(
        "xgboost / scikit-learn not installed — ML backtester will "
        "degrade to rule-based mode.  pip install xgboost scikit-learn"
    )

# ── Constants ────────────────────────────────────────────────────
MIN_HISTORY_DAYS = 180
FORWARD_WINDOW = 7            # look-ahead for target (7 data points)
TARGET_RETURN_PCT = 5.0       # price must rise ≥ 5 % within window
ACCURACY_THRESHOLD = 0.55     # minimum to pass ML gate
MODEL_CACHE_DIR = Path(os.path.dirname(__file__)) / ".." / "ml_models"
MODEL_TTL_SECONDS = 7 * 86400  # retrain weekly


# ── Result dataclass ──────────────────────────────────────────────

@dataclass
class MLBacktestResult:
    """Output of the XGBoost backtest validation."""
    coin_id: str
    symbol: str

    ml_accuracy: float = 0.0              # 0-1
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    top_feature: str = ""
    top_feature_importance: float = 0.0

    confidence_interval_low: float = 0.0
    confidence_interval_high: float = 0.0

    passes_threshold: bool = False
    recommended_parameters: Dict[str, Any] = field(default_factory=dict)

    n_samples: int = 0
    n_positive: int = 0
    n_splits: int = 5

    error: str = ""
    is_fallback: bool = False             # True when ML unavailable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coin_id": self.coin_id,
            "symbol": self.symbol,
            "ml_accuracy": round(self.ml_accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "feature_importance": {
                k: round(v, 4) for k, v in self.feature_importance.items()
            },
            "top_feature": self.top_feature,
            "top_feature_importance": round(self.top_feature_importance, 4),
            "confidence_interval": [
                round(self.confidence_interval_low, 4),
                round(self.confidence_interval_high, 4),
            ],
            "passes_threshold": self.passes_threshold,
            "recommended_parameters": self.recommended_parameters,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "error": self.error,
            "is_fallback": self.is_fallback,
            "label": "HISTORICAL SIMULATION — not guaranteed future performance",
        }


# ── Feature engineering helpers ──────────────────────────────────

def _ema(values: List[float], period: int) -> List[float]:
    """Full-length EMA series (delegates to ta_utils)."""
    return _ta_ema_series(values, period)


def _rsi_series(closes: List[float], period: int = 14) -> List[float]:
    """RSI series (delegates to ta_utils)."""
    return _ta_rsi_series(closes, period)


def _macd_histogram(closes: List[float], fast: int = 12, slow: int = 26,
                    sig: int = 9) -> List[float]:
    """MACD histogram series (delegates to ta_utils)."""
    _, _, histogram = _ta_macd_series(closes, fast, slow, sig)
    return histogram


def _bollinger_position(closes: List[float], period: int = 20,
                        std_mult: float = 2.0) -> List[float]:
    """Position of price within Bollinger bands (delegates to ta_utils)."""
    return _ta_bollinger_position(closes, period, std_mult)


def _volume_ratio(volumes: List[float], period: int = 20) -> List[float]:
    n = len(volumes)
    ratio = [1.0] * n
    for i in range(period, n):
        ma = sum(volumes[i - period: i]) / period
        ratio[i] = volumes[i] / ma if ma > 0 else 1.0
    return ratio


def _momentum(closes: List[float], period: int = 5) -> List[float]:
    n = len(closes)
    mom = [0.0] * n
    for i in range(period, n):
        if closes[i - period] > 0:
            mom[i] = (closes[i] - closes[i - period]) / closes[i - period]
    return mom


def _volatility(closes: List[float], period: int = 10) -> List[float]:
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


def _ema_crossover(closes: List[float], fast: int = 9,
                   slow: int = 21) -> List[float]:
    """EMA crossover normalised signal (uses ta_utils for EMA)."""
    ema_f = _ta_ema_series(closes, fast)
    ema_s = _ta_ema_series(closes, slow)
    n = len(closes)
    cross = [0.0] * n
    for i in range(slow, n):
        if ema_s[i] > 0:
            cross[i] = (ema_f[i] - ema_s[i]) / ema_s[i]
    return cross


def _buy_sell_proxy(closes: List[float], volumes: List[float]) -> List[float]:
    """Heuristic: volume-weighted direction indicates buy vs sell pressure."""
    n = len(closes)
    proxy = [1.0] * n
    for i in range(1, n):
        direction = 1.0 if closes[i] >= closes[i - 1] else -1.0
        vol_norm = volumes[i] / max(volumes[i - 1], 1.0) if volumes[i - 1] > 0 else 1.0
        proxy[i] = direction * vol_norm
    return proxy


# ── Build feature matrix ─────────────────────────────────────────

FEATURE_NAMES = [
    "rsi", "macd_hist", "bb_position", "volume_ratio",
    "momentum_5d", "volatility_10d", "ema_crossover", "buy_sell_ratio",
]


def build_features(closes: List[float],
                   volumes: List[float]) -> Tuple[Any, List[float]]:
    """
    Build (X, target_labels) numpy arrays.

    Returns (X: ndarray[N, 8],  y: list[int])  where
    N = len(closes) − FORWARD_WINDOW − warm_up.
    """
    if not _HAS_ML:
        return None, []

    n = len(closes)
    rsi = _rsi_series(closes)
    macd_h = _macd_histogram(closes)
    bb_pos = _bollinger_position(closes)
    vol_r = _volume_ratio(volumes)
    mom = _momentum(closes)
    volat = _volatility(closes)
    ema_x = _ema_crossover(closes)
    bsr = _buy_sell_proxy(closes, volumes)

    warm_up = 30  # skip initial indicator warm-up period
    end = n - FORWARD_WINDOW

    rows = []
    targets = []
    for i in range(warm_up, end):
        row = [
            rsi[i], macd_h[i], bb_pos[i], vol_r[i],
            mom[i], volat[i], ema_x[i], bsr[i],
        ]
        # Target: did price rise ≥ 5 % in next FORWARD_WINDOW points?
        future_max = max(closes[i + 1: i + 1 + FORWARD_WINDOW])
        target = 1 if (future_max - closes[i]) / closes[i] >= TARGET_RETURN_PCT / 100 else 0
        rows.append(row)
        targets.append(target)

    X = np.array(rows, dtype=np.float32)
    return X, targets


# ══════════════════════════════════════════════════════════════════
# ML Backtester
# ══════════════════════════════════════════════════════════════════

class MLBacktester:
    """
    XGBoost-based backtester with walk-forward TimeSeriesSplit validation.

    Usage::

        bt = MLBacktester()
        result = await bt.run("bitcoin", "Bitcoin", "BTC", cg_collector)
    """

    def __init__(self, n_splits: int = 5, cache_dir: Optional[Path] = None):
        self.n_splits = n_splits
        self.cache_dir = cache_dir or MODEL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── cache helpers ─────────────────────────────────────────────

    def _model_path(self, coin_id: str) -> Path:
        return self.cache_dir / f"xgb_{coin_id}.pkl"

    def _load_cached(self, coin_id: str) -> Optional[Tuple[Any, float]]:
        path = self.cache_dir / f"xgb_{coin_id}.pkl"
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > MODEL_TTL_SECONDS:
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cached(self, coin_id: str, model: Any, accuracy: float):
        try:
            path = self._model_path(coin_id)
            with open(path, "wb") as f:
                pickle.dump((model, accuracy), f)
        except Exception as exc:
            logger.warning("Failed to cache XGB model for %s: %s", coin_id, exc)

    # ── main entry point ──────────────────────────────────────────

    async def run(
        self,
        coin_id: str,
        coin_name: str,
        symbol: str,
        cg_collector,
        days: int = MIN_HISTORY_DAYS,
    ) -> MLBacktestResult:
        """Run XGBoost ML backtest for a token."""
        result = MLBacktestResult(coin_id=coin_id, symbol=symbol.upper())

        # ── graceful degradation ──
        if not _HAS_ML:
            result.is_fallback = True
            result.error = "ML libraries not installed (xgboost/scikit-learn)"
            result.ml_accuracy = 0.0
            result.passes_threshold = False
            return result

        # ── fetch data ──
        try:
            history = await cg_collector.get_price_history(coin_id, days=max(days, MIN_HISTORY_DAYS))
        except Exception as exc:
            result.error = f"Data fetch failed: {exc}"
            result.is_fallback = True
            return result

        if not history or not history.prices or len(history.prices) < 60:
            result.error = f"Insufficient data: {len(history.prices) if history and history.prices else 0} points"
            result.is_fallback = True
            return result

        closes = [p[1] for p in history.prices]
        volumes = [v[1] for v in history.volumes] if history.volumes else [0.0] * len(closes)
        if len(volumes) < len(closes):
            volumes.extend([0.0] * (len(closes) - len(volumes)))

        # ── build features ──
        X, y = build_features(closes, volumes)
        if X is None or len(y) < 50:
            result.error = f"Not enough samples after feature engineering: {len(y)}"
            result.is_fallback = True
            return result

        y_arr = np.array(y, dtype=np.int32)
        result.n_samples = len(y)
        result.n_positive = int(y_arr.sum())

        # ── walk-forward cross-validation with TimeSeriesSplit ──
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        last_model = None
        last_importances = None

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = float((preds == y_test).mean())
            fold_accuracies.append(acc)

            # Precision / recall
            tp = int(((preds == 1) & (y_test == 1)).sum())
            fp = int(((preds == 1) & (y_test == 0)).sum())
            fn = int(((preds == 0) & (y_test == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fold_precisions.append(prec)
            fold_recalls.append(rec)

            last_model = model
            last_importances = model.feature_importances_

        # ── aggregate results ──
        result.ml_accuracy = float(np.mean(fold_accuracies))
        result.precision = float(np.mean(fold_precisions))
        result.recall = float(np.mean(fold_recalls))
        if result.precision + result.recall > 0:
            result.f1_score = 2 * result.precision * result.recall / (result.precision + result.recall)

        # Confidence interval (±1.96 * std / sqrt(n_splits))
        acc_std = float(np.std(fold_accuracies))
        margin = 1.96 * acc_std / math.sqrt(self.n_splits)
        result.confidence_interval_low = max(0, result.ml_accuracy - margin)
        result.confidence_interval_high = min(1, result.ml_accuracy + margin)

        result.n_splits = self.n_splits
        result.passes_threshold = result.ml_accuracy >= ACCURACY_THRESHOLD

        # Feature importance
        if last_importances is not None:
            imp = {FEATURE_NAMES[i]: float(last_importances[i])
                   for i in range(len(FEATURE_NAMES))}
            # Normalise to sum=1
            total = sum(imp.values())
            if total > 0:
                imp = {k: v / total for k, v in imp.items()}
            result.feature_importance = dict(
                sorted(imp.items(), key=lambda x: x[1], reverse=True)
            )
            if result.feature_importance:
                top = next(iter(result.feature_importance.items()))
                result.top_feature = top[0]
                result.top_feature_importance = top[1]

        # Recommended parameters (from feature importance ranking)
        result.recommended_parameters = self._suggest_parameters(result.feature_importance)

        # Cache model
        if last_model is not None:
            self._save_cached(coin_id, last_model, result.ml_accuracy)

        logger.info(
            "ML Backtest [%s]: accuracy=%.3f  precision=%.3f  recall=%.3f  "
            "passes=%s  top_feature=%s (%.1f%%)",
            symbol, result.ml_accuracy, result.precision, result.recall,
            result.passes_threshold, result.top_feature,
            result.top_feature_importance * 100,
        )

        return result

    # ── predict on latest data ────────────────────────────────────

    async def predict_latest(
        self, coin_id: str, cg_collector, days: int = 90,
    ) -> Optional[Dict[str, Any]]:
        """Use cached model to predict on latest market data."""
        cached = self._load_cached(coin_id)
        if cached is None:
            return None

        model, cached_acc = cached

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

        # Get latest features
        rsi = _rsi_series(closes)
        macd_h = _macd_histogram(closes)
        bb_pos = _bollinger_position(closes)
        vol_r = _volume_ratio(volumes)
        mom = _momentum(closes)
        volat = _volatility(closes)
        ema_x = _ema_crossover(closes)
        bsr = _buy_sell_proxy(closes, volumes)

        i = len(closes) - 1
        row = np.array([[
            rsi[i], macd_h[i], bb_pos[i], vol_r[i],
            mom[i], volat[i], ema_x[i], bsr[i],
        ]], dtype=np.float32)

        prob = model.predict_proba(row)[0]
        buy_prob = float(prob[1]) if len(prob) > 1 else 0.0

        return {
            "buy_probability": round(buy_prob, 4),
            "prediction": "BUY" if buy_prob > 0.5 else "HOLD",
            "model_accuracy": cached_acc,
        }

    @staticmethod
    def _suggest_parameters(importance: Dict[str, float]) -> Dict[str, Any]:
        """Suggest indicator parameters based on which features matter most."""
        params: Dict[str, Any] = {}
        if importance.get("rsi", 0) > 0.2:
            params["rsi_period"] = 14
            params["rsi_note"] = "RSI is a top driver — keep standard period"
        if importance.get("macd_hist", 0) > 0.2:
            params["macd_fast"] = 12
            params["macd_slow"] = 26
        if importance.get("volume_ratio", 0) > 0.25:
            params["volume_ma_period"] = 20
            params["volume_note"] = "Volume is key — monitor 20-day volume MA closely"
        if importance.get("momentum_5d", 0) > 0.2:
            params["momentum_period"] = 5
        return params


# ── Module-level singleton ────────────────────────────────────────

_ml_backtester: Optional[MLBacktester] = None


def get_ml_backtester() -> MLBacktester:
    global _ml_backtester
    if _ml_backtester is None:
        _ml_backtester = MLBacktester()
    return _ml_backtester
