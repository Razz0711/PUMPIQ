"""
Watchlist Manager
===================
Step 4.3 – CRUD operations and alert evaluation for the user watchlist.

Features:
    - Add / remove tokens from watchlist
    - Set price alerts (target reached, drops below threshold)
    - Evaluate alerts against live prices
    - Export watchlist summary
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .user_config import AlertType, UserPreferences, WatchlistItem

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Triggered Alert
# ══════════════════════════════════════════════════════════════════

class TriggeredAlert:
    """Represents a watchlist alert that has fired."""

    def __init__(
        self,
        token: str,
        alert_type: AlertType,
        alert_price: float,
        current_price: float,
        message: str,
    ):
        self.token = token
        self.alert_type = alert_type
        self.alert_price = alert_price
        self.current_price = current_price
        self.message = message
        self.triggered_at = datetime.utcnow()

    def __repr__(self) -> str:
        return f"<TriggeredAlert {self.token} {self.alert_type.value} @ ${self.current_price}>"


# ══════════════════════════════════════════════════════════════════
# Watchlist Manager
# ══════════════════════════════════════════════════════════════════

class WatchlistManager:
    """
    Manage a user's watchlist and evaluate price alerts.

    Usage::

        mgr = WatchlistManager(prefs)
        mgr.add("PEPE", alert_price=0.005, alert_type=AlertType.TARGET_REACHED)
        alerts = mgr.evaluate_alerts({"PEPE": 0.0052})
    """

    def __init__(self, prefs: UserPreferences):
        self._prefs = prefs

    # ── CRUD ──────────────────────────────────────────────────────

    def add(
        self,
        token: str,
        alert_price: Optional[float] = None,
        alert_type: AlertType = AlertType.TARGET_REACHED,
        notes: str = "",
    ) -> WatchlistItem:
        """Add a token to the watchlist. If already present, update it."""
        token_upper = token.upper()

        # Check if already on watchlist
        for item in self._prefs.watchlist:
            if item.token.upper() == token_upper:
                item.alert_price = alert_price
                item.alert_type = alert_type
                if notes:
                    item.notes = notes
                logger.info("Updated watchlist entry for %s", token_upper)
                return item

        item = WatchlistItem(
            token=token_upper,
            alert_price=alert_price,
            alert_type=alert_type,
            notes=notes,
        )
        self._prefs.watchlist.append(item)
        logger.info("Added %s to watchlist", token_upper)
        return item

    def remove(self, token: str) -> bool:
        """Remove a token from the watchlist. Returns True if found."""
        token_upper = token.upper()
        before = len(self._prefs.watchlist)
        self._prefs.watchlist = [
            w for w in self._prefs.watchlist
            if w.token.upper() != token_upper
        ]
        removed = len(self._prefs.watchlist) < before
        if removed:
            logger.info("Removed %s from watchlist", token_upper)
        return removed

    def get(self, token: str) -> Optional[WatchlistItem]:
        """Get a single watchlist entry."""
        token_upper = token.upper()
        for item in self._prefs.watchlist:
            if item.token.upper() == token_upper:
                return item
        return None

    def list_all(self) -> List[WatchlistItem]:
        """Return all watchlist items."""
        return list(self._prefs.watchlist)

    @property
    def count(self) -> int:
        return len(self._prefs.watchlist)

    # ── Alert Evaluation ──────────────────────────────────────────

    def evaluate_alerts(
        self,
        current_prices: Dict[str, float],
    ) -> List[TriggeredAlert]:
        """
        Check all watchlist alerts against *current_prices*.

        Parameters
        ----------
        current_prices : dict
            Mapping of token ticker (upper) → current price.

        Returns
        -------
        list[TriggeredAlert]
            Alerts whose conditions have been met.
        """
        triggered: List[TriggeredAlert] = []

        for item in self._prefs.watchlist:
            price = current_prices.get(item.token.upper())
            if price is None or item.alert_price is None:
                continue

            alert = self._check_single(item, price)
            if alert:
                triggered.append(alert)

        return triggered

    def _check_single(
        self, item: WatchlistItem, current_price: float,
    ) -> Optional[TriggeredAlert]:
        """Evaluate a single watchlist item against current price."""
        if item.alert_price is None:
            return None

        if item.alert_type == AlertType.TARGET_REACHED:
            if current_price >= item.alert_price:
                return TriggeredAlert(
                    token=item.token,
                    alert_type=item.alert_type,
                    alert_price=item.alert_price,
                    current_price=current_price,
                    message=(
                        f"{item.token} reached ${current_price:.8g} "
                        f"(target was ${item.alert_price:.8g})"
                    ),
                )

        elif item.alert_type == AlertType.STOP_LOSS_TRIGGERED:
            if current_price <= item.alert_price:
                return TriggeredAlert(
                    token=item.token,
                    alert_type=item.alert_type,
                    alert_price=item.alert_price,
                    current_price=current_price,
                    message=(
                        f"⚠️ {item.token} dropped to ${current_price:.8g} "
                        f"(stop-loss at ${item.alert_price:.8g})"
                    ),
                )

        elif item.alert_type == AlertType.ENTRY_ZONE:
            if current_price <= item.alert_price:
                return TriggeredAlert(
                    token=item.token,
                    alert_type=item.alert_type,
                    alert_price=item.alert_price,
                    current_price=current_price,
                    message=(
                        f"{item.token} entered buy zone at ${current_price:.8g} "
                        f"(entry target ${item.alert_price:.8g})"
                    ),
                )

        return None

    # ── Summary ───────────────────────────────────────────────────

    def summary(
        self,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Export watchlist as a list of dicts with current price and change.
        """
        results = []
        for item in self._prefs.watchlist:
            entry: Dict[str, Any] = {
                "token": item.token,
                "alert_price": item.alert_price,
                "alert_type": item.alert_type.value,
                "added_at": item.added_at.isoformat(),
                "notes": item.notes,
            }
            if current_prices:
                price = current_prices.get(item.token.upper())
                entry["current_price"] = price
                if price and item.alert_price:
                    diff = (price - item.alert_price) / item.alert_price * 100
                    entry["distance_to_alert_pct"] = round(diff, 2)
            results.append(entry)
        return results
