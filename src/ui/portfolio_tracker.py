"""
Portfolio Tracker
===================
Step 4.3 – Track user holdings, calculate P&L, and provide
AI-augmented position status updates.

Features:
    - Record buy / sell events
    - Real-time P&L per position and total
    - AI status annotation (on-track / approaching stop / target hit)
    - Position history
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .user_config import PortfolioHolding, UserPreferences

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Position Status
# ══════════════════════════════════════════════════════════════════

@dataclass
class PositionStatus:
    """Enriched view of a single portfolio position."""
    token: str
    entry_price: float
    entry_date: datetime
    quantity: float
    current_price: float = 0.0

    # Computed
    pnl_dollar: float = 0.0
    pnl_percent: float = 0.0
    current_value: float = 0.0
    invested_value: float = 0.0
    ai_status: str = ""           # "HOLD", "WATCH", "SELL"
    ai_status_emoji: str = ""     # 🟢 🟡 🔴
    ai_comment: str = ""          # Free text reason

    recommendation_id: Optional[str] = None
    notes: str = ""


@dataclass
class PortfolioSummary:
    """Aggregated portfolio view."""
    positions: List[PositionStatus] = dc_field(default_factory=list)
    total_invested: float = 0.0
    total_current_value: float = 0.0
    total_pnl_dollar: float = 0.0
    total_pnl_percent: float = 0.0
    winning_count: int = 0
    losing_count: int = 0
    biggest_winner: Optional[PositionStatus] = None
    biggest_loser: Optional[PositionStatus] = None


# ══════════════════════════════════════════════════════════════════
# Portfolio Tracker
# ══════════════════════════════════════════════════════════════════

class PortfolioTracker:
    """
    Manage portfolio holdings and compute P&L.

    Usage::

        tracker = PortfolioTracker(prefs)
        tracker.add_position("PEPE", entry_price=0.04, quantity=10000)
        summary = tracker.get_summary({"PEPE": 0.05})
    """

    def __init__(self, prefs: UserPreferences):
        self._prefs = prefs

    # ── Position Management ───────────────────────────────────────

    def add_position(
        self,
        token: str,
        entry_price: float,
        quantity: float = 0,
        entry_date: Optional[datetime] = None,
        recommendation_id: Optional[str] = None,
        notes: str = "",
    ) -> PortfolioHolding:
        """Record a new position (buy)."""
        holding = PortfolioHolding(
            token=token.upper(),
            entry_price=entry_price,
            entry_date=entry_date or datetime.utcnow(),
            quantity=quantity,
            recommendation_id=recommendation_id,
            notes=notes,
        )
        self._prefs.holdings.append(holding)
        logger.info("Added position: %s @ $%.8g × %.2f", token, entry_price, quantity)
        return holding

    def close_position(self, token: str) -> bool:
        """Remove a holding (record a sell). Returns True if found."""
        token_upper = token.upper()
        before = len(self._prefs.holdings)
        self._prefs.holdings = [
            h for h in self._prefs.holdings
            if h.token.upper() != token_upper
        ]
        return len(self._prefs.holdings) < before

    def update_quantity(self, token: str, new_quantity: float) -> bool:
        """Update quantity for an existing holding (partial sell / add)."""
        for h in self._prefs.holdings:
            if h.token.upper() == token.upper():
                h.quantity = new_quantity
                return True
        return False

    def get_holding(self, token: str) -> Optional[PortfolioHolding]:
        """Get a single holding by ticker."""
        token_upper = token.upper()
        for h in self._prefs.holdings:
            if h.token.upper() == token_upper:
                return h
        return None

    @property
    def count(self) -> int:
        return len(self._prefs.holdings)

    # ── P&L Computation ───────────────────────────────────────────

    def get_summary(
        self,
        current_prices: Dict[str, float],
        entry_exit_plans: Optional[Dict[str, Any]] = None,
    ) -> PortfolioSummary:
        """
        Compute full portfolio summary with real-time prices.

        Parameters
        ----------
        current_prices : dict
            Token ticker (upper) → current price.
        entry_exit_plans : dict, optional
            Token ticker → EntryExitPlan (for AI status annotation).
        """
        positions: List[PositionStatus] = []
        total_invested = 0.0
        total_current = 0.0

        for h in self._prefs.holdings:
            price = current_prices.get(h.token.upper(), 0.0)
            pos = self._compute_position(h, price, entry_exit_plans)
            positions.append(pos)
            total_invested += pos.invested_value
            total_current += pos.current_value

        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        winners = [p for p in positions if p.pnl_dollar > 0]
        losers = [p for p in positions if p.pnl_dollar < 0]

        return PortfolioSummary(
            positions=positions,
            total_invested=round(total_invested, 2),
            total_current_value=round(total_current, 2),
            total_pnl_dollar=round(total_pnl, 2),
            total_pnl_percent=round(total_pnl_pct, 2),
            winning_count=len(winners),
            losing_count=len(losers),
            biggest_winner=max(winners, key=lambda p: p.pnl_percent, default=None),
            biggest_loser=min(losers, key=lambda p: p.pnl_percent, default=None),
        )

    def _compute_position(
        self,
        holding: PortfolioHolding,
        current_price: float,
        plans: Optional[Dict[str, Any]] = None,
    ) -> PositionStatus:
        invested = holding.entry_price * holding.quantity
        current_val = current_price * holding.quantity
        pnl = current_val - invested
        pnl_pct = (pnl / invested * 100) if invested > 0 else 0

        # AI status based on entry/exit plan
        ai_status, ai_emoji, ai_comment = self._annotate_status(
            holding, current_price, plans,
        )

        return PositionStatus(
            token=holding.token,
            entry_price=holding.entry_price,
            entry_date=holding.entry_date,
            quantity=holding.quantity,
            current_price=current_price,
            pnl_dollar=round(pnl, 2),
            pnl_percent=round(pnl_pct, 2),
            current_value=round(current_val, 2),
            invested_value=round(invested, 2),
            ai_status=ai_status,
            ai_status_emoji=ai_emoji,
            ai_comment=ai_comment,
            recommendation_id=holding.recommendation_id,
            notes=holding.notes,
        )

    @staticmethod
    def _annotate_status(
        holding: PortfolioHolding,
        current_price: float,
        plans: Optional[Dict[str, Any]],
    ) -> tuple:
        """
        Determine AI status (HOLD / WATCH / SELL) based on how
        the current price relates to the recommendation's entry/exit plan.
        """
        if not plans or holding.token.upper() not in plans:
            # No plan data – give a basic P&L-based status
            pct = ((current_price - holding.entry_price) / holding.entry_price * 100) if holding.entry_price > 0 else 0
            if pct > 15:
                return "HOLD", "🟢", "On track – approaching first target"
            if pct > 0:
                return "HOLD", "🟢", "In profit – holding"
            if pct > -10:
                return "WATCH", "🟡", "Slight drawdown – monitor closely"
            return "WATCH", "🟡", "Approaching stop-loss level"

        plan = plans[holding.token.upper()]
        sl = getattr(plan, "stop_loss", 0)
        t1 = getattr(plan, "target_1", 0)
        t2 = getattr(plan, "target_2", 0)

        if sl and current_price <= sl:
            return "SELL", "🔴", f"Stop-loss triggered (${sl:.8g})"
        if t2 and current_price >= t2:
            return "SELL", "🟢", f"Target 2 reached (${t2:.8g}) – consider full exit"
        if t1 and current_price >= t1:
            return "HOLD", "🟢", f"Target 1 hit (${t1:.8g}) – consider partial profit"
        if sl and current_price < sl * 1.1:
            return "WATCH", "🟡", "Approaching stop-loss – watch closely"

        return "HOLD", "🟢", "On track for targets"

    # ── Text Summary ──────────────────────────────────────────────

    def format_summary_text(self, summary: PortfolioSummary) -> str:
        """Render a plain-text portfolio summary."""
        lines: List[str] = []
        lines.append("═" * 50)
        lines.append("  MY PORTFOLIO")
        lines.append("═" * 50)

        for p in summary.positions:
            sign = "+" if p.pnl_percent >= 0 else ""
            lines.append(
                f"\n{p.token}"
                f"\n  Entry: ${p.entry_price:.8g} on {p.entry_date.strftime('%b %d')}"
                f"\n  Current: ${p.current_price:.8g} ({sign}{p.pnl_percent:.1f}%)"
                f"\n  AI Status: {p.ai_status_emoji} {p.ai_status} – {p.ai_comment}"
            )

        sign = "+" if summary.total_pnl_dollar >= 0 else ""
        lines.append(f"\n{'─' * 50}")
        lines.append(
            f"Total P&L: {sign}${summary.total_pnl_dollar:,.2f} "
            f"({sign}{summary.total_pnl_percent:.1f}%)"
        )
        lines.append(f"Winners: {summary.winning_count} | Losers: {summary.losing_count}")
        return "\n".join(lines)
