"""
Notification Formatter
========================
Step 4.2 – Format alert and notification content for push, email,
SMS, and in-app channels.

Notification types:
  1. Price alerts       (target hit, stop-loss triggered)
  2. New recommendations
  3. Risk warnings      (liquidity drop, whale dump, metrics deterioration)
  4. Portfolio updates   (P&L milestones)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Notification Enums
# ══════════════════════════════════════════════════════════════════

class NotificationChannel(str, Enum):
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    TELEGRAM = "telegram"
    DISCORD = "discord"


class NotificationType(str, Enum):
    PRICE_ALERT = "price_alert"
    NEW_RECOMMENDATION = "new_recommendation"
    RISK_WARNING = "risk_warning"
    PORTFOLIO_UPDATE = "portfolio_update"
    TARGET_HIT = "target_hit"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    WATCHLIST_CHANGE = "watchlist_change"


class NotificationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ══════════════════════════════════════════════════════════════════
# Notification Data
# ══════════════════════════════════════════════════════════════════

@dataclass
class Notification:
    """Rendered notification ready for delivery."""
    ntype: NotificationType
    priority: NotificationPriority = NotificationPriority.MEDIUM
    title: str = ""
    body: str = ""
    short_body: str = ""         # SMS / push
    html_body: str = ""          # email
    action_url: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


# ══════════════════════════════════════════════════════════════════
# Formatter
# ══════════════════════════════════════════════════════════════════

class NotificationFormatter:
    """
    Build channel-specific notification content from recommendation
    and alert events.

    Usage::

        fmt = NotificationFormatter()
        notif = fmt.price_alert(token="PEPE", current=0.005, target=0.005,
                                entry=0.0042, alert_type="target_reached")
    """

    # ── Price Alert ───────────────────────────────────────────────

    def price_alert(
        self,
        token: str,
        current_price: float,
        target_price: float,
        entry_price: Optional[float] = None,
        alert_type: str = "target_reached",
    ) -> Notification:
        pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0

        if alert_type == "target_reached":
            emoji = "🎯"
            action = "Consider taking partial profits"
        elif alert_type == "stop_loss":
            emoji = "🛑"
            action = "Consider exiting position to protect capital"
        else:
            emoji = "🔔"
            action = "Review current position"

        title = f"{emoji} PRICE ALERT: {token}"
        body = (
            f"{token} reached ${current_price:.8g}"
        )
        if entry_price:
            sign = "+" if pct >= 0 else ""
            body += f" ({sign}{pct:.1f}% from entry ${entry_price:.8g})"
        body += f"\n{action}"

        short_body = f"{token} hit ${current_price:.8g}"
        if entry_price:
            sign = "+" if pct >= 0 else ""
            short_body += f" ({sign}{pct:.0f}%)"

        html_body = (
            f"<h2>{emoji} Price Alert: {token}</h2>"
            f"<p><strong>Current Price:</strong> ${current_price:.8g}</p>"
        )
        if entry_price:
            html_body += f"<p><strong>Entry:</strong> ${entry_price:.8g} &rarr; <strong>{pct:+.1f}%</strong></p>"
        html_body += f"<p>{action}</p>"
        html_body += '<p><a href="{{action_url}}">View Full Analysis</a></p>'

        return Notification(
            ntype=NotificationType.PRICE_ALERT,
            priority=NotificationPriority.HIGH if alert_type == "stop_loss" else NotificationPriority.MEDIUM,
            title=title,
            body=body,
            short_body=short_body,
            html_body=html_body,
            action_url=f"/analyze/{token.lower()}",
            data={
                "token": token,
                "current_price": current_price,
                "target_price": target_price,
                "entry_price": entry_price,
                "pct_change": round(pct, 2),
            },
        )

    # ── Target Hit ────────────────────────────────────────────────

    def target_hit(
        self,
        token: str,
        target_num: int,
        current_price: float,
        target_price: float,
        entry_price: float,
        next_target: Optional[float] = None,
    ) -> Notification:
        pct = (current_price - entry_price) / entry_price * 100

        title = f"🎯 {token} hit Target {target_num}!"
        body = (
            f"Entry: ${entry_price:.8g} → Current: ${current_price:.8g} (+{pct:.0f}%)\n"
        )
        if next_target:
            body += f"Next: Consider taking 50% profit, hold rest for Target {target_num+1} (${next_target:.8g})"
        else:
            body += "Consider taking profits."

        short_body = f"{token} T{target_num} hit! +{pct:.0f}%"

        html_body = (
            f"<h2>🎯 PumpIQ Alert: {token} hit Target {target_num}</h2>"
            f"<p>Entry: ${entry_price:.8g} &rarr; Current: ${current_price:.8g} (<strong>+{pct:.0f}%</strong>)</p>"
        )
        if next_target:
            html_body += (
                f"<p>Next: Consider taking 50% profit, hold rest for "
                f"Target {target_num+1} (${next_target:.8g})</p>"
            )
        html_body += '<p><a href="{{action_url}}">View Full Analysis</a></p>'

        return Notification(
            ntype=NotificationType.TARGET_HIT,
            priority=NotificationPriority.HIGH,
            title=title,
            body=body,
            short_body=short_body,
            html_body=html_body,
            action_url=f"/analyze/{token.lower()}",
        )

    # ── Risk Warning ──────────────────────────────────────────────

    def risk_warning(
        self,
        token: str,
        warning_type: str,
        details: str,
    ) -> Notification:
        emoji_map = {
            "liquidity_drop": "💧",
            "whale_dump": "🐋",
            "metrics_deterioration": "📉",
            "social_red_flag": "🚨",
            "smart_money_exit": "🏃",
        }
        emoji = emoji_map.get(warning_type, "⚠️")
        readable = warning_type.replace("_", " ").title()

        title = f"{emoji} Risk Warning: {token}"
        body = f"{readable}: {details}"
        short_body = f"⚠️ {token}: {readable}"

        html_body = (
            f"<h2>{emoji} Risk Warning for {token}</h2>"
            f"<p><strong>{readable}</strong></p>"
            f"<p>{details}</p>"
            f'<p><a href="{{action_url}}">View Updated Analysis</a></p>'
        )

        return Notification(
            ntype=NotificationType.RISK_WARNING,
            priority=NotificationPriority.HIGH,
            title=title,
            body=body,
            short_body=short_body,
            html_body=html_body,
            action_url=f"/analyze/{token.lower()}",
            data={"token": token, "warning_type": warning_type},
        )

    # ── New Recommendation ────────────────────────────────────────

    def new_recommendation(
        self,
        token: str,
        verdict: str,
        confidence: float,
        risk: str,
        entry_low: float,
        target_1_pct: float,
    ) -> Notification:
        title = f"🆕 New Pick: {token} – {verdict}"
        body = (
            f"{verdict} | Confidence {confidence:.1f}/10 | Risk: {risk}\n"
            f"Entry near ${entry_low:.8g} → Target +{target_1_pct:.0f}%"
        )
        short_body = f"NEW: {token} {verdict} ({confidence:.0f}/10)"

        html_body = (
            f"<h2>🆕 New PumpIQ Pick: {token}</h2>"
            f"<p><strong>{verdict}</strong> | Confidence: {confidence:.1f}/10 | Risk: {risk}</p>"
            f"<p>Entry: ${entry_low:.8g} | Upside: +{target_1_pct:.0f}%</p>"
            f'<p><a href="{{action_url}}">View Full Recommendation</a></p>'
        )

        return Notification(
            ntype=NotificationType.NEW_RECOMMENDATION,
            priority=NotificationPriority.MEDIUM,
            title=title,
            body=body,
            short_body=short_body,
            html_body=html_body,
            action_url=f"/analyze/{token.lower()}",
        )

    # ── Portfolio Update ──────────────────────────────────────────

    def portfolio_update(
        self,
        total_pnl: float,
        total_pnl_pct: float,
        positions: List[Dict[str, Any]],
    ) -> Notification:
        sign = "+" if total_pnl >= 0 else ""
        emoji = "📈" if total_pnl >= 0 else "📉"

        title = f"{emoji} Portfolio: {sign}${total_pnl:,.2f} ({sign}{total_pnl_pct:.1f}%)"
        lines = [title, ""]
        for pos in positions[:5]:
            tok = pos.get("token", "?")
            pct = pos.get("pnl_pct", 0)
            status = pos.get("status", "")
            s = "+" if pct >= 0 else ""
            lines.append(f"  {tok}: {s}{pct:.1f}% – {status}")

        body = "\n".join(lines)
        short_body = f"Portfolio {sign}{total_pnl_pct:.0f}%"

        return Notification(
            ntype=NotificationType.PORTFOLIO_UPDATE,
            priority=NotificationPriority.LOW,
            title=title,
            body=body,
            short_body=short_body,
            action_url="/portfolio",
        )
