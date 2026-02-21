"""
User Configuration & Preferences
===================================
Step 4.3 – Pydantic models and backend logic for personalized settings.

Covers all 8 settings categories from spec:
    1. Data source preferences (modes + weights)
    2. Risk profile
    3. Investment timeframe
    4. Position size guidance
    5. Notification preferences
    6. Advanced filters
    7. Watchlist (model only – logic in watchlist_manager.py)
    8. Portfolio holdings (model only – logic in portfolio_tracker.py)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class TradingStyle(str, Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"


class PositionSizingStrategy(str, Enum):
    FIXED_PERCENT = "fixed_percent"
    FIXED_DOLLAR = "fixed_dollar"
    RISK_BASED = "risk_based"


class NotificationFrequency(str, Enum):
    REAL_TIME = "real-time"
    HOURLY = "hourly"
    DAILY = "daily"


class AlertType(str, Enum):
    ENTRY_ZONE = "entry_zone"
    TARGET_REACHED = "target_reached"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    METRICS_CHANGE = "metrics_change"
    DAILY_SUMMARY = "daily_summary"


class TokenAgeFilter(str, Enum):
    ANY = "any"
    UNDER_7D = "<7d"
    UNDER_30D = "<30d"
    OVER_30D = ">30d"


# ══════════════════════════════════════════════════════════════════
# Sub-models
# ══════════════════════════════════════════════════════════════════

class DataSourceConfig(BaseModel):
    """Single data-source toggle and weight."""
    enabled: bool = True
    weight: float = Field(default=0.25, ge=0, le=1.0)


class DataSourcePreferences(BaseModel):
    """All four data sources with weights that auto-normalise."""
    news: DataSourceConfig = Field(default_factory=lambda: DataSourceConfig(weight=0.20))
    onchain: DataSourceConfig = Field(default_factory=lambda: DataSourceConfig(weight=0.35))
    technical: DataSourceConfig = Field(default_factory=lambda: DataSourceConfig(weight=0.25))
    social: DataSourceConfig = Field(default_factory=lambda: DataSourceConfig(weight=0.20))

    @property
    def enabled_modes(self) -> List[str]:
        modes = []
        if self.news.enabled:
            modes.append("news")
        if self.onchain.enabled:
            modes.append("onchain")
        if self.technical.enabled:
            modes.append("technical")
        if self.social.enabled:
            modes.append("social")
        return modes

    @property
    def normalised_weights(self) -> Dict[str, float]:
        """Return weights normalised to sum=1.0 across enabled sources only."""
        raw: Dict[str, float] = {}
        if self.news.enabled:
            raw["news"] = self.news.weight
        if self.onchain.enabled:
            raw["onchain"] = self.onchain.weight
        if self.technical.enabled:
            raw["technical"] = self.technical.weight
        if self.social.enabled:
            raw["social"] = self.social.weight

        total = sum(raw.values())
        if total == 0:
            return raw
        return {k: round(v / total, 4) for k, v in raw.items()}


class PortfolioConfig(BaseModel):
    """Position sizing preferences."""
    total_size: float = 0              # user's total portfolio $
    position_sizing: PositionSizingStrategy = PositionSizingStrategy.FIXED_PERCENT
    position_size_percent: float = Field(default=3.0, ge=0.5, le=25.0)
    position_size_dollar: float = 0    # for FIXED_DOLLAR strategy
    max_concurrent_positions: int = Field(default=5, ge=1, le=50)

    def suggested_size(self, risk_level: str = "MEDIUM") -> float:
        """
        Compute suggested position size in dollars.

        For RISK_BASED strategy, adjusts percent by risk:
            LOW    → position_size_percent × 1.5
            MEDIUM → position_size_percent × 1.0
            HIGH   → position_size_percent × 0.5
        """
        if self.total_size <= 0:
            return 0.0

        if self.position_sizing == PositionSizingStrategy.FIXED_DOLLAR:
            return self.position_size_dollar

        pct = self.position_size_percent
        if self.position_sizing == PositionSizingStrategy.RISK_BASED:
            mult = {"LOW": 1.5, "MEDIUM": 1.0, "HIGH": 0.5}.get(risk_level.upper(), 1.0)
            pct *= mult

        return round(self.total_size * pct / 100, 2)


class NotificationPreferences(BaseModel):
    """What to notify and how."""
    new_recommendations: List[str] = Field(default_factory=lambda: ["push", "email"])
    price_alerts: List[str] = Field(default_factory=lambda: ["push"])
    risk_warnings: List[str] = Field(default_factory=lambda: ["push", "email"])
    frequency: NotificationFrequency = NotificationFrequency.REAL_TIME

    # Granular alert types
    alert_on_entry_zone: bool = True
    alert_on_target_reached: bool = True
    alert_on_stop_loss: bool = True
    alert_on_metrics_deterioration: bool = True
    alert_on_whale_dump: bool = True
    alert_on_liquidity_drop: bool = True
    daily_watchlist_summary: bool = False


class AdvancedFilters(BaseModel):
    """User-defined token screening criteria."""
    token_age: TokenAgeFilter = TokenAgeFilter.ANY
    min_liquidity: float = Field(default=10_000, ge=0)
    min_holders: int = Field(default=500, ge=0)
    min_confidence_score: float = Field(default=0.65, ge=0, le=1)

    # Required conditions (AND logic)
    require_positive_news_24h: bool = False
    require_rsi_range: bool = False
    rsi_min: float = 40
    rsi_max: float = 70
    require_increasing_holders: bool = False
    require_smart_money: bool = False

    # Exclusions
    exclude_already_owned: bool = True
    exclude_flagged_scams: bool = True
    exclude_meme_coins: bool = False


class WatchlistItem(BaseModel):
    """A single token on the user's watchlist."""
    token: str
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    alert_price: Optional[float] = None
    alert_type: AlertType = AlertType.TARGET_REACHED
    notes: str = ""


class PortfolioHolding(BaseModel):
    """A token the user currently holds."""
    token: str
    entry_price: float
    entry_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    quantity: float = 0
    recommendation_id: Optional[str] = None
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# Top-level User Preferences
# ══════════════════════════════════════════════════════════════════

class UserPreferences(BaseModel):
    """
    Complete user preference document.

    Stored per user_id in the database and loaded at the start of
    every request to personalise the recommendation pipeline.

    JSON schema maps directly to the wireframe in Step 4.3.
    """
    user_id: str = ""
    data_sources: DataSourcePreferences = Field(default_factory=DataSourcePreferences)
    risk_profile: RiskProfile = RiskProfile.MODERATE
    trading_style: TradingStyle = TradingStyle.DAY_TRADING
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    notifications: NotificationPreferences = Field(default_factory=NotificationPreferences)
    filters: AdvancedFilters = Field(default_factory=AdvancedFilters)
    watchlist: List[WatchlistItem] = Field(default_factory=list)
    holdings: List[PortfolioHolding] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Convenience helpers ───────────────────────────────────────

    @property
    def held_tickers(self) -> List[str]:
        return [h.token.upper() for h in self.holdings]

    @property
    def watchlist_tickers(self) -> List[str]:
        return [w.token.upper() for w in self.watchlist]

    def max_risk_for_profile(self) -> str:
        """Return the maximum allowed risk level for the user's profile."""
        return {
            RiskProfile.CONSERVATIVE: "LOW",
            RiskProfile.MODERATE: "MEDIUM",
            RiskProfile.AGGRESSIVE: "HIGH",
        }[self.risk_profile]

    def token_age_days_range(self) -> tuple:
        """Return (min_days, max_days) for the token age filter."""
        return {
            TokenAgeFilter.ANY: (0, 999_999),
            TokenAgeFilter.UNDER_7D: (0, 7),
            TokenAgeFilter.UNDER_30D: (0, 30),
            TokenAgeFilter.OVER_30D: (30, 999_999),
        }[self.filters.token_age]


# ══════════════════════════════════════════════════════════════════
# Default Factory
# ══════════════════════════════════════════════════════════════════

def default_preferences(user_id: str = "") -> UserPreferences:
    """Create a UserPreferences instance with sensible defaults."""
    return UserPreferences(user_id=user_id)
