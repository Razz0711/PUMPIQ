"""
NexYpher – User Interface & Experience Layer
=============================================
Phase 4: Query Intent Recognition, Response Formatting,
and User Configuration & Personalization.

Quick-start::

    from src.ui import parse_user_query, ResponseFormatter, PersonalizationEngine

    # 1. Parse a natural-language query
    parsed = parse_user_query("What are the best coins to buy?")
    if not parsed.ready:
        # Need clarification – show parsed.clarification to user
        ...

    # 2. Format recommendations for web / mobile / API
    formatter = ResponseFormatter()
    web_data  = formatter.format_web(rec_set)
    api_json  = formatter.format_api(rec_set)

    # 3. Apply user preferences
    engine = PersonalizationEngine(prefs)
    config = engine.build_config()
    query  = engine.build_query("best coins", intent="discovery")
"""

# ── Step 4.1: Query Intent Recognition ────────────────────────────
from .intent_recognizer import (
    ClassificationResult,
    Intent,
    IntentRecognizer,
)
from .parameter_extractor import (
    AlertCondition,
    ExtractedParams,
    FilterParams,
    ParameterExtractor,
    PriceContext,
)
from .clarification_engine import (
    ClarificationEngine,
    ClarificationOption,
    ClarificationPrompt,
    ClarificationType,
    ParsedQuery,
    parse_user_query,
)

# ── Step 4.2: Response Formatting & Display ───────────────────────
from .response_formatter import OutputFormat, ResponseFormatter
from .visual_indicators import (
    confidence_bar,
    confidence_bar_html,
    confidence_label,
    data_freshness_html,
    data_freshness_indicator,
    risk_badge,
    risk_badge_html,
    score_sparkline,
    trend_arrow,
    trend_arrow_html,
    verdict_colour,
    verdict_emoji,
)
from .api_schemas import (
    AnalysisResponse,
    ClarificationResponse,
    ComparisonResponse,
    DetailedAnalysisBlock,
    EntryExitBlock,
    ErrorResponse,
    MarketContext,
    RecommendationResponse,
    RecommendationSetResponse,
    ResponseMetadata,
    ScoreBlock,
    TokenInfo,
)
from .notification_formatter import (
    Notification,
    NotificationChannel,
    NotificationFormatter,
    NotificationPriority,
    NotificationType,
)

# ── Step 4.3: User Configuration & Personalization ────────────────
from .user_config import (
    AdvancedFilters,
    AlertType,
    DataSourceConfig,
    DataSourcePreferences,
    NotificationFrequency,
    NotificationPreferences,
    PortfolioConfig,
    PortfolioHolding,
    PositionSizingStrategy,
    RiskProfile,
    TokenAgeFilter,
    TradingStyle,
    UserPreferences,
    WatchlistItem,
    default_preferences,
)
from .watchlist_manager import TriggeredAlert, WatchlistManager
from .portfolio_tracker import PortfolioSummary, PortfolioTracker, PositionStatus
from .personalization_engine import PersonalizationEngine

__all__ = [
    # 4.1 – Intent Recognition
    "Intent",
    "IntentRecognizer",
    "ClassificationResult",
    "ParameterExtractor",
    "ExtractedParams",
    "FilterParams",
    "PriceContext",
    "AlertCondition",
    "ClarificationEngine",
    "ClarificationType",
    "ClarificationOption",
    "ClarificationPrompt",
    "ParsedQuery",
    "parse_user_query",
    # 4.2 – Response Formatting
    "OutputFormat",
    "ResponseFormatter",
    "confidence_bar",
    "confidence_bar_html",
    "confidence_label",
    "risk_badge",
    "risk_badge_html",
    "trend_arrow",
    "trend_arrow_html",
    "data_freshness_indicator",
    "data_freshness_html",
    "verdict_colour",
    "verdict_emoji",
    "score_sparkline",
    "NotificationFormatter",
    "Notification",
    "NotificationChannel",
    "NotificationType",
    "NotificationPriority",
    # 4.2 – API Schemas
    "RecommendationSetResponse",
    "RecommendationResponse",
    "AnalysisResponse",
    "ComparisonResponse",
    "ClarificationResponse",
    "ErrorResponse",
    "TokenInfo",
    "ScoreBlock",
    "EntryExitBlock",
    "DetailedAnalysisBlock",
    "MarketContext",
    "ResponseMetadata",
    # 4.3 – User Config
    "UserPreferences",
    "DataSourcePreferences",
    "DataSourceConfig",
    "PortfolioConfig",
    "NotificationPreferences",
    "AdvancedFilters",
    "WatchlistItem",
    "PortfolioHolding",
    "RiskProfile",
    "TradingStyle",
    "PositionSizingStrategy",
    "NotificationFrequency",
    "AlertType",
    "TokenAgeFilter",
    "default_preferences",
    # 4.3 – Watchlist & Portfolio
    "WatchlistManager",
    "TriggeredAlert",
    "PortfolioTracker",
    "PortfolioSummary",
    "PositionStatus",
    # 4.3 – Personalization
    "PersonalizationEngine",
]
