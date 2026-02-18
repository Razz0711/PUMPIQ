"""
NexYpher Social Collector Package
================================
Social sentiment analysis engine for cryptocurrency discussions.

Modules:
- sentiment_analyzer: Core NLP sentiment analysis with crypto-specific lexicon
- twitter_collector: Twitter/X data collection and sentiment scoring
- farcaster_collector: Farcaster protocol analysis with on-chain reputation
- telegram_collector: Telegram/Discord community health analysis
- reddit_collector: Reddit sentiment and engagement analysis
- social_aggregator: Multi-platform score aggregation (0-12 point system)
- bot_detector: Bot/shill/coordinated campaign detection
- influencer_tracker: Influencer identification and weighting
- social_recommendation_engine: SOCIAL_ONLY_MODE recommendation logic
"""

from .sentiment_analyzer import CryptoSentimentAnalyzer, SentimentResult, WeightedSentiment
from .twitter_collector import TwitterCollector, TwitterTokenMetrics
from .farcaster_collector import FarcasterCollector, FarcasterTokenMetrics
from .telegram_collector import TelegramDiscordCollector, CommunityHealthMetrics
from .reddit_collector import RedditCollector, RedditTokenMetrics
from .bot_detector import BotDetector, BotDetectionReport
from .influencer_tracker import InfluencerTracker, InfluencerSignalReport
from .social_aggregator import (
    SocialAggregator,
    SocialScoreReport,
    SocialDataBundle,
    CategoryScore,
)
from .social_recommendation_engine import (
    SocialRecommendationEngine,
    SocialRecommendation,
    Recommendation,
    RiskLevel,
    SignalAction,
)

__all__ = [
    # Sentiment
    "CryptoSentimentAnalyzer",
    "SentimentResult",
    "WeightedSentiment",
    # Platform collectors
    "TwitterCollector",
    "TwitterTokenMetrics",
    "FarcasterCollector",
    "FarcasterTokenMetrics",
    "TelegramDiscordCollector",
    "CommunityHealthMetrics",
    "RedditCollector",
    "RedditTokenMetrics",
    # Bot detection
    "BotDetector",
    "BotDetectionReport",
    # Influencer tracking
    "InfluencerTracker",
    "InfluencerSignalReport",
    # Aggregation
    "SocialAggregator",
    "SocialScoreReport",
    "SocialDataBundle",
    "CategoryScore",
    # Recommendation engine
    "SocialRecommendationEngine",
    "SocialRecommendation",
    "Recommendation",
    "RiskLevel",
    "SignalAction",
]
