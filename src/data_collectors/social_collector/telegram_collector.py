"""
Telegram / Discord Community Health Collector
===============================================
Analyzes community health from messaging platforms.

Scoring Dimensions:
- Message frequency & tone
- Admin responsiveness & transparency
- Member growth rate
- Quality of discussion (technical vs "wen moon")
- Activity level relative to member count
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .sentiment_analyzer import CryptoSentimentAnalyzer, SentimentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring Constants
# ---------------------------------------------------------------------------

# Quality keywords (higher quality = better score)
QUALITY_KEYWORDS = {
    "technical", "roadmap", "development", "audit", "partnership",
    "update", "release", "milestone", "smart contract", "utility",
    "governance", "staking", "tokenomics", "liquidity", "deployment",
    "github", "commit", "integration", "protocol", "mainnet", "testnet",
}

# Low-quality / hype-only keywords
HYPE_KEYWORDS = {
    "wen moon", "wen lambo", "1000x", "100x", "easy money",
    "free money", "airdrop", "pump it", "buy now", "last chance",
    "going to the moon", "to the moon", "lfg", "get in now",
}

# FUD / red-flag keywords from community
RED_FLAG_KEYWORDS = {
    "rug", "scam", "where is dev", "dev left", "abandoned",
    "no update", "team silent", "admin muted", "can't sell",
    "honeypot", "locked", "exit scam", "ponzi",
}


@dataclass
class CommunityMessage:
    """A single message from Telegram or Discord."""
    message_id: str
    text: str
    author_id: str
    author_username: str
    is_admin: bool = False
    is_bot: bool = False
    timestamp: Optional[datetime] = None
    platform: str = "telegram"  # "telegram" or "discord"
    reply_to: Optional[str] = None


@dataclass
class CommunityHealthMetrics:
    """Comprehensive community health assessment."""
    token_ticker: str
    platform: str = "telegram"

    # Member metrics
    total_members: int = 0
    online_members: int = 0
    member_growth_24h: float = 0.0   # % growth
    member_growth_7d: float = 0.0    # % growth

    # Activity metrics
    messages_24h: int = 0
    messages_7d: int = 0
    unique_chatters_24h: int = 0
    messages_per_member: float = 0.0  # activity ratio
    activity_level: str = "low"       # low / medium / high

    # Sentiment metrics
    avg_sentiment: float = 0.0        # -10 to +10
    positive_message_pct: float = 0.0
    negative_message_pct: float = 0.0
    neutral_message_pct: float = 0.0

    # Quality metrics
    discussion_quality: float = 0.0    # 0-10 scale
    technical_discussion_pct: float = 0.0
    hype_discussion_pct: float = 0.0
    quality_label: str = "unknown"     # "high", "medium", "low"

    # Admin metrics
    admin_messages_24h: int = 0
    admin_response_rate: float = 0.0   # % of questions answered
    admin_responsiveness: str = "unknown"  # "responsive", "moderate", "unresponsive"

    # Red flags
    red_flags: List[str] = field(default_factory=list)
    red_flag_count: int = 0
    fud_message_pct: float = 0.0

    # Tone summary
    community_tone: str = "neutral"    # "bullish", "neutral", "bearish", "fearful"

    collected_at: Optional[datetime] = None


class TelegramDiscordCollector:
    """
    Collects and analyses community health from Telegram and Discord.

    Activity Scoring:
    - High activity + positive tone = bullish signal
    - Low activity or fearful tone = bearish signal
    - Admin FUD or no communication = red flag
    """

    def __init__(
        self,
        telegram_api_token: Optional[str] = None,
        discord_bot_token: Optional[str] = None,
        analyzer: Optional[CryptoSentimentAnalyzer] = None,
    ):
        self.telegram_token = telegram_api_token
        self.discord_token = discord_bot_token
        self.analyzer = analyzer or CryptoSentimentAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def collect_community_health(
        self,
        token_ticker: str,
        group_id: str,
        platform: str = "telegram",
        member_count: int = 0,
        member_count_yesterday: int = 0,
        member_count_last_week: int = 0,
    ) -> CommunityHealthMetrics:
        """
        Collect and analyze community health.

        Args:
            token_ticker: Token symbol
            group_id: Telegram group ID or Discord guild ID
            platform: "telegram" or "discord"
            member_count: Current member count
            member_count_yesterday: Member count 24h ago
            member_count_last_week: Member count 7d ago

        Returns:
            CommunityHealthMetrics with all computed scores
        """
        # Fetch messages (last 24h and 7d)
        messages_24h = await self._fetch_messages(group_id, platform, hours=24)
        messages_7d = await self._fetch_messages(group_id, platform, hours=168)

        # Analyze
        return self._analyze_community(
            token_ticker=token_ticker,
            platform=platform,
            messages_24h=messages_24h,
            messages_7d=messages_7d,
            member_count=member_count,
            member_count_yesterday=member_count_yesterday,
            member_count_last_week=member_count_last_week,
        )

    # ------------------------------------------------------------------
    # Internal – Message Fetching
    # ------------------------------------------------------------------

    async def _fetch_messages(
        self, group_id: str, platform: str, hours: int
    ) -> List[CommunityMessage]:
        """
        Fetch messages from a Telegram group or Discord channel.
        Production implementation uses respective bot APIs.
        """
        # Placeholder – in production, integrate with:
        # - Telegram Bot API: getUpdates / getChat + getChatHistory
        # - Discord.py: fetch_message_history
        logger.info(
            f"Fetching {platform} messages for group {group_id} (last {hours}h)"
        )
        return []

    # ------------------------------------------------------------------
    # Internal – Analysis
    # ------------------------------------------------------------------

    def _analyze_community(
        self,
        token_ticker: str,
        platform: str,
        messages_24h: List[CommunityMessage],
        messages_7d: List[CommunityMessage],
        member_count: int,
        member_count_yesterday: int,
        member_count_last_week: int,
    ) -> CommunityHealthMetrics:
        """Run full community health analysis."""

        metrics = CommunityHealthMetrics(
            token_ticker=token_ticker,
            platform=platform,
            total_members=member_count,
            collected_at=datetime.utcnow(),
        )

        # ---------- Member growth ----------
        if member_count_yesterday > 0:
            metrics.member_growth_24h = round(
                ((member_count - member_count_yesterday) / member_count_yesterday) * 100, 2
            )
        if member_count_last_week > 0:
            metrics.member_growth_7d = round(
                ((member_count - member_count_last_week) / member_count_last_week) * 100, 2
            )

        # ---------- Activity metrics ----------
        # Filter out bot messages
        human_24h = [m for m in messages_24h if not m.is_bot]
        human_7d = [m for m in messages_7d if not m.is_bot]

        metrics.messages_24h = len(human_24h)
        metrics.messages_7d = len(human_7d)
        metrics.unique_chatters_24h = len({m.author_id for m in human_24h})

        if member_count > 0:
            metrics.messages_per_member = round(
                metrics.messages_24h / member_count, 4
            )

        # Activity level classification
        if metrics.messages_per_member > 0.10:
            metrics.activity_level = "high"
        elif metrics.messages_per_member > 0.03:
            metrics.activity_level = "medium"
        else:
            metrics.activity_level = "low"

        # ---------- Sentiment analysis ----------
        sentiments: List[SentimentResult] = []
        for msg in human_24h:
            s = self.analyzer.analyze(msg.text)
            sentiments.append(s)

        if sentiments:
            metrics.avg_sentiment = round(
                sum(s.raw_score for s in sentiments) / len(sentiments), 2
            )
            positive = sum(1 for s in sentiments if s.raw_score > 1.5)
            negative = sum(1 for s in sentiments if s.raw_score < -1.5)
            neutral = len(sentiments) - positive - negative

            total = len(sentiments)
            metrics.positive_message_pct = round(positive / total * 100, 1)
            metrics.negative_message_pct = round(negative / total * 100, 1)
            metrics.neutral_message_pct = round(neutral / total * 100, 1)

        # ---------- Discussion quality ----------
        quality_score = self._assess_discussion_quality(human_24h)
        metrics.discussion_quality = quality_score
        metrics.quality_label = (
            "high" if quality_score >= 7
            else "medium" if quality_score >= 4
            else "low"
        )

        # Technical vs hype breakdown
        tech_count = 0
        hype_count = 0
        for msg in human_24h:
            text_lower = msg.text.lower()
            if any(kw in text_lower for kw in QUALITY_KEYWORDS):
                tech_count += 1
            if any(kw in text_lower for kw in HYPE_KEYWORDS):
                hype_count += 1

        if human_24h:
            metrics.technical_discussion_pct = round(
                tech_count / len(human_24h) * 100, 1
            )
            metrics.hype_discussion_pct = round(
                hype_count / len(human_24h) * 100, 1
            )

        # ---------- Admin analysis ----------
        admin_msgs = [m for m in human_24h if m.is_admin]
        metrics.admin_messages_24h = len(admin_msgs)

        # Check for questions without admin responses
        questions = [m for m in human_24h if "?" in m.text and not m.is_admin]
        if questions:
            answered = sum(
                1 for q in questions
                if any(
                    a.is_admin and a.reply_to == q.message_id
                    for a in human_24h
                )
            )
            metrics.admin_response_rate = round(answered / len(questions) * 100, 1)
        else:
            metrics.admin_response_rate = 100.0  # No questions = not applicable

        if metrics.admin_response_rate >= 70:
            metrics.admin_responsiveness = "responsive"
        elif metrics.admin_response_rate >= 30:
            metrics.admin_responsiveness = "moderate"
        else:
            metrics.admin_responsiveness = "unresponsive"

        # ---------- Red flags ----------
        metrics.red_flags = self._detect_community_red_flags(
            human_24h, admin_msgs, metrics
        )
        metrics.red_flag_count = len(metrics.red_flags)

        # FUD percentage
        fud_count = sum(
            1 for msg in human_24h
            if any(kw in msg.text.lower() for kw in RED_FLAG_KEYWORDS)
        )
        if human_24h:
            metrics.fud_message_pct = round(fud_count / len(human_24h) * 100, 1)

        # ---------- Community tone ----------
        if metrics.avg_sentiment > 3:
            metrics.community_tone = "bullish"
        elif metrics.avg_sentiment > 0:
            metrics.community_tone = "neutral"
        elif metrics.avg_sentiment > -3:
            metrics.community_tone = "cautious"
        else:
            metrics.community_tone = "fearful"

        return metrics

    def _assess_discussion_quality(self, messages: List[CommunityMessage]) -> float:
        """
        Score discussion quality from 0-10.

        Higher quality:
        - Technical discussion, roadmap talk, development updates
        - Longer messages with substance
        - Questions about technology, not just price

        Lower quality:
        - Pure price speculation ("wen moon")
        - Very short messages
        - Repetitive shilling
        """
        if not messages:
            return 0.0

        score = 5.0  # baseline

        # Average message length
        avg_len = sum(len(m.text) for m in messages) / len(messages)
        if avg_len > 100:
            score += 1.5
        elif avg_len > 50:
            score += 0.5
        elif avg_len < 20:
            score -= 1.5

        # Technical content
        tech_count = sum(
            1 for m in messages
            if any(kw in m.text.lower() for kw in QUALITY_KEYWORDS)
        )
        tech_ratio = tech_count / len(messages) if messages else 0
        score += min(2.0, tech_ratio * 10)

        # Hype content (negative)
        hype_count = sum(
            1 for m in messages
            if any(kw in m.text.lower() for kw in HYPE_KEYWORDS)
        )
        hype_ratio = hype_count / len(messages) if messages else 0
        score -= min(2.0, hype_ratio * 8)

        # Unique posters (diverse discussion = better)
        unique_authors = len({m.author_id for m in messages})
        if unique_authors > 20:
            score += 1.0
        elif unique_authors < 5:
            score -= 1.0

        # Red flag content (strong negative)
        red_count = sum(
            1 for m in messages
            if any(kw in m.text.lower() for kw in RED_FLAG_KEYWORDS)
        )
        red_ratio = red_count / len(messages) if messages else 0
        score -= min(3.0, red_ratio * 15)

        return round(max(0, min(10, score)), 1)

    def _detect_community_red_flags(
        self,
        messages: List[CommunityMessage],
        admin_messages: List[CommunityMessage],
        metrics: CommunityHealthMetrics,
    ) -> List[str]:
        """Detect red flags in community activity."""
        flags: List[str] = []

        # No admin activity in 24h
        if not admin_messages:
            flags.append("No admin activity in 24 hours")

        # Admin posting FUD
        for msg in admin_messages:
            text_lower = msg.text.lower()
            if any(kw in text_lower for kw in {"selling", "exit", "leaving", "stepping down"}):
                flags.append(f"Admin potentially posting concerning content: '{msg.text[:80]}...'")

        # High FUD percentage
        if metrics.fud_message_pct > 30:
            flags.append(f"High FUD content ({metrics.fud_message_pct}% of messages)")

        # Very low activity despite having members
        if metrics.total_members > 500 and metrics.messages_24h < 10:
            flags.append("Very low activity relative to member count (possible dead community)")

        # Rapid member decline
        if metrics.member_growth_24h < -5:
            flags.append(f"Members declining rapidly ({metrics.member_growth_24h}% in 24h)")

        # Unresponsive admins
        if metrics.admin_responsiveness == "unresponsive":
            flags.append("Admin team is unresponsive to community questions")

        # Mostly hype, no substance
        if metrics.hype_discussion_pct > 60 and metrics.technical_discussion_pct < 5:
            flags.append("Community is mostly hype with no technical discussion")

        return flags
