"""
Social Score Aggregator
=========================
Aggregates all social signals into a unified 0–12 point score.

Scoring Breakdown:
╔══════════════════════╦═══════════╗
║ Category             ║ Max Points║
╠══════════════════════╬═══════════╣
║ Mention Volume       ║ 0–3       ║
║ Sentiment Quality    ║ 0–3       ║
║ Influencer Signal    ║ 0–2       ║
║ Organic Activity     ║ 0–2       ║
║ Trend Momentum       ║ -2 to +2  ║
╠══════════════════════╬═══════════╣
║ TOTAL (pre-penalty)  ║ 0–12      ║
╠══════════════════════╬═══════════╣
║ Red Flag Deductions  ║ up to -8  ║
╠══════════════════════╬═══════════╣
║ FINAL (floored @ 0)  ║ 0–12      ║
╚══════════════════════╩═══════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .bot_detector import BotDetectionReport
from .influencer_tracker import InfluencerSignalReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SocialScoreCategory(str, Enum):
    MENTION_VOLUME = "mention_volume"
    SENTIMENT_QUALITY = "sentiment_quality"
    INFLUENCER_SIGNAL = "influencer_signal"
    ORGANIC_ACTIVITY = "organic_activity"
    TREND_MOMENTUM = "trend_momentum"


class OverallTone(str, Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CategoryScore:
    """Score for a single aggregation category."""
    category: SocialScoreCategory
    score: float  # Actual score
    max_score: float  # Maximum possible
    details: str = ""


@dataclass
class PlatformSummary:
    """Summary of a single platform's contribution."""
    platform: str
    mentions: int = 0
    avg_sentiment: float = 0.0
    weighted_sentiment: float = 0.0
    influencer_mentions: int = 0
    top_post_preview: str = ""
    engagement_total: int = 0


@dataclass
class SocialScoreReport:
    """
    Complete social score report for a token.

    This is the primary output consumed by the Social Recommendation Engine.
    """
    token_ticker: str

    # Scores
    category_scores: List[CategoryScore] = field(default_factory=list)
    raw_total: float = 0.0         # Sum of category scores (before penalties)
    penalty_total: float = 0.0     # Red-flag deductions
    final_score: float = 0.0       # raw_total - penalty_total, floored at 0

    # Overall tone
    overall_tone: OverallTone = OverallTone.NEUTRAL

    # Per-platform summaries
    platform_summaries: List[PlatformSummary] = field(default_factory=list)

    # Red flag details
    red_flags: List[str] = field(default_factory=list)
    red_flag_count: int = 0

    # Metadata
    total_data_points: int = 0
    analysis_window_hours: int = 24
    confidence: float = 0.0  # 0-1, based on data volume & quality

    generated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Input container (pass in all sub-module outputs)
# ---------------------------------------------------------------------------

@dataclass
class SocialDataBundle:
    """
    Bundle of all social data collected for a token.
    Created by the orchestrator before passing to the aggregator.
    """
    token_ticker: str

    # Platform metrics
    twitter_mentions: int = 0
    twitter_avg_sentiment: float = 0.0
    twitter_weighted_sentiment: float = 0.0
    twitter_influencer_mentions: int = 0
    twitter_engagement: int = 0
    twitter_mention_velocity: float = 0.0
    twitter_unique_authors: int = 0
    twitter_top_text: str = ""

    farcaster_mentions: int = 0
    farcaster_avg_sentiment: float = 0.0
    farcaster_weighted_sentiment: float = 0.0
    farcaster_high_rep_mentions: int = 0
    farcaster_engagement: int = 0
    farcaster_unique_authors: int = 0
    farcaster_top_text: str = ""

    reddit_posts: int = 0
    reddit_avg_sentiment: float = 0.0
    reddit_weighted_sentiment: float = 0.0
    reddit_upvote_ratio: float = 0.0
    reddit_engagement: int = 0
    reddit_unique_authors: int = 0
    reddit_trending_score: float = 0.0
    reddit_top_text: str = ""

    telegram_messages_24h: int = 0
    telegram_avg_sentiment: float = 0.0
    telegram_member_count: int = 0
    telegram_member_growth_24h: float = 0.0
    telegram_activity_level: str = "low"
    telegram_discussion_quality: float = 0.0
    telegram_admin_responsive: str = "unknown"
    telegram_community_tone: str = "neutral"

    discord_messages_24h: int = 0
    discord_avg_sentiment: float = 0.0
    discord_member_count: int = 0
    discord_member_growth_24h: float = 0.0
    discord_activity_level: str = "low"
    discord_discussion_quality: float = 0.0
    discord_admin_responsive: str = "unknown"
    discord_community_tone: str = "neutral"

    # Sub-module reports
    influencer_report: Optional[InfluencerSignalReport] = None
    bot_report: Optional[BotDetectionReport] = None

    # Previous period (for trend momentum)
    prev_twitter_mentions: int = 0
    prev_farcaster_mentions: int = 0
    prev_reddit_posts: int = 0
    prev_total_engagement: int = 0
    prev_avg_sentiment: float = 0.0


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class SocialAggregator:
    """
    Aggregates all social sub-scores into a final 0–12 point score.

    Usage::

        aggregator = SocialAggregator()
        report = aggregator.compute_social_score(bundle)
        print(report.final_score)  # 0–12
    """

    def compute_social_score(self, bundle: SocialDataBundle) -> SocialScoreReport:
        """Compute the unified social score from all social data."""
        category_scores: List[CategoryScore] = []

        # 1. Mention Volume (0-3)
        mv = self._score_mention_volume(bundle)
        category_scores.append(mv)

        # 2. Sentiment Quality (0-3)
        sq = self._score_sentiment_quality(bundle)
        category_scores.append(sq)

        # 3. Influencer Signal (0-2)
        ins = self._score_influencer_signal(bundle)
        category_scores.append(ins)

        # 4. Organic Activity (0-2)
        oa = self._score_organic_activity(bundle)
        category_scores.append(oa)

        # 5. Trend Momentum (-2 to +2)
        tm = self._score_trend_momentum(bundle)
        category_scores.append(tm)

        # Raw total
        raw_total = sum(cs.score for cs in category_scores)

        # Red-flag penalties from bot detector
        penalty = 0.0
        red_flags: List[str] = []
        if bundle.bot_report:
            penalty = bundle.bot_report.total_penalty
            for campaign in bundle.bot_report.campaigns:
                red_flags.extend(campaign.evidence)
            if bundle.bot_report.pump_group_detected:
                red_flags.append("⚠️ Pump group activity detected (-3)")
            if bundle.bot_report.coordinated_shilling:
                red_flags.append("⚠️ Coordinated shilling detected (-2)")
            if bundle.bot_report.fake_engagement_detected:
                red_flags.append("⚠️ Fake engagement patterns detected (-2)")
            if bundle.bot_report.suspicious_promotions:
                red_flags.append("⚠️ Suspicious influencer promotions (-1)")

        # Community red flags (from Telegram/Discord)
        if bundle.telegram_admin_responsive == "unresponsive":
            red_flags.append("⚠️ Telegram admins unresponsive")
            penalty += 0.5
        if bundle.discord_admin_responsive == "unresponsive":
            red_flags.append("⚠️ Discord admins unresponsive")
            penalty += 0.5

        # Final score
        final_score = max(0.0, min(12.0, raw_total - penalty))

        # Overall tone
        overall_tone = self._determine_tone(final_score, bundle)

        # Platform summaries
        platforms = self._build_platform_summaries(bundle)

        # Data points & confidence
        total_points = (
            bundle.twitter_mentions + bundle.farcaster_mentions
            + bundle.reddit_posts + bundle.telegram_messages_24h
            + bundle.discord_messages_24h
        )
        confidence = self._compute_confidence(total_points, bundle)

        return SocialScoreReport(
            token_ticker=bundle.token_ticker,
            category_scores=category_scores,
            raw_total=round(raw_total, 2),
            penalty_total=round(penalty, 2),
            final_score=round(final_score, 2),
            overall_tone=overall_tone,
            platform_summaries=platforms,
            red_flags=red_flags,
            red_flag_count=len(red_flags),
            total_data_points=total_points,
            confidence=round(confidence, 2),
            generated_at=datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Category 1: Mention Volume (0-3)
    # ------------------------------------------------------------------

    def _score_mention_volume(self, b: SocialDataBundle) -> CategoryScore:
        """
        Score based on total mention volume across platforms.

        Thresholds:
        - 0 mentions: 0 pts
        - 1-50: 0.5 pts
        - 51-200: 1.0 pts
        - 201-500: 1.5 pts
        - 501-1000: 2.0 pts
        - 1001-5000: 2.5 pts
        - 5000+: 3.0 pts
        """
        total = (
            b.twitter_mentions + b.farcaster_mentions
            + b.reddit_posts + b.telegram_messages_24h
            + b.discord_messages_24h
        )

        if total == 0:
            score = 0.0
        elif total <= 50:
            score = 0.5
        elif total <= 200:
            score = 1.0
        elif total <= 500:
            score = 1.5
        elif total <= 1000:
            score = 2.0
        elif total <= 5000:
            score = 2.5
        else:
            score = 3.0

        # Bonus for multi-platform coverage (max +0.3)
        platforms_active = sum(1 for x in [
            b.twitter_mentions, b.farcaster_mentions,
            b.reddit_posts, b.telegram_messages_24h,
            b.discord_messages_24h,
        ] if x > 0)
        if platforms_active >= 4:
            score = min(3.0, score + 0.3)
        elif platforms_active >= 3:
            score = min(3.0, score + 0.15)

        return CategoryScore(
            category=SocialScoreCategory.MENTION_VOLUME,
            score=round(score, 2),
            max_score=3.0,
            details=f"{total} total mentions across {platforms_active} platforms",
        )

    # ------------------------------------------------------------------
    # Category 2: Sentiment Quality (0-3)
    # ------------------------------------------------------------------

    def _score_sentiment_quality(self, b: SocialDataBundle) -> CategoryScore:
        """
        Score based on weighted sentiment across all platforms.

        Maps weighted average sentiment (-10 to +10) to 0-3 score.
        """
        # Compute a weighted average of all platform sentiments
        weights_and_scores = []
        if b.twitter_mentions > 0:
            weights_and_scores.append((b.twitter_mentions, b.twitter_weighted_sentiment))
        if b.farcaster_mentions > 0:
            weights_and_scores.append((b.farcaster_mentions, b.farcaster_weighted_sentiment))
        if b.reddit_posts > 0:
            weights_and_scores.append((b.reddit_posts, b.reddit_weighted_sentiment))
        if b.telegram_messages_24h > 0:
            weights_and_scores.append((b.telegram_messages_24h, b.telegram_avg_sentiment))
        if b.discord_messages_24h > 0:
            weights_and_scores.append((b.discord_messages_24h, b.discord_avg_sentiment))

        if not weights_and_scores:
            return CategoryScore(
                category=SocialScoreCategory.SENTIMENT_QUALITY,
                score=0.0, max_score=3.0,
                details="No sentiment data",
            )

        total_weight = sum(w for w, _ in weights_and_scores)
        weighted_avg = sum(w * s for w, s in weights_and_scores) / total_weight

        # Map -10..+10 → 0..3
        # -10 → 0, 0 → 1.5, +10 → 3
        score = ((weighted_avg + 10) / 20) * 3.0
        score = max(0.0, min(3.0, score))

        # Quality bonus: if community discussion quality is high, add 0.2
        avg_quality = 0.0
        quality_count = 0
        if b.telegram_discussion_quality > 0:
            avg_quality += b.telegram_discussion_quality
            quality_count += 1
        if b.discord_discussion_quality > 0:
            avg_quality += b.discord_discussion_quality
            quality_count += 1
        if quality_count > 0:
            avg_quality /= quality_count
            if avg_quality >= 7:
                score = min(3.0, score + 0.2)

        return CategoryScore(
            category=SocialScoreCategory.SENTIMENT_QUALITY,
            score=round(score, 2),
            max_score=3.0,
            details=f"Weighted avg sentiment: {weighted_avg:+.2f} / 10",
        )

    # ------------------------------------------------------------------
    # Category 3: Influencer Signal (0-2)
    # ------------------------------------------------------------------

    def _score_influencer_signal(self, b: SocialDataBundle) -> CategoryScore:
        """
        Score based on influencer mentions and credibility-weighted sentiment.
        Directly uses the InfluencerSignalReport score (already 0-2).
        """
        if b.influencer_report and b.influencer_report.influencer_signal_score > 0:
            score = b.influencer_report.influencer_signal_score
            detail = (
                f"{b.influencer_report.unique_influencers} influencers, "
                f"consensus: {b.influencer_report.consensus}"
            )
        else:
            # Fallback: derive from platform data
            total_inf = (
                b.twitter_influencer_mentions
                + b.farcaster_high_rep_mentions
            )
            if total_inf == 0:
                score = 0.0
            elif total_inf <= 2:
                score = 0.5
            elif total_inf <= 5:
                score = 1.0
            elif total_inf <= 10:
                score = 1.5
            else:
                score = 2.0
            detail = f"{total_inf} influencer mentions (fallback scoring)"

        return CategoryScore(
            category=SocialScoreCategory.INFLUENCER_SIGNAL,
            score=round(min(2.0, score), 2),
            max_score=2.0,
            details=detail,
        )

    # ------------------------------------------------------------------
    # Category 4: Organic Activity (0-2)
    # ------------------------------------------------------------------

    def _score_organic_activity(self, b: SocialDataBundle) -> CategoryScore:
        """
        Score based on how organic the activity appears.
        Higher score = more organic (diverse authors, genuine discussion).
        Bot presence reduces this score.
        """
        score = 1.0  # Baseline

        # Unique author diversity
        total_mentions = max(
            b.twitter_mentions + b.farcaster_mentions + b.reddit_posts, 1
        )
        total_unique = (
            b.twitter_unique_authors + b.farcaster_unique_authors
            + b.reddit_unique_authors
        )
        diversity_ratio = total_unique / total_mentions if total_mentions > 0 else 0

        if diversity_ratio > 0.7:
            score += 0.5  # Very diverse = organic
        elif diversity_ratio > 0.4:
            score += 0.2
        elif diversity_ratio < 0.2:
            score -= 0.5  # Few authors = suspicious

        # Community health
        tg_healthy = b.telegram_activity_level in ("medium", "high") and b.telegram_discussion_quality >= 5
        dc_healthy = b.discord_activity_level in ("medium", "high") and b.discord_discussion_quality >= 5
        if tg_healthy or dc_healthy:
            score += 0.3

        # Bot penalty
        if b.bot_report:
            if b.bot_report.bot_percentage > 30:
                score -= 0.8
            elif b.bot_report.bot_percentage > 15:
                score -= 0.4

        score = max(0.0, min(2.0, score))

        return CategoryScore(
            category=SocialScoreCategory.ORGANIC_ACTIVITY,
            score=round(score, 2),
            max_score=2.0,
            details=f"Author diversity: {diversity_ratio:.0%}, "
                    f"Bot %: {b.bot_report.bot_percentage if b.bot_report else 0:.1f}%",
        )

    # ------------------------------------------------------------------
    # Category 5: Trend Momentum (-2 to +2)
    # ------------------------------------------------------------------

    def _score_trend_momentum(self, b: SocialDataBundle) -> CategoryScore:
        """
        Score based on momentum: is social activity increasing or decreasing?
        Compares current period to previous period.
        """
        current_mentions = (
            b.twitter_mentions + b.farcaster_mentions + b.reddit_posts
        )
        prev_mentions = (
            b.prev_twitter_mentions + b.prev_farcaster_mentions + b.prev_reddit_posts
        )

        if prev_mentions == 0 and current_mentions == 0:
            mention_momentum = 0.0
        elif prev_mentions == 0:
            mention_momentum = 2.0  # New trend appearing
        else:
            change_pct = ((current_mentions - prev_mentions) / prev_mentions) * 100
            if change_pct > 200:
                mention_momentum = 2.0
            elif change_pct > 100:
                mention_momentum = 1.5
            elif change_pct > 50:
                mention_momentum = 1.0
            elif change_pct > 10:
                mention_momentum = 0.5
            elif change_pct > -10:
                mention_momentum = 0.0
            elif change_pct > -50:
                mention_momentum = -0.5
            elif change_pct > -75:
                mention_momentum = -1.0
            else:
                mention_momentum = -2.0

        # Sentiment momentum
        sentiment_change = b.twitter_weighted_sentiment - b.prev_avg_sentiment
        if sentiment_change > 3:
            sentiment_momentum = 0.5
        elif sentiment_change > 1:
            sentiment_momentum = 0.2
        elif sentiment_change < -3:
            sentiment_momentum = -0.5
        elif sentiment_change < -1:
            sentiment_momentum = -0.2
        else:
            sentiment_momentum = 0.0

        score = mention_momentum + sentiment_momentum
        score = max(-2.0, min(2.0, score))

        return CategoryScore(
            category=SocialScoreCategory.TREND_MOMENTUM,
            score=round(score, 2),
            max_score=2.0,
            details=f"Mention change: {current_mentions} vs {prev_mentions} prev, "
                    f"Sentiment Δ: {sentiment_change:+.2f}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _determine_tone(
        self, final_score: float, b: SocialDataBundle
    ) -> OverallTone:
        """Map final score to an overall tone label."""
        if final_score >= 9:
            return OverallTone.STRONG_BULLISH
        elif final_score >= 6:
            return OverallTone.BULLISH
        elif final_score >= 4:
            return OverallTone.NEUTRAL
        elif final_score >= 2:
            return OverallTone.BEARISH
        else:
            return OverallTone.STRONG_BEARISH

    def _build_platform_summaries(
        self, b: SocialDataBundle
    ) -> List[PlatformSummary]:
        """Build per-platform summary objects."""
        summaries: List[PlatformSummary] = []

        if b.twitter_mentions > 0:
            summaries.append(PlatformSummary(
                platform="Twitter/X",
                mentions=b.twitter_mentions,
                avg_sentiment=b.twitter_avg_sentiment,
                weighted_sentiment=b.twitter_weighted_sentiment,
                influencer_mentions=b.twitter_influencer_mentions,
                top_post_preview=b.twitter_top_text[:120],
                engagement_total=b.twitter_engagement,
            ))
        if b.farcaster_mentions > 0:
            summaries.append(PlatformSummary(
                platform="Farcaster",
                mentions=b.farcaster_mentions,
                avg_sentiment=b.farcaster_avg_sentiment,
                weighted_sentiment=b.farcaster_weighted_sentiment,
                influencer_mentions=b.farcaster_high_rep_mentions,
                top_post_preview=b.farcaster_top_text[:120],
                engagement_total=b.farcaster_engagement,
            ))
        if b.reddit_posts > 0:
            summaries.append(PlatformSummary(
                platform="Reddit",
                mentions=b.reddit_posts,
                avg_sentiment=b.reddit_avg_sentiment,
                weighted_sentiment=b.reddit_weighted_sentiment,
                top_post_preview=b.reddit_top_text[:120],
                engagement_total=b.reddit_engagement,
            ))
        if b.telegram_messages_24h > 0:
            summaries.append(PlatformSummary(
                platform="Telegram",
                mentions=b.telegram_messages_24h,
                avg_sentiment=b.telegram_avg_sentiment,
            ))
        if b.discord_messages_24h > 0:
            summaries.append(PlatformSummary(
                platform="Discord",
                mentions=b.discord_messages_24h,
                avg_sentiment=b.discord_avg_sentiment,
            ))

        return summaries

    def _compute_confidence(
        self, total_points: int, b: SocialDataBundle
    ) -> float:
        """
        How confident are we in this score?

        Higher confidence when:
        - More data points
        - Multiple platforms
        - Low bot percentage
        - High author diversity
        """
        confidence = 0.0

        # Data volume (max 0.4)
        if total_points >= 500:
            confidence += 0.4
        elif total_points >= 100:
            confidence += 0.3
        elif total_points >= 30:
            confidence += 0.2
        elif total_points >= 5:
            confidence += 0.1

        # Platform coverage (max 0.3)
        platforms = sum(1 for x in [
            b.twitter_mentions, b.farcaster_mentions,
            b.reddit_posts, b.telegram_messages_24h,
            b.discord_messages_24h,
        ] if x > 0)
        confidence += min(0.3, platforms * 0.075)

        # Bot cleanliness (max 0.2)
        if b.bot_report:
            if b.bot_report.bot_percentage < 5:
                confidence += 0.2
            elif b.bot_report.bot_percentage < 15:
                confidence += 0.1
        else:
            confidence += 0.1  # No bot data → moderate trust

        # Author diversity (max 0.1)
        total_unique = (
            b.twitter_unique_authors + b.farcaster_unique_authors
            + b.reddit_unique_authors
        )
        if total_unique >= 50:
            confidence += 0.1
        elif total_unique >= 10:
            confidence += 0.05

        return min(1.0, confidence)
