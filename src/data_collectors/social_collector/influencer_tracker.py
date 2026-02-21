"""
Influencer Tracker
====================
Identifies and tracks crypto influencers mentioning tokens.

Influence Weighting:
- Mega influencer (100K+ followers):   15x weight
- Major influencer (50K-100K):         10x weight
- Influencer (10K-50K):                5x weight
- Mid-tier (1K-10K):                   2x weight
- Micro (<1K):                         1x weight

Credibility is adjusted by historical accuracy of their calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

INFLUENCE_TIERS: List[Tuple[str, int, float]] = [
    ("mega",  100_000, 15.0),
    ("major",  50_000, 10.0),
    ("influencer", 10_000, 5.0),
    ("mid_tier",  1_000, 2.0),
    ("micro",        0, 1.0),
]


def get_influence_tier(follower_count: int) -> Tuple[str, float]:
    """Return tier name and weight for an author's follower count."""
    for tier_name, threshold, weight in INFLUENCE_TIERS:
        if follower_count >= threshold:
            return tier_name, weight
    return "micro", 1.0


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class InfluencerProfile:
    """Cached profile of a known influencer."""
    author_id: str
    username: str
    platform: str = "twitter"
    follower_count: int = 0
    tier: str = "micro"
    weight: float = 1.0

    # Historical accuracy
    total_calls: int = 0
    successful_calls: int = 0  # Price went in predicted direction
    accuracy: float = 0.0      # 0-1
    credibility_score: float = 1.0  # Adjusted weight

    # On-chain (Farcaster)
    onchain_reputation: float = 0.0

    last_updated: Optional[datetime] = None


@dataclass
class InfluencerMention:
    """A single mention of a token by an influencer."""
    influencer: InfluencerProfile
    text: str
    sentiment_score: float = 0.0
    engagement: int = 0
    timestamp: Optional[datetime] = None
    platform: str = "twitter"


@dataclass
class InfluencerSignalReport:
    """Aggregated influencer signal for a token."""
    token_ticker: str

    # Counts
    total_influencer_mentions: int = 0
    unique_influencers: int = 0
    mega_influencer_mentions: int = 0
    major_influencer_mentions: int = 0

    # Sentiment
    influencer_avg_sentiment: float = 0.0
    credibility_weighted_sentiment: float = 0.0

    # Top mentions
    top_mentions: List[InfluencerMention] = field(default_factory=list)

    # Influencer consensus
    bullish_influencers: int = 0
    bearish_influencers: int = 0
    neutral_influencers: int = 0
    consensus: str = "mixed"  # "strong_bullish", "bullish", "mixed", "bearish", "strong_bearish"

    # Signal score for aggregator (0-2 range)
    influencer_signal_score: float = 0.0

    # Total reach (sum of followers of all mentioning influencers)
    total_reach: int = 0

    generated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class InfluencerTracker:
    """
    Tracks influencer mentions of crypto tokens and computes
    credibility-weighted sentiment signals.

    The tracker maintains a registry of known influencers with historical
    accuracy. New influencers are profiled on first encounter.

    Usage::

        tracker = InfluencerTracker()
        signal = tracker.compute_influencer_signal(
            token_ticker="WIF",
            mentions=[...],
        )
        print(signal.influencer_signal_score)  # 0-2 for aggregator
    """

    def __init__(self):
        # Persistent influencer registry (production: backed by DB)
        self._registry: Dict[str, InfluencerProfile] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_influencer(
        self,
        author_id: str,
        username: str,
        platform: str,
        follower_count: int,
        total_calls: int = 0,
        successful_calls: int = 0,
        onchain_reputation: float = 0.0,
    ) -> InfluencerProfile:
        """Register or update an influencer in the tracker."""
        tier, weight = get_influence_tier(follower_count)

        accuracy = (
            successful_calls / total_calls if total_calls > 0 else 0.5
        )
        # Credibility = base weight * accuracy modifier
        # New influencers get neutral (1.0) modifier
        # Accurate influencers get up to 1.5x boost
        # Inaccurate influencers get as low as 0.5x penalty
        if total_calls >= 5:
            accuracy_modifier = 0.5 + accuracy  # 0.5-1.5 range
        else:
            accuracy_modifier = 1.0  # Not enough data; neutral

        credibility = round(weight * accuracy_modifier, 2)

        profile = InfluencerProfile(
            author_id=author_id,
            username=username,
            platform=platform,
            follower_count=follower_count,
            tier=tier,
            weight=weight,
            total_calls=total_calls,
            successful_calls=successful_calls,
            accuracy=round(accuracy, 3),
            credibility_score=credibility,
            onchain_reputation=onchain_reputation,
            last_updated=datetime.now(timezone.utc),
        )
        self._registry[author_id] = profile
        return profile

    def get_or_create_profile(
        self,
        author_id: str,
        username: str = "",
        platform: str = "twitter",
        follower_count: int = 0,
    ) -> InfluencerProfile:
        """Retrieve existing profile or create a new one."""
        if author_id in self._registry:
            profile = self._registry[author_id]
            # Update follower count if changed significantly
            if abs(profile.follower_count - follower_count) > 100:
                profile.follower_count = follower_count
                profile.tier, profile.weight = get_influence_tier(follower_count)
                profile.last_updated = datetime.now(timezone.utc)
            return profile

        return self.register_influencer(
            author_id=author_id,
            username=username,
            platform=platform,
            follower_count=follower_count,
        )

    def compute_influencer_signal(
        self,
        token_ticker: str,
        mentions: List[Dict[str, Any]],
    ) -> InfluencerSignalReport:
        """
        Compute aggregated influencer signal.

        Args:
            token_ticker: Token symbol
            mentions: List of dicts with keys:
                author_id, author_username, follower_count, platform,
                text, sentiment_score (float -10 to +10), engagement (int),
                timestamp (datetime)

        Returns:
            InfluencerSignalReport with 0-2 signal score
        """
        # Filter to influencer-tier (1K+ followers) only
        influencer_mentions = [
            m for m in mentions
            if m.get("follower_count", 0) >= 1_000
        ]

        if not influencer_mentions:
            return InfluencerSignalReport(
                token_ticker=token_ticker, generated_at=datetime.now(timezone.utc)
            )

        # Build InfluencerMention objects
        mention_objects: List[InfluencerMention] = []
        unique_influencers: Dict[str, InfluencerProfile] = {}
        total_reach = 0

        for m in influencer_mentions:
            profile = self.get_or_create_profile(
                author_id=m.get("author_id", ""),
                username=m.get("author_username", ""),
                platform=m.get("platform", "twitter"),
                follower_count=m.get("follower_count", 0),
            )
            unique_influencers[profile.author_id] = profile
            total_reach += profile.follower_count

            mention_objects.append(
                InfluencerMention(
                    influencer=profile,
                    text=m.get("text", ""),
                    sentiment_score=m.get("sentiment_score", 0),
                    engagement=m.get("engagement", 0),
                    timestamp=m.get("timestamp"),
                    platform=m.get("platform", "twitter"),
                )
            )

        # Compute sentiment metrics
        raw_sentiments = [m.sentiment_score for m in mention_objects]
        avg_sentiment = sum(raw_sentiments) / len(raw_sentiments)

        # Credibility-weighted sentiment
        weighted_sum = sum(
            m.sentiment_score * m.influencer.credibility_score
            for m in mention_objects
        )
        total_credibility = sum(
            m.influencer.credibility_score for m in mention_objects
        )
        cred_weighted = weighted_sum / total_credibility if total_credibility else 0

        # Consensus
        bullish = sum(1 for s in raw_sentiments if s > 2)
        bearish = sum(1 for s in raw_sentiments if s < -2)
        neutral = len(raw_sentiments) - bullish - bearish

        total = len(raw_sentiments)
        bullish_pct = bullish / total
        bearish_pct = bearish / total

        if bullish_pct > 0.7:
            consensus = "strong_bullish"
        elif bullish_pct > 0.5:
            consensus = "bullish"
        elif bearish_pct > 0.7:
            consensus = "strong_bearish"
        elif bearish_pct > 0.5:
            consensus = "bearish"
        else:
            consensus = "mixed"

        # Tier counts
        mega = sum(1 for p in unique_influencers.values() if p.tier == "mega")
        major = sum(1 for p in unique_influencers.values() if p.tier == "major")

        # Signal score (0-2 for aggregator)
        signal_score = self._compute_signal_score(
            mention_count=len(mention_objects),
            unique_count=len(unique_influencers),
            cred_weighted_sentiment=cred_weighted,
            consensus=consensus,
            mega_count=mega,
            major_count=major,
        )

        # Top mentions by credibility-weighted engagement
        sorted_mentions = sorted(
            mention_objects,
            key=lambda m: m.engagement * m.influencer.credibility_score,
            reverse=True,
        )

        return InfluencerSignalReport(
            token_ticker=token_ticker,
            total_influencer_mentions=len(mention_objects),
            unique_influencers=len(unique_influencers),
            mega_influencer_mentions=mega,
            major_influencer_mentions=major,
            influencer_avg_sentiment=round(avg_sentiment, 2),
            credibility_weighted_sentiment=round(cred_weighted, 2),
            top_mentions=sorted_mentions[:10],
            bullish_influencers=bullish,
            bearish_influencers=bearish,
            neutral_influencers=neutral,
            consensus=consensus,
            influencer_signal_score=round(signal_score, 2),
            total_reach=total_reach,
            generated_at=datetime.now(timezone.utc),
        )

    def record_outcome(
        self,
        author_id: str,
        was_successful: bool,
    ) -> None:
        """Record the outcome of an influencer's call for accuracy tracking."""
        if author_id in self._registry:
            profile = self._registry[author_id]
            profile.total_calls += 1
            if was_successful:
                profile.successful_calls += 1
            profile.accuracy = round(
                profile.successful_calls / profile.total_calls, 3
            )
            # Recalculate credibility
            if profile.total_calls >= 5:
                modifier = 0.5 + profile.accuracy
            else:
                modifier = 1.0
            profile.credibility_score = round(profile.weight * modifier, 2)
            profile.last_updated = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Internal – Signal Score
    # ------------------------------------------------------------------

    def _compute_signal_score(
        self,
        mention_count: int,
        unique_count: int,
        cred_weighted_sentiment: float,
        consensus: str,
        mega_count: int,
        major_count: int,
    ) -> float:
        """
        Compute Influencer Signal Score (0-2 range for aggregator).

        Components:
        - Mention volume from influencers (0-0.5)
        - Sentiment quality from credible sources (0-0.8)
        - Mega/major influencer bonus (0-0.4)
        - Consensus bonus/penalty (0-0.3)
        """
        # Volume component (log-scaled, max 0.5)
        volume_score = min(0.5, math.log10(max(mention_count, 1)) * 0.25)

        # Sentiment component (mapped from -10/+10 to 0-0.8)
        # Positive sentiment → higher score
        sentiment_normalized = (cred_weighted_sentiment + 10) / 20  # 0-1
        sentiment_score = sentiment_normalized * 0.8

        # Mega/major bonus (each mega = 0.15, each major = 0.08, max 0.4)
        tier_bonus = min(0.4, mega_count * 0.15 + major_count * 0.08)

        # Consensus bonus
        consensus_map = {
            "strong_bullish": 0.3,
            "bullish": 0.15,
            "mixed": 0.0,
            "bearish": -0.15,
            "strong_bearish": -0.3,
        }
        consensus_bonus = consensus_map.get(consensus, 0.0)

        total = volume_score + sentiment_score + tier_bonus + consensus_bonus
        return max(0.0, min(2.0, total))
