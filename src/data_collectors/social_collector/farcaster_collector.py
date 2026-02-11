"""
Farcaster Protocol Data Collector
===================================
Analyzes Farcaster casts (posts) for crypto sentiment.
Weights by on-chain reputation – higher-reputation accounts carry more weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .sentiment_analyzer import CryptoSentimentAnalyzer, WeightedSentiment

logger = logging.getLogger(__name__)


@dataclass
class FarcasterCast:
    """Represents a single Farcaster cast (post)."""
    cast_hash: str
    text: str
    author_fid: int
    author_username: str
    follower_count: int = 0
    following_count: int = 0
    onchain_reputation: float = 0.0  # 0-1 scale
    like_count: int = 0
    recast_count: int = 0
    reply_count: int = 0
    created_at: Optional[datetime] = None
    channel: str = ""  # e.g. /crypto, /defi


@dataclass
class FarcasterCastSentiment:
    """Cast paired with sentiment analysis."""
    cast: FarcasterCast
    weighted_sentiment: WeightedSentiment
    engagement_score: float = 0.0


@dataclass
class FarcasterTokenMetrics:
    """Aggregated Farcaster metrics for a single token."""
    token_ticker: str
    total_casts: int = 0
    unique_authors: int = 0
    total_engagement: int = 0
    avg_sentiment: float = 0.0
    weighted_avg_sentiment: float = 0.0
    high_reputation_mentions: int = 0       # Authors with reputation > 0.7
    high_reputation_total_followers: int = 0
    top_casts: List[FarcasterCastSentiment] = field(default_factory=list)
    channel_breakdown: Dict[str, int] = field(default_factory=dict)
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    mention_velocity: float = 0.0
    collected_at: Optional[datetime] = None


class FarcasterCollector:
    """
    Collects Farcaster casts mentioning crypto tokens and performs
    reputation-weighted sentiment analysis.

    On-chain reputation weighting:
    - Reputation > 0.8: 2x additional multiplier
    - Reputation 0.5-0.8: 1.5x multiplier
    - Reputation < 0.5: 1x (no bonus)
    - Reputation 0 or new: 0.5x (discount)
    """

    NEYNAR_SEARCH_URL = "https://api.neynar.com/v2/farcaster/cast/search"

    def __init__(
        self,
        api_key: str,
        analyzer: Optional[CryptoSentimentAnalyzer] = None,
    ):
        self.api_key = api_key
        self.analyzer = analyzer or CryptoSentimentAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def collect_token_metrics(
        self,
        token_ticker: str,
        hours: int = 24,
        max_casts: int = 300,
    ) -> FarcasterTokenMetrics:
        """
        Collect casts mentioning *token_ticker* and compute aggregated
        sentiment metrics weighted by on-chain reputation.
        """
        casts = await self._search_casts(token_ticker, hours, max_casts)
        scored = self._score_casts(casts)
        return self._aggregate(token_ticker, scored)

    # ------------------------------------------------------------------
    # Internal – API
    # ------------------------------------------------------------------

    async def _search_casts(
        self, query: str, hours: int, max_casts: int
    ) -> List[FarcasterCast]:
        """Search casts via Neynar API (interface spec – production implementation)."""
        try:
            import httpx

            params = {
                "q": query,
                "limit": min(max_casts, 100),
            }
            headers = {"api_key": self.api_key, "accept": "application/json"}

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    self.NEYNAR_SEARCH_URL, params=params, headers=headers, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()

            return self._parse_response(data)

        except ImportError:
            logger.warning("httpx not installed – returning empty cast list")
            return []
        except Exception as e:
            logger.error(f"Farcaster search error: {e}")
            return []

    def _parse_response(self, data: Dict[str, Any]) -> List[FarcasterCast]:
        """Parse Neynar API response into FarcasterCast objects."""
        casts: List[FarcasterCast] = []
        for item in data.get("result", {}).get("casts", []):
            author = item.get("author", {})

            created_str = item.get("timestamp", "")
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except Exception:
                created_at = None

            # Compute on-chain reputation from available signals
            reputation = self._compute_onchain_reputation(author)

            casts.append(
                FarcasterCast(
                    cast_hash=item.get("hash", ""),
                    text=item.get("text", ""),
                    author_fid=author.get("fid", 0),
                    author_username=author.get("username", ""),
                    follower_count=author.get("follower_count", 0),
                    following_count=author.get("following_count", 0),
                    onchain_reputation=reputation,
                    like_count=item.get("reactions", {}).get("likes_count", 0),
                    recast_count=item.get("reactions", {}).get("recasts_count", 0),
                    reply_count=item.get("replies", {}).get("count", 0),
                    created_at=created_at,
                    channel=item.get("root_parent_url", ""),
                )
            )
        return casts

    def _compute_onchain_reputation(self, author: Dict[str, Any]) -> float:
        """
        Compute on-chain reputation from Farcaster profile data.

        Factors:
        - Follower count (normalized logarithmically)
        - Account age / FID (lower FID = earlier adopter)
        - Active engagement ratio (following/follower ratio)
        - Verified Ethereum address linked
        """
        followers = author.get("follower_count", 0)
        following = author.get("following_count", 0)
        fid = author.get("fid", 999_999)
        has_eth_address = bool(author.get("verified_addresses", {}).get("eth_addresses"))

        # Follower component (0-0.4)
        import math
        follower_score = min(0.4, math.log10(max(followers, 1)) / 12.5)

        # FID component – early adopters (0-0.2)
        fid_score = max(0, 0.2 - (fid / 500_000) * 0.2)

        # Engagement ratio (0-0.2)
        if followers > 0:
            ratio = min(following / followers, 2.0)
            # Healthy ratio around 0.1-0.5 is best
            engagement_score = 0.2 if 0.05 < ratio < 1.0 else 0.1
        else:
            engagement_score = 0.0

        # Verified address (0-0.2)
        address_score = 0.2 if has_eth_address else 0.0

        reputation = follower_score + fid_score + engagement_score + address_score
        return round(min(1.0, reputation), 3)

    # ------------------------------------------------------------------
    # Internal – Scoring
    # ------------------------------------------------------------------

    def _score_casts(self, casts: List[FarcasterCast]) -> List[FarcasterCastSentiment]:
        """Score each cast with reputation-weighted sentiment."""
        results: List[FarcasterCastSentiment] = []
        for cast in casts:
            weighted = self.analyzer.compute_weighted_sentiment(
                text=cast.text,
                follower_count=cast.follower_count,
                account_age_days=365,  # Approximate
                is_verified=cast.onchain_reputation > 0.6,
                onchain_reputation=cast.onchain_reputation,
                platform="farcaster",
            )
            engagement = (
                cast.like_count + cast.recast_count * 2 + cast.reply_count * 1.5
            )
            results.append(
                FarcasterCastSentiment(
                    cast=cast,
                    weighted_sentiment=weighted,
                    engagement_score=engagement,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internal – Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, token_ticker: str, scored: List[FarcasterCastSentiment]
    ) -> FarcasterTokenMetrics:
        if not scored:
            return FarcasterTokenMetrics(
                token_ticker=token_ticker, collected_at=datetime.utcnow()
            )

        raw_scores = [s.weighted_sentiment.sentiment.raw_score for s in scored]
        weighted_scores = [s.weighted_sentiment.weighted_score for s in scored]
        total_weights = sum(s.weighted_sentiment.author_weight for s in scored)

        high_rep = [s for s in scored if s.cast.onchain_reputation > 0.7]

        channel_counts: Dict[str, int] = {}
        for s in scored:
            ch = s.cast.channel or "no_channel"
            channel_counts[ch] = channel_counts.get(ch, 0) + 1

        dist: Dict[str, int] = {}
        for s in scored:
            tone = s.weighted_sentiment.sentiment.emotional_tone.value
            dist[tone] = dist.get(tone, 0) + 1

        return FarcasterTokenMetrics(
            token_ticker=token_ticker,
            total_casts=len(scored),
            unique_authors=len({s.cast.author_fid for s in scored}),
            total_engagement=int(sum(s.engagement_score for s in scored)),
            avg_sentiment=round(sum(raw_scores) / len(raw_scores), 2),
            weighted_avg_sentiment=round(
                sum(weighted_scores) / total_weights if total_weights else 0, 2
            ),
            high_reputation_mentions=len(high_rep),
            high_reputation_total_followers=sum(
                s.cast.follower_count for s in high_rep
            ),
            top_casts=sorted(scored, key=lambda x: x.engagement_score, reverse=True)[:10],
            channel_breakdown=channel_counts,
            sentiment_distribution=dist,
            collected_at=datetime.utcnow(),
        )
