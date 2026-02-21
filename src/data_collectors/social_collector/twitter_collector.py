"""
Twitter / X Data Collector
===========================
Collects and scores cryptocurrency-related tweets.

Weighting by Author Influence:
- Influencer (10K+ followers): 10x weight
- Mid-tier (1K-10K followers): 3x weight
- Regular user: 1x weight
- New/suspicious account: 0.3x weight
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .sentiment_analyzer import CryptoSentimentAnalyzer, WeightedSentiment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Tweet:
    """Represents a single tweet."""
    tweet_id: str
    text: str
    author_id: str
    author_username: str
    follower_count: int = 0
    is_verified: bool = False
    account_created_at: Optional[datetime] = None
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    created_at: Optional[datetime] = None
    hashtags: List[str] = field(default_factory=list)
    mentioned_tickers: List[str] = field(default_factory=list)
    language: str = "en"


@dataclass
class TweetSentiment:
    """Tweet paired with its sentiment analysis."""
    tweet: Tweet
    weighted_sentiment: WeightedSentiment
    engagement_score: float = 0.0  # normalized engagement metric


@dataclass
class TwitterTokenMetrics:
    """Aggregated Twitter metrics for a single token."""
    token_ticker: str
    total_mentions: int = 0
    unique_authors: int = 0
    total_engagement: int = 0  # likes + retweets + replies + quotes
    avg_sentiment: float = 0.0
    weighted_avg_sentiment: float = 0.0
    influencer_mentions: int = 0  # authors with 10K+ followers
    influencer_total_followers: int = 0
    top_tweets: List[TweetSentiment] = field(default_factory=list)
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    hourly_mention_counts: Dict[str, int] = field(default_factory=dict)
    mention_velocity: float = 0.0  # % change vs. previous period
    collected_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class TwitterCollector:
    """
    Collects tweets for crypto tokens, performs sentiment analysis, and
    computes aggregated metrics.

    Usage::

        collector = TwitterCollector(bearer_token="…")
        metrics = await collector.collect_token_metrics("BONK", hours=24)
    """

    SEARCH_ENDPOINT = "https://api.twitter.com/2/tweets/search/recent"
    USER_ENDPOINT = "https://api.twitter.com/2/users"

    def __init__(
        self,
        bearer_token: str,
        analyzer: Optional[CryptoSentimentAnalyzer] = None,
        max_results_per_query: int = 100,
        rate_limit_sleep: float = 2.0,
    ):
        self.bearer_token = bearer_token
        self.analyzer = analyzer or CryptoSentimentAnalyzer()
        self.max_results_per_query = max_results_per_query
        self.rate_limit_sleep = rate_limit_sleep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def collect_token_metrics(
        self,
        token_ticker: str,
        hours: int = 24,
        max_tweets: int = 500,
    ) -> TwitterTokenMetrics:
        """
        Collect tweets mentioning *token_ticker* from the last *hours* and
        compute aggregated sentiment metrics.

        Args:
            token_ticker: Token symbol (e.g. "BONK", "WIF")
            hours: Lookback window
            max_tweets: Maximum tweets to analyse

        Returns:
            TwitterTokenMetrics with all computed scores
        """
        # Build search queries (cashtag + plain mention)
        queries = [
            f"${token_ticker} crypto -is:retweet lang:en",
            f"{token_ticker} token -is:retweet lang:en",
        ]

        all_tweets: List[Tweet] = []
        for query in queries:
            tweets = await self._search_tweets(query, hours, max_tweets // len(queries))
            all_tweets.extend(tweets)

        # De-duplicate by tweet_id
        seen_ids = set()
        unique_tweets: List[Tweet] = []
        for t in all_tweets:
            if t.tweet_id not in seen_ids:
                seen_ids.add(t.tweet_id)
                unique_tweets.append(t)

        # Analyse sentiment for each tweet
        scored_tweets = self._score_tweets(unique_tweets)

        # Aggregate
        metrics = self._aggregate_metrics(token_ticker, scored_tweets, hours)
        return metrics

    async def collect_trending_tokens(
        self,
        token_tickers: List[str],
        hours: int = 24,
    ) -> List[TwitterTokenMetrics]:
        """Collect metrics for multiple tokens."""
        results = []
        for ticker in token_tickers:
            try:
                m = await self.collect_token_metrics(ticker, hours)
                results.append(m)
            except Exception as e:
                logger.error(f"Twitter collection failed for {ticker}: {e}")
        return results

    # ------------------------------------------------------------------
    # Internal – Twitter API
    # ------------------------------------------------------------------

    async def _search_tweets(
        self, query: str, hours: int, max_tweets: int
    ) -> List[Tweet]:
        """
        Search recent tweets via Twitter API v2.

        NOTE: In production, replace with real httpx/aiohttp calls.
        This is the interface specification – returns empty when no
        bearer token configured.
        """
        try:
            import httpx

            start_time = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            params = {
                "query": query,
                "start_time": start_time,
                "max_results": min(max_tweets, self.max_results_per_query),
                "tweet.fields": "created_at,public_metrics,entities,lang",
                "user.fields": "public_metrics,verified,created_at",
                "expansions": "author_id",
            }
            headers = {"Authorization": f"Bearer {self.bearer_token}"}

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    self.SEARCH_ENDPOINT, params=params, headers=headers, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()

            tweets = self._parse_twitter_response(data)
            await asyncio.sleep(self.rate_limit_sleep)  # respect rate limits
            return tweets[:max_tweets]

        except ImportError:
            logger.warning("httpx not installed – returning empty tweet list")
            return []
        except Exception as e:
            logger.error(f"Twitter search error: {e}")
            return []

    def _parse_twitter_response(self, data: Dict[str, Any]) -> List[Tweet]:
        """Parse Twitter API v2 response into Tweet objects."""
        tweets: List[Tweet] = []

        # Build user lookup
        users: Dict[str, Dict] = {}
        for u in data.get("includes", {}).get("users", []):
            users[u["id"]] = u

        for item in data.get("data", []):
            author = users.get(item.get("author_id", ""), {})
            pub_metrics = item.get("public_metrics", {})
            author_metrics = author.get("public_metrics", {})

            # Extract hashtags
            hashtags = [
                h["tag"]
                for h in item.get("entities", {}).get("hashtags", [])
            ]
            # Extract cashtags
            tickers = [
                c["tag"]
                for c in item.get("entities", {}).get("cashtags", [])
            ]

            created_str = item.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except Exception:
                created_at = None

            account_created_str = author.get("created_at", "")
            try:
                account_created = datetime.fromisoformat(
                    account_created_str.replace("Z", "+00:00")
                )
            except Exception:
                account_created = None

            tweets.append(
                Tweet(
                    tweet_id=item.get("id", ""),
                    text=item.get("text", ""),
                    author_id=item.get("author_id", ""),
                    author_username=author.get("username", ""),
                    follower_count=author_metrics.get("followers_count", 0),
                    is_verified=author.get("verified", False),
                    account_created_at=account_created,
                    like_count=pub_metrics.get("like_count", 0),
                    retweet_count=pub_metrics.get("retweet_count", 0),
                    reply_count=pub_metrics.get("reply_count", 0),
                    quote_count=pub_metrics.get("quote_count", 0),
                    created_at=created_at,
                    hashtags=hashtags,
                    mentioned_tickers=tickers,
                    language=item.get("lang", "en"),
                )
            )
        return tweets

    # ------------------------------------------------------------------
    # Internal – Scoring
    # ------------------------------------------------------------------

    def _score_tweets(self, tweets: List[Tweet]) -> List[TweetSentiment]:
        """Apply sentiment analysis and author weighting to each tweet."""
        results: List[TweetSentiment] = []
        for tweet in tweets:
            account_age = 365  # default
            if tweet.account_created_at:
                account_age = max(1, (datetime.now(timezone.utc) - tweet.account_created_at.replace(tzinfo=None)).days)

            weighted = self.analyzer.compute_weighted_sentiment(
                text=tweet.text,
                follower_count=tweet.follower_count,
                account_age_days=account_age,
                is_verified=tweet.is_verified,
                platform="twitter",
            )

            engagement = (
                tweet.like_count
                + tweet.retweet_count * 2
                + tweet.reply_count * 1.5
                + tweet.quote_count * 3
            )

            results.append(
                TweetSentiment(
                    tweet=tweet,
                    weighted_sentiment=weighted,
                    engagement_score=engagement,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internal – Aggregation
    # ------------------------------------------------------------------

    def _aggregate_metrics(
        self,
        token_ticker: str,
        scored_tweets: List[TweetSentiment],
        hours: int,
    ) -> TwitterTokenMetrics:
        """Compute aggregated metrics from scored tweets."""
        if not scored_tweets:
            return TwitterTokenMetrics(
                token_ticker=token_ticker, collected_at=datetime.now(timezone.utc)
            )

        total_mentions = len(scored_tweets)
        unique_authors = len({st.tweet.author_id for st in scored_tweets})

        raw_scores = [st.weighted_sentiment.sentiment.raw_score for st in scored_tweets]
        weighted_scores = [st.weighted_sentiment.weighted_score for st in scored_tweets]
        total_weights = sum(st.weighted_sentiment.author_weight for st in scored_tweets)

        avg_sentiment = sum(raw_scores) / total_mentions
        weighted_avg = (
            sum(weighted_scores) / total_weights if total_weights > 0 else 0
        )

        total_engagement = sum(
            st.tweet.like_count + st.tweet.retweet_count
            + st.tweet.reply_count + st.tweet.quote_count
            for st in scored_tweets
        )

        # Influencer metrics
        influencer_tweets = [
            st for st in scored_tweets if st.tweet.follower_count >= 10_000
        ]
        influencer_mentions = len(influencer_tweets)
        influencer_total_followers = sum(
            st.tweet.follower_count for st in influencer_tweets
        )

        # Top tweets by engagement
        top_tweets = sorted(scored_tweets, key=lambda x: x.engagement_score, reverse=True)[:10]

        # Sentiment distribution
        dist: Dict[str, int] = {}
        for st in scored_tweets:
            tone = st.weighted_sentiment.sentiment.emotional_tone.value
            dist[tone] = dist.get(tone, 0) + 1

        # Hourly mentions (for velocity calculation)
        hourly: Dict[str, int] = {}
        for st in scored_tweets:
            if st.tweet.created_at:
                hour_key = st.tweet.created_at.strftime("%Y-%m-%d %H:00")
                hourly[hour_key] = hourly.get(hour_key, 0) + 1

        # Mention velocity (first half vs second half of period)
        if hourly:
            sorted_hours = sorted(hourly.keys())
            mid = len(sorted_hours) // 2
            first_half = sum(hourly[h] for h in sorted_hours[:mid]) if mid > 0 else 0
            second_half = sum(hourly[h] for h in sorted_hours[mid:])
            if first_half > 0:
                mention_velocity = ((second_half - first_half) / first_half) * 100
            else:
                mention_velocity = 100.0 if second_half > 0 else 0.0
        else:
            mention_velocity = 0.0

        return TwitterTokenMetrics(
            token_ticker=token_ticker,
            total_mentions=total_mentions,
            unique_authors=unique_authors,
            total_engagement=total_engagement,
            avg_sentiment=round(avg_sentiment, 2),
            weighted_avg_sentiment=round(weighted_avg, 2),
            influencer_mentions=influencer_mentions,
            influencer_total_followers=influencer_total_followers,
            top_tweets=top_tweets,
            sentiment_distribution=dist,
            hourly_mention_counts=hourly,
            mention_velocity=round(mention_velocity, 2),
            collected_at=datetime.now(timezone.utc),
        )
