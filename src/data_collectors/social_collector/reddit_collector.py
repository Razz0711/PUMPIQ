"""
Reddit Sentiment Collector
============================
Analyzes Reddit posts and comments for crypto sentiment.

Scoring factors:
- Upvote/downvote ratio as sentiment proxy
- Comment sentiment analysis
- Post title sentiment
- Subreddit relevance weighting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .sentiment_analyzer import CryptoSentimentAnalyzer, SentimentResult

logger = logging.getLogger(__name__)


CRYPTO_SUBREDDITS = {
    "cryptocurrency": 1.5,
    "cryptomoonshots": 1.0,
    "wallstreetbetscrypto": 1.2,
    "defi": 1.3,
    "solana": 1.4,
    "ethereum": 1.4,
    "altcoins": 1.1,
    "satoshistreetbets": 1.0,
    "memecoins": 0.8,  # Lower weight – less credible
    "cryptomarkets": 1.3,
}


@dataclass
class RedditPost:
    """A Reddit post with metadata."""
    post_id: str
    title: str
    body: str
    author: str
    subreddit: str
    upvotes: int = 0
    downvotes: int = 0
    upvote_ratio: float = 0.5
    num_comments: int = 0
    score: int = 0
    created_at: Optional[datetime] = None
    is_self_post: bool = True
    url: str = ""
    flair: str = ""


@dataclass
class RedditComment:
    """A Reddit comment."""
    comment_id: str
    body: str
    author: str
    post_id: str
    upvotes: int = 0
    score: int = 0
    is_op: bool = False  # Is the original poster
    created_at: Optional[datetime] = None


@dataclass
class RedditPostSentiment:
    """Post with sentiment analysis."""
    post: RedditPost
    title_sentiment: SentimentResult
    body_sentiment: SentimentResult
    combined_score: float = 0.0
    upvote_sentiment_signal: float = 0.0  # Derived from upvote ratio
    subreddit_weight: float = 1.0
    top_comment_sentiments: List[SentimentResult] = field(default_factory=list)


@dataclass
class RedditTokenMetrics:
    """Aggregated Reddit metrics for a single token."""
    token_ticker: str
    total_posts: int = 0
    total_comments_analyzed: int = 0
    unique_authors: int = 0
    total_upvotes: int = 0
    avg_upvote_ratio: float = 0.0
    avg_post_sentiment: float = 0.0
    avg_comment_sentiment: float = 0.0
    weighted_sentiment: float = 0.0  # Weighted by upvotes + subreddit credibility
    subreddit_breakdown: Dict[str, int] = field(default_factory=dict)
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    top_posts: List[RedditPostSentiment] = field(default_factory=list)
    mention_velocity: float = 0.0
    trending_score: float = 0.0
    collected_at: Optional[datetime] = None


class RedditCollector:
    """
    Collects Reddit posts/comments mentioning crypto tokens and performs
    sentiment analysis weighted by upvotes and subreddit relevance.

    Reddit-specific scoring:
    - Upvote ratio > 0.8 = community agrees (amplify sentiment)
    - Upvote ratio < 0.4 = community disagrees (dampen/flip sentiment)
    - Crypto subreddits get higher weight
    - Posts with many comments get higher weight
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "PumpIQ/1.0",
        analyzer: Optional[CryptoSentimentAnalyzer] = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.analyzer = analyzer or CryptoSentimentAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def collect_token_metrics(
        self,
        token_ticker: str,
        hours: int = 24,
        max_posts: int = 100,
        comments_per_post: int = 10,
    ) -> RedditTokenMetrics:
        """
        Search Reddit for token mentions, analyze sentiment, and aggregate.
        """
        posts = await self._search_posts(token_ticker, hours, max_posts)
        scored_posts: List[RedditPostSentiment] = []

        for post in posts:
            comments = await self._fetch_comments(post.post_id, comments_per_post)
            scored = self._score_post(post, comments)
            scored_posts.append(scored)

        return self._aggregate(token_ticker, scored_posts)

    # ------------------------------------------------------------------
    # Internal – Reddit API
    # ------------------------------------------------------------------

    async def _search_posts(
        self, query: str, hours: int, max_posts: int
    ) -> List[RedditPost]:
        """
        Search Reddit via API.

        Production: Use PRAW (Python Reddit API Wrapper) or httpx.
        """
        try:
            import httpx

            # Reddit OAuth flow (production implementation)
            auth_url = "https://www.reddit.com/api/v1/access_token"
            search_url = "https://oauth.reddit.com/search"

            async with httpx.AsyncClient() as client:
                # Get access token
                auth_resp = await client.post(
                    auth_url,
                    data={"grant_type": "client_credentials"},
                    auth=(self.client_id, self.client_secret),
                    headers={"User-Agent": self.user_agent},
                    timeout=15,
                )
                token = auth_resp.json().get("access_token", "")

                # Search
                resp = await client.get(
                    search_url,
                    params={
                        "q": f"{query} crypto",
                        "sort": "relevance",
                        "t": "day" if hours <= 24 else "week",
                        "limit": min(max_posts, 100),
                        "type": "link",
                    },
                    headers={
                        "Authorization": f"Bearer {token}",
                        "User-Agent": self.user_agent,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                return self._parse_posts(resp.json())

        except ImportError:
            logger.warning("httpx not installed – returning empty post list")
            return []
        except Exception as e:
            logger.error(f"Reddit search error: {e}")
            return []

    def _parse_posts(self, data: Dict[str, Any]) -> List[RedditPost]:
        """Parse Reddit API response."""
        posts: List[RedditPost] = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            try:
                created = datetime.utcfromtimestamp(d.get("created_utc", 0))
            except Exception:
                created = None

            posts.append(
                RedditPost(
                    post_id=d.get("id", ""),
                    title=d.get("title", ""),
                    body=d.get("selftext", ""),
                    author=d.get("author", ""),
                    subreddit=d.get("subreddit", "").lower(),
                    upvotes=d.get("ups", 0),
                    downvotes=d.get("downs", 0),
                    upvote_ratio=d.get("upvote_ratio", 0.5),
                    num_comments=d.get("num_comments", 0),
                    score=d.get("score", 0),
                    created_at=created,
                    is_self_post=d.get("is_self", True),
                    url=d.get("url", ""),
                    flair=d.get("link_flair_text", ""),
                )
            )
        return posts

    async def _fetch_comments(
        self, post_id: str, limit: int
    ) -> List[RedditComment]:
        """Fetch top comments for a post (production: PRAW or API call)."""
        # Placeholder – production uses Reddit API
        return []

    # ------------------------------------------------------------------
    # Internal – Scoring
    # ------------------------------------------------------------------

    def _score_post(
        self, post: RedditPost, comments: List[RedditComment]
    ) -> RedditPostSentiment:
        """Analyze sentiment for a post and its comments."""
        title_sent = self.analyzer.analyze(post.title)
        body_sent = self.analyzer.analyze(post.body) if post.body else SentimentResult(
            raw_score=0, normalized_score=0,
            emotional_tone=title_sent.emotional_tone,
            confidence=0, word_count=0,
        )

        # Upvote ratio as sentiment signal
        # ratio > 0.8 = community strongly agrees → amplify
        # ratio 0.5-0.8 = mixed → neutral
        # ratio < 0.5 = community disagrees → dampen
        if post.upvote_ratio > 0.8:
            upvote_signal = 1.5
        elif post.upvote_ratio > 0.6:
            upvote_signal = 1.0
        elif post.upvote_ratio > 0.4:
            upvote_signal = 0.5
        else:
            upvote_signal = -0.5  # Community disagrees

        # Subreddit credibility weight
        sub_weight = CRYPTO_SUBREDDITS.get(post.subreddit, 0.8)

        # Combined score: title carries more weight for quick assessment
        combined = (
            (title_sent.raw_score * 0.6 + body_sent.raw_score * 0.4)
            * upvote_signal
            * sub_weight
        )

        # Analyze top comments
        comment_sentiments: List[SentimentResult] = []
        for comment in comments:
            cs = self.analyzer.analyze(comment.body)
            comment_sentiments.append(cs)

        return RedditPostSentiment(
            post=post,
            title_sentiment=title_sent,
            body_sentiment=body_sent,
            combined_score=round(combined, 2),
            upvote_sentiment_signal=upvote_signal,
            subreddit_weight=sub_weight,
            top_comment_sentiments=comment_sentiments,
        )

    # ------------------------------------------------------------------
    # Internal – Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, token_ticker: str, scored_posts: List[RedditPostSentiment]
    ) -> RedditTokenMetrics:
        """Compute aggregated Reddit metrics."""
        if not scored_posts:
            return RedditTokenMetrics(
                token_ticker=token_ticker, collected_at=datetime.utcnow()
            )

        total = len(scored_posts)
        all_authors = set()
        total_upvotes = 0
        combined_scores: List[float] = []
        comment_scores: List[float] = []
        sub_counts: Dict[str, int] = {}
        dist: Dict[str, int] = {}

        for sp in scored_posts:
            all_authors.add(sp.post.author)
            total_upvotes += sp.post.upvotes
            combined_scores.append(sp.combined_score)

            sub = sp.post.subreddit
            sub_counts[sub] = sub_counts.get(sub, 0) + 1

            tone = sp.title_sentiment.emotional_tone.value
            dist[tone] = dist.get(tone, 0) + 1

            for cs in sp.top_comment_sentiments:
                comment_scores.append(cs.raw_score)

        avg_upvote_ratio = sum(
            sp.post.upvote_ratio for sp in scored_posts
        ) / total

        avg_post_sentiment = sum(
            sp.title_sentiment.raw_score for sp in scored_posts
        ) / total

        avg_comment_sentiment = (
            sum(comment_scores) / len(comment_scores) if comment_scores else 0
        )

        weighted_sentiment = sum(combined_scores) / total

        # Trending score: combination of upvotes, comments, and post count
        trending = (
            total * 3               # More posts = more trending
            + total_upvotes * 0.1   # Upvotes matter
            + sum(sp.post.num_comments for sp in scored_posts) * 0.5
        )

        return RedditTokenMetrics(
            token_ticker=token_ticker,
            total_posts=total,
            total_comments_analyzed=len(comment_scores),
            unique_authors=len(all_authors),
            total_upvotes=total_upvotes,
            avg_upvote_ratio=round(avg_upvote_ratio, 3),
            avg_post_sentiment=round(avg_post_sentiment, 2),
            avg_comment_sentiment=round(avg_comment_sentiment, 2),
            weighted_sentiment=round(weighted_sentiment, 2),
            subreddit_breakdown=sub_counts,
            sentiment_distribution=dist,
            top_posts=sorted(scored_posts, key=lambda x: x.combined_score, reverse=True)[:10],
            trending_score=round(trending, 2),
            collected_at=datetime.utcnow(),
        )
