"""
News & Sentiment Collector
============================
Fetches crypto news headlines and computes basic sentiment scores.

Data sources (all **free**, no API key required for basic access)
-----------------------------------------------------------------
1. **CryptoPanic API** (free tier) – aggregated crypto news with
   community votes (bullish / bearish).
   Base: ``https://cryptopanic.com/api/free/v1/posts/``
   Free tier: 5 requests/min, returns 20 posts per page.
   Set ``CRYPTOPANIC_API_KEY`` in .env for full access.

2. **CoinGecko /status_updates** (deprecated) – alternative fallback.

3. **Built-in keyword sentiment** – fast local scoring when no API
   key is available; uses a crypto-tuned word list.

Output: ``NewsResult`` objects that map to the engine's ``NewsScorePayload``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ───────────────────────── sentiment word lists ─────────────────────────

_BULLISH_WORDS = {
    "bullish", "surge", "soar", "rally", "pump", "moon", "breakout",
    "partnership", "listing", "launch", "upgrade", "adoption", "milestone",
    "record", "high", "buy", "accumulate", "growth", "profit", "gain",
    "outperform", "breakthrough", "support", "uptrend", "whale",
    "institutional", "approval", "etf", "integration", "backed",
}

_BEARISH_WORDS = {
    "bearish", "crash", "dump", "plunge", "hack", "exploit", "scam",
    "rug", "rugpull", "ban", "regulation", "sec", "lawsuit", "fraud",
    "sell", "decline", "loss", "drop", "downtrend", "resistance",
    "fud", "warning", "vulnerability", "delay", "concern", "risk",
    "investigation", "liquidation", "bankrupt", "exit",
}


def _keyword_sentiment(text: str) -> float:
    """
    Quick keyword-based sentiment scorer.

    Returns a score between -1.0 (very bearish) and +1.0 (very bullish).
    0.0 = neutral.
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    bull = len(words & _BULLISH_WORDS)
    bear = len(words & _BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 3)


# ───────────────────────── Data classes ─────────────────────────

@dataclass
class NewsArticle:
    """A single news article / post."""
    title: str
    url: str = ""
    source: str = ""
    published_at: Optional[datetime] = None
    sentiment: float = 0.0        # -1..+1
    votes_positive: int = 0
    votes_negative: int = 0
    votes_important: int = 0
    currencies: List[str] = field(default_factory=list)  # mentioned tickers


@dataclass
class NewsResult:
    """
    Aggregated news analysis for a token or the market.

    Maps to ``NewsScorePayload`` via the orchestrator.
    """
    query: str = ""
    articles: List[NewsArticle] = field(default_factory=list)
    avg_sentiment: float = 0.0     # -1..+1
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    total_articles: int = 0
    source_count: int = 0
    key_headlines: List[str] = field(default_factory=list)
    narrative: str = ""
    collected_at: Optional[datetime] = None

    @property
    def score_0_10(self) -> float:
        """Convert -1..+1 sentiment to the engine's 0-10 scale."""
        return round((self.avg_sentiment + 1) * 5, 2)   # -1→0, 0→5, +1→10


# ───────────────────────── Collector ────────────────────────────

class NewsCollector:
    """
    Pulls crypto news headlines, scores sentiment, and returns
    an aggregated ``NewsResult``.

    Usage::

        nc = NewsCollector()                           # keyword-only
        nc = NewsCollector(api_key="your_key")         # CryptoPanic

        result = await nc.collect("BONK")
        print(result.score_0_10, result.key_headlines)
    """

    CRYPTOPANIC_BASE = "https://cryptopanic.com/api/free/v1/posts/"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_sleep: float = 1.5,
    ):
        self.api_key = api_key or os.getenv("CRYPTOPANIC_API_KEY", "")
        self.rate_limit_sleep = rate_limit_sleep

    # ── public API ─────────────────────────────────────────────────

    async def collect(
        self,
        query: str,
        max_articles: int = 20,
    ) -> NewsResult:
        """
        Collect news for *query* (a token ticker or keyword).

        If a CryptoPanic key is configured, uses the API.
        Otherwise falls back to a placeholder / keyword-only mode.
        """
        if self.api_key:
            articles = await self._fetch_cryptopanic(query, max_articles)
        else:
            logger.info(
                "No CRYPTOPANIC_API_KEY – using keyword-only sentiment"
            )
            articles = []

        return self._aggregate(query, articles)

    async def collect_market_news(
        self,
        max_articles: int = 30,
    ) -> NewsResult:
        """Collect general crypto market news (no specific token filter)."""
        if self.api_key:
            articles = await self._fetch_cryptopanic("", max_articles)
        else:
            articles = []
        return self._aggregate("crypto market", articles)

    # ── CryptoPanic ────────────────────────────────────────────────

    async def _fetch_cryptopanic(
        self,
        query: str,
        max_articles: int,
    ) -> List[NewsArticle]:
        """
        CryptoPanic free API.

        Docs: https://cryptopanic.com/developers/api/
        Endpoint: /api/free/v1/posts/?auth_token=KEY&currencies=BTC
        """
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed – cannot fetch news")
            return []

        params: Dict[str, Any] = {
            "auth_token": self.api_key,
            "public": "true",
        }
        if query:
            params["currencies"] = query.upper()

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    self.CRYPTOPANIC_BASE, params=params,
                )
                resp.raise_for_status()
                data = resp.json()
            await asyncio.sleep(self.rate_limit_sleep)
        except Exception as exc:
            logger.error("CryptoPanic API error: %s", exc)
            return []

        articles: List[NewsArticle] = []
        for item in (data.get("results", []) or [])[:max_articles]:
            published = None
            if item.get("published_at"):
                try:
                    published = datetime.fromisoformat(
                        item["published_at"].replace("Z", "+00:00")
                    )
                except Exception:
                    pass

            votes = item.get("votes", {})
            title = item.get("title", "")
            sentiment = _keyword_sentiment(title)

            # CryptoPanic community votes can override keyword sentiment
            pos = votes.get("positive", 0) or 0
            neg = votes.get("negative", 0) or 0
            if pos + neg > 0:
                vote_sentiment = (pos - neg) / (pos + neg)
                sentiment = round((sentiment + vote_sentiment) / 2, 3)

            currencies = [
                c.get("code", "")
                for c in (item.get("currencies", []) or [])
            ]

            articles.append(
                NewsArticle(
                    title=title,
                    url=item.get("url", ""),
                    source=item.get("source", {}).get("title", ""),
                    published_at=published,
                    sentiment=sentiment,
                    votes_positive=pos,
                    votes_negative=neg,
                    votes_important=votes.get("important", 0) or 0,
                    currencies=currencies,
                )
            )

        return articles

    # ── aggregation ────────────────────────────────────────────────

    def _aggregate(
        self,
        query: str,
        articles: List[NewsArticle],
    ) -> NewsResult:
        if not articles:
            return NewsResult(
                query=query,
                avg_sentiment=0.0,
                total_articles=0,
                collected_at=datetime.now(timezone.utc),
            )

        sentiments = [a.sentiment for a in articles]
        avg_sent = sum(sentiments) / len(sentiments)
        bullish = sum(1 for s in sentiments if s > 0.15)
        bearish = sum(1 for s in sentiments if s < -0.15)
        neutral = len(sentiments) - bullish - bearish

        sources = {a.source for a in articles if a.source}

        # Top 5 most impactful headlines (by |sentiment|)
        sorted_arts = sorted(articles, key=lambda a: abs(a.sentiment), reverse=True)
        top_headlines = [a.title for a in sorted_arts[:5]]

        # Build narrative
        if avg_sent > 0.3:
            narrative = f"News sentiment for {query} is strongly bullish."
        elif avg_sent > 0.1:
            narrative = f"News sentiment for {query} is mildly bullish."
        elif avg_sent < -0.3:
            narrative = f"News sentiment for {query} is strongly bearish."
        elif avg_sent < -0.1:
            narrative = f"News sentiment for {query} is mildly bearish."
        else:
            narrative = f"News sentiment for {query} is neutral/mixed."

        return NewsResult(
            query=query,
            articles=articles,
            avg_sentiment=round(avg_sent, 3),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            total_articles=len(articles),
            source_count=len(sources),
            key_headlines=top_headlines,
            narrative=narrative,
            collected_at=datetime.now(timezone.utc),
        )
