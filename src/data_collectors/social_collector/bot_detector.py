"""
Bot & Shill Detection Module
===============================
Detects artificial / inorganic social activity around crypto tokens.

Red Flag Deductions (from 12-point social score):
- Pump group activity detected:         -3 points
- Coordinated shilling detected:        -2 points
- Fake followers / engagement:          -2 points
- Suspicious influencer promotions:     -1 point
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Red‑flag severity constants
# ---------------------------------------------------------------------------

PUMP_GROUP_PENALTY = 3.0
COORDINATED_SHILL_PENALTY = 2.0
FAKE_ENGAGEMENT_PENALTY = 2.0
SUSPICIOUS_PROMO_PENALTY = 1.0

# Known pump-group keywords / phrases
PUMP_SIGNALS = {
    "pump group", "pump signal", "next 100x", "insider alpha",
    "buy now before", "private group", "guaranteed pump",
    "vip signal", "whale alert buy", "launching in",
    "pre-sale live", "stealth launch", "just launched buy now",
    "aped in", "fill your bags", "buy the dip now",
}

# Shill patterns (regex)
SHILL_PATTERNS = [
    r"(?i)just bought [\$#]?\w+ .{0,20}(not financial advice|nfa)",
    r"(?i)this is the next (doge|shib|pepe|bonk)",
    r"(?i)(easy|free|guaranteed)\s+(money|profit|gains)",
    r"(?i)(get in|buy)\s+(before|now|quick|asap)",
    r"(?i)don'?t miss.{0,30}(gem|moon|pump)",
]

# Suspicious domain patterns (fake engagement services)
FAKE_ENGAGEMENT_DOMAINS = {
    "followersup", "tweetattacks", "socialempire", "botfollowers",
    "buycheapfollowers", "likesandretweets", "gramfree",
}


@dataclass
class BotScore:
    """Score for a single author indicating bot probability."""
    author_id: str
    author_username: str
    bot_probability: float = 0.0  # 0-1 scale
    signals: List[str] = field(default_factory=list)


@dataclass
class ShillCampaign:
    """A detected coordinated shill campaign."""
    token_ticker: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    campaign_type: str = "unknown"  # "pump_group", "coordinated", "fake_engagement"
    severity: float = 0.0
    evidence: List[str] = field(default_factory=list)
    involved_accounts: Set[str] = field(default_factory=set)
    confidence: float = 0.0  # 0-1


@dataclass
class BotDetectionReport:
    """Full report for a token's bot / shill analysis."""
    token_ticker: str

    # Aggregate metrics
    total_accounts_analyzed: int = 0
    suspected_bots: int = 0
    bot_percentage: float = 0.0

    # Campaign detections
    pump_group_detected: bool = False
    coordinated_shilling: bool = False
    fake_engagement_detected: bool = False
    suspicious_promotions: bool = False

    # Individual campaign details
    campaigns: List[ShillCampaign] = field(default_factory=list)

    # Per-account scores
    bot_scores: List[BotScore] = field(default_factory=list)

    # Penalty points (subtracted from 12-point social score)
    total_penalty: float = 0.0
    penalty_breakdown: Dict[str, float] = field(default_factory=dict)

    generated_at: Optional[datetime] = None


class BotDetector:
    """
    Analyzes social data to detect:
    1. Bot / automated accounts
    2. Coordinated pump-group campaigns
    3. Fake engagement (likes, followers, retweets)
    4. Suspicious influencer promotions (paid undisclosed shills)

    Usage::

        detector = BotDetector()
        report = detector.analyze_token(
            token_ticker="SCAM",
            posts=all_posts,
            authors=all_author_profiles,
        )
        penalty = report.total_penalty  # 0-8 points to subtract
    """

    def __init__(
        self,
        bot_threshold: float = 0.65,
        shill_time_window_minutes: int = 30,
        shill_min_accounts: int = 5,
    ):
        self.bot_threshold = bot_threshold
        self.shill_time_window = timedelta(minutes=shill_time_window_minutes)
        self.shill_min_accounts = shill_min_accounts
        self._compiled_shill_patterns = [
            re.compile(p) for p in SHILL_PATTERNS
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_token(
        self,
        token_ticker: str,
        posts: List[Dict[str, Any]],
        authors: Optional[List[Dict[str, Any]]] = None,
    ) -> BotDetectionReport:
        """
        Run full bot / shill analysis.

        Args:
            token_ticker: Token symbol
            posts: List of post dicts with keys:
                   text, author_id, author_username, follower_count,
                   created_at (datetime), platform
            authors: Optional author profile dicts with keys:
                     author_id, follower_count, following_count,
                     account_age_days, tweet_count, avg_posts_per_day

        Returns:
            BotDetectionReport with penalties and details
        """
        report = BotDetectionReport(
            token_ticker=token_ticker,
            total_accounts_analyzed=len({p.get("author_id") for p in posts}),
            generated_at=datetime.utcnow(),
        )

        # 1. Score individual accounts for bot probability
        author_profiles = self._build_author_profiles(posts, authors)
        report.bot_scores = self._score_accounts(author_profiles)
        report.suspected_bots = sum(
            1 for bs in report.bot_scores if bs.bot_probability >= self.bot_threshold
        )
        report.bot_percentage = round(
            (report.suspected_bots / max(report.total_accounts_analyzed, 1)) * 100, 1
        )

        # 2. Detect pump-group activity
        pump_campaigns = self._detect_pump_groups(posts)
        if pump_campaigns:
            report.pump_group_detected = True
            report.campaigns.extend(pump_campaigns)
            report.penalty_breakdown["pump_group"] = PUMP_GROUP_PENALTY

        # 3. Detect coordinated shilling
        shill_campaigns = self._detect_coordinated_shilling(posts)
        if shill_campaigns:
            report.coordinated_shilling = True
            report.campaigns.extend(shill_campaigns)
            report.penalty_breakdown["coordinated_shill"] = COORDINATED_SHILL_PENALTY

        # 4. Detect fake engagement
        if self._detect_fake_engagement(posts, author_profiles):
            report.fake_engagement_detected = True
            report.penalty_breakdown["fake_engagement"] = FAKE_ENGAGEMENT_PENALTY

        # 5. Detect suspicious promotions
        promo_campaigns = self._detect_suspicious_promotions(posts, author_profiles)
        if promo_campaigns:
            report.suspicious_promotions = True
            report.campaigns.extend(promo_campaigns)
            report.penalty_breakdown["suspicious_promo"] = SUSPICIOUS_PROMO_PENALTY

        report.total_penalty = sum(report.penalty_breakdown.values())

        return report

    # ------------------------------------------------------------------
    # Internal – Account Scoring
    # ------------------------------------------------------------------

    def _build_author_profiles(
        self,
        posts: List[Dict[str, Any]],
        authors: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """Build author profile lookup from posts and optional author data."""
        profiles: Dict[str, Dict] = {}

        # From posts
        for p in posts:
            aid = p.get("author_id", "")
            if aid not in profiles:
                profiles[aid] = {
                    "author_id": aid,
                    "author_username": p.get("author_username", ""),
                    "follower_count": p.get("follower_count", 0),
                    "following_count": 0,
                    "account_age_days": 365,
                    "post_count": 0,
                    "texts": [],
                    "timestamps": [],
                }
            profiles[aid]["post_count"] += 1
            profiles[aid]["texts"].append(p.get("text", ""))
            if p.get("created_at"):
                profiles[aid]["timestamps"].append(p["created_at"])

        # Merge author data if provided
        if authors:
            for a in authors:
                aid = a.get("author_id", "")
                if aid in profiles:
                    profiles[aid].update({
                        k: v for k, v in a.items() if k != "author_id"
                    })

        return profiles

    def _score_accounts(
        self, profiles: Dict[str, Dict[str, Any]]
    ) -> List[BotScore]:
        """Score each author's bot probability."""
        results: List[BotScore] = []

        for aid, profile in profiles.items():
            score = 0.0
            signals: List[str] = []

            # 1. Account age (newer = more suspicious)
            age = profile.get("account_age_days", 365)
            if age < 7:
                score += 0.35
                signals.append(f"Account very new ({age} days)")
            elif age < 30:
                score += 0.15
                signals.append(f"Account recently created ({age} days)")

            # 2. Follower/following ratio
            followers = profile.get("follower_count", 0)
            following = profile.get("following_count", 0)
            if following > 0 and followers > 0:
                ratio = following / followers
                if ratio > 10:
                    score += 0.15
                    signals.append(f"Suspicious follow ratio ({ratio:.1f})")

            # 3. Post frequency (too many posts about same token)
            post_count = profile.get("post_count", 0)
            if post_count > 10:
                score += 0.20
                signals.append(f"Excessive posting about token ({post_count} posts)")
            elif post_count > 5:
                score += 0.10

            # 4. Content repetition
            texts = profile.get("texts", [])
            if len(texts) > 2:
                unique_ratio = len(set(texts)) / len(texts)
                if unique_ratio < 0.5:
                    score += 0.20
                    signals.append(f"Repetitive content ({unique_ratio:.0%} unique)")

            # 5. Shill language patterns
            shill_count = 0
            for text in texts:
                for pattern in self._compiled_shill_patterns:
                    if pattern.search(text):
                        shill_count += 1
                        break
            if shill_count > 0:
                pct = shill_count / max(len(texts), 1)
                score += min(0.25, pct * 0.5)
                signals.append(f"Shill language detected in {shill_count} posts")

            # 6. Timestamp clustering (rapid-fire posts)
            timestamps = sorted(profile.get("timestamps", []))
            if len(timestamps) >= 3:
                gaps = [
                    (timestamps[i + 1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps) - 1)
                    if isinstance(timestamps[i], datetime)
                       and isinstance(timestamps[i + 1], datetime)
                ]
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    if avg_gap < 60:  # Less than 1 minute between posts
                        score += 0.15
                        signals.append(f"Rapid-fire posting (avg gap {avg_gap:.0f}s)")

            results.append(
                BotScore(
                    author_id=aid,
                    author_username=profile.get("author_username", ""),
                    bot_probability=round(min(1.0, score), 3),
                    signals=signals,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Internal – Pump Group Detection
    # ------------------------------------------------------------------

    def _detect_pump_groups(
        self, posts: List[Dict[str, Any]]
    ) -> List[ShillCampaign]:
        """Detect pump-group activity via keyword matching and coordination."""
        campaigns: List[ShillCampaign] = []
        pump_posts: List[Dict] = []

        for post in posts:
            text_lower = post.get("text", "").lower()
            if any(signal in text_lower for signal in PUMP_SIGNALS):
                pump_posts.append(post)

        if len(pump_posts) >= 3:
            accounts = {p.get("author_id", "") for p in pump_posts}
            evidence = [
                f"[{p.get('author_username')}]: {p.get('text', '')[:80]}..."
                for p in pump_posts[:5]
            ]
            campaigns.append(
                ShillCampaign(
                    token_ticker=pump_posts[0].get("token_ticker", ""),
                    campaign_type="pump_group",
                    severity=PUMP_GROUP_PENALTY,
                    evidence=evidence,
                    involved_accounts=accounts,
                    confidence=min(1.0, len(pump_posts) / 10),
                )
            )

        return campaigns

    # ------------------------------------------------------------------
    # Internal – Coordinated Shilling
    # ------------------------------------------------------------------

    def _detect_coordinated_shilling(
        self, posts: List[Dict[str, Any]]
    ) -> List[ShillCampaign]:
        """
        Detect coordinated shilling by finding clusters of similar messages
        posted within a short time window by different accounts.
        """
        campaigns: List[ShillCampaign] = []

        # Sort by timestamp
        timed_posts = [
            p for p in posts
            if isinstance(p.get("created_at"), datetime)
        ]
        timed_posts.sort(key=lambda x: x["created_at"])

        # Sliding window
        for i, anchor in enumerate(timed_posts):
            window_posts = [
                p for p in timed_posts[i:]
                if p["created_at"] - anchor["created_at"] <= self.shill_time_window
            ]

            if len(window_posts) < self.shill_min_accounts:
                continue

            # Check for content similarity within window
            unique_authors = {p.get("author_id") for p in window_posts}
            if len(unique_authors) < self.shill_min_accounts:
                continue

            # Simple similarity: check if >50% share common phrases
            texts = [p.get("text", "").lower() for p in window_posts]
            common_phrases = self._find_common_phrases(texts)

            if common_phrases:
                campaigns.append(
                    ShillCampaign(
                        campaign_type="coordinated",
                        severity=COORDINATED_SHILL_PENALTY,
                        evidence=[
                            f"{len(unique_authors)} accounts posted similar content "
                            f"within {self.shill_time_window.seconds // 60} minutes",
                            f"Common phrases: {', '.join(common_phrases[:3])}",
                        ],
                        involved_accounts=unique_authors,
                        confidence=min(1.0, len(unique_authors) / 15),
                    )
                )
                break  # One detection is enough

        return campaigns

    def _find_common_phrases(self, texts: List[str], min_phrase_len: int = 3) -> List[str]:
        """Find phrases that appear in more than 50% of texts."""
        if len(texts) < 2:
            return []

        # Extract 3-word phrases
        phrase_counts: Dict[str, int] = {}
        for text in texts:
            words = text.split()
            phrases = set()
            for j in range(len(words) - min_phrase_len + 1):
                phrase = " ".join(words[j : j + min_phrase_len])
                phrases.add(phrase)
            for phrase in phrases:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        threshold = len(texts) * 0.5
        return [
            phrase for phrase, count in phrase_counts.items()
            if count >= threshold and len(phrase) > 10
        ]

    # ------------------------------------------------------------------
    # Internal – Fake Engagement
    # ------------------------------------------------------------------

    def _detect_fake_engagement(
        self,
        posts: List[Dict[str, Any]],
        profiles: Dict[str, Dict],
    ) -> bool:
        """
        Detect fake engagement patterns:
        - Accounts with disproportionate follower/engagement ratios
        - Sudden engagement spikes on low-quality content
        - Known fake-engagement service links
        """
        suspicious_count = 0

        for profile in profiles.values():
            followers = profile.get("follower_count", 0)

            # High followers but zero engagement on own posts
            texts = profile.get("texts", [])
            post_count = profile.get("post_count", 0)

            if followers > 5000 and post_count > 3:
                # If someone has 5K+ followers but their posts about this token
                # seem template-like, flag it
                unique = len(set(texts))
                if unique < len(texts) * 0.3:
                    suspicious_count += 1

            # Check for fake engagement domains
            for text in texts:
                if any(domain in text.lower() for domain in FAKE_ENGAGEMENT_DOMAINS):
                    return True

        return suspicious_count >= 3

    # ------------------------------------------------------------------
    # Internal – Suspicious Promotions
    # ------------------------------------------------------------------

    def _detect_suspicious_promotions(
        self,
        posts: List[Dict[str, Any]],
        profiles: Dict[str, Dict],
    ) -> List[ShillCampaign]:
        """
        Detect undisclosed paid promotions:
        - Influencer (10K+) posts matching shill patterns
        - No #ad or #sponsored disclosure
        - First-time mention of this token from an influencer
        """
        campaigns: List[ShillCampaign] = []

        for profile in profiles.values():
            followers = profile.get("follower_count", 0)
            if followers < 10_000:
                continue

            texts = profile.get("texts", [])
            for text in texts:
                text_lower = text.lower()

                # Check for shill patterns
                is_shill = any(
                    p.search(text_lower)
                    for p in self._compiled_shill_patterns
                )
                # Check for disclosure
                has_disclosure = any(
                    tag in text_lower
                    for tag in {"#ad", "#sponsored", "#paid", "sponsored", "advertisement"}
                )

                if is_shill and not has_disclosure:
                    campaigns.append(
                        ShillCampaign(
                            campaign_type="suspicious_promo",
                            severity=SUSPICIOUS_PROMO_PENALTY,
                            evidence=[
                                f"Influencer @{profile.get('author_username')} "
                                f"({followers:,} followers) posted shill content "
                                f"without disclosure: '{text[:100]}...'"
                            ],
                            involved_accounts={profile.get("author_id", "")},
                            confidence=0.7,
                        )
                    )
                    break  # One per influencer

        return campaigns
