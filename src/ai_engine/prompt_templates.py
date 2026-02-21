"""
GPT-4o Prompt Engineering
===========================
Step 3.2 – System prompt + user prompt templates for every query type.

Templates are Jinja2-style (using str.format) with dynamic sections
that are toggled based on which data modes are enabled.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import (
    DataMode,
    InvestmentTimeframe,
    MarketCondition,
    RiskTolerance,
    TokenData,
    UserConfig,
    UserQuery,
)


# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert cryptocurrency trading analyst AI called NexYpher.
Your role is to synthesize data from multiple sources (news, on-chain metrics,
technical analysis, social sentiment) and provide clear, actionable trading
recommendations for crypto tokens.

Your analysis style:
- Direct and concise (traders need clarity, not fluff)
- Evidence-based (always cite which data informed your decision)
- Risk-aware (call out dangers, not just opportunities)
- Honest about uncertainty (if data conflicts, say so)

When providing recommendations:
1. Clearly state your confidence level (0-10)
2. Explain WHY you're recommending this token (which data sources made the case)
3. Highlight the STRONGEST signals (the most compelling reason to buy)
4. Call out CONCERNS or conflicting signals
5. Provide specific entry prices, targets, and stop-loss levels
6. Estimate the expected timeframe for the trade
7. Assign a risk rating (Low/Medium/High)

ENABLED DATA SOURCES:
- News Sentiment: {news_status}
- On-Chain Metrics: {onchain_status}
- Technical Analysis: {technical_status}
- Social Sentiment: {social_status}

IMPORTANT: When data sources conflict, you MUST:
1. Explicitly call out the conflict
2. Explain both sides of the argument
3. Weigh which data source is more reliable for this specific situation
4. Make a decision with appropriate confidence adjustment

Remember: Users are risking real money. Be thorough but realistic.
If a token is risky, say so clearly.
"""


def build_system_prompt(enabled_modes: List[DataMode]) -> str:
    """Render system prompt with enabled/disabled markers per mode."""
    return SYSTEM_PROMPT.format(
        news_status="ENABLED" if DataMode.NEWS in enabled_modes else "DISABLED",
        onchain_status="ENABLED" if DataMode.ONCHAIN in enabled_modes else "DISABLED",
        technical_status="ENABLED" if DataMode.TECHNICAL in enabled_modes else "DISABLED",
        social_status="ENABLED" if DataMode.SOCIAL in enabled_modes else "DISABLED",
    )


# ══════════════════════════════════════════════════════════════════
# USER PROMPT – "Best coins to buy now"
# ══════════════════════════════════════════════════════════════════

_BEST_COINS_HEADER = """\
I need you to analyze the following token data and recommend the \
TOP {num_recs} tokens to buy RIGHT NOW.

USER CONTEXT:
- Risk Tolerance: {risk_tolerance}
- Investment Timeframe: {timeframe}
- Position Size Preference: {position_size}

MARKET CONTEXT:
- Current Date: {date}
- Overall Crypto Market: {market_condition} (based on BTC trend)
"""

_TOKEN_SECTION_HEADER = """
═══════════════════════════════════════════════════════════════
TOKEN {idx}: {name} ({ticker})
═══════════════════════════════════════════════════════════════
Current Price: ${price}
Overall Score: {composite_score}/10
"""

_NEWS_BLOCK = """
📰 NEWS DATA:
- Sentiment Score: {score}/10
- Key Headlines:
{headlines}
- Narrative: {narrative}
"""

_ONCHAIN_BLOCK = """
⛓️ ON-CHAIN DATA:
- Health Score: {score}/10
- Holder Count: {holder_count} ({holder_growth}% in 24h)
- Top 10 Concentration: {top10_pct}%
- 24h Volume: ${volume_24h} ({volume_trend})
- Liquidity: ${liquidity} (ratio: {liq_ratio})
- Smart Money Activity: {smart_money}
"""

_TECHNICAL_BLOCK = """
📊 TECHNICAL DATA:
- Technical Score: {score}/10
- Trend: {trend}
- RSI: {rsi} ({rsi_label})
- MACD: {macd_signal}
- Key Levels: Support ${support}, Resistance ${resistance}
- Pattern: {pattern}
"""

_SOCIAL_BLOCK = """
💬 SOCIAL DATA:
- Sentiment Score: {score}/{score_max}
- 24h Mentions: {mentions} ({mention_trend})
- Influencer Signals: {influencer_count} influencers mentioned
- Community: {telegram_members} members ({growth}% growth)
- Trending: {trending}
"""

_RISK_FACTORS_BLOCK = """
⚠️ RISK FACTORS:
{risk_factors}
"""

_CONFLICT_BLOCK = """
⚡ CONFLICTING SIGNALS:
{conflicts}
"""

_BEST_COINS_INSTRUCTIONS = """
───────────────────────────────────────────────────────────────
INSTRUCTIONS:
Analyze the data above and provide recommendations in this EXACT format:

🏆 RECOMMENDATION #N: {Token Name}
Confidence: X/10 | Risk: Low/Medium/High
Current Price: $X.XX
Entry Zone: $X.XX - $X.XX
Target 1: $X.XX (+XX%)
Target 2: $X.XX (+XX%)
Stop Loss: $X.XX (-XX%)
Timeframe: {estimate}

🎯 THE CASE FOR THIS PICK:
{2-3 sentences explaining why this is your top pick. Highlight the STRONGEST signals from the data.}

📊 KEY DATA POINTS:
{Bullet points of the most compelling metrics from each enabled data source}

⚠️ WHAT COULD GO WRONG:
{1-2 sentences on the main risks}

Repeat for each recommendation.

FINAL THOUGHTS:
{Brief summary comparing all picks, and any general market advice}
"""


# ══════════════════════════════════════════════════════════════════
# USER PROMPT – "Analyze [TOKEN]"
# ══════════════════════════════════════════════════════════════════

_ANALYZE_TOKEN_HEADER = """\
Perform a deep analysis of {name} ({ticker}) and tell me if I should \
buy, hold, or avoid it.

USER CONTEXT:
- Current Holdings: {holdings}
- Risk Tolerance: {risk_tolerance}

MARKET CONTEXT:
- Current Date: {date}
- Overall Crypto Market: {market_condition}
"""

_ANALYZE_TOKEN_INSTRUCTIONS = """
───────────────────────────────────────────────────────────────
INSTRUCTIONS:
Provide a comprehensive analysis with the following sections:

📈 OVERALL VERDICT: {{Strong Buy / Moderate Buy / Hold / Avoid / Sell}}
Confidence: X/10 | Risk: Low/Medium/High

💡 EXECUTIVE SUMMARY:
{{3-4 sentence overview of whether this is a good buy right now and why}}

📊 DETAILED BREAKDOWN:
{mode_sections}

🎯 ACTIONABLE ADVICE:
IF BUY:
- Entry: $X.XX - $X.XX
- Target 1: $X.XX
- Target 2: $X.XX
- Stop Loss: $X.XX
- Position Size: {{based on risk}}

IF HOLD:
- Current holders should {{add more / maintain / reduce}}
- Watch for {{specific conditions to change decision}}

IF AVOID/SELL:
- Why you should stay away or exit
- Alternative tokens to consider instead

⚠️ RISKS & CONCERNS:
{{Comprehensive risk assessment}}

⏰ TIMING:
{{When to execute this trade, and expected timeframe for results}}
"""

_ANALYZE_MODE_SECTION_NEWS = """\
[News Analysis]
Analyze the news sentiment, key narratives, and how news might impact price."""

_ANALYZE_MODE_SECTION_ONCHAIN = """\
[On-Chain Analysis]
Analyze holder distribution, volume trends, liquidity health, whale activity."""

_ANALYZE_MODE_SECTION_TECHNICAL = """\
[Technical Analysis]
Analyze chart setup, indicators, support/resistance, entry/exit levels."""

_ANALYZE_MODE_SECTION_SOCIAL = """\
[Social Sentiment Analysis]
Analyze community buzz, influencer activity, trending status."""


# ══════════════════════════════════════════════════════════════════
# Edge-case: no qualifying tokens
# ══════════════════════════════════════════════════════════════════

NO_TOKENS_PROMPT = """\
I asked for the top tokens to buy, but after scoring and filtering \
no tokens met the minimum criteria (composite score > 6/10, no critical \
red flags, minimum liquidity met).

Please respond with:
1. A brief explanation of why no recommendations can be made right now
2. What general market conditions users should watch for
3. A reminder that no recommendation is sometimes the best recommendation

Do NOT fabricate or force a recommendation.
"""

# ══════════════════════════════════════════════════════════════════
# Edge-case: only 1 mode enabled
# ══════════════════════════════════════════════════════════════════

SINGLE_MODE_DISCLAIMER = """\
⚠️ NOTE: Only {mode_name} data is enabled for this analysis.
Confidence is capped at 6/10 because a single data source cannot
provide the multi-dimensional view needed for high-conviction calls.
Consider enabling more data modes for stronger recommendations.
"""


# ══════════════════════════════════════════════════════════════════
# Builder Functions
# ══════════════════════════════════════════════════════════════════

class PromptBuilder:
    """
    Builds ready-to-send GPT-4o prompts from structured data.

    Usage::

        builder = PromptBuilder()
        system_msg = builder.system_prompt(config)
        user_msg   = builder.best_coins_prompt(query, config, tokens, market)
    """

    # ── System prompt ─────────────────────────────────────────────

    @staticmethod
    def system_prompt(config: UserConfig) -> str:
        return build_system_prompt(config.enabled_modes)

    # ── Best coins prompt ─────────────────────────────────────────

    def best_coins_prompt(
        self,
        query: UserQuery,
        config: UserConfig,
        tokens: List[TokenData],
        market: MarketCondition = MarketCondition.SIDEWAYS,
    ) -> str:
        parts: List[str] = []

        # Header
        parts.append(_BEST_COINS_HEADER.format(
            num_recs=query.num_recommendations,
            risk_tolerance=query.risk_tolerance.value.title(),
            timeframe=query.timeframe.value.replace("_", " ").title(),
            position_size=query.position_size.value.title(),
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            market_condition=market.value.title(),
        ))

        # Single-mode disclaimer
        if len(config.enabled_modes) == 1:
            mode_name = config.enabled_modes[0].value.title()
            parts.append(SINGLE_MODE_DISCLAIMER.format(mode_name=mode_name))

        # Token sections
        for idx, token in enumerate(tokens, 1):
            parts.append(self._render_token_section(
                idx, token, config.enabled_modes,
            ))

        # Instructions
        parts.append(_BEST_COINS_INSTRUCTIONS)
        return "\n".join(parts)

    # ── Analyse single token prompt ───────────────────────────────

    def analyze_token_prompt(
        self,
        query: UserQuery,
        config: UserConfig,
        token: TokenData,
        market: MarketCondition = MarketCondition.SIDEWAYS,
    ) -> str:
        parts: List[str] = []

        holdings_text = (
            f"User already holds {token.token_ticker}"
            if token.token_ticker in query.held_tokens
            else "None"
        )

        parts.append(_ANALYZE_TOKEN_HEADER.format(
            name=token.token_name,
            ticker=token.token_ticker,
            holdings=holdings_text,
            risk_tolerance=query.risk_tolerance.value.title(),
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            market_condition=market.value.title(),
        ))

        if len(config.enabled_modes) == 1:
            parts.append(SINGLE_MODE_DISCLAIMER.format(
                mode_name=config.enabled_modes[0].value.title(),
            ))

        parts.append(self._render_token_section(
            1, token, config.enabled_modes,
        ))

        # Mode sections for instructions
        mode_sections: List[str] = []
        if DataMode.NEWS in config.enabled_modes:
            mode_sections.append(_ANALYZE_MODE_SECTION_NEWS)
        if DataMode.ONCHAIN in config.enabled_modes:
            mode_sections.append(_ANALYZE_MODE_SECTION_ONCHAIN)
        if DataMode.TECHNICAL in config.enabled_modes:
            mode_sections.append(_ANALYZE_MODE_SECTION_TECHNICAL)
        if DataMode.SOCIAL in config.enabled_modes:
            mode_sections.append(_ANALYZE_MODE_SECTION_SOCIAL)

        parts.append(_ANALYZE_TOKEN_INSTRUCTIONS.format(
            mode_sections="\n".join(mode_sections),
        ))

        return "\n".join(parts)

    # ── No tokens fallback ────────────────────────────────────────

    @staticmethod
    def no_tokens_prompt() -> str:
        return NO_TOKENS_PROMPT

    # ── Internal rendering ────────────────────────────────────────

    def _render_token_section(
        self,
        idx: int,
        token: TokenData,
        modes: List[DataMode],
    ) -> str:
        parts: List[str] = []

        parts.append(_TOKEN_SECTION_HEADER.format(
            idx=idx,
            name=token.token_name,
            ticker=token.token_ticker,
            price=f"{token.current_price:.8g}",
            composite_score=f"{token.composite_score:.1f}",
        ))

        if DataMode.NEWS in modes and token.news:
            n = token.news
            headlines = "\n".join(f"  • {h}" for h in n.key_headlines[:3]) or "  (none)"
            parts.append(_NEWS_BLOCK.format(
                score=f"{n.score:.1f}",
                headlines=headlines,
                narrative=n.narrative or n.summary,
            ))

        if DataMode.ONCHAIN in modes and token.onchain:
            o = token.onchain
            parts.append(_ONCHAIN_BLOCK.format(
                score=f"{o.score:.1f}",
                holder_count=f"{o.holder_count:,}",
                holder_growth=f"{o.holder_growth_24h:+.1f}",
                top10_pct=f"{o.top_10_concentration:.1f}",
                volume_24h=f"{o.volume_24h:,.0f}",
                volume_trend=o.volume_trend or "N/A",
                liquidity=f"{o.liquidity:,.0f}",
                liq_ratio=f"{o.liquidity_mcap_ratio:.2f}",
                smart_money=o.smart_money_summary or "N/A",
            ))

        if DataMode.TECHNICAL in modes and token.technical:
            t = token.technical
            parts.append(_TECHNICAL_BLOCK.format(
                score=f"{t.score:.1f}",
                trend=t.trend,
                rsi=f"{t.rsi:.1f}",
                rsi_label=t.rsi_label,
                macd_signal=t.macd_signal,
                support=f"{t.support:.8g}",
                resistance=f"{t.resistance:.8g}",
                pattern=t.pattern,
            ))

        if DataMode.SOCIAL in modes and token.social:
            s = token.social
            parts.append(_SOCIAL_BLOCK.format(
                score=f"{s.score:.1f}",
                score_max=f"{s.score_max:.0f}",
                mentions=f"{s.mention_count_24h:,}",
                mention_trend=s.mention_trend or "N/A",
                influencer_count=s.influencer_count,
                telegram_members=f"{s.telegram_members:,}",
                growth=f"{s.community_growth:+.1f}",
                trending=s.trending_status or "N/A",
            ))

        # Risk factors
        if token.risk_factors:
            factors = "\n".join(f"  • {r}" for r in token.risk_factors)
            parts.append(_RISK_FACTORS_BLOCK.format(risk_factors=factors))

        # Conflicts
        if token.conflicts:
            conflicts = "\n".join(
                f"  ⚡ {c.description}" for c in token.conflicts
            )
            parts.append(_CONFLICT_BLOCK.format(conflicts=conflicts))

        return "\n".join(parts)
