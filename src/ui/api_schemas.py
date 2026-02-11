"""
API Response Schema
=====================
Step 4.2 – Pydantic models defining the canonical JSON response
returned by the PumpIQ REST API.

These schemas serve as:
  1. Response validation (FastAPI will auto-generate OpenAPI docs)
  2. Client SDK type hints
  3. API contract documentation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
# Nested sub-schemas
# ══════════════════════════════════════════════════════════════════

class TokenInfo(BaseModel):
    """Basic token identity + current price."""
    name: str
    ticker: str
    contract_address: Optional[str] = None
    current_price: float = 0.0
    logo_url: Optional[str] = None


class ScoreBlock(BaseModel):
    """Scores breakdown."""
    overall: float = Field(0, ge=0, le=10)
    confidence: float = Field(0, ge=0, le=10)
    risk: str = "MEDIUM"               # LOW | MEDIUM | HIGH
    news: Optional[float] = None       # 0-10
    onchain: Optional[float] = None    # 0-10
    technical: Optional[float] = None  # 0-10
    social: Optional[float] = None     # 0-12 (raw) or 0-10 (normalised)


class EntryExitBlock(BaseModel):
    """Entry zone, targets, stop-loss."""
    entry_min: float = 0
    entry_max: float = 0
    target_1: float = 0
    target_1_percent: float = 0
    target_2: float = 0
    target_2_percent: float = 0
    stop_loss: float = 0
    stop_loss_percent: float = 0
    timeframe: str = ""


class DetailedAnalysisBlock(BaseModel):
    """Per-module analysis summaries (populated when expanded)."""
    news: Optional[str] = None
    onchain: Optional[str] = None
    technical: Optional[str] = None
    social: Optional[str] = None


class ConflictBlock(BaseModel):
    """A single conflict between two modules."""
    severity: str = "minor"
    modules: str = ""            # "news vs onchain"
    description: str = ""


# ══════════════════════════════════════════════════════════════════
# Single Recommendation
# ══════════════════════════════════════════════════════════════════

class RecommendationResponse(BaseModel):
    """A single recommendation in the API response."""
    rank: int
    token: TokenInfo
    scores: ScoreBlock
    entry_exit: EntryExitBlock
    verdict: str = "Watch"
    thesis: str = ""
    key_data_points: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    conflicts: List[ConflictBlock] = Field(default_factory=list)
    detailed_analysis: DetailedAnalysisBlock = Field(
        default_factory=DetailedAnalysisBlock
    )
    position_sizing: str = ""


# ══════════════════════════════════════════════════════════════════
# Market Context
# ══════════════════════════════════════════════════════════════════

class MarketContext(BaseModel):
    """Broad market context included with every response."""
    condition: str = "sideways"           # bull | bear | sideways
    bitcoin_price: Optional[float] = None
    bitcoin_trend: Optional[str] = None
    overall_sentiment: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# Metadata
# ══════════════════════════════════════════════════════════════════

class ResponseMetadata(BaseModel):
    """Processing metadata."""
    modes_enabled: List[str] = Field(default_factory=list)
    tokens_analyzed: int = 0
    tokens_filtered_out: int = 0
    data_freshness_minutes: float = 0
    processing_time_ms: float = 0


# ══════════════════════════════════════════════════════════════════
# Top-level Responses
# ══════════════════════════════════════════════════════════════════

class RecommendationSetResponse(BaseModel):
    """
    Top-level API response for recommendation endpoints.

    FastAPI usage::

        @app.get("/api/v1/recommendations", response_model=RecommendationSetResponse)
        async def get_recommendations(...):
            ...
    """
    query_timestamp: str = ""
    recommendations: List[RecommendationResponse] = Field(default_factory=list)
    market_context: MarketContext = Field(default_factory=MarketContext)
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class AnalysisResponse(BaseModel):
    """
    Detailed single-token analysis response.

    FastAPI usage::

        @app.get("/api/v1/analyze/{ticker}", response_model=AnalysisResponse)
        async def analyze_token(ticker: str, ...):
            ...
    """
    query_timestamp: str = ""
    token: TokenInfo = Field(default_factory=TokenInfo)
    verdict: str = "Watch"
    scores: ScoreBlock = Field(default_factory=ScoreBlock)
    entry_exit: EntryExitBlock = Field(default_factory=EntryExitBlock)
    thesis: str = ""
    key_data_points: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    conflicts: List[ConflictBlock] = Field(default_factory=list)
    detailed_analysis: DetailedAnalysisBlock = Field(
        default_factory=DetailedAnalysisBlock
    )
    position_sizing: str = ""
    market_context: MarketContext = Field(default_factory=MarketContext)
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class ComparisonResponse(BaseModel):
    """
    Side-by-side comparison of 2-4 tokens.
    """
    query_timestamp: str = ""
    tokens: List[RecommendationResponse] = Field(default_factory=list)
    highlights: Dict[str, str] = Field(default_factory=dict)
    market_context: MarketContext = Field(default_factory=MarketContext)
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class ClarificationResponse(BaseModel):
    """
    Response indicating the server needs clarification.
    """
    needs_clarification: bool = True
    message: str = ""
    options: List[Dict[str, str]] = Field(default_factory=list)
    free_text_allowed: bool = False


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    error: str
    detail: Optional[str] = None
    code: int = 400
