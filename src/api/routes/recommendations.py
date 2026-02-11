"""
Recommendations Route
=======================
POST /api/v1/recommendations – Get AI-powered token recommendations.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from src.ui.api_schemas import RecommendationSetResponse
from src.api.dependencies import get_service, get_user_prefs, verify_api_key

router = APIRouter()


# ── Request Body ──────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    """Request body for the recommendation endpoint."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural-language query, e.g. 'What are the best coins to buy?'",
        json_schema_extra={"examples": ["What are the best coins to buy right now?"]},
    )
    num_recommendations: int = Field(
        default=3, ge=1, le=10,
        description="Number of recommendations to return.",
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Override timeframe: scalp | day | swing | long",
    )
    risk_preference: Optional[str] = Field(
        default=None,
        description="Override risk: conservative | moderate | aggressive",
    )
    modes: Optional[List[str]] = Field(
        default=None,
        description="Restrict data modes: [news, onchain, technical, social]",
    )


# ── Endpoint ──────────────────────────────────────────────────────

@router.post(
    "/recommendations",
    response_model=RecommendationSetResponse,
    summary="Get AI-powered recommendations",
    description=(
        "Submit a natural-language query and receive ranked token "
        "recommendations with confidence scores, entry/exit plans, "
        "and risk assessments."
    ),
)
async def get_recommendations(
    body: RecommendationRequest,
    service=Depends(get_service),
    prefs=Depends(get_user_prefs),
    _api_key: str = Depends(verify_api_key),
):
    """
    **POST /api/v1/recommendations**

    Pipeline: parse query → collect data → composite scoring →
    conflict detection → rank & filter → GPT-4o synthesis.
    """
    return await service.get_recommendations(
        query_text=body.query,
        prefs=prefs,
        num_recommendations=body.num_recommendations,
        timeframe=body.timeframe,
        risk=body.risk_preference,
        modes=body.modes,
    )
