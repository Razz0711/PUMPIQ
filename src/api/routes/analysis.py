"""
Analysis Route
================
GET  /api/v1/token/{ticker}/analysis – Deep single-token analysis.
POST /api/v1/compare                 – Side-by-side token comparison.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, Field

from src.ui.api_schemas import AnalysisResponse, ComparisonResponse
from src.api.dependencies import get_service, get_user_prefs, verify_api_key

router = APIRouter()


# ══════════════════════════════════════════════════════════════════
# Single-token analysis
# ══════════════════════════════════════════════════════════════════

@router.get(
    "/token/{ticker}/analysis",
    response_model=AnalysisResponse,
    summary="Analyse a specific token",
    description=(
        "Run the full PumpIQ analysis pipeline on a single token "
        "and return a detailed verdict with entry/exit plan."
    ),
)
async def analyze_token(
    ticker: str = Path(
        ..., min_length=1, max_length=20,
        description="Token ticker symbol, e.g. BONK, SOL, WIF",
    ),
    modes: Optional[str] = Query(
        None,
        description="Comma-separated modes to enable: news,onchain,technical,social",
    ),
    service=Depends(get_service),
    prefs=Depends(get_user_prefs),
    _api_key: str = Depends(verify_api_key),
):
    """
    **GET /api/v1/token/{ticker}/analysis**

    Returns: verdict, confidence, risk, entry/exit, per-module analysis.
    """
    mode_list = [m.strip() for m in modes.split(",")] if modes else None
    return await service.analyze_token(
        ticker=ticker.upper(),
        prefs=prefs,
        modes=mode_list,
    )


# ══════════════════════════════════════════════════════════════════
# Token comparison
# ══════════════════════════════════════════════════════════════════

class CompareRequest(BaseModel):
    """Request body for the comparison endpoint."""
    tickers: List[str] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="List of 2-4 tickers to compare.",
        json_schema_extra={"examples": [["SOL", "BONK", "WIF"]]},
    )


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare tokens side-by-side",
    description="Run analysis on 2-4 tokens and return a comparative view.",
)
async def compare_tokens(
    body: CompareRequest,
    service=Depends(get_service),
    prefs=Depends(get_user_prefs),
    _api_key: str = Depends(verify_api_key),
):
    """
    **POST /api/v1/compare**

    Returns: per-token scores + comparison highlights.
    """
    return await service.compare_tokens(
        tickers=body.tickers,
        prefs=prefs,
    )
