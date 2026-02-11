"""
Portfolio & Watchlist Routes
==============================
POST /api/v1/watchlist  – Manage watchlist (CRUD).
GET  /api/v1/portfolio  – Portfolio status with AI annotations.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.ui.api_schemas import ErrorResponse
from src.ui.user_config import PortfolioHolding
from src.ui.watchlist_manager import WatchlistManager
from src.ui.portfolio_tracker import PortfolioTracker, PortfolioSummary
from src.api.dependencies import get_user_prefs, verify_api_key

router = APIRouter()


# ══════════════════════════════════════════════════════════════════
# Watchlist
# ══════════════════════════════════════════════════════════════════

class WatchlistAddRequest(BaseModel):
    """Add a token to the watchlist."""
    ticker: str = Field(..., max_length=20)
    target_price: Optional[float] = None
    notes: Optional[str] = None


class WatchlistRemoveRequest(BaseModel):
    """Remove a token from the watchlist."""
    ticker: str


class WatchlistResponse(BaseModel):
    """Complete watchlist state."""
    items: List[dict] = Field(default_factory=list)
    alerts: List[dict] = Field(default_factory=list)
    count: int = 0


@router.post(
    "/watchlist",
    response_model=WatchlistResponse,
    summary="Manage watchlist",
    description="Add, remove, or list tokens on your watchlist.",
)
async def manage_watchlist(
    body: WatchlistAddRequest,
    prefs=Depends(get_user_prefs),
    _api_key: str = Depends(verify_api_key),
):
    """
    **POST /api/v1/watchlist**

    Add a token to the watchlist with optional price targets.
    """
    mgr = WatchlistManager(prefs)

    mgr.add(
        token=body.ticker.upper(),
        alert_price=body.target_price,
        notes=body.notes or "",
    )

    items = [itm.model_dump() for itm in mgr.list_all()]
    return WatchlistResponse(
        items=items,
        alerts=[],
        count=len(items),
    )


@router.get(
    "/watchlist",
    response_model=WatchlistResponse,
    summary="List watchlist",
)
async def list_watchlist(
    prefs=Depends(get_user_prefs),
    _api_key: str = Depends(verify_api_key),
):
    """Return the user's current watchlist."""
    mgr = WatchlistManager(prefs)
    items = [itm.model_dump() for itm in mgr.list_all()]
    return WatchlistResponse(items=items, alerts=[], count=len(items))


@router.delete(
    "/watchlist/{ticker}",
    response_model=WatchlistResponse,
    summary="Remove from watchlist",
)
async def remove_from_watchlist(
    ticker: str,
    prefs=Depends(get_user_prefs),
    _api_key: str = Depends(verify_api_key),
):
    """Remove a token from the watchlist."""
    mgr = WatchlistManager(prefs)
    mgr.remove(ticker.upper())
    items = [itm.model_dump() for itm in mgr.list_all()]
    return WatchlistResponse(items=items, alerts=[], count=len(items))


# ══════════════════════════════════════════════════════════════════
# Portfolio
# ══════════════════════════════════════════════════════════════════

class PortfolioResponse(BaseModel):
    """Portfolio summary with AI annotations."""
    total_value: float = 0
    total_pnl: float = 0
    total_pnl_pct: float = 0
    positions: List[dict] = Field(default_factory=list)
    ai_summary: str = ""


@router.get(
    "/portfolio",
    response_model=PortfolioResponse,
    summary="Get portfolio status",
    description="View portfolio positions with P&L and AI status annotations.",
)
async def get_portfolio(
    prefs=Depends(get_user_prefs),
    _api_key: str = Depends(verify_api_key),
):
    """
    **GET /api/v1/portfolio**

    Returns positions, P&L, and AI-generated HOLD / WATCH / SELL flags.
    """
    tracker = PortfolioTracker(prefs)
    summary = tracker.get_summary(current_prices={})

    return PortfolioResponse(
        total_value=summary.total_current_value,
        total_pnl=summary.total_pnl_dollar,
        total_pnl_pct=summary.total_pnl_percent,
        positions=[
            {"token": pos.token, "entry_price": pos.entry_price,
             "current_price": pos.current_price, "pnl_percent": pos.pnl_percent,
             "ai_status": pos.ai_status, "ai_comment": pos.ai_comment}
            for pos in summary.positions
        ],
        ai_summary="",
    )
