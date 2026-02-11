"""
PumpIQ – API Route Modules
=============================
All FastAPI routers are registered from here.
"""

from .recommendations import router as recommendations_router
from .analysis import router as analysis_router
from .portfolio import router as portfolio_router
from .health import router as health_router

__all__ = [
    "recommendations_router",
    "analysis_router",
    "portfolio_router",
    "health_router",
]
