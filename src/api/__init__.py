"""
PumpIQ – API Integration Layer
=================================
Step 5.1: End-to-End System Integration.

Public API::

    from src.api import create_app

    app = create_app()
    # uvicorn src.api.app:app --reload
"""

from .app import create_app
from .service_layer import PumpIQService
from .dependencies import get_service, get_user_prefs
from .error_handlers import register_error_handlers

__all__ = [
    "create_app",
    "PumpIQService",
    "get_service",
    "get_user_prefs",
    "register_error_handlers",
]
