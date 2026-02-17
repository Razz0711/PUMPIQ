"""
PumpIQ — Supabase Database Client
====================================
Configures and exports a Supabase client singleton.

Required environment variables (set in .env):
  SUPABASE_URL  – Your Supabase project URL   (e.g. https://xyzcompany.supabase.co)
  SUPABASE_KEY  – Your Supabase service-role key (or anon key for client-side)

Get these from:  https://app.supabase.com → Your Project → Settings → API
"""

from __future__ import annotations

import os
from supabase import create_client, Client

SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

_client: Client | None = None


def get_supabase() -> Client:
    """Return the Supabase client singleton. Raises RuntimeError if not configured."""
    global _client
    if _client is None:
        url = SUPABASE_URL or os.getenv("SUPABASE_URL", "")
        key = SUPABASE_KEY or os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError(
                "Supabase is not configured.\n"
                "Set SUPABASE_URL and SUPABASE_KEY in your .env file.\n"
                "Get these from: https://app.supabase.com → Your Project → Settings → API"
            )
        _client = create_client(url, key)
    return _client
