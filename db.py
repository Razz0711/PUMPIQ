"""
PumpIQ Database Abstraction Layer
=====================================
Provides a unified sqlite3-compatible interface for both:
  - Local SQLite (development)
  - Turso cloud database (production on Vercel)

When TURSO_DATABASE_URL and TURSO_AUTH_TOKEN env vars are set,
all database operations go to Turso (persistent cloud SQLite).
Otherwise, falls back to local SQLite file.

Required env vars for Turso (production):
    TURSO_DATABASE_URL  – e.g. libsql://your-db-name-org.turso.io
    TURSO_AUTH_TOKEN     – Bearer token from Turso dashboard
"""

from __future__ import annotations

import os
import sqlite3
import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────

IS_VERCEL = bool(os.getenv("VERCEL"))
TURSO_URL = os.getenv("TURSO_DATABASE_URL", "").strip()
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "").strip()
USE_TURSO = bool(TURSO_URL and TURSO_TOKEN)

LOCAL_DB_PATH = (
    "/tmp/pumpiq.db" if (IS_VERCEL and not USE_TURSO)
    else os.path.join(os.path.dirname(__file__), "pumpiq.db")
)

if USE_TURSO:
    logger.info("Database: Turso cloud (%s)", TURSO_URL.split("//")[-1].split(".")[0] if "//" in TURSO_URL else "remote")
elif IS_VERCEL:
    logger.warning("Database: Ephemeral /tmp SQLite (set TURSO_DATABASE_URL for persistence)")
else:
    logger.info("Database: Local SQLite (%s)", LOCAL_DB_PATH)


# ── Turso sqlite3-compatible wrapper ───────────────────────────

class _TursoRow:
    """Row object that supports dict-like access by column name (like sqlite3.Row)."""

    __slots__ = ("_columns", "_values", "_map")

    def __init__(self, columns: Tuple[str, ...], values: tuple):
        self._columns = columns
        self._values = values
        self._map = {c: v for c, v in zip(columns, values)}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return self._map[key]

    def __contains__(self, key):
        return key in self._map

    def __repr__(self):
        return f"Row({self._map})"

    def keys(self):
        return self._columns

    def values(self):
        return self._values

    def items(self):
        return zip(self._columns, self._values)

    def get(self, key, default=None):
        return self._map.get(key, default)


class _TursoCursor:
    """Cursor-like wrapper for Turso ResultSet."""

    def __init__(self, columns: Tuple[str, ...], rows: list, last_insert_rowid: int, rows_affected: int):
        self._columns = columns
        self._rows = rows
        self.lastrowid = last_insert_rowid
        self.rowcount = rows_affected
        self._pos = 0

    def fetchone(self) -> Optional[_TursoRow]:
        if self._pos < len(self._rows):
            row = self._rows[self._pos]
            self._pos += 1
            return row
        return None

    def fetchall(self) -> List[_TursoRow]:
        remaining = self._rows[self._pos:]
        self._pos = len(self._rows)
        return remaining

    def __iter__(self):
        return iter(self._rows)


class _TursoConnection:
    """sqlite3.Connection-compatible wrapper for Turso via libsql_client."""

    def __init__(self, client):
        self._client = client
        self.row_factory = None  # Accepted but not used (always returns _TursoRow)

    def execute(self, sql: str, parameters: Sequence = ()) -> _TursoCursor:
        """Execute a single SQL statement. Parameters use ? placeholders (sqlite3 style)."""
        try:
            # Convert sqlite3-style tuple params to list for libsql_client
            params = list(parameters) if parameters else []
            result = self._client.execute(sql, params)
            columns = tuple(result.columns) if result.columns else ()
            rows = []
            for r in result.rows:
                values = r.astuple() if hasattr(r, "astuple") else tuple(r)
                rows.append(_TursoRow(columns, values))
            return _TursoCursor(
                columns,
                rows,
                result.last_insert_rowid or 0,
                result.rows_affected or 0,
            )
        except Exception as e:
            logger.error("Turso execute error: %s | SQL: %s", e, sql[:200])
            raise

    def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements separated by semicolons."""
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                self._client.execute(stmt)
            except Exception as e:
                # Log but continue (CREATE IF NOT EXISTS may fail on some edge cases)
                logger.debug("Turso executescript stmt error (continuing): %s", e)

    def commit(self) -> None:
        """No-op: Turso auto-commits each statement."""
        pass

    def close(self) -> None:
        """Close the Turso client."""
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Public API ──────────────────────────────────────────────────

def get_db():
    """
    Get a database connection.
    Returns a sqlite3.Connection (local) or _TursoConnection (cloud).
    Both support the same interface: execute, executescript, commit, close.
    Row access by column name: row["column"].
    """
    if USE_TURSO:
        try:
            from libsql_client import create_client_sync
            client = create_client_sync(url=TURSO_URL, auth_token=TURSO_TOKEN)
            return _TursoConnection(client)
        except ImportError:
            logger.error("libsql-client not installed. Run: pip install libsql-client")
            raise
        except Exception as e:
            logger.error("Failed to connect to Turso: %s", e)
            raise

    # Local SQLite fallback
    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_db_path() -> str:
    """Return the local DB file path (for backward compat)."""
    return LOCAL_DB_PATH


def is_turso() -> bool:
    """Check if using Turso cloud database."""
    return USE_TURSO
