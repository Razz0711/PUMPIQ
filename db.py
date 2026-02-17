"""
PumpIQ Database Abstraction Layer
=====================================
Provides a unified sqlite3-compatible interface for:
  - Local SQLite (development)
  - Google Cloud Storage-synced SQLite (production on Vercel — persistent)
  - Turso cloud database (alternative production option)

Priority: Turso > GCS-synced SQLite > ephemeral /tmp SQLite > local file

Required env vars for GCS (recommended for Vercel):
    GCS_BUCKET_NAME            – e.g. pumpiq-db
    GOOGLE_CREDENTIALS_BASE64  – base64-encoded service account JSON

Required env vars for Turso (alternative):
    TURSO_DATABASE_URL  – e.g. libsql://your-db-name-org.turso.io
    TURSO_AUTH_TOKEN     – Bearer token from Turso dashboard
"""

from __future__ import annotations

import os
import sqlite3
import logging
import threading
from typing import Any, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────

IS_VERCEL = bool(os.getenv("VERCEL"))

# Turso (highest priority)
TURSO_URL = os.getenv("TURSO_DATABASE_URL", "").strip()
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "").strip()
USE_TURSO = bool(TURSO_URL and TURSO_TOKEN)

# Google Cloud Storage (second priority)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "").strip()
GCS_CREDENTIALS_B64 = os.getenv("GOOGLE_CREDENTIALS_BASE64", "").strip()
USE_GCS = bool(GCS_BUCKET_NAME) and not USE_TURSO

GCS_DB_BLOB_NAME = "pumpiq.db"  # name of the DB file in the bucket
_gcs_synced = False  # track if we've downloaded from GCS this cold start
_gcs_lock = threading.Lock()  # thread-safe GCS sync

LOCAL_DB_PATH = (
    "/tmp/pumpiq.db" if (IS_VERCEL and not USE_TURSO)
    else os.path.join(os.path.dirname(__file__), "pumpiq.db")
)

if USE_TURSO:
    logger.info("Database: Turso cloud (%s)", TURSO_URL.split("//")[-1].split(".")[0] if "//" in TURSO_URL else "remote")
elif USE_GCS:
    logger.info("Database: GCS-synced SQLite (bucket=%s)", GCS_BUCKET_NAME)
elif IS_VERCEL:
    logger.warning("Database: Ephemeral /tmp SQLite (set GCS_BUCKET_NAME or TURSO_DATABASE_URL for persistence)")
else:
    logger.info("Database: Local SQLite (%s)", LOCAL_DB_PATH)


# ── Google Cloud Storage sync ──────────────────────────────────

def _get_gcs_client():
    """Get an authenticated GCS client."""
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        import base64
        import json

        if GCS_CREDENTIALS_B64:
            creds_json = json.loads(base64.b64decode(GCS_CREDENTIALS_B64))
            credentials = service_account.Credentials.from_service_account_info(creds_json)
            return storage.Client(credentials=credentials, project=creds_json.get("project_id"))
        else:
            # Fall back to default credentials (e.g. GOOGLE_APPLICATION_CREDENTIALS env)
            return storage.Client()
    except ImportError:
        logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
        raise
    except Exception as e:
        logger.error("Failed to create GCS client: %s", e)
        raise


def _download_from_gcs():
    """Download pumpiq.db from GCS to LOCAL_DB_PATH if it exists in the bucket."""
    global _gcs_synced
    if _gcs_synced:
        return  # Already downloaded this cold start

    with _gcs_lock:
        if _gcs_synced:
            return
        try:
            client = _get_gcs_client()
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(GCS_DB_BLOB_NAME)

            if blob.exists():
                blob.download_to_filename(LOCAL_DB_PATH)
                size_kb = os.path.getsize(LOCAL_DB_PATH) / 1024
                logger.info("GCS: Downloaded pumpiq.db (%.1f KB) from bucket %s", size_kb, GCS_BUCKET_NAME)
            else:
                logger.info("GCS: No existing pumpiq.db in bucket %s — starting fresh", GCS_BUCKET_NAME)

            _gcs_synced = True
        except Exception as e:
            logger.error("GCS download failed: %s — using local/empty DB", e)
            _gcs_synced = True  # Don't retry endlessly


def _upload_to_gcs():
    """Upload LOCAL_DB_PATH to GCS bucket (called after writes)."""
    try:
        if not os.path.exists(LOCAL_DB_PATH):
            return
        client = _get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_DB_BLOB_NAME)
        blob.upload_from_filename(LOCAL_DB_PATH)
        size_kb = os.path.getsize(LOCAL_DB_PATH) / 1024
        logger.debug("GCS: Uploaded pumpiq.db (%.1f KB) to bucket %s", size_kb, GCS_BUCKET_NAME)
    except Exception as e:
        logger.error("GCS upload failed: %s — data may be lost on cold start", e)


class _GCSWrappedConnection:
    """
    Wraps a sqlite3.Connection and auto-syncs to GCS on commit().
    All existing SQL code works unchanged.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self.row_factory = conn.row_factory

    def execute(self, sql: str, parameters: Sequence = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, parameters)

    def executescript(self, sql: str):
        result = self._conn.executescript(sql)
        _upload_to_gcs()  # executescript auto-commits in sqlite3
        return result

    def commit(self) -> None:
        self._conn.commit()
        # Sync to GCS after every commit (ensures persistence)
        _upload_to_gcs()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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
    Returns a sqlite3.Connection (local), _GCSWrappedConnection (GCS-synced),
    or _TursoConnection (cloud).
    All support the same interface: execute, executescript, commit, close.
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

    # GCS-synced SQLite: download DB from cloud on first access
    if USE_GCS:
        _download_from_gcs()

    # Local/GCS SQLite
    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA foreign_keys=ON")

    if USE_GCS:
        return _GCSWrappedConnection(conn)
    return conn


def get_db_path() -> str:
    """Return the local DB file path (for backward compat)."""
    return LOCAL_DB_PATH


def is_turso() -> bool:
    """Check if using Turso cloud database."""
    return USE_TURSO
