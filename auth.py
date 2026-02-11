"""
PumpIQ Authentication Module
==============================
SQLite-backed user management with JWT tokens, wallet linking, and portfolio tracking.
"""

from __future__ import annotations

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

import smtp_service

# ── Config ──────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "pumpiq_dev_secret_key_change_in_production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

DB_PATH = os.path.join(os.path.dirname(__file__), "pumpiq.db")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Pydantic Models ─────────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    username: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    email_verified: bool = False
    wallets: List[Dict[str, str]] = []
    watchlist: List[str] = []
    created_at: str

class WalletAdd(BaseModel):
    address: str
    chain: str = "ethereum"  # ethereum, solana, base
    label: str = ""

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ── Database Setup ──────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email_verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_login TEXT
        );

        CREATE TABLE IF NOT EXISTS wallets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            address TEXT NOT NULL,
            chain TEXT NOT NULL DEFAULT 'ethereum',
            label TEXT DEFAULT '',
            added_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, address, chain)
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            coin_id TEXT NOT NULL,
            added_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, coin_id)
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            coin_id TEXT NOT NULL,
            amount REAL NOT NULL DEFAULT 0,
            avg_buy_price REAL NOT NULL DEFAULT 0,
            notes TEXT DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, coin_id)
        );

        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            coin_id TEXT NOT NULL,
            action TEXT NOT NULL,
            amount REAL NOT NULL,
            price REAL NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            wallet_address TEXT DEFAULT '',
            tx_hash TEXT DEFAULT '',
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS email_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            token_type TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)
    # Migration: add email_verified column if missing (existing DBs)
    try:
        conn.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    conn.close()


# ── Auth Functions ──────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_access_token(user_id: int, email: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expires,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ── User CRUD ───────────────────────────────────────────────────

def register_user(email: str, username: str, password: str) -> Optional[UserResponse]:
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO users (email, username, password_hash) VALUES (?, ?, ?)",
            (email.lower(), username, hash_password(password)),
        )
        conn.commit()
        user_id = cursor.lastrowid
        return _get_user_response(conn, user_id)
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> Optional[UserResponse]:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower(),)
        ).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            return None
        conn.execute(
            "UPDATE users SET last_login = datetime('now') WHERE id = ?", (row["id"],)
        )
        conn.commit()
        return _get_user_response(conn, row["id"])
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[UserResponse]:
    conn = get_db()
    try:
        return _get_user_response(conn, user_id)
    finally:
        conn.close()


def _get_user_response(conn: sqlite3.Connection, user_id: int) -> Optional[UserResponse]:
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        return None

    wallets = conn.execute(
        "SELECT address, chain, label FROM wallets WHERE user_id = ?", (user_id,)
    ).fetchall()
    watchlist = conn.execute(
        "SELECT coin_id FROM watchlist WHERE user_id = ?", (user_id,)
    ).fetchall()

    return UserResponse(
        id=row["id"],
        email=row["email"],
        username=row["username"],
        email_verified=bool(row["email_verified"]) if "email_verified" in row.keys() else False,
        wallets=[dict(w) for w in wallets],
        watchlist=[w["coin_id"] for w in watchlist],
        created_at=row["created_at"],
    )


# ── Wallet Management ──────────────────────────────────────────

def add_wallet(user_id: int, address: str, chain: str = "ethereum", label: str = "") -> bool:
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO wallets (user_id, address, chain, label) VALUES (?, ?, ?, ?)",
            (user_id, address.lower(), chain.lower(), label),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def remove_wallet(user_id: int, address: str, chain: str = "ethereum") -> bool:
    conn = get_db()
    try:
        conn.execute(
            "DELETE FROM wallets WHERE user_id = ? AND address = ? AND chain = ?",
            (user_id, address.lower(), chain.lower()),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_wallets(user_id: int) -> List[Dict[str, str]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT address, chain, label, added_at FROM wallets WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Watchlist ───────────────────────────────────────────────────

def add_to_watchlist(user_id: int, coin_id: str) -> bool:
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (user_id, coin_id) VALUES (?, ?)",
            (user_id, coin_id.lower()),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def remove_from_watchlist(user_id: int, coin_id: str) -> bool:
    conn = get_db()
    try:
        conn.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND coin_id = ?",
            (user_id, coin_id.lower()),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_watchlist(user_id: int) -> List[str]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT coin_id FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [r["coin_id"] for r in rows]
    finally:
        conn.close()


# ── Portfolio ───────────────────────────────────────────────────

def update_portfolio(user_id: int, coin_id: str, amount: float, avg_price: float, notes: str = "") -> bool:
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO portfolio (user_id, coin_id, amount, avg_buy_price, notes, updated_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(user_id, coin_id) DO UPDATE SET
                   amount = excluded.amount,
                   avg_buy_price = excluded.avg_buy_price,
                   notes = excluded.notes,
                   updated_at = datetime('now')""",
            (user_id, coin_id.lower(), amount, avg_price, notes),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_portfolio(user_id: int) -> List[Dict[str, Any]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT coin_id, amount, avg_buy_price, notes, updated_at FROM portfolio WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Email Verification ──────────────────────────────────────────

def send_verification_email(user_id: int) -> bool:
    """Generate token and send verification email."""
    conn = get_db()
    try:
        row = conn.execute("SELECT email, username FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return False

        token = smtp_service.generate_verification_token()
        expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()

        conn.execute(
            "INSERT INTO email_tokens (user_id, token, token_type, expires_at) VALUES (?, ?, 'verify', ?)",
            (user_id, token, expires),
        )
        conn.commit()

        return smtp_service.send_verification_email(row["email"], row["username"], token)
    finally:
        conn.close()


def verify_email(token: str) -> Optional[str]:
    """Verify email from token. Returns error string or None on success."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM email_tokens WHERE token = ? AND token_type = 'verify' AND used = 0",
            (token,),
        ).fetchone()
        if not row:
            return "Invalid or expired verification link"

        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            return "Verification link has expired"

        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (row["user_id"],))
        conn.execute("UPDATE email_tokens SET used = 1 WHERE id = ?", (row["id"],))
        conn.commit()

        # Send welcome email
        user = conn.execute("SELECT email, username FROM users WHERE id = ?", (row["user_id"],)).fetchone()
        if user:
            smtp_service.send_welcome_email(user["email"], user["username"])

        return None  # success
    finally:
        conn.close()


# ── Password Reset ──────────────────────────────────────────────

def request_password_reset(email: str) -> bool:
    """Generate reset token and send email. Returns True if user exists."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, username FROM users WHERE email = ?", (email.lower(),)
        ).fetchone()
        if not row:
            return False  # user not found (don't reveal this to frontend)

        token = smtp_service.generate_reset_token()
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

        # Invalidate old reset tokens
        conn.execute(
            "UPDATE email_tokens SET used = 1 WHERE user_id = ? AND token_type = 'reset' AND used = 0",
            (row["id"],),
        )
        conn.execute(
            "INSERT INTO email_tokens (user_id, token, token_type, expires_at) VALUES (?, ?, 'reset', ?)",
            (row["id"], token, expires),
        )
        conn.commit()

        return smtp_service.send_password_reset_email(email.lower(), row["username"], token)
    finally:
        conn.close()


def reset_password(token: str, new_password: str) -> Optional[str]:
    """Reset password from token. Returns error string or None on success."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM email_tokens WHERE token = ? AND token_type = 'reset' AND used = 0",
            (token,),
        ).fetchone()
        if not row:
            return "Invalid or expired reset link"

        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            return "Reset link has expired"

        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(new_password), row["user_id"]),
        )
        conn.execute("UPDATE email_tokens SET used = 1 WHERE id = ?", (row["id"],))
        conn.commit()
        return None  # success
    finally:
        conn.close()


def resend_verification(user_id: int) -> bool:
    """Resend verification email, invalidating old tokens."""
    conn = get_db()
    try:
        conn.execute(
            "UPDATE email_tokens SET used = 1 WHERE user_id = ? AND token_type = 'verify' AND used = 0",
            (user_id,),
        )
        conn.commit()
    finally:
        conn.close()
    return send_verification_email(user_id)


# Initialize DB on import
init_db()
