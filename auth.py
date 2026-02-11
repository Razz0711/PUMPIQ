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
    bank_accounts: List[Dict[str, Any]] = []
    wallet_balance: float = 0.0
    created_at: str

class WalletAdd(BaseModel):
    address: str
    chain: str = "ethereum"  # ethereum, solana, base
    label: str = ""

class BankAccountAdd(BaseModel):
    account_holder: str
    account_number: str
    ifsc_code: str
    bank_name: str

class DepositRequest(BaseModel):
    amount: float
    bank_id: int

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

        CREATE TABLE IF NOT EXISTS bank_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            account_holder TEXT NOT NULL,
            account_number_hash TEXT NOT NULL,
            account_last4 TEXT NOT NULL,
            ifsc_code TEXT NOT NULL,
            bank_name TEXT NOT NULL,
            verified INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            added_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS wallet_balance (
            user_id INTEGER PRIMARY KEY,
            balance REAL NOT NULL DEFAULT 0.0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS wallet_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            bank_id INTEGER,
            description TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'completed',
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
        user_id = cursor.lastrowid
        # Give new users a starting wallet balance of $10,000
        conn.execute(
            "INSERT INTO wallet_balance (user_id, balance, updated_at) VALUES (?, 10000.0, datetime('now'))",
            (user_id,),
        )
        conn.execute(
            "INSERT INTO wallet_transactions (user_id, type, amount, description, status) VALUES (?, 'signup_bonus', 10000.0, 'Welcome bonus - $10,000 starting balance', 'completed')",
            (user_id,),
        )
        conn.commit()
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
    bank_accounts = conn.execute(
        "SELECT id, account_holder, account_last4, ifsc_code, bank_name, verified, status, added_at FROM bank_accounts WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    bal_row = conn.execute(
        "SELECT balance FROM wallet_balance WHERE user_id = ?", (user_id,)
    ).fetchone()

    return UserResponse(
        id=row["id"],
        email=row["email"],
        username=row["username"],
        email_verified=bool(row["email_verified"]) if "email_verified" in row.keys() else False,
        wallets=[dict(w) for w in wallets],
        watchlist=[w["coin_id"] for w in watchlist],
        bank_accounts=[dict(b) for b in bank_accounts],
        wallet_balance=bal_row["balance"] if bal_row else 0.0,
        created_at=row["created_at"],
    )


# ── Bank Account Management ────────────────────────────────────

def _validate_ifsc(ifsc: str) -> bool:
    """Validate IFSC code format: 4 alpha + 0 + 6 alphanumeric."""
    import re
    return bool(re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', ifsc.upper()))


def _validate_account_number(acc: str) -> bool:
    """Validate Indian bank account number: 9-18 digits."""
    return acc.isdigit() and 9 <= len(acc) <= 18


def verify_bank_details(account_number: str, ifsc_code: str, account_holder: str, bank_name: str) -> Dict[str, Any]:
    """Verify bank details. Returns verification result."""
    errors = []
    if not _validate_account_number(account_number):
        errors.append("Account number must be 9-18 digits")
    if not _validate_ifsc(ifsc_code):
        errors.append("Invalid IFSC code format (e.g. SBIN0001234)")
    if len(account_holder.strip()) < 3:
        errors.append("Account holder name must be at least 3 characters")
    if len(bank_name.strip()) < 2:
        errors.append("Bank name is required")

    if errors:
        return {"verified": False, "errors": errors}

    # Auto-verify based on IFSC lookup (simplified — in production use Razorpay/Cashfree API)
    ifsc_upper = ifsc_code.upper()
    bank_prefix = ifsc_upper[:4]
    known_banks = {
        "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
        "UTIB": "Axis Bank", "PUNB": "Punjab National Bank", "CNRB": "Canara Bank",
        "UBIN": "Union Bank of India", "BARB": "Bank of Baroda", "IOBA": "Indian Overseas Bank",
        "KKBK": "Kotak Mahindra Bank", "YESB": "Yes Bank", "IDIB": "Indian Bank",
        "BKID": "Bank of India", "CBIN": "Central Bank of India", "ALLA": "Allahabad Bank",
        "INDB": "IndusInd Bank", "FDRL": "Federal Bank", "RATN": "RBL Bank",
        "MAHB": "Bank of Maharashtra", "UCBA": "UCO Bank",
    }

    detected_bank = known_banks.get(bank_prefix)
    return {
        "verified": True,
        "bank_detected": detected_bank or bank_name,
        "ifsc_valid": True,
        "account_last4": account_number[-4:],
    }


def add_bank_account(user_id: int, account_holder: str, account_number: str, ifsc_code: str, bank_name: str) -> Dict[str, Any]:
    """Add a verified bank account for the user."""
    # Step 1: validate
    result = verify_bank_details(account_number, ifsc_code, account_holder, bank_name)
    if not result["verified"]:
        return {"success": False, "errors": result["errors"]}

    # Step 2: hash account number for security, store last 4
    acc_hash = hashlib.sha256(account_number.encode()).hexdigest()
    last4 = account_number[-4:]
    detected_bank = result.get("bank_detected", bank_name)

    conn = get_db()
    try:
        # Check for duplicate
        existing = conn.execute(
            "SELECT id FROM bank_accounts WHERE user_id = ? AND account_number_hash = ?",
            (user_id, acc_hash),
        ).fetchone()
        if existing:
            return {"success": False, "errors": ["This bank account is already linked"]}

        conn.execute(
            """INSERT INTO bank_accounts (user_id, account_holder, account_number_hash, account_last4, ifsc_code, bank_name, verified, status)
               VALUES (?, ?, ?, ?, ?, ?, 1, 'verified')""",
            (user_id, account_holder.strip(), acc_hash, last4, ifsc_code.upper(), detected_bank),
        )
        conn.commit()
        return {"success": True, "bank_name": detected_bank, "last4": last4}
    except Exception as e:
        return {"success": False, "errors": [str(e)]}
    finally:
        conn.close()


def remove_bank_account(user_id: int, bank_id: int) -> bool:
    conn = get_db()
    try:
        conn.execute("DELETE FROM bank_accounts WHERE id = ? AND user_id = ?", (bank_id, user_id))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_bank_accounts(user_id: int) -> List[Dict[str, Any]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, account_holder, account_last4, ifsc_code, bank_name, verified, status, added_at FROM bank_accounts WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Wallet Balance Management ──────────────────────────────────

def get_wallet_balance(user_id: int) -> float:
    conn = get_db()
    try:
        row = conn.execute("SELECT balance FROM wallet_balance WHERE user_id = ?", (user_id,)).fetchone()
        return row["balance"] if row else 0.0
    finally:
        conn.close()


def deposit_to_wallet(user_id: int, amount: float, bank_id: int) -> Dict[str, Any]:
    """Deposit money from a verified bank account into wallet."""
    if amount <= 0:
        return {"success": False, "error": "Amount must be greater than 0"}
    if amount > 1000000:
        return {"success": False, "error": "Maximum deposit limit is ₹10,00,000"}

    conn = get_db()
    try:
        # Verify bank account belongs to user and is verified
        bank = conn.execute(
            "SELECT * FROM bank_accounts WHERE id = ? AND user_id = ? AND verified = 1",
            (bank_id, user_id),
        ).fetchone()
        if not bank:
            return {"success": False, "error": "Bank account not found or not verified"}

        # Upsert wallet balance
        conn.execute(
            """INSERT INTO wallet_balance (user_id, balance, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(user_id) DO UPDATE SET
                   balance = balance + ?,
                   updated_at = datetime('now')""",
            (user_id, amount, amount),
        )
        # Record transaction
        conn.execute(
            "INSERT INTO wallet_transactions (user_id, type, amount, bank_id, description, status) VALUES (?, 'deposit', ?, ?, ?, 'completed')",
            (user_id, amount, bank_id, f"Deposit from {bank['bank_name']} ****{bank['account_last4']}"),
        )
        conn.commit()

        new_balance = conn.execute("SELECT balance FROM wallet_balance WHERE user_id = ?", (user_id,)).fetchone()
        return {"success": True, "new_balance": new_balance["balance"], "bank_name": bank["bank_name"]}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def withdraw_from_wallet(user_id: int, amount: float, bank_id: int) -> Dict[str, Any]:
    """Withdraw money from wallet to a verified bank account."""
    if amount <= 0:
        return {"success": False, "error": "Amount must be greater than 0"}

    conn = get_db()
    try:
        bal_row = conn.execute("SELECT balance FROM wallet_balance WHERE user_id = ?", (user_id,)).fetchone()
        current = bal_row["balance"] if bal_row else 0.0
        if amount > current:
            return {"success": False, "error": "Insufficient wallet balance"}

        bank = conn.execute(
            "SELECT * FROM bank_accounts WHERE id = ? AND user_id = ? AND verified = 1",
            (bank_id, user_id),
        ).fetchone()
        if not bank:
            return {"success": False, "error": "Bank account not found or not verified"}

        conn.execute(
            "UPDATE wallet_balance SET balance = balance - ?, updated_at = datetime('now') WHERE user_id = ?",
            (amount, user_id),
        )
        conn.execute(
            "INSERT INTO wallet_transactions (user_id, type, amount, bank_id, description, status) VALUES (?, 'withdraw', ?, ?, ?, 'completed')",
            (user_id, amount, bank_id, f"Withdrawal to {bank['bank_name']} ****{bank['account_last4']}"),
        )
        conn.commit()

        new_balance = conn.execute("SELECT balance FROM wallet_balance WHERE user_id = ?", (user_id,)).fetchone()
        return {"success": True, "new_balance": new_balance["balance"]}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def get_wallet_transactions(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, type, amount, description, status, created_at FROM wallet_transactions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


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
