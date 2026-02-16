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
import sms_service
from blockchain_service import blockchain

import logging
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────
_env_secret = os.getenv("SECRET_KEY", "").strip()
if _env_secret:
    SECRET_KEY = _env_secret
else:
    # Generate a random secret key at startup — secure by default
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "SECRET_KEY not set in environment — generated a random key. "
        "JWTs will be invalidated on server restart. "
        "Set SECRET_KEY in .env for persistent sessions."
    )
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# On Vercel (serverless), use /tmp for writable SQLite; locally use project dir
IS_VERCEL = bool(os.getenv("VERCEL"))
DB_PATH = "/tmp/pumpiq.db" if IS_VERCEL else os.path.join(os.path.dirname(__file__), "pumpiq.db")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Pydantic Models ─────────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    username: str

    def validate_password_strength(self) -> list:
        """Return a list of password weakness messages (empty = strong)."""
        issues = []
        if len(self.password) < 8:
            issues.append("Password must be at least 8 characters")
        if not any(c.isupper() for c in self.password):
            issues.append("Password should contain at least one uppercase letter")
        if not any(c.isdigit() for c in self.password):
            issues.append("Password should contain at least one digit")
        return issues

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
    phone_number: str = ""
    otp_code: str = ""  # required for final add

class DepositRequest(BaseModel):
    amount: float
    bank_id: int
    otp_code: str = ""

class PhoneOTPRequest(BaseModel):
    phone_number: str
    action: str = "bank_verify"  # bank_verify

class OTPRequest(BaseModel):
    bank_id: int
    action: str = "deposit"  # deposit or withdraw

class OTPVerifyRequest(BaseModel):
    bank_id: int
    otp_code: str
    action: str = "deposit"

class UserPreferencesUpdate(BaseModel):
    risk_profile: Optional[str] = None        # conservative, balanced, aggressive
    ai_sensitivity: Optional[float] = None     # 0.0 – 1.0 (how eagerly AI acts)
    auto_trade_threshold: Optional[float] = None  # min confidence (0-10) to auto-execute
    max_daily_trades: Optional[int] = None
    preferred_chains: Optional[List[str]] = None   # ["ethereum","solana","base"]
    notification_email: Optional[bool] = None
    notification_push: Optional[bool] = None
    dark_mode: Optional[bool] = None
    language: Optional[str] = None             # e.g. "en", "hi"

class UserPreferencesResponse(BaseModel):
    risk_profile: str = "balanced"
    ai_sensitivity: float = 0.5
    auto_trade_threshold: float = 4.0
    max_daily_trades: int = 10
    preferred_chains: List[str] = ["ethereum", "solana"]
    notification_email: bool = True
    notification_push: bool = True
    dark_mode: bool = True
    language: str = "en"

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ── Database Setup ──────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass  # WAL may not be supported on some serverless filesystems
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
            phone_number TEXT NOT NULL DEFAULT '',
            phone_last4 TEXT NOT NULL DEFAULT '',
            verified INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            added_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS otp_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            bank_id INTEGER NOT NULL DEFAULT 0,
            otp_code TEXT NOT NULL,
            action TEXT NOT NULL DEFAULT 'deposit',
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
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
            tx_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'completed',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id INTEGER PRIMARY KEY,
            risk_profile TEXT NOT NULL DEFAULT 'balanced',
            ai_sensitivity REAL NOT NULL DEFAULT 0.5,
            auto_trade_threshold REAL NOT NULL DEFAULT 4.0,
            max_daily_trades INTEGER NOT NULL DEFAULT 10,
            preferred_chains TEXT NOT NULL DEFAULT '["ethereum","solana"]',
            notification_email INTEGER NOT NULL DEFAULT 1,
            notification_push INTEGER NOT NULL DEFAULT 1,
            dark_mode INTEGER NOT NULL DEFAULT 1,
            language TEXT NOT NULL DEFAULT 'en',
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)
    # Migration: add email_verified column if missing (existing DBs)
    try:
        conn.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    # Migration: add tx_hash column to wallet_transactions
    try:
        conn.execute("ALTER TABLE wallet_transactions ADD COLUMN tx_hash TEXT NOT NULL DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    # Migration: add phone_number & phone_last4 to bank_accounts
    for col, default in [("phone_number", "''"), ("phone_last4", "''")]:
        try:
            conn.execute(f"ALTER TABLE bank_accounts ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    # Migration: recreate otp_codes table without FK on bank_id (allows bank_id=0 for phone verification)
    # Check if existing table has the problematic FK constraint by trying to drop and recreate
    try:
        conn.execute("DROP TABLE IF EXISTS otp_codes")
    except Exception:
        pass
    conn.execute('''
        CREATE TABLE IF NOT EXISTS otp_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            bank_id INTEGER NOT NULL DEFAULT 0,
            otp_code TEXT NOT NULL,
            action TEXT NOT NULL DEFAULT 'deposit',
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    conn.commit()
    conn.close()


def _generate_wallet_tx_hash(user_id: int, tx_type: str, amount: float, timestamp: str) -> str:
    """Generate a SHA-256 hash for a wallet transaction."""
    raw = f"{user_id}|{tx_type}|{amount:.8f}|{timestamp}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


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
        timestamp = datetime.now(timezone.utc).isoformat()
        bonus_hash = _generate_wallet_tx_hash(user_id, 'signup_bonus', 10000.0, timestamp)
        conn.execute(
            "INSERT INTO wallet_transactions (user_id, type, amount, description, tx_hash, status, created_at) VALUES (?, 'signup_bonus', 10000.0, 'Welcome bonus - $10,000 starting balance', ?, 'completed', ?)",
            (user_id, bonus_hash, timestamp),
        )
        conn.commit()

        # Record signup bonus on blockchain
        blockchain.record_transaction_async(bonus_hash, "signup_bonus", "USD", 10000.0)

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
        "SELECT id, account_holder, account_last4, ifsc_code, bank_name, phone_last4, verified, status, added_at FROM bank_accounts WHERE user_id = ?",
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


def _validate_phone(phone: str) -> bool:
    """Validate Indian phone number (10 digits, optionally with +91 prefix)."""
    import re
    cleaned = re.sub(r'[\s\-\(\)]+', '', phone)
    return bool(re.match(r'^(\+91|91)?[6-9]\d{9}$', cleaned))


def _clean_phone(phone: str) -> str:
    """Normalize phone to +91XXXXXXXXXX format."""
    import re
    cleaned = re.sub(r'[\s\-\(\)]+', '', phone)
    cleaned = re.sub(r'^(\+91|91)', '', cleaned)
    return '+91' + cleaned


def add_bank_account(user_id: int, account_holder: str, account_number: str, ifsc_code: str, bank_name: str, phone_number: str = "", otp_code: str = "") -> Dict[str, Any]:
    """Add a verified bank account for the user. Requires OTP verification of the phone number."""
    # Step 1: validate bank details
    result = verify_bank_details(account_number, ifsc_code, account_holder, bank_name)
    if not result["verified"]:
        return {"success": False, "errors": result["errors"]}

    # Validate phone number
    if not phone_number or not _validate_phone(phone_number):
        return {"success": False, "errors": ["A valid phone number linked to this bank account is required (10-digit Indian mobile)"]}

    # Step 2: Verify OTP — mandatory
    if not otp_code:
        return {"success": False, "errors": ["OTP verification is required to link a bank account"]}

    otp_result = verify_phone_otp(user_id, otp_code, action="bank_verify", phone_number=phone_number)
    if not otp_result["success"]:
        return {"success": False, "errors": [otp_result["error"]]}

    # Step 3: hash account number for security, store last 4
    acc_hash = hashlib.sha256(account_number.encode()).hexdigest()
    last4 = account_number[-4:]
    detected_bank = result.get("bank_detected", bank_name)
    clean_phone = _clean_phone(phone_number)
    phone_last4 = clean_phone[-4:]

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
            """INSERT INTO bank_accounts (user_id, account_holder, account_number_hash, account_last4, ifsc_code, bank_name, phone_number, phone_last4, verified, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 'verified')""",
            (user_id, account_holder.strip(), acc_hash, last4, ifsc_code.upper(), detected_bank, clean_phone, phone_last4),
        )
        conn.commit()
        return {"success": True, "bank_name": detected_bank, "last4": last4, "phone_last4": phone_last4}
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
            "SELECT id, account_holder, account_last4, ifsc_code, bank_name, phone_last4, verified, status, added_at FROM bank_accounts WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── OTP Management ─────────────────────────────────────────────

import random

def generate_phone_otp(user_id: int, phone_number: str, action: str = "bank_verify") -> Dict[str, Any]:
    """Send OTP to a phone number for verification (e.g. before bank linking).
       Uses Twilio Verify when configured, falls back to DB-based debug OTP."""
    if not phone_number or not _validate_phone(phone_number):
        return {"success": False, "error": "A valid 10-digit phone number is required"}

    clean_phone = _clean_phone(phone_number)
    phone_last4 = clean_phone[-4:]

    conn = get_db()
    try:
        # Rate limit: max 5 OTPs per hour per user
        recent = conn.execute(
            "SELECT COUNT(*) as cnt FROM otp_codes WHERE user_id = ? AND created_at > datetime('now', '-1 hour')",
            (user_id,),
        ).fetchone()["cnt"]
        if recent >= 5:
            return {"success": False, "error": "Too many OTP requests. Please try again later."}

        # Record the OTP request (for rate limiting + audit)
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

        if sms_service.is_configured():
            # ── Twilio Verify: sends OTP automatically ──
            sms_result = sms_service.send_otp(clean_phone)
            if not sms_result["success"]:
                return {"success": False, "error": sms_result.get("error", "Failed to send OTP")}

            # Log for rate limiting (no actual code stored — Twilio manages it)
            conn.execute(
                "INSERT INTO otp_codes (user_id, bank_id, otp_code, action, expires_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, 0, "TWILIO", action, expires_at),
            )
            conn.commit()
            logger.info("Twilio OTP sent for user %d, phone ****%s, action %s", user_id, phone_last4, action)

            return {
                "success": True,
                "message": f"OTP sent to ••••••{phone_last4}",
                "phone_last4": phone_last4,
                "expires_in": 600,
                "sms_sent": True,
            }
        else:
            # ── Fallback: DB-based debug OTP ──
            conn.execute(
                "UPDATE otp_codes SET used = 1 WHERE user_id = ? AND action = ? AND used = 0",
                (user_id, action),
            )
            otp_code = f"{random.randint(100000, 999999)}"
            conn.execute(
                "INSERT INTO otp_codes (user_id, bank_id, otp_code, action, expires_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, 0, otp_code, action, expires_at),
            )
            conn.commit()
            logger.warning("Twilio not configured — debug OTP for user %d, phone ****%s", user_id, phone_last4)

            return {
                "success": True,
                "message": f"OTP generated for ••••••{phone_last4} (debug mode — configure Twilio for real SMS)",
                "phone_last4": phone_last4,
                "otp_code_debug": otp_code,
                "expires_in": 600,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def verify_phone_otp(user_id: int, otp_code: str, action: str = "bank_verify", phone_number: str = "") -> Dict[str, Any]:
    """Verify OTP for phone verification (bank linking).
       Uses Twilio Verify when configured, falls back to DB check."""
    if sms_service.is_configured() and phone_number:
        # ── Twilio Verify: check via API ──
        result = sms_service.verify_otp(phone_number, otp_code)
        if result["success"]:
            logger.info("Twilio OTP verified for user %d, phone ****%s", user_id, phone_number[-4:])
        return result
    else:
        # ── Fallback: DB-based verification ──
        conn = get_db()
        try:
            row = conn.execute(
                """SELECT * FROM otp_codes
                   WHERE user_id = ? AND otp_code = ? AND action = ? AND used = 0
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id, otp_code, action),
            ).fetchone()

            if not row:
                return {"success": False, "error": "Invalid OTP code"}

            expires_at = datetime.fromisoformat(row["expires_at"].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > expires_at:
                conn.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (row["id"],))
                conn.commit()
                return {"success": False, "error": "OTP has expired. Please request a new one."}

            conn.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (row["id"],))
            conn.commit()
            return {"success": True, "message": "OTP verified successfully"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()


def generate_otp(user_id: int, bank_id: int, action: str = "deposit") -> Dict[str, Any]:
    """Send OTP for deposit/withdraw via the bank account's linked phone.
       Uses Twilio Verify when configured, falls back to DB-based debug OTP."""
    conn = get_db()
    try:
        # Verify bank account belongs to user
        bank = conn.execute(
            "SELECT * FROM bank_accounts WHERE id = ? AND user_id = ? AND verified = 1",
            (bank_id, user_id),
        ).fetchone()
        if not bank:
            return {"success": False, "error": "Bank account not found or not verified"}

        phone = bank["phone_number"]
        phone_last4 = bank["phone_last4"]
        if not phone:
            return {"success": False, "error": "No phone number linked to this bank account"}

        # Rate limit: max 5 OTPs per hour per user
        recent = conn.execute(
            "SELECT COUNT(*) as cnt FROM otp_codes WHERE user_id = ? AND created_at > datetime('now', '-1 hour')",
            (user_id,),
        ).fetchone()["cnt"]
        if recent >= 5:
            return {"success": False, "error": "Too many OTP requests. Please try again later."}

        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

        if sms_service.is_configured():
            # ── Twilio Verify ──
            sms_result = sms_service.send_otp(phone)
            if not sms_result["success"]:
                return {"success": False, "error": sms_result.get("error", "Failed to send OTP")}

            conn.execute(
                "INSERT INTO otp_codes (user_id, bank_id, otp_code, action, expires_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, bank_id, "TWILIO", action, expires_at),
            )
            conn.commit()
            logger.info("Twilio OTP sent for user %d, bank %d, action %s, phone ****%s", user_id, bank_id, action, phone_last4)

            return {
                "success": True,
                "message": f"OTP sent to ••••••{phone_last4}",
                "phone_last4": phone_last4,
                "expires_in": 600,
                "sms_sent": True,
            }
        else:
            # ── Fallback: DB-based debug OTP ──
            conn.execute(
                "UPDATE otp_codes SET used = 1 WHERE user_id = ? AND bank_id = ? AND action = ? AND used = 0",
                (user_id, bank_id, action),
            )
            otp_code = f"{random.randint(100000, 999999)}"
            conn.execute(
                "INSERT INTO otp_codes (user_id, bank_id, otp_code, action, expires_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, bank_id, otp_code, action, expires_at),
            )
            conn.commit()
            logger.warning("Twilio not configured — debug OTP for user %d, bank %d", user_id, bank_id)

            return {
                "success": True,
                "message": f"OTP generated for ••••••{phone_last4} (debug mode — configure Twilio for real SMS)",
                "phone_last4": phone_last4,
                "otp_code_debug": otp_code,
                "expires_in": 600,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def verify_otp(user_id: int, bank_id: int, otp_code: str, action: str = "deposit") -> Dict[str, Any]:
    """Verify OTP for deposit/withdraw.
       Uses Twilio Verify when configured, falls back to DB check."""
    if sms_service.is_configured():
        # ── Twilio Verify: look up the phone from bank account and check ──
        conn = get_db()
        try:
            bank = conn.execute(
                "SELECT phone_number FROM bank_accounts WHERE id = ? AND user_id = ?",
                (bank_id, user_id),
            ).fetchone()
            if not bank or not bank["phone_number"]:
                return {"success": False, "error": "Bank account phone not found"}
            result = sms_service.verify_otp(bank["phone_number"], otp_code)
            if result["success"]:
                logger.info("Twilio OTP verified for user %d, bank %d, action %s", user_id, bank_id, action)
            return result
        finally:
            conn.close()
    else:
        # ── Fallback: DB-based verification ──
        conn = get_db()
        try:
            row = conn.execute(
                """SELECT * FROM otp_codes
                   WHERE user_id = ? AND bank_id = ? AND otp_code = ? AND action = ? AND used = 0
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id, bank_id, otp_code, action),
            ).fetchone()

            if not row:
                return {"success": False, "error": "Invalid OTP code"}

            expires_at = datetime.fromisoformat(row["expires_at"].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > expires_at:
                conn.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (row["id"],))
                conn.commit()
                return {"success": False, "error": "OTP has expired. Please request a new one."}

            conn.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (row["id"],))
            conn.commit()
            return {"success": True, "message": "OTP verified successfully"}
        except Exception as e:
            return {"success": False, "error": str(e)}
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


def deposit_to_wallet(user_id: int, amount: float, bank_id: int, otp_code: str = "") -> Dict[str, Any]:
    """Deposit money from a verified bank account into wallet. Requires OTP."""
    if amount <= 0:
        return {"success": False, "error": "Amount must be greater than 0"}
    if amount > 1000000:
        return {"success": False, "error": "Maximum deposit limit is $1,000,000"}
    if not otp_code:
        return {"success": False, "error": "OTP verification is required for deposits"}

    # Verify OTP
    otp_result = verify_otp(user_id, bank_id, otp_code, "deposit")
    if not otp_result["success"]:
        return {"success": False, "error": otp_result["error"]}

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
        timestamp = datetime.now(timezone.utc).isoformat()
        dep_hash = _generate_wallet_tx_hash(user_id, 'deposit', amount, timestamp)
        conn.execute(
            "INSERT INTO wallet_transactions (user_id, type, amount, bank_id, description, tx_hash, status, created_at) VALUES (?, 'deposit', ?, ?, ?, ?, 'completed', ?)",
            (user_id, amount, bank_id, f"Deposit from {bank['bank_name']} ****{bank['account_last4']}", dep_hash, timestamp),
        )
        conn.commit()

        new_balance = conn.execute("SELECT balance FROM wallet_balance WHERE user_id = ?", (user_id,)).fetchone()

        # Record deposit on blockchain
        blockchain.record_transaction_async(dep_hash, "deposit", "USD", amount)

        return {"success": True, "new_balance": new_balance["balance"], "bank_name": bank["bank_name"], "tx_hash": dep_hash}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def withdraw_from_wallet(user_id: int, amount: float, bank_id: int, otp_code: str = "") -> Dict[str, Any]:
    """Withdraw money from wallet to a verified bank account. Requires OTP."""
    if amount <= 0:
        return {"success": False, "error": "Amount must be greater than 0"}
    if not otp_code:
        return {"success": False, "error": "OTP verification is required for withdrawals"}

    # Verify OTP
    otp_result = verify_otp(user_id, bank_id, otp_code, "withdraw")
    if not otp_result["success"]:
        return {"success": False, "error": otp_result["error"]}

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
        timestamp = datetime.now(timezone.utc).isoformat()
        wd_hash = _generate_wallet_tx_hash(user_id, 'withdraw', amount, timestamp)
        conn.execute(
            "INSERT INTO wallet_transactions (user_id, type, amount, bank_id, description, tx_hash, status, created_at) VALUES (?, 'withdraw', ?, ?, ?, ?, 'completed', ?)",
            (user_id, amount, bank_id, f"Withdrawal to {bank['bank_name']} ****{bank['account_last4']}", wd_hash, timestamp),
        )
        conn.commit()

        new_balance = conn.execute("SELECT balance FROM wallet_balance WHERE user_id = ?", (user_id,)).fetchone()

        # Record withdrawal on blockchain
        blockchain.record_transaction_async(wd_hash, "withdraw", "USD", amount)

        return {"success": True, "new_balance": new_balance["balance"], "tx_hash": wd_hash}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def get_wallet_transactions(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, type, amount, description, tx_hash, status, created_at FROM wallet_transactions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
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


# ── User Preferences ───────────────────────────────────────────

def get_user_preferences(user_id: int) -> UserPreferencesResponse:
    """Get user AI/trading preferences. Returns defaults if no row exists."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return UserPreferencesResponse()
        import json as _json
        chains = ["ethereum", "solana"]
        try:
            chains = _json.loads(row["preferred_chains"])
        except Exception:
            pass
        return UserPreferencesResponse(
            risk_profile=row["risk_profile"],
            ai_sensitivity=row["ai_sensitivity"],
            auto_trade_threshold=row["auto_trade_threshold"],
            max_daily_trades=row["max_daily_trades"],
            preferred_chains=chains,
            notification_email=bool(row["notification_email"]),
            notification_push=bool(row["notification_push"]),
            dark_mode=bool(row["dark_mode"]),
            language=row["language"],
        )
    finally:
        conn.close()


def update_user_preferences(user_id: int, updates: Dict[str, Any]) -> UserPreferencesResponse:
    """Upsert user preferences. Only supplied fields are updated."""
    import json as _json
    current = get_user_preferences(user_id)
    merged = current.model_dump()

    valid_risk = {"conservative", "balanced", "aggressive"}
    if "risk_profile" in updates and updates["risk_profile"] in valid_risk:
        merged["risk_profile"] = updates["risk_profile"]
    if "ai_sensitivity" in updates and updates["ai_sensitivity"] is not None:
        merged["ai_sensitivity"] = max(0.0, min(1.0, float(updates["ai_sensitivity"])))
    if "auto_trade_threshold" in updates and updates["auto_trade_threshold"] is not None:
        merged["auto_trade_threshold"] = max(0.0, min(10.0, float(updates["auto_trade_threshold"])))
    if "max_daily_trades" in updates and updates["max_daily_trades"] is not None:
        merged["max_daily_trades"] = max(1, min(100, int(updates["max_daily_trades"])))
    if "preferred_chains" in updates and updates["preferred_chains"] is not None:
        merged["preferred_chains"] = updates["preferred_chains"]
    if "notification_email" in updates and updates["notification_email"] is not None:
        merged["notification_email"] = bool(updates["notification_email"])
    if "notification_push" in updates and updates["notification_push"] is not None:
        merged["notification_push"] = bool(updates["notification_push"])
    if "dark_mode" in updates and updates["dark_mode"] is not None:
        merged["dark_mode"] = bool(updates["dark_mode"])
    if "language" in updates and updates["language"] is not None:
        merged["language"] = str(updates["language"])[:5]

    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO user_preferences
                (user_id, risk_profile, ai_sensitivity, auto_trade_threshold,
                 max_daily_trades, preferred_chains, notification_email,
                 notification_push, dark_mode, language, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                risk_profile = excluded.risk_profile,
                ai_sensitivity = excluded.ai_sensitivity,
                auto_trade_threshold = excluded.auto_trade_threshold,
                max_daily_trades = excluded.max_daily_trades,
                preferred_chains = excluded.preferred_chains,
                notification_email = excluded.notification_email,
                notification_push = excluded.notification_push,
                dark_mode = excluded.dark_mode,
                language = excluded.language,
                updated_at = datetime('now')
        """, (
            user_id,
            merged["risk_profile"],
            merged["ai_sensitivity"],
            merged["auto_trade_threshold"],
            merged["max_daily_trades"],
            _json.dumps(merged["preferred_chains"]),
            1 if merged["notification_email"] else 0,
            1 if merged["notification_push"] else 0,
            1 if merged["dark_mode"] else 0,
            merged["language"],
        ))
        conn.commit()
    finally:
        conn.close()

    return get_user_preferences(user_id)


# DB is initialized in web_app.py startup event
