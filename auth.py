"""
NEXYPHER Authentication Module
==============================
Supabase-backed user management with JWT tokens, wallet linking, and portfolio tracking.
"""

from __future__ import annotations

import os
import hashlib
import secrets
import json as _json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

import smtp_service
from blockchain_service import blockchain
from supabase_db import get_supabase

# ── Config ──────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "NEXYPHER_dev_secret_key_change_in_production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))  # 7 days

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
    auto_trade_threshold: float = 7.0
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

def init_db():
    """No-op for Supabase — tables are created via the Supabase SQL Editor.
    Run database/supabase_schema.sql in your Supabase project's SQL Editor."""
    pass


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
    sb = get_supabase()
    try:
        result = sb.table("users").insert({
            "email": email.lower(),
            "username": username,
            "password_hash": hash_password(password),
        }).execute()

        if not result.data:
            return None

        user_id = result.data[0]["id"]

        # Give new users a starting wallet balance of $10,000
        sb.table("wallet_balance").insert({
            "user_id": user_id,
            "balance": 10000.0,
        }).execute()

        timestamp = datetime.now(timezone.utc).isoformat()
        bonus_hash = _generate_wallet_tx_hash(user_id, 'signup_bonus', 10000.0, timestamp)

        sb.table("wallet_transactions").insert({
            "user_id": user_id,
            "type": "signup_bonus",
            "amount": 10000.0,
            "description": "Welcome bonus - $10,000 starting balance",
            "tx_hash": bonus_hash,
            "status": "completed",
            "created_at": timestamp,
        }).execute()

        # Record signup bonus on blockchain
        blockchain.record_transaction_async(bonus_hash, "signup_bonus", "USD", 10000.0)

        return _get_user_response(user_id)
    except Exception as e:
        # Unique constraint violation (duplicate email/username)
        if "duplicate" in str(e).lower() or "23505" in str(e):
            return None
        raise


def authenticate_user(email: str, password: str) -> Optional[UserResponse]:
    sb = get_supabase()
    result = sb.table("users").select("*").eq("email", email.lower()).execute()
    if not result.data:
        return None
    row = result.data[0]
    if not verify_password(password, row["password_hash"]):
        return None
    sb.table("users").update({"last_login": datetime.now(timezone.utc).isoformat()}).eq("id", row["id"]).execute()
    return _get_user_response(row["id"])


def get_user_by_id(user_id: int) -> Optional[UserResponse]:
    return _get_user_response(user_id)


def _get_user_response(user_id: int) -> Optional[UserResponse]:
    sb = get_supabase()
    result = sb.table("users").select("*").eq("id", user_id).execute()
    if not result.data:
        return None
    row = result.data[0]

    wallets = sb.table("wallets").select("address, chain, label").eq("user_id", user_id).execute()
    watchlist = sb.table("watchlist").select("coin_id").eq("user_id", user_id).execute()
    bank_accounts = sb.table("bank_accounts").select(
        "id, account_holder, account_last4, ifsc_code, bank_name, verified, status, added_at"
    ).eq("user_id", user_id).execute()
    bal = sb.table("wallet_balance").select("balance").eq("user_id", user_id).execute()

    return UserResponse(
        id=row["id"],
        email=row["email"],
        username=row["username"],
        email_verified=bool(row.get("email_verified", False)),
        wallets=[dict(w) for w in wallets.data] if wallets.data else [],
        watchlist=[w["coin_id"] for w in watchlist.data] if watchlist.data else [],
        bank_accounts=[dict(b) for b in bank_accounts.data] if bank_accounts.data else [],
        wallet_balance=bal.data[0]["balance"] if bal.data else 0.0,
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

    sb = get_supabase()
    try:
        # Check for duplicate
        existing = sb.table("bank_accounts").select("id").eq("user_id", user_id).eq("account_number_hash", acc_hash).execute()
        if existing.data:
            return {"success": False, "errors": ["This bank account is already linked"]}

        sb.table("bank_accounts").insert({
            "user_id": user_id,
            "account_holder": account_holder.strip(),
            "account_number_hash": acc_hash,
            "account_last4": last4,
            "ifsc_code": ifsc_code.upper(),
            "bank_name": detected_bank,
            "verified": True,
            "status": "verified",
        }).execute()
        return {"success": True, "bank_name": detected_bank, "last4": last4}
    except Exception as e:
        return {"success": False, "errors": [str(e)]}


def remove_bank_account(user_id: int, bank_id: int) -> bool:
    sb = get_supabase()
    try:
        sb.table("bank_accounts").delete().eq("id", bank_id).eq("user_id", user_id).execute()
        return True
    except Exception:
        return False


def get_bank_accounts(user_id: int) -> List[Dict[str, Any]]:
    sb = get_supabase()
    result = sb.table("bank_accounts").select(
        "id, account_holder, account_last4, ifsc_code, bank_name, verified, status, added_at"
    ).eq("user_id", user_id).execute()
    return [dict(r) for r in result.data] if result.data else []


# ── Wallet Balance Management ──────────────────────────────────

def get_wallet_balance(user_id: int) -> float:
    sb = get_supabase()
    result = sb.table("wallet_balance").select("balance").eq("user_id", user_id).execute()
    return result.data[0]["balance"] if result.data else 0.0


def deposit_to_wallet(user_id: int, amount: float, bank_id: int) -> Dict[str, Any]:
    """Deposit money from a verified bank account into wallet."""
    if amount <= 0:
        return {"success": False, "error": "Amount must be greater than 0"}
    if amount > 1000000:
        return {"success": False, "error": "Maximum deposit limit is $1,000,000"}

    sb = get_supabase()
    try:
        # Verify bank account belongs to user and is verified
        bank = sb.table("bank_accounts").select("*").eq("id", bank_id).eq("user_id", user_id).eq("verified", True).execute()
        if not bank.data:
            return {"success": False, "error": "Bank account not found or not verified"}
        bank_row = bank.data[0]

        # Atomically update wallet balance via RPC
        new_balance = sb.rpc("update_wallet_balance", {"p_user_id": user_id, "p_delta": amount}).execute()

        # Record transaction
        timestamp = datetime.now(timezone.utc).isoformat()
        dep_hash = _generate_wallet_tx_hash(user_id, 'deposit', amount, timestamp)
        sb.table("wallet_transactions").insert({
            "user_id": user_id,
            "type": "deposit",
            "amount": amount,
            "bank_id": bank_id,
            "description": f"Deposit from {bank_row['bank_name']} ****{bank_row['account_last4']}",
            "tx_hash": dep_hash,
            "status": "completed",
            "created_at": timestamp,
        }).execute()

        bal = sb.table("wallet_balance").select("balance").eq("user_id", user_id).execute()
        final_balance = bal.data[0]["balance"] if bal.data else amount

        # Record deposit on blockchain
        blockchain.record_transaction_async(dep_hash, "deposit", "USD", amount)

        return {"success": True, "new_balance": final_balance, "bank_name": bank_row["bank_name"], "tx_hash": dep_hash}
    except Exception as e:
        return {"success": False, "error": str(e)}


def withdraw_from_wallet(user_id: int, amount: float, bank_id: int) -> Dict[str, Any]:
    """Withdraw money from wallet to a verified bank account."""
    if amount <= 0:
        return {"success": False, "error": "Amount must be greater than 0"}

    sb = get_supabase()
    try:
        bal = sb.table("wallet_balance").select("balance").eq("user_id", user_id).execute()
        current = bal.data[0]["balance"] if bal.data else 0.0
        if amount > current:
            return {"success": False, "error": "Insufficient wallet balance"}

        bank = sb.table("bank_accounts").select("*").eq("id", bank_id).eq("user_id", user_id).eq("verified", True).execute()
        if not bank.data:
            return {"success": False, "error": "Bank account not found or not verified"}
        bank_row = bank.data[0]

        # Atomically deduct balance via RPC
        sb.rpc("update_wallet_balance", {"p_user_id": user_id, "p_delta": -amount}).execute()

        timestamp = datetime.now(timezone.utc).isoformat()
        wd_hash = _generate_wallet_tx_hash(user_id, 'withdraw', amount, timestamp)
        sb.table("wallet_transactions").insert({
            "user_id": user_id,
            "type": "withdraw",
            "amount": amount,
            "bank_id": bank_id,
            "description": f"Withdrawal to {bank_row['bank_name']} ****{bank_row['account_last4']}",
            "tx_hash": wd_hash,
            "status": "completed",
            "created_at": timestamp,
        }).execute()

        new_bal = sb.table("wallet_balance").select("balance").eq("user_id", user_id).execute()
        final_balance = new_bal.data[0]["balance"] if new_bal.data else 0.0

        # Record withdrawal on blockchain
        blockchain.record_transaction_async(wd_hash, "withdraw", "USD", amount)

        return {"success": True, "new_balance": final_balance, "tx_hash": wd_hash}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_wallet_transactions(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    sb = get_supabase()
    result = sb.table("wallet_transactions").select(
        "id, type, amount, description, tx_hash, status, created_at"
    ).eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
    return [dict(r) for r in result.data] if result.data else []


# ── Wallet Management ──────────────────────────────────────────

def add_wallet(user_id: int, address: str, chain: str = "ethereum", label: str = "") -> bool:
    sb = get_supabase()
    try:
        sb.table("wallets").upsert({
            "user_id": user_id,
            "address": address.lower(),
            "chain": chain.lower(),
            "label": label,
        }, on_conflict="user_id,address,chain").execute()
        return True
    except Exception:
        return False


def remove_wallet(user_id: int, address: str, chain: str = "ethereum") -> bool:
    sb = get_supabase()
    try:
        sb.table("wallets").delete().eq("user_id", user_id).eq("address", address.lower()).eq("chain", chain.lower()).execute()
        return True
    except Exception:
        return False


def get_wallets(user_id: int) -> List[Dict[str, str]]:
    sb = get_supabase()
    result = sb.table("wallets").select("address, chain, label, added_at").eq("user_id", user_id).execute()
    return [dict(r) for r in result.data] if result.data else []


# ── Watchlist ───────────────────────────────────────────────────

def add_to_watchlist(user_id: int, coin_id: str) -> bool:
    sb = get_supabase()
    try:
        sb.table("watchlist").upsert({
            "user_id": user_id,
            "coin_id": coin_id.lower(),
        }, on_conflict="user_id,coin_id").execute()
        return True
    except Exception:
        return False


def remove_from_watchlist(user_id: int, coin_id: str) -> bool:
    sb = get_supabase()
    try:
        sb.table("watchlist").delete().eq("user_id", user_id).eq("coin_id", coin_id.lower()).execute()
        return True
    except Exception:
        return False


def get_watchlist(user_id: int) -> List[str]:
    sb = get_supabase()
    result = sb.table("watchlist").select("coin_id").eq("user_id", user_id).execute()
    return [r["coin_id"] for r in result.data] if result.data else []


# ── Portfolio ───────────────────────────────────────────────────

def update_portfolio(user_id: int, coin_id: str, amount: float, avg_price: float, notes: str = "") -> bool:
    sb = get_supabase()
    try:
        sb.table("portfolio").upsert({
            "user_id": user_id,
            "coin_id": coin_id.lower(),
            "amount": amount,
            "avg_buy_price": avg_price,
            "notes": notes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="user_id,coin_id").execute()
        return True
    except Exception:
        return False


def get_portfolio(user_id: int) -> List[Dict[str, Any]]:
    sb = get_supabase()
    result = sb.table("portfolio").select(
        "coin_id, amount, avg_buy_price, notes, updated_at"
    ).eq("user_id", user_id).execute()
    return [dict(r) for r in result.data] if result.data else []


# ── Email Verification ──────────────────────────────────────────

def send_verification_email(user_id: int) -> bool:
    """Generate token and send verification email."""
    sb = get_supabase()
    result = sb.table("users").select("email, username").eq("id", user_id).execute()
    if not result.data:
        return False
    row = result.data[0]

    token = smtp_service.generate_verification_token()
    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()

    sb.table("email_tokens").insert({
        "user_id": user_id,
        "token": token,
        "token_type": "verify",
        "expires_at": expires,
    }).execute()

    return smtp_service.send_verification_email(row["email"], row["username"], token)


def verify_email(token: str) -> Optional[str]:
    """Verify email from token. Returns error string or None on success."""
    sb = get_supabase()
    result = sb.table("email_tokens").select("*").eq("token", token).eq("token_type", "verify").eq("used", False).execute()
    if not result.data:
        return "Invalid or expired verification link"
    row = result.data[0]

    if datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00")) < datetime.now(timezone.utc):
        return "Verification link has expired"

    sb.table("users").update({"email_verified": True}).eq("id", row["user_id"]).execute()
    sb.table("email_tokens").update({"used": True}).eq("id", row["id"]).execute()

    # Send welcome email
    user = sb.table("users").select("email, username").eq("id", row["user_id"]).execute()
    if user.data:
        smtp_service.send_welcome_email(user.data[0]["email"], user.data[0]["username"])

    return None  # success


# ── Password Reset ──────────────────────────────────────────────

def request_password_reset(email: str) -> bool:
    """Generate reset token and send email. Returns True if user exists."""
    sb = get_supabase()
    result = sb.table("users").select("id, username").eq("email", email.lower()).execute()
    if not result.data:
        return False
    row = result.data[0]

    token = smtp_service.generate_reset_token()
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    # Invalidate old reset tokens
    sb.table("email_tokens").update({"used": True}).eq("user_id", row["id"]).eq("token_type", "reset").eq("used", False).execute()

    sb.table("email_tokens").insert({
        "user_id": row["id"],
        "token": token,
        "token_type": "reset",
        "expires_at": expires,
    }).execute()

    return smtp_service.send_password_reset_email(email.lower(), row["username"], token)


def reset_password(token: str, new_password: str) -> Optional[str]:
    """Reset password from token. Returns error string or None on success."""
    sb = get_supabase()
    result = sb.table("email_tokens").select("*").eq("token", token).eq("token_type", "reset").eq("used", False).execute()
    if not result.data:
        return "Invalid or expired reset link"
    row = result.data[0]

    if datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00")) < datetime.now(timezone.utc):
        return "Reset link has expired"

    sb.table("users").update({"password_hash": hash_password(new_password)}).eq("id", row["user_id"]).execute()
    sb.table("email_tokens").update({"used": True}).eq("id", row["id"]).execute()
    return None  # success


def resend_verification(user_id: int) -> bool:
    """Resend verification email, invalidating old tokens."""
    sb = get_supabase()
    sb.table("email_tokens").update({"used": True}).eq("user_id", user_id).eq("token_type", "verify").eq("used", False).execute()
    return send_verification_email(user_id)


# ── User Preferences ───────────────────────────────────────────

def get_user_preferences(user_id: int) -> UserPreferencesResponse:
    """Get user AI/trading preferences. Returns defaults if no row exists."""
    sb = get_supabase()
    result = sb.table("user_preferences").select("*").eq("user_id", user_id).execute()
    if not result.data:
        return UserPreferencesResponse()
    row = result.data[0]
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


def update_user_preferences(user_id: int, updates: Dict[str, Any]) -> UserPreferencesResponse:
    """Upsert user preferences. Only supplied fields are updated."""
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

    sb = get_supabase()
    sb.table("user_preferences").upsert({
        "user_id": user_id,
        "risk_profile": merged["risk_profile"],
        "ai_sensitivity": merged["ai_sensitivity"],
        "auto_trade_threshold": merged["auto_trade_threshold"],
        "max_daily_trades": merged["max_daily_trades"],
        "preferred_chains": _json.dumps(merged["preferred_chains"]),
        "notification_email": merged["notification_email"],
        "notification_push": merged["notification_push"],
        "dark_mode": merged["dark_mode"],
        "language": merged["language"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).execute()

    return get_user_preferences(user_id)


# DB is initialized in web_app.py startup event
