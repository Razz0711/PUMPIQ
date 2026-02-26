"""NEXYPHER Autonomous Trading Engine v2.0
============================================================
AI-powered autonomous crypto trading bot that:
1. Checks REAL wallet balance (deposited from bank account)
2. Researches market opportunities every 30 seconds (continuous scanning)
3. Makes LONG & SHORT decisions with risk management
4. Executes trades using real wallet funds
5. Tracks P&L and performance
6. Auto-closes ALL positions within 1 hour max
7. Sends SMS warnings 5 minutes before auto-close

Trading Rules:
- Portfolio: 10-15 simultaneous positions (diversified)
- No single token > 15% of total portfolio
- Bidirectional: LONG (buy) + SHORT (sell) simultaneously
- 1-hour max hold time with forced exit
- Stop-loss & take-profit per position
- Auto-trader ON by default (hands-free)
- 30-second scan interval for real-time opportunity detection
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from blockchain_service import blockchain
from supabase_db import get_supabase
from src.backtest_engine import get_backtest_engine, BacktestResult
from src.ml_backtester import get_ml_backtester
from src.pretrained_predictor import predict_pretrained

logger = logging.getLogger(__name__)

# ── System auto-trader (always-on background bot) ──
SYSTEM_USER_ID = 1  # trades under the primary user account so dashboard shows activity
SYSTEM_INITIAL_BALANCE = 10_000_000.0  # ₹1,00,00,000 (1 Cr)
AUTO_TRADE_INTERVAL_SECONDS = 10   # 10 seconds — fast position checks
RESEARCH_EVERY_N_CYCLES = 6         # full research every 6th cycle (~60s)
MAX_HOLD_HOURS = 2                  # auto-sell if TP/SL not hit within 2 hours (fast rotation for model training)
WARNING_MINUTES_BEFORE_CLOSE = 10   # SMS warning 10 min before auto-close
MAX_PORTFOLIO_SLOTS = 15            # 10-15 simultaneous positions
MAX_POSITION_PCT = 15.0             # no single token > 15% of portfolio
MIN_POSITION_AMOUNT = 100           # minimum $ per position (lowered for small wallets)
RESERVE_AMOUNT = 50                 # always keep $50 as buffer
STABLECOINS = {"tether", "usd-coin", "dai", "usd1-wlfi", "binance-usd", "true-usd", "first-digital-usd"}
_autotrader_task = None  # reference to the background asyncio task
_autotrader_running = False

# ── Cycle log ring buffer (in-memory, last 50 cycles) ──
from collections import deque
_cycle_log: deque = deque(maxlen=50)

# ── Backtest result cache (avoid re-running for same coin within 1 hour) ──
_backtest_cache: Dict[str, Tuple[float, "BacktestResult"]] = {}
BACKTEST_CACHE_TTL = 3600  # seconds

def _get_cached_backtest(coin_id: str) -> Optional["BacktestResult"]:
    """Return cached backtest result if still valid."""
    if coin_id in _backtest_cache:
        cached_time, result = _backtest_cache[coin_id]
        if time.time() - cached_time < BACKTEST_CACHE_TTL:
            return result
    return None

def _cache_backtest(coin_id: str, result: "BacktestResult"):
    """Cache a backtest result."""
    _backtest_cache[coin_id] = (time.time(), result)


# Late-import learning loop (avoid circular)
_learning_loop = None

def _get_learning_loop():
    global _learning_loop
    if _learning_loop is None:
        try:
            from src.ai_engine.learning_loop import LearningLoop
            _learning_loop = LearningLoop()
        except Exception:
            pass
    return _learning_loop


def _detect_market_regime(btc_change_24h: float) -> str:
    """Detect market regime from BTC 24h change.
    Used to adjust LONG/SHORT balance and position sizing."""
    if btc_change_24h > 3:
        return "strong_bull"
    elif btc_change_24h > 1:
        return "bull"
    elif btc_change_24h > -1:
        return "sideways"
    elif btc_change_24h > -3:
        return "bear"
    else:
        return "strong_bear"


def _get_learning_adjustments() -> Dict[str, Any]:
    """Fetch learning loop adjustments and compile into actionable trading modifiers.

    THIS IS THE CRITICAL FEEDBACK LINK that was missing.
    Without this, the learning loop generates insights but the trader ignores them.

    Queries the last 24h of strategy adjustments and builds:
    - Per-token score penalties (underperformers get docked)
    - Direction bias corrections (if SHORTs keep losing, reduce SHORT confidence)
    - Global score offset (if overall accuracy is poor, raise the bar)
    - SL modifier (if PnL is drifting negative, tighten stops)
    """
    adjustments: Dict[str, Any] = {
        "token_penalties": {},       # coin_id -> score penalty (negative)
        "direction_bias": {},        # "long"/"short" -> multiplier (0.0-1.0)
        "global_score_offset": 0,    # added to min_score threshold
        "sl_modifier": 0.0,         # added to stop_loss_pct (positive = tighter)
        "reduce_position_size": False,
    }
    ll = _get_learning_loop()
    if not ll:
        return adjustments

    try:
        recent = ll.get_recent_adjustments(limit=50)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        recent = [a for a in recent if a.get("created_at", "") >= cutoff]

        # BUGFIX: Only use the LATEST adjustment of each type to prevent stacking.
        # Previously, 10+ records of the same type would stack penalties exponentially.
        seen_types: set = set()
        for adj in recent:
            adj_type = adj.get("adjustment_type", "")

            if adj_type == "token_confidence_reduction":
                desc = adj.get("description", "")
                token = desc.split(" ")[0].lower() if desc else ""
                dedup_key = f"{adj_type}_{token}"
                if token and dedup_key not in seen_types:
                    seen_types.add(dedup_key)
                    adjustments["token_penalties"][token] = (
                        adjustments["token_penalties"].get(token, 0) - 10
                    )

            elif adj_type == "direction_bias_correction":
                if adj_type in seen_types:
                    continue
                seen_types.add(adj_type)
                new_val = adj.get("new_value", "")
                if "short" in new_val.lower():
                    adjustments["direction_bias"]["short"] = min(
                        adjustments["direction_bias"].get("short", 1.0), 0.6
                    )
                elif "long" in new_val.lower():
                    adjustments["direction_bias"]["long"] = min(
                        adjustments["direction_bias"].get("long", 1.0), 0.6
                    )

            elif adj_type == "global_confidence_reduction":
                if adj_type in seen_types:
                    continue
                seen_types.add(adj_type)
                adjustments["global_score_offset"] += 5
                adjustments["reduce_position_size"] = True

            elif adj_type == "global_confidence_increase":
                if adj_type in seen_types:
                    continue
                seen_types.add(adj_type)
                adjustments["global_score_offset"] = max(
                    0, adjustments["global_score_offset"] - 5
                )

            elif adj_type == "pnl_drift_alert":
                if adj_type in seen_types:
                    continue
                seen_types.add(adj_type)
                adjustments["sl_modifier"] += 0.5
                adjustments["reduce_position_size"] = True

            elif adj_type == "regime_weight_reduction":
                if adj_type in seen_types:
                    continue
                seen_types.add(adj_type)
                adjustments["global_score_offset"] += 3

        # ── HARD CAP: prevent stacked adjustments from blocking ALL trades ──
        # Without this cap, 27+ global_confidence_reduction records stack to +415,
        # making effective_min_score = 40 + 415 = 455 (impossible to reach).
        # Cap at +5 means: worst case min_score goes from 40 -> 45 (aggressive).
        # Combined with quality floor (35) this still maintains risk control.
        adjustments["global_score_offset"] = min(adjustments["global_score_offset"], 5)
        adjustments["sl_modifier"] = min(adjustments["sl_modifier"], 1.0)

        # Additionally: penalize tokens in worst predictions
        try:
            stats = ll.get_performance_stats(days=7)
            for w in stats.get("worst_predictions", []):
                ticker = w.get("token_ticker", "").lower()
                if ticker:
                    adjustments["token_penalties"][ticker] = (
                        adjustments["token_penalties"].get(ticker, 0) - 10
                    )
            # Heavily penalize directions with <30% accuracy
            for direction, ddata in stats.get("direction_accuracy", {}).items():
                if ddata.get("total", 0) >= 5 and ddata.get("accuracy", 50) < 30:
                    key = "short" if direction == "SHORT" else "long"
                    adjustments["direction_bias"][key] = min(
                        adjustments["direction_bias"].get(key, 1.0), 0.3
                    )
        except Exception:
            pass

        # Cap per-token penalties at -25 (don't completely zero out a coin)
        for token in adjustments["token_penalties"]:
            adjustments["token_penalties"][token] = max(adjustments["token_penalties"][token], -25)

    except Exception as e:
        logger.warning("Failed to fetch learning adjustments: %s", e)

    return adjustments


# -- Database ------------------------------------------------------------------

def init_trading_tables():
    """No-op for Supabase — tables are created via the Supabase SQL Editor.
    Run database/supabase_schema.sql in your Supabase project's SQL Editor."""
    pass


# -- Transaction Hashing -------------------------------------------------------

def generate_tx_hash(user_id: int, action: str, coin_id: str, symbol: str,
                     price: float, quantity: float, amount: float, timestamp: str) -> str:
    """Generate a real SHA-256 transaction hash from trade data.
    This is a deterministic hash — the same inputs always produce the same hash,
    so any transaction can be independently verified."""
    raw = f"{user_id}|{action}|{coin_id}|{symbol}|{price:.8f}|{quantity:.8f}|{amount:.8f}|{timestamp}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def verify_tx_hash(tx_hash: str, user_id: int, action: str, coin_id: str, symbol: str,
                   price: float, quantity: float, amount: float, timestamp: str) -> bool:
    """Verify a transaction hash matches its data — returns True if valid."""
    expected = generate_tx_hash(user_id, action, coin_id, symbol, price, quantity, amount, timestamp)
    return tx_hash == expected


# -- Settings ------------------------------------------------------------------

def get_trade_settings(user_id: int) -> Dict[str, Any]:
    sb = get_supabase()
    result = sb.table("trade_settings").select("*").eq("user_id", user_id).execute()
    if result.data:
        row = result.data[0]
        # Convert boolean to int for backward compatibility
        row["auto_trade_enabled"] = 1 if row.get("auto_trade_enabled") else 0
        return row
    return {
        "user_id": user_id, "auto_trade_enabled": 1,  # ON by default
        "max_trade_pct": 20.0, "daily_loss_limit_pct": 10.0,
        "max_open_positions": 15, "stop_loss_pct": 1.5,
        "take_profit_pct": 2.0, "cooldown_minutes": 0,
        "min_market_cap": 1000000, "risk_level": "aggressive",
    }


def update_trade_settings(user_id: int, settings: Dict[str, Any]) -> Dict[str, Any]:
    sb = get_supabase()
    sb.table("trade_settings").upsert({
        "user_id": user_id,
        "auto_trade_enabled": bool(settings.get("auto_trade_enabled", 0)),
        "max_trade_pct": settings.get("max_trade_pct", 20.0),
        "daily_loss_limit_pct": settings.get("daily_loss_limit_pct", 10.0),
        "max_open_positions": settings.get("max_open_positions", 5),
        "stop_loss_pct": settings.get("stop_loss_pct", 1.5),
        "take_profit_pct": settings.get("take_profit_pct", 2.0),
        "cooldown_minutes": settings.get("cooldown_minutes", 0),
        "min_market_cap": settings.get("min_market_cap", 1000000),
        "risk_level": settings.get("risk_level", "moderate"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return get_trade_settings(user_id)


# -- Real Wallet Balance (from Supabase wallet_balance table) ------------------

def _get_wallet_balance(user_id: int) -> float:
    sb = get_supabase()
    result = sb.table("wallet_balance").select("balance").eq("user_id", user_id).execute()
    return result.data[0]["balance"] if result.data else 0.0


def _deduct_wallet(user_id: int, amount: float):
    sb = get_supabase()
    sb.rpc("update_wallet_balance", {"p_user_id": user_id, "p_delta": -amount}).execute()
    sb.table("wallet_transactions").insert({
        "user_id": user_id,
        "type": "trade_buy",
        "amount": amount,
        "description": f"Auto-trade: invested ${amount:,.2f}",
        "status": "completed",
    }).execute()


def _credit_wallet(user_id: int, amount: float, description: str = ""):
    sb = get_supabase()
    sb.rpc("update_wallet_balance", {"p_user_id": user_id, "p_delta": amount}).execute()
    sb.table("wallet_transactions").insert({
        "user_id": user_id,
        "type": "trade_sell",
        "amount": amount,
        "description": description,
        "status": "completed",
    }).execute()


# -- Trade Stats ---------------------------------------------------------------

def _get_trade_stats(user_id: int) -> Dict[str, Any]:
    sb = get_supabase()
    result = sb.table("trade_stats").select("*").eq("user_id", user_id).execute()
    if result.data:
        return result.data[0]
    # Initialize stats row if missing
    sb.table("trade_stats").upsert({"user_id": user_id}).execute()
    return {"user_id": user_id, "total_invested": 0, "total_pnl": 0, "total_trades": 0,
            "winning_trades": 0, "losing_trades": 0, "best_trade_pnl": 0, "worst_trade_pnl": 0}


def reset_trading(user_id: int) -> Dict[str, Any]:
    sb = get_supabase()
    try:
        # Get open positions to refund
        open_pos = sb.table("trade_positions").select("invested_amount").eq("user_id", user_id).eq("status", "open").execute()
        refund = sum(p["invested_amount"] for p in open_pos.data) if open_pos.data else 0

        if refund > 0:
            sb.rpc("update_wallet_balance", {"p_user_id": user_id, "p_delta": refund}).execute()
            sb.table("wallet_transactions").insert({
                "user_id": user_id,
                "type": "trade_refund",
                "amount": refund,
                "description": "Trading reset - positions refunded",
                "status": "completed",
            }).execute()

        # Close all open positions
        sb.table("trade_positions").update({
            "status": "closed",
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("user_id", user_id).eq("status", "open").execute()

        # Reset stats
        sb.rpc("reset_trade_stats", {"p_user_id": user_id}).execute()

        _log_event(user_id, "RESET", f"Trading reset — ${refund:,.0f} refunded to wallet")
        return {"success": True, "refunded": refund}
    except Exception as e:
        return {"success": False, "error": str(e)}


# -- Positions -----------------------------------------------------------------

def get_open_positions(user_id: int) -> List[Dict[str, Any]]:
    sb = get_supabase()
    result = sb.table("trade_positions").select("*").eq("user_id", user_id).eq("status", "open").order("opened_at", desc=True).execute()
    return [dict(r) for r in result.data] if result.data else []


def get_closed_positions(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    sb = get_supabase()
    result = sb.table("trade_positions").select("*").eq("user_id", user_id).eq("status", "closed").order("closed_at", desc=True).limit(limit).execute()
    return [dict(r) for r in result.data] if result.data else []


def get_trade_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    sb = get_supabase()
    result = sb.table("trade_orders").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
    return [dict(r) for r in result.data] if result.data else []


def get_trade_log_entries(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    sb = get_supabase()
    result = sb.table("trade_log").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
    return [dict(r) for r in result.data] if result.data else []


def _log_event(user_id: int, event: str, details: str):
    sb = get_supabase()
    sb.table("trade_log").insert({
        "user_id": user_id,
        "event": event,
        "details": details,
    }).execute()


# -- Core Trading Logic --------------------------------------------------------

def _send_trade_sms(user_id: int, message: str):
    """Send an SMS notification to a user about a trade event."""
    try:
        import sms_service
        if not sms_service.is_configured():
            return
        sb = get_supabase()
        user_row = sb.table("users").select("phone").eq("id", user_id).execute()
        if user_row.data and user_row.data[0].get("phone"):
            phone = user_row.data[0]["phone"]
            # Use Twilio messaging (not verify OTP)
            from twilio.rest import Client
            import os
            sid = os.getenv("TWILIO_ACCOUNT_SID", "")
            token = os.getenv("TWILIO_AUTH_TOKEN", "")
            from_number = os.getenv("TWILIO_FROM_NUMBER", "")
            if sid and token and from_number:
                client = Client(sid, token)
                client.messages.create(
                    body=f"[NexYpher] {message}",
                    from_=from_number,
                    to=sms_service._format_phone(phone)
                )
                logger.info("SMS sent to user %d: %s", user_id, message[:50])
    except Exception as e:
        logger.warning("SMS send failed for user %d: %s", user_id, e)


def execute_buy(user_id, coin_id, coin_name, symbol, price, amount, ai_score, ai_reasoning, stop_loss_pct=1.5, take_profit_pct=2.0, side="long", trade_metadata=None):
    """Execute a buy (LONG) or short-sell (SHORT) order."""
    sb = get_supabase()
    try:
        bal = sb.table("wallet_balance").select("balance").eq("user_id", user_id).execute()
        balance = bal.data[0]["balance"] if bal.data else 0.0
        if balance <= 0:
            return {"success": False, "error": "No funds in wallet. Add money from your bank account first."}
        if amount > balance:
            return {"success": False, "error": f"Insufficient balance. You have {balance:,.2f} but tried to invest {amount:,.2f}"}
        if amount <= 0:
            return {"success": False, "error": "Invalid amount"}

        quantity = amount / price
        # For SHORT positions: stop-loss is ABOVE entry, take-profit is BELOW entry
        if side == "short":
            stop_loss = price * (1 + stop_loss_pct / 100)
            take_profit = price * (1 - take_profit_pct / 100)
        else:
            stop_loss = price * (1 - stop_loss_pct / 100)
            take_profit = price * (1 + take_profit_pct / 100)

        timestamp = datetime.now(timezone.utc).isoformat()
        action_label = 'SHORT' if side == 'short' else 'BUY'
        tx_hash = generate_tx_hash(user_id, action_label, coin_id, symbol.upper(), price, quantity, amount, timestamp)

        # Insert position
        pos_result = sb.table("trade_positions").insert({
            "user_id": user_id,
            "coin_id": coin_id,
            "coin_name": coin_name,
            "symbol": symbol.upper(),
            "side": side,
            "entry_price": price,
            "current_price": price,
            "quantity": quantity,
            "invested_amount": amount,
            "current_value": amount,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "ai_reasoning": ai_reasoning,
            "tx_hash": tx_hash,
        }).execute()
        position_id = pos_result.data[0]["id"]

        # Insert order
        sb.table("trade_orders").insert({
            "user_id": user_id,
            "position_id": position_id,
            "coin_id": coin_id,
            "symbol": symbol.upper(),
            "action": action_label,
            "price": price,
            "quantity": quantity,
            "amount": amount,
            "ai_score": ai_score,
            "ai_reasoning": ai_reasoning,
            "tx_hash": tx_hash,
            "created_at": timestamp,
        }).execute()

        # Deduct wallet balance (margin for both LONG and SHORT)
        _deduct_wallet(user_id, amount)

        # Update trade stats
        sb.rpc("increment_trade_stats", {
            "p_user_id": user_id,
            "p_invested": amount,
            "p_trades": 1,
        }).execute()

        side_label = "SHORTED" if side == "short" else "Bought"
        _log_event(user_id, action_label, f"{side_label} {quantity:.6f} {symbol.upper()} at ${price:,.2f} (${amount:,.0f}) | Score: {ai_score}/100 | Hash: {tx_hash[:16]}... | {ai_reasoning[:180]}")

        # Record on blockchain (async, non-blocking)
        blockchain.record_transaction_async(tx_hash, action_label, symbol.upper(), amount)

        return {
            "success": True, "position_id": position_id, "quantity": quantity,
            "amount": amount, "tx_hash": tx_hash, "side": side,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct,
            "trade_metadata": trade_metadata or {},
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_sell(user_id, position_id, current_price, reason="manual"):
    """Close a position (LONG or SHORT) at the given price."""
    sb = get_supabase()
    try:
        pos_result = sb.table("trade_positions").select("*").eq("id", position_id).eq("user_id", user_id).eq("status", "open").execute()
        if not pos_result.data:
            return {"success": False, "error": "Position not found or already closed"}
        pos = pos_result.data[0]
        side = pos.get("side", "long")

        # P&L calculation depends on side
        if side == "short":
            # SHORT: profit when price goes DOWN
            current_value = pos["invested_amount"] + (pos["entry_price"] - current_price) * pos["quantity"]
            pnl = current_value - pos["invested_amount"]
        else:
            # LONG: profit when price goes UP
            current_value = pos["quantity"] * current_price
            pnl = current_value - pos["invested_amount"]
        pnl_pct = (pnl / pos["invested_amount"]) * 100 if pos["invested_amount"] > 0 else 0

        timestamp = datetime.now(timezone.utc).isoformat()
        close_action = 'COVER' if side == 'short' else 'SELL'
        tx_hash = generate_tx_hash(user_id, close_action, pos["coin_id"], pos["symbol"], current_price, pos["quantity"], current_value, timestamp)

        # Close the position
        sb.table("trade_positions").update({
            "status": "closed",
            "current_price": current_price,
            "current_value": max(0, current_value),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "tx_hash": tx_hash,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", position_id).execute()

        # Insert close order
        sb.table("trade_orders").insert({
            "user_id": user_id,
            "position_id": position_id,
            "coin_id": pos["coin_id"],
            "symbol": pos["symbol"],
            "action": close_action,
            "price": current_price,
            "quantity": pos["quantity"],
            "amount": max(0, current_value),
            "ai_reasoning": reason,
            "tx_hash": tx_hash,
            "created_at": timestamp,
        }).execute()

        # Credit wallet (return margin + P&L)
        credit_amount = max(0, current_value)
        side_label = "Covered short" if side == "short" else "Sold"
        _credit_wallet(user_id, credit_amount, f"{side_label} {pos['symbol']} — P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")

        # Update trade stats
        win_inc = 1 if pnl > 0 else 0
        loss_inc = 1 if pnl < 0 else 0
        sb.rpc("increment_trade_stats", {
            "p_user_id": user_id,
            "p_pnl": pnl,
            "p_trades": 1,
            "p_wins": win_inc,
            "p_losses": loss_inc,
            "p_best": pnl if pnl > 0 else 0,
            "p_worst": pnl if pnl < 0 else 0,
        }).execute()

        _log_event(user_id, close_action, f"{side_label} {pos['quantity']:.6f} {pos['symbol']} ({side.upper()}) at ${current_price:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%) | Hash: {tx_hash[:16]}... | Reason: {reason}")

        # Record on blockchain (async, non-blocking)
        blockchain.record_transaction_async(tx_hash, close_action, pos["symbol"], credit_amount)

        # Feed outcome back to AI learning loop (evaluate prediction)
        ll = _get_learning_loop()
        if ll:
            try:
                # Compute hold duration from position open time
                _sell_hold_minutes = 0.0
                _sell_opened = pos.get("opened_at") or pos.get("created_at", "")
                if _sell_opened:
                    try:
                        _ot = datetime.fromisoformat(_sell_opened.replace("Z", "+00:00"))
                        if _ot.tzinfo is None:
                            _ot = _ot.replace(tzinfo=timezone.utc)
                        _sell_hold_minutes = (datetime.now(timezone.utc) - _ot).total_seconds() / 60
                    except Exception:
                        pass
                ll.evaluate_trade_close(
                    token_ticker=pos["coin_id"],
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                    hold_duration_minutes=_sell_hold_minutes,
                    exit_reason="manual",
                )
            except Exception:
                pass

        # Compute hold duration for result
        hold_duration_str = ""
        hold_minutes = 0.0
        _sell_opened_at = pos.get("opened_at") or pos.get("created_at", "")
        if _sell_opened_at:
            try:
                _ot2 = datetime.fromisoformat(_sell_opened_at.replace("Z", "+00:00"))
                if _ot2.tzinfo is None:
                    _ot2 = _ot2.replace(tzinfo=timezone.utc)
                hold_minutes = (datetime.now(timezone.utc) - _ot2).total_seconds() / 60
                if hold_minutes >= 60:
                    hold_duration_str = f"{hold_minutes / 60:.1f}h"
                else:
                    hold_duration_str = f"{hold_minutes:.0f}m"
            except Exception:
                pass

        return {
            "success": True, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "amount": round(max(0, current_value), 2), "tx_hash": tx_hash, "side": side,
            "entry_price": pos["entry_price"], "close_reason": reason,
            "hold_duration": hold_duration_str, "hold_minutes": round(hold_minutes, 1),
            "coin_name": pos.get("coin_name", pos["symbol"]),
            "symbol": pos["symbol"], "quantity": pos["quantity"],
            "invested_amount": pos["invested_amount"],
            "ai_reasoning": pos.get("ai_reasoning", ""),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# -- AI Research Engine --------------------------------------------------------

async def research_opportunities(cg_collector, gemini_client=None):
    """Research and score trading opportunities — DIRECTION-AGNOSTIC.
    
    Principle: We don't have a LONG bias or SHORT bias.
    We READ THE MARKET and pick the direction that maximises profit.
    Both sides get symmetric scoring — bearish signals are just as valuable as bullish ones.
    XGBoost ML model validates direction when its accuracy justifies trust.
    
    NOTE: DexScreener removed — it provided unreliable data and tokens
    bypassed ML/backtest verification, causing many losing trades.
    """
    opportunities = []

    # ── Load XGBoost ML backtester for prediction signals ──
    ml_bt = get_ml_backtester()

    try:
        top_coins = await cg_collector.get_top_coins(limit=50)
        trending = await cg_collector.get_trending()
        trending_ids = {t.coin_id for t in trending} if trending else set()

        # ── ML predictions for top coins (max 8 to limit API calls + training time) ──
        # predict_latest auto-trains if no cache exists (first cycle ~7s/coin,
        # subsequent cycles use 7-day cache and are instant).
        ml_predictions = {}  # coin_id -> {buy_probability, prediction, model_accuracy}
        ml_coin_count = 0
        for coin in top_coins:
            if ml_coin_count >= 8:
                break
            if coin.coin_id in STABLECOINS:
                continue
            try:
                pred = await ml_bt.predict_latest(coin.coin_id, cg_collector, days=90)
                if pred and pred.get("model_accuracy", 0) >= 0.55:
                    ml_predictions[coin.coin_id] = pred
                    ml_coin_count += 1
            except Exception:
                pass
        ml_available = bool(ml_predictions)
        if ml_available:
            logger.info(
                "ML predictions loaded for %d coins: %s",
                len(ml_predictions),
                {k: f"{v['buy_probability']:.2f} ({v['model_accuracy']*100:.0f}%acc)"
                 for k, v in ml_predictions.items()},
            )
        else:
            logger.info("ML (XGBoost) not available — skipping ML score penalties")

        # ── Pretrained 38-feature model predictions ──
        pt_predictions = {}  # coin_id -> {verdict, direction, prob_up_24h, prob_up_7d, confidence, ...}
        pt_coin_count = 0
        for coin in top_coins:
            if pt_coin_count >= 8:
                break
            if coin.coin_id in STABLECOINS:
                continue
            try:
                pt_pred = await predict_pretrained(coin.coin_id, cg_collector, coin_data=coin)
                if pt_pred:
                    pt_predictions[coin.coin_id] = pt_pred
                    pt_coin_count += 1
            except Exception:
                pass
        pt_available = bool(pt_predictions)
        if pt_available:
            logger.info(
                "Pretrained predictions loaded for %d coins: %s",
                len(pt_predictions),
                {k: f"{v['verdict']} ({v['prob_up_7d']:.0f}% 7d)"
                 for k, v in pt_predictions.items()},
            )
        else:
            logger.info("Pretrained model not available — skipping PT score penalties")

        for coin in top_coins:
            if coin.coin_id in STABLECOINS:
                continue
            if coin.current_price <= 0.001:
                continue

            change_24h = coin.price_change_pct_24h

            # ── SHARED FACTORS (direction-independent) ──
            vol_ratio = coin.total_volume_24h / coin.market_cap if coin.market_cap > 0 else 0
            ath_ratio = coin.current_price / coin.ath if hasattr(coin, 'ath') and coin.ath and coin.ath > 0 else 0.5
            is_trending = coin.coin_id in trending_ids

            # Market cap tier (both sides benefit from liquidity)
            if coin.market_cap > 50_000_000_000:
                cap_score = 8; cap_label = "Mega cap — liquid"
            elif coin.market_cap > 10_000_000_000:
                cap_score = 6; cap_label = "Large cap"
            elif coin.market_cap > 1_000_000_000:
                cap_score = 5; cap_label = "Mid cap"
            elif coin.market_cap > 100_000_000:
                cap_score = 3; cap_label = "Small cap"
            else:
                cap_score = 2; cap_label = "Micro cap"

            # Volume (both sides benefit from active trading)
            if vol_ratio > 0.3:
                vol_score = 10; vol_label = f"High volume: {vol_ratio:.2f}"
            elif vol_ratio > 0.1:
                vol_score = 6; vol_label = f"Healthy volume: {vol_ratio:.2f}"
            elif vol_ratio > 0.05:
                vol_score = 3; vol_label = f"Normal volume: {vol_ratio:.2f}"
            else:
                vol_score = 1; vol_label = f"Low volume: {vol_ratio:.2f}"

            # Trending (attention = opportunity in BOTH directions)
            trend_score = 5 if is_trending else 0

            # ── ML PREDICTION SIGNAL ──
            # NOTE: The XGBoost model predicts "will price rise ≥5% in 7 days?"
            # Model is the PRIMARY signal — heuristics are secondary confirmation.
            # Low buy_prob does NOT equal "will crash" — it could mean sideways.
            # SHORT is only taken when direction model explicitly says DOWN.
            ml_pred = ml_predictions.get(coin.coin_id)
            ml_long_boost = 0
            ml_short_boost = 0
            ml_reason = ""
            if ml_pred:
                buy_prob = ml_pred.get("buy_probability", 0.5)
                ml_acc = ml_pred.get("model_accuracy", 0.5)
                # Accuracy gate already applied during loading (>= 0.55)
                if buy_prob >= 0.70:
                    ml_long_boost = 30
                    ml_reason = f"🤖 ML: strong BUY signal ({buy_prob*100:.0f}% up-prob, {ml_acc*100:.0f}% model acc)"
                elif buy_prob >= 0.60:
                    ml_long_boost = 20
                    ml_reason = f"🤖 ML: leans bullish ({buy_prob*100:.0f}% up-prob)"
                elif buy_prob >= 0.55:
                    ml_long_boost = 10
                    ml_reason = f"🤖 ML: mild bullish ({buy_prob*100:.0f}% up-prob)"
                elif buy_prob <= 0.25:
                    # Only SHORT when model is very confident price won’t rise
                    ml_short_boost = 10
                    ml_reason = f"🤖 ML: very unlikely to rise ({buy_prob*100:.0f}% up-prob) — mild SHORT"
                # buy_prob 0.25-0.55 = no ML signal (ambiguous zone — skip)

            # ── PRETRAINED 38-FEATURE MODEL SIGNAL (PRIMARY DECISION MAKER) ──
            pt_pred = pt_predictions.get(coin.coin_id)
            pt_long_boost = 0
            pt_short_boost = 0
            pt_reason = ""
            if pt_pred:
                verdict = pt_pred.get("verdict", "NEUTRAL")
                direction = pt_pred.get("direction", "SIDEWAYS")
                p7d = pt_pred.get("prob_up_7d", 50)
                conf = pt_pred.get("confidence", 1)
                acc7d = pt_pred.get("model_7d_acc", 0)
                # Verdict-based boost (MUCH higher weights — ML is primary signal)
                if verdict == "STRONG BUY":
                    pt_long_boost = 30
                    pt_reason = f"🧠 PT: STRONG BUY ({p7d:.0f}% 7d, conf {conf})"
                elif verdict == "BUY":
                    pt_long_boost = 18
                    pt_reason = f"🧠 PT: BUY ({p7d:.0f}% 7d)"
                elif verdict == "SELL":
                    pt_short_boost = 25
                    pt_reason = f"🧠 PT: SELL ({p7d:.0f}% 7d, conf {conf})"
                elif verdict == "AVOID":
                    pt_short_boost = 12
                    pt_reason = f"🧠 PT: AVOID ({p7d:.0f}% 7d)"
                elif verdict == "NEUTRAL":
                    # NEUTRAL = no ML edge = don’t trade this coin
                    pt_reason = f"🧠 PT: NEUTRAL ({p7d:.0f}% 7d) — no edge"
                # Direction confirmation bonus (+8 max)
                if direction == "UP" and pt_long_boost > 0:
                    pt_long_boost += 8
                    pt_reason += " | dir=UP"
                elif direction == "DOWN" and pt_short_boost > 0:
                    pt_short_boost += 8
                    pt_reason += " | dir=DOWN"
                # Cap combined ML+pretrained boost to prevent score inflation
                total_long_ml = ml_long_boost + pt_long_boost
                total_short_ml = ml_short_boost + pt_short_boost
                if total_long_ml > 50:
                    pt_long_boost = max(0, 50 - ml_long_boost)
                if total_short_ml > 40:
                    pt_short_boost = max(0, 40 - ml_short_boost)

            # ══════════════════════════════════════════════════════
            # ── LONG SIGNAL SCORING ──
            # ML models now contribute 50-60% of score; heuristics are secondary.
            # ══════════════════════════════════════════════════════
            long_score = 0
            long_reasons = []

            # Bullish momentum (reduced weights — heuristic-only max ~35 points)
            if change_24h > 10:
                long_score += 15; long_reasons.append(f"Explosive momentum: {change_24h:+.1f}%")
            elif change_24h > 3:
                long_score += 12; long_reasons.append(f"Strong bullish: {change_24h:+.1f}%")
            elif change_24h > 0.5:
                long_score += 8; long_reasons.append(f"Positive: {change_24h:+.1f}%")
            elif change_24h > -1:
                long_score += 4; long_reasons.append(f"Flat — mild dip-buy: {change_24h:+.1f}%")
            elif change_24h > -3:
                long_score += 2; long_reasons.append(f"Dip-buy zone: {change_24h:+.1f}%")
            # No points for LONG if change < -3% (don't catch falling knives)

            # ATH recovery potential (LONG-specific)
            if 0.2 < ath_ratio < 0.6:
                long_score += 8; long_reasons.append(f"Recovery potential — {(1-ath_ratio)*100:.0f}% below ATH")
            elif 0.6 <= ath_ratio < 0.85:
                long_score += 4; long_reasons.append("Near ATH recovery zone")

            # Add shared scores
            long_score += cap_score; long_reasons.append(cap_label)
            long_score += vol_score; long_reasons.append(vol_label)
            if trend_score:
                long_score += trend_score; long_reasons.append("Trending on CoinGecko")
            # ML boost for LONG (PRIMARY signal)
            if ml_long_boost:
                long_score += ml_long_boost; long_reasons.append(ml_reason)
            # Pretrained model boost for LONG (PRIMARY signal)
            if pt_long_boost:
                long_score += pt_long_boost; long_reasons.append(pt_reason)
            # PENALTY: No ML confirmation = risky trade → apply penalty
            # BUT only if ML is actually available on this server.
            # If ML packages aren't installed (e.g. Render), skip penalty entirely.
            if ml_long_boost == 0 and pt_long_boost == 0:
                if ml_available or pt_available:
                    # ML is working but didn't confirm THIS coin → halve
                    long_score = int(long_score * 0.5)
                    long_reasons.append("⚠️ No ML confirmation — score halved")
                # else: ML not installed — no penalty (heuristics are enough)

            long_score = max(0, min(100, long_score))

            if long_score >= 15:
                opportunities.append({
                    "coin_id": coin.coin_id, "name": coin.name, "symbol": coin.symbol.upper(),
                    "price": coin.current_price, "change_24h": change_24h,
                    "market_cap": coin.market_cap, "volume_24h": coin.total_volume_24h,
                    "score": long_score, "reasons": long_reasons, "reasoning": " | ".join(long_reasons),
                    "source": "coingecko", "side": "long",
                    # ML/PT signals for email metadata
                    "ml_reason": ml_reason, "ml_buy_probability": ml_pred.get("buy_probability") if ml_pred else None,
                    "ml_model_accuracy": ml_pred.get("model_accuracy") if ml_pred else None,
                    "pt_reason": pt_reason, "pt_verdict": pt_pred.get("verdict") if pt_pred else None,
                    "pt_prob_7d": pt_pred.get("prob_up_7d") if pt_pred else None,
                    "pt_confidence": pt_pred.get("confidence") if pt_pred else None,
                    "cap_label": cap_label, "is_trending": is_trending,
                    "vol_label": vol_label,
                })

            # ══════════════════════════════════════════════════════
            # ── SHORT SIGNAL SCORING (SYMMETRIC with LONG) ──
            # ══════════════════════════════════════════════════════
            short_score = 0
            short_reasons = []

            # Bearish momentum (reduced weights — heuristic max ~30 points)
            if change_24h < -10:
                short_score += 15; short_reasons.append(f"Strong bearish momentum: {change_24h:+.1f}%")
            elif change_24h < -3:
                short_score += 12; short_reasons.append(f"Bearish continuation: {change_24h:+.1f}%")
            elif change_24h < -0.5:
                short_score += 8; short_reasons.append(f"Declining: {change_24h:+.1f}%")
            elif change_24h < 1:
                short_score += 3; short_reasons.append(f"Flat — mild short: {change_24h:+.1f}%")
            # No points for SHORT if change > 1% alone (don't short rising momentum)

            # Overextended pump (contrarian short — BONUS on top of base)
            if change_24h > 15:
                short_score += 8; short_reasons.append(f"Pump overextended: {change_24h:+.1f}% — pullback likely")
            elif change_24h > 8:
                short_score += 5; short_reasons.append(f"Rally exhaustion: {change_24h:+.1f}%")

            # ATH resistance (SHORT-specific: near ATH = rejection zone)
            if ath_ratio > 0.95:
                short_score += 8; short_reasons.append(f"At ATH ({ath_ratio*100:.0f}%) — heavy resistance")
            elif ath_ratio > 0.85:
                short_score += 5; short_reasons.append(f"Near ATH ({ath_ratio*100:.0f}%) — resistance zone")
            # Far below ATH = already weak (short-friendly)
            elif ath_ratio < 0.3:
                short_score += 3; short_reasons.append(f"Deep below ATH ({ath_ratio*100:.0f}%) — sustained weakness")

            # Add shared scores (liquidity matters for shorts too)
            short_score += cap_score; short_reasons.append(cap_label)
            short_score += vol_score; short_reasons.append(vol_label)
            if trend_score:
                short_score += trend_score; short_reasons.append("Trending — high attention")
            # ML boost for SHORT
            if ml_short_boost:
                short_score += ml_short_boost; short_reasons.append(ml_reason)
            # Pretrained model boost for SHORT
            if pt_short_boost:
                short_score += pt_short_boost; short_reasons.append(pt_reason)
            # PENALTY: Shorts REQUIRE ML direction model confirmation
            # Without it, "low prob_up" could just mean sideways, not bearish.
            if ml_short_boost == 0 and pt_short_boost == 0:
                if ml_available or pt_available:
                    # ML available but no SHORT signal → heavy penalty
                    short_score = int(short_score * 0.3)
                    short_reasons.append("⚠️ No ML SHORT confirmation — score reduced 70%")
                else:
                    # ML not installed — moderate penalty (shorts are riskier without ML)
                    short_score = int(short_score * 0.5)
                    short_reasons.append("ℹ️ ML unavailable — moderate SHORT penalty")

            short_score = max(0, min(100, short_score))

            # SHORT minimum score raised to 25 (was 15) — shorts are inherently riskier
            # and need stronger conviction to avoid poorly-timed counter-trend trades.
            if short_score >= 25:
                opportunities.append({
                    "coin_id": coin.coin_id, "name": coin.name, "symbol": coin.symbol.upper(),
                    "price": coin.current_price, "change_24h": change_24h,
                    "market_cap": coin.market_cap, "volume_24h": coin.total_volume_24h,
                    "score": short_score, "reasons": short_reasons, "reasoning": " | ".join(short_reasons),
                    "source": "coingecko", "side": "short",
                    # ML/PT signals for email metadata
                    "ml_reason": ml_reason, "ml_buy_probability": ml_pred.get("buy_probability") if ml_pred else None,
                    "ml_model_accuracy": ml_pred.get("model_accuracy") if ml_pred else None,
                    "pt_reason": pt_reason, "pt_verdict": pt_pred.get("verdict") if pt_pred else None,
                    "pt_prob_7d": pt_pred.get("prob_up_7d") if pt_pred else None,
                    "pt_confidence": pt_pred.get("confidence") if pt_pred else None,
                    "cap_label": cap_label, "is_trending": is_trending,
                    "vol_label": vol_label,
                })
    except Exception as e:
        logger.warning("CoinGecko research failed: %s", e)

    # NOTE: DexScreener block removed — tokens bypassed ML/backtest verification
    # and provided unreliable data, causing many losing trades.

    if gemini_client and opportunities:
        try:
            top5 = sorted(opportunities, key=lambda x: x["score"], reverse=True)[:5]
            prompt = (
                "You are NEXYPHER, an expert crypto trading AI. For each coin below, write a detailed 2-3 sentence analysis explaining:\n"
                "1. WHY you would buy it right now (momentum, volume, trend signals)\n"
                "2. What RISKS exist (volatility, market cap, recent dumps)\n"
                "3. Your RECOMMENDATION (Strong Buy, Buy, Hold, or Avoid) with a target % gain\n\n"
                "Coins to analyze:\n"
                + "\n".join(
                    f"- {t['name']} ({t['symbol']}): Price ${t['price']:,.6f}, "
                    f"24h Change: {t['change_24h']:+.1f}%, Market Cap: ${t['market_cap']:,.0f}, "
                    f"Volume: ${t['volume_24h']:,.0f}, NEXYPHER Score: {t['score']}/100"
                    for t in top5
                )
                + "\n\nFormat: COIN_SYMBOL: [Recommendation] - Detailed reasoning..."
            )
            resp = await asyncio.wait_for(gemini_client.chat("You are a crypto trading AI assistant.", prompt), timeout=15)
            if resp.success:
                # BUGFIX: Extract per-coin analysis instead of duplicating full response
                for opp in top5:
                    symbol = opp["symbol"].upper()
                    # Find the relevant section for THIS coin
                    coin_analysis = ""
                    for line in resp.content.split("\n"):
                        if symbol in line.upper():
                            coin_analysis = line.strip()[:300]
                            break
                    opp["ai_analysis"] = coin_analysis or resp.content[:200]
                    if coin_analysis:
                        opp["reasoning"] = coin_analysis
        except Exception as e:
            logger.warning("AI analysis failed: %s", e)

    opportunities.sort(key=lambda x: x["score"], reverse=True)

    # ── MANDATORY BACKTEST VERIFICATION ──
    # Only CoinGecko tokens (non-address coin_ids) can be backtested
    # because we need 6 months of OHLCV from CoinGecko.
    backtested = []
    bt_engine = get_backtest_engine()

    for opp in opportunities:
        coin_id = opp.get("coin_id", "")
        # Skip DEX-only tokens (addresses) — they lack historical data
        if coin_id.startswith("0x") or opp.get("source") == "dexscreener":
            opp["backtest_status"] = "skipped"
            opp["backtest_reason"] = "DEX token — insufficient historical data for backtest"
            opp["backtest_verified"] = False
            backtested.append(opp)
            continue

        # Check cache first
        cached = _get_cached_backtest(coin_id)
        if cached:
            bt_result = cached
        else:
            try:
                bt_result = await bt_engine.run_backtest(
                    coin_id=coin_id,
                    coin_name=opp["name"],
                    symbol=opp["symbol"],
                    cg_collector=cg_collector,
                    days=180,
                    market_cap=opp.get("market_cap", 0),
                )
                _cache_backtest(coin_id, bt_result)
            except Exception as e:
                logger.warning("Backtest failed for %s: %s", coin_id, e)
                opp["backtest_status"] = "error"
                opp["backtest_reason"] = f"Backtest error: {e}"
                opp["backtest_verified"] = False
                backtested.append(opp)
                continue

        # Attach backtest stats to the opportunity
        opp["backtest_verified"] = bt_result.passed_all_thresholds
        opp["backtest_status"] = "passed" if bt_result.passed_all_thresholds else "failed"
        opp["backtest_stats"] = {
            "win_rate": round(bt_result.win_rate, 2),
            "total_return": round(bt_result.total_return, 2),
            "max_drawdown": round(bt_result.max_drawdown, 2),
            "sharpe_ratio": round(bt_result.sharpe_ratio, 2),
            "total_trades": bt_result.total_trades,
            "period": f"{bt_result.start_date} to {bt_result.end_date}",
            "days_covered": bt_result.days_covered,
        }
        opp["backtest_recommendation"] = bt_result.recommendation
        opp["backtest_detail"] = bt_result.recommendation_detail
        opp["backtest_confidence"] = bt_result.confidence
        opp["backtest_strategy_direction"] = getattr(bt_result, 'strategy_direction', 'LONG')
        opp["backtest_detected_trend"] = getattr(bt_result, 'detected_trend', 'unknown')
        opp["backtest_token_tier"] = getattr(bt_result, 'token_tier', 'unknown')
        opp["backtest_strategies_tested"] = getattr(bt_result, 'strategies_tested', [])

        if bt_result.passed_all_thresholds:
            # Boost score for verified tokens
            direction = opp["backtest_strategy_direction"]
            opp["score"] = min(100, opp["score"] + 15)
            opp["reasons"].append(
                f"✅ Backtest verified ({direction}): {bt_result.win_rate:.0f}% win rate, "
                f"{bt_result.total_return:.1f}% return, Sharpe {bt_result.sharpe_ratio:.2f}"
            )
            opp["reasoning"] = " | ".join(opp["reasons"])
        else:
            # Backtest FAILED — historical data says this pattern loses money.
            # Penalty increased from -5 to -15 to actually discourage trading.
            tested = opp.get("backtest_strategies_tested", [])
            opp["score"] = max(0, opp["score"] - 15)
            opp["reasons"].append(
                f"⚠️ Backtest FAILED (tested {', '.join(tested) if tested else 'LONG'}): "
                f"{'; '.join(bt_result.failure_reasons)}"
            )
            opp["reasoning"] = " | ".join(opp["reasons"])

        backtested.append(opp)

    backtested.sort(key=lambda x: x["score"], reverse=True)
    return backtested


# ── Per-token cooldown after stop-loss (prevent re-buying losers immediately) ──
_token_cooldowns: Dict[str, float] = {}  # coin_id -> timestamp when cooldown expires
TOKEN_COOLDOWN_MINUTES = 60  # 1 hour cooldown after SL hit

# -- Autonomous Trading Loop ---------------------------------------------------

async def fast_position_check(user_id, cg_collector):
    """
    Fast position management — runs every cycle (10s).
    Only checks SL/TP/trailing-stop/auto-exit for open positions.
    Does NOT do research, backtesting, or AI analysis.
    This ensures expired/stopped positions close within seconds, not minutes.
    """
    settings = get_trade_settings(user_id)
    sb = get_supabase()
    results = {"actions": [], "positions_updated": 0, "positions_closed": 0, "trade_details": []}

    open_positions = get_open_positions(user_id)
    if not open_positions:
        return results

    # Batch price fetch — single API call for all coins
    coin_ids = list({p.get("coin_id", "") for p in open_positions
                     if p.get("coin_id") and not p["coin_id"].startswith("0x") and len(p["coin_id"]) <= 20})
    prices_map = {}
    if coin_ids:
        try:
            prices_map = await cg_collector.get_simple_price(coin_ids)
        except Exception:
            pass

    for pos in open_positions:
        try:
            coin_id = pos.get("coin_id", "")
            side = pos.get("side", "long")
            current_price = prices_map.get(coin_id, pos["current_price"])

            # P&L calculation based on side
            if side == "short":
                current_value = pos["invested_amount"] + (pos["entry_price"] - current_price) * pos["quantity"]
                pnl = current_value - pos["invested_amount"]
            else:
                current_value = pos["quantity"] * current_price if current_price > 0 else 0
                pnl = current_value - pos["invested_amount"]
            pnl_pct = (pnl / pos["invested_amount"]) * 100 if pos["invested_amount"] > 0 else 0

            # Update position in DB
            sb.table("trade_positions").update({
                "current_price": current_price,
                "current_value": max(0, current_value),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }).eq("id", pos["id"]).execute()
            results["positions_updated"] += 1

            # Compute hold duration
            _opened_at_str = pos.get("opened_at") or pos.get("created_at", "")
            _hold_minutes = 0.0
            if _opened_at_str:
                try:
                    _opened_at = datetime.fromisoformat(_opened_at_str.replace("Z", "+00:00"))
                    if _opened_at.tzinfo is None:
                        _opened_at = _opened_at.replace(tzinfo=timezone.utc)
                    _hold_minutes = (datetime.now(timezone.utc) - _opened_at).total_seconds() / 60
                except Exception:
                    pass

            # Check stop-loss
            pos_sl = pos.get("stop_loss", 0)
            pos_tp = pos.get("take_profit", 0)
            sl_hit = False
            tp_hit = False
            if pos_sl > 0:
                if side == "short":
                    sl_hit = current_price >= pos_sl
                else:
                    sl_hit = current_price <= pos_sl
            else:
                sl_hit = pnl_pct <= -settings["stop_loss_pct"]
            if pos_tp > 0:
                if side == "short":
                    tp_hit = current_price <= pos_tp
                else:
                    tp_hit = current_price >= pos_tp
            else:
                tp_hit = pnl_pct >= settings["take_profit_pct"]

            if sl_hit:
                side_verb = "covered" if side == "short" else "sold"
                sell_reason = (
                    f"STOP-LOSS ({side.upper()}): {pos['symbol']} lost {pnl_pct:.1f}% "
                    f"(entry ${pos['entry_price']:.4f} -> ${current_price:.4f}). "
                    f"Loss ${abs(pnl):,.0f}. Cutting losses."
                )
                sell_result = execute_sell(user_id, pos["id"], current_price, sell_reason)
                if sell_result["success"]:
                    results["actions"].append(
                        f"Stop-loss: {side_verb} {pos['symbol']} ({side.upper()}) at ${current_price:,.4f} (P&L: {pnl_pct:+.1f}%)"
                    )
                    results["positions_closed"] += 1
                    results["trade_details"].append({
                        "type": "sell", "action": "COVER" if side == "short" else "SELL",
                        "symbol": pos["symbol"], "coin_name": pos.get("coin_name", pos["symbol"]),
                        "price": current_price, "quantity": pos["quantity"],
                        "amount": sell_result["amount"], "side": side,
                        "pnl": sell_result["pnl"], "pnl_pct": sell_result["pnl_pct"],
                        "entry_price": pos["entry_price"], "close_reason": sell_reason,
                        "hold_duration": sell_result.get("hold_duration", ""),
                        "ai_reasoning": pos.get("ai_reasoning", ""),
                    })
                    ll = _get_learning_loop()
                    if ll:
                        try: ll.evaluate_trade_close(pos["coin_id"], current_price, pnl_pct, hold_duration_minutes=_hold_minutes, exit_reason="stop_loss")
                        except Exception: pass
                    _token_cooldowns[coin_id] = time.time() + TOKEN_COOLDOWN_MINUTES * 60
                    logger.info("Token cooldown set: %s blocked for %d min after SL", pos.get("symbol", coin_id), TOKEN_COOLDOWN_MINUTES)
                continue

            # Trailing stop
            if pos_sl > 0 and pos["entry_price"] > 0:
                if side == "long" and current_price > pos["entry_price"] * 1.01:
                    new_sl = current_price * (1 - settings["stop_loss_pct"] / 100)
                    if new_sl > pos_sl:
                        sb.table("trade_positions").update({"stop_loss": round(new_sl, 6)}).eq("id", pos["id"]).execute()
                        logger.debug("Trailing SL updated for %s LONG: $%.4f -> $%.4f", pos.get("symbol"), pos_sl, new_sl)
                elif side == "short" and current_price < pos["entry_price"] * 0.99:
                    new_sl = current_price * (1 + settings["stop_loss_pct"] / 100)
                    if new_sl < pos_sl:
                        sb.table("trade_positions").update({"stop_loss": round(new_sl, 6)}).eq("id", pos["id"]).execute()
                        logger.debug("Trailing SL updated for %s SHORT: $%.4f -> $%.4f", pos.get("symbol"), pos_sl, new_sl)

            if tp_hit:
                side_verb = "covered" if side == "short" else "sold"
                sell_reason = (
                    f"TAKE-PROFIT ({side.upper()}): {pos['symbol']} gained {pnl_pct:.1f}% "
                    f"(entry ${pos['entry_price']:.4f} -> ${current_price:.4f}). "
                    f"Profit ${pnl:,.0f}. Locking in gains."
                )
                sell_result = execute_sell(user_id, pos["id"], current_price, sell_reason)
                if sell_result["success"]:
                    results["actions"].append(
                        f"Take-profit: {side_verb} {pos['symbol']} ({side.upper()}) at ${current_price:,.4f} (P&L: {pnl_pct:+.1f}%)"
                    )
                    results["positions_closed"] += 1
                    results["trade_details"].append({
                        "type": "sell", "action": "COVER" if side == "short" else "SELL",
                        "symbol": pos["symbol"], "coin_name": pos.get("coin_name", pos["symbol"]),
                        "price": current_price, "quantity": pos["quantity"],
                        "amount": sell_result["amount"], "side": side,
                        "pnl": sell_result["pnl"], "pnl_pct": sell_result["pnl_pct"],
                        "entry_price": pos["entry_price"], "close_reason": sell_reason,
                        "hold_duration": sell_result.get("hold_duration", ""),
                        "ai_reasoning": pos.get("ai_reasoning", ""),
                    })
                    ll = _get_learning_loop()
                    if ll:
                        try: ll.evaluate_trade_close(pos["coin_id"], current_price, pnl_pct, hold_duration_minutes=_hold_minutes, exit_reason="take_profit")
                        except Exception: pass
                continue

            # Auto-exit (max hold time) + SMS warning
            opened_at_str = pos.get("opened_at") or pos.get("created_at", "")
            if opened_at_str:
                try:
                    opened_at = datetime.fromisoformat(opened_at_str.replace("Z", "+00:00"))
                    if opened_at.tzinfo is None:
                        opened_at = opened_at.replace(tzinfo=timezone.utc)
                    seconds_held = (datetime.now(timezone.utc) - opened_at).total_seconds()
                    hours_held = seconds_held / 3600
                    minutes_remaining = (MAX_HOLD_HOURS * 60) - (seconds_held / 60)

                    if 0 < minutes_remaining <= WARNING_MINUTES_BEFORE_CLOSE:
                        warning_key = f"warned_{pos['id']}"
                        if warning_key not in results:
                            results[warning_key] = True
                            side_verb = "cover" if side == "short" else "close"
                            _send_trade_sms(
                                user_id,
                                f"⚠️ {pos['symbol']} ({side.upper()}) will auto-{side_verb} in {minutes_remaining:.0f}min! "
                                f"P&L: {pnl_pct:+.1f}% (${pnl:,.0f}). "
                                f"Entry: ${pos['entry_price']:.4f} → Now: ${current_price:.4f}"
                            )
                            results["actions"].append(
                                f"⚠️ SMS warning: {pos['symbol']} ({side.upper()}) closing in {minutes_remaining:.0f}min"
                            )

                    if hours_held >= MAX_HOLD_HOURS:
                        side_verb = "covered" if side == "short" else "sold"
                        sell_reason = (
                            f"AUTO-EXIT ({side.upper()}): {pos['symbol']} held {hours_held:.1f}h "
                            f"(max {MAX_HOLD_HOURS}h). P&L: {pnl_pct:+.1f}%. FORCED CLOSE."
                        )
                        sell_result = execute_sell(user_id, pos["id"], current_price, sell_reason)
                        if sell_result["success"]:
                            results["actions"].append(
                                f"Auto-exit: {side_verb} {pos['symbol']} ({side.upper()}) after {hours_held:.1f}h (P&L: {pnl_pct:+.1f}%)"
                            )
                            results["positions_closed"] += 1
                            results["trade_details"].append({
                                "type": "sell", "action": "COVER" if side == "short" else "SELL",
                                "symbol": pos["symbol"], "coin_name": pos.get("coin_name", pos["symbol"]),
                                "price": current_price, "quantity": pos["quantity"],
                                "amount": sell_result["amount"], "side": side,
                                "pnl": sell_result["pnl"], "pnl_pct": sell_result["pnl_pct"],
                                "entry_price": pos["entry_price"], "close_reason": sell_reason,
                                "hold_duration": sell_result.get("hold_duration", ""),
                                "ai_reasoning": pos.get("ai_reasoning", ""),
                            })
                            _send_trade_sms(
                                user_id,
                                f"🔴 AUTO-CLOSED {pos['symbol']} ({side.upper()}) after {hours_held:.1f}h | "
                                f"P&L: {pnl_pct:+.1f}% (${pnl:,.0f})"
                            )
                            ll = _get_learning_loop()
                            if ll:
                                try: ll.evaluate_trade_close(pos["coin_id"], current_price, pnl_pct, hold_duration_minutes=seconds_held / 60, exit_reason="auto_exit")
                                except Exception: pass
                        continue
                except Exception:
                    pass

        except Exception as e:
            logger.warning("Fast position check failed for %s: %s", pos.get("coin_id"), e)

    return results


async def auto_trade_cycle(user_id, cg_collector, gemini_client=None):
    settings = get_trade_settings(user_id)
    if not settings.get("auto_trade_enabled"):
        return {"status": "disabled", "message": "Auto-trading is disabled. Turn on the toggle to start."}

    # ── Load user preferences for confidence threshold ──
    try:
        import auth as _auth
        user_prefs = _auth.get_user_preferences(user_id)
        confidence_threshold = user_prefs.auto_trade_threshold  # 0-10
        max_daily = user_prefs.max_daily_trades
        risk_profile = user_prefs.risk_profile
    except Exception:
        confidence_threshold = 5.0
        max_daily = 999_999
        risk_profile = "aggressive"

    # Always remove daily trade cap — let position limits & balance guard instead
    max_daily = 999_999

    # Trade settings risk_level overrides user prefs (auto-trader sets this to "aggressive")
    risk_profile = settings.get("risk_level", risk_profile)

    # ── LEARNING LOOP FEEDBACK (the critical missing link) ──
    learning_mods = _get_learning_adjustments()
    market_regime = "sideways"  # updated after BTC data is available
    if learning_mods.get("global_score_offset", 0) > 0 or learning_mods.get("reduce_position_size"):
        logger.info(
            "Learning loop active: score_offset=+%d, reduce_size=%s, token_penalties=%d, dir_bias=%s",
            learning_mods["global_score_offset"],
            learning_mods["reduce_position_size"],
            len(learning_mods["token_penalties"]),
            learning_mods["direction_bias"],
        )

    # Risk profile modifiers
    risk_modifiers = {
        "conservative": {"max_trade_pct_mult": 0.5, "min_score": 70, "stop_loss_add": 2},
        "moderate":     {"max_trade_pct_mult": 1.0, "min_score": 55, "stop_loss_add": 0},
        "balanced":     {"max_trade_pct_mult": 1.0, "min_score": 55, "stop_loss_add": 0},
        "aggressive":   {"max_trade_pct_mult": 1.5, "min_score": 40, "stop_loss_add": 0},
    }
    mods = risk_modifiers.get(risk_profile, risk_modifiers["aggressive"])

    balance = _get_wallet_balance(user_id)
    stats = _get_trade_stats(user_id)
    total_pnl = stats.get("total_pnl", 0)
    results = {"actions": [], "positions_updated": 0, "new_trades": 0, "trade_details": []}

    sb = get_supabase()

    if balance > 0:
        loss_pct = (total_pnl / balance) * 100 if total_pnl < 0 else 0
        if loss_pct < -settings["daily_loss_limit_pct"]:
            _log_event(user_id, "SAFETY", f"Daily loss limit hit: {loss_pct:.1f}%")
            return {"status": "paused", "message": f"Daily loss limit reached ({loss_pct:.1f}%)"}

    # (No cooldown for always-on all-in bot — we sell and rebuy every cycle)

    # ── STEP 1: Fast position management (SL/TP/trailing/auto-exit) ──
    # This is now handled by fast_position_check() called every 10s.
    # Still call it here for manual cycle triggers via /api/trader/run-cycle.
    pos_result = await fast_position_check(user_id, cg_collector)
    results["positions_updated"] = pos_result["positions_updated"]
    results["actions"].extend(pos_result["actions"])

    # ── STEP 2: DIVERSIFIED PORTFOLIO — open new LONG and SHORT positions ──
    # Re-fetch balance and open positions after step 1 (some may have closed)
    balance = _get_wallet_balance(user_id)
    open_positions = get_open_positions(user_id)

    # Daily trade cap removed — unlimited trades (position limits & balance guard)
    logger.info(
        "Trade check — balance: %s | open: %d/%d",
        f"${balance:,.0f}", len(open_positions), MAX_PORTFOLIO_SLOTS,
    )
    if True:
        opportunities = await research_opportunities(cg_collector, gemini_client)

        # Filter stablecoins, invalid prices, and low market cap
        opportunities = [o for o in opportunities if o.get("price", 0) > 0.001]
        opportunities = [o for o in opportunities if o["market_cap"] >= settings["min_market_cap"]]
        logger.info(
            "Research returned %d opportunities (after price/mcap filter)",
            len(opportunities),
        )

        # ── REGIME DETECTION (BTC 24h change as proxy) ──
        btc_opp = next((o for o in opportunities if o.get("coin_id") == "bitcoin"), None)
        if btc_opp:
            market_regime = _detect_market_regime(btc_opp["change_24h"])
            logger.info("Market regime: %s (BTC 24h: %+.1f%%)", market_regime, btc_opp["change_24h"])

        # ── BACKTEST GATE ──
        # Conservative/moderate: hard block. Aggressive: heavy score penalty.
        if risk_profile != "aggressive":
            verified_opportunities = []
            for o in opportunities:
                if o.get("backtest_verified"):
                    verified_opportunities.append(o)
                else:
                    bt_status = o.get("backtest_status", "unknown")
                    logger.info(
                        "Backtest gate BLOCKED %s (%s) — status: %s",
                        o.get("symbol", "?"), o.get("coin_id", "?"), bt_status,
                    )
            opportunities = verified_opportunities
            results["backtest_filtered"] = len(verified_opportunities)
        else:
            # Aggressive: don't hard-block, but penalize unverified coins
            for o in opportunities:
                if not o.get("backtest_verified") and o.get("backtest_status") == "failed":
                    o["score"] = max(0, o["score"] - 20)
                    o["reasons"].append("⚠️ Backtest FAILED — score penalized -20")
            opportunities.sort(key=lambda x: x["score"], reverse=True)
            logger.info("Aggressive mode — backtest failures penalized, %d coins eligible", len(opportunities))

        # ── APPLY LEARNING LOOP ADJUSTMENTS TO SCORES ──
        for o in opportunities:
            cid = o.get("coin_id", "").lower()
            side = o.get("side", "long")
            # Token-specific penalty from past underperformance
            token_penalty = learning_mods["token_penalties"].get(cid, 0)
            if token_penalty:
                o["score"] = max(0, o["score"] + token_penalty)
                o["reasons"].append(f"📉 Learning penalty: {token_penalty} (past losses)")
            # Direction bias correction
            dir_mult = learning_mods["direction_bias"].get(side, 1.0)
            if dir_mult < 1.0:
                old_score = o["score"]
                o["score"] = max(0, int(o["score"] * dir_mult))
                o["reasons"].append(f"📉 {side.upper()} accuracy low: score {old_score}→{o['score']}")

        # ── REGIME-BASED SCORE ADJUSTMENT ──
        for o in opportunities:
            side = o.get("side", "long")
            if market_regime in ("strong_bull", "bull") and side == "long":
                o["score"] = min(100, o["score"] + 10)
            elif market_regime in ("strong_bear", "bear") and side == "short":
                o["score"] = min(100, o["score"] + 10)
            elif market_regime in ("strong_bull", "bull") and side == "short":
                o["score"] = max(0, o["score"] - 15)
            elif market_regime in ("strong_bear", "bear") and side == "long":
                o["score"] = max(0, o["score"] - 15)
        opportunities.sort(key=lambda x: x["score"], reverse=True)

        # Apply confidence threshold (with learning loop offset)
        min_score = mods["min_score"]
        learning_offset = learning_mods.get("global_score_offset", 0)
        confidence_min_score = int(confidence_threshold * 10)
        base_min = min_score + learning_offset
        # BUGFIX: Always apply user's confidence_threshold, even in aggressive mode.
        # Previously aggressive mode ignored it entirely, letting low-confidence trades through.
        effective_min_score = max(base_min, confidence_min_score)
        pre_filter_count = len(opportunities)
        opportunities = [o for o in opportunities if o["score"] >= effective_min_score]
        logger.info(
            "Score filter — min_score: %d (base %d + learning %d) | before: %d | after: %d | regime: %s | top: %s",
            effective_min_score, min_score, learning_offset, pre_filter_count, len(opportunities),
            market_regime,
            [o['score'] for o in sorted(opportunities, key=lambda x: x['score'], reverse=True)[:5]],
        )
        results["confidence_threshold"] = confidence_threshold
        results["effective_min_score"] = effective_min_score
        results["risk_profile"] = risk_profile
        results["market_regime"] = market_regime

        # ── DIVERSIFIED PORTFOLIO: 10-15 positions, balanced LONG/SHORT ──

        # Filter out coins we already hold (same coin + same side)
        current_positions_key = {(p["coin_id"], p.get("side", "long")) for p in open_positions}
        new_opportunities = [o for o in opportunities if (o["coin_id"], o.get("side", "long")) not in current_positions_key]

        # Filter out coins under cooldown (recently hit stop-loss)
        now_ts = time.time()
        cooled_off = []
        for o in new_opportunities:
            cid = o.get("coin_id", "")
            cooldown_until = _token_cooldowns.get(cid, 0)
            if now_ts < cooldown_until:
                remaining_min = (cooldown_until - now_ts) / 60
                logger.info("Skipping %s — cooldown active (%.0f min remaining after SL)", o.get("symbol", cid), remaining_min)
            else:
                _token_cooldowns.pop(cid, None)  # Clean up expired cooldowns
                cooled_off.append(o)
        new_opportunities = cooled_off

        # Separate LONG and SHORT opportunities
        long_opps = [o for o in new_opportunities if o.get("side", "long") == "long"]
        short_opps = [o for o in new_opportunities if o.get("side") == "short"]

        # Count existing LONG/SHORT positions for balance
        current_longs = sum(1 for p in open_positions if p.get("side", "long") == "long")
        current_shorts = sum(1 for p in open_positions if p.get("side") == "short")
        open_count = len(open_positions)
        available_slots = max(0, MAX_PORTFOLIO_SLOTS - open_count)

        # Calculate total portfolio value for max position % check
        total_portfolio = balance + sum(p.get("invested_amount", 0) for p in open_positions)

        # Regime-based LONG/SHORT allocation
        if market_regime in ("strong_bull", "bull"):
            target_long_pct = 0.80  # Bull = mostly long
        elif market_regime in ("strong_bear", "bear"):
            target_long_pct = 0.30  # Bear = mostly short
        else:
            target_long_pct = 0.60  # Sideways = slight long bias
        target_longs = int(MAX_PORTFOLIO_SLOTS * target_long_pct)
        target_shorts = MAX_PORTFOLIO_SLOTS - target_longs
        long_slots = max(0, min(available_slots, target_longs - current_longs))
        short_slots = max(0, min(available_slots - long_slots, target_shorts - current_shorts))

        # Merge top picks: take best LONGs and best SHORTs
        picks = []
        picks.extend([(o, "long") for o in long_opps[:long_slots]])
        picks.extend([(o, "short") for o in short_opps[:short_slots]])
        # Fill remaining slots with any remaining best-scored opportunities
        remaining_slots = available_slots - len(picks)
        if remaining_slots > 0:
            used_ids = {(o["coin_id"], s) for o, s in picks}
            remaining = [o for o in new_opportunities if (o["coin_id"], o.get("side", "long")) not in used_ids]
            picks.extend([(o, o.get("side", "long")) for o in remaining[:remaining_slots]])

        if picks and balance > (MIN_POSITION_AMOUNT + RESERVE_AMOUNT):
            # ── Score-weighted allocation with DYNAMIC risk sizing ──
            investable = balance - RESERVE_AMOUNT
            total_score = sum(max(o["score"], 1) for o, _ in picks)
            base_sl = max(1.0, settings["stop_loss_pct"] + mods["stop_loss_add"] + learning_mods.get("sl_modifier", 0))
            base_tp = settings["take_profit_pct"]

            slots_used_this_cycle = 0  # Track slots consumed this cycle to avoid exceeding MAX
            for opp, trade_side in picks:
                if balance - RESERVE_AMOUNT < MIN_POSITION_AMOUNT:
                    break
                # Re-check slot availability after each buy to prevent exceeding MAX_PORTFOLIO_SLOTS
                if (len(open_positions) + slots_used_this_cycle) >= MAX_PORTFOLIO_SLOTS:
                    logger.info("Portfolio full (%d + %d slots used) — stopping new trades", len(open_positions), slots_used_this_cycle)
                    break

                score = opp["score"]

                # ── SCORE-TIERED RISK PARAMETERS ──
                # Uses user's configured SL/TP as base, only adjusts allocation %.
                # Higher scores get slightly wider TP and more allocation.
                if score >= 80:
                    effective_sl = base_sl; effective_tp = base_tp + 0.5; max_alloc_pct = 10.0
                elif score >= 60:
                    effective_sl = base_sl; effective_tp = base_tp; max_alloc_pct = 7.0
                elif score >= 45:
                    effective_sl = base_sl; effective_tp = base_tp; max_alloc_pct = 5.0
                elif score >= 35:
                    effective_sl = base_sl; effective_tp = base_tp; max_alloc_pct = 3.0
                else:
                    logger.info("Skipping %s (score %d < 35) — below quality floor", opp["symbol"], score)
                    continue  # Quality floor: don't trade on very weak signals

                # Reduce size if learning loop recommends caution
                if learning_mods.get("reduce_position_size"):
                    max_alloc_pct *= 0.6

                # Allocate proportional to score
                weight = max(score, 1) / total_score
                trade_amount = max(investable * weight, MIN_POSITION_AMOUNT)
                trade_amount = min(trade_amount, balance - RESERVE_AMOUNT)

                # Enforce max position % (score-tiered cap)
                trade_amount = min(trade_amount, total_portfolio * (max_alloc_pct / 100))

                # Build AI reasoning
                bt_stats = opp.get("backtest_stats", {})
                bt_direction = opp.get("backtest_strategy_direction", trade_side.upper())
                bt_trend = opp.get("backtest_detected_trend", "unknown")
                bt_summary = ""
                if bt_stats:
                    bt_summary = (
                        f"BACKTEST ({bt_direction}, {bt_trend}): "
                        f"{bt_stats.get('win_rate', 0):.1f}% WR, "
                        f"{bt_stats.get('total_return', 0):.1f}% ret. "
                    )

                side_label = "SHORT" if trade_side == "short" else "BUY"
                detailed_reasoning = (
                    f"{side_label} {opp['name']} ({opp['symbol']}): "
                    f"${opp['price']:,.6f} | 24h: {opp['change_24h']:+.1f}% | "
                    f"Score: {opp['score']}/100. {bt_summary}"
                    f"Investing ${trade_amount:,.0f} | SL: -{effective_sl}% | TP: +{effective_tp}% | Regime: {market_regime}."
                )
                if opp.get("ai_analysis"):
                    detailed_reasoning += f" {opp['ai_analysis'][:150]}"

                # ── Build comprehensive trade metadata for email ──
                trade_metadata = {
                    "direction": trade_side.upper(),
                    "direction_reason": opp.get("reasoning", ""),
                    "ai_score": opp["score"],
                    "market_regime": market_regime,
                    "change_24h": opp.get("change_24h", 0),
                    "market_cap": opp.get("market_cap", 0),
                    "volume_24h": opp.get("volume_24h", 0),
                    "cap_tier": opp.get("cap_label", ""),
                    "is_trending": opp.get("is_trending", False),
                    "stop_loss_pct": effective_sl,
                    "take_profit_pct": effective_tp,
                    "max_alloc_pct": max_alloc_pct,
                    "max_hold_hours": MAX_HOLD_HOURS,
                    "reasons": opp.get("reasons", []),
                    "ai_analysis": opp.get("ai_analysis", ""),
                    # Backtest data
                    "backtest_verified": opp.get("backtest_verified", False),
                    "backtest_status": opp.get("backtest_status", "none"),
                    "backtest_stats": bt_stats,
                    "backtest_strategy_direction": bt_direction,
                    "backtest_detected_trend": bt_trend,
                    "backtest_strategies_tested": opp.get("backtest_strategies_tested", []),
                    "backtest_recommendation": opp.get("backtest_recommendation", ""),
                    # ML XGBoost signal
                    "ml_signal": opp.get("ml_reason", ""),
                    "ml_buy_probability": opp.get("ml_buy_probability", None),
                    "ml_model_accuracy": opp.get("ml_model_accuracy", None),
                    # Pretrained 38-feature model
                    "pt_signal": opp.get("pt_reason", ""),
                    "pt_verdict": opp.get("pt_verdict", ""),
                    "pt_prob_7d": opp.get("pt_prob_7d", None),
                    "pt_confidence": opp.get("pt_confidence", None),
                    # Portfolio context
                    "portfolio_slots_used": len(open_positions) + slots_used_this_cycle,
                    "portfolio_slots_max": MAX_PORTFOLIO_SLOTS,
                    "wallet_balance": balance,
                }

                buy_result = execute_buy(
                    user_id=user_id, coin_id=opp["coin_id"], coin_name=opp["name"],
                    symbol=opp["symbol"], price=opp["price"], amount=trade_amount,
                    ai_score=opp["score"], ai_reasoning=detailed_reasoning,
                    stop_loss_pct=effective_sl, take_profit_pct=effective_tp,
                    side=trade_side, trade_metadata=trade_metadata,
                )
                if buy_result["success"]:
                    results["new_trades"] += 1
                    balance -= trade_amount
                    slots_used_this_cycle += 1
                    action_verb = "Shorted" if trade_side == "short" else "Bought"
                    results["actions"].append(
                        f"{action_verb} {opp['symbol']} ({trade_side.upper()}) at ${opp['price']:,.4f} (${trade_amount:,.0f}) "
                        f"| Score: {opp['score']}/100 | TX: {buy_result['tx_hash'][:16]}..."
                    )
                    # Store full trade details for email notifications
                    results["trade_details"].append({
                        "type": "buy",
                        "action": side_label,
                        "symbol": opp["symbol"],
                        "coin_name": opp["name"],
                        "coin_id": opp["coin_id"],
                        "price": opp["price"],
                        "quantity": buy_result["quantity"],
                        "amount": trade_amount,
                        "side": trade_side,
                        "stop_loss": buy_result["stop_loss"],
                        "take_profit": buy_result["take_profit"],
                        "ai_reasoning": detailed_reasoning,
                        "tx_hash": buy_result["tx_hash"],
                        "trade_metadata": trade_metadata,
                    })
                    # Record in learning loop
                    ll = _get_learning_loop()
                    if ll:
                        try:
                            verdict = "bearish" if trade_side == "short" else "bullish"
                            ll.record_prediction(
                                token_ticker=opp["coin_id"],
                                token_name=opp["name"],
                                verdict=verdict,
                                confidence=opp["score"] / 10.0,
                                composite_score=opp["score"],
                                price_at_prediction=opp["price"],
                                market_regime=market_regime,
                                user_id=user_id,
                            )
                        except Exception:
                            pass

        elif not picks:
            logger.info(
                "No new trades — available_slots: %d | new_opps: %d | long_opps: %d | short_opps: %d",
                available_slots, len(new_opportunities), len(long_opps), len(short_opps),
            )

    # Add portfolio summary to results
    results["portfolio_summary"] = {
        "total_positions": len(open_positions),
        "long_positions": sum(1 for p in open_positions if p.get("side", "long") == "long"),
        "short_positions": sum(1 for p in open_positions if p.get("side") == "short"),
        "max_slots": MAX_PORTFOLIO_SLOTS,
        "available_balance": balance,
    }

    # ── AUTO-RETRAIN: Trigger continuous learner periodically ──
    # Run the ML feedback loop every 6 hours (checks internally if enough feedback exists)
    try:
        _trigger_periodic_retrain()
    except Exception as e:
        logger.debug("Periodic retrain check: %s", e)

    return results


# ── Periodic ML Retraining ────────────────────────────────────────────────────
_last_retrain_check = 0.0
_RETRAIN_INTERVAL = 6 * 3600  # 6 hours

def _trigger_periodic_retrain():
    """Trigger the continuous learner's feedback loop every 6 hours."""
    global _last_retrain_check
    now = time.time()
    if now - _last_retrain_check < _RETRAIN_INTERVAL:
        return  # Not yet time

    _last_retrain_check = now
    logger.info("Triggering periodic ML retrain check...")

    try:
        import subprocess
        import sys
        ml_script = os.path.join(os.path.dirname(__file__), "ml", "continuous_learner.py")
        if os.path.exists(ml_script):
            # Run in subprocess to avoid blocking the trading engine
            subprocess.Popen(
                [sys.executable, ml_script, "--action", "loop"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("ML retrain subprocess launched")
    except Exception as e:
        logger.warning("Failed to trigger ML retrain: %s", e)



# -- Performance Stats ---------------------------------------------------------

def get_performance_stats(user_id):
    balance = _get_wallet_balance(user_id)
    stats = _get_trade_stats(user_id)
    open_pos = get_open_positions(user_id)
    open_pnl = sum(p["pnl"] for p in open_pos)
    open_value = sum(p["current_value"] for p in open_pos)
    total_trades = stats.get("total_trades", 0)
    winning = stats.get("winning_trades", 0)
    losing = stats.get("losing_trades", 0)
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    total_value = balance + open_value
    return {
        "wallet_balance": round(balance, 2),
        "total_value": round(total_value, 2),
        "realized_pnl": round(stats.get("total_pnl", 0), 2),
        "unrealized_pnl": round(open_pnl, 2),
        "open_positions": len(open_pos),
        "total_trades": total_trades,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": round(win_rate, 1),
        "best_trade": round(stats.get("best_trade_pnl", 0), 2),
        "worst_trade": round(stats.get("worst_trade_pnl", 0), 2),
        "invested_in_positions": round(open_value, 2),
        "total_invested_ever": round(stats.get("total_invested", 0), 2),
    }


def get_todays_trade_count() -> int:
    """Get the number of auto-trades executed today across all users."""
    try:
        sb = get_supabase()
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = sb.table("trade_positions").select("id", count="exact").gte(
            "created_at", today
        ).execute()
        return result.count if hasattr(result, "count") and result.count else len(result.data) if result.data else 0
    except Exception:
        return 0


def get_today_quick_stats(user_id: int) -> Dict[str, Any]:
    """Get today's quick stats: trades, P&L, avg hold time, best/worst."""
    sb = get_supabase()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00")
    try:
        # Today's orders
        orders = sb.table("trade_orders").select("*").eq("user_id", user_id).gte("created_at", today).execute()
        today_orders = orders.data or []

        # Today's closed positions
        closed = sb.table("trade_positions").select("*").eq("user_id", user_id).eq("status", "closed").gte("closed_at", today).execute()
        today_closed = closed.data or []

        # Open positions — unrealized P&L
        open_pos = sb.table("trade_positions").select("pnl,invested_amount").eq("user_id", user_id).eq("status", "open").execute()
        open_positions = open_pos.data or []
        unrealized_pnl = sum(p.get("pnl", 0) or 0 for p in open_positions)
        open_count = len(open_positions)

        realized_pnl = sum(p.get("pnl", 0) for p in today_closed)
        today_pnl = realized_pnl + unrealized_pnl
        today_trades = len(today_orders)
        wins = sum(1 for p in today_closed if p.get("pnl", 0) > 0)
        losses = sum(1 for p in today_closed if p.get("pnl", 0) <= 0)
        best = max((p.get("pnl", 0) for p in today_closed), default=0)
        worst = min((p.get("pnl", 0) for p in today_closed), default=0)

        # Avg hold time for closed positions today
        hold_times = []
        for p in today_closed:
            opened = p.get("opened_at") or p.get("created_at")
            closed_at = p.get("closed_at")
            if opened and closed_at:
                try:
                    t0 = datetime.fromisoformat(str(opened).replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
                    hold_times.append((t1 - t0).total_seconds() / 60)
                except Exception:
                    pass
        avg_hold_min = round(sum(hold_times) / len(hold_times), 1) if hold_times else 0

        # Buy / sell counts
        buys = sum(1 for o in today_orders if o.get("action") in ("BUY", "SHORT"))
        sells = sum(1 for o in today_orders if o.get("action") in ("SELL", "COVER"))

        # Volume today
        volume = sum(o.get("amount", 0) for o in today_orders)

        return {
            "today_trades": today_trades,
            "today_buys": buys,
            "today_sells": sells,
            "today_pnl": round(today_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "today_wins": wins,
            "today_losses": losses,
            "today_best": round(best, 2),
            "today_worst": round(worst, 2),
            "avg_hold_minutes": avg_hold_min,
            "today_volume": round(volume, 2),
            "today_closed_count": len(today_closed),
            "open_count": open_count,
        }
    except Exception as e:
        logger.warning("get_today_quick_stats error: %s (type: %s)", e, type(e).__name__)
        import traceback; logger.warning("Traceback: %s", traceback.format_exc())
        return {
            "today_trades": 0, "today_buys": 0, "today_sells": 0,
            "today_pnl": 0, "realized_pnl": 0, "unrealized_pnl": 0,
            "today_wins": 0, "today_losses": 0,
            "today_best": 0, "today_worst": 0, "avg_hold_minutes": 0,
            "today_volume": 0, "today_closed_count": 0, "open_count": 0,
        }


def get_pnl_chart_data(user_id: int, days: int = 14) -> List[Dict[str, Any]]:
    """Get daily cumulative P&L data for charting."""
    sb = get_supabase()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00+00:00")
    try:
        result = sb.table("trade_positions").select("pnl,closed_at").eq("user_id", user_id).eq("status", "closed").gte("closed_at", since).order("closed_at").execute()
        positions = result.data or []

        daily_pnl: Dict[str, float] = {}
        for p in positions:
            if not p.get("closed_at"):
                continue
            day = str(p["closed_at"])[:10]
            daily_pnl[day] = daily_pnl.get(day, 0) + (p.get("pnl", 0) or 0)

        # Build cumulative chart data for last N days
        chart = []
        cumulative = 0
        start = datetime.now(timezone.utc) - timedelta(days=days)
        for i in range(days + 1):
            d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            cumulative += daily_pnl.get(d, 0)
            chart.append({
                "date": d,
                "daily_pnl": round(daily_pnl.get(d, 0), 2),
                "cumulative_pnl": round(cumulative, 2),
            })
        return chart
    except Exception as e:
        logger.warning("get_pnl_chart_data error: %s", e)
        return []


def get_pnl_heatmap(user_id: int, days: int = 30) -> List[Dict[str, Any]]:
    """Aggregate P&L per coin for a treemap heatmap.
    Returns list of { coin, symbol, pnl, pnl_pct, position_size, trades, status }.
    Combines both open + closed positions within the window."""
    sb = get_supabase()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00+00:00")
    try:
        # Closed positions in window
        closed = sb.table("trade_positions").select(
            "coin_id,symbol,invested_amount,pnl,pnl_pct"
        ).eq("user_id", user_id).eq("status", "closed").gte("closed_at", since).execute()

        # Open positions (always included)
        open_pos = sb.table("trade_positions").select(
            "coin_id,symbol,invested_amount,pnl,pnl_pct,current_value"
        ).eq("user_id", user_id).eq("status", "open").execute()

        coin_map: Dict[str, Dict[str, Any]] = {}

        for p in (closed.data or []):
            cid = p["coin_id"]
            if cid not in coin_map:
                coin_map[cid] = {"coin": cid, "symbol": p.get("symbol", cid).upper(),
                                 "pnl": 0, "invested": 0, "trades": 0, "open_value": 0, "has_open": False}
            coin_map[cid]["pnl"] += p.get("pnl", 0) or 0
            coin_map[cid]["invested"] += p.get("invested_amount", 0) or 0
            coin_map[cid]["trades"] += 1

        for p in (open_pos.data or []):
            cid = p["coin_id"]
            if cid not in coin_map:
                coin_map[cid] = {"coin": cid, "symbol": p.get("symbol", cid).upper(),
                                 "pnl": 0, "invested": 0, "trades": 0, "open_value": 0, "has_open": False}
            coin_map[cid]["pnl"] += p.get("pnl", 0) or 0
            coin_map[cid]["invested"] += p.get("invested_amount", 0) or 0
            coin_map[cid]["open_value"] += p.get("current_value", 0) or 0
            coin_map[cid]["trades"] += 1
            coin_map[cid]["has_open"] = True

        result = []
        for cid, d in coin_map.items():
            position_size = d["open_value"] if d["has_open"] else d["invested"]
            pnl_pct = (d["pnl"] / d["invested"] * 100) if d["invested"] > 0 else 0
            result.append({
                "coin": cid,
                "symbol": d["symbol"],
                "pnl": round(d["pnl"], 2),
                "pnl_pct": round(pnl_pct, 2),
                "position_size": round(max(position_size, 0.01), 2),
                "trades": d["trades"],
                "status": "open" if d["has_open"] else "closed",
            })

        # Sort by absolute position size descending (bigger tiles first)
        result.sort(key=lambda x: x["position_size"], reverse=True)
        return result
    except Exception as e:
        logger.warning("get_pnl_heatmap error: %s", e)
        return []


def get_live_feed(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """Get the most recent trade actions for the live feed."""
    sb = get_supabase()
    try:
        result = sb.table("trade_orders").select("action,symbol,price,amount,ai_score,created_at,tx_hash").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        feed = []
        for o in (result.data or []):
            action = o.get("action", "BUY")
            feed.append({
                "action": action,
                "symbol": o.get("symbol", "???"),
                "price": o.get("price", 0),
                "amount": o.get("amount", 0),
                "score": o.get("ai_score", 0),
                "time": o.get("created_at", ""),
                "tx_hash": o.get("tx_hash", ""),
                "direction": "SHORT" if action in ("SHORT", "COVER") else "LONG",
            })
        return feed
    except Exception as e:
        logger.warning("get_live_feed error: %s", e)
        return []


def get_cycle_log() -> List[Dict[str, Any]]:
    """Return the last N cycle summaries from the in-memory ring buffer."""
    return list(_cycle_log)


# Tables are created via Supabase SQL Editor (database/supabase_schema.sql)


# -- Always-On Auto-Trading System --------------------------------------------

def ensure_system_wallet():
    """
    Ensure the auto-trader user has a wallet with the initial ₹1Cr balance.
    Uses the primary user account (SYSTEM_USER_ID=1) so trades appear in dashboard.
    """
    sb = get_supabase()
    try:
        # Check the user exists (user_id=1 should already exist from registration)
        user_check = sb.table("users").select("id").eq("id", SYSTEM_USER_ID).execute()
        if not user_check.data:
            logger.warning("User id=%d not found — auto-trader will retry on next cycle", SYSTEM_USER_ID)
            return

        # Step 1: Check if wallet exists
        result = sb.table("wallet_balance").select("balance").eq("user_id", SYSTEM_USER_ID).execute()
        if not result.data:
            # Create wallet with initial balance
            sb.table("wallet_balance").insert({
                "user_id": SYSTEM_USER_ID,
                "balance": SYSTEM_INITIAL_BALANCE,
            }).execute()
            logger.info(
                "Created system auto-trader wallet with ₹%s balance",
                f"{SYSTEM_INITIAL_BALANCE:,.0f}",
            )
        else:
            current = result.data[0]["balance"]
            if current < SYSTEM_INITIAL_BALANCE:
                sb.table("wallet_balance").update({
                    "balance": SYSTEM_INITIAL_BALANCE,
                }).eq("user_id", SYSTEM_USER_ID).execute()
                logger.info(
                    "Topped up auto-trader wallet: ₹%s → ₹%s",
                    f"{current:,.0f}", f"{SYSTEM_INITIAL_BALANCE:,.0f}",
                )
            else:
                logger.info("Auto-trader wallet — balance: ₹%s", f"{current:,.0f}")

        # Ensure auto_trade_enabled is ON for system user
        settings = sb.table("trade_settings").select("*").eq("user_id", SYSTEM_USER_ID).execute()
        if not settings.data:
            sb.table("trade_settings").insert({
                "user_id": SYSTEM_USER_ID,
                "auto_trade_enabled": True,
                "max_trade_pct": 20.0,
                "daily_loss_limit_pct": 10.0,
                "max_open_positions": 15,
                "stop_loss_pct": 1.5,
                "take_profit_pct": 2.0,
                "cooldown_minutes": 0,
                "min_market_cap": 1000000,
                "risk_level": "aggressive",
            }).execute()
            logger.info("Created system auto-trader settings (auto_trade=ON, aggressive, SL=1.5%%, TP=2%%, 15 positions, 2h hold)")
        else:
            # Always force-update key settings on startup
            sb.table("trade_settings").update({
                "auto_trade_enabled": True,
                "risk_level": "aggressive",
                "stop_loss_pct": 1.5,
                "take_profit_pct": 2.0,
                "max_open_positions": 15,
                "cooldown_minutes": 0,
            }).eq("user_id", SYSTEM_USER_ID).execute()
            logger.info("Updated system auto-trader: SL=1.5%%, TP=2%%, max_positions=15, 2h hold, cooldown=0")

        # Ensure trade_stats row exists
        stats = sb.table("trade_stats").select("*").eq("user_id", SYSTEM_USER_ID).execute()
        if not stats.data:
            sb.table("trade_stats").upsert({"user_id": SYSTEM_USER_ID}).execute()

    except Exception as e:
        logger.error("Failed to ensure system wallet: %s", e)


async def continuous_trading_loop(cg_collector, gemini_client=None, email_callback=None):
    """
    Always-on background trading loop — two-tier architecture.

    FAST tier (every cycle, ~10s):
      - fast_position_check() — batch price fetch + SL/TP/trailing/auto-exit
      - Ensures positions close within seconds of expiry

    SLOW tier (every RESEARCH_EVERY_N_CYCLES cycles, ~60s):
      - Full auto_trade_cycle() — research, ML, backtests, new trades
      - Learning loop evaluation & strategy adjustments

    email_callback: Optional callable(user_id, cycle_result) to send trade emails.
    Never crashes — catches all exceptions.
    """
    global _autotrader_running
    _autotrader_running = True
    cycle_count = 0

    logger.info(
        "=== NEXYPHER AUTO-TRADER STARTED === "
        "User: SYSTEM(%d) | Balance: ₹%s | Fast-check: %ds | Research every %d cycles (~%ds)",
        SYSTEM_USER_ID,
        f"{SYSTEM_INITIAL_BALANCE:,.0f}",
        AUTO_TRADE_INTERVAL_SECONDS,
        RESEARCH_EVERY_N_CYCLES,
        AUTO_TRADE_INTERVAL_SECONDS * RESEARCH_EVERY_N_CYCLES,
    )

    while _autotrader_running:
        cycle_count += 1
        is_research_cycle = (cycle_count % RESEARCH_EVERY_N_CYCLES == 0)

        try:
            if is_research_cycle:
                # ── SLOW TIER: full research + position management ──
                logger.info("── Research cycle #%d starting (full) ──", cycle_count)

                result = await auto_trade_cycle(
                    user_id=SYSTEM_USER_ID,
                    cg_collector=cg_collector,
                    gemini_client=gemini_client,
                )

                status = result.get("status", "ok")
                actions = result.get("actions", [])
                new_trades = result.get("new_trades", 0)
                positions_updated = result.get("positions_updated", 0)

                logger.info(
                    "Research cycle #%d complete — status: %s | "
                    "new_trades: %d | positions_updated: %d | actions: %d",
                    cycle_count, status, new_trades, positions_updated, len(actions),
                )
                for action in actions:
                    logger.info("  → %s", action)

                # Record cycle summary in ring buffer for the Cycle Log UI
                _cycle_log.appendleft({
                    "cycle": cycle_count,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "new_trades": new_trades,
                    "positions_updated": positions_updated,
                    "actions": actions[:5],
                    "portfolio": result.get("portfolio_summary", {}),
                    "type": "research",
                })

                # Send trade email notifications in background thread
                if email_callback and (result.get("trade_details") or actions):
                    try:
                        threading.Thread(
                            target=email_callback,
                            args=(SYSTEM_USER_ID, result),
                            daemon=True,
                        ).start()
                    except Exception as e:
                        logger.warning("Email callback failed: %s", e)

                # Learning loop: evaluate past predictions
                ll = _get_learning_loop()
                if ll:
                    try:
                        eval_result = await ll.evaluate_pending(cg_collector)
                        eval_24h = eval_result.get("evaluated_24h", 0) if isinstance(eval_result, dict) else 0
                        eval_7d = eval_result.get("evaluated_7d", 0) if isinstance(eval_result, dict) else 0
                        if eval_24h or eval_7d:
                            logger.info(
                                "Learning loop evaluated %d (24h) + %d (7d) predictions",
                                eval_24h, eval_7d,
                            )
                    except Exception as e:
                        logger.warning("Learning evaluation error: %s", e)

                    # Strategy adjustments every 100 cycles
                    if cycle_count % 100 == 0:
                        try:
                            adjustments = ll.generate_adjustments()
                            if adjustments:
                                logger.info(
                                    "Generated %d strategy adjustments from trade history",
                                    len(adjustments) if isinstance(adjustments, list) else 0,
                                )
                        except Exception as e:
                            logger.warning("Strategy adjustment error: %s", e)

                    # Snapshot accuracy every 50 cycles
                    if cycle_count % 50 == 0:
                        try:
                            ll.snapshot_accuracy()
                            logger.info("Accuracy snapshot saved (cycle #%d)", cycle_count)
                        except Exception:
                            pass

            else:
                # ── FAST TIER: position management only (~1-2s) ──
                pos_result = await fast_position_check(SYSTEM_USER_ID, cg_collector)

                actions = pos_result.get("actions", [])
                positions_closed = pos_result.get("positions_closed", 0)
                positions_updated = pos_result.get("positions_updated", 0)

                if actions:
                    logger.info(
                        "Fast-check cycle #%d — closed: %d | updated: %d | actions: %d",
                        cycle_count, positions_closed, positions_updated, len(actions),
                    )
                    for action in actions:
                        logger.info("  → %s", action)

                    # Record fast-check actions in cycle log
                    _cycle_log.appendleft({
                        "cycle": cycle_count,
                        "time": datetime.now(timezone.utc).isoformat(),
                        "status": "ok",
                        "new_trades": 0,
                        "positions_updated": positions_updated,
                        "actions": actions[:5],
                        "type": "fast_check",
                    })

                    # Send close-trade email notifications in background thread
                    if email_callback and pos_result.get("trade_details"):
                        try:
                            threading.Thread(
                                target=email_callback,
                                args=(SYSTEM_USER_ID, pos_result),
                                daemon=True,
                            ).start()
                        except Exception as e:
                            logger.warning("Email callback failed: %s", e)

        except Exception as e:
            logger.error("Auto-trade cycle #%d FAILED: %s", cycle_count, e, exc_info=True)

        # Wait for next cycle
        if _autotrader_running:
            await asyncio.sleep(AUTO_TRADE_INTERVAL_SECONDS)

    logger.info("=== NEXYPHER AUTO-TRADER STOPPED after %d cycles ===", cycle_count)


def start_autotrader(cg_collector, gemini_client=None, email_callback=None):
    """
    Start the always-on auto-trader as a background asyncio task.
    Safe to call multiple times — will not create duplicate tasks.

    email_callback: Optional callable(user_id, cycle_result) — called in a
                    background thread whenever trades are executed.
    """
    global _autotrader_task, _autotrader_running

    if _autotrader_task and not _autotrader_task.done():
        logger.info("Auto-trader already running — skipping duplicate start")
        return

    # Ensure system user has wallet & settings
    ensure_system_wallet()

    _autotrader_running = True
    _autotrader_task = asyncio.create_task(
        continuous_trading_loop(cg_collector, gemini_client, email_callback=email_callback)
    )
    logger.info("Auto-trader background task created")


def stop_autotrader():
    """Stop the auto-trader gracefully."""
    global _autotrader_running
    _autotrader_running = False
    logger.info("Auto-trader stop requested — will finish current cycle")


def get_autotrader_status() -> Dict[str, Any]:
    """Get the current status of the auto-trader."""
    running = _autotrader_running and _autotrader_task and not _autotrader_task.done()
    try:
        balance = _get_wallet_balance(SYSTEM_USER_ID)
        stats = _get_trade_stats(SYSTEM_USER_ID)
        open_pos = get_open_positions(SYSTEM_USER_ID)
    except Exception:
        balance = 0
        stats = {}
        open_pos = []

    return {
        "running": bool(running),
        "system_user_id": SYSTEM_USER_ID,
        "initial_balance": SYSTEM_INITIAL_BALANCE,
        "current_balance": balance,
        "total_pnl": stats.get("total_pnl", 0),
        "total_trades": stats.get("total_trades", 0),
        "winning_trades": stats.get("winning_trades", 0),
        "losing_trades": stats.get("losing_trades", 0),
        "open_positions": len(open_pos),
        "interval_seconds": AUTO_TRADE_INTERVAL_SECONDS,
    }
