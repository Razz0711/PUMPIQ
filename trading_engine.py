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

logger = logging.getLogger(__name__)

# ── System auto-trader (always-on background bot) ──
SYSTEM_USER_ID = 1  # trades under the primary user account so dashboard shows activity
SYSTEM_INITIAL_BALANCE = 10_000_000.0  # ₹1,00,00,000 (1 Cr)
AUTO_TRADE_INTERVAL_SECONDS = 30   # 30 seconds — continuous market scanning
MAX_HOLD_HOURS = 1                  # auto-sell if TP/SL not hit within 1 hour
WARNING_MINUTES_BEFORE_CLOSE = 5    # SMS warning 5 min before auto-close
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
        "max_open_positions": 15, "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0, "cooldown_minutes": 0,
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
        "stop_loss_pct": settings.get("stop_loss_pct", 8.0),
        "take_profit_pct": settings.get("take_profit_pct", 20.0),
        "cooldown_minutes": settings.get("cooldown_minutes", 5),
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


def execute_buy(user_id, coin_id, coin_name, symbol, price, amount, ai_score, ai_reasoning, stop_loss_pct=8.0, take_profit_pct=20.0, side="long"):
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

        return {"success": True, "position_id": position_id, "quantity": quantity, "amount": amount, "tx_hash": tx_hash, "side": side}
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
                ll.evaluate_trade_close(
                    token_ticker=pos["coin_id"],
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                )
            except Exception:
                pass

        return {"success": True, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2), "amount": round(max(0, current_value), 2), "tx_hash": tx_hash, "side": side}
    except Exception as e:
        return {"success": False, "error": str(e)}


# -- AI Research Engine --------------------------------------------------------

async def research_opportunities(cg_collector, dex_collector, gemini_client=None):
    """Research and score trading opportunities (LONG and SHORT signals)."""
    opportunities = []
    try:
        top_coins = await cg_collector.get_top_coins(limit=50)  # Increased from 30 to 50 for more diversity
        trending = await cg_collector.get_trending()
        trending_ids = {t.coin_id for t in trending} if trending else set()

        for coin in top_coins:
            # Skip stablecoins — they don't move
            if coin.coin_id in STABLECOINS:
                continue
            # Skip coins with invalid prices
            if coin.current_price <= 0.001:
                continue

            change_24h = coin.price_change_pct_24h

            # ── LONG SIGNAL SCORING (widened for aggressive trading) ──
            long_score = 0
            long_reasons = []
            # Momentum scoring (broader bands)
            if change_24h > 10:
                long_score += 30; long_reasons.append(f"Explosive momentum: {change_24h:+.1f}%")
            elif change_24h > 3:
                long_score += 25; long_reasons.append(f"Strong 24h momentum: {change_24h:+.1f}%")
            elif change_24h > 0.5:
                long_score += 15; long_reasons.append(f"Positive momentum: {change_24h:+.1f}%")
            elif change_24h > -1:
                long_score += 8; long_reasons.append(f"Stable/flat: {change_24h:+.1f}%")
            # Volume analysis
            if coin.market_cap > 0:
                vol_ratio = coin.total_volume_24h / coin.market_cap
                if vol_ratio > 0.3:
                    long_score += 20; long_reasons.append(f"High volume/mcap ratio: {vol_ratio:.2f}")
                elif vol_ratio > 0.1:
                    long_score += 12; long_reasons.append(f"Healthy volume: {vol_ratio:.2f}")
                elif vol_ratio > 0.05:
                    long_score += 6; long_reasons.append(f"Normal volume: {vol_ratio:.2f}")
            # Trending
            if coin.coin_id in trending_ids:
                long_score += 15; long_reasons.append("Trending on CoinGecko")
            # Market cap tiers
            if coin.market_cap > 50_000_000_000:
                long_score += 12; long_reasons.append("Mega cap - liquid & safe")
            elif coin.market_cap > 10_000_000_000:
                long_score += 10; long_reasons.append("Large cap - lower risk")
            elif coin.market_cap > 1_000_000_000:
                long_score += 8; long_reasons.append("Mid cap")
            elif coin.market_cap > 100_000_000:
                long_score += 5; long_reasons.append("Small cap - higher potential")
            # ATH analysis
            if hasattr(coin, 'ath') and coin.ath > 0:
                ath_ratio = coin.current_price / coin.ath
                if 0.2 < ath_ratio < 0.6:
                    long_score += 12; long_reasons.append(f"Recovery potential - {(1-ath_ratio)*100:.0f}% below ATH")
                elif 0.6 <= ath_ratio < 0.85:
                    long_score += 6; long_reasons.append(f"Near ATH recovery zone")
            long_score = max(0, min(100, long_score))

            if long_score >= 15:
                opportunities.append({
                    "coin_id": coin.coin_id, "name": coin.name, "symbol": coin.symbol.upper(),
                    "price": coin.current_price, "change_24h": change_24h,
                    "market_cap": coin.market_cap, "volume_24h": coin.total_volume_24h,
                    "score": long_score, "reasons": long_reasons, "reasoning": " | ".join(long_reasons),
                    "source": "coingecko", "side": "long",
                })

            # ── SHORT SIGNAL SCORING (widened for aggressive trading) ──
            short_score = 0
            short_reasons = []
            # Bearish momentum (broader bands)
            if change_24h < -8:
                short_score += 30; short_reasons.append(f"Strong dump: {change_24h:+.1f}%")
            elif change_24h < -3:
                short_score += 25; short_reasons.append(f"Bearish momentum: {change_24h:+.1f}%")
            elif change_24h < -1:
                short_score += 15; short_reasons.append(f"Declining: {change_24h:+.1f}%")
            elif change_24h < 0:
                short_score += 8; short_reasons.append(f"Slightly bearish: {change_24h:+.1f}%")
            # Volume in decline
            if coin.market_cap > 0:
                vol_ratio = coin.total_volume_24h / coin.market_cap
                if vol_ratio > 0.3:
                    short_score += 15; short_reasons.append(f"Panic volume: {vol_ratio:.2f}")
                elif vol_ratio > 0.15 and change_24h < -1:
                    short_score += 8; short_reasons.append(f"Elevated sell volume")
            # ATH analysis — near ATH = likely pullback
            if hasattr(coin, 'ath') and coin.ath > 0:
                ath_ratio = coin.current_price / coin.ath
                if ath_ratio > 0.92:
                    short_score += 18; short_reasons.append(f"Near ATH ({ath_ratio*100:.0f}%) — pullback likely")
                elif ath_ratio > 0.8:
                    short_score += 10; short_reasons.append(f"Close to ATH ({ath_ratio*100:.0f}%) — resistance zone")
            # Large cap decline
            if change_24h < -1 and coin.market_cap > 1_000_000_000:
                short_score += 10; short_reasons.append("Large cap decline — shorting opportunity")
            short_score = max(0, min(100, short_score))

            if short_score >= 15:
                opportunities.append({
                    "coin_id": coin.coin_id, "name": coin.name, "symbol": coin.symbol.upper(),
                    "price": coin.current_price, "change_24h": change_24h,
                    "market_cap": coin.market_cap, "volume_24h": coin.total_volume_24h,
                    "score": short_score, "reasons": short_reasons, "reasoning": " | ".join(short_reasons),
                    "source": "coingecko", "side": "short",
                })
    except Exception as e:
        logger.warning("CoinGecko research failed: %s", e)

    try:
        for term in ["SOL", "ETH", "PEPE"]:
            pairs = await dex_collector.search_pairs(term)
            for p in (pairs or [])[:10]:
                buys = p.txns_buys_24h
                sells = p.txns_sells_24h
                bsr = buys / max(sells, 1)
                score = 0; reasons = []
                if p.volume_24h > 50000:
                    score += 20; reasons.append(f"Volume: ${p.volume_24h:,.0f}")
                if p.liquidity_usd > 50000:
                    score += 15; reasons.append(f"Liquidity: ${p.liquidity_usd:,.0f}")
                if 1.5 < bsr < 5:
                    score += 15; reasons.append(f"Buy pressure: {bsr:.1f}x")
                if 2 < p.price_change_24h < 30:
                    score += 20; reasons.append(f"Price up {p.price_change_24h:+.1f}%")
                if p.market_cap > 1000000:
                    score += 10
                score = max(0, min(100, score))
                if score >= 30:
                    opportunities.append({
                        "coin_id": p.base_token_address, "name": p.base_token_name,
                        "symbol": p.base_token_symbol, "price": p.price_usd,
                        "change_24h": p.price_change_24h, "market_cap": p.market_cap,
                        "volume_24h": p.volume_24h, "score": score, "reasons": reasons,
                        "reasoning": " | ".join(reasons), "source": "dexscreener",
                    })
    except Exception as e:
        logger.warning("DexScreener research failed: %s", e)

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
                for opp in top5:
                    opp["ai_analysis"] = resp.content
                    # Also enhance the reasoning field with AI insight
                    symbol = opp["symbol"].upper()
                    for line in resp.content.split("\n"):
                        if symbol in line.upper():
                            opp["reasoning"] = line.strip()[:300]
                            break
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
            # Mild penalty — don't kill opportunities in aggressive mode
            tested = opp.get("backtest_strategies_tested", [])
            opp["score"] = max(0, opp["score"] - 5)
            opp["reasons"].append(
                f"⚠️ Backtest caution (tested {', '.join(tested) if tested else 'LONG'}): "
                f"{'; '.join(bt_result.failure_reasons)}"
            )
            opp["reasoning"] = " | ".join(opp["reasons"])

        backtested.append(opp)

    backtested.sort(key=lambda x: x["score"], reverse=True)
    return backtested


# -- Autonomous Trading Loop ---------------------------------------------------

async def auto_trade_cycle(user_id, cg_collector, dex_collector, gemini_client=None):
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
        max_daily = 100
        risk_profile = "aggressive"

    # Override max_daily for aggressive mode (never stop trading due to daily limit)
    if settings.get("risk_level") == "aggressive":
        max_daily = max(max_daily, 200)

    # Trade settings risk_level overrides user prefs (auto-trader sets this to "aggressive")
    risk_profile = settings.get("risk_level", risk_profile)

    # Risk profile modifiers
    risk_modifiers = {
        "conservative": {"max_trade_pct_mult": 0.5, "min_score": 70, "stop_loss_add": 2},
        "moderate":     {"max_trade_pct_mult": 1.0, "min_score": 50, "stop_loss_add": 0},
        "balanced":     {"max_trade_pct_mult": 1.0, "min_score": 50, "stop_loss_add": 0},
        "aggressive":   {"max_trade_pct_mult": 1.5, "min_score": 15, "stop_loss_add": -2},
    }
    mods = risk_modifiers.get(risk_profile, risk_modifiers["aggressive"])

    balance = _get_wallet_balance(user_id)
    stats = _get_trade_stats(user_id)
    total_pnl = stats.get("total_pnl", 0)
    results = {"actions": [], "positions_updated": 0, "new_trades": 0}

    sb = get_supabase()

    if balance > 0:
        loss_pct = (total_pnl / balance) * 100 if total_pnl < 0 else 0
        if loss_pct < -settings["daily_loss_limit_pct"]:
            _log_event(user_id, "SAFETY", f"Daily loss limit hit: {loss_pct:.1f}%")
            return {"status": "paused", "message": f"Daily loss limit reached ({loss_pct:.1f}%)"}

    # (No cooldown for always-on all-in bot — we sell and rebuy every cycle)

    # ── STEP 1: Update all open positions with live prices & check stop/take/auto-exit ──
    open_positions = get_open_positions(user_id)
    for pos in open_positions:
        try:
            coin_id = pos.get("coin_id", "")
            side = pos.get("side", "long")
            if coin_id and not coin_id.startswith("0x") and len(coin_id) <= 20:
                try:
                    prices = await cg_collector.get_simple_price([coin_id])
                    current_price = prices.get(coin_id, pos["current_price"])
                except Exception:
                    current_price = pos["current_price"]
            else:
                current_price = pos["current_price"]

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

            # Check stop-loss (works for both LONG and SHORT)
            if pnl_pct <= -settings["stop_loss_pct"]:
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
                    ll = _get_learning_loop()
                    if ll:
                        try: ll.evaluate_trade_close(pos["coin_id"], current_price, pnl_pct)
                        except Exception: pass
                continue

            # Check take-profit
            if pnl_pct >= settings["take_profit_pct"]:
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
                    ll = _get_learning_loop()
                    if ll:
                        try: ll.evaluate_trade_close(pos["coin_id"], current_price, pnl_pct)
                        except Exception: pass
                continue

            # Check auto-exit (max hold time = 1 hour) + 5-minute SMS warning
            opened_at_str = pos.get("opened_at") or pos.get("created_at", "")
            if opened_at_str:
                try:
                    opened_at = datetime.fromisoformat(opened_at_str.replace("Z", "+00:00"))
                    if opened_at.tzinfo is None:
                        opened_at = opened_at.replace(tzinfo=timezone.utc)
                    seconds_held = (datetime.now(timezone.utc) - opened_at).total_seconds()
                    hours_held = seconds_held / 3600
                    minutes_remaining = (MAX_HOLD_HOURS * 60) - (seconds_held / 60)

                    # ── 5-minute SMS warning before auto-close ──
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

                    # ── FORCE EXIT after 1 hour ──
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
                            # Send SMS notification for auto-close
                            _send_trade_sms(
                                user_id,
                                f"🔴 AUTO-CLOSED {pos['symbol']} ({side.upper()}) after {hours_held:.1f}h | "
                                f"P&L: {pnl_pct:+.1f}% (${pnl:,.0f})"
                            )
                            ll = _get_learning_loop()
                            if ll:
                                try: ll.evaluate_trade_close(pos["coin_id"], current_price, pnl_pct)
                                except Exception: pass
                        continue
                except Exception:
                    pass

        except Exception as e:
            logger.warning("Position update failed for %s: %s", pos.get("coin_id"), e)

    # ── STEP 2: DIVERSIFIED PORTFOLIO — open new LONG and SHORT positions ──
    # Re-fetch balance and open positions after step 1 (some may have closed)
    balance = _get_wallet_balance(user_id)
    open_positions = get_open_positions(user_id)

    # Check daily trade count limit
    today_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00")
    today_orders = sb.table("trade_orders").select("id").eq("user_id", user_id).gte("created_at", today_start).execute()
    trades_today = len(today_orders.data) if today_orders.data else 0

    logger.info(
        "Trade check — trades_today: %d | max_daily: %d | balance: %s | open: %d/%d",
        trades_today, max_daily, f"${balance:,.0f}", len(open_positions), MAX_PORTFOLIO_SLOTS,
    )
    if trades_today < max_daily:
        opportunities = await research_opportunities(cg_collector, dex_collector, gemini_client)

        # Filter stablecoins, invalid prices, and low market cap
        opportunities = [o for o in opportunities if o.get("price", 0) > 0.001]
        opportunities = [o for o in opportunities if o["market_cap"] >= settings["min_market_cap"]]
        logger.info(
            "Research returned %d opportunities (after price/mcap filter)",
            len(opportunities),
        )

        # ── BACKTEST GATE ──
        # Aggressive: allow all coins (skip backtest gate for maximum trading)
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
            logger.info("Aggressive mode — backtest gate BYPASSED, %d coins eligible", len(opportunities))

        # Apply confidence threshold
        min_score = mods["min_score"]
        confidence_min_score = int(confidence_threshold * 10)
        effective_min_score = min_score if risk_profile == "aggressive" else max(min_score, confidence_min_score)
        pre_filter_count = len(opportunities)
        opportunities = [o for o in opportunities if o["score"] >= effective_min_score]
        logger.info(
            "Score filter — min_score: %d | before: %d | after: %d | top_scores: %s",
            effective_min_score, pre_filter_count, len(opportunities),
            [o['score'] for o in sorted(opportunities, key=lambda x: x['score'], reverse=True)[:5]],
        )
        results["confidence_threshold"] = confidence_threshold
        results["effective_min_score"] = effective_min_score
        results["risk_profile"] = risk_profile

        # ── DIVERSIFIED PORTFOLIO: 10-15 positions, balanced LONG/SHORT ──

        # Filter out coins we already hold (same coin + same side)
        current_positions_key = {(p["coin_id"], p.get("side", "long")) for p in open_positions}
        new_opportunities = [o for o in opportunities if (o["coin_id"], o.get("side", "long")) not in current_positions_key]

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

        # Balance LONG/SHORT: aim for ~60% LONG, ~40% SHORT
        target_longs = int(MAX_PORTFOLIO_SLOTS * 0.6)
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
            # ── Score-weighted allocation ──
            investable = balance - RESERVE_AMOUNT
            total_score = sum(max(o["score"], 1) for o, _ in picks)
            effective_sl = max(1.0, settings["stop_loss_pct"] + mods["stop_loss_add"])

            for opp, trade_side in picks:
                if balance - RESERVE_AMOUNT < MIN_POSITION_AMOUNT:
                    break

                # Allocate proportional to score, floor at MIN_POSITION_AMOUNT
                weight = max(opp["score"], 1) / total_score
                trade_amount = max(investable * weight, MIN_POSITION_AMOUNT)
                trade_amount = min(trade_amount, balance - RESERVE_AMOUNT)

                # Enforce max position % (no single token > 15% of portfolio)
                max_allowed = total_portfolio * (MAX_POSITION_PCT / 100)
                trade_amount = min(trade_amount, max_allowed)

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
                    f"Investing ${trade_amount:,.0f} | SL: -{effective_sl}% | TP: +{settings['take_profit_pct']}%."
                )
                if opp.get("ai_analysis"):
                    detailed_reasoning += f" {opp['ai_analysis'][:150]}"

                buy_result = execute_buy(
                    user_id=user_id, coin_id=opp["coin_id"], coin_name=opp["name"],
                    symbol=opp["symbol"], price=opp["price"], amount=trade_amount,
                    ai_score=opp["score"], ai_reasoning=detailed_reasoning,
                    stop_loss_pct=effective_sl, take_profit_pct=settings["take_profit_pct"],
                    side=trade_side,
                )
                if buy_result["success"]:
                    results["new_trades"] += 1
                    balance -= trade_amount
                    action_verb = "Shorted" if trade_side == "short" else "Bought"
                    results["actions"].append(
                        f"{action_verb} {opp['symbol']} ({trade_side.upper()}) at ${opp['price']:,.4f} (${trade_amount:,.0f}) "
                        f"| Score: {opp['score']}/100 | TX: {buy_result['tx_hash'][:16]}..."
                    )
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
                                market_regime="unknown",
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

    return results



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

        today_pnl = sum(p.get("pnl", 0) for p in today_closed)
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
            "today_wins": wins,
            "today_losses": losses,
            "today_best": round(best, 2),
            "today_worst": round(worst, 2),
            "avg_hold_minutes": avg_hold_min,
            "today_volume": round(volume, 2),
            "today_closed_count": len(today_closed),
        }
    except Exception as e:
        logger.warning("get_today_quick_stats error: %s", e)
        return {
            "today_trades": 0, "today_buys": 0, "today_sells": 0,
            "today_pnl": 0, "today_wins": 0, "today_losses": 0,
            "today_best": 0, "today_worst": 0, "avg_hold_minutes": 0,
            "today_volume": 0, "today_closed_count": 0,
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
                "stop_loss_pct": 3.0,
                "take_profit_pct": 5.0,
                "cooldown_minutes": 0,
                "min_market_cap": 1000000,
                "risk_level": "aggressive",
            }).execute()
            logger.info("Created system auto-trader settings (auto_trade=ON, aggressive, SL=3%%, TP=5%%, 15 positions)")
        else:
            # Always force-update key settings on startup
            sb.table("trade_settings").update({
                "auto_trade_enabled": True,
                "risk_level": "aggressive",
                "stop_loss_pct": 3.0,
                "take_profit_pct": 5.0,
                "max_open_positions": 15,
                "cooldown_minutes": 0,
            }).eq("user_id", SYSTEM_USER_ID).execute()
            logger.info("Updated system auto-trader: SL=3%%, TP=5%%, max_positions=15, cooldown=0")

        # Ensure trade_stats row exists
        stats = sb.table("trade_stats").select("*").eq("user_id", SYSTEM_USER_ID).execute()
        if not stats.data:
            sb.table("trade_stats").upsert({"user_id": SYSTEM_USER_ID}).execute()

    except Exception as e:
        logger.error("Failed to ensure system wallet: %s", e)


async def continuous_trading_loop(cg_collector, dex_collector, gemini_client=None):
    """
    Always-on background trading loop.

    Runs every AUTO_TRADE_INTERVAL_SECONDS:
      1. Calls auto_trade_cycle() to research + buy/sell
      2. Evaluates past predictions via LearningLoop
      3. Generates strategy adjustments from outcomes
      4. Never crashes — catches all exceptions
    """
    global _autotrader_running
    _autotrader_running = True
    cycle_count = 0

    logger.info(
        "=== NEXYPHER AUTO-TRADER STARTED === "
        "User: SYSTEM(%d) | Balance: ₹%s | Interval: %ds",
        SYSTEM_USER_ID,
        f"{SYSTEM_INITIAL_BALANCE:,.0f}",
        AUTO_TRADE_INTERVAL_SECONDS,
    )

    while _autotrader_running:
        cycle_count += 1
        try:
            logger.info("── Auto-trade cycle #%d starting ──", cycle_count)

            # 1. Run the main trading cycle (research → buy/sell)
            result = await auto_trade_cycle(
                user_id=SYSTEM_USER_ID,
                cg_collector=cg_collector,
                dex_collector=dex_collector,
                gemini_client=gemini_client,
            )

            status = result.get("status", "ok")
            actions = result.get("actions", [])
            new_trades = result.get("new_trades", 0)
            positions_updated = result.get("positions_updated", 0)

            logger.info(
                "Auto-trade cycle #%d complete — status: %s | "
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
                "actions": actions[:5],  # keep max 5 action strings per entry
                "portfolio": result.get("portfolio_summary", {}),
            })

            # 2. Evaluate past predictions (learning from mistakes)
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

                # 3. Generate strategy adjustments every 10 cycles
                if cycle_count % 10 == 0:
                    try:
                        adjustments = ll.generate_adjustments()
                        if adjustments:
                            logger.info(
                                "Generated %d strategy adjustments from trade history",
                                len(adjustments) if isinstance(adjustments, list) else 0,
                            )
                    except Exception as e:
                        logger.warning("Strategy adjustment error: %s", e)

                # 4. Snapshot accuracy every 50 cycles
                if cycle_count % 50 == 0:
                    try:
                        ll.snapshot_accuracy()
                        logger.info("Accuracy snapshot saved (cycle #%d)", cycle_count)
                    except Exception:
                        pass

        except Exception as e:
            logger.error("Auto-trade cycle #%d FAILED: %s", cycle_count, e, exc_info=True)

        # Wait for next cycle
        if _autotrader_running:
            await asyncio.sleep(AUTO_TRADE_INTERVAL_SECONDS)

    logger.info("=== NEXYPHER AUTO-TRADER STOPPED after %d cycles ===", cycle_count)


def start_autotrader(cg_collector, dex_collector, gemini_client=None):
    """
    Start the always-on auto-trader as a background asyncio task.
    Safe to call multiple times — will not create duplicate tasks.
    """
    global _autotrader_task, _autotrader_running

    if _autotrader_task and not _autotrader_task.done():
        logger.info("Auto-trader already running — skipping duplicate start")
        return

    # Ensure system user has wallet & settings
    ensure_system_wallet()

    _autotrader_running = True
    _autotrader_task = asyncio.create_task(
        continuous_trading_loop(cg_collector, dex_collector, gemini_client)
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
