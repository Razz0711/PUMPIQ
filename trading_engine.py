"""
NEXYPHER Autonomous Trading Engine
============================================================
AI-powered autonomous crypto trading bot that:
1. Checks REAL wallet balance (deposited from bank account)
2. Researches market opportunities using CoinGecko + DexScreener + Gemini AI
3. Makes buy/sell decisions with risk management
4. Executes trades using real wallet funds
5. Tracks P&L and performance

Safety Controls:
- Max trade size: 20% of wallet per trade
- Daily loss limit: 10% of wallet
- Max open positions: 5
- Stop-loss: -8% per position
- Take-profit: +20% per position
- Cooldown: 5 min between trades
- Only trades top coins (whitelist or market cap > $1M)
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
        "user_id": user_id, "auto_trade_enabled": 0,
        "max_trade_pct": 20.0, "daily_loss_limit_pct": 10.0,
        "max_open_positions": 5, "stop_loss_pct": 8.0,
        "take_profit_pct": 20.0, "cooldown_minutes": 5,
        "min_market_cap": 1000000, "risk_level": "moderate",
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

def execute_buy(user_id, coin_id, coin_name, symbol, price, amount, ai_score, ai_reasoning, stop_loss_pct=8.0, take_profit_pct=20.0):
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
        stop_loss = price * (1 - stop_loss_pct / 100)
        take_profit = price * (1 + take_profit_pct / 100)

        timestamp = datetime.now(timezone.utc).isoformat()
        tx_hash = generate_tx_hash(user_id, 'BUY', coin_id, symbol.upper(), price, quantity, amount, timestamp)

        # Insert position
        pos_result = sb.table("trade_positions").insert({
            "user_id": user_id,
            "coin_id": coin_id,
            "coin_name": coin_name,
            "symbol": symbol.upper(),
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
            "action": "BUY",
            "price": price,
            "quantity": quantity,
            "amount": amount,
            "ai_score": ai_score,
            "ai_reasoning": ai_reasoning,
            "tx_hash": tx_hash,
            "created_at": timestamp,
        }).execute()

        # Deduct wallet balance
        _deduct_wallet(user_id, amount)

        # Update trade stats
        sb.rpc("increment_trade_stats", {
            "p_user_id": user_id,
            "p_invested": amount,
            "p_trades": 1,
        }).execute()

        _log_event(user_id, "BUY", f"Bought {quantity:.6f} {symbol.upper()} at ${price:,.2f} (${amount:,.0f}) | Score: {ai_score}/100 | Hash: {tx_hash[:16]}... | {ai_reasoning[:180]}")

        # Record on blockchain (async, non-blocking)
        blockchain.record_transaction_async(tx_hash, "BUY", symbol.upper(), amount)

        return {"success": True, "position_id": position_id, "quantity": quantity, "amount": amount, "tx_hash": tx_hash}
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_sell(user_id, position_id, current_price, reason="manual"):
    sb = get_supabase()
    try:
        pos_result = sb.table("trade_positions").select("*").eq("id", position_id).eq("user_id", user_id).eq("status", "open").execute()
        if not pos_result.data:
            return {"success": False, "error": "Position not found or already closed"}
        pos = pos_result.data[0]

        current_value = pos["quantity"] * current_price
        pnl = current_value - pos["invested_amount"]
        pnl_pct = (pnl / pos["invested_amount"]) * 100

        timestamp = datetime.now(timezone.utc).isoformat()
        tx_hash = generate_tx_hash(user_id, 'SELL', pos["coin_id"], pos["symbol"], current_price, pos["quantity"], current_value, timestamp)

        # Close the position
        sb.table("trade_positions").update({
            "status": "closed",
            "current_price": current_price,
            "current_value": current_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "tx_hash": tx_hash,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", position_id).execute()

        # Insert sell order
        sb.table("trade_orders").insert({
            "user_id": user_id,
            "position_id": position_id,
            "coin_id": pos["coin_id"],
            "symbol": pos["symbol"],
            "action": "SELL",
            "price": current_price,
            "quantity": pos["quantity"],
            "amount": current_value,
            "ai_reasoning": reason,
            "tx_hash": tx_hash,
            "created_at": timestamp,
        }).execute()

        # Credit wallet
        _credit_wallet(user_id, current_value, f"Sold {pos['symbol']} — P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")

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

        _log_event(user_id, "SELL", f"Sold {pos['quantity']:.6f} {pos['symbol']} at ${current_price:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%) | Hash: {tx_hash[:16]}... | Reason: {reason}")

        # Record on blockchain (async, non-blocking)
        blockchain.record_transaction_async(tx_hash, "SELL", pos["symbol"], current_value)

        # Feed outcome back to AI learning loop (evaluate prediction)
        ll = _get_learning_loop()
        if ll:
            try:
                ll.evaluate_predictions_with_price(
                    token_ticker=pos["coin_id"],
                    actual_price=current_price,
                )
            except Exception:
                # Fallback: try direct DB update if method doesn't exist
                try:
                    import sqlite3 as _sq
                    _ldb = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "NEXYPHER_learning.db")
                    if os.path.exists(_ldb):
                        _lconn = _sq.connect(_ldb)
                        _lconn.execute("""
                            UPDATE predictions SET
                                actual_price_7d = ?,
                                direction_correct_7d = CASE
                                    WHEN predicted_direction = 'up' AND ? > price_at_prediction THEN 1
                                    WHEN predicted_direction = 'down' AND ? < price_at_prediction THEN 1
                                    ELSE 0 END,
                                pnl_pct_7d = ?,
                                evaluated_7d_at = datetime('now')
                            WHERE token_ticker = ?
                              AND evaluated_7d_at IS NULL
                        """, (current_price, current_price, current_price,
                              round(pnl_pct, 2), pos["coin_id"]))
                        _lconn.commit()
                        _lconn.close()
                except Exception:
                    pass

        return {"success": True, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2), "amount": round(current_value, 2), "tx_hash": tx_hash}
    except Exception as e:
        return {"success": False, "error": str(e)}


# -- AI Research Engine --------------------------------------------------------

async def research_opportunities(cg_collector, dex_collector, gemini_client=None):
    opportunities = []
    try:
        top_coins = await cg_collector.get_top_coins(limit=30)
        trending = await cg_collector.get_trending()
        trending_ids = {t.coin_id for t in trending} if trending else set()

        for coin in top_coins:
            score = 0
            reasons = []
            change_24h = coin.price_change_pct_24h
            if 3 < change_24h < 15:
                score += 25; reasons.append(f"Strong 24h momentum: {change_24h:+.1f}%")
            elif 1 < change_24h <= 3:
                score += 15; reasons.append(f"Positive momentum: {change_24h:+.1f}%")
            elif change_24h < -10:
                score -= 10; reasons.append(f"Heavy decline: {change_24h:+.1f}%")
            if coin.market_cap > 0:
                vol_ratio = coin.total_volume_24h / coin.market_cap
                if vol_ratio > 0.3:
                    score += 20; reasons.append(f"High volume/mcap ratio: {vol_ratio:.2f}")
                elif vol_ratio > 0.1:
                    score += 10; reasons.append(f"Healthy volume: {vol_ratio:.2f}")
            if coin.coin_id in trending_ids:
                score += 15; reasons.append("Trending on CoinGecko")
            if coin.market_cap > 10_000_000_000:
                score += 10; reasons.append("Large cap - lower risk")
            elif coin.market_cap > 1_000_000_000:
                score += 5; reasons.append("Mid cap")
            if hasattr(coin, 'ath') and coin.ath > 0:
                ath_ratio = coin.current_price / coin.ath
                if 0.3 < ath_ratio < 0.7:
                    score += 10; reasons.append(f"Room to grow - {(1-ath_ratio)*100:.0f}% below ATH")
            score = max(0, min(100, score))
            if score >= 25:
                opportunities.append({
                    "coin_id": coin.coin_id, "name": coin.name, "symbol": coin.symbol.upper(),
                    "price": coin.current_price, "change_24h": change_24h,
                    "market_cap": coin.market_cap, "volume_24h": coin.total_volume_24h,
                    "score": score, "reasons": reasons, "reasoning": " | ".join(reasons), "source": "coingecko",
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
            opp["score"] = min(100, opp["score"] + 10)
            opp["reasons"].append(
                f"✅ Backtest verified ({direction}): {bt_result.win_rate:.0f}% win rate, "
                f"{bt_result.total_return:.1f}% return, Sharpe {bt_result.sharpe_ratio:.2f}"
            )
            opp["reasoning"] = " | ".join(opp["reasons"])
        else:
            # Penalize and flag
            tested = opp.get("backtest_strategies_tested", [])
            opp["score"] = max(0, opp["score"] - 20)
            opp["reasons"].append(
                f"⚠️ Backtest WARNING (tested {', '.join(tested) if tested else 'LONG'}): "
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
        confidence_threshold = 7.0
        max_daily = 10
        risk_profile = "balanced"

    # Risk profile modifiers
    risk_modifiers = {
        "conservative": {"max_trade_pct_mult": 0.5, "min_score": 70, "stop_loss_add": 2},
        "balanced":     {"max_trade_pct_mult": 1.0, "min_score": 50, "stop_loss_add": 0},
        "aggressive":   {"max_trade_pct_mult": 1.5, "min_score": 30, "stop_loss_add": -2},
    }
    mods = risk_modifiers.get(risk_profile, risk_modifiers["balanced"])

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

    last_trade = sb.table("trade_orders").select("created_at").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
    if last_trade.data:
        last_time = datetime.fromisoformat(last_trade.data[0]["created_at"].replace("Z", "+00:00"))
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        cooldown = timedelta(minutes=settings["cooldown_minutes"])
        if datetime.now(timezone.utc) - last_time < cooldown:
            return {"status": "cooldown", "message": f"Cooldown active ({settings['cooldown_minutes']}min)"}

    open_positions = get_open_positions(user_id)
    for pos in open_positions:
        try:
            if pos["coin_id"] and not pos["coin_id"].startswith("0x"):
                prices = await cg_collector.get_simple_price([pos["coin_id"]])
                current_price = prices.get(pos["coin_id"], pos["current_price"])
            else:
                current_price = pos["current_price"]
            current_value = pos["quantity"] * current_price
            pnl = current_value - pos["invested_amount"]
            pnl_pct = (pnl / pos["invested_amount"]) * 100
            sb.table("trade_positions").update({
                "current_price": current_price,
                "current_value": current_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }).eq("id", pos["id"]).execute()
            results["positions_updated"] += 1
            if pnl_pct <= -settings["stop_loss_pct"]:
                sell_reason = (f"STOP-LOSS TRIGGERED: {pos['symbol']} dropped {pnl_pct:.1f}% from entry price ${pos['entry_price']:.2f} to ${current_price:.2f}. "
                               f"Loss of ${abs(pnl):.2f} on ${pos['invested_amount']:.2f} invested. Selling to prevent further losses.")
                sell_result = execute_sell(user_id, pos["id"], current_price, sell_reason)
                if sell_result["success"]:
                    results["actions"].append(f"Stop-loss: Sold {pos['symbol']} at ${current_price:,.2f} (P&L: {pnl_pct:+.1f}%) \u2014 TX: {sell_result['tx_hash'][:16]}...")
                continue
            if pnl_pct >= settings["take_profit_pct"]:
                sell_reason = (f"TAKE-PROFIT TRIGGERED: {pos['symbol']} gained {pnl_pct:.1f}% from entry ${pos['entry_price']:.2f} to ${current_price:.2f}. "
                               f"Profit of ${pnl:.2f} on ${pos['invested_amount']:.2f} invested. Locking in profits at target.")
                sell_result = execute_sell(user_id, pos["id"], current_price, sell_reason)
                if sell_result["success"]:
                    results["actions"].append(f"Take-profit: Sold {pos['symbol']} at ${current_price:,.2f} (P&L: {pnl_pct:+.1f}%) \u2014 TX: {sell_result['tx_hash'][:16]}...")
                continue
        except Exception as e:
            logger.warning("Position update failed for %s: %s", pos["coin_id"], e)

    balance = _get_wallet_balance(user_id)
    open_count = len([p for p in get_open_positions(user_id) if p["status"] == "open"])

    # Check daily trade count limit
    today_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00")
    today_orders = sb.table("trade_orders").select("id").eq("user_id", user_id).eq("action", "BUY").gte("created_at", today_start).execute()
    trades_today = len(today_orders.data) if today_orders.data else 0

    if open_count < settings["max_open_positions"] and balance > 100 and trades_today < max_daily:
        opportunities = await research_opportunities(cg_collector, dex_collector, gemini_client)
        held_coins = {p["coin_id"] for p in get_open_positions(user_id)}
        opportunities = [o for o in opportunities if o["coin_id"] not in held_coins]
        opportunities = [o for o in opportunities if o["market_cap"] >= settings["min_market_cap"]]

        # ── MANDATORY BACKTEST GATE ──
        # Only allow tokens that passed backtest verification
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

        # Apply confidence threshold — convert score (0-100) to confidence (0-10)
        min_score = mods["min_score"]
        confidence_min_score = int(confidence_threshold * 10)  # e.g. 7.0 → 70
        effective_min_score = max(min_score, confidence_min_score)
        opportunities = [o for o in opportunities if o["score"] >= effective_min_score]
        results["confidence_threshold"] = confidence_threshold
        results["effective_min_score"] = effective_min_score
        results["risk_profile"] = risk_profile

        max_buys = min(3, max_daily - trades_today)
        for opp in opportunities[:max_buys]:
            if open_count >= settings["max_open_positions"]:
                break
            # Apply risk-profile trade size modifier
            effective_max_pct = settings["max_trade_pct"] * mods["max_trade_pct_mult"]
            max_trade = balance * (effective_max_pct / 100)
            trade_amount = min(max_trade, balance * 0.15)
            if trade_amount < 50:
                break

            effective_sl = max(1.0, settings["stop_loss_pct"] + mods["stop_loss_add"])

            # Build detailed AI reasoning for the buy (includes backtest verification)
            bt_stats = opp.get("backtest_stats", {})
            bt_direction = opp.get("backtest_strategy_direction", "LONG")
            bt_trend = opp.get("backtest_detected_trend", "unknown")
            bt_summary = ""
            if bt_stats:
                bt_summary = (
                    f"BACKTEST VERIFIED ({bt_direction} strategy, {bt_trend} trend): "
                    f"{bt_stats.get('win_rate', 0):.1f}% win rate, "
                    f"{bt_stats.get('total_return', 0):.1f}% return, "
                    f"{bt_stats.get('max_drawdown', 0):.1f}% max DD, "
                    f"Sharpe {bt_stats.get('sharpe_ratio', 0):.2f}, "
                    f"{bt_stats.get('total_trades', 0)} trades over "
                    f"{bt_stats.get('days_covered', 0)} days. "
                )

            detailed_reasoning = (
                f"BUY SIGNAL for {opp['name']} ({opp['symbol']}): "
                f"Price ${opp['price']:,.6f} | 24h: {opp['change_24h']:+.1f}% | "
                f"Market Cap: ${opp['market_cap']:,.0f} | Volume: ${opp['volume_24h']:,.0f} | "
                f"Score: {opp['score']}/100 (threshold: {effective_min_score}). "
                f"{bt_summary}"
                f"Risk profile: {risk_profile} | Confidence gate: {confidence_threshold}/10. "
                f"Analysis: {opp['reasoning']}. "
                f"Investing ${trade_amount:,.2f} ({trade_amount/balance*100:.1f}% of wallet) with "
                f"stop-loss at -{effective_sl}% and take-profit at +{settings['take_profit_pct']}%."
            )
            if opp.get("ai_analysis"):
                detailed_reasoning += f" AI Insight: {opp['ai_analysis'][:200]}"

            buy_result = execute_buy(
                user_id=user_id, coin_id=opp["coin_id"], coin_name=opp["name"],
                symbol=opp["symbol"], price=opp["price"], amount=trade_amount,
                ai_score=opp["score"], ai_reasoning=detailed_reasoning,
                stop_loss_pct=effective_sl, take_profit_pct=settings["take_profit_pct"],
            )
            if buy_result["success"]:
                results["new_trades"] += 1
                open_count += 1
                balance -= trade_amount
                results["actions"].append(
                    f"Bought {opp['symbol']} at ${opp['price']:,.2f} (${trade_amount:,.0f}) "
                    f"| Score: {opp['score']}/100 | Strategy: {bt_direction} "
                    f"| Profile: {risk_profile} "
                    f"| TX: {buy_result['tx_hash'][:16]}..."
                )

                # Record in learning loop
                ll = _get_learning_loop()
                if ll:
                    try:
                        ll.record_prediction(
                            token_ticker=opp["coin_id"],
                            token_name=opp["name"],
                            verdict=bt_direction or "bullish",
                            confidence=opp["score"] / 10.0,
                            composite_score=opp["score"],
                            price_at_prediction=opp["price"],
                            market_regime="unknown",
                        )
                    except Exception:
                        pass

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


# Tables are created via Supabase SQL Editor (database/supabase_schema.sql)
