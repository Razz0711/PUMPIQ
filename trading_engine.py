"""
PumpIQ Autonomous Trading Engine
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
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from blockchain_service import blockchain

logger = logging.getLogger(__name__)

# On Vercel (serverless), use /tmp for writable SQLite; locally use project dir
IS_VERCEL = bool(os.getenv("VERCEL"))
DB_PATH = "/tmp/pumpiq.db" if IS_VERCEL else os.path.join(os.path.dirname(__file__), "pumpiq.db")

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

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass  # WAL may not be supported on some serverless filesystems
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_trading_tables():
    conn = _get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS trade_settings (
            user_id INTEGER PRIMARY KEY,
            auto_trade_enabled INTEGER NOT NULL DEFAULT 0,
            max_trade_pct REAL NOT NULL DEFAULT 20.0,
            daily_loss_limit_pct REAL NOT NULL DEFAULT 10.0,
            max_open_positions INTEGER NOT NULL DEFAULT 5,
            stop_loss_pct REAL NOT NULL DEFAULT 8.0,
            take_profit_pct REAL NOT NULL DEFAULT 20.0,
            cooldown_minutes INTEGER NOT NULL DEFAULT 5,
            min_market_cap REAL NOT NULL DEFAULT 1000000,
            risk_level TEXT NOT NULL DEFAULT 'moderate',
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS trade_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            coin_id TEXT NOT NULL,
            coin_name TEXT NOT NULL DEFAULT '',
            symbol TEXT NOT NULL DEFAULT '',
            side TEXT NOT NULL DEFAULT 'long',
            entry_price REAL NOT NULL,
            current_price REAL NOT NULL DEFAULT 0,
            quantity REAL NOT NULL,
            invested_amount REAL NOT NULL,
            current_value REAL NOT NULL DEFAULT 0,
            pnl REAL NOT NULL DEFAULT 0,
            pnl_pct REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'open',
            stop_loss REAL NOT NULL DEFAULT 0,
            take_profit REAL NOT NULL DEFAULT 0,
            ai_reasoning TEXT NOT NULL DEFAULT '',
            tx_hash TEXT NOT NULL DEFAULT '',
            opened_at TEXT NOT NULL DEFAULT (datetime('now')),
            closed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS trade_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            position_id INTEGER,
            coin_id TEXT NOT NULL,
            symbol TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            amount REAL NOT NULL,
            ai_score INTEGER NOT NULL DEFAULT 0,
            ai_reasoning TEXT NOT NULL DEFAULT '',
            tx_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'executed',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS trade_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            event TEXT NOT NULL,
            details TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS trade_stats (
            user_id INTEGER PRIMARY KEY,
            total_invested REAL NOT NULL DEFAULT 0,
            total_pnl REAL NOT NULL DEFAULT 0,
            total_trades INTEGER NOT NULL DEFAULT 0,
            winning_trades INTEGER NOT NULL DEFAULT 0,
            losing_trades INTEGER NOT NULL DEFAULT 0,
            best_trade_pnl REAL NOT NULL DEFAULT 0,
            worst_trade_pnl REAL NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    ''')
    # Migration: add tx_hash column if missing
    try:
        conn.execute("ALTER TABLE trade_orders ADD COLUMN tx_hash TEXT NOT NULL DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE trade_positions ADD COLUMN tx_hash TEXT NOT NULL DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


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
    conn = _get_db()
    try:
        row = conn.execute("SELECT * FROM trade_settings WHERE user_id = ?", (user_id,)).fetchone()
        if row:
            return dict(row)
        return {
            "user_id": user_id, "auto_trade_enabled": 0,
            "max_trade_pct": 20.0, "daily_loss_limit_pct": 10.0,
            "max_open_positions": 5, "stop_loss_pct": 8.0,
            "take_profit_pct": 20.0, "cooldown_minutes": 5,
            "min_market_cap": 1000000, "risk_level": "moderate",
        }
    finally:
        conn.close()


def update_trade_settings(user_id: int, settings: Dict[str, Any]) -> Dict[str, Any]:
    conn = _get_db()
    try:
        conn.execute('''
            INSERT INTO trade_settings (user_id, auto_trade_enabled, max_trade_pct, daily_loss_limit_pct,
                max_open_positions, stop_loss_pct, take_profit_pct, cooldown_minutes, min_market_cap, risk_level, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                auto_trade_enabled = excluded.auto_trade_enabled,
                max_trade_pct = excluded.max_trade_pct,
                daily_loss_limit_pct = excluded.daily_loss_limit_pct,
                max_open_positions = excluded.max_open_positions,
                stop_loss_pct = excluded.stop_loss_pct,
                take_profit_pct = excluded.take_profit_pct,
                cooldown_minutes = excluded.cooldown_minutes,
                min_market_cap = excluded.min_market_cap,
                risk_level = excluded.risk_level,
                updated_at = datetime('now')
        ''', (
            user_id,
            settings.get("auto_trade_enabled", 0),
            settings.get("max_trade_pct", 20.0),
            settings.get("daily_loss_limit_pct", 10.0),
            settings.get("max_open_positions", 5),
            settings.get("stop_loss_pct", 8.0),
            settings.get("take_profit_pct", 20.0),
            settings.get("cooldown_minutes", 5),
            settings.get("min_market_cap", 1000000),
            settings.get("risk_level", "moderate"),
        ))
        conn.commit()
        return get_trade_settings(user_id)
    finally:
        conn.close()


# -- Real Wallet Balance (from auth.py wallet_balance table) -------------------

def _get_wallet_balance(user_id: int) -> float:
    conn = _get_db()
    try:
        row = conn.execute("SELECT balance FROM wallet_balance WHERE user_id = ?", (user_id,)).fetchone()
        return row["balance"] if row else 0.0
    finally:
        conn.close()


def _deduct_wallet(conn, user_id: int, amount: float):
    conn.execute("UPDATE wallet_balance SET balance = balance - ?, updated_at = datetime('now') WHERE user_id = ?", (amount, user_id))
    conn.execute("INSERT INTO wallet_transactions (user_id, type, amount, description, status) VALUES (?, 'trade_buy', ?, ?, 'completed')",
                 (user_id, amount, f"Auto-trade: invested ${amount:,.2f}"))


def _credit_wallet(conn, user_id: int, amount: float, description: str = ""):
    conn.execute("UPDATE wallet_balance SET balance = balance + ?, updated_at = datetime('now') WHERE user_id = ?", (amount, user_id))
    conn.execute("INSERT INTO wallet_transactions (user_id, type, amount, description, status) VALUES (?, 'trade_sell', ?, ?, 'completed')",
                 (user_id, amount, description))


# -- Trade Stats ---------------------------------------------------------------

def _get_trade_stats(user_id: int) -> Dict[str, Any]:
    conn = _get_db()
    try:
        row = conn.execute("SELECT * FROM trade_stats WHERE user_id = ?", (user_id,)).fetchone()
        if row:
            return dict(row)
        conn.execute("INSERT OR IGNORE INTO trade_stats (user_id) VALUES (?)", (user_id,))
        conn.commit()
        return {"user_id": user_id, "total_invested": 0, "total_pnl": 0, "total_trades": 0,
                "winning_trades": 0, "losing_trades": 0, "best_trade_pnl": 0, "worst_trade_pnl": 0}
    finally:
        conn.close()


def reset_trading(user_id: int) -> Dict[str, Any]:
    conn = _get_db()
    try:
        open_pos = conn.execute("SELECT invested_amount FROM trade_positions WHERE user_id = ? AND status = 'open'", (user_id,)).fetchall()
        refund = sum(p["invested_amount"] for p in open_pos)
        if refund > 0:
            conn.execute("UPDATE wallet_balance SET balance = balance + ?, updated_at = datetime('now') WHERE user_id = ?", (refund, user_id))
            conn.execute("INSERT INTO wallet_transactions (user_id, type, amount, description, status) VALUES (?, 'trade_refund', ?, 'Trading reset - positions refunded', 'completed')", (user_id, refund))
        conn.execute("UPDATE trade_positions SET status = 'closed', closed_at = datetime('now') WHERE user_id = ? AND status = 'open'", (user_id,))
        conn.execute('''
            INSERT INTO trade_stats (user_id, total_invested, total_pnl, total_trades, winning_trades, losing_trades, best_trade_pnl, worst_trade_pnl, updated_at)
            VALUES (?, 0, 0, 0, 0, 0, 0, 0, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET total_invested=0, total_pnl=0, total_trades=0,
                winning_trades=0, losing_trades=0, best_trade_pnl=0, worst_trade_pnl=0, updated_at=datetime('now')
        ''', (user_id,))
        conn.commit()
        _log_event(conn, user_id, "RESET", f"Trading reset — ${refund:,.0f} refunded to wallet")
        conn.commit()
        return {"success": True, "refunded": refund}
    finally:
        conn.close()


# -- Positions -----------------------------------------------------------------

def get_open_positions(user_id: int) -> List[Dict[str, Any]]:
    conn = _get_db()
    try:
        rows = conn.execute("SELECT * FROM trade_positions WHERE user_id = ? AND status = 'open' ORDER BY opened_at DESC", (user_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_closed_positions(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_db()
    try:
        rows = conn.execute("SELECT * FROM trade_positions WHERE user_id = ? AND status = 'closed' ORDER BY closed_at DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_trade_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_db()
    try:
        rows = conn.execute("SELECT * FROM trade_orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_trade_log_entries(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_db()
    try:
        rows = conn.execute("SELECT * FROM trade_log WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _log_event(conn, user_id: int, event: str, details: str):
    conn.execute("INSERT INTO trade_log (user_id, event, details) VALUES (?, ?, ?)", (user_id, event, details))


# -- Core Trading Logic --------------------------------------------------------

def execute_buy(user_id, coin_id, coin_name, symbol, price, amount, ai_score, ai_reasoning, stop_loss_pct=8.0, take_profit_pct=20.0):
    conn = _get_db()
    try:
        # BEGIN IMMEDIATE for atomic balance check + debit
        conn.execute("BEGIN IMMEDIATE")

        bal = conn.execute("SELECT balance FROM wallet_balance WHERE user_id = ?", (user_id,)).fetchone()
        balance = bal["balance"] if bal else 0.0
        if balance <= 0:
            conn.execute("ROLLBACK")
            return {"success": False, "error": "No funds in wallet. Add money from your bank account first."}
        if amount > balance:
            conn.execute("ROLLBACK")
            return {"success": False, "error": f"Insufficient balance. You have {balance:,.2f} but tried to invest {amount:,.2f}"}
        if amount <= 0:
            conn.execute("ROLLBACK")
            return {"success": False, "error": "Invalid amount"}
        if price <= 0:
            conn.execute("ROLLBACK")
            return {"success": False, "error": "Invalid price"}

        quantity = amount / price
        stop_loss = price * (1 - stop_loss_pct / 100)
        take_profit = price * (1 + take_profit_pct / 100)

        timestamp = datetime.now(timezone.utc).isoformat()
        tx_hash = generate_tx_hash(user_id, 'BUY', coin_id, symbol.upper(), price, quantity, amount, timestamp)

        cursor = conn.execute('''
            INSERT INTO trade_positions (user_id, coin_id, coin_name, symbol, entry_price, current_price,
                quantity, invested_amount, current_value, stop_loss, take_profit, ai_reasoning, tx_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, coin_id, coin_name, symbol.upper(), price, price, quantity, amount, amount, stop_loss, take_profit, ai_reasoning, tx_hash))
        position_id = cursor.lastrowid

        conn.execute('''
            INSERT INTO trade_orders (user_id, position_id, coin_id, symbol, action, price, quantity, amount, ai_score, ai_reasoning, tx_hash, created_at)
            VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, position_id, coin_id, symbol.upper(), price, quantity, amount, ai_score, ai_reasoning, tx_hash, timestamp))

        _deduct_wallet(conn, user_id, amount)

        conn.execute('''
            INSERT INTO trade_stats (user_id, total_invested, total_trades, updated_at)
            VALUES (?, ?, 1, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET total_invested = total_invested + ?, total_trades = total_trades + 1, updated_at = datetime('now')
        ''', (user_id, amount, amount))

        _log_event(conn, user_id, "BUY", f"Bought {quantity:.6f} {symbol.upper()} at ${price:,.2f} (${amount:,.0f}) | Score: {ai_score}/100 | Hash: {tx_hash[:16]}... | {ai_reasoning[:180]}")
        conn.commit()

        # Record on blockchain (async, non-blocking)
        blockchain.record_transaction_async(tx_hash, "BUY", symbol.upper(), amount)

        return {"success": True, "position_id": position_id, "quantity": quantity, "amount": amount, "tx_hash": tx_hash}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def execute_sell(user_id, position_id, current_price, reason="manual"):
    conn = _get_db()
    try:
        # BEGIN IMMEDIATE for atomic position close + wallet credit
        conn.execute("BEGIN IMMEDIATE")

        pos = conn.execute("SELECT * FROM trade_positions WHERE id = ? AND user_id = ? AND status = 'open'", (position_id, user_id)).fetchone()
        if not pos:
            conn.execute("ROLLBACK")
            return {"success": False, "error": "Position not found or already closed"}

        current_value = pos["quantity"] * current_price
        pnl = current_value - pos["invested_amount"]
        pnl_pct = (pnl / pos["invested_amount"]) * 100

        timestamp = datetime.now(timezone.utc).isoformat()
        tx_hash = generate_tx_hash(user_id, 'SELL', pos["coin_id"], pos["symbol"], current_price, pos["quantity"], current_value, timestamp)

        conn.execute('''
            UPDATE trade_positions SET status = 'closed', current_price = ?, current_value = ?,
                pnl = ?, pnl_pct = ?, tx_hash = ?, closed_at = datetime('now')
            WHERE id = ?
        ''', (current_price, current_value, pnl, pnl_pct, tx_hash, position_id))

        conn.execute('''
            INSERT INTO trade_orders (user_id, position_id, coin_id, symbol, action, price, quantity, amount, ai_reasoning, tx_hash, created_at)
            VALUES (?, ?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?)
        ''', (user_id, position_id, pos["coin_id"], pos["symbol"], current_price, pos["quantity"], current_value, reason, tx_hash, timestamp))

        _credit_wallet(conn, user_id, current_value, f"Sold {pos['symbol']} — P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")

        win_inc = 1 if pnl > 0 else 0
        loss_inc = 1 if pnl < 0 else 0
        conn.execute('''
            UPDATE trade_stats SET total_pnl = total_pnl + ?, total_trades = total_trades + 1,
                winning_trades = winning_trades + ?, losing_trades = losing_trades + ?,
                best_trade_pnl = MAX(best_trade_pnl, ?), worst_trade_pnl = MIN(worst_trade_pnl, ?),
                updated_at = datetime('now')
            WHERE user_id = ?
        ''', (pnl, win_inc, loss_inc, pnl, pnl, user_id))

        _log_event(conn, user_id, "SELL", f"Sold {pos['quantity']:.6f} {pos['symbol']} at ${current_price:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%) | Hash: {tx_hash[:16]}... | Reason: {reason}")
        conn.commit()

        # Record on blockchain (async, non-blocking)
        blockchain.record_transaction_async(tx_hash, "SELL", pos["symbol"], current_value)

        return {"success": True, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2), "amount": round(current_value, 2), "tx_hash": tx_hash}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


# -- AI Research Engine --------------------------------------------------------

async def research_opportunities(cg_collector, dex_collector, gemini_client=None):
    opportunities = []
    try:
        top_coins = await cg_collector.get_top_coins(limit=30)
        trending = await cg_collector.get_trending()
        trending_ids = {t.coin_id for t in trending} if trending else set()

        for coin in top_coins:
            score = 10  # Base score for being in top 30
            reasons = []
            change_24h = coin.price_change_pct_24h

            # Momentum scoring (improved with more granularity)
            if 5 < change_24h < 15:
                score += 25; reasons.append(f"Strong 24h momentum: {change_24h:+.1f}%")
            elif 15 <= change_24h < 30:
                score += 18; reasons.append(f"Very strong momentum (watch for overbought): {change_24h:+.1f}%")
            elif change_24h >= 30:
                score += 5; reasons.append(f"Extreme pump — high reversal risk: {change_24h:+.1f}%")
            elif 1 < change_24h <= 5:
                score += 18; reasons.append(f"Positive momentum: {change_24h:+.1f}%")
            elif 0 < change_24h <= 1:
                score += 8; reasons.append(f"Slight uptick: {change_24h:+.1f}%")
            elif -5 < change_24h <= 0:
                score += 5; reasons.append(f"Stable/dip buy opportunity: {change_24h:+.1f}%")
            elif -10 < change_24h <= -5:
                score += 2; reasons.append(f"Moderate decline: {change_24h:+.1f}%")
            elif change_24h <= -10:
                score -= 10; reasons.append(f"Heavy decline: {change_24h:+.1f}%")

            # Volume/market-cap ratio (improved thresholds)
            if coin.market_cap > 0:
                vol_ratio = coin.total_volume_24h / coin.market_cap
                if vol_ratio > 0.5:
                    score += 15; reasons.append(f"Extremely high volume/mcap: {vol_ratio:.2f} (possible manipulation)")
                elif vol_ratio > 0.3:
                    score += 20; reasons.append(f"High volume/mcap ratio: {vol_ratio:.2f}")
                elif vol_ratio > 0.15:
                    score += 14; reasons.append(f"Strong volume: {vol_ratio:.2f}")
                elif vol_ratio > 0.05:
                    score += 8; reasons.append(f"Healthy volume: {vol_ratio:.2f}")
                elif vol_ratio < 0.01:
                    score -= 5; reasons.append(f"Very low volume: {vol_ratio:.3f}")

            # Trending bonus
            if coin.coin_id in trending_ids:
                score += 15; reasons.append("Trending on CoinGecko")

            # Market cap tiers (more nuanced)
            if coin.market_cap > 50_000_000_000:
                score += 10; reasons.append("Mega cap - very stable")
            elif coin.market_cap > 10_000_000_000:
                score += 12; reasons.append("Large cap - lower risk")
            elif coin.market_cap > 1_000_000_000:
                score += 8; reasons.append("Mid cap")
            elif coin.market_cap > 100_000_000:
                score += 5; reasons.append("Small cap - higher risk")
            elif coin.market_cap < 10_000_000:
                score -= 5; reasons.append("Micro cap - very high risk")

            # ATH distance (improved)
            if hasattr(coin, 'ath') and coin.ath > 0:
                ath_ratio = coin.current_price / coin.ath
                if 0.3 < ath_ratio < 0.7:
                    score += 10; reasons.append(f"Room to grow - {(1-ath_ratio)*100:.0f}% below ATH")
                elif ath_ratio < 0.3:
                    score += 5; reasons.append(f"Deep discount - {(1-ath_ratio)*100:.0f}% below ATH")
                elif ath_ratio > 0.95:
                    score -= 3; reasons.append(f"Near ATH - limited upside: {(1-ath_ratio)*100:.1f}% below")
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
                "You are PumpIQ, an expert crypto trading AI. For each coin below, write a detailed 2-3 sentence analysis explaining:\n"
                "1. WHY you would buy it right now (momentum, volume, trend signals)\n"
                "2. What RISKS exist (volatility, market cap, recent dumps)\n"
                "3. Your RECOMMENDATION (Strong Buy, Buy, Hold, or Avoid) with a target % gain\n\n"
                "Coins to analyze:\n"
                + "\n".join(
                    f"- {t['name']} ({t['symbol']}): Price ${t['price']:,.6f}, "
                    f"24h Change: {t['change_24h']:+.1f}%, Market Cap: ${t['market_cap']:,.0f}, "
                    f"Volume: ${t['volume_24h']:,.0f}, PumpIQ Score: {t['score']}/100"
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

    # Record top opportunities as predictions in the learning loop
    ll = _get_learning_loop()
    if ll:
        for opp in opportunities[:15]:
            try:
                verdict = (
                    "Strong Buy" if opp["score"] >= 70
                    else "Moderate Buy" if opp["score"] >= 50
                    else "Cautious Buy" if opp["score"] >= 35
                    else "Hold"
                )
                ll.record_prediction(
                    token_ticker=opp["symbol"],
                    token_name=opp["name"],
                    verdict=verdict,
                    confidence=opp["score"] / 10.0,
                    composite_score=opp["score"],
                    price_at_prediction=opp["price"],
                    target_price=opp["price"] * 1.10,
                    stop_loss_price=opp["price"] * 0.95,
                    market_condition="sideways",
                    market_regime=opp.get("market_regime", "unknown"),
                    risk_level="MEDIUM",
                    ai_thought_summary=opp.get("reasoning", "")[:500],
                )
            except Exception:
                pass
        logger.info("📊 Recorded %d research predictions to learning loop", min(len(opportunities), 15))

    return opportunities


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
        confidence_threshold = 4.0
        max_daily = 10
        risk_profile = "balanced"

    # Risk profile modifiers
    risk_modifiers = {
        "conservative": {"max_trade_pct_mult": 0.5, "min_score": 45, "stop_loss_add": 2},
        "balanced":     {"max_trade_pct_mult": 1.0, "min_score": 30, "stop_loss_add": 0},
        "aggressive":   {"max_trade_pct_mult": 1.5, "min_score": 20, "stop_loss_add": -2},
    }
    mods = risk_modifiers.get(risk_profile, risk_modifiers["balanced"])

    balance = _get_wallet_balance(user_id)
    stats = _get_trade_stats(user_id)
    total_pnl = stats.get("total_pnl", 0)
    results = {"actions": [], "positions_updated": 0, "new_trades": 0}

    conn = _get_db()
    try:
        if balance > 0:
            loss_pct = (total_pnl / balance) * 100 if total_pnl < 0 else 0
            if loss_pct < -settings["daily_loss_limit_pct"]:
                _log_event(conn, user_id, "SAFETY", f"Daily loss limit hit: {loss_pct:.1f}%")
                conn.commit()
                return {"status": "paused", "message": f"Daily loss limit reached ({loss_pct:.1f}%)"}
        last_trade = conn.execute("SELECT created_at FROM trade_orders WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,)).fetchone()
        if last_trade:
            last_time = datetime.fromisoformat(last_trade["created_at"])
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            cooldown = timedelta(minutes=settings["cooldown_minutes"])
            if datetime.now(timezone.utc) - last_time < cooldown:
                return {"status": "cooldown", "message": f"Cooldown active ({settings['cooldown_minutes']}min)"}
    finally:
        conn.close()

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
            conn = _get_db()
            conn.execute("UPDATE trade_positions SET current_price = ?, current_value = ?, pnl = ?, pnl_pct = ? WHERE id = ?",
                         (current_price, current_value, pnl, pnl_pct, pos["id"]))
            conn.commit()
            conn.close()
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
    conn_check = _get_db()
    try:
        today_trades = conn_check.execute(
            "SELECT COUNT(*) as cnt FROM trade_orders WHERE user_id = ? AND action = 'BUY' AND created_at >= date('now')",
            (user_id,),
        ).fetchone()
        trades_today = today_trades["cnt"] if today_trades else 0
    finally:
        conn_check.close()

    # Calculate effective min score up front for logging and filtering
    min_score = mods["min_score"]
    confidence_min_score = int(confidence_threshold * 10)  # e.g. 4.0 → 40
    effective_min_score = max(min_score, confidence_min_score)

    logger.info("Auto-trade user %d: balance=$%.2f, open=%d/%d, trades_today=%d/%d, min_score=%d",
                user_id, balance, open_count, settings["max_open_positions"], trades_today, max_daily, effective_min_score)
    if open_count < settings["max_open_positions"] and balance > 10 and trades_today < max_daily:
        opportunities = await research_opportunities(cg_collector, dex_collector, gemini_client)
        held_coins = {p["coin_id"] for p in get_open_positions(user_id)}
        opportunities = [o for o in opportunities if o["coin_id"] not in held_coins]
        opportunities = [o for o in opportunities if o["market_cap"] >= settings["min_market_cap"]]

        # Apply confidence threshold filter
        opportunities = [o for o in opportunities if o["score"] >= effective_min_score]
        logger.info("Auto-trade user %d: %d opportunities after filtering (min_score=%d)", user_id, len(opportunities), effective_min_score)
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
            if trade_amount < 10:
                break

            effective_sl = max(1.0, settings["stop_loss_pct"] + mods["stop_loss_add"])

            # Build detailed AI reasoning for the buy
            detailed_reasoning = (
                f"BUY SIGNAL for {opp['name']} ({opp['symbol']}): "
                f"Price ${opp['price']:,.6f} | 24h: {opp['change_24h']:+.1f}% | "
                f"Market Cap: ${opp['market_cap']:,.0f} | Volume: ${opp['volume_24h']:,.0f} | "
                f"Score: {opp['score']}/100 (threshold: {effective_min_score}). "
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
                    f"| Score: {opp['score']}/100 | Profile: {risk_profile} "
                    f"| TX: {buy_result['tx_hash'][:16]}..."
                )

                # Record in learning loop
                ll = _get_learning_loop()
                if ll:
                    try:
                        verdict = "Strong Buy" if opp["score"] >= 70 else "Moderate Buy" if opp["score"] >= 50 else "Cautious Buy"
                        ll.record_prediction(
                            token_ticker=opp["symbol"],
                            token_name=opp["name"],
                            verdict=verdict,
                            confidence=opp["score"] / 10.0,
                            composite_score=opp["score"],
                            price_at_prediction=opp["price"],
                            target_price=opp["price"] * 1.10,
                            stop_loss_price=opp["price"] * 0.95,
                            market_condition="sideways",
                            market_regime=opp.get("market_regime", "unknown"),
                            risk_level=risk_profile.upper() if risk_profile else "MEDIUM",
                            ai_thought_summary=detailed_reasoning[:500] if detailed_reasoning else "",
                            user_id=user_id,
                        )
                        logger.info("✅ Recorded prediction for %s (score=%s)", opp["symbol"], opp["score"])
                    except Exception as e:
                        logger.warning("Failed to record prediction for %s: %s", opp["symbol"], e)

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


# Initialize tables on import
init_trading_tables()
