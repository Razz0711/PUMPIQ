-- ═══════════════════════════════════════════════════════════════════════════════
-- NexYpher — Supabase Schema
-- ═══════════════════════════════════════════════════════════════════════════════
-- Run this ONCE in your Supabase SQL Editor:
--   https://app.supabase.com → Your Project → SQL Editor → New Query → Paste → Run
-- ═══════════════════════════════════════════════════════════════════════════════


-- ─── CORE TABLES (auth.py) ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS wallets (
    id        BIGSERIAL PRIMARY KEY,
    user_id   BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    address   TEXT NOT NULL,
    chain     TEXT NOT NULL DEFAULT 'ethereum',
    label     TEXT DEFAULT '',
    added_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, address, chain)
);

CREATE TABLE IF NOT EXISTS watchlist (
    id        BIGSERIAL PRIMARY KEY,
    user_id   BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    coin_id   TEXT NOT NULL,
    added_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, coin_id)
);

CREATE TABLE IF NOT EXISTS portfolio (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    coin_id       TEXT NOT NULL,
    amount        NUMERIC(18,8) NOT NULL DEFAULT 0,
    avg_buy_price NUMERIC(18,8) NOT NULL DEFAULT 0,
    notes         TEXT DEFAULT '',
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, coin_id)
);

CREATE TABLE IF NOT EXISTS trade_history (
    id             BIGSERIAL PRIMARY KEY,
    user_id        BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    coin_id        TEXT NOT NULL,
    action         TEXT NOT NULL,
    amount         NUMERIC(18,8) NOT NULL,
    price          NUMERIC(18,8) NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    wallet_address TEXT DEFAULT '',
    tx_hash        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS email_tokens (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token       TEXT UNIQUE NOT NULL,
    token_type  TEXT NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    used        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bank_accounts (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_holder      TEXT NOT NULL,
    account_number_hash TEXT NOT NULL,
    account_last4       TEXT NOT NULL,
    ifsc_code           TEXT NOT NULL,
    bank_name           TEXT NOT NULL,
    verified            BOOLEAN NOT NULL DEFAULT FALSE,
    status              TEXT NOT NULL DEFAULT 'pending',
    added_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wallet_balance (
    user_id    BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    balance    NUMERIC(18,8) NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wallet_transactions (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,
    amount      NUMERIC(18,8) NOT NULL,
    bank_id     BIGINT,
    description TEXT DEFAULT '',
    tx_hash     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'completed',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id              BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    risk_profile         TEXT NOT NULL DEFAULT 'balanced',
    ai_sensitivity       DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    auto_trade_threshold DOUBLE PRECISION NOT NULL DEFAULT 7.0,
    max_daily_trades     INTEGER NOT NULL DEFAULT 10,
    preferred_chains     TEXT NOT NULL DEFAULT '["ethereum","solana"]',
    notification_email   BOOLEAN NOT NULL DEFAULT TRUE,
    notification_push    BOOLEAN NOT NULL DEFAULT TRUE,
    dark_mode            BOOLEAN NOT NULL DEFAULT TRUE,
    language             TEXT NOT NULL DEFAULT 'en',
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ─── TRADING TABLES (trading_engine.py) ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS trade_settings (
    user_id              BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    auto_trade_enabled   BOOLEAN NOT NULL DEFAULT FALSE,
    max_trade_pct        NUMERIC(8,4) NOT NULL DEFAULT 20.0,
    daily_loss_limit_pct NUMERIC(8,4) NOT NULL DEFAULT 10.0,
    max_open_positions   INTEGER NOT NULL DEFAULT 5,
    stop_loss_pct        NUMERIC(8,4) NOT NULL DEFAULT 8.0,
    take_profit_pct      NUMERIC(8,4) NOT NULL DEFAULT 20.0,
    cooldown_minutes     INTEGER NOT NULL DEFAULT 5,
    min_market_cap       NUMERIC(18,2) NOT NULL DEFAULT 1000000,
    risk_level           TEXT NOT NULL DEFAULT 'moderate',
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trade_positions (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    coin_id         TEXT NOT NULL,
    coin_name       TEXT NOT NULL DEFAULT '',
    symbol          TEXT NOT NULL DEFAULT '',
    side            TEXT NOT NULL DEFAULT 'long',
    entry_price     NUMERIC(18,8) NOT NULL,
    current_price   NUMERIC(18,8) NOT NULL DEFAULT 0,
    quantity        NUMERIC(18,8) NOT NULL,
    invested_amount NUMERIC(18,8) NOT NULL,
    current_value   NUMERIC(18,8) NOT NULL DEFAULT 0,
    pnl             NUMERIC(18,8) NOT NULL DEFAULT 0,
    pnl_pct         NUMERIC(8,4) NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'open',
    stop_loss       NUMERIC(18,8) NOT NULL DEFAULT 0,
    take_profit     NUMERIC(18,8) NOT NULL DEFAULT 0,
    ai_reasoning    TEXT NOT NULL DEFAULT '',
    tx_hash         TEXT NOT NULL DEFAULT '',
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS trade_orders (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    position_id  BIGINT,
    coin_id      TEXT NOT NULL,
    symbol       TEXT NOT NULL DEFAULT '',
    action       TEXT NOT NULL,
    price        NUMERIC(18,8) NOT NULL,
    quantity     NUMERIC(18,8) NOT NULL,
    amount       NUMERIC(18,8) NOT NULL,
    ai_score     INTEGER NOT NULL DEFAULT 0,
    ai_reasoning TEXT NOT NULL DEFAULT '',
    tx_hash      TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'executed',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trade_log (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event      TEXT NOT NULL,
    details    TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trade_stats (
    user_id        BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    total_invested NUMERIC(18,8) NOT NULL DEFAULT 0,
    total_pnl      NUMERIC(18,8) NOT NULL DEFAULT 0,
    total_trades   INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades  INTEGER NOT NULL DEFAULT 0,
    best_trade_pnl NUMERIC(18,8) NOT NULL DEFAULT 0,
    worst_trade_pnl NUMERIC(18,8) NOT NULL DEFAULT 0,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ─── RPC FUNCTIONS (atomic operations called via supabase.rpc()) ────────────

-- Atomically adjust wallet balance (positive = credit, negative = debit)
-- Raises an exception if a debit would make the balance negative.
CREATE OR REPLACE FUNCTION update_wallet_balance(p_user_id BIGINT, p_delta DOUBLE PRECISION)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    new_bal DOUBLE PRECISION;
    cur_bal DOUBLE PRECISION;
BEGIN
    -- Ensure the row exists (idempotent)
    INSERT INTO wallet_balance (user_id, balance, updated_at)
    VALUES (p_user_id, 0, NOW())
    ON CONFLICT (user_id) DO NOTHING;

    -- Lock the row and read current balance
    SELECT balance INTO cur_bal
    FROM wallet_balance
    WHERE user_id = p_user_id
    FOR UPDATE;

    -- Guard: reject debits that exceed available balance
    IF (cur_bal + p_delta) < 0 THEN
        RAISE EXCEPTION 'Insufficient wallet balance. Current: %, Requested delta: %', cur_bal, p_delta;
    END IF;

    -- Apply the delta
    UPDATE wallet_balance
    SET balance = cur_bal + p_delta,
        updated_at = NOW()
    WHERE user_id = p_user_id;

    new_bal := cur_bal + p_delta;
    RETURN new_bal;
END;
$$ LANGUAGE plpgsql;


-- Atomically increment trade statistics
CREATE OR REPLACE FUNCTION increment_trade_stats(
    p_user_id  BIGINT,
    p_invested DOUBLE PRECISION DEFAULT 0,
    p_pnl      DOUBLE PRECISION DEFAULT 0,
    p_trades   INTEGER DEFAULT 0,
    p_wins     INTEGER DEFAULT 0,
    p_losses   INTEGER DEFAULT 0,
    p_best     DOUBLE PRECISION DEFAULT 0,
    p_worst    DOUBLE PRECISION DEFAULT 0
)
RETURNS void AS $$
BEGIN
    INSERT INTO trade_stats
        (user_id, total_invested, total_pnl, total_trades,
         winning_trades, losing_trades, best_trade_pnl, worst_trade_pnl, updated_at)
    VALUES
        (p_user_id, p_invested, p_pnl, p_trades, p_wins, p_losses, p_best, p_worst, NOW())
    ON CONFLICT (user_id) DO UPDATE SET
        total_invested  = trade_stats.total_invested + p_invested,
        total_pnl       = trade_stats.total_pnl + p_pnl,
        total_trades    = trade_stats.total_trades + p_trades,
        winning_trades  = trade_stats.winning_trades + p_wins,
        losing_trades   = trade_stats.losing_trades + p_losses,
        best_trade_pnl  = GREATEST(trade_stats.best_trade_pnl, p_best),
        worst_trade_pnl = LEAST(trade_stats.worst_trade_pnl, p_worst),
        updated_at      = NOW();
END;
$$ LANGUAGE plpgsql;


-- Reset trade stats to zero
CREATE OR REPLACE FUNCTION reset_trade_stats(p_user_id BIGINT)
RETURNS void AS $$
BEGIN
    INSERT INTO trade_stats
        (user_id, total_invested, total_pnl, total_trades,
         winning_trades, losing_trades, best_trade_pnl, worst_trade_pnl, updated_at)
    VALUES (p_user_id, 0, 0, 0, 0, 0, 0, 0, NOW())
    ON CONFLICT (user_id) DO UPDATE SET
        total_invested = 0, total_pnl = 0, total_trades = 0,
        winning_trades = 0, losing_trades = 0,
        best_trade_pnl = 0, worst_trade_pnl = 0, updated_at = NOW();
END;
$$ LANGUAGE plpgsql;


-- ─── INDEXES ────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_wallets_user              ON wallets(user_id);
CREATE INDEX IF NOT EXISTS idx_watchlist_user             ON watchlist(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_user             ON portfolio(user_id);
CREATE INDEX IF NOT EXISTS idx_trade_history_user         ON trade_history(user_id);
CREATE INDEX IF NOT EXISTS idx_email_tokens_token         ON email_tokens(token);
CREATE INDEX IF NOT EXISTS idx_bank_accounts_user         ON bank_accounts(user_id);
CREATE INDEX IF NOT EXISTS idx_wallet_transactions_user   ON wallet_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_trade_positions_user_status ON trade_positions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_trade_orders_user          ON trade_orders(user_id);
CREATE INDEX IF NOT EXISTS idx_trade_log_user             ON trade_log(user_id);


-- ─── ROW LEVEL SECURITY ─────────────────────────────────────────────────────
-- Enable RLS on all user-facing tables to prevent cross-user data access.
-- The backend service role bypasses RLS; direct client access is restricted.

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE wallets ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE email_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE bank_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE wallet_balance ENABLE ROW LEVEL SECURITY;
ALTER TABLE wallet_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_stats ENABLE ROW LEVEL SECURITY;

-- ─── RLS POLICIES ───────────────────────────────────────────────────────────
-- Users can only read/write their own rows.  The service_role key (used by
-- the backend) bypasses RLS automatically.

-- users: allow read own row only
CREATE POLICY users_self_access ON users
    FOR ALL USING (id = current_setting('app.current_user_id', true)::BIGINT);

-- wallets
CREATE POLICY wallets_owner ON wallets
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- watchlist
CREATE POLICY watchlist_owner ON watchlist
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- portfolio
CREATE POLICY portfolio_owner ON portfolio
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- trade_history
CREATE POLICY trade_history_owner ON trade_history
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- email_tokens
CREATE POLICY email_tokens_owner ON email_tokens
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- bank_accounts
CREATE POLICY bank_accounts_owner ON bank_accounts
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- wallet_balance
CREATE POLICY wallet_balance_owner ON wallet_balance
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- wallet_transactions
CREATE POLICY wallet_transactions_owner ON wallet_transactions
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- user_preferences
CREATE POLICY user_preferences_owner ON user_preferences
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- trade_settings
CREATE POLICY trade_settings_owner ON trade_settings
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- trade_positions
CREATE POLICY trade_positions_owner ON trade_positions
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- trade_orders
CREATE POLICY trade_orders_owner ON trade_orders
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- trade_log
CREATE POLICY trade_log_owner ON trade_log
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);

-- trade_stats
CREATE POLICY trade_stats_owner ON trade_stats
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::BIGINT);
-- ALTER TABLE wallet_balance ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE wallet_transactions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trade_settings ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trade_positions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trade_orders ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trade_log ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trade_stats ENABLE ROW LEVEL SECURITY;
