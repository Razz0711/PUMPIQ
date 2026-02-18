-- ═══════════════════════════════════════════════════════════════
-- NexYpher AlgoTrader Tables — Supabase (PostgreSQL)
-- Run this in Supabase SQL Editor
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS algo_strategies (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    instruments TEXT NOT NULL DEFAULT '[]',
    legs TEXT NOT NULL DEFAULT '[]',
    strategy_type TEXT NOT NULL DEFAULT 'time_based',
    order_type TEXT NOT NULL DEFAULT 'market',
    risk_config TEXT NOT NULL DEFAULT '{}',
    advanced_config TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'stopped',
    pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_trades INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS algo_exchanges (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    api_key_last4 TEXT NOT NULL DEFAULT '',
    connected INTEGER NOT NULL DEFAULT 1,
    connected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS algo_backtest_results (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_id BIGINT NOT NULL REFERENCES algo_strategies(id) ON DELETE CASCADE,
    time_range TEXT NOT NULL DEFAULT '1M',
    total_return DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_drawdown DOUBLE PRECISION NOT NULL DEFAULT 0,
    win_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    sharpe_ratio DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_trades INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS algo_trade_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_id BIGINT,
    severity TEXT NOT NULL DEFAULT 'INFO',
    message TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS algo_trade_reports (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_id BIGINT,
    strategy_name TEXT NOT NULL DEFAULT '',
    pair TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT 'BUY',
    qty DOUBLE PRECISION NOT NULL DEFAULT 0,
    buy_price DOUBLE PRECISION NOT NULL DEFAULT 0,
    sell_price DOUBLE PRECISION NOT NULL DEFAULT 0,
    pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    fees DOUBLE PRECISION NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'closed',
    exchange TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT 'live',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_algo_strategies_user ON algo_strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_algo_exchanges_user ON algo_exchanges(user_id);
CREATE INDEX IF NOT EXISTS idx_algo_backtest_user ON algo_backtest_results(user_id);
CREATE INDEX IF NOT EXISTS idx_algo_logs_user ON algo_trade_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_algo_reports_user ON algo_trade_reports(user_id);
