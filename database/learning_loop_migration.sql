-- ═══════════════════════════════════════════════════════════════════════════════
-- NexYpher — Learning Loop Migration (SQLite → Supabase)
-- ═══════════════════════════════════════════════════════════════════════════════
-- Run this ONCE in your Supabase SQL Editor:
--   https://app.supabase.com → Your Project → SQL Editor → New Query → Paste → Run
--
-- These tables are also included in database/supabase_schema.sql for new setups.
-- This file is for EXISTING deployments that need to add the learning loop tables.
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS ll_predictions (
    id                    BIGSERIAL PRIMARY KEY,
    prediction_id         TEXT UNIQUE NOT NULL,
    user_id               BIGINT DEFAULT 0,
    token_ticker          TEXT NOT NULL,
    token_name            TEXT NOT NULL DEFAULT '',
    verdict               TEXT NOT NULL,
    confidence            DOUBLE PRECISION NOT NULL,
    composite_score       DOUBLE PRECISION NOT NULL DEFAULT 0,
    predicted_direction   TEXT NOT NULL DEFAULT 'up',
    price_at_prediction   DOUBLE PRECISION NOT NULL,
    target_price          DOUBLE PRECISION NOT NULL DEFAULT 0,
    stop_loss_price       DOUBLE PRECISION NOT NULL DEFAULT 0,
    market_condition      TEXT NOT NULL DEFAULT 'sideways',
    market_regime         TEXT NOT NULL DEFAULT 'unknown',
    risk_level            TEXT NOT NULL DEFAULT 'MEDIUM',
    enabled_modes         TEXT NOT NULL DEFAULT '[]',
    ai_thought_summary    TEXT NOT NULL DEFAULT '',
    -- Outcome fields (filled by evaluation)
    actual_price_24h      DOUBLE PRECISION DEFAULT NULL,
    actual_price_7d       DOUBLE PRECISION DEFAULT NULL,
    direction_correct_24h BOOLEAN DEFAULT NULL,
    direction_correct_7d  BOOLEAN DEFAULT NULL,
    pnl_pct_24h          DOUBLE PRECISION DEFAULT NULL,
    pnl_pct_7d           DOUBLE PRECISION DEFAULT NULL,
    target_hit            BOOLEAN DEFAULT NULL,
    stop_loss_hit         BOOLEAN DEFAULT NULL,
    -- Metadata
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    evaluated_24h_at      TIMESTAMPTZ DEFAULT NULL,
    evaluated_7d_at       TIMESTAMPTZ DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS ll_strategy_adjustments (
    id              BIGSERIAL PRIMARY KEY,
    adjustment_type TEXT NOT NULL,
    description     TEXT NOT NULL,
    old_value       TEXT NOT NULL DEFAULT '',
    new_value       TEXT NOT NULL DEFAULT '',
    reason          TEXT NOT NULL DEFAULT '',
    market_regime   TEXT NOT NULL DEFAULT '',
    applied         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ll_accuracy_snapshots (
    id                       BIGSERIAL PRIMARY KEY,
    period                   TEXT NOT NULL,
    total_predictions        INTEGER NOT NULL DEFAULT 0,
    correct_24h              INTEGER NOT NULL DEFAULT 0,
    correct_7d               INTEGER NOT NULL DEFAULT 0,
    accuracy_24h             DOUBLE PRECISION NOT NULL DEFAULT 0,
    accuracy_7d              DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_confidence_correct   DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_confidence_incorrect DOUBLE PRECISION NOT NULL DEFAULT 0,
    best_mode                TEXT NOT NULL DEFAULT '',
    worst_mode               TEXT NOT NULL DEFAULT '',
    market_regime            TEXT NOT NULL DEFAULT '',
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ll_predictions_ticker  ON ll_predictions(token_ticker);
CREATE INDEX IF NOT EXISTS idx_ll_predictions_created ON ll_predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_ll_predictions_eval24  ON ll_predictions(evaluated_24h_at);
CREATE INDEX IF NOT EXISTS idx_ll_adj_created         ON ll_strategy_adjustments(created_at);
