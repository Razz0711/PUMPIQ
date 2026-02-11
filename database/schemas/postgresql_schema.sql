-- PumpIQ PostgreSQL Database Schema
-- Version: 1.0.0
-- Database: PostgreSQL 15+ with TimescaleDB extension for time-series data

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- ============================================================================
-- 1. TOKENS TABLE
-- ============================================================================
CREATE TABLE tokens (
    token_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    contract_address VARCHAR(255) UNIQUE NOT NULL,
    blockchain VARCHAR(50) NOT NULL DEFAULT 'solana',
    
    -- Current market data
    current_price DECIMAL(20, 10),
    market_cap DECIMAL(20, 2),
    liquidity DECIMAL(20, 2),
    volume_24h DECIMAL(20, 2),
    
    -- RobinPump specific data
    bonding_curve_percent DECIMAL(5, 2),
    is_graduated BOOLEAN DEFAULT FALSE,
    graduation_timestamp TIMESTAMP,
    
    -- Metadata
    description TEXT,
    website VARCHAR(500),
    twitter VARCHAR(255),
    telegram VARCHAR(255),
    logo_url VARCHAR(500),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_price_update TIMESTAMP,
    
    -- Indexes for common queries
    CONSTRAINT valid_bonding_curve CHECK (bonding_curve_percent >= 0 AND bonding_curve_percent <= 100)
);

CREATE INDEX idx_tokens_ticker ON tokens(ticker);
CREATE INDEX idx_tokens_blockchain ON tokens(blockchain);
CREATE INDEX idx_tokens_bonding_curve ON tokens(bonding_curve_percent);
CREATE INDEX idx_tokens_market_cap ON tokens(market_cap);
CREATE INDEX idx_tokens_updated_at ON tokens(updated_at);

-- ============================================================================
-- 2. HISTORICAL_DATA TABLE (TimescaleDB Hypertable)
-- ============================================================================
CREATE TABLE historical_data (
    id UUID DEFAULT uuid_generate_v4(),
    token_id UUID NOT NULL REFERENCES tokens(token_id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    
    -- OHLCV data
    open_price DECIMAL(20, 10),
    high_price DECIMAL(20, 10),
    low_price DECIMAL(20, 10),
    close_price DECIMAL(20, 10),
    volume DECIMAL(20, 2),
    
    -- On-chain metrics
    holder_count INTEGER,
    whale_count INTEGER,
    top_10_holder_percent DECIMAL(5, 2),
    total_transactions INTEGER,
    buy_transactions INTEGER,
    sell_transactions INTEGER,
    
    -- Whale movements (last hour/day)
    whale_buys_1h INTEGER DEFAULT 0,
    whale_sells_1h INTEGER DEFAULT 0,
    whale_buy_volume_1h DECIMAL(20, 2) DEFAULT 0,
    whale_sell_volume_1h DECIMAL(20, 2) DEFAULT 0,
    
    -- Sentiment scores (0-100 scale)
    social_sentiment_score DECIMAL(5, 2),
    news_sentiment_score DECIMAL(5, 2),
    
    -- Technical indicators
    rsi_14 DECIMAL(5, 2),
    macd DECIMAL(20, 10),
    macd_signal DECIMAL(20, 10),
    bb_upper DECIMAL(20, 10),
    bb_middle DECIMAL(20, 10),
    bb_lower DECIMAL(20, 10),
    
    PRIMARY KEY (token_id, timestamp)
);

-- Convert to TimescaleDB hypertable for efficient time-series queries
SELECT create_hypertable('historical_data', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes for time-series queries
CREATE INDEX idx_historical_token_time ON historical_data(token_id, timestamp DESC);
CREATE INDEX idx_historical_timestamp ON historical_data(timestamp DESC);

-- ============================================================================
-- 3. RECOMMENDATIONS TABLE
-- ============================================================================
CREATE TABLE recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_id UUID NOT NULL REFERENCES tokens(token_id) ON DELETE CASCADE,
    user_id UUID,
    
    -- Recommendation details
    recommendation_type VARCHAR(20) NOT NULL, -- 'BUY', 'HOLD', 'SELL', 'AVOID'
    entry_price DECIMAL(20, 10) NOT NULL,
    current_price DECIMAL(20, 10),
    
    -- Target prices
    target_price_1 DECIMAL(20, 10),
    target_price_2 DECIMAL(20, 10),
    target_price_3 DECIMAL(20, 10),
    stop_loss DECIMAL(20, 10),
    
    -- Recommendation metadata
    confidence_score DECIMAL(5, 4) NOT NULL, -- 0.0000 to 1.0000
    risk_rating VARCHAR(20), -- 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'
    timeframe VARCHAR(50), -- 'SCALP', 'DAY_TRADE', 'SWING', 'LONG_TERM'
    
    -- Data source contribution flags
    news_contribution BOOLEAN DEFAULT FALSE,
    onchain_contribution BOOLEAN DEFAULT FALSE,
    technical_contribution BOOLEAN DEFAULT FALSE,
    social_contribution BOOLEAN DEFAULT FALSE,
    
    -- AI analysis
    ai_reasoning TEXT NOT NULL,
    key_factors JSONB, -- Array of key factors that influenced the recommendation
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'ACTIVE', -- 'ACTIVE', 'COMPLETED', 'STOPPED', 'EXPIRED'
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT valid_recommendation_type CHECK (recommendation_type IN ('BUY', 'HOLD', 'SELL', 'AVOID'))
);

CREATE INDEX idx_recommendations_token ON recommendations(token_id);
CREATE INDEX idx_recommendations_user ON recommendations(user_id);
CREATE INDEX idx_recommendations_created ON recommendations(created_at DESC);
CREATE INDEX idx_recommendations_confidence ON recommendations(confidence_score DESC);
CREATE INDEX idx_recommendations_status ON recommendations(status);
CREATE INDEX idx_recommendations_type ON recommendations(recommendation_type);

-- ============================================================================
-- 4. PERFORMANCE_TRACKING TABLE
-- ============================================================================
CREATE TABLE performance_tracking (
    tracking_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recommendation_id UUID NOT NULL REFERENCES recommendations(recommendation_id) ON DELETE CASCADE,
    
    -- Outcome tracking
    outcome VARCHAR(50), -- 'TARGET_1_HIT', 'TARGET_2_HIT', 'TARGET_3_HIT', 'STOP_LOSS_HIT', 'STILL_OPEN', 'EXPIRED'
    actual_exit_price DECIMAL(20, 10),
    actual_return_percent DECIMAL(10, 4), -- Can be negative
    
    -- Time tracking
    time_to_outcome_hours DECIMAL(10, 2),
    recommendation_created_at TIMESTAMP,
    outcome_timestamp TIMESTAMP,
    
    -- Success metrics
    is_successful BOOLEAN,
    hit_target_1 BOOLEAN DEFAULT FALSE,
    hit_target_2 BOOLEAN DEFAULT FALSE,
    hit_target_3 BOOLEAN DEFAULT FALSE,
    hit_stop_loss BOOLEAN DEFAULT FALSE,
    
    -- Price movement tracking
    max_price_reached DECIMAL(20, 10),
    min_price_reached DECIMAL(20, 10),
    max_gain_percent DECIMAL(10, 4),
    max_drawdown_percent DECIMAL(10, 4),
    
    -- Performance notes
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_recommendation ON performance_tracking(recommendation_id);
CREATE INDEX idx_performance_outcome ON performance_tracking(outcome);
CREATE INDEX idx_performance_success ON performance_tracking(is_successful);
CREATE INDEX idx_performance_return ON performance_tracking(actual_return_percent DESC);

-- ============================================================================
-- 5. USER_PREFERENCES TABLE
-- ============================================================================
CREATE TABLE user_preferences (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE,
    email VARCHAR(255) UNIQUE NOT NULL,
    
    -- Configuration (stored as JSONB for flexibility)
    config JSONB NOT NULL DEFAULT '{}',
    
    -- Watchlist (array of token IDs)
    watchlist UUID[] DEFAULT ARRAY[]::UUID[],
    
    -- Portfolio holdings
    portfolio JSONB DEFAULT '[]', -- Array of {token_id, quantity, avg_buy_price, current_value}
    
    -- Notification preferences
    notification_settings JSONB,
    
    -- Subscription info
    subscription_tier VARCHAR(50) DEFAULT 'free', -- 'free', 'basic', 'premium', 'pro'
    subscription_expires_at TIMESTAMP,
    
    -- Activity tracking
    last_login TIMESTAMP,
    total_recommendations_received INTEGER DEFAULT 0,
    total_trades_made INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON user_preferences(email);
CREATE INDEX idx_users_subscription ON user_preferences(subscription_tier);

-- ============================================================================
-- 6. DATA_SOURCE_CACHE TABLE
-- ============================================================================
CREATE TABLE data_source_cache (
    cache_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_identifier VARCHAR(255) NOT NULL, -- e.g., 'news_api_crypto', 'twitter_trending'
    cache_key VARCHAR(500) NOT NULL, -- Unique key for this cached data
    
    -- Cached data
    data JSONB NOT NULL,
    
    -- TTL management
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    ttl_seconds INTEGER,
    
    -- Metadata
    hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(source_identifier, cache_key)
);

CREATE INDEX idx_cache_source ON data_source_cache(source_identifier);
CREATE INDEX idx_cache_expires ON data_source_cache(expires_at);
CREATE INDEX idx_cache_key ON data_source_cache(cache_key);

-- Auto-delete expired cache entries
CREATE INDEX idx_cache_cleanup ON data_source_cache(expires_at) WHERE expires_at < CURRENT_TIMESTAMP;

-- ============================================================================
-- 7. ANALYSIS_LOGS TABLE
-- ============================================================================
CREATE TABLE analysis_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_id UUID REFERENCES tokens(token_id) ON DELETE SET NULL,
    
    analysis_type VARCHAR(50), -- 'NEWS', 'ONCHAIN', 'TECHNICAL', 'SOCIAL', 'AI_SYNTHESIS'
    
    -- Input data snapshot
    input_data JSONB,
    
    -- Output/Results
    output_data JSONB,
    analysis_result TEXT,
    
    -- Performance metrics
    execution_time_ms INTEGER,
    tokens_used INTEGER, -- For AI API calls
    
    -- Status
    status VARCHAR(20), -- 'SUCCESS', 'FAILED', 'PARTIAL'
    error_message TEXT,
    
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_logs_token ON analysis_logs(token_id);
CREATE INDEX idx_analysis_logs_type ON analysis_logs(analysis_type);
CREATE INDEX idx_analysis_logs_timestamp ON analysis_logs(timestamp DESC);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Active recommendations with token details
CREATE VIEW active_recommendations_view AS
SELECT 
    r.recommendation_id,
    r.token_id,
    t.name AS token_name,
    t.ticker,
    r.recommendation_type,
    r.entry_price,
    r.current_price,
    r.target_price_1,
    r.target_price_2,
    r.target_price_3,
    r.stop_loss,
    r.confidence_score,
    r.risk_rating,
    r.ai_reasoning,
    r.created_at,
    r.expires_at,
    CASE 
        WHEN r.current_price IS NOT NULL THEN 
            ROUND(((r.current_price - r.entry_price) / r.entry_price * 100)::NUMERIC, 2)
        ELSE NULL
    END AS current_return_percent
FROM recommendations r
JOIN tokens t ON r.token_id = t.token_id
WHERE r.status = 'ACTIVE'
ORDER BY r.created_at DESC;

-- View: Performance summary by token
CREATE VIEW token_performance_summary AS
SELECT 
    t.token_id,
    t.ticker,
    t.name,
    COUNT(DISTINCT r.recommendation_id) AS total_recommendations,
    COUNT(DISTINCT CASE WHEN pt.is_successful = TRUE THEN pt.tracking_id END) AS successful_recommendations,
    ROUND(AVG(pt.actual_return_percent)::NUMERIC, 2) AS avg_return_percent,
    ROUND(AVG(CASE WHEN pt.is_successful = TRUE THEN pt.actual_return_percent END)::NUMERIC, 2) AS avg_winning_return,
    ROUND(AVG(CASE WHEN pt.is_successful = FALSE THEN pt.actual_return_percent END)::NUMERIC, 2) AS avg_losing_return,
    ROUND((COUNT(DISTINCT CASE WHEN pt.is_successful = TRUE THEN pt.tracking_id END)::NUMERIC / 
           NULLIF(COUNT(DISTINCT pt.tracking_id), 0) * 100), 2) AS win_rate_percent
FROM tokens t
LEFT JOIN recommendations r ON t.token_id = r.token_id
LEFT JOIN performance_tracking pt ON r.recommendation_id = pt.recommendation_id
GROUP BY t.token_id, t.ticker, t.name;

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Function to update 'updated_at' timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
CREATE TRIGGER update_tokens_updated_at BEFORE UPDATE ON tokens
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_recommendations_updated_at BEFORE UPDATE ON recommendations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_performance_updated_at BEFORE UPDATE ON performance_tracking
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM data_source_cache WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================

-- Query 1: Get top performing recommendations from last 7 days
-- SELECT * FROM performance_tracking pt
-- JOIN recommendations r ON pt.recommendation_id = r.recommendation_id
-- JOIN tokens t ON r.token_id = t.token_id
-- WHERE pt.is_successful = TRUE
--   AND pt.outcome_timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days'
-- ORDER BY pt.actual_return_percent DESC
-- LIMIT 10;

-- Query 2: Get current price data for all tokens in watchlist
-- SELECT t.* FROM tokens t
-- WHERE t.token_id = ANY(
--     SELECT unnest(watchlist) FROM user_preferences WHERE user_id = 'YOUR_USER_ID'
-- );

-- Query 3: Get historical data for a token (last 24 hours, 5-minute intervals)
-- SELECT * FROM historical_data
-- WHERE token_id = 'YOUR_TOKEN_ID'
--   AND timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
-- ORDER BY timestamp DESC;

-- Query 4: Calculate AI model accuracy over time
-- SELECT 
--     DATE(pt.outcome_timestamp) as date,
--     COUNT(*) as total_recommendations,
--     SUM(CASE WHEN pt.is_successful THEN 1 ELSE 0 END) as successful,
--     ROUND(AVG(pt.actual_return_percent), 2) as avg_return
-- FROM performance_tracking pt
-- WHERE pt.outcome_timestamp IS NOT NULL
-- GROUP BY DATE(pt.outcome_timestamp)
-- ORDER BY date DESC;

-- Query 5: Get tokens with bonding curve in preferred range
-- SELECT * FROM tokens
-- WHERE bonding_curve_percent BETWEEN 60 AND 85
--   AND liquidity > 50000
--   AND is_graduated = FALSE
-- ORDER BY bonding_curve_percent DESC;
