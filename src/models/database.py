"""
Database Models for PumpIQ using SQLAlchemy
"""

from datetime import datetime
from typing import List, Optional
from decimal import Decimal
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    Text, ForeignKey, Enum as SQLEnum, DECIMAL, ARRAY, JSON,
    CheckConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid
import enum

Base = declarative_base()


# Enums
class RecommendationType(str, enum.Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    AVOID = "AVOID"


class RiskRating(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class RecommendationStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    EXPIRED = "EXPIRED"


class OutcomeType(str, enum.Enum):
    TARGET_1_HIT = "TARGET_1_HIT"
    TARGET_2_HIT = "TARGET_2_HIT"
    TARGET_3_HIT = "TARGET_3_HIT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    STILL_OPEN = "STILL_OPEN"
    EXPIRED = "EXPIRED"


class Timeframe(str, enum.Enum):
    SCALP = "SCALP"
    DAY_TRADE = "DAY_TRADE"
    SWING = "SWING"
    LONG_TERM = "LONG_TERM"


# Models
class Token(Base):
    """Token metadata and current market data"""
    __tablename__ = "tokens"
    
    token_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    ticker = Column(String(20), nullable=False)
    contract_address = Column(String(255), unique=True, nullable=False)
    blockchain = Column(String(50), nullable=False, default='solana')
    
    # Current market data
    current_price = Column(DECIMAL(20, 10))
    market_cap = Column(DECIMAL(20, 2))
    liquidity = Column(DECIMAL(20, 2))
    volume_24h = Column(DECIMAL(20, 2))
    
    # RobinPump specific
    bonding_curve_percent = Column(DECIMAL(5, 2))
    is_graduated = Column(Boolean, default=False)
    graduation_timestamp = Column(DateTime)
    
    # Metadata
    description = Column(Text)
    website = Column(String(500))
    twitter = Column(String(255))
    telegram = Column(String(255))
    logo_url = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_price_update = Column(DateTime)
    
    # Relationships
    recommendations = relationship("Recommendation", back_populates="token")
    historical_data = relationship("HistoricalData", back_populates="token")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            'bonding_curve_percent >= 0 AND bonding_curve_percent <= 100',
            name='valid_bonding_curve'
        ),
        Index('idx_tokens_ticker', 'ticker'),
        Index('idx_tokens_blockchain', 'blockchain'),
        Index('idx_tokens_bonding_curve', 'bonding_curve_percent'),
        Index('idx_tokens_market_cap', 'market_cap'),
        Index('idx_tokens_updated_at', 'updated_at'),
    )
    
    def __repr__(self):
        return f"<Token {self.ticker} ({self.name})>"


class HistoricalData(Base):
    """Time-series data for tokens"""
    __tablename__ = "historical_data"
    
    id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    token_id = Column(UUID(as_uuid=True), ForeignKey('tokens.token_id', ondelete='CASCADE'), nullable=False, primary_key=True)
    timestamp = Column(DateTime, nullable=False, primary_key=True)
    
    # OHLCV data
    open_price = Column(DECIMAL(20, 10))
    high_price = Column(DECIMAL(20, 10))
    low_price = Column(DECIMAL(20, 10))
    close_price = Column(DECIMAL(20, 10))
    volume = Column(DECIMAL(20, 2))
    
    # On-chain metrics
    holder_count = Column(Integer)
    whale_count = Column(Integer)
    top_10_holder_percent = Column(DECIMAL(5, 2))
    total_transactions = Column(Integer)
    buy_transactions = Column(Integer)
    sell_transactions = Column(Integer)
    
    # Whale movements
    whale_buys_1h = Column(Integer, default=0)
    whale_sells_1h = Column(Integer, default=0)
    whale_buy_volume_1h = Column(DECIMAL(20, 2), default=0)
    whale_sell_volume_1h = Column(DECIMAL(20, 2), default=0)
    
    # Sentiment scores
    social_sentiment_score = Column(DECIMAL(5, 2))
    news_sentiment_score = Column(DECIMAL(5, 2))
    
    # Technical indicators
    rsi_14 = Column(DECIMAL(5, 2))
    macd = Column(DECIMAL(20, 10))
    macd_signal = Column(DECIMAL(20, 10))
    bb_upper = Column(DECIMAL(20, 10))
    bb_middle = Column(DECIMAL(20, 10))
    bb_lower = Column(DECIMAL(20, 10))
    
    # Relationships
    token = relationship("Token", back_populates="historical_data")
    
    __table_args__ = (
        Index('idx_historical_token_time', 'token_id', 'timestamp'),
        Index('idx_historical_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<HistoricalData {self.token_id} @ {self.timestamp}>"


class Recommendation(Base):
    """AI-generated trading recommendations"""
    __tablename__ = "recommendations"
    
    recommendation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_id = Column(UUID(as_uuid=True), ForeignKey('tokens.token_id', ondelete='CASCADE'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user_preferences.user_id'))
    
    # Recommendation details
    recommendation_type = Column(SQLEnum(RecommendationType), nullable=False)
    entry_price = Column(DECIMAL(20, 10), nullable=False)
    current_price = Column(DECIMAL(20, 10))
    
    # Targets
    target_price_1 = Column(DECIMAL(20, 10))
    target_price_2 = Column(DECIMAL(20, 10))
    target_price_3 = Column(DECIMAL(20, 10))
    stop_loss = Column(DECIMAL(20, 10))
    
    # Metadata
    confidence_score = Column(DECIMAL(5, 4), nullable=False)
    risk_rating = Column(SQLEnum(RiskRating))
    timeframe = Column(SQLEnum(Timeframe))
    
    # Data source contributions
    news_contribution = Column(Boolean, default=False)
    onchain_contribution = Column(Boolean, default=False)
    technical_contribution = Column(Boolean, default=False)
    social_contribution = Column(Boolean, default=False)
    
    # AI analysis
    ai_reasoning = Column(Text, nullable=False)
    key_factors = Column(JSONB)
    
    # Status
    status = Column(SQLEnum(RecommendationStatus), default=RecommendationStatus.ACTIVE)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    token = relationship("Token", back_populates="recommendations")
    user = relationship("UserPreference", back_populates="recommendations")
    performance = relationship("PerformanceTracking", back_populates="recommendation", uselist=False)
    
    __table_args__ = (
        CheckConstraint(
            'confidence_score >= 0 AND confidence_score <= 1',
            name='valid_confidence'
        ),
        Index('idx_recommendations_token', 'token_id'),
        Index('idx_recommendations_user', 'user_id'),
        Index('idx_recommendations_created', 'created_at'),
        Index('idx_recommendations_confidence', 'confidence_score'),
        Index('idx_recommendations_status', 'status'),
        Index('idx_recommendations_type', 'recommendation_type'),
    )
    
    def __repr__(self):
        return f"<Recommendation {self.recommendation_type} {self.token_id}>"


class PerformanceTracking(Base):
    """Track recommendation performance and outcomes"""
    __tablename__ = "performance_tracking"
    
    tracking_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recommendation_id = Column(UUID(as_uuid=True), ForeignKey('recommendations.recommendation_id', ondelete='CASCADE'), nullable=False)
    
    # Outcome
    outcome = Column(SQLEnum(OutcomeType))
    actual_exit_price = Column(DECIMAL(20, 10))
    actual_return_percent = Column(DECIMAL(10, 4))
    
    # Timing
    time_to_outcome_hours = Column(DECIMAL(10, 2))
    recommendation_created_at = Column(DateTime)
    outcome_timestamp = Column(DateTime)
    
    # Success metrics
    is_successful = Column(Boolean)
    hit_target_1 = Column(Boolean, default=False)
    hit_target_2 = Column(Boolean, default=False)
    hit_target_3 = Column(Boolean, default=False)
    hit_stop_loss = Column(Boolean, default=False)
    
    # Price movement
    max_price_reached = Column(DECIMAL(20, 10))
    min_price_reached = Column(DECIMAL(20, 10))
    max_gain_percent = Column(DECIMAL(10, 4))
    max_drawdown_percent = Column(DECIMAL(10, 4))
    
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    recommendation = relationship("Recommendation", back_populates="performance")
    
    __table_args__ = (
        Index('idx_performance_recommendation', 'recommendation_id'),
        Index('idx_performance_outcome', 'outcome'),
        Index('idx_performance_success', 'is_successful'),
        Index('idx_performance_return', 'actual_return_percent'),
    )
    
    def __repr__(self):
        return f"<PerformanceTracking {self.outcome} for {self.recommendation_id}>"


class UserPreference(Base):
    """User preferences and configuration"""
    __tablename__ = "user_preferences"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True)
    email = Column(String(255), unique=True, nullable=False)
    
    # Configuration (JSON)
    config = Column(JSONB, nullable=False, default={})
    
    # Watchlist
    watchlist = Column(ARRAY(UUID(as_uuid=True)), default=[])
    
    # Portfolio
    portfolio = Column(JSONB, default=[])
    
    # Notification settings
    notification_settings = Column(JSONB)
    
    # Subscription
    subscription_tier = Column(String(50), default='free')
    subscription_expires_at = Column(DateTime)
    
    # Activity
    last_login = Column(DateTime)
    total_recommendations_received = Column(Integer, default=0)
    total_trades_made = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    recommendations = relationship("Recommendation", back_populates="user")
    
    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_subscription', 'subscription_tier'),
    )
    
    def __repr__(self):
        return f"<User {self.email}>"


class DataSourceCache(Base):
    """Cache for external data source responses"""
    __tablename__ = "data_source_cache"
    
    cache_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_identifier = Column(String(255), nullable=False)
    cache_key = Column(String(500), nullable=False)
    
    # Cached data
    data = Column(JSONB, nullable=False)
    
    # TTL
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    ttl_seconds = Column(Integer)
    
    # Metadata
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_cache_source', 'source_identifier'),
        Index('idx_cache_expires', 'expires_at'),
        Index('idx_cache_key', 'cache_key'),
        Index('idx_cache_cleanup', 'expires_at'),  # For cleanup jobs
    )
    
    def __repr__(self):
        return f"<Cache {self.source_identifier}:{self.cache_key}>"


class AnalysisLog(Base):
    """Logs for analysis operations"""
    __tablename__ = "analysis_logs"
    
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_id = Column(UUID(as_uuid=True), ForeignKey('tokens.token_id', ondelete='SET NULL'))
    
    analysis_type = Column(String(50))  # NEWS, ONCHAIN, TECHNICAL, SOCIAL, AI_SYNTHESIS
    
    # Data
    input_data = Column(JSONB)
    output_data = Column(JSONB)
    analysis_result = Column(Text)
    
    # Performance
    execution_time_ms = Column(Integer)
    tokens_used = Column(Integer)  # For AI API calls
    
    # Status
    status = Column(String(20))  # SUCCESS, FAILED, PARTIAL
    error_message = Column(Text)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_analysis_logs_token', 'token_id'),
        Index('idx_analysis_logs_type', 'analysis_type'),
        Index('idx_analysis_logs_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<AnalysisLog {self.analysis_type} @ {self.timestamp}>"
