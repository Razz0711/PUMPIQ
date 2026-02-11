# System Architecture Documentation

## Overview

PumpIQ is an AI-powered cryptocurrency trading recommendation system that aggregates data from multiple sources, synthesizes insights using GPT-4o, and provides actionable trading recommendations to users.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT APPLICATIONS                            │
│                    (Web App, Mobile App, API Clients)                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ HTTPS/REST
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY                                   │
│              (Rate Limiting, Authentication, Load Balancing)             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API LAYER (FastAPI)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Tokens     │  │Recommendations│  │  Analysis   │  │  User Mgmt  │ │
│  │  Endpoints   │  │  Endpoints    │  │  Endpoints  │  │  Endpoints  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR SERVICE                             │
│         (Coordination, Workflow Management, Cache Strategy)              │
└──┬──────────────┬──────────────┬──────────────┬────────────────────────┘
   │              │              │              │
   │              │              │              │
   ▼              ▼              ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│  News    │ │ On-Chain │ │Technical │ │   Social     │
│Collector │ │Collector │ │Collector │ │  Collector   │
└─────┬────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
      │           │            │               │
      │           │            │               │
      └───────────┴────────────┴───────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Redis Cache    │
                    └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI SYNTHESIS ENGINE                              │
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│  │  Data Aggregator │ -> │  GPT-4o Analysis │ -> │  Recommendation  │ │
│  │                  │    │                  │    │    Generator     │ │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PERSISTENCE LAYER                            │
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│  │   PostgreSQL     │    │   TimescaleDB    │    │   Redis Cache    │ │
│  │ (Relational Data)│    │ (Time-Series)    │    │ (Hot Data/Queue) │ │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MONITORING & LOGGING                                │
│              (Prometheus, Grafana, ELK Stack)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Data Collection Flow
```
External APIs → Data Collectors → Validation → Redis Cache → PostgreSQL/TimescaleDB
     │                                                              │
     ├─ News APIs (CoinDesk, CryptoNews)                          │
     ├─ On-Chain APIs (Solscan, Etherscan)                        │
     ├─ Market APIs (CoinGecko, DexScreener)                      │
     └─ Social APIs (Twitter, Reddit)                             │
                                                                   │
                        Results stored & cached ─────────────────┘
```

### 2. Analysis & Recommendation Flow
```
User Request → API Layer → Orchestrator
                              │
                              ├─ Check Cache (Redis)
                              │     └─ If cached & fresh → Return
                              │
                              ├─ Fetch Multi-Source Data
                              │     ├─ News Collector
                              │     ├─ On-Chain Collector
                              │     ├─ Technical Collector
                              │     └─ Social Collector
                              │
                              ├─ Aggregate Data
                              │
                              ├─ Send to AI Engine
                              │     ├─ Format prompt with user preferences
                              │     ├─ Call GPT-4o API
                              │     └─ Parse AI response
                              │
                              ├─ Generate Recommendation
                              │     ├─ Calculate targets & stop loss
                              │     ├─ Assign confidence score
                              │     └─ Determine risk rating
                              │
                              ├─ Store in Database
                              └─ Return to User
```

### 3. Performance Tracking Flow
```
Recommendation Created
     │
     ├─ Background Monitor starts
     │     │
     │     ├─ Check price every N seconds
     │     ├─ Compare to targets/stop loss
     │     └─ Update tracking metrics
     │
     ├─ Target Hit OR Stop Loss Hit OR Expiration
     │     │
     │     └─ Update Performance Table
     │           ├─ Calculate actual return
     │           ├─ Record outcome
     │           ├─ Update success metrics
     │           └─ Notify user
     │
     └─ Aggregate for ML Model Improvement
```

## Component Details

### 1. Orchestrator Service

**Responsibilities:**
- Coordinate between data collectors
- Manage data collection schedules based on user config
- Cache management and invalidation
- Workflow orchestration for AI analysis
- Background task scheduling

**Technology:**
- Python 3.11+
- Celery for task queue
- Redis for message broker
- APScheduler for scheduling

**Key Files:**
- `src/orchestrator/main.py` - Main orchestrator logic
- `src/orchestrator/scheduler.py` - Task scheduling
- `src/orchestrator/cache_manager.py` - Cache strategy
- `src/orchestrator/workflow.py` - Analysis workflows

### 2. Data Collectors

Each collector is an independent microservice that can be scaled separately.

#### News Collector
- **Sources:** CoinDesk, CryptoPanic, NewsAPI, Google News
- **Metrics:** Sentiment score, relevance, credibility
- **Update Frequency:** 5-10 minutes
- **Storage:** Raw articles in PostgreSQL, sentiment scores in cache

#### On-Chain Collector
- **Sources:** Solscan, Etherscan, DexScreener, Pump.fun API
- **Metrics:** Holder count, whale movements, liquidity, bonding curve
- **Update Frequency:** 5-10 minutes
- **Storage:** TimescaleDB for historical metrics

#### Technical Collector
- **Sources:** TradingView, Exchange APIs (Binance, Coinbase)
- **Metrics:** RSI, MACD, Bollinger Bands, volume, support/resistance
- **Update Frequency:** 1-5 minutes
- **Storage:** TimescaleDB for OHLCV and indicators

#### Social Collector
- **Sources:** Twitter API, Reddit API, Telegram (if possible)
- **Metrics:** Mention volume, sentiment, influencer activity
- **Update Frequency:** 3-5 minutes
- **Storage:** PostgreSQL for posts, aggregated scores in cache

### 3. AI Synthesis Engine

**Responsibilities:**
- Aggregate data from multiple sources
- Format prompts based on user preferences and mode selection
- Call GPT-4o API with structured prompts
- Parse and validate AI responses
- Generate structured recommendations

**Prompt Engineering Strategy:**
```python
# Example prompt structure
f"""
You are a cryptocurrency trading analyst. Analyze the following data for {token_name} ({ticker}):

USER PROFILE:
- Risk Tolerance: {risk_tolerance}
- Timeframe: {timeframe}
- Portfolio Size: {portfolio_size}

DATA SOURCES (Weight: {weight}%):
{enabled_sources}

NEWS DATA:
{news_summary}

ON-CHAIN DATA:
{onchain_metrics}

TECHNICAL DATA:
{technical_indicators}

SOCIAL DATA:
{social_sentiment}

Provide a recommendation in the following JSON format:
{{
  "recommendation_type": "BUY|HOLD|SELL|AVOID",
  "confidence_score": 0.0-1.0,
  "entry_price": number,
  "targets": [target1, target2, target3],
  "stop_loss": number,
  "reasoning": "detailed explanation",
  "key_factors": [{{factor, impact, weight}}],
  "risk_rating": "LOW|MEDIUM|HIGH|VERY_HIGH"
}}
"""
```

**Technology:**
- OpenAI GPT-4o API
- LangChain for prompt templates
- Pydantic for response validation
- Retry logic with exponential backoff

### 4. API Layer

**Framework:** FastAPI
- High performance async support
- Automatic OpenAPI documentation
- Built-in validation with Pydantic
- WebSocket support for real-time updates

**Features:**
- JWT authentication
- Rate limiting (per user tier)
- Request validation
- Response caching
- CORS configuration
- API versioning

### 5. Configuration Management

**How it Works:**
1. User preferences stored in `user_preferences` table
2. Orchestrator reads config on each analysis request
3. Config determines:
   - Which data collectors to invoke
   - Weight distribution in hybrid mode
   - Cache TTL values
   - Refresh intervals
   - AI model parameters

**Config Priority:**
- User-specific config (highest)
- Account tier defaults
- System defaults (lowest)

### 6. Database Strategy

**PostgreSQL (Primary Relational Store):**
- Tokens metadata
- User accounts and preferences
- Recommendations
- Performance tracking
- Analysis logs

**TimescaleDB (Time-Series Extension):**
- Historical OHLCV data
- On-chain metrics over time
- Sentiment scores over time
- Technical indicators time-series

**Redis (Cache & Queue):**
- API response caching
- Data collector results (hot data)
- Celery task queue
- Rate limiting counters
- Session storage

**Backup Strategy:**
- Daily PostgreSQL snapshots
- Continuous WAL archiving
- Redis persistence (AOF + RDB)
- S3 backup storage

## Scalability Considerations

### Horizontal Scaling

**Can Scale Horizontally (Stateless):**
- API Layer (multiple instances behind load balancer)
- Data Collectors (one per source, can duplicate)
- AI Engine workers (multiple workers for parallel processing)

**Requires Coordination (Stateful):**
- Orchestrator (use leader election if multiple instances)
- Database (read replicas for scaling reads)

### Vertical Scaling

**CPU-Intensive:**
- Technical analysis calculations
- Sentiment analysis processing

**Memory-Intensive:**
- Redis cache (increase for more cached data)
- AI Engine (model loading)

**I/O-Intensive:**
- Database queries on large historical datasets
- API calls to external services

### Load Distribution

```
                    Load Balancer (nginx)
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      API Instance 1  API Instance 2  API Instance 3
            │               │               │
            └───────────────┼───────────────┘
                            ▼
                   PostgreSQL (Primary)
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     Read Replica 1  Read Replica 2  Read Replica 3
```

## Microservices Deployment

**Recommended Production Setup:**

### Microservices:
1. **api-service** - FastAPI application
2. **orchestrator-service** - Main coordinator
3. **news-collector-service** - News data collection
4. **onchain-collector-service** - On-chain data collection
5. **technical-collector-service** - Technical analysis
6. **social-collector-service** - Social sentiment
7. **ai-engine-service** - AI recommendation generation

### Infrastructure:
- **Container Orchestration:** Kubernetes
- **Service Mesh:** Istio (for inter-service communication)
- **Message Queue:** RabbitMQ or Apache Kafka
- **Monitoring:** Prometheus + Grafana
- **Logging:** ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing:** Jaeger

## Security Considerations

### Authentication & Authorization
- JWT tokens with 1-hour expiration
- Refresh token mechanism
- Role-based access control (RBAC)
- API key for service-to-service communication

### Data Protection
- Encryption at rest (database encryption)
- Encryption in transit (TLS 1.3)
- API secrets in environment variables
- Vault for secret management

### Rate Limiting
- Per-user tier limits
- DDoS protection at API gateway
- Redis-based rate limiting

### Validation & Sanitization
- Input validation on all endpoints
- SQL injection protection (parameterized queries)
- XSS protection
- CSRF tokens for state-changing operations

## Monitoring & Alerting

### Key Metrics to Monitor

**Application Metrics:**
- API response times (p50, p95, p99)
- Request rate and error rate
- AI API call latency and cost
- Data collector success/failure rates
- Recommendation generation time

**System Metrics:**
- CPU, memory, disk usage
- Database connection pool
- Redis memory usage
- Queue depth (Celery)

**Business Metrics:**
- Recommendation accuracy (win rate)
- Average return per recommendation
- User engagement (API calls, recommendations viewed)
- Data freshness (time since last update)

### Alerts

**Critical:**
- Database connection failures
- AI API failures (OpenAI down)
- Authentication service down
- Multiple data collectors failing

**Warning:**
- High API latency (>2s)
- Cache hit rate below 70%
- Database CPU >80%
- High error rate (>5%)

## Development vs Production

### Development (Monolithic):
- All services in one Python application
- SQLite for quick setup (optional)
- File-based logging
- Minimal caching

### Production (Microservices):
- Separate Docker containers per service
- PostgreSQL + TimescaleDB + Redis cluster
- Centralized logging
- Multi-layer caching
- CDN for static assets
- Auto-scaling based on load

## Future Enhancements

### Phase 2:
- WebSocket support for real-time price updates
- Mobile app (React Native)
- Portfolio backtesting
- Custom indicator builder

### Phase 3:
- Machine learning model for pattern recognition
- Automated trade execution (with user approval)
- Multi-chain support (Ethereum, BSC, Polygon)
- Community sharing of successful recommendations

### Phase 4:
- Copy-trading features
- AI model fine-tuning based on performance data
- Advanced charting with TradingView integration
- Token launch prediction model
