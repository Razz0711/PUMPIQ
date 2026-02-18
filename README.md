# NexYpher - AI-Powered Cryptocurrency Trading Recommendation System

## Overview
NexYpher is an intelligent cryptocurrency trading recommendation system that analyzes multiple data sources using AI to generate actionable trading insights.

## System Architecture

### Core Components
1. **Orchestrator Service** - Main coordinator for all system components
2. **Data Collection Modules** - Four independent collectors (News, On-Chain, Technical, Social)
3. **AI Synthesis Engine** - GPT-4o powered analysis and recommendation generation
4. **User API Layer** - REST API for client applications
5. **Configuration Manager** - Flexible user preference system

### Technology Stack
- **Backend**: Python 3.11+ with FastAPI
- **AI/ML**: OpenAI GPT-4o, LangChain
- **Databases**: PostgreSQL (relational data), Redis (caching), TimescaleDB (time-series)
- **Message Queue**: RabbitMQ or Apache Kafka
- **Monitoring**: Prometheus + Grafana
- **Deployment**: Docker + Kubernetes

## Project Structure
```
NexYpher/
├── src/
│   ├── orchestrator/           # Main coordination service
│   ├── data_collectors/        # Independent data collection services
│   │   ├── news_collector/     
│   │   ├── onchain_collector/
│   │   ├── technical_collector/
│   │   └── social_collector/
│   ├── ai_engine/              # AI synthesis and recommendation engine
│   ├── api/                    # User-facing REST API
│   ├── config/                 # Configuration management
│   ├── models/                 # Database models and schemas
│   └── utils/                  # Shared utilities
├── database/
│   ├── schemas/                # SQL and NoSQL schemas
│   └── migrations/             # Database migration scripts
├── config/
│   ├── default_config.json     # Default system configuration
│   └── config_schema.json      # Configuration validation schema
├── tests/                      # Test suites
├── docs/                       # Documentation
├── docker/                     # Docker configurations
└── scripts/                    # Utility scripts
```

## Getting Started
See [docs/setup.md](docs/setup.md) for installation and setup instructions.

## Architecture Philosophy

### Microservices vs Monolithic
- **Microservices** (recommended for production):
  - Data collectors (independent scaling)
  - AI Engine (GPU-optimized instances)
  - API Layer (high availability)
  
- **Monolithic** (acceptable for MVP):
  - Configuration Manager
  - Orchestrator (can split later)

## Data Flow
1. **Collection**: Data collectors continuously fetch data from various sources
2. **Storage**: Raw data stored in PostgreSQL/TimescaleDB with Redis caching
3. **Processing**: Orchestrator triggers AI engine when new data is available
4. **Analysis**: AI engine synthesizes data and generates recommendations
5. **Delivery**: API layer serves recommendations to users
6. **Tracking**: Performance tracker monitors recommendation outcomes

## License
Proprietary
