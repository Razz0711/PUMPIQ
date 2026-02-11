# PumpIQ - Setup Guide

## Prerequisites

- Python 3.11 or higher
- PostgreSQL 15+ with TimescaleDB extension
- Redis 7.0+
- OpenAI API key (GPT-4o access)
- API keys for data sources (News APIs, Blockchain explorers, etc.)

## Installation Steps

### 1. Clone the Repository

```bash
cd "C:\Users\RAJ\OneDrive\Documents\Desktop\PumpIQ"
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up PostgreSQL Database

#### Install PostgreSQL and TimescaleDB

- Download and install PostgreSQL 15+ from https://www.postgresql.org/download/
- Install TimescaleDB extension: https://docs.timescale.com/install/latest/

#### Create Database

```bash
psql -U postgres

CREATE DATABASE pumpiq;
\c pumpiq

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

\q
```

#### Run Schema

```bash
psql -U postgres -d pumpiq -f database/schemas/postgresql_schema.sql
```

### 5. Set Up Redis

#### Install Redis

**Windows:**
- Download Redis from https://github.com/microsoftarchive/redis/releases
- Or use WSL and install via: `sudo apt-get install redis-server`

**macOS:**
```bash
brew install redis
brew services start redis
```

**Linux:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

### 6. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/pumpiq
REDIS_URL=redis://localhost:6379/0

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Data Source APIs
NEWS_API_KEY=your_news_api_key
COINGECKO_API_KEY=your_coingecko_api_key
SOLSCAN_API_KEY=your_solscan_api_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Security
SECRET_KEY=your_secret_key_for_jwt
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Environment
ENVIRONMENT=development  # development, staging, production

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

### 7. Initialize Database Migrations

```bash
# Initialize Alembic
alembic init alembic

# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

### 8. Test the Setup

```bash
# Test configuration loading
python -m src.config.config_manager

# Test database connection
python -c "from src.models.database import Base; print('Database models loaded successfully')"
```

## Running the Application

### Development Mode

#### Start the API Server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Start Celery Worker (for background tasks)

```bash
celery -A src.orchestrator.celery_app worker --loglevel=info
```

#### Start Celery Beat (for scheduled tasks)

```bash
celery -A src.orchestrator.celery_app beat --loglevel=info
```

### Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Project Structure

```
PumpIQ/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py            # API entry point
│   │   ├── routes/            # API route handlers
│   │   └── middleware/        # Custom middleware
│   ├── orchestrator/          # Main coordinator
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── scheduler.py
│   │   └── cache_manager.py
│   ├── data_collectors/       # Data collection modules
│   │   ├── news_collector/
│   │   ├── onchain_collector/
│   │   ├── technical_collector/
│   │   └── social_collector/
│   ├── ai_engine/             # AI analysis engine
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   └── prompts.py
│   ├── config/                # Configuration management
│   │   ├── __init__.py
│   │   └── config_manager.py
│   ├── models/                # Database models
│   │   ├── __init__.py
│   │   └── database.py
│   └── utils/                 # Shared utilities
├── database/
│   ├── schemas/               # Database schemas
│   └── migrations/            # Alembic migrations
├── config/                    # Configuration files
├── tests/                     # Test suites
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables
```

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test Module

```bash
pytest tests/test_config.py
```

## Deployment

### Docker Deployment

#### Build Docker Image

```bash
docker build -t pumpiq:latest .
```

#### Run with Docker Compose

```bash
docker-compose up -d
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

## Monitoring

### Access Monitoring Tools

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Flower (Celery monitoring)**: http://localhost:5555

## Troubleshooting

### Database Connection Issues

```bash
# Test PostgreSQL connection
psql -U postgres -d pumpiq -c "SELECT version();"

# Check if TimescaleDB is loaded
psql -U postgres -d pumpiq -c "SELECT default_version FROM pg_available_extensions WHERE name='timescaledb';"
```

### Redis Connection Issues

```bash
# Test Redis connection
redis-cli ping
# Should return: PONG
```

### API Not Starting

1. Check if port 8000 is already in use
2. Verify all environment variables are set
3. Check logs for specific error messages

### Celery Workers Not Processing Tasks

```bash
# Check Redis connection
redis-cli
> KEYS *

# Restart Celery workers
celery -A src.orchestrator.celery_app purge  # Clear queue
celery -A src.orchestrator.celery_app worker --loglevel=debug
```

## Next Steps

1. Configure your API keys in `.env`
2. Run database migrations
3. Start the API server
4. Visit http://localhost:8000/docs to explore the API
5. Create a user account
6. Configure your preferences
7. Start receiving recommendations!

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run tests: `pytest`
4. Format code: `black src/`
5. Lint: `flake8 src/`
6. Commit and push
7. Create pull request

## Support

For issues and questions:
- Check the documentation in `docs/`
- Review API specification in `docs/api_specification.yaml`
- Read architecture guide in `docs/architecture.md`
