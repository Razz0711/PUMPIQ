# NexYpher Data Flow Diagram

## High-Level Data Flow

```mermaid
graph TB
    subgraph "External Data Sources"
        NEWS[News APIs]
        ONCHAIN[Blockchain APIs]
        MARKET[Market Data APIs]
        SOCIAL[Social Media APIs]
    end
    
    subgraph "Data Collection Layer"
        NC[News Collector]
        OC[OnChain Collector]
        TC[Technical Collector]
        SC[Social Collector]
    end
    
    subgraph "Caching Layer"
        REDIS[(Redis Cache)]
    end
    
    subgraph "Orchestration Layer"
        ORCH[Orchestrator Service]
        SCHED[Scheduler]
        CACHE[Cache Manager]
    end
    
    subgraph "AI Analysis Layer"
        AGG[Data Aggregator]
        GPT[GPT-4o Engine]
        RECGEN[Recommendation Generator]
    end
    
    subgraph "Persistence Layer"
        PSQL[(PostgreSQL)]
        TSDB[(TimescaleDB)]
    end
    
    subgraph "API Layer"
        API[FastAPI Server]
        AUTH[Authentication]
        RATE[Rate Limiter]
    end
    
    subgraph "Client Applications"
        WEB[Web App]
        MOBILE[Mobile App]
        CLIENT[API Client]
    end
    
    NEWS --> NC
    ONCHAIN --> OC
    MARKET --> TC
    SOCIAL --> SC
    
    NC --> REDIS
    OC --> REDIS
    TC --> REDIS
    SC --> REDIS
    
    REDIS --> ORCH
    
    SCHED --> ORCH
    CACHE --> ORCH
    
    ORCH --> AGG
    AGG --> GPT
    GPT --> RECGEN
    
    RECGEN --> PSQL
    RECGEN --> TSDB
    
    NC --> PSQL
    OC --> TSDB
    TC --> TSDB
    SC --> PSQL
    
    API <--> PSQL
    API <--> REDIS
    API <--> ORCH
    
    AUTH --> API
    RATE --> API
    
    WEB --> API
    MOBILE --> API
    CLIENT --> API
```

## Detailed Recommendation Generation Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Orchestrator
    participant Cache
    participant Collectors
    participant AI Engine
    participant Database
    
    User->>API: POST /recommendations/batch-generate
    API->>API: Authenticate & Rate Limit
    API->>Orchestrator: Request recommendations
    
    Orchestrator->>Cache: Check for cached data
    
    alt Cache Hit
        Cache-->>Orchestrator: Return cached data
    else Cache Miss
        Orchestrator->>Collectors: Request fresh data
        
        par Parallel Data Collection
            Collectors->>Collectors: News Collection
            Collectors->>Collectors: OnChain Collection
            Collectors->>Collectors: Technical Collection
            Collectors->>Collectors: Social Collection
        end
        
        Collectors-->>Orchestrator: Return collected data
        Orchestrator->>Cache: Store in cache
    end
    
    Orchestrator->>AI Engine: Send aggregated data
    AI Engine->>AI Engine: Format prompt with user config
    AI Engine->>AI Engine: Call GPT-4o API
    AI Engine->>AI Engine: Parse response
    AI Engine->>AI Engine: Validate recommendation
    AI Engine-->>Orchestrator: Return recommendation
    
    Orchestrator->>Database: Store recommendation
    Database-->>Orchestrator: Confirm stored
    
    Orchestrator-->>API: Return recommendation
    API-->>User: HTTP 201 Created
    
    Note over Database: Background monitoring starts
```

## Data Collection Flow Details

```mermaid
flowchart TD
    START([Start Data Collection]) --> CHECK_CONFIG{Read User<br/>Config}
    
    CHECK_CONFIG --> |Get Active Sources| SOURCES{Determine<br/>Data Sources}
    
    SOURCES --> |News Enabled| NEWS_FLOW
    SOURCES --> |OnChain Enabled| ONCHAIN_FLOW
    SOURCES --> |Technical Enabled| TECH_FLOW
    SOURCES --> |Social Enabled| SOCIAL_FLOW
    
    NEWS_FLOW[Fetch News Articles] --> NEWS_ANALYZE[Sentiment Analysis]
    NEWS_ANALYZE --> NEWS_CACHE[Cache Results]
    NEWS_CACHE --> NEWS_DB[Store in PostgreSQL]
    
    ONCHAIN_FLOW[Query Blockchain] --> ONCHAIN_PARSE[Parse Metrics]
    ONCHAIN_PARSE --> ONCHAIN_CACHE[Cache Results]
    ONCHAIN_CACHE --> ONCHAIN_DB[Store in TimescaleDB]
    
    TECH_FLOW[Fetch OHLCV Data] --> TECH_CALC[Calculate Indicators]
    TECH_CALC --> TECH_CACHE[Cache Results]
    TECH_CACHE --> TECH_DB[Store in TimescaleDB]
    
    SOCIAL_FLOW[Fetch Social Posts] --> SOCIAL_ANALYZE[Sentiment Analysis]
    SOCIAL_ANALYZE --> SOCIAL_CACHE[Cache Results]
    SOCIAL_CACHE --> SOCIAL_DB[Store in PostgreSQL]
    
    NEWS_DB --> AGGREGATE
    ONCHAIN_DB --> AGGREGATE
    TECH_DB --> AGGREGATE
    SOCIAL_DB --> AGGREGATE
    
    AGGREGATE[Aggregate All Data] --> WEIGHTED{Apply<br/>Weights}
    
    WEIGHTED --> AI_ENGINE[Send to AI Engine]
    AI_ENGINE --> END([End Collection])
```

## Performance Tracking Flow

```mermaid
stateDiagram-v2
    [*] --> RecommendationCreated
    
    RecommendationCreated --> Monitoring: Start Background Monitor
    
    Monitoring --> CheckPrice: Every N seconds
    CheckPrice --> Monitoring: Price not at target/stop
    
    CheckPrice --> Target1Hit: Price >= Target 1
    CheckPrice --> Target2Hit: Price >= Target 2
    CheckPrice --> Target3Hit: Price >= Target 3
    CheckPrice --> StopLossHit: Price <= Stop Loss
    CheckPrice --> Expired: Expiration time reached
    
    Target1Hit --> UpdatePerformance
    Target2Hit --> UpdatePerformance
    Target3Hit --> UpdatePerformance
    StopLossHit --> UpdatePerformance
    Expired --> UpdatePerformance
    
    UpdatePerformance --> CalculateMetrics
    CalculateMetrics --> StoreResults
    StoreResults --> NotifyUser
    NotifyUser --> [*]
```

## Cache Strategy Flow

```mermaid
graph LR
    subgraph "Cache Layers"
        L1[Redis L1<br/>Hot Data<br/>TTL: 1-5 min]
        L2[Application Cache<br/>Processed Data<br/>TTL: 5-10 min]
        DB[(Database<br/>Persistent Storage)]
    end
    
    REQUEST[API Request] --> CHECK_L1{L1 Cache<br/>Hit?}
    
    CHECK_L1 --> |Yes| RETURN_L1[Return from Redis]
    CHECK_L1 --> |No| CHECK_L2{L2 Cache<br/>Hit?}
    
    CHECK_L2 --> |Yes| STORE_L1[Store in Redis]
    CHECK_L2 --> |No| FETCH_DB{Database<br/>Has Data?}
    
    STORE_L1 --> RETURN_L2[Return from L2]
    
    FETCH_DB --> |Yes| STORE_BOTH[Store in Both Caches]
    FETCH_DB --> |No| COLLECT[Collect Fresh Data]
    
    COLLECT --> VALIDATE[Validate Data]
    VALIDATE --> STORE_ALL[Store in All Layers]
    
    STORE_BOTH --> RETURN_DB[Return from DB]
    STORE_ALL --> RETURN_FRESH[Return Fresh Data]
    
    RETURN_L1 --> RESPONSE[HTTP Response]
    RETURN_L2 --> RESPONSE
    RETURN_DB --> RESPONSE
    RETURN_FRESH --> RESPONSE
```

## Configuration Application Flow

```mermaid
flowchart TD
    USER_REQUEST[User API Request] --> GET_USER{Get User ID<br/>from Token}
    
    GET_USER --> LOAD_CONFIG[Load User<br/>Configuration]
    
    LOAD_CONFIG --> MERGE{Merge Configs}
    
    MERGE --> |Layer 1| DEFAULT[Default Config]
    MERGE --> |Layer 2| TIER[Tier Defaults]
    MERGE --> |Layer 3| USER_OVERRIDE[User Overrides]
    
    DEFAULT --> VALIDATE
    TIER --> VALIDATE
    USER_OVERRIDE --> VALIDATE
    
    VALIDATE{Validate Against<br/>JSON Schema} --> |Invalid| ERROR[Return 400<br/>Bad Request]
    
    VALIDATE --> |Valid| CHECK_MODE{Determine<br/>Active Mode}
    
    CHECK_MODE --> |Single Mode| SINGLE[Enable One Source<br/>Weight: 100%]
    CHECK_MODE --> |Hybrid Mode| HYBRID[Enable Multiple<br/>Custom Weights]
    
    SINGLE --> APPLY_CONFIG
    HYBRID --> APPLY_CONFIG
    
    APPLY_CONFIG[Apply to Orchestrator] --> PROCESS[Process Request<br/>with Config]
    
    PROCESS --> RESPONSE[Return Response]
```

## System Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
│  Web UI, Mobile Apps, API Clients, WebSocket Connections   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                      API GATEWAY LAYER                       │
│   Authentication, Rate Limiting, Request Routing, CORS      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                     APPLICATION LAYER                        │
│  FastAPI Endpoints, Request Validation, Response Formatting │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                     ORCHESTRATION LAYER                      │
│   Workflow Coordination, Task Scheduling, Cache Management  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                   DATA COLLECTION LAYER                      │
│    News, OnChain, Technical, Social Data Collectors         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                      AI ANALYSIS LAYER                       │
│   Data Aggregation, GPT-4o Processing, Recommendation Gen   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    PERSISTENCE LAYER                         │
│   PostgreSQL (Relational), TimescaleDB (Time-Series),       │
│   Redis (Cache & Queue)                                      │
└─────────────────────────────────────────────────────────────┘
```
