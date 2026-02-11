# Configuration Management Guide

## Overview

PumpIQ uses a flexible configuration system that allows users to customize their trading recommendations by enabling/disabling data sources and adjusting various parameters.

## Configuration Files

### 1. `config/default_config.json`
The default system configuration that all users start with. This file contains:
- Default mode settings
- Standard user preferences
- System-wide settings
- Database configurations

### 2. `config/config_schema.json`
JSON Schema that validates all configuration updates. Ensures:
- Required fields are present
- Values are within acceptable ranges
- Data types are correct
- Enum values are valid

## Configuration Modes

### Single-Source Modes

When a single mode is enabled, only that data source is used for analysis:

#### NEWS_ONLY_MODE
- **Use Case:** Users who want recommendations based purely on market news and announcements
- **Weight:** 100% news sentiment
- **Best For:** Long-term investors who follow fundamental news
- **Update Frequency:** Every 5 minutes

#### ONCHAIN_ONLY_MODE
- **Use Case:** Users who trust blockchain data above all else
- **Weight:** 100% on-chain metrics
- **Best For:** Whale watchers, liquidity analysts
- **Update Frequency:** Every 10 minutes

#### TECHNICAL_ONLY_MODE
- **Use Case:** Traditional chart analysts, day traders
- **Weight:** 100% technical indicators
- **Best For:** Scalpers, day traders
- **Update Frequency:** Every 1 minute

#### SOCIAL_ONLY_MODE
- **Use Case:** Users who follow social media trends
- **Weight:** 100% social sentiment
- **Best For:** Meme coin traders, trend followers
- **Update Frequency:** Every 3 minutes

### Hybrid Mode (Recommended)

Combines multiple data sources with custom weights:

```json
{
  "hybrid_mode": {
    "enabled": true,
    "weights": {
      "news": 25,      // 25% weight to news
      "onchain": 35,   // 35% weight to on-chain data
      "technical": 25, // 25% weight to technical analysis
      "social": 15     // 15% weight to social sentiment
    }
  }
}
```

**Requirements:**
- At least 2 sources must have weight > 0
- Total weights must sum to 100
- Individual weights: 0-100

## User Preferences

### Risk Tolerance

Affects recommendation filtering and position sizing:

- **conservative**: 
  - Only high-confidence recommendations (>0.80)
  - Lower position sizes
  - Tighter stop losses
  - Prefers established tokens

- **moderate** (default):
  - Medium confidence threshold (>0.65)
  - Standard position sizing
  - Balanced risk/reward

- **aggressive**:
  - Lower confidence threshold (>0.50)
  - Larger position sizes
  - Wider stop losses
  - Open to new/volatile tokens

### Investment Timeframe

Influences recommendation type and analysis period:

- **scalping**: 
  - Very short-term (minutes to hours)
  - Focuses on technical analysis
  - Tight targets and stops

- **day_trading**:
  - Intraday positions
  - Mix of technical and news
  - Closes positions by EOD

- **swing_trading** (default):
  - Multi-day to week-long holds
  - Balanced analysis
  - Larger price targets

- **long_term**:
  - Weeks to months
  - Heavy fundamental focus
  - Patient approach

### Portfolio Size Category

Affects recommendation suitability:

- **small** (<$1,000):
  - Fewer, higher-conviction recommendations
  - Avoids extremely illiquid tokens
  - Focus on quality over quantity

- **medium** ($1,000-$10,000):
  - Balanced recommendation count
  - Moderate diversification

- **large** (>$10,000):
  - More diversified recommendations
  - Can handle less liquid tokens
  - Higher position sizing flexibility

## System Settings

### Data Refresh Intervals

Control how frequently each collector fetches new data (in seconds):

```json
{
  "data_refresh_intervals": {
    "news_collector": 300,      // 5 minutes
    "onchain_collector": 600,   // 10 minutes
    "technical_collector": 60,  // 1 minute
    "social_collector": 180,    // 3 minutes
    "ai_analysis": 900          // 15 minutes
  }
}
```

**Considerations:**
- Shorter intervals = more API calls = higher costs
- Longer intervals = less fresh data = potentially missed opportunities
- Balance based on subscription tier

### Cache TTL Settings

Cache time-to-live values (in seconds):

```json
{
  "cache_settings": {
    "news_cache_ttl": 600,           // 10 minutes
    "onchain_cache_ttl": 900,        // 15 minutes
    "technical_cache_ttl": 300,      // 5 minutes
    "social_cache_ttl": 450,         // 7.5 minutes
    "api_response_cache_ttl": 60     // 1 minute
  }
}
```

**Cache Strategy:**
- Short TTL for volatile data (price, technical)
- Longer TTL for stable data (on-chain, news)
- API responses cached briefly to handle duplicate requests

### API Rate Limits

Per-user tier rate limits:

```json
{
  "api_settings": {
    "rate_limits": {
      "requests_per_minute": 60,
      "requests_per_hour": 1000,
      "requests_per_day": 10000
    }
  }
}
```

| Tier    | RPM | Per Hour | Per Day |
|---------|-----|----------|---------|
| Free    | 60  | 1,000    | 10,000  |
| Basic   | 120 | 5,000    | 50,000  |
| Premium | 300 | 20,000   | 200,000 |
| Pro     | 600 | 50,000   | 500,000 |

## AI Engine Configuration

Fine-tune GPT-4o parameters:

```json
{
  "ai_engine": {
    "model": "gpt-4o",
    "temperature": 0.7,        // Creativity (0=deterministic, 1=creative)
    "max_tokens": 2000,        // Response length limit
    "top_p": 0.9,              // Nucleus sampling
    "frequency_penalty": 0.0,  // Reduce repetition
    "presence_penalty": 0.0    // Encourage new topics
  }
}
```

**Parameter Guide:**
- **temperature**: 0.5-0.7 for consistent analysis, 0.8-1.0 for creative insights
- **max_tokens**: 1500-2500 for detailed recommendations
- **top_p**: Keep at 0.9-0.95 for balanced responses

## Trading Parameters

Risk management settings:

```json
{
  "trading_parameters": {
    "max_position_size_percent": 10,  // Max % of portfolio per trade
    "stop_loss_percent": 5,           // Default stop loss
    "take_profit_levels": [10, 20, 35], // Target levels
    "min_market_cap": 100000,         // Minimum token market cap
    "max_market_cap": null,           // No maximum
    "min_liquidity": 50000            // Minimum liquidity
  }
}
```

## RobinPump-Specific Settings

For RobinPump tokens with bonding curves:

```json
{
  "robinpump_specific": {
    "min_bonding_curve_percent": 30,    // Avoid very early stage
    "max_bonding_curve_percent": 95,    // Avoid nearly graduated
    "preferred_bonding_range": [60, 85] // Sweet spot range
  }
}
```

**Bonding Curve Guide:**
- **0-30%**: Very early, extremely high risk
- **30-60%**: Early stage, high risk/reward
- **60-85%**: Optimal range (momentum + liquidity)
- **85-95%**: Late stage, lower upside potential
- **95-100%**: About to graduate, different dynamics

## Notification Settings

Control how users receive recommendations:

```json
{
  "notification_settings": {
    "email_enabled": true,
    "push_enabled": false,
    "sms_enabled": false,
    "notification_frequency": "realtime",  // realtime|hourly|daily|off
    "min_confidence_for_notification": 0.75
  }
}
```

## How Orchestrator Reads Configuration

### 1. Configuration Loading

```python
# Pseudocode
def get_user_config(user_id):
    # 1. Get user-specific config from database
    user_config = db.get_user_preferences(user_id).config
    
    # 2. Merge with tier defaults
    tier_config = get_tier_defaults(user.subscription_tier)
    merged_config = merge_configs(tier_config, user_config)
    
    # 3. Validate against schema
    validate_config(merged_config, config_schema)
    
    return merged_config
```

### 2. Mode Application

```python
def determine_active_sources(config):
    active_sources = []
    weights = {}
    
    # Check if any single mode is enabled
    if config['modes']['news_only_mode']['enabled']:
        return ['news'], {'news': 100}
    # ... check other single modes
    
    # Default to hybrid mode
    if config['modes']['hybrid_mode']['enabled']:
        hybrid_weights = config['modes']['hybrid_mode']['weights']
        for source, weight in hybrid_weights.items():
            if weight > 0:
                active_sources.append(source)
                weights[source] = weight
    
    return active_sources, weights
```

### 3. Data Collection Orchestration

```python
def collect_data_for_token(token_id, config):
    active_sources, weights = determine_active_sources(config)
    data = {}
    
    for source in active_sources:
        collector = get_collector(source)
        cache_ttl = config['cache_settings'][f'{source}_cache_ttl']
        
        # Check cache first
        cached_data = redis.get(f'{source}:{token_id}')
        if cached_data and not expired(cached_data, cache_ttl):
            data[source] = cached_data
        else:
            # Fetch fresh data
            fresh_data = collector.fetch(token_id)
            redis.set(f'{source}:{token_id}', fresh_data, ttl=cache_ttl)
            data[source] = fresh_data
    
    return data, weights
```

## Configuration Update API

### Example: Update User Configuration

```bash
curl -X PUT https://api.pumpiq.com/v1/user/preferences \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "modes": {
      "hybrid_mode": {
        "enabled": true,
        "weights": {
          "news": 20,
          "onchain": 40,
          "technical": 30,
          "social": 10
        }
      }
    },
    "user_preferences": {
      "risk_tolerance": "aggressive",
      "investment_timeframe": "day_trading",
      "min_confidence_score": 0.60
    }
  }'
```

## Validation Rules

The system validates:

1. **Mode Exclusivity**: Only one single mode OR hybrid mode can be active
2. **Weight Sum**: Hybrid mode weights must sum to 100
3. **Value Ranges**: All numeric values within acceptable ranges
4. **Required Fields**: Essential configuration fields must be present
5. **Type Checking**: All values match expected data types

## Best Practices

### For Conservative Traders:
```json
{
  "modes": {"hybrid_mode": {"enabled": true, "weights": {"news": 30, "onchain": 40, "technical": 20, "social": 10}}},
  "user_preferences": {
    "risk_tolerance": "conservative",
    "investment_timeframe": "swing_trading",
    "min_confidence_score": 0.75
  },
  "trading_parameters": {
    "max_position_size_percent": 5,
    "stop_loss_percent": 3
  }
}
```

### For Aggressive Traders:
```json
{
  "modes": {"hybrid_mode": {"enabled": true, "weights": {"news": 15, "onchain": 25, "technical": 40, "social": 20}}},
  "user_preferences": {
    "risk_tolerance": "aggressive",
    "investment_timeframe": "day_trading",
    "min_confidence_score": 0.55
  },
  "trading_parameters": {
    "max_position_size_percent": 15,
    "stop_loss_percent": 7
  }
}
```

### For Social Trend Followers:
```json
{
  "modes": {"social_only_mode": {"enabled": true, "weight": 100}},
  "user_preferences": {
    "risk_tolerance": "aggressive",
    "investment_timeframe": "scalping",
    "min_confidence_score": 0.50
  }
}
```
