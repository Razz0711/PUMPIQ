# Vercel Deployment Guide

## Prerequisites
- Vercel account connected to your GitHub repository
- Required API keys (see .env.example)

## Vercel Configuration Overview

The `vercel.json` file configures the serverless deployment:

- **Entry Point**: `api/index.py` exports the FastAPI app
- **Static Files**: `/static/*` URLs map to `/web/static/` directory in the repository
- **API Routes**: All other requests route to the FastAPI application
- **Environment**: Production mode is set by default

## Required Environment Variables (Vercel Dashboard)
Configure these in Project Settings → Environment Variables:

### Core Configuration
- `ENVIRONMENT=production`
- `SECRET_KEY=<generate-strong-secret>`
- `ALGORITHM=HS256`

### Database (Use Vercel Postgres or external service)
- `DATABASE_URL=<vercel-postgres-connection-string>`

### API Keys
- `GEMINI_API_KEY=<your-gemini-key>`
- `COINGECKO_API_KEY=<your-coingecko-key>`
- `DEXSCREENER_API_KEY=<your-dexscreener-key>`
- `CRYPTOPANIC_API_KEY=<your-cryptopanic-key>`
- `NEWS_API_KEY=<your-news-api-key>`
- `SOLANA_RPC_URL=<solana-rpc-endpoint>`

### Optional API Keys
- `APIFY_API_KEY=<optional-apify-key>`
- `SOLSCAN_API_KEY=<optional-solscan-key>`
- `ETHERSCAN_API_KEY=<optional-etherscan-key>`

## Deployment Steps
1. Connect repository to Vercel
2. Configure environment variables
3. Deploy: Vercel will auto-detect the configuration
4. Access your app at the provided Vercel URL

## Limitations on Vercel
- **No background tasks**: Auto-trading loop removed (use Vercel Cron Jobs or external scheduler)
- **Execution timeout**: 10s (Hobby), 60s (Pro), 900s (Enterprise)
- **No Redis/Celery**: Use Vercel KV or external Redis if needed
- **Stateless**: Each request may hit a different serverless instance

## Database Recommendations
- **Vercel Postgres**: Integrated PostgreSQL database
- **Supabase**: Free tier with PostgreSQL
- **PlanetScale**: Serverless MySQL alternative
- **MongoDB Atlas**: Cloud MongoDB (if using MongoDB schemas)

## Next Steps
- Set up Vercel Cron Jobs for periodic tasks (market data updates, etc.)
- Consider using Vercel KV (Redis) for caching
- Monitor function execution times and optimize slow endpoints

## Vercel Cron Jobs (Optional)
To enable periodic auto-trading, add this to `vercel.json`:

```json
{
  "crons": [
    {
      "path": "/api/cron/auto-trade",
      "schedule": "*/5 * * * *"
    }
  ]
}
```

Then create an endpoint in `web_app.py`:

**Note**: The example below is pseudo-code for illustration. In production, you should add a public method to `trading_engine` to get enabled users instead of using the private `_get_db()` method.

```python
@app.get("/api/cron/auto-trade")
async def cron_auto_trade(request: Request):
    # Verify the request is from Vercel Cron
    # Vercel sends the CRON_SECRET in the Authorization header
    auth_header = request.headers.get("Authorization")
    expected_auth = f"Bearer {os.getenv('CRON_SECRET')}"
    
    if not auth_header or auth_header != expected_auth:
        raise HTTPException(401, "Unauthorized")
    
    # Run auto-trade for all enabled users
    _ensure_collectors_initialized()
    
    # PSEUDO-CODE: Add a public method like trading_engine.get_enabled_users()
    # instead of using the private _get_db() method shown below
    conn = trading_engine._get_db()
    enabled_users = conn.execute(
        "SELECT user_id FROM trade_settings WHERE auto_trade_enabled = 1"
    ).fetchall()
    conn.close()
    
    results = []
    for row in enabled_users:
        try:
            result = await trading_engine.auto_trade_cycle(
                row["user_id"], cg, dex, gemini_client
            )
            results.append({"user_id": row["user_id"], "success": True, "result": result})
        except Exception as e:
            results.append({"user_id": row["user_id"], "success": False, "error": str(e)})
    
    return {"status": "completed", "results": results}
```

Don't forget to set `CRON_SECRET` in Vercel environment variables for security.
