# PumpIQ — AI-Powered Cryptocurrency Intelligence Platform
### Detailed Project Report

---

**Project Name:** PumpIQ  
**Type:** AI-Powered Crypto Trading Intelligence & Autonomous Trading Platform  
**Tech Stack:** Python (FastAPI), Google Gemini AI, SQLite, Base L2 Blockchain, Vercel  
**Live URL:** [https://pumpiq.vercel.app](https://pumpiq.vercel.app)  
**Date:** February 2026

---

## 1. Executive Summary

PumpIQ is a full-stack AI-powered cryptocurrency intelligence platform that aggregates real-time data from multiple sources — including CoinGecko, DexScreener, crypto news feeds, and social media channels — and synthesizes actionable trading insights using Google Gemini AI. The platform features an autonomous trading bot with built-in risk management, on-chain transaction recording via Base L2 / Ethereum smart contracts, and a complete user authentication system with email verification. It is deployed as a serverless application on Vercel.

---

## 2. System Architecture

```
┌─────────────────────── FRONTEND (Single-Page App) ───────────────────────┐
│   Home  │  Token Feed  │  AI Recs  │  Leaderboard  │  Auto Trader       │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ REST API (53 Endpoints)
┌──────────────────────────────▼───────────────────────────────────────────┐
│                        FastAPI Backend (web_app.py)                       │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │   Auth   │  │  Market   │  │  AI Engine   │  │  Trading Engine   │   │
│  │  (JWT)   │  │  Data API │  │  (Gemini)    │  │  (Autonomous Bot) │   │
│  └──────────┘  └───────────┘  └──────────────┘  └───────────────────┘   │
└──────┬──────────────┬────────────────┬───────────────────┬───────────────┘
       │              │                │                   │
  ┌────▼────┐  ┌──────▼──────┐  ┌─────▼─────┐   ┌────────▼────────┐
  │ SQLite  │  │  CoinGecko  │  │  Gemini   │   │  Blockchain     │
  │  (Auth  │  │ DexScreener │  │  AI API   │   │  (Base L2 /     │
  │   & DB) │  │  News APIs  │  │           │   │   Ethereum)     │
  └─────────┘  └─────────────┘  └───────────┘   └─────────────────┘
```

---

## 3. Core Modules & Features

### 3.1 Data Collection Layer (15 Modules)
| Module                  | Source           | Data Provided                           |
|-------------------------|------------------|-----------------------------------------|
| CoinGecko Collector     | CoinGecko API    | Prices, market cap, volume, sparklines  |
| DexScreener Collector   | DexScreener      | DEX pairs, liquidity, buy/sell ratios   |
| News Collector          | CryptoPanic      | Headlines, sentiment scores             |
| Technical Analyzer      | Computed          | RSI, MACD, support/resistance, trends   |
| Social Collectors (×8)  | Twitter, Reddit, Telegram, Farcaster | Mentions, sentiment, bot filtering |
| RobinPump Collector     | Custom           | Pump/dump pattern detection             |

### 3.2 AI Engine (8 Modules)
- **Gemini Client** — Primary AI integration using Google Gemini 3 Flash for real-time token analysis and recommendations.
- **GPT Client** — Legacy OpenAI GPT-4o fallback support.
- **Orchestrator** — Aggregates multi-source data and coordinates AI analysis pipeline.
- **Confidence Scorer** — Assigns confidence scores (0–100) to each recommendation.
- **Conflict Detector** — Identifies conflicting signals across data sources (e.g., bullish price + bearish social sentiment).
- **NLG Engine** — Generates natural language summaries and verdicts.

### 3.3 Autonomous Trading Engine (684 lines)
The built-in trading bot operates with strict risk management:

| Parameter              | Value                    |
|------------------------|--------------------------|
| Max Trade Size         | 20% of wallet per trade  |
| Daily Loss Limit       | 10% of wallet            |
| Max Open Positions     | 5                        |
| Stop-Loss              | −8% per position         |
| Take-Profit            | +20% per position        |
| Trade Cooldown         | 5 minutes                |
| Min Market Cap Filter  | $1,000,000               |

The bot runs a research pipeline (CoinGecko → DexScreener → Gemini AI) before each trade decision, with full P&L tracking and trade history.

### 3.4 Authentication & Security
- JWT-based authentication with bcrypt password hashing
- Email verification via SMTP (Gmail)
- Password reset flow with secure tokens
- Wallet linking (Ethereum, Solana, Base chains)
- Bank account integration for deposits/withdrawals

### 3.5 Blockchain Integration
- Smart contract on Base L2 (Ethereum L2) for on-chain transaction recording
- Every trade gets a SHA-256 hash + blockchain transaction receipt
- Verifiable transaction audit trail via `TransactionRegistry.sol`

---

## 4. API Overview

The platform exposes **53 REST API endpoints** across 7 domains:

| Domain          | Endpoints | Key Operations                                      |
|-----------------|-----------|------------------------------------------------------|
| Authentication  | 8         | Register, login, email verify, password reset        |
| Wallet & Bank   | 10        | Add/remove wallets, deposits, withdrawals, balance   |
| Market Data     | 5         | Top coins, trending, token feed, detail, DEX search  |
| AI Intelligence | 2         | AI recommendations, conversational bot               |
| Trading         | 11        | Buy, sell, positions, auto-trade, performance, P&L   |
| Portfolio       | 4         | Holdings, watchlist, leaderboard                     |
| Blockchain      | 3         | Verify transactions, status, hash lookup             |

---

## 5. Frontend

Single-page application served as static HTML via FastAPI, featuring:
- **Home Dashboard** — Market overview with trending coins and AI insights
- **Token Feed** — Live DexScreener data with filters (volume, price change, liquidity)
- **AI Recommendations** — Gemini-powered analysis with PumpIQ Score (0–100)
- **Auto Trader** — Toggle autonomous trading with real-time position monitoring
- **Leaderboard** — Top-performing portfolios and traders
- **Conversational AI Bot** — Ask questions about any token or market trend

---

## 6. Project Statistics

| Metric                     | Value         |
|----------------------------|---------------|
| Total Python Files         | 67            |
| Main Application           | 1,949 lines   |
| Trading Engine             | 684 lines     |
| Authentication Module      | 802 lines     |
| Blockchain Service         | 377 lines     |
| Frontend (HTML/CSS/JS)     | 2,935 lines   |
| API Endpoints              | 53            |
| Data Collector Modules     | 15            |
| AI Engine Modules          | 8             |
| UI/UX Modules              | 12            |
| Test Files                 | 5             |
| Deployment                 | Vercel (Serverless) |

---

## 7. Deployment

- **Platform:** Vercel (Serverless Python Functions)
- **Production URL:** [https://pumpiq.vercel.app](https://pumpiq.vercel.app)
- **Database:** SQLite with `/tmp` path for serverless compatibility
- **Environment Variables:** Managed via Vercel dashboard (API keys, secrets)
- **CI/CD:** `vercel --prod` CLI deployment

---

## 8. Future Roadmap

1. **PostgreSQL Migration** — Move from SQLite to a managed PostgreSQL database (Supabase/Neon) for persistent data across serverless invocations.
2. **WebSocket Support** — Real-time price feeds and live trade notifications.
3. **Mobile App** — React Native client consuming the existing API.
4. **Advanced ML Models** — On-chain pattern recognition and whale activity alerts.
5. **Multi-Exchange Support** — Binance, Coinbase, Uniswap integration for live order execution.

---

*Report prepared on February 14, 2026 — PumpIQ v2.0*
