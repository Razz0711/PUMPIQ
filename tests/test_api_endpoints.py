"""
Integration Tests – API Endpoints
====================================
Step 5.2 – End-to-end tests exercising the FastAPI routes via TestClient.

These tests use the sync ``TestClient`` (no real GPT / Redis / DB calls).
The Orchestrator returns an empty set when there is no data_fetcher,
so we verify status codes, response schemas, and error handling.
"""

from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════════════
# Health Endpoints
# ══════════════════════════════════════════════════════════════════

class TestHealthEndpoints:

    def test_health_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data

    def test_deep_health(self, client):
        resp = client.get("/api/v1/health/deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert isinstance(data["components"], list)

    def test_deep_health_lists_components(self, client):
        resp = client.get("/api/v1/health/deep")
        names = [c["name"] for c in resp.json()["components"]]
        assert "redis" in names
        assert "database" in names
        assert "ai_client" in names


# ══════════════════════════════════════════════════════════════════
# Recommendations Endpoint
# ══════════════════════════════════════════════════════════════════

class TestRecommendationsEndpoint:

    def test_post_recommendations_200(self, client):
        resp = client.post(
            "/api/v1/recommendations",
            json={"query": "What are the best coins to buy?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert "metadata" in data

    def test_recommendations_with_params(self, client):
        resp = client.post(
            "/api/v1/recommendations",
            json={
                "query": "best coins to buy",
                "num_recommendations": 5,
                "timeframe": "swing",
                "risk_preference": "conservative",
                "modes": ["news", "onchain"],
            },
        )
        assert resp.status_code == 200

    def test_recommendations_short_query_422(self, client):
        resp = client.post(
            "/api/v1/recommendations",
            json={"query": "ab"},  # min_length=3
        )
        assert resp.status_code == 422

    def test_recommendations_missing_query_422(self, client):
        resp = client.post("/api/v1/recommendations", json={})
        assert resp.status_code == 422

    def test_recommendations_response_schema(self, client):
        resp = client.post(
            "/api/v1/recommendations",
            json={"query": "best coins to buy right now"},
        )
        data = resp.json()
        assert "query_timestamp" in data
        assert "market_context" in data
        assert isinstance(data["recommendations"], list)


# ══════════════════════════════════════════════════════════════════
# Token Analysis Endpoint
# ══════════════════════════════════════════════════════════════════

class TestAnalysisEndpoint:

    def test_analyze_token_path(self, client):
        """GET /api/v1/token/BONK/analysis should not 500."""
        resp = client.get("/api/v1/token/BONK/analysis")
        # May 404 (token not found) or 200 depending on data
        assert resp.status_code in (200, 404)

    def test_analyze_with_modes(self, client):
        resp = client.get("/api/v1/token/SOL/analysis?modes=news,onchain")
        assert resp.status_code in (200, 404)

    def test_analyze_invalid_ticker(self, client):
        resp = client.get("/api/v1/token//analysis")
        assert resp.status_code in (404, 422)  # empty path param


# ══════════════════════════════════════════════════════════════════
# Comparison Endpoint
# ══════════════════════════════════════════════════════════════════

class TestComparisonEndpoint:

    def test_compare_tokens(self, client):
        resp = client.post(
            "/api/v1/compare",
            json={"tickers": ["SOL", "BONK"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "tokens" in data
        assert "highlights" in data

    def test_compare_single_ticker_422(self, client):
        """At least 2 tickers required."""
        resp = client.post(
            "/api/v1/compare",
            json={"tickers": ["SOL"]},
        )
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════
# Watchlist Endpoint
# ══════════════════════════════════════════════════════════════════

class TestWatchlistEndpoint:

    def test_add_to_watchlist(self, client):
        resp = client.post(
            "/api/v1/watchlist",
            json={"ticker": "BONK", "target_price": 0.01},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1

    def test_list_watchlist(self, client):
        resp = client.get("/api/v1/watchlist")
        assert resp.status_code == 200
        assert "items" in resp.json()


# ══════════════════════════════════════════════════════════════════
# Portfolio Endpoint
# ══════════════════════════════════════════════════════════════════

class TestPortfolioEndpoint:

    def test_get_portfolio(self, client):
        resp = client.get("/api/v1/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_value" in data
        assert "positions" in data


# ══════════════════════════════════════════════════════════════════
# Error Handling
# ══════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_404_unknown_route(self, client):
        resp = client.get("/api/v1/nonexistent")
        assert resp.status_code == 404

    def test_process_time_header(self, client):
        resp = client.get("/api/v1/health")
        assert "x-process-time-ms" in resp.headers

    def test_cors_headers_present(self, client):
        resp = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should respond
        assert resp.status_code in (200, 400)
