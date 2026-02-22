"""
NexYpher – AI Synthesis Engine
===============================
Phase 3: AI Orchestration, Confidence Scoring, Prompt Engineering,
and Natural Language Generation.

Supported AI backends:
    - GeminiClient (Google Gemini 2.0 Flash – default, free tier)

Public API
----------
>>> from src.ai_engine import Orchestrator, GeminiClient, NLGEngine
>>> client = GeminiClient(api_key="AIza...")
>>> orch   = Orchestrator(ai_client=client)
>>> result = await orch.run(query, config)
"""

# ── Models ────────────────────────────────────────────────────────
from .models import (
    ConfidenceBreakdown,
    ConflictFlag,
    ConflictSeverity,
    DataMode,
    EntryExitPlan,
    InvestmentTimeframe,
    MarketCondition,
    NewsScorePayload,
    OnchainScorePayload,
    PositionSizePreference,
    QueryType,
    RecommendationSet,
    RecommendationVerdict,
    RiskAssessment,
    RiskLevel,
    RiskTolerance,
    SocialScorePayload,
    TechnicalScorePayload,
    TokenData,
    TokenRecommendation,
    UserConfig,
    UserQuery,
)

# ── Sub-engines ───────────────────────────────────────────────────
from .confidence_scorer import (
    ConfidenceScorer,
    EntryExitCalculator,
    RiskRater,
    confidence_risk_verdict,
)
from .conflict_detector import ConflictDetector
from .gemini_client import GeminiClient, GeminiResponse
from .nlg_engine import NLGEngine
from .prompt_templates import PromptBuilder

# ── Orchestrator (main entry point) ──────────────────────────────
from .orchestrator import Orchestrator

__all__ = [
    # Orchestrator
    "Orchestrator",
    # AI clients
    "GeminiClient",
    "GeminiResponse",
    # Sub-engines
    "ConfidenceScorer",
    "RiskRater",
    "EntryExitCalculator",
    "ConflictDetector",
    "PromptBuilder",
    "NLGEngine",
    "confidence_risk_verdict",
    # Models – Enums
    "DataMode",
    "QueryType",
    "InvestmentTimeframe",
    "RiskTolerance",
    "RiskLevel",
    "PositionSizePreference",
    "MarketCondition",
    "RecommendationVerdict",
    "ConflictSeverity",
    # Models – Data classes
    "UserQuery",
    "UserConfig",
    "TokenData",
    "NewsScorePayload",
    "OnchainScorePayload",
    "TechnicalScorePayload",
    "SocialScorePayload",
    "ConflictFlag",
    "ConfidenceBreakdown",
    "RiskAssessment",
    "EntryExitPlan",
    "TokenRecommendation",
    "RecommendationSet",
]
