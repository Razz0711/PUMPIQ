"""
Focused test: reproduce exactly what the enhanced-recommendations endpoint does.
"""
import asyncio, logging, traceback, os, sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s %(levelname)s: %(message)s",
)

async def test():
    # Load .env like the server does
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass

    # Step 1: import the same things the endpoint imports
    print("=" * 60)
    print("STEP 1: Imports")
    print("=" * 60)
    try:
        from src.ai_engine.models import (
            DataMode, MarketCondition, UserQuery, UserConfig, QueryType
        )
        print("  models OK")
    except Exception as e:
        print(f"  FAIL importing models: {e}")
        traceback.print_exc()
        return

    try:
        from src.ai_engine.orchestrator import Orchestrator
        print("  orchestrator OK")
    except Exception as e:
        print(f"  FAIL importing orchestrator: {e}")
        traceback.print_exc()
        return

    try:
        from src.data_collectors.data_pipeline import DataPipeline
        print("  data_pipeline OK")
    except Exception as e:
        print(f"  FAIL importing data_pipeline: {e}")
        traceback.print_exc()
        return


    # Step 2: Create DataPipeline
    print("\n" + "=" * 60)
    print("STEP 2: Create DataPipeline")
    print("=" * 60)
    try:
        dp = DataPipeline()
        print(f"  DataPipeline OK: {dp}")
        print(f"  dp.fetch = {dp.fetch}")
    except Exception as e:
        print(f"  FAIL creating DataPipeline: {e}")
        traceback.print_exc()
        dp = None

    # Step 3: Create GeminiClient
    print("\n" + "=" * 60)
    print("STEP 3: Create GeminiClient")
    print("=" * 60)
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    print(f"  GEMINI_API_KEY present: {bool(gemini_key)}")
    ai_client = None
    if gemini_key:
        try:
            from src.ai_engine.gemini_client import GeminiClient
            ai_client = GeminiClient(api_key=gemini_key)
            print(f"  GeminiClient OK: {ai_client}")
        except Exception as e:
            print(f"  FAIL creating GeminiClient: {e}")
            traceback.print_exc()
    else:
        print("  WARNING: No GEMINI_API_KEY - ai_client will be None!")

    # Step 4: Create Orchestrator
    print("\n" + "=" * 60)
    print("STEP 4: Create Orchestrator")
    print("=" * 60)
    try:
        orch = Orchestrator(
            gpt_client=ai_client,
            data_fetcher=dp.fetch if dp else None,
            market_condition=MarketCondition.SIDEWAYS,
        )
        print(f"  Orchestrator OK")
        print(f"  orch.gpt = {orch.gpt}")
        print(f"  orch.data_fetcher = {orch.data_fetcher}")
    except Exception as e:
        print(f"  FAIL creating Orchestrator: {e}")
        traceback.print_exc()
        return

    # Step 5: Create UserQuery and UserConfig
    print("\n" + "=" * 60)
    print("STEP 5: Create UserQuery & UserConfig")
    print("=" * 60)
    try:
        uq = UserQuery(
            raw_query="Top crypto picks",
            query_type=QueryType.BEST_COINS,
            num_recommendations=3,
        )
        uc = UserConfig()
        print(f"  UserQuery OK: type={uq.query_type}, n={uq.num_recommendations}")
        print(f"  UserConfig OK: modes={uc.enabled_modes}")
    except Exception as e:
        print(f"  FAIL creating query/config: {e}")
        traceback.print_exc()
        return

    # Step 6: Run orchestrator
    print("\n" + "=" * 60)
    print("STEP 6: Run Orchestrator")
    print("=" * 60)
    try:
        rec_set = await orch.run(uq, uc)
        print(f"\n  SUCCESS! Recommendations: {len(rec_set.recommendations)}")
        for r in rec_set.recommendations:
            print(f"    #{r.rank} {r.token_name} ({r.token_ticker}) score={r.composite_score}")
    except Exception as e:
        print(f"\n  FAILED during orch.run(): {type(e).__name__}: {e}")
        traceback.print_exc()

asyncio.run(test())
