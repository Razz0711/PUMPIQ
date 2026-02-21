"""Quick diagnostic — run: python _diag.py"""
from dotenv import load_dotenv
load_dotenv()
from supabase_db import get_supabase

sb = get_supabase()

# 1. ll_predictions
print("=" * 60)
print("1. ll_predictions")
print("=" * 60)
try:
    r = sb.table("ll_predictions").select("*", count="exact").limit(5).execute()
    print(f"   Total rows: {r.count}")
    if r.data:
        for row in r.data[:3]:
            pid = row.get("prediction_id", "?")
            tkr = row.get("token_ticker", "?")
            d = row.get("predicted_direction", "?")
            e24 = row.get("evaluated_24h_at")
            c24 = row.get("direction_correct_24h")
            pnl = row.get("pnl_pct_24h")
            print(f"   {pid}: {tkr} dir={d} eval24={e24} correct={c24} pnl={pnl}")
    else:
        print("   >>> TABLE IS EMPTY <<<")
except Exception as e:
    print(f"   ERROR: {e}")

# 2. trade_orders
print()
print("=" * 60)
print("2. trade_orders")
print("=" * 60)
try:
    r2 = sb.table("trade_orders").select("*", count="exact").limit(5).execute()
    print(f"   Total rows: {r2.count}")
    if r2.data:
        for row in r2.data[:3]:
            print(f"   action={row.get('action')} coin={row.get('coin_id')} sym={row.get('symbol')} price={row.get('price')} score={row.get('ai_score')} pos_id={row.get('position_id')}")
    else:
        print("   >>> TABLE IS EMPTY <<<")
except Exception as e:
    print(f"   ERROR: {e}")

# 3. trade_positions
print()
print("=" * 60)
print("3. trade_positions")
print("=" * 60)
try:
    r3 = sb.table("trade_positions").select("*", count="exact").limit(5).execute()
    print(f"   Total rows: {r3.count}")
    if r3.data:
        for row in r3.data[:3]:
            print(f"   id={row.get('id')} coin={row.get('coin_id')} side={row.get('side')} status={row.get('status')} entry={row.get('entry_price')} pnl={row.get('pnl')}")
    closed = sb.table("trade_positions").select("id", count="exact").eq("status", "closed").limit(0).execute()
    opened = sb.table("trade_positions").select("id", count="exact").eq("status", "open").limit(0).execute()
    print(f"   open={opened.count}  closed={closed.count}")
except Exception as e:
    print(f"   ERROR: {e}")

# 4. Test backfill
print()
print("=" * 60)
print("4. Testing backfill_from_trades()")
print("=" * 60)
try:
    from src.ai_engine.learning_loop import LearningLoop
    ll = LearningLoop()
    created = ll.backfill_from_trades()
    print(f"   Backfilled: {created} predictions created")
except Exception as e:
    import traceback
    print(f"   ERROR: {e}")
    traceback.print_exc()

# 5. Re-check ll_predictions after backfill
print()
print("=" * 60)
print("5. ll_predictions AFTER backfill")
print("=" * 60)
try:
    r = sb.table("ll_predictions").select("*", count="exact").limit(5).execute()
    print(f"   Total rows: {r.count}")
    if r.data:
        for row in r.data[:5]:
            pid = row.get("prediction_id", "?")
            tkr = row.get("token_ticker", "?")
            d = row.get("predicted_direction", "?")
            e24 = row.get("evaluated_24h_at")
            c24 = row.get("direction_correct_24h")
            pnl = row.get("pnl_pct_24h")
            print(f"   {pid}: {tkr} dir={d} eval24={e24} correct={c24} pnl={pnl}")
    else:
        print("   >>> STILL EMPTY after backfill <<<")
except Exception as e:
    print(f"   ERROR: {e}")

# 6. Test get_performance_stats
print()
print("=" * 60)
print("6. get_performance_stats() output")
print("=" * 60)
try:
    stats = ll.get_performance_stats()
    for k, v in stats.items():
        if k not in ("best_predictions", "worst_predictions", "regime_accuracy", "direction_accuracy"):
            print(f"   {k}: {v}")
        else:
            print(f"   {k}: {len(v) if isinstance(v, (list, dict)) else v} items")
except Exception as e:
    print(f"   ERROR: {e}")
