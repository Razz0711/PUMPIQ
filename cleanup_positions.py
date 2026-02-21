"""Close all bad positions (DEX tokens with address-based coin_ids or zero prices)."""
from dotenv import load_dotenv
load_dotenv()
from supabase_db import get_supabase
from datetime import datetime, timezone

sb = get_supabase()
positions = sb.table("trade_positions").select("*").eq("user_id", 1).eq("status", "open").execute()

print(f"Found {len(positions.data)} open positions:")
closed = 0
for p in positions.data:
    coin_id = p.get("coin_id", "")
    entry = p.get("entry_price", 0)
    name = p.get("coin_name", "")
    symbol = p.get("symbol", "")
    
    # Bad position = DEX address token (long hash) OR zero price
    is_bad = len(coin_id) > 20 or entry <= 0.001
    
    print(f"  ID:{p['id']} | {symbol} {name} | entry=${entry} | BAD={is_bad}")
    
    if is_bad:
        # Force close the position
        sb.table("trade_positions").update({
            "status": "closed",
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "pnl": 0,
            "pnl_pct": 0,
        }).eq("id", p["id"]).execute()
        
        # Refund the invested amount back to wallet
        invested = p.get("invested_amount", 0)
        if invested > 0:
            sb.rpc("update_wallet_balance", {
                "p_user_id": 1,
                "p_delta": invested,
            }).execute()
        
        closed += 1
        print(f"    -> CLOSED and refunded ${invested:,.0f}")

print(f"\nDone: closed {closed} bad positions")
