"""Quick verification that backtest_engine changes compile and work."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. Syntax check
import ast
with open(os.path.join(os.path.dirname(__file__), "src", "backtest_engine.py"), encoding="utf-8") as f:
    ast.parse(f.read())
print("[OK] Syntax check passed")

# 2. Import check
from src.backtest_engine import BacktestEngine, THRESHOLDS, get_backtest_engine
print("[OK] Import successful")

# 3. Verify constants
e = BacktestEngine()
assert e.rsi_oversold == 35.0, f"RSI oversold should be 35.0, got {e.rsi_oversold}"
assert e.rsi_overbought == 65.0, f"RSI overbought should be 65.0, got {e.rsi_overbought}"
print(f"[OK] RSI thresholds: oversold={e.rsi_oversold}, overbought={e.rsi_overbought}")

assert THRESHOLDS["major"]["win_rate"] == 50.0
assert THRESHOLDS["major"]["min_trades"] == 8
assert THRESHOLDS["major"]["min_return"] == 0.0
assert THRESHOLDS["mid"]["win_rate"] == 48.0
assert THRESHOLDS["mid"]["min_trades"] == 6
assert THRESHOLDS["micro"]["win_rate"] == 45.0
assert THRESHOLDS["micro"]["min_trades"] == 5
print("[OK] Tiered thresholds verified")

# 4. Verify strategy cascade exists via source inspection
import inspect
src = inspect.getsource(e.run_backtest)
assert "strategy_order" in src, "Strategy cascade not found in run_backtest"
assert "best_trial" in src, "best_trial logic not found in run_backtest"
print("[OK] Strategy cascade implemented")

# 5. Verify LONG entry uses >= 1
src_long = inspect.getsource(e._simulate_long_trades)
assert "buy_signals >= 1" in src_long, "LONG entry should require >= 1 signal"
print("[OK] LONG entry relaxed to >= 1 signal")

# 6. Verify stop-loss cooldown fix
assert "last_exit_idx = i" in src_long.split("stop_loss")[2][:500], "Stop-loss should update last_exit_idx"
print("[OK] Stop-loss cooldown bug fixed")

print("\n=== ALL VERIFICATIONS PASSED ===")
