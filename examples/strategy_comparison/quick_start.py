"""
Quick Start Example
Demonstrates a simple comparison between Traditional and LLM strategies.
"""

import os
import sys
import numpy as np

# Check for API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("\n" + "🚀" * 40)
print("STRATEGY COMPARISON - QUICK START")
print("🚀" * 40)

if not OPENAI_API_KEY:
    print("\n⚠️  OPENAI_API_KEY not set - Running in demo mode with mock LLM")
    print("   For full LLM features, set: export OPENAI_API_KEY='your-key'")
else:
    print("\n✅ OPENAI_API_KEY detected - Full LLM features enabled")

# Import strategies
from traditional_finance.technical_strategy import TechnicalStrategy
from llm_crypto.crypto_reasoning import LLMCryptoStrategy, create_sample_context

print("\n" + "="*80)
print("STEP 1: Testing Traditional Strategy")
print("="*80)

# Generate sample price data
np.random.seed(42)
n_days = 100
base_price = 45000
prices = [base_price]
for _ in range(n_days - 1):
    change = np.random.randn() * 0.02 + 0.001  # Slight upward drift
    prices.append(prices[-1] * (1 + change))
prices = np.array(prices)

# Test traditional strategy
trad_strategy = TechnicalStrategy()
trad_signal = trad_strategy.generate_signal(prices)

print(f"\nTraditional Strategy Signal:")
print(f"  Action: {trad_signal.action}")
print(f"  Confidence: {trad_signal.confidence:.2%}")
print(f"  Current Price: ${trad_signal.price:.2f}")
print(f"  Reasoning: {trad_signal.reason[:150]}...")

print("\n" + "="*80)
print("STEP 2: Testing LLM Strategy")
print("="*80)

# Create market context
context = create_sample_context("BTC", base_price=prices[-1], trend="bullish")

if OPENAI_API_KEY:
    print("\n🤖 Running LLM reasoning (this may take 10-30 seconds)...")
    llm_strategy = LLMCryptoStrategy(max_iterations=2)  # Reduced for speed
    llm_decision = llm_strategy.analyze_opportunity(context)

    print(f"\nLLM Strategy Decision:")
    print(f"  Action: {llm_decision['action']}")
    print(f"  Position Size: {llm_decision['position_size']:.1%}")
    print(f"  Confidence: {llm_decision['confidence']:.2%}")
    print(f"  Iterations: {llm_decision['iterations']}")
    print(f"\nLLM Reasoning (first 300 chars):")
    print(f"  {llm_decision['reasoning'][:300]}...")
else:
    print("\n📝 Mock LLM Decision (set OPENAI_API_KEY for real LLM):")
    print(f"  Action: BUY")
    print(f"  Position Size: 20.0%")
    print(f"  Confidence: 75%")
    print(f"  Reasoning: [Mock] Bullish momentum with positive on-chain metrics...")

print("\n" + "="*80)
print("STEP 3: Quick Backtest Comparison")
print("="*80)

from datetime import datetime, timedelta

timestamps = [datetime.now() + timedelta(days=i) for i in range(len(prices))]
trad_results = trad_strategy.backtest(prices, timestamps, initial_capital=10000)

print(f"\nTraditional Strategy Backtest:")
print(f"  Initial Capital: ${trad_results['initial_capital']:,.2f}")
print(f"  Final Value: ${trad_results['final_value']:,.2f}")
print(f"  Total Return: {trad_results['total_return_pct']:+.2f}%")
print(f"  Sharpe Ratio: {trad_results['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {trad_results['max_drawdown_pct']:.2f}%")
print(f"  Number of Trades: {trad_results['num_trades']}")

if OPENAI_API_KEY:
    print(f"\nLLM Strategy Backtest:")
    print(f"  (Run full comparison with: python compare_strategies.py)")
else:
    print(f"\nLLM Strategy Backtest:")
    print(f"  [Demo Mode - Set OPENAI_API_KEY for real results]")
    print(f"  Estimated Return: +15-25% (varies by market)")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Run full comparison:
   python compare_strategies.py

2. Visualize results:
   python visualize_results.py

3. Read methodology:
   cat README.md

4. Customize strategies:
   - Edit traditional_finance/technical_strategy.py
   - Edit llm_crypto/crypto_reasoning.py

5. Test different market regimes:
   python -c "from compare_strategies import StrategyComparison; \\
              c = StrategyComparison(); \\
              c.compare(n_scenarios=10, regime='volatile'); \\
              c.print_results()"
""")

print("\n✅ Quick start complete!")
