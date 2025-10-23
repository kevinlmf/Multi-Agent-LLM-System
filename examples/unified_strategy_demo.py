"""
Unified Strategy Demo - Combines LLM Reasoning + Performance Comparison

This demo showcases:
1. LLM's reasoning capability on various trading strategies
2. Performance comparison between Traditional vs LLM strategies
3. Interactive menu for exploring different scenarios
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ReasoningGraph
import json
from datetime import datetime


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    """Print section divider."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


# ============================================================================
# PART 1: LLM REASONING DEMONSTRATIONS
# ============================================================================

def demo_mean_reversion_reasoning():
    """Demo 1: LLM reasoning on mean reversion strategy."""
    print_header("DEMO 1: Mean Reversion Strategy Analysis (LLM Reasoning)")

    graph = ReasoningGraph(max_iterations=3)

    question = """
    Evaluate the following mean reversion trading strategy:

    Strategy Details:
    - Asset: SPY (S&P 500 ETF)
    - Signal: Z-score of 20-day moving average
    - Entry: When Z-score < -2 (oversold), go LONG
    - Exit: When Z-score > 0 (back to mean), close position
    - Stop-loss: -3% from entry
    - Position sizing: Kelly criterion with 0.5 factor

    Market Context:
    - Current volatility (VIX): 18
    - Recent trend: Sideways with occasional spikes
    - Correlation with bonds: -0.3

    Questions to address:
    1. Is this strategy suitable for current market conditions?
    2. What are the key risks?
    3. What improvements would you suggest?
    4. What should be the expected Sharpe ratio range?
    """

    result = graph.reason(question)

    print_section("LLM Analysis Result")
    print(result["final_answer"])
    print(f"\nConfidence: {result['confidence_score']:.2%}")
    print(f"Reasoning Iterations: {result['iterations']}")

    return result


def demo_pairs_trading_reasoning():
    """Demo 2: LLM reasoning on pairs trading."""
    print_header("DEMO 2: Pairs Trading Analysis (LLM Reasoning)")

    graph = ReasoningGraph(max_iterations=3)

    question = """
    Analyze this pairs trading opportunity:

    Pair:
    - Stock A: JPM (JP Morgan)
    - Stock B: BAC (Bank of America)

    Statistical Analysis:
    - Cointegration test p-value: 0.03
    - Half-life of mean reversion: 5 days
    - Historical correlation: 0.85
    - Current spread Z-score: 2.3 (2 std dev above mean)

    Proposed Trade:
    - Short JPM: $100,000
    - Long BAC: $100,000
    - Hold period: 10 days or until Z-score < 0.5

    Context:
    - Both stocks report earnings in 2 weeks
    - Recent regulatory news affecting financial sector

    Should we enter this trade? What are the risks?
    """

    result = graph.reason(question)

    print_section("LLM Analysis Result")
    print(result["final_answer"])
    print(f"\nConfidence: {result['confidence_score']:.2%}")
    print(f"Reasoning Iterations: {result['iterations']}")

    return result


def demo_portfolio_optimization_reasoning():
    """Demo 3: LLM reasoning on portfolio optimization."""
    print_header("DEMO 3: Portfolio Optimization (LLM Reasoning)")

    graph = ReasoningGraph(max_iterations=3)

    question = """
    Design an optimal portfolio allocation strategy:

    Investor Profile:
    - Risk tolerance: Moderate
    - Investment horizon: 5 years
    - Current portfolio: 60% stocks, 40% bonds
    - AUM: $1M

    Available Assets:
    1. US Large Cap (SPY): Expected return 8%, Vol 15%
    2. Emerging Markets (EEM): Expected return 12%, Vol 28%
    3. Corporate Bonds (LQD): Expected return 4%, Vol 8%
    4. Gold (GLD): Expected return 5%, Vol 16%

    Constraints:
    - Max allocation per asset: 40%
    - Min allocation per asset: 10%

    Propose optimal allocation and explain your reasoning.
    """

    result = graph.reason(question)

    print_section("LLM Analysis Result")
    print(result["final_answer"])
    print(f"\nConfidence: {result['confidence_score']:.2%}")
    print(f"Reasoning Iterations: {result['iterations']}")

    return result


# ============================================================================
# PART 2: PERFORMANCE COMPARISON DEMONSTRATIONS
# ============================================================================

def demo_crypto_comparison():
    """Demo 4: Compare Traditional vs LLM on crypto trading."""
    print_header("DEMO 4: Traditional vs LLM - Crypto Trading Comparison")

    # Import comparison framework
    sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_comparison'))
    from strategy_comparison.compare_strategies import StrategyComparison

    print("\n🔬 Running strategy comparison experiment...")
    print("   This will test both strategies on cryptocurrency data")

    comparison = StrategyComparison(initial_capital=10000)

    # Test on bullish crypto market
    print_section("Scenario: Bullish Crypto Market")
    results = comparison.compare(n_scenarios=5, regime='bullish')
    comparison.print_results()

    return results


def demo_volatile_market_comparison():
    """Demo 5: Compare strategies in volatile market."""
    print_header("DEMO 5: Traditional vs LLM - Volatile Market Comparison")

    sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_comparison'))
    from strategy_comparison.compare_strategies import StrategyComparison

    print("\n🔬 Testing strategies under high volatility...")

    comparison = StrategyComparison(initial_capital=10000)

    print_section("Scenario: Volatile Market Conditions")
    results = comparison.compare(n_scenarios=5, regime='volatile')
    comparison.print_results()

    return results


def demo_multi_regime_comparison():
    """Demo 6: Compare strategies across all market regimes."""
    print_header("DEMO 6: Comprehensive Multi-Regime Comparison")

    sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_comparison'))
    from strategy_comparison.compare_strategies import StrategyComparison

    regimes = ['bullish', 'bearish', 'sideways', 'volatile']
    all_results = []

    for regime in regimes:
        print_section(f"Testing {regime.upper()} Market")
        comparison = StrategyComparison(initial_capital=10000)
        results = comparison.compare(n_scenarios=5, regime=regime)

        # Print compact results
        trad = results['traditional_strategy']
        llm = results['llm_strategy']
        comp = results['comparison_metrics']

        print(f"\n  Traditional: {trad['total_return_pct']:+.2f}% | "
              f"LLM: {llm['total_return_pct']:+.2f}% | "
              f"Winner: {comp['winner']}")

        all_results.append({
            'regime': regime,
            'traditional_return': trad['total_return_pct'],
            'llm_return': llm['total_return_pct'],
            'winner': comp['winner']
        })

    # Summary
    print_section("SUMMARY ACROSS ALL REGIMES")
    print("\n  Regime        Traditional    LLM         Winner")
    print("  " + "─" * 55)
    for r in all_results:
        print(f"  {r['regime']:12}  {r['traditional_return']:+7.2f}%     {r['llm_return']:+7.2f}%    {r['winner']:>11}")

    llm_wins = sum(1 for r in all_results if r['winner'] == 'LLM')
    print(f"\n  LLM won in {llm_wins}/{len(regimes)} market regimes")

    return all_results


# ============================================================================
# INTERACTIVE MENU
# ============================================================================

def show_menu():
    """Display interactive menu."""
    print("\n" + "🎯" * 40)
    print("UNIFIED STRATEGY DEMO - Interactive Menu")
    print("🎯" * 40)

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                     PART 1: LLM REASONING DEMOS                           ║
║  These demos show how LLM analyzes and reasons about trading strategies  ║
╚═══════════════════════════════════════════════════════════════════════════╝

  1. Mean Reversion Strategy Analysis
     → LLM evaluates a mean reversion strategy on SPY
     → Shows reasoning process and risk assessment

  2. Pairs Trading Analysis
     → LLM analyzes a JPM/BAC pairs trade opportunity
     → Considers cointegration, earnings risk, and timing

  3. Portfolio Optimization
     → LLM designs optimal portfolio allocation
     → Balances risk/return across multiple assets

╔═══════════════════════════════════════════════════════════════════════════╗
║                PART 2: PERFORMANCE COMPARISON DEMOS                       ║
║     These demos compare actual trading performance: Traditional vs LLM   ║
╚═══════════════════════════════════════════════════════════════════════════╝

  4. Crypto Trading Comparison (Bullish Market)
     → Tests both strategies on rising crypto prices
     → Shows which approach captures upside better

  5. Volatile Market Comparison
     → Tests both strategies under high volatility
     → Evaluates risk management effectiveness

  6. Multi-Regime Comprehensive Test
     → Tests across all market conditions
     → Identifies which strategy works best where

╔═══════════════════════════════════════════════════════════════════════════╗
║              PART 3: INVESTMENT MASTER PERSONAS 🎩                        ║
║    Analyze crypto from different legendary investors' perspectives       ║
╚═══════════════════════════════════════════════════════════════════════════╝

  10. Jim Simons Style (Quantitative Renaissance)
      → Pure statistical analysis, high-frequency patterns

  11. Warren Buffett Style (Value Investing)
      → Long-term value, competitive moats, intrinsic value

  12. George Soros Style (Macro Trading & Reflexivity)
      → Macro imbalances, market psychology, reflexivity

  13. Ray Dalio Style (All-Weather Risk Parity)
      → Balanced portfolio, economic cycles, diversification

  14. Cathie Wood Style (Disruptive Innovation)
      → Exponential tech growth, innovation disruption

  15. ALL MASTERS Debate (Compare all 5 perspectives!)
      → See how different masters view the same opportunity

╔═══════════════════════════════════════════════════════════════════════════╗
║                           SPECIAL OPTIONS                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

  7. Run ALL LLM Reasoning Demos (1-3)
  8. Run ALL Performance Comparisons (4-6)
  9. Run EVERYTHING (Full Demo Suite)

  0. Exit

""")


def demo_master_analysis(master_name: str):
    """Run analysis from specific master's perspective."""
    sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_comparison'))
    from strategy_comparison.llm_crypto.master_strategies import MasterStrategyAnalyzer, PERSONAS
    from strategy_comparison.llm_crypto.crypto_reasoning import create_sample_context

    master_info = PERSONAS[master_name]
    print_header(f"{master_info.name} Analysis: {master_info.style}")

    print(f"\n📖 Philosophy Preview:")
    print(f"   {master_info.philosophy.strip().split(chr(10))[1][:150]}...")
    print(f"\n⏱️  Time Horizon: {master_info.time_horizon}")
    print(f"🎯 Risk Approach: {master_info.risk_approach}")

    # Create market scenario
    context = create_sample_context("BTC", base_price=45000, trend="bullish")

    # Analyze
    analyzer = MasterStrategyAnalyzer(master_name, max_iterations=2)
    result = analyzer.analyze(context)

    print_section("Analysis Result")
    print(result['analysis'])
    print(f"\nConfidence: {result['confidence']:.2%}")


def demo_all_masters_debate():
    """Run ALL MASTERS debate on same opportunity."""
    sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_comparison'))
    from strategy_comparison.llm_crypto.master_strategies import compare_masters
    from strategy_comparison.llm_crypto.crypto_reasoning import create_sample_context

    print_header("ALL MASTERS DEBATE: Who's Right?")

    context = create_sample_context("ETH", base_price=2500, trend="volatile")
    context.recent_news = [
        "Ethereum 2.0 staking yields attracting institutions",
        "DeFi protocols seeing record TVL growth",
        "Regulatory clarity improving in major markets"
    ]

    results = compare_masters(context)

    print("\n" + "🏆" * 40)
    print("VERDICT SUMMARY")
    print("🏆" * 40)

    decisions = []
    for result in results:
        analysis_lower = result['analysis'].lower()
        if 'buy' in analysis_lower and 'not buy' not in analysis_lower:
            decision = "BUY"
        elif 'sell' in analysis_lower:
            decision = "SELL"
        else:
            decision = "HOLD"
        decisions.append(decision)
        print(f"{result['master']:20} → {decision:6} @ {result['confidence']:.0%} confidence")

    # Consensus
    buy_count = decisions.count('BUY')
    sell_count = decisions.count('SELL')
    hold_count = decisions.count('HOLD')

    print(f"\n📊 Consensus: {buy_count} BUY | {hold_count} HOLD | {sell_count} SELL")

    if buy_count > sell_count + hold_count:
        print("✅ Majority BULLISH - Multiple masters see opportunity")
    elif sell_count > buy_count + hold_count:
        print("❌ Majority BEARISH - Masters advise caution")
    else:
        print("⚖️  NO CONSENSUS - Diverse opinions, proceed carefully")


def run_demo(choice: int) -> bool:
    """Run selected demo. Returns True if should continue."""

    # Check API key for options that need it
    if choice in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        if not os.getenv("OPENAI_API_KEY"):
            print("\n⚠️  OPENAI_API_KEY not set!")
            print("   Some features will run in mock mode.")
            print("   For full LLM features: export OPENAI_API_KEY='your-key'")
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return True

    try:
        if choice == 1:
            demo_mean_reversion_reasoning()
        elif choice == 2:
            demo_pairs_trading_reasoning()
        elif choice == 3:
            demo_portfolio_optimization_reasoning()
        elif choice == 4:
            demo_crypto_comparison()
        elif choice == 5:
            demo_volatile_market_comparison()
        elif choice == 6:
            demo_multi_regime_comparison()
        elif choice == 7:
            print_header("Running ALL LLM Reasoning Demos")
            demo_mean_reversion_reasoning()
            demo_pairs_trading_reasoning()
            demo_portfolio_optimization_reasoning()
        elif choice == 8:
            print_header("Running ALL Performance Comparisons")
            demo_crypto_comparison()
            demo_volatile_market_comparison()
        elif choice == 9:
            print_header("Running COMPLETE DEMO SUITE")
            # LLM Reasoning
            demo_mean_reversion_reasoning()
            demo_pairs_trading_reasoning()
            demo_portfolio_optimization_reasoning()
            # Performance Comparison
            demo_crypto_comparison()
            demo_volatile_market_comparison()
            demo_multi_regime_comparison()
        elif choice == 10:
            demo_master_analysis('simons')
        elif choice == 11:
            demo_master_analysis('buffett')
        elif choice == 12:
            demo_master_analysis('soros')
        elif choice == 13:
            demo_master_analysis('dalio')
        elif choice == 14:
            demo_master_analysis('wood')
        elif choice == 15:
            demo_all_masters_debate()
        elif choice == 0:
            print("\n👋 Goodbye!")
            return False
        else:
            print("\n❌ Invalid choice. Please select 0-15.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()

    return True


def main():
    """Main interactive loop."""
    print("\n" + "🚀" * 40)
    print("UNIFIED STRATEGY DEMONSTRATION SYSTEM")
    print("Combining LLM Reasoning + Performance Comparison")
    print("🚀" * 40)

    print("""
This system demonstrates two complementary approaches:

1. LLM REASONING: How AI analyzes and evaluates trading strategies
   → Deep analysis of strategy logic
   → Risk assessment and suggestions
   → Qualitative reasoning capabilities

2. PERFORMANCE COMPARISON: How strategies perform in practice
   → Traditional technical analysis (rule-based)
   → LLM-based reasoning (adaptive)
   → Head-to-head backtesting

Together, they show both the "thinking process" and the "actual results"
""")

    while True:
        show_menu()
        try:
            choice = int(input("Select demo (0-15): "))
            if not run_demo(choice):
                break

            if choice != 0:
                input("\n✅ Demo complete. Press Enter to return to menu...")

        except ValueError:
            print("\n❌ Please enter a number 0-15")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
