"""
Quantitative Strategy Reasoning Example
Demonstrates multi-agent reasoning for quantitative trading strategy evaluation.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ReasoningGraph
import json


def print_reasoning_history(result: dict):
    """Pretty print the reasoning history."""
    print("\n" + "=" * 80)
    print("📚 REASONING HISTORY")
    print("=" * 80)

    for i, step in enumerate(result["reasoning_history"], 1):
        role = step["role"].upper()
        content = step["content"]
        timestamp = step.get("timestamp", "N/A")

        print(f"\n[{i}] {role} ({timestamp})")
        print("-" * 80)
        print(content)
        print("-" * 80)


def example_1_mean_reversion():
    """Example 1: Mean reversion strategy evaluation."""
    print("\n" + "📈" * 40)
    print("EXAMPLE 1: Mean Reversion Strategy")
    print("📈" * 40)

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

    print("\n" + "=" * 80)
    print("🎯 FINAL ANSWER")
    print("=" * 80)
    print(result["final_answer"])

    print_reasoning_history(result)

    return result


def example_2_pairs_trading():
    """Example 2: Pairs trading strategy reasoning."""
    print("\n" + "📊" * 40)
    print("EXAMPLE 2: Pairs Trading Strategy")
    print("📊" * 40)

    graph = ReasoningGraph(max_iterations=4)

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
    - Stop-loss: If spread widens to Z-score > 3.0

    Context:
    - Both stocks report earnings in 2 weeks
    - Recent regulatory news affecting financial sector
    - Interest rate environment: Fed on hold

    Evaluate:
    1. Is the cointegration relationship still valid?
    2. What is the risk/reward profile?
    3. Should we enter this trade given the earnings risk?
    4. What position sizing adjustments would you recommend?
    """

    result = graph.reason(question)

    print("\n" + "=" * 80)
    print("🎯 FINAL ANSWER")
    print("=" * 80)
    print(result["final_answer"])

    print_reasoning_history(result)

    return result


def example_3_momentum_strategy():
    """Example 3: Momentum strategy with ML signals."""
    print("\n" + "🚀" * 40)
    print("EXAMPLE 3: ML-Enhanced Momentum Strategy")
    print("🚀" * 40)

    graph = ReasoningGraph(max_iterations=3)

    question = """
    Evaluate this ML-enhanced momentum strategy:

    Strategy Framework:
    - Universe: Russell 2000 constituents
    - Lookback: 60 days momentum
    - ML Model: LSTM predicting next 5-day returns
    - Position: Long top 20 stocks (highest combined momentum + ML score)
    - Rebalancing: Weekly

    Backtesting Results (2020-2023):
    - Annual Return: 18.5%
    - Sharpe Ratio: 1.35
    - Max Drawdown: -22%
    - Win Rate: 58%
    - Avg Holding Period: 12 days

    Concerns:
    1. Model was trained on 2015-2019 data (pre-COVID)
    2. Recent 3-month performance: -5% (market +2%)
    3. Model accuracy dropped from 62% to 54%
    4. Higher turnover than expected (3.2x per month)

    Questions:
    1. Is the model degrading? How to diagnose?
    2. Should we retrain, adjust, or pause the strategy?
    3. How to incorporate regime detection?
    4. What risk management improvements would you add?
    """

    result = graph.reason(question)

    print("\n" + "=" * 80)
    print("🎯 FINAL ANSWER")
    print("=" * 80)
    print(result["final_answer"])

    print_reasoning_history(result)

    return result


def example_4_portfolio_optimization():
    """Example 4: Portfolio optimization reasoning."""
    print("\n" + "💼" * 40)
    print("EXAMPLE 4: Portfolio Optimization")
    print("💼" * 40)

    graph = ReasoningGraph(max_iterations=4)

    question = """
    Design an optimal portfolio allocation strategy:

    Investor Profile:
    - Risk tolerance: Moderate
    - Investment horizon: 5 years
    - Current portfolio: 60% stocks, 40% bonds
    - AUM: $1M

    Available Assets:
    1. US Large Cap (SPY): Expected return 8%, Vol 15%
    2. US Small Cap (IWM): Expected return 10%, Vol 22%
    3. International (EFA): Expected return 7%, Vol 18%
    4. Emerging Markets (EEM): Expected return 12%, Vol 28%
    5. Corporate Bonds (LQD): Expected return 4%, Vol 8%
    6. Treasury Bonds (TLT): Expected return 3%, Vol 12%
    7. Real Estate (VNQ): Expected return 9%, Vol 20%
    8. Gold (GLD): Expected return 5%, Vol 16%

    Correlation Matrix:
    - SPY-IWM: 0.85, SPY-EFA: 0.75, SPY-Bonds: -0.20
    - SPY-Gold: -0.10, Bonds-Gold: 0.05
    (Assume reasonable correlations for others)

    Constraints:
    - Max allocation per asset: 30%
    - Min allocation per asset: 5%
    - Total equity (stocks): 50-70%
    - Total fixed income: 20-40%

    Tasks:
    1. Propose optimal allocation using modern portfolio theory
    2. Calculate expected portfolio Sharpe ratio
    3. Identify key risk factors
    4. Suggest rebalancing frequency
    5. How would you adjust for a recession scenario?
    """

    result = graph.reason(question)

    print("\n" + "=" * 80)
    print("🎯 FINAL ANSWER")
    print("=" * 80)
    print(result["final_answer"])

    print_reasoning_history(result)

    return result


def save_results(result: dict, filename: str):
    """Save reasoning results to JSON file."""
    output = {
        "question": result["question"],
        "final_answer": result["final_answer"],
        "confidence_score": result["confidence_score"],
        "iterations": result["iterations"],
        "reasoning_history": result["reasoning_history"]
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {filename}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # Run examples
    print("\n" + "🚀" * 40)
    print("MULTI-AGENT REASONING SYSTEM - QUANT STRATEGY EXAMPLES")
    print("🚀" * 40)

    # Example 1: Mean reversion
    result1 = example_1_mean_reversion()
    save_results(result1, "results_mean_reversion.json")

    # Example 2: Pairs trading
    result2 = example_2_pairs_trading()
    save_results(result2, "results_pairs_trading.json")

    # Example 3: ML momentum
    result3 = example_3_momentum_strategy()
    save_results(result3, "results_ml_momentum.json")

    # Example 4: Portfolio optimization
    result4 = example_4_portfolio_optimization()
    save_results(result4, "results_portfolio_optimization.json")

    print("\n" + "✅" * 40)
    print("ALL QUANT EXAMPLES COMPLETED!")
    print("✅" * 40)
