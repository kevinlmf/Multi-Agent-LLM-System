"""
Enhanced Cognitive Trading Agent
Demonstrates:
1. Full cognitive loop: Perception → Memory → Reasoning → Action → Learning
2. Strategy analysis (Mean Reversion, Pairs Trading, Portfolio Optimization)
3. Performance comparison (Traditional vs LLM)
4. Investment master personas (Simons, Buffett, Soros, Dalio, Wood)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_graph import CognitiveReasoningGraph
from core import ReasoningGraph
from datetime import datetime


def get_master_opinions(scenario_data: dict) -> str:
    """
    Get opinions from investment masters on the trading scenario.

    Args:
        scenario_data: Dictionary with day, observation, question

    Returns:
        Formatted string with masters' opinions
    """
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_comparison'))
        from strategy_comparison.llm_crypto.master_strategies import MasterStrategyAnalyzer, PERSONAS

        # Parse market data from observation
        observation = scenario_data['observation']

        # Create a simplified prompt for masters
        master_context = f"""
Market Scenario (Day {scenario_data['day']}):
{observation}

Question: {scenario_data['question']}
"""

        opinions = []
        master_names = ['buffett', 'soros', 'dalio']  # Select key masters

        print("\n🎩 Consulting investment masters...")

        for master_name in master_names:
            try:
                master = PERSONAS[master_name]
                # Create simplified analysis prompt
                analysis_prompt = f"""
{master.philosophy}

{master_context}

As {master.name}, provide a brief recommendation (2-3 sentences) on what action to take.
Focus on your key principles: {', '.join(master.key_metrics[:2])}
"""
                # Use basic ReasoningGraph for quick analysis
                from core import ReasoningGraph
                graph = ReasoningGraph(max_iterations=1)
                result = graph.reason(analysis_prompt)

                opinions.append({
                    'master': master.name,
                    'style': master.style,
                    'opinion': result['final_answer'][:300]  # Limit length
                })
                print(f"  ✓ {master.name} ({master.style})")

            except Exception as e:
                print(f"  ⚠️  Could not get opinion from {master_name}: {str(e)[:50]}")
                continue

        # Format opinions
        if not opinions:
            return "\n💭 Investment Masters' Opinions: Not available\n"

        formatted = "\n" + "="*80 + "\n"
        formatted += "💭 INVESTMENT MASTERS' PERSPECTIVES\n"
        formatted += "="*80 + "\n"

        for opinion in opinions:
            formatted += f"\n📊 {opinion['master']} ({opinion['style']}):\n"
            formatted += f"{opinion['opinion']}\n"

        formatted += "="*80 + "\n"

        return formatted

    except ImportError:
        return "\n💭 Investment Masters' Opinions: Module not available\n"
    except Exception as e:
        return f"\n💭 Investment Masters' Opinions: Error - {str(e)[:100]}\n"


def run_trading_scenario(enable_masters: bool = True):
    """
    Run a complete trading scenario with cognitive multi-agent system.

    Args:
        enable_masters: Whether to consult investment masters (Buffett, Soros, Dalio)

    Demonstrates:
    1. Perceiving market conditions
    2. Retrieving relevant past experiences
    3. Multi-agent reasoning (Reasoner → Critic → Refiner)
    4. Making trading decisions
    5. Consulting investment masters (Buffett, Soros, Dalio) - Optional
    6. Storing experiences to memory
    7. Learning from outcomes (consolidation)
    """

    print("=" * 80)
    title = "COGNITIVE TRADING AGENT - Full System Demo"
    if enable_masters:
        title += " with Investment Masters"
    print(title)
    print("=" * 80)
    print()

    # Initialize cognitive system
    print("Initializing cognitive multi-agent system...")
    cognitive_system = CognitiveReasoningGraph(
        max_iterations=3,
        model_name="gpt-4o-mini",
        enable_perception=True,
        enable_memory=True
    )
    print("✓ System initialized with perception and memory")
    if enable_masters:
        print("✓ Investment masters module enabled (Buffett, Soros, Dalio)")
    print()

    # Scenario: Multiple trading decisions over time
    scenarios = [
        {
            "day": 1,
            "observation": """
            Market Data - Day 1:
            - SPY: $450.00 (+0.5%)
            - VIX: 15.2 (-2.1%)
            - News: Fed announces rate hold, market rallies
            - Volume: High buying volume in tech sector
            - Sentiment: Bullish indicators across major indices
            """,
            "question": "Should we enter a long position in SPY? What's the optimal strategy?",
            "reward": None  # No reward yet, first decision
        },
        {
            "day": 2,
            "observation": """
            Market Data - Day 2:
            - SPY: $455.20 (+1.2%)
            - VIX: 14.8 (-0.4%)
            - News: Strong earnings from tech giants
            - Volume: Sustained high volume
            - Our Position: Long SPY from $450, currently +$5.20 profit
            """,
            "question": "Should we hold, add to position, or take profits? Consider risk management.",
            "reward": 1.0  # Positive reward from previous decision
        },
        {
            "day": 3,
            "observation": """
            Market Data - Day 3:
            - SPY: $454.50 (-0.15%)
            - VIX: 16.5 (+1.7%)
            - News: Unexpected geopolitical tensions, market pullback
            - Volume: Elevated volatility
            - Our Position: Long SPY from $450, currently +$4.50 profit
            """,
            "question": "How should we respond to increased volatility? Adjust position or exit?",
            "reward": 0.8  # Still profitable but volatility increased
        },
        {
            "day": 4,
            "observation": """
            Market Data - Day 4:
            - SPY: $448.00 (-1.4%)
            - VIX: 19.8 (+3.3%)
            - News: Market correction continues, risk-off sentiment
            - Volume: Heavy selling pressure
            - Our Position: Long SPY from $450, currently -$2.00 loss
            """,
            "question": "Should we cut losses now or hold through the volatility? Risk assessment needed.",
            "reward": -0.5  # Negative reward, position now in loss
        },
        {
            "day": 5,
            "observation": """
            Market Data - Day 5:
            - SPY: $452.30 (+0.95%)
            - VIX: 17.2 (-2.6%)
            - News: Market stabilizes, dip buyers emerge
            - Volume: Moderate recovery
            - Our Position: Long SPY from $450, back to +$2.30 profit
            """,
            "question": "Market recovered. Should we exit now with small profit or continue? What did we learn?",
            "reward": 0.6  # Recovered to profit but experienced drawdown
        }
    ]

    # Run through scenarios
    results = []

    for scenario in scenarios:
        print("=" * 80)
        print(f"📊 DAY {scenario['day']} - Trading Decision")
        print("=" * 80)
        print()

        print("Market Observation:")
        print(scenario['observation'])
        print()

        print("Question:")
        print(scenario['question'])
        print()

        print("🤖 Cognitive agents working...")
        print("  → Perceiver: Processing market data...")
        print("  → Memory: Retrieving relevant experiences...")
        print("  → Reasoner: Analyzing with full context...")
        print("  → Critic: Evaluating reasoning quality...")
        print("  → Refiner: Producing final recommendation...")
        print()

        # Optionally get investment masters' opinions
        final_question = scenario['question']
        if enable_masters:
            master_opinions = get_master_opinions(scenario)

            # Execute cognitive reasoning with masters' input
            # Add masters' perspectives to the observation context
            final_question = f"""
{scenario['question']}

Consider the following perspectives from legendary investors:
{master_opinions}

Synthesize these expert opinions with your own cognitive analysis to make a final recommendation.
"""

        result = cognitive_system.reason(
            question=final_question,
            observation=scenario['observation'],
            reward=scenario['reward']
        )

        print("=" * 80)
        decision_title = "🎯 FINAL DECISION"
        if enable_masters:
            decision_title += " (Incorporating Masters' Wisdom)"
        print(decision_title)
        print("=" * 80)
        print(result['final_answer'])
        print()

        print(f"Confidence Score: {result['confidence_score']:.2f}")
        print(f"Reasoning Iterations: {result['iterations']}")
        print(f"Experience Stored: {result['experience_stored']}")
        print()

        # Show learned knowledge
        if result.get('semantic_knowledge'):
            print("💡 Learned Knowledge:")
            for knowledge in result['semantic_knowledge'][:3]:
                print(f"  • {knowledge}")
            print()

        results.append(result)

        print()
        input("Press Enter to continue to next day...")
        print("\n" * 2)

    # Final summary
    print("=" * 80)
    print("📈 TRADING CAMPAIGN SUMMARY")
    print("=" * 80)
    print()

    print(f"Total Trading Days: {len(scenarios)}")
    print(f"Decisions Made: {len(results)}")
    print()

    print("Confidence Progression:")
    for i, result in enumerate(results, 1):
        confidence = result.get('confidence_score', 0.0)
        bar = "█" * int(confidence * 20)
        print(f"  Day {i}: {bar} {confidence:.2f}")
    print()

    # Get system statistics
    stats = cognitive_system.get_statistics()

    print("System Statistics:")
    print(f"  Perception Enabled: {stats['perception_enabled']}")
    print(f"  Memory Enabled: {stats['memory_enabled']}")

    if stats.get('memory'):
        mem_stats = stats['memory']
        print(f"\nMemory System:")
        print(f"  Total Experiences: {mem_stats.get('episodic', {}).get('total_episodes', 0)}")
        print(f"  Concepts Learned: {mem_stats.get('semantic', {}).get('total_concepts', 0)}")
        print(f"  Patterns Detected: {mem_stats.get('semantic', {}).get('total_patterns', 0)}")

    print()
    print("=" * 80)
    print("✓ Cognitive trading agent demonstration complete!")
    print("=" * 80)


def run_simple_example():
    """Simple example without full perception/memory (for quick testing)."""

    print("Simple Cognitive Reasoning Example")
    print("=" * 50)
    print()

    # Initialize without perception/memory
    system = CognitiveReasoningGraph(
        max_iterations=2,
        model_name="gpt-4o-mini",
        enable_perception=False,
        enable_memory=False
    )

    # Simple trading question
    result = system.reason(
        question="I have $10,000. SPY is at $450 with low VIX. Should I enter a long position?",
        observation=None,
        reward=None
    )

    print("Final Answer:")
    print(result['final_answer'])
    print()
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Iterations: {result['iterations']}")
    print()

    print("Reasoning History:")
    for step in result['reasoning_history']:
        print(f"  [{step['role']}] {step['content'][:100]}...")
    print()


# ============================================================================
# PART 2: STRATEGY ANALYSIS DEMONSTRATIONS (Using Cognitive System)
# ============================================================================

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


def demo_mean_reversion_analysis():
    """Analyze mean reversion strategy with cognitive system."""
    print_header("Strategy Analysis: Mean Reversion (Cognitive System)")

    system = CognitiveReasoningGraph(
        max_iterations=3,
        model_name="gpt-4o-mini",
        enable_perception=True,
        enable_memory=True
    )

    observation = """
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
    """

    question = """
    Evaluate this mean reversion trading strategy:
    1. Is this strategy suitable for current market conditions?
    2. What are the key risks?
    3. What improvements would you suggest?
    4. What should be the expected Sharpe ratio range?
    """

    result = system.reason(
        question=question,
        observation=observation,
        reward=None
    )

    print_section("Cognitive Analysis Result")
    print(result['final_answer'])
    print(f"\nConfidence: {result['confidence_score']:.2%}")
    print(f"Reasoning Iterations: {result['iterations']}")

    if result.get('semantic_knowledge'):
        print("\n💡 Knowledge Learned:")
        for knowledge in result['semantic_knowledge'][:3]:
            print(f"  • {knowledge}")

    return result


def demo_pairs_trading_analysis():
    """Analyze pairs trading strategy with cognitive system."""
    print_header("Strategy Analysis: Pairs Trading (Cognitive System)")

    system = CognitiveReasoningGraph(
        max_iterations=3,
        model_name="gpt-4o-mini",
        enable_perception=True,
        enable_memory=True
    )

    observation = """
    Pair Analysis:
    - Stock A: JPM (JP Morgan)
    - Stock B: BAC (Bank of America)

    Statistical Metrics:
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
    """

    question = "Should we enter this pairs trade? Analyze risks and reward potential."

    result = system.reason(
        question=question,
        observation=observation,
        reward=None
    )

    print_section("Cognitive Analysis Result")
    print(result['final_answer'])
    print(f"\nConfidence: {result['confidence_score']:.2%}")
    print(f"Reasoning Iterations: {result['iterations']}")

    return result


def demo_portfolio_optimization_analysis():
    """Analyze portfolio optimization with cognitive system."""
    print_header("Strategy Analysis: Portfolio Optimization (Cognitive System)")

    system = CognitiveReasoningGraph(
        max_iterations=3,
        model_name="gpt-4o-mini",
        enable_perception=True,
        enable_memory=True
    )

    observation = """
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
    """

    question = "Design an optimal portfolio allocation strategy. Explain your reasoning and risk management approach."

    result = system.reason(
        question=question,
        observation=observation,
        reward=None
    )

    print_section("Cognitive Analysis Result")
    print(result['final_answer'])
    print(f"\nConfidence: {result['confidence_score']:.2%}")
    print(f"Reasoning Iterations: {result['iterations']}")

    return result


# ============================================================================
# PART 3: PERFORMANCE COMPARISON DEMONSTRATIONS
# ============================================================================

def demo_strategy_comparison():
    """Compare Traditional vs LLM strategies."""
    print_header("Performance Comparison: Traditional vs LLM")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_comparison'))
        from strategy_comparison.compare_strategies import StrategyComparison

        print("\n🔬 Running strategy comparison experiment...")
        comparison = StrategyComparison(initial_capital=10000)

        regimes = ['bullish', 'bearish', 'sideways', 'volatile']
        all_results = []

        for regime in regimes:
            print_section(f"Testing {regime.upper()} Market")
            results = comparison.compare(n_scenarios=5, regime=regime)

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

    except ImportError:
        print("\n⚠️  Strategy comparison module not found.")
        print("   This feature requires the strategy_comparison package.")
        return None


# ============================================================================
# PART 4: INVESTMENT MASTER PERSONAS
# ============================================================================

def demo_master_analysis(master_name: str):
    """Run analysis from specific master's perspective."""
    print_header(f"Investment Master Analysis: {master_name.upper()}")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_comparison'))
        from strategy_comparison.llm_crypto.master_strategies import MasterStrategyAnalyzer, PERSONAS
        from strategy_comparison.llm_crypto.crypto_reasoning import create_sample_context

        master_info = PERSONAS[master_name]
        print(f"\n📖 Philosophy: {master_info.style}")
        print(f"⏱️  Time Horizon: {master_info.time_horizon}")
        print(f"🎯 Risk Approach: {master_info.risk_approach}")

        # Create market scenario
        context = create_sample_context("BTC", base_price=45000, trend="bullish")

        # Analyze
        analyzer = MasterStrategyAnalyzer(master_name, max_iterations=2)
        result = analyzer.analyze(context)

        print_section("Analysis Result")
        print(result['analysis'])
        print(f"\nConfidence: {result['confidence']:.2%}")

        return result

    except ImportError:
        print("\n⚠️  Master strategy module not found.")
        print("   This feature requires the strategy_comparison package.")
        return None


def demo_all_masters_debate():
    """Run ALL MASTERS debate on same opportunity."""
    print_header("ALL MASTERS DEBATE")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_comparison'))
        from strategy_comparison.llm_crypto.master_strategies import compare_masters
        from strategy_comparison.llm_crypto.crypto_reasoning import create_sample_context

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

        return results

    except ImportError:
        print("\n⚠️  Master strategy module not found.")
        print("   This feature requires the strategy_comparison package.")
        return None


# ============================================================================
# INTERACTIVE MENU SYSTEM
# ============================================================================

def show_menu():
    """Display interactive menu."""
    print("\n" + "🎯" * 40)
    print("ENHANCED COGNITIVE TRADING AGENT - Interactive Menu")
    print("🎯" * 40)

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              PART 1: COGNITIVE LOOP DEMONSTRATIONS                        ║
║   Full cognitive system with Perception → Memory → Reasoning → Learning  ║
╚═══════════════════════════════════════════════════════════════════════════╝

  1. Full Trading Scenario (5-day campaign with Investment Masters)
     → Complete cognitive loop with memory and learning
     → Multi-day trading decisions with feedback
     → Incorporates Buffett, Soros, Dalio perspectives

  2. Simple Reasoning Example
     → Quick test without perception/memory
     → Basic cognitive reasoning demonstration

╔═══════════════════════════════════════════════════════════════════════════╗
║              PART 2: STRATEGY ANALYSIS (Cognitive System)                 ║
║      Analyze trading strategies using cognitive reasoning framework       ║
╚═══════════════════════════════════════════════════════════════════════════╝

  3. Mean Reversion Strategy Analysis
     → Evaluate Z-score based mean reversion on SPY
     → Risk assessment with cognitive framework

  4. Pairs Trading Analysis
     → Analyze JPM/BAC pairs trade opportunity
     → Consider cointegration and market timing

  5. Portfolio Optimization
     → Design optimal multi-asset allocation
     → Balance risk/return across asset classes

╔═══════════════════════════════════════════════════════════════════════════╗
║          PART 3: PERFORMANCE COMPARISON (Traditional vs LLM)              ║
║       Test actual trading performance across different market regimes     ║
╚═══════════════════════════════════════════════════════════════════════════╝

  6. Multi-Regime Strategy Comparison
     → Test across bullish, bearish, sideways, volatile markets
     → Compare Traditional technical analysis vs LLM reasoning

╔═══════════════════════════════════════════════════════════════════════════╗
║              PART 4: INVESTMENT MASTER PERSONAS                           ║
║     Analyze markets from legendary investors' perspectives                ║
╚═══════════════════════════════════════════════════════════════════════════╝

  10. Jim Simons Style (Quantitative Renaissance)
  11. Warren Buffett Style (Value Investing)
  12. George Soros Style (Macro Trading & Reflexivity)
  13. Ray Dalio Style (All-Weather Risk Parity)
  14. Cathie Wood Style (Disruptive Innovation)
  15. ALL MASTERS Debate (Compare all perspectives!)

╔═══════════════════════════════════════════════════════════════════════════╗
║                         BATCH OPERATIONS                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

  7. Run ALL Strategy Analysis Demos (3-5)
  8. Run ALL Cognitive Demos (1-5)
  9. Run EVERYTHING (Complete Suite)

  0. Exit

""")


def run_demo(choice: int) -> bool:
    """Run selected demo. Returns True if should continue."""

    # Check API key for LLM-based demos
    if choice in range(1, 16):
        if not os.getenv("OPENAI_API_KEY"):
            print("\n⚠️  OPENAI_API_KEY not set!")
            print("   Some features will run in mock mode.")
            print("   For full LLM features: export OPENAI_API_KEY='your-key'")
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return True

    try:
        if choice == 1:
            run_trading_scenario()
        elif choice == 2:
            run_simple_example()
        elif choice == 3:
            demo_mean_reversion_analysis()
        elif choice == 4:
            demo_pairs_trading_analysis()
        elif choice == 5:
            demo_portfolio_optimization_analysis()
        elif choice == 6:
            demo_strategy_comparison()
        elif choice == 7:
            print_header("Running ALL Strategy Analysis Demos")
            demo_mean_reversion_analysis()
            demo_pairs_trading_analysis()
            demo_portfolio_optimization_analysis()
        elif choice == 8:
            print_header("Running ALL Cognitive Demos")
            run_trading_scenario()
            demo_mean_reversion_analysis()
            demo_pairs_trading_analysis()
            demo_portfolio_optimization_analysis()
        elif choice == 9:
            print_header("Running COMPLETE DEMO SUITE")
            run_trading_scenario()
            demo_mean_reversion_analysis()
            demo_pairs_trading_analysis()
            demo_portfolio_optimization_analysis()
            demo_strategy_comparison()
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
    print("ENHANCED COGNITIVE TRADING AGENT")
    print("🚀" * 40)

    print("""
This system demonstrates:

1. COGNITIVE LOOP: Full cognitive reasoning with memory and learning
   → Perception → Memory → Reasoning → Action → Learning
   → Multi-day trading scenarios with feedback

2. STRATEGY ANALYSIS: Evaluate trading strategies using cognitive framework
   → Mean reversion, pairs trading, portfolio optimization
   → Deep analysis with risk assessment

3. PERFORMANCE COMPARISON: Traditional vs LLM strategies
   → Test across multiple market regimes
   → Head-to-head performance metrics

4. INVESTMENT MASTERS: Legendary investors' perspectives
   → Simons, Buffett, Soros, Dalio, Wood
   → Multi-perspective debate mode
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
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Cognitive Trading Agent")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "simple", "interactive"],
        default="interactive",
        help="Demo mode: 'full' for complete cognitive loop, 'simple' for basic reasoning, 'interactive' for menu"
    )

    args = parser.parse_args()

    if args.mode == "full":
        run_trading_scenario()
    elif args.mode == "simple":
        run_simple_example()
    else:
        main()
