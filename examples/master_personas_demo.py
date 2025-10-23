"""
Investment Master Personas Demo
Quick demonstration of how different legendary investors would analyze the same opportunity.

Usage:
    python master_personas_demo.py              # Interactive mode
    python master_personas_demo.py simons       # Specific master
    python master_personas_demo.py compare      # All masters debate
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show_masters_info():
    """Display information about all masters."""
    from strategy_comparison.llm_crypto.master_strategies import PERSONAS

    print("\n" + "🎩" * 40)
    print("INVESTMENT MASTER PERSONAS")
    print("🎩" * 40)

    print("""
This system allows you to analyze cryptocurrency opportunities through the lens
of 5 legendary investors. Each has a distinct philosophy and decision-making style.
""")

    for key, persona in PERSONAS.items():
        print(f"\n{'='*80}")
        print(f"  {persona.name} - {persona.style}")
        print(f"{'='*80}")
        print(f"  Time Horizon: {persona.time_horizon}")
        print(f"  Risk Approach: {persona.risk_approach}")
        print(f"  Key Metrics: {', '.join(persona.key_metrics[:3])}...")


def quick_demo(master_name: str = 'simons'):
    """Run quick demo with specified master."""
    from strategy_comparison.llm_crypto.master_strategies import MasterStrategyAnalyzer, PERSONAS
    from strategy_comparison.llm_crypto.crypto_reasoning import create_sample_context

    if master_name not in PERSONAS:
        print(f"❌ Unknown master: {master_name}")
        print(f"   Available: {', '.join(PERSONAS.keys())}")
        return

    master = PERSONAS[master_name]

    print("\n" + "🎩" * 40)
    print(f"ANALYZING AS: {master.name}")
    print(f"Style: {master.style}")
    print("🎩" * 40)

    # Create scenario
    print("\n📊 Creating market scenario...")
    context = create_sample_context("BTC", base_price=45000, trend="bullish")

    print(f"\n💰 Bitcoin @ ${context.current_price:,.2f}")
    print(f"📈 30-day trend: Bullish")
    print(f"😱 Fear & Greed: {context.fear_greed_index}/100")

    # Analyze
    print(f"\n🧠 {master.name} is analyzing...")
    analyzer = MasterStrategyAnalyzer(master_name, max_iterations=2)
    result = analyzer.analyze(context)

    print("\n" + "="*80)
    print("ANALYSIS RESULT")
    print("="*80)
    print(result['analysis'])
    print(f"\nConfidence: {result['confidence']:.2%}")
    print(f"Reasoning Iterations: {result['iterations']}")


def compare_all_masters():
    """Compare all masters on the same opportunity."""
    from strategy_comparison.llm_crypto.master_strategies import compare_masters, PERSONAS
    from strategy_comparison.llm_crypto.crypto_reasoning import create_sample_context

    print("\n" + "🏆" * 40)
    print("ALL MASTERS DEBATE")
    print("🏆" * 40)

    print("""
We'll present the same cryptocurrency opportunity to all 5 masters
and see how their different philosophies lead to different conclusions.

Scenario: Ethereum during a volatile market period
""")

    context = create_sample_context("ETH", base_price=2500, trend="volatile")
    context.recent_news = [
        "Ethereum 2.0 staking yields attracting institutional capital",
        "DeFi protocols experiencing record Total Value Locked (TVL)",
        "Regulatory clarity improving in major developed markets",
        "Network activity at all-time highs"
    ]

    print(f"\n💎 Ethereum @ ${context.current_price:,.2f}")
    print(f"📊 Market Cap: ${context.market_cap/1e9:.1f}B")
    print(f"📈 Trend: Volatile with positive catalysts")

    input("\n⏸️  Press Enter to start the debate...")

    results = compare_masters(context)

    # Consensus analysis
    print("\n" + "📊" * 40)
    print("CONSENSUS ANALYSIS")
    print("📊" * 40)

    decisions = []
    confidence_scores = []

    for result in results:
        analysis_lower = result['analysis'].lower()

        # Extract decision
        if 'buy' in analysis_lower and 'not buy' not in analysis_lower:
            decision = "BUY"
        elif 'sell' in analysis_lower and 'not sell' not in analysis_lower:
            decision = "SELL"
        else:
            decision = "HOLD"

        decisions.append(decision)
        confidence_scores.append(result['confidence'])

        # Extract position size if mentioned
        import re
        position_match = re.search(r'(\d+)%\s+(?:of\s+)?portfolio', analysis_lower)
        position_size = position_match.group(1) if position_match else "?"

        print(f"\n{result['master']:20} → {decision:6} ({position_size}% position) @ {result['confidence']:.0%} confidence")

    # Summary
    buy_count = decisions.count('BUY')
    sell_count = decisions.count('SELL')
    hold_count = decisions.count('HOLD')
    avg_confidence = sum(confidence_scores) / len(confidence_scores)

    print(f"\n" + "="*80)
    print(f"VERDICT: {buy_count} BUY | {hold_count} HOLD | {sell_count} SELL")
    print(f"Average Confidence: {avg_confidence:.1%}")
    print("="*80)

    if buy_count > sell_count + hold_count:
        print("\n✅ BULLISH CONSENSUS")
        print("   Multiple masters see opportunity despite different methodologies")
        print("   → Consider position sizing based on your risk tolerance")
    elif sell_count > buy_count + hold_count:
        print("\n❌ BEARISH CONSENSUS")
        print("   Masters advise caution or avoidance")
        print("   → Respect the collective wisdom, wait for better entry")
    else:
        print("\n⚖️  NO CLEAR CONSENSUS")
        print("   Diverse opinions across different investment philosophies")
        print("   → Indicates complex trade-off, proceed with extra caution")
        print("   → Consider hybrid approach or smaller position size")


def interactive_mode():
    """Interactive menu for exploring master personas."""
    from strategy_comparison.llm_crypto.master_strategies import PERSONAS

    while True:
        print("\n" + "🎯" * 40)
        print("MASTER PERSONAS - Interactive Mode")
        print("🎯" * 40)

        print("""
Choose an option:

  1. Jim Simons (Quantitative)
  2. Warren Buffett (Value Investing)
  3. George Soros (Macro Trading)
  4. Ray Dalio (Risk Parity)
  5. Cathie Wood (Innovation)

  6. Compare ALL Masters (Recommended!)
  7. Show Masters Info
  0. Exit
""")

        try:
            choice = int(input("Select option (0-7): "))

            if choice == 0:
                print("\n👋 Goodbye!")
                break
            elif choice == 1:
                quick_demo('simons')
            elif choice == 2:
                quick_demo('buffett')
            elif choice == 3:
                quick_demo('soros')
            elif choice == 4:
                quick_demo('dalio')
            elif choice == 5:
                quick_demo('wood')
            elif choice == 6:
                compare_all_masters()
            elif choice == 7:
                show_masters_info()
            else:
                print("\n❌ Invalid choice")

            if choice != 0:
                input("\n✅ Press Enter to continue...")

        except ValueError:
            print("\n❌ Please enter a number 0-7")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


def main():
    """Main entry point."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set")
        print("   This demo requires OpenAI API access")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        response = input("\n   Continue anyway to see menu? (y/n): ")
        if response.lower() != 'y':
            return

    # Check command line args
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg == 'compare':
            compare_all_masters()
        elif arg == 'info':
            show_masters_info()
        elif arg in ['simons', 'buffett', 'soros', 'dalio', 'wood']:
            quick_demo(arg)
        else:
            print(f"❌ Unknown command: {arg}")
            print("   Usage: python master_personas_demo.py [simons|buffett|soros|dalio|wood|compare|info]")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
