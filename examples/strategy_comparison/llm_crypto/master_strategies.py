"""
Investment Master Strategy Personas
Simulates different investment masters' thinking styles and strategies.

Inspired by:
- Jim Simons (Renaissance Technologies) - Quantitative, data-driven
- Warren Buffett (Berkshire Hathaway) - Value investing, long-term
- George Soros (Quantum Fund) - Macro trading, reflexivity
- Ray Dalio (Bridgewater) - Risk parity, all-weather
- Cathie Wood (ARK Invest) - Innovation, disruptive tech
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core import ReasoningGraph
from .crypto_reasoning import CryptoMarketContext, create_sample_context
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MasterPersona:
    """Investment master persona configuration."""
    name: str
    style: str
    philosophy: str
    key_metrics: List[str]
    risk_approach: str
    time_horizon: str


# ============================================================================
# MASTER PERSONAS
# ============================================================================

PERSONAS = {
    "simons": MasterPersona(
        name="Jim Simons",
        style="Quantitative Renaissance",
        philosophy="""
I am Jim Simons. I believe markets are inefficient and can be beaten through:
- Pure statistical analysis and pattern recognition
- No fundamental narratives - only data speaks
- High-frequency signals and mean reversion
- Mathematical models over human intuition
- Diversification across thousands of uncorrelated bets

I look for:
- Statistical anomalies and arbitrage opportunities
- Short-term predictable patterns
- Market microstructure inefficiencies
- Risk-adjusted returns through diversification
""",
        key_metrics=["sharpe_ratio", "statistical_significance", "turnover", "diversification"],
        risk_approach="Diversification + statistical risk management",
        time_horizon="Minutes to days"
    ),

    "buffett": MasterPersona(
        name="Warren Buffett",
        style="Value Investing",
        philosophy="""
I am Warren Buffett. My approach is simple:
- Buy wonderful businesses at fair prices
- Hold forever if possible
- Margin of safety is paramount
- Understand the business deeply
- Management quality matters

I ask:
- Does this asset have a moat (competitive advantage)?
- Is it undervalued relative to intrinsic value?
- Can I understand how it makes money?
- Will it still be relevant in 10-20 years?
- Am I buying at a price that protects me if I'm wrong?

I avoid:
- Speculation and momentum trading
- Assets I don't understand (like complex derivatives)
- Short-term market noise
- Following the crowd
""",
        key_metrics=["intrinsic_value", "margin_of_safety", "moat_strength", "long_term_growth"],
        risk_approach="Margin of safety + business quality",
        time_horizon="Years to decades"
    ),

    "soros": MasterPersona(
        name="George Soros",
        style="Macro Trading & Reflexivity",
        philosophy="""
I am George Soros. I believe in reflexivity - markets shape reality:
- Markets are always biased in one direction
- Identify boom-bust cycles early
- Make large, concentrated bets when thesis is strong
- Fundamentals matter, but market psychology matters more
- Be willing to change my mind quickly

I look for:
- Macroeconomic imbalances and bubbles
- Central bank policy mistakes
- Currency and sovereign debt crises
- Market sentiment extremes
- Positive feedback loops that can break

My trades are:
- Concentrated (high conviction)
- Macro-driven (top-down)
- Flexible (change quickly with new information)
""",
        key_metrics=["macro_imbalances", "sentiment_extremes", "policy_errors", "reflexive_loops"],
        risk_approach="Concentrated conviction + quick exits",
        time_horizon="Weeks to months"
    ),

    "dalio": MasterPersona(
        name="Ray Dalio",
        style="All-Weather Risk Parity",
        philosophy="""
I am Ray Dalio. I believe in balanced, diversified portfolios:
- No one knows what will happen
- Build an "All Weather" portfolio for any environment
- Balance risk, not dollars
- Economic machine runs in cycles
- Four scenarios: Growth↑/↓ + Inflation↑/↓

I construct portfolios by:
- Identifying uncorrelated return streams
- Equal risk contribution from each asset
- Protection in all economic environments
- Systematic rebalancing
- Stress testing for tail risks

Key principles:
- Diversification is the only free lunch
- Understand the economic machine
- Radically transparent analysis
- Pain + Reflection = Progress
""",
        key_metrics=["risk_parity", "correlation_matrix", "tail_risk", "all_weather_score"],
        risk_approach="Balanced diversification across scenarios",
        time_horizon="Years (with tactical adjustments)"
    ),

    "wood": MasterPersona(
        name="Cathie Wood",
        style="Disruptive Innovation",
        philosophy="""
I am Cathie Wood. I invest in the future:
- Focus on exponential growth technologies
- Disruptive innovation drives massive returns
- 5-year time horizon minimum
- High conviction, concentrated portfolio
- Willing to be early and volatile

I seek:
- Companies enabling technological disruption
- Exponential growth trajectories
- Winner-take-most network effects
- First movers with sustainable advantages
- Technologies reaching inflection points

Sectors I love:
- Artificial Intelligence / Machine Learning
- Blockchain / Cryptocurrency
- Genomic Revolution
- Energy Storage / Electric Vehicles
- Fintech / Digital Wallets

I embrace:
- High volatility for high potential returns
- Being contrarian when conviction is strong
- Long-term vision over short-term noise
""",
        key_metrics=["innovation_score", "growth_potential", "network_effects", "disruption_timeline"],
        risk_approach="High conviction + long-term vision",
        time_horizon="5-10 years"
    )
}


class MasterStrategyAnalyzer:
    """Analyze crypto strategies from different master perspectives."""

    def __init__(self, master_name: str, max_iterations: int = 3):
        """
        Initialize analyzer with specific master's style.

        Args:
            master_name: One of ['simons', 'buffett', 'soros', 'dalio', 'wood']
            max_iterations: Reasoning iterations
        """
        if master_name not in PERSONAS:
            raise ValueError(f"Unknown master: {master_name}. Choose from {list(PERSONAS.keys())}")

        self.master = PERSONAS[master_name]
        self.reasoning_graph = ReasoningGraph(max_iterations=max_iterations)

    def _build_master_prompt(self, context: CryptoMarketContext) -> str:
        """Build prompt with master's perspective."""
        # Calculate basic metrics
        prices = context.price_history
        returns_1d = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) > 1 else 0
        returns_7d = (prices[-1] - prices[-8]) / prices[-8] * 100 if len(prices) > 7 else 0
        returns_30d = (prices[-1] - prices[-31]) / prices[-31] * 100 if len(prices) > 30 else 0

        prompt = f"""
{self.master.philosophy}

Now, analyze this cryptocurrency opportunity through MY lens:

═══════════════════════════════════════════════════════════════
MARKET DATA: {context.symbol}
═══════════════════════════════════════════════════════════════

**Current Price:** ${context.current_price:,.2f}
**Market Cap:** ${context.market_cap / 1e9:.2f}B

**Performance:**
- 1-day:  {returns_1d:+.2f}%
- 7-day:  {returns_7d:+.2f}%
- 30-day: {returns_30d:+.2f}%

**Market Sentiment:**
- Fear & Greed Index: {context.fear_greed_index}/100
- Social Sentiment: {context.social_sentiment}
- BTC Correlation: {context.btc_correlation:.2f}

**On-Chain Metrics:**
{self._format_onchain(context.on_chain_metrics)}

**Recent News:**
{self._format_news(context.recent_news)}

═══════════════════════════════════════════════════════════════
ANALYSIS REQUIRED (in {self.master.name}'s style)
═══════════════════════════════════════════════════════════════

As {self.master.name}, answer these questions based on MY investment philosophy:

1. **Through My Lens:**
   - How do I view this opportunity given my style ({self.master.style})?
   - What metrics matter most to me? ({', '.join(self.master.key_metrics)})
   - Does this fit my time horizon ({self.master.time_horizon})?

2. **Decision Framework:**
   - Would I invest in this? WHY or WHY NOT?
   - If YES: How much (% of portfolio)? Entry strategy?
   - If NO: What would need to change for me to invest?

3. **Risk Assessment (my approach: {self.master.risk_approach}):**
   - What are the key risks from my perspective?
   - How would I manage or hedge these risks?
   - What's my exit strategy?

4. **Specific to My Style:**
"""

        # Add persona-specific questions
        if self.master.name == "Jim Simons":
            prompt += """
   - What statistical patterns do I see?
   - Is there enough trading volume for my models?
   - What's the Sharpe ratio potential?
   - Are there arbitrage opportunities?
"""
        elif self.master.name == "Warren Buffett":
            prompt += """
   - Does crypto have a sustainable competitive moat?
   - What is the intrinsic value (if any)?
   - Will this be relevant in 10-20 years?
   - Is there a margin of safety at current price?
"""
        elif self.master.name == "George Soros":
            prompt += """
   - What macro imbalances am I seeing?
   - Is there a reflexive boom-bust pattern forming?
   - What are central banks doing that affects this?
   - Where is market sentiment extreme?
"""
        elif self.master.name == "Ray Dalio":
            prompt += """
   - How does this fit in an All Weather portfolio?
   - What economic scenarios does this protect against?
   - What's the correlation with other assets?
   - How do I balance risk across assets?
"""
        elif self.master.name == "Cathie Wood":
            prompt += """
   - Is this truly disruptive innovation?
   - What's the 5-10 year growth potential?
   - Are we at an inflection point?
   - What network effects exist?
"""

        prompt += """

5. **Final Verdict:**
   - Action: BUY / SELL / HOLD
   - Confidence: 0-100%
   - Position Size: X% of portfolio
   - Reasoning: Explain in MY style and philosophy

IMPORTANT: Stay in character as {self.master.name}. Think and speak like I would.
Use my terminology, reference my key principles, and make decisions consistent with my track record.
"""

        return prompt

    def _format_onchain(self, metrics: Dict[str, float]) -> str:
        """Format on-chain metrics."""
        lines = []
        for key, value in metrics.items():
            if 'flow' in key.lower() or 'volume' in key.lower():
                lines.append(f"  - {key}: ${value/1e6:.2f}M")
            else:
                lines.append(f"  - {key}: {value:.2f}")
        return "\n".join(lines) if lines else "  - No data"

    def _format_news(self, news: List[str]) -> str:
        """Format news items."""
        if not news:
            return "  - No significant news"
        return "\n".join([f"  - {item}" for item in news])

    def analyze(self, context: CryptoMarketContext) -> Dict:
        """
        Analyze crypto opportunity as the chosen master.

        Returns:
            Analysis result with decision and reasoning
        """
        print("\n" + "🎩" * 40)
        print(f"ANALYZING AS: {self.master.name}")
        print(f"Style: {self.master.style}")
        print("🎩" * 40)

        # Build master-specific prompt
        prompt = self._build_master_prompt(context)

        # Run reasoning
        result = self.reasoning_graph.reason(prompt)

        return {
            'master': self.master.name,
            'style': self.master.style,
            'analysis': result["final_answer"],
            'confidence': result["confidence_score"],
            'iterations': result["iterations"],
            'reasoning_history': result["reasoning_history"]
        }


def compare_masters(context: CryptoMarketContext, masters: List[str] = None):
    """
    Compare multiple masters' perspectives on the same opportunity.

    Args:
        context: Market context to analyze
        masters: List of master names (default: all)
    """
    if masters is None:
        masters = list(PERSONAS.keys())

    print("\n" + "🏆" * 40)
    print("MULTI-MASTER COMPARISON")
    print("🏆" * 40)
    print(f"\nAnalyzing {context.symbol} @ ${context.current_price:,.2f}")
    print(f"Through the eyes of {len(masters)} investment masters...\n")

    results = []
    for master_name in masters:
        analyzer = MasterStrategyAnalyzer(master_name, max_iterations=2)
        result = analyzer.analyze(context)
        results.append(result)

        # Print summary
        print(f"\n{'='*80}")
        print(f"{result['master']} ({result['style']})")
        print(f"{'='*80}")
        print(result['analysis'][:500] + "..." if len(result['analysis']) > 500 else result['analysis'])
        print(f"\nConfidence: {result['confidence']:.2%}")

    # Summary comparison
    print("\n" + "📊" * 40)
    print("CONSENSUS ANALYSIS")
    print("📊" * 40)

    for result in results:
        analysis_lower = result['analysis'].lower()

        # Simple decision extraction
        if 'buy' in analysis_lower and 'not buy' not in analysis_lower:
            decision = "BUY"
        elif 'sell' in analysis_lower and 'not sell' not in analysis_lower:
            decision = "SELL"
        else:
            decision = "HOLD"

        print(f"\n{result['master']:20} → {decision:6} (Confidence: {result['confidence']:.0%})")

    return results


def example_simons_vs_buffett():
    """Example: Compare Simons (quant) vs Buffett (value) on Bitcoin."""
    print("\n" + "🥊" * 40)
    print("EXAMPLE: Simons vs Buffett on Bitcoin")
    print("🥊" * 40)

    context = create_sample_context("BTC", base_price=45000, trend="bullish")

    compare_masters(context, masters=['simons', 'buffett'])


def example_all_masters_crypto():
    """Example: All 5 masters analyze the same crypto."""
    print("\n" + "🌟" * 40)
    print("EXAMPLE: All Masters on Ethereum")
    print("🌟" * 40)

    context = create_sample_context("ETH", base_price=2500, trend="volatile")
    context.recent_news = [
        "Ethereum 2.0 staking yields attracting institutions",
        "DeFi protocols seeing record TVL growth",
        "Regulatory clarity improving in major markets"
    ]

    compare_masters(context, masters=['simons', 'buffett', 'soros', 'dalio', 'wood'])


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Run examples
    example_simons_vs_buffett()
    print("\n" + "="*80 + "\n")
    example_all_masters_crypto()
