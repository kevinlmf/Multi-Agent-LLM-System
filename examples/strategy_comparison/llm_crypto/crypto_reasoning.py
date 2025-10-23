"""
LLM-based Crypto Strategy Reasoning
Uses multi-agent reasoning system for cryptocurrency trading decisions.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core import ReasoningGraph
import numpy as np
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class CryptoMarketContext:
    """Crypto market context for LLM reasoning."""
    symbol: str
    current_price: float
    price_history: List[float]
    volume_history: List[float]
    market_cap: float
    btc_correlation: float
    fear_greed_index: float
    social_sentiment: str
    recent_news: List[str]
    on_chain_metrics: Dict[str, float]


class LLMCryptoStrategy:
    """
    LLM-powered cryptocurrency trading strategy.
    Uses multi-agent reasoning to analyze market conditions and make decisions.
    """

    def __init__(self, max_iterations: int = 3):
        """
        Initialize LLM crypto strategy.

        Args:
            max_iterations: Maximum reasoning iterations
        """
        self.reasoning_graph = ReasoningGraph(max_iterations=max_iterations)
        self.decision_history = []

    def _build_market_context_prompt(self, context: CryptoMarketContext) -> str:
        """Build comprehensive prompt with market context."""
        # Calculate technical indicators for context
        prices = np.array(context.price_history)
        returns = np.diff(prices) / prices[:-1] * 100
        volatility = np.std(returns)
        momentum_1d = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) > 1 else 0
        momentum_7d = (prices[-1] - prices[-8]) / prices[-8] * 100 if len(prices) > 7 else 0
        momentum_30d = (prices[-1] - prices[-31]) / prices[-31] * 100 if len(prices) > 30 else 0

        prompt = f"""
Analyze the following cryptocurrency trading opportunity and provide a comprehensive decision:

**Asset:** {context.symbol}
**Current Price:** ${context.current_price:,.2f}
**Market Cap:** ${context.market_cap / 1e9:.2f}B

**Price Action:**
- 1-day change: {momentum_1d:+.2f}%
- 7-day change: {momentum_7d:+.2f}%
- 30-day change: {momentum_30d:+.2f}%
- 30-day volatility: {volatility:.2f}%

**Market Sentiment:**
- Fear & Greed Index: {context.fear_greed_index}/100 ({self._interpret_fear_greed(context.fear_greed_index)})
- Social Sentiment: {context.social_sentiment}
- BTC Correlation: {context.btc_correlation:.2f}

**On-Chain Metrics:**
{self._format_onchain_metrics(context.on_chain_metrics)}

**Recent Market News:**
{self._format_news(context.recent_news)}

**Analysis Required:**

1. **Market Structure Analysis:**
   - What is the current market regime (trending/ranging/volatile)?
   - How does this crypto compare to BTC and broader market?
   - Are we in a risk-on or risk-off environment?

2. **Fundamental Analysis:**
   - What do the on-chain metrics suggest about network health?
   - Is there unusual activity (whale movements, exchange flows)?
   - How credible are the recent news catalysts?

3. **Risk Assessment:**
   - What are the key risks (technical, fundamental, macro)?
   - What is the potential downside vs upside?
   - How liquid is this asset?

4. **Trading Decision:**
   - Recommended Action: BUY / SELL / HOLD
   - Position Size: % of portfolio (0-100%)
   - Entry Strategy: Market / Limit / DCA
   - Stop Loss Level: $ price
   - Take Profit Targets: $ prices
   - Confidence Level: 0-100%
   - Time Horizon: Short-term (days) / Medium-term (weeks) / Long-term (months)

5. **Reasoning Quality:**
   - List all key assumptions
   - Identify potential blind spots
   - Rate your confidence in this analysis (0-100%)

Please provide a detailed, well-reasoned analysis considering all factors.
"""
        return prompt

    def _interpret_fear_greed(self, index: float) -> str:
        """Interpret Fear & Greed Index."""
        if index < 25:
            return "Extreme Fear"
        elif index < 45:
            return "Fear"
        elif index < 55:
            return "Neutral"
        elif index < 75:
            return "Greed"
        else:
            return "Extreme Greed"

    def _format_onchain_metrics(self, metrics: Dict[str, float]) -> str:
        """Format on-chain metrics for prompt."""
        lines = []
        for key, value in metrics.items():
            if 'ratio' in key.lower() or 'index' in key.lower():
                lines.append(f"  - {key}: {value:.2f}")
            elif 'flow' in key.lower() or 'volume' in key.lower():
                lines.append(f"  - {key}: ${value/1e6:.2f}M")
            else:
                lines.append(f"  - {key}: {value:.2f}")
        return "\n".join(lines) if lines else "  - No on-chain data available"

    def _format_news(self, news: List[str]) -> str:
        """Format news items for prompt."""
        if not news:
            return "  - No significant news"
        return "\n".join([f"  - {item}" for item in news])

    def analyze_opportunity(self, context: CryptoMarketContext) -> Dict:
        """
        Analyze a cryptocurrency trading opportunity using LLM reasoning.

        Args:
            context: Market context for the cryptocurrency

        Returns:
            Dictionary containing decision, reasoning, and metrics
        """
        print("\n" + "🔮" * 40)
        print(f"LLM REASONING: {context.symbol}")
        print("🔮" * 40)

        # Build prompt
        prompt = self._build_market_context_prompt(context)

        # Run multi-agent reasoning
        result = self.reasoning_graph.reason(prompt)

        # Parse decision from LLM response
        decision = self._parse_decision(result["final_answer"], context)

        # Store in history
        self.decision_history.append({
            'timestamp': datetime.now(),
            'symbol': context.symbol,
            'decision': decision,
            'reasoning': result["final_answer"],
            'confidence': result["confidence_score"],
            'iterations': result["iterations"]
        })

        return {
            'symbol': context.symbol,
            'action': decision['action'],
            'position_size': decision['position_size'],
            'confidence': result["confidence_score"],
            'entry_strategy': decision.get('entry_strategy', 'MARKET'),
            'stop_loss': decision.get('stop_loss', None),
            'take_profit': decision.get('take_profit', []),
            'reasoning': result["final_answer"],
            'reasoning_history': result["reasoning_history"],
            'iterations': result["iterations"],
            'llm_confidence': decision.get('llm_confidence', None)
        }

    def _parse_decision(self, llm_response: str, context: CryptoMarketContext) -> Dict:
        """
        Parse trading decision from LLM response.
        Uses heuristics to extract structured information.
        """
        response_lower = llm_response.lower()

        # Determine action
        if 'recommend' in response_lower:
            if 'buy' in response_lower and 'not buy' not in response_lower:
                action = 'BUY'
            elif 'sell' in response_lower and 'not sell' not in response_lower:
                action = 'SELL'
            else:
                action = 'HOLD'
        else:
            # Default to HOLD if unclear
            action = 'HOLD'

        # Extract position size (look for percentages)
        position_size = 0.0
        import re
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', llm_response)
        if percentages:
            # Take the first reasonable percentage mentioned
            for pct in percentages:
                pct_val = float(pct)
                if 1 <= pct_val <= 100:
                    position_size = pct_val / 100.0
                    break

        # If BUY action but no size, use conservative default
        if action == 'BUY' and position_size == 0:
            position_size = 0.10  # 10% default

        # Extract confidence
        confidence_patterns = [
            r'confidence[:\s]+(\d+)%',
            r'confidence[:\s]+(\d+)/100',
            r'(\d+)%\s+confidence'
        ]
        llm_confidence = None
        for pattern in confidence_patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                llm_confidence = float(matches[0]) / 100.0
                break

        # Extract stop loss (look for $ prices)
        stop_loss = None
        stop_patterns = [
            r'stop\s+loss[:\s]+\$?([\d,]+(?:\.\d+)?)',
            r'stop[:\s]+\$?([\d,]+(?:\.\d+)?)'
        ]
        for pattern in stop_patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                stop_loss = float(matches[0].replace(',', ''))
                break

        return {
            'action': action,
            'position_size': position_size,
            'entry_strategy': 'MARKET',  # Default
            'stop_loss': stop_loss,
            'take_profit': [],
            'llm_confidence': llm_confidence
        }

    def backtest(
        self,
        market_contexts: List[CryptoMarketContext],
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Backtest LLM strategy on historical scenarios.

        Args:
            market_contexts: List of market contexts over time
            initial_capital: Starting capital

        Returns:
            Performance metrics
        """
        capital = initial_capital
        position = 0  # units held
        trades = []
        portfolio_values = []

        print("\n" + "=" * 80)
        print("LLM STRATEGY BACKTEST")
        print("=" * 80)

        for i, context in enumerate(market_contexts):
            print(f"\n[{i+1}/{len(market_contexts)}] Analyzing {context.symbol} @ ${context.current_price:.2f}")

            # Get LLM decision
            decision = self.analyze_opportunity(context)

            # Execute trade based on decision
            if decision['action'] == 'BUY' and decision['position_size'] > 0 and capital > 0:
                investment = capital * decision['position_size']
                shares_to_buy = investment / context.current_price
                position += shares_to_buy
                capital -= investment

                trades.append({
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'price': context.current_price,
                    'shares': shares_to_buy,
                    'confidence': decision['confidence'],
                    'reasoning_summary': decision['reasoning'][:200] + "..."
                })
                print(f"  ✅ BUY {shares_to_buy:.4f} units @ ${context.current_price:.2f}")

            elif decision['action'] == 'SELL' and position > 0:
                sell_amount = position * decision['position_size']
                capital += sell_amount * context.current_price
                position -= sell_amount

                trades.append({
                    'timestamp': datetime.now(),
                    'action': 'SELL',
                    'price': context.current_price,
                    'shares': sell_amount,
                    'confidence': decision['confidence'],
                    'reasoning_summary': decision['reasoning'][:200] + "..."
                })
                print(f"  ✅ SELL {sell_amount:.4f} units @ ${context.current_price:.2f}")

            # Calculate portfolio value
            portfolio_value = capital + (position * context.current_price)
            portfolio_values.append(portfolio_value)
            print(f"  💰 Portfolio Value: ${portfolio_value:,.2f}")

        # Final metrics
        final_value = capital + (position * market_contexts[-1].current_price if position > 0 else 0)
        total_return = (final_value - initial_capital) / initial_capital

        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

        max_drawdown = 0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'strategy_type': 'LLM_REASONING',
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values.tolist(),
            'avg_confidence': np.mean([t.get('confidence', 0) for t in trades]) if trades else 0
        }


def create_sample_context(
    symbol: str = "BTC",
    base_price: float = 45000,
    trend: str = "bullish"
) -> CryptoMarketContext:
    """Create sample market context for testing."""
    np.random.seed(42)

    # Generate price history
    n_days = 30
    if trend == "bullish":
        drift = 0.02
    elif trend == "bearish":
        drift = -0.02
    else:
        drift = 0

    prices = [base_price]
    for _ in range(n_days):
        change = np.random.randn() * 0.03 + drift
        prices.append(prices[-1] * (1 + change))

    volumes = [np.random.uniform(1e9, 5e9) for _ in range(n_days)]

    return CryptoMarketContext(
        symbol=symbol,
        current_price=prices[-1],
        price_history=prices,
        volume_history=volumes,
        market_cap=850e9,
        btc_correlation=1.0 if symbol == "BTC" else 0.75,
        fear_greed_index=65,  # Greed
        social_sentiment="Bullish - High social media activity",
        recent_news=[
            "Institutional adoption increasing",
            "Major exchange reports record volumes",
            "Regulatory clarity improving in key markets"
        ],
        on_chain_metrics={
            'exchange_netflow': -50e6,  # Negative = outflow (bullish)
            'active_addresses': 950000,
            'transaction_volume': 25e9,
            'mvrv_ratio': 2.1,
            'nvt_ratio': 45
        }
    )


def example_usage():
    """Example usage of LLM crypto strategy."""
    # Create strategy
    strategy = LLMCryptoStrategy(max_iterations=3)

    # Analyze single opportunity
    context = create_sample_context("BTC", base_price=45000, trend="bullish")
    decision = strategy.analyze_opportunity(context)

    print("\n" + "=" * 80)
    print("LLM DECISION SUMMARY")
    print("=" * 80)
    print(f"Action: {decision['action']}")
    print(f"Position Size: {decision['position_size']:.1%}")
    print(f"Confidence: {decision['confidence']:.2%}")
    print(f"Iterations: {decision['iterations']}")
    print("\nFull Reasoning:")
    print(decision['reasoning'])

    return strategy, decision


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    example_usage()
