"""
Strategy Comparison Framework
Compares Traditional Finance vs LLM-based Crypto strategies.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List
import matplotlib.pyplot as plt
from traditional_finance.technical_strategy import TechnicalStrategy
from llm_crypto.crypto_reasoning import LLMCryptoStrategy, CryptoMarketContext, create_sample_context


class StrategyComparison:
    """
    Framework for comparing Traditional vs LLM-based trading strategies.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.results = {}

    def generate_market_scenarios(
        self,
        n_scenarios: int = 10,
        base_price: float = 45000,
        regime: str = "mixed"
    ) -> tuple:
        """
        Generate market scenarios for testing.

        Args:
            n_scenarios: Number of scenarios to generate
            base_price: Starting price
            regime: Market regime ("bullish", "bearish", "mixed", "volatile")

        Returns:
            Tuple of (prices, contexts) for testing
        """
        np.random.seed(42)
        scenarios = []

        # Generate diverse market conditions
        regimes = []
        if regime == "mixed":
            regimes = ["bullish", "bearish", "sideways", "volatile"] * (n_scenarios // 4 + 1)
            regimes = regimes[:n_scenarios]
        else:
            regimes = [regime] * n_scenarios

        all_prices = [base_price]
        contexts = []

        for i, current_regime in enumerate(regimes):
            # Generate prices based on regime
            if current_regime == "bullish":
                drift = 0.03
                volatility = 0.02
            elif current_regime == "bearish":
                drift = -0.03
                volatility = 0.02
            elif current_regime == "sideways":
                drift = 0.0
                volatility = 0.015
            else:  # volatile
                drift = 0.0
                volatility = 0.05

            # Generate daily prices
            daily_prices = []
            current_price = all_prices[-1]
            for day in range(30):  # 30 days per scenario
                change = np.random.randn() * volatility + drift / 30
                current_price *= (1 + change)
                daily_prices.append(current_price)

            all_prices.extend(daily_prices)

            # Create market context for LLM
            fear_greed = self._calculate_fear_greed(current_regime, daily_prices)
            news = self._generate_news(current_regime)

            context = CryptoMarketContext(
                symbol="BTC",
                current_price=daily_prices[-1],
                price_history=all_prices[-60:],  # Last 60 days
                volume_history=[np.random.uniform(1e9, 5e9) for _ in range(len(daily_prices))],
                market_cap=850e9,
                btc_correlation=1.0,
                fear_greed_index=fear_greed,
                social_sentiment=self._get_sentiment(current_regime),
                recent_news=news,
                on_chain_metrics=self._generate_onchain_metrics(current_regime)
            )
            contexts.append(context)

        return np.array(all_prices), contexts

    def _calculate_fear_greed(self, regime: str, prices: List[float]) -> float:
        """Calculate fear & greed index based on regime."""
        base_values = {
            "bullish": 70,
            "bearish": 25,
            "sideways": 50,
            "volatile": 40
        }
        base = base_values.get(regime, 50)
        noise = np.random.uniform(-10, 10)
        return np.clip(base + noise, 0, 100)

    def _get_sentiment(self, regime: str) -> str:
        """Get social sentiment based on regime."""
        sentiments = {
            "bullish": "Very bullish - Strong social media FOMO",
            "bearish": "Bearish - Fear and negative sentiment prevailing",
            "sideways": "Neutral - Mixed sentiment, awaiting catalyst",
            "volatile": "Uncertain - High fear due to volatility"
        }
        return sentiments.get(regime, "Neutral")

    def _generate_news(self, regime: str) -> List[str]:
        """Generate news based on regime."""
        news_items = {
            "bullish": [
                "Major institutional investment announced",
                "Positive regulatory developments",
                "Strong adoption metrics reported"
            ],
            "bearish": [
                "Regulatory concerns increasing",
                "Major exchange reports issues",
                "Macroeconomic headwinds strengthening"
            ],
            "sideways": [
                "Market consolidating after recent moves",
                "Institutional interest steady",
                "Awaiting macro catalysts"
            ],
            "volatile": [
                "High volatility due to uncertainty",
                "Mixed signals from regulators",
                "Conflicting macro data"
            ]
        }
        return news_items.get(regime, ["No significant news"])

    def _generate_onchain_metrics(self, regime: str) -> Dict[str, float]:
        """Generate on-chain metrics based on regime."""
        base_metrics = {
            "bullish": {
                'exchange_netflow': -100e6,  # Strong outflow
                'active_addresses': 1000000,
                'transaction_volume': 30e9,
                'mvrv_ratio': 2.5,
                'nvt_ratio': 40
            },
            "bearish": {
                'exchange_netflow': 150e6,  # Inflow (selling pressure)
                'active_addresses': 700000,
                'transaction_volume': 15e9,
                'mvrv_ratio': 1.2,
                'nvt_ratio': 65
            },
            "sideways": {
                'exchange_netflow': 10e6,
                'active_addresses': 850000,
                'transaction_volume': 22e9,
                'mvrv_ratio': 1.8,
                'nvt_ratio': 50
            },
            "volatile": {
                'exchange_netflow': 50e6,
                'active_addresses': 900000,
                'transaction_volume': 28e9,
                'mvrv_ratio': 1.9,
                'nvt_ratio': 55
            }
        }
        return base_metrics.get(regime, base_metrics["sideways"])

    def run_traditional_strategy(self, prices: np.ndarray) -> Dict:
        """Run traditional technical analysis strategy."""
        print("\n" + "📊" * 40)
        print("RUNNING TRADITIONAL STRATEGY")
        print("📊" * 40)

        strategy = TechnicalStrategy(
            short_window=20,
            long_window=50,
            rsi_period=14
        )

        timestamps = [datetime.now() + timedelta(days=i) for i in range(len(prices))]
        results = strategy.backtest(prices, timestamps, self.initial_capital)
        results['strategy_type'] = 'TRADITIONAL'

        return results

    def run_llm_strategy(self, contexts: List[CryptoMarketContext]) -> Dict:
        """Run LLM-based reasoning strategy."""
        print("\n" + "🤖" * 40)
        print("RUNNING LLM STRATEGY")
        print("🤖" * 40)

        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not set. Using mock LLM responses.")
            # Return mock results for demonstration
            return self._mock_llm_results(contexts)

        strategy = LLMCryptoStrategy(max_iterations=2)  # Reduce iterations for speed
        results = strategy.backtest(contexts, self.initial_capital)

        return results

    def _mock_llm_results(self, contexts: List[CryptoMarketContext]) -> Dict:
        """Generate mock LLM results for demonstration when no API key."""
        # Simple momentum-based mock
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = []

        for context in contexts:
            prices = context.price_history[-30:]
            momentum = (prices[-1] - prices[0]) / prices[0]

            # Mock decision based on momentum
            if momentum > 0.05 and capital > 0:  # Strong positive momentum
                investment = capital * 0.2
                shares = investment / context.current_price
                position += shares
                capital -= investment
                trades.append({
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'price': context.current_price,
                    'confidence': 0.7
                })
            elif momentum < -0.05 and position > 0:  # Strong negative momentum
                proceeds = position * context.current_price
                capital += proceeds
                trades.append({
                    'timestamp': datetime.now(),
                    'action': 'SELL',
                    'price': context.current_price,
                    'confidence': 0.7
                })
                position = 0

            portfolio_value = capital + (position * context.current_price)
            portfolio_values.append(portfolio_value)

        final_value = capital + (position * contexts[-1].current_price if position > 0 else 0)
        total_return = (final_value - self.initial_capital) / self.initial_capital

        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        return {
            'strategy_type': 'LLM_REASONING (MOCK)',
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': 0.15,
            'max_drawdown_pct': 15.0,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values.tolist()
        }

    def compare(self, n_scenarios: int = 10, regime: str = "mixed") -> Dict:
        """
        Run comprehensive comparison between strategies.

        Args:
            n_scenarios: Number of market scenarios to test
            regime: Market regime type

        Returns:
            Comparison results
        """
        print("\n" + "🔬" * 40)
        print("STRATEGY COMPARISON EXPERIMENT")
        print("🔬" * 40)
        print(f"Scenarios: {n_scenarios}")
        print(f"Regime: {regime}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")

        # Generate scenarios
        prices, contexts = self.generate_market_scenarios(n_scenarios, regime=regime)

        # Run both strategies
        trad_results = self.run_traditional_strategy(prices)
        llm_results = self.run_llm_strategy(contexts)

        # Compare results
        comparison = {
            'experiment_config': {
                'n_scenarios': n_scenarios,
                'regime': regime,
                'initial_capital': self.initial_capital,
                'timestamp': datetime.now().isoformat()
            },
            'traditional_strategy': trad_results,
            'llm_strategy': llm_results,
            'comparison_metrics': self._calculate_comparison_metrics(trad_results, llm_results)
        }

        self.results = comparison
        return comparison

    def _calculate_comparison_metrics(self, trad: Dict, llm: Dict) -> Dict:
        """Calculate relative comparison metrics."""
        return {
            'return_difference': llm['total_return_pct'] - trad['total_return_pct'],
            'sharpe_difference': llm['sharpe_ratio'] - trad['sharpe_ratio'],
            'drawdown_difference': llm['max_drawdown_pct'] - trad['max_drawdown_pct'],
            'trade_count_difference': llm['num_trades'] - trad['num_trades'],
            'winner': 'LLM' if llm['total_return'] > trad['total_return'] else 'Traditional',
            'return_improvement': ((llm['total_return'] - trad['total_return']) / abs(trad['total_return']) * 100) if trad['total_return'] != 0 else 0
        }

    def print_results(self):
        """Pretty print comparison results."""
        if not self.results:
            print("No results to display. Run compare() first.")
            return

        trad = self.results['traditional_strategy']
        llm = self.results['llm_strategy']
        comp = self.results['comparison_metrics']

        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        print("\n📊 TRADITIONAL STRATEGY")
        print(f"  Final Value: ${trad['final_value']:,.2f}")
        print(f"  Total Return: {trad['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio: {trad['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {trad['max_drawdown_pct']:.2f}%")
        print(f"  Number of Trades: {trad['num_trades']}")

        print("\n🤖 LLM STRATEGY")
        print(f"  Final Value: ${llm['final_value']:,.2f}")
        print(f"  Total Return: {llm['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio: {llm['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {llm['max_drawdown_pct']:.2f}%")
        print(f"  Number of Trades: {llm['num_trades']}")

        print("\n🏆 COMPARISON")
        print(f"  Winner: {comp['winner']}")
        print(f"  Return Difference: {comp['return_difference']:+.2f}%")
        print(f"  Sharpe Difference: {comp['sharpe_difference']:+.2f}")
        print(f"  Return Improvement: {comp['return_improvement']:+.2f}%")

    def save_results(self, filename: str = None):
        """Save results to JSON file."""
        if not self.results:
            print("No results to save.")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_results/strategy_comparison_{timestamp}.json"

        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main execution function."""
    print("\n" + "🚀" * 40)
    print("TRADITIONAL vs LLM STRATEGY COMPARISON")
    print("🚀" * 40)

    # Initialize comparison
    comparison = StrategyComparison(initial_capital=10000)

    # Run comparison across different market regimes
    regimes = ["bullish", "bearish", "mixed", "volatile"]

    all_results = []
    for regime in regimes:
        print(f"\n\n{'='*80}")
        print(f"Testing Regime: {regime.upper()}")
        print(f"{'='*80}")

        results = comparison.compare(n_scenarios=8, regime=regime)
        comparison.print_results()

        # Save individual results
        comparison.save_results(f"comparison_results/comparison_{regime}.json")

        all_results.append({
            'regime': regime,
            'results': results
        })

    # Summary across all regimes
    print("\n\n" + "=" * 80)
    print("SUMMARY ACROSS ALL REGIMES")
    print("=" * 80)

    for result in all_results:
        regime = result['regime']
        comp = result['results']['comparison_metrics']
        print(f"\n{regime.upper():>12}: Winner = {comp['winner']:>11} | Return Diff = {comp['return_difference']:+7.2f}%")

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
