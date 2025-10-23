"""
Visualization and Analysis Dashboard
Creates charts and analysis for strategy comparison results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os


class ResultsVisualizer:
    """Visualize strategy comparison results."""

    def __init__(self, results: Dict = None, results_file: str = None):
        """
        Initialize visualizer.

        Args:
            results: Results dictionary
            results_file: Path to results JSON file
        """
        if results:
            self.results = results
        elif results_file:
            with open(results_file, 'r') as f:
                self.results = json.load(f)
        else:
            raise ValueError("Must provide either results dict or results_file path")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)

    def plot_portfolio_comparison(self, save_path: str = None):
        """Plot portfolio value evolution for both strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        trad = self.results['traditional_strategy']
        llm = self.results['llm_strategy']

        # Portfolio values over time
        ax1 = axes[0, 0]
        if trad['portfolio_values']:
            ax1.plot(trad['portfolio_values'], label='Traditional', linewidth=2, color='blue')
        if llm['portfolio_values']:
            ax1.plot(llm['portfolio_values'], label='LLM', linewidth=2, color='green')
        ax1.axhline(y=self.results['experiment_config']['initial_capital'],
                    color='red', linestyle='--', label='Initial Capital', alpha=0.5)
        ax1.set_title('Portfolio Value Evolution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Returns comparison
        ax2 = axes[0, 1]
        strategies = ['Traditional', 'LLM']
        returns = [trad['total_return_pct'], llm['total_return_pct']]
        colors = ['blue' if r < 0 else 'green' for r in returns]
        bars = ax2.bar(strategies, returns, color=colors, alpha=0.7)
        ax2.set_title('Total Return Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom' if height > 0 else 'top')

        # Sharpe ratio comparison
        ax3 = axes[1, 0]
        sharpe_ratios = [trad['sharpe_ratio'], llm['sharpe_ratio']]
        colors = ['blue', 'green']
        bars = ax3.bar(strategies, sharpe_ratios, color=colors, alpha=0.7)
        ax3.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top')

        # Max drawdown comparison
        ax4 = axes[1, 1]
        drawdowns = [trad['max_drawdown_pct'], llm['max_drawdown_pct']]
        bars = ax4.bar(strategies, drawdowns, color=['red', 'orange'], alpha=0.7)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved portfolio comparison to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_trade_analysis(self, save_path: str = None):
        """Analyze trading patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        trad = self.results['traditional_strategy']
        llm = self.results['llm_strategy']

        # Trade count
        ax1 = axes[0, 0]
        strategies = ['Traditional', 'LLM']
        trade_counts = [trad['num_trades'], llm['num_trades']]
        ax1.bar(strategies, trade_counts, color=['blue', 'green'], alpha=0.7)
        ax1.set_title('Number of Trades', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Trade Count')
        ax1.grid(True, alpha=0.3, axis='y')

        for i, count in enumerate(trade_counts):
            ax1.text(i, count, str(count), ha='center', va='bottom')

        # Buy vs Sell distribution
        ax2 = axes[0, 1]
        trad_buys = sum(1 for t in trad['trades'] if t['action'] == 'BUY')
        trad_sells = trad['num_trades'] - trad_buys
        llm_buys = sum(1 for t in llm['trades'] if t['action'] == 'BUY')
        llm_sells = llm['num_trades'] - llm_buys

        x = np.arange(2)
        width = 0.35
        ax2.bar(x - width/2, [trad_buys, trad_sells], width, label='Traditional', color='blue', alpha=0.7)
        ax2.bar(x + width/2, [llm_buys, llm_sells], width, label='LLM', color='green', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Buy', 'Sell'])
        ax2.set_title('Trade Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Confidence scores (if available)
        ax3 = axes[1, 0]
        if 'avg_confidence' in llm:
            confidence_data = []
            labels = []

            if trad.get('trades'):
                trad_conf = [t.get('confidence', 0) for t in trad['trades'] if t.get('confidence')]
                if trad_conf:
                    confidence_data.append(trad_conf)
                    labels.append('Traditional')

            if llm.get('trades'):
                llm_conf = [t.get('confidence', 0) for t in llm['trades'] if t.get('confidence')]
                if llm_conf:
                    confidence_data.append(llm_conf)
                    labels.append('LLM')

            if confidence_data:
                ax3.boxplot(confidence_data, labels=labels)
                ax3.set_title('Trade Confidence Distribution', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Confidence Score')
                ax3.grid(True, alpha=0.3, axis='y')
            else:
                ax3.text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=ax3.transAxes)

        # Key metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f"""
        SUMMARY STATISTICS
        ═══════════════════════════════════

        Traditional Strategy:
        • Final Value: ${trad['final_value']:,.2f}
        • Return: {trad['total_return_pct']:+.2f}%
        • Sharpe: {trad['sharpe_ratio']:.2f}
        • Max DD: {trad['max_drawdown_pct']:.2f}%
        • Trades: {trad['num_trades']}

        LLM Strategy:
        • Final Value: ${llm['final_value']:,.2f}
        • Return: {llm['total_return_pct']:+.2f}%
        • Sharpe: {llm['sharpe_ratio']:.2f}
        • Max DD: {llm['max_drawdown_pct']:.2f}%
        • Trades: {llm['num_trades']}

        Winner: {self.results['comparison_metrics']['winner']}
        Return Improvement: {self.results['comparison_metrics']['return_improvement']:+.2f}%
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved trade analysis to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_risk_metrics(self, save_path: str = None):
        """Plot risk-adjusted performance metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        trad = self.results['traditional_strategy']
        llm = self.results['llm_strategy']

        # Return vs Risk scatter
        ax1 = axes[0]
        returns = [trad['total_return_pct'], llm['total_return_pct']]
        risks = [trad['max_drawdown_pct'], llm['max_drawdown_pct']]
        colors = ['blue', 'green']
        labels = ['Traditional', 'LLM']

        for i in range(2):
            ax1.scatter(risks[i], returns[i], s=300, c=colors[i], alpha=0.6, label=labels[i])
            ax1.annotate(labels[i], (risks[i], returns[i]),
                        xytext=(10, 10), textcoords='offset points', fontsize=12)

        ax1.set_xlabel('Max Drawdown (%)', fontsize=12)
        ax1.set_ylabel('Total Return (%)', fontsize=12)
        ax1.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Risk-adjusted return (Return / Drawdown ratio)
        ax2 = axes[1]
        risk_adj_trad = trad['total_return_pct'] / trad['max_drawdown_pct'] if trad['max_drawdown_pct'] > 0 else 0
        risk_adj_llm = llm['total_return_pct'] / llm['max_drawdown_pct'] if llm['max_drawdown_pct'] > 0 else 0

        strategies = ['Traditional', 'LLM']
        risk_adj_returns = [risk_adj_trad, risk_adj_llm]
        bars = ax2.bar(strategies, risk_adj_returns, color=['blue', 'green'], alpha=0.7)
        ax2.set_title('Risk-Adjusted Return\n(Return / Max Drawdown)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved risk metrics to: {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_full_report(self, output_dir: str = "comparison_results"):
        """Generate full visualization report."""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "📊" * 40)
        print("GENERATING VISUALIZATION REPORT")
        print("📊" * 40)

        regime = self.results['experiment_config'].get('regime', 'unknown')

        # Generate all plots
        self.plot_portfolio_comparison(f"{output_dir}/portfolio_comparison_{regime}.png")
        self.plot_trade_analysis(f"{output_dir}/trade_analysis_{regime}.png")
        self.plot_risk_metrics(f"{output_dir}/risk_metrics_{regime}.png")

        print(f"\n✅ Full report generated in: {output_dir}/")


def visualize_all_results(results_dir: str = "comparison_results"):
    """Visualize all result files in a directory."""
    if not os.path.exists(results_dir):
        print(f"❌ Directory not found: {results_dir}")
        return

    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    if not json_files:
        print(f"❌ No JSON result files found in {results_dir}")
        return

    print(f"\n📊 Found {len(json_files)} result files")

    for json_file in json_files:
        filepath = os.path.join(results_dir, json_file)
        print(f"\n{'='*80}")
        print(f"Visualizing: {json_file}")
        print(f"{'='*80}")

        try:
            visualizer = ResultsVisualizer(results_file=filepath)
            visualizer.generate_full_report(results_dir)
        except Exception as e:
            print(f"❌ Error visualizing {json_file}: {e}")

    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    # Visualize all results in the comparison_results directory
    visualize_all_results("comparison_results")
