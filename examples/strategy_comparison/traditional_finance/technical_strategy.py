"""
Traditional Finance Strategy - Technical Indicators Based
A baseline implementation using classic technical analysis methods.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    """Trading signal."""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price: float
    indicators: Dict[str, float]
    reason: str


class TechnicalStrategy:
    """
    Traditional technical analysis strategy combining multiple indicators:
    - Moving Averages (SMA, EMA)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Volume analysis
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def calculate_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        return np.convolve(prices, np.ones(window)/window, mode='valid')

    def calculate_ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2 / (window + 1)

        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    def calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, Signal line, and Histogram."""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd, signal)
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(prices, window)
        # Pad the beginning to match length
        sma_full = np.concatenate([np.full(window-1, np.nan), sma])

        std = np.array([
            np.std(prices[max(0, i-window+1):i+1])
            for i in range(len(prices))
        ])

        upper_band = sma_full + (std * num_std)
        lower_band = sma_full - (std * num_std)

        return upper_band, sma_full, lower_band

    def generate_signal(
        self,
        prices: np.ndarray,
        volumes: np.ndarray = None,
        timestamp: datetime = None
    ) -> Signal:
        """
        Generate trading signal based on technical indicators.

        Returns:
            Signal object with action, confidence, and reasoning
        """
        if len(prices) < self.long_window:
            return Signal(
                timestamp=timestamp or datetime.now(),
                action='HOLD',
                confidence=0.0,
                price=prices[-1],
                indicators={},
                reason="Insufficient data for analysis"
            )

        current_price = prices[-1]

        # Calculate indicators
        sma_short = self.calculate_sma(prices, self.short_window)[-1]
        sma_long = self.calculate_sma(prices, self.long_window)[-1]
        rsi = self.calculate_rsi(prices, self.rsi_period)[-1]
        macd, signal_line, histogram = self.calculate_macd(prices)
        macd_val, signal_val, hist_val = macd[-1], signal_line[-1], histogram[-1]

        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(prices)
        bb_position = (current_price - lower_bb[-1]) / (upper_bb[-1] - lower_bb[-1])

        indicators = {
            'sma_short': sma_short,
            'sma_long': sma_long,
            'rsi': rsi,
            'macd': macd_val,
            'signal_line': signal_val,
            'macd_histogram': hist_val,
            'bb_position': bb_position,
            'current_price': current_price
        }

        # Decision logic
        buy_signals = 0
        sell_signals = 0
        reasons = []

        # 1. Moving Average Crossover
        if sma_short > sma_long:
            buy_signals += 1
            reasons.append(f"MA bullish (SMA{self.short_window}={sma_short:.2f} > SMA{self.long_window}={sma_long:.2f})")
        else:
            sell_signals += 1
            reasons.append(f"MA bearish (SMA{self.short_window}={sma_short:.2f} < SMA{self.long_window}={sma_long:.2f})")

        # 2. RSI
        if rsi < self.rsi_oversold:
            buy_signals += 1
            reasons.append(f"RSI oversold ({rsi:.2f} < {self.rsi_oversold})")
        elif rsi > self.rsi_overbought:
            sell_signals += 1
            reasons.append(f"RSI overbought ({rsi:.2f} > {self.rsi_overbought})")

        # 3. MACD
        if hist_val > 0 and macd_val > signal_val:
            buy_signals += 1
            reasons.append(f"MACD bullish (histogram={hist_val:.2f} > 0)")
        elif hist_val < 0 and macd_val < signal_val:
            sell_signals += 1
            reasons.append(f"MACD bearish (histogram={hist_val:.2f} < 0)")

        # 4. Bollinger Bands
        if bb_position < 0.2:
            buy_signals += 1
            reasons.append(f"Price near lower BB (position={bb_position:.2%})")
        elif bb_position > 0.8:
            sell_signals += 1
            reasons.append(f"Price near upper BB (position={bb_position:.2%})")

        # Final decision
        total_signals = buy_signals + sell_signals
        if buy_signals > sell_signals:
            action = 'BUY'
            confidence = buy_signals / 4.0  # 4 indicators max
            reason_text = "BUY: " + "; ".join(reasons)
        elif sell_signals > buy_signals:
            action = 'SELL'
            confidence = sell_signals / 4.0
            reason_text = "SELL: " + "; ".join(reasons)
        else:
            action = 'HOLD'
            confidence = 0.5
            reason_text = "HOLD: Mixed signals - " + "; ".join(reasons)

        return Signal(
            timestamp=timestamp or datetime.now(),
            action=action,
            confidence=confidence,
            price=current_price,
            indicators=indicators,
            reason=reason_text
        )

    def backtest(
        self,
        prices: np.ndarray,
        timestamps: List[datetime] = None,
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Backtest the strategy on historical data.

        Returns:
            Dictionary with performance metrics
        """
        capital = initial_capital
        position = 0  # shares held
        trades = []
        portfolio_values = []

        for i in range(self.long_window, len(prices)):
            current_prices = prices[:i+1]
            signal = self.generate_signal(
                current_prices,
                timestamp=timestamps[i] if timestamps else None
            )

            # Execute trade
            if signal.action == 'BUY' and signal.confidence > 0.6 and capital > 0:
                shares_to_buy = capital / signal.price
                position += shares_to_buy
                capital = 0
                trades.append({
                    'timestamp': signal.timestamp,
                    'action': 'BUY',
                    'price': signal.price,
                    'shares': shares_to_buy,
                    'confidence': signal.confidence,
                    'reason': signal.reason
                })

            elif signal.action == 'SELL' and signal.confidence > 0.6 and position > 0:
                capital = position * signal.price
                trades.append({
                    'timestamp': signal.timestamp,
                    'action': 'SELL',
                    'price': signal.price,
                    'shares': position,
                    'confidence': signal.confidence,
                    'reason': signal.reason
                })
                position = 0

            # Calculate portfolio value
            portfolio_value = capital + (position * prices[i])
            portfolio_values.append(portfolio_value)

        # Calculate metrics
        final_value = capital + (position * prices[-1])
        total_return = (final_value - initial_capital) / initial_capital

        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        max_drawdown = 0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values.tolist()
        }


def example_usage():
    """Example usage of the traditional strategy."""
    # Generate sample data
    np.random.seed(42)
    n_days = 200
    prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
    prices = np.maximum(prices, 50)  # Floor price

    timestamps = [datetime(2024, 1, 1) + np.timedelta64(i, 'D') for i in range(n_days)]

    # Initialize strategy
    strategy = TechnicalStrategy()

    # Get current signal
    signal = strategy.generate_signal(prices)
    print("\n" + "=" * 80)
    print("TRADITIONAL TECHNICAL ANALYSIS SIGNAL")
    print("=" * 80)
    print(f"Action: {signal.action}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Current Price: ${signal.price:.2f}")
    print(f"\nIndicators:")
    for key, value in signal.indicators.items():
        print(f"  {key}: {value:.2f}")
    print(f"\nReasoning: {signal.reason}")

    # Backtest
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    results = strategy.backtest(prices, timestamps)
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")

    return strategy, signal, results


if __name__ == "__main__":
    example_usage()
