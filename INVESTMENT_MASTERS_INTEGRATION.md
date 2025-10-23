# Investment Masters Integration

## Overview

The cognitive trading agent now integrates insights from legendary investors to provide multi-perspective analysis for trading decisions.

## Investment Masters Included

### 1. Warren Buffett - Value Investing
- **Philosophy**: Buy wonderful businesses at fair prices, hold forever
- **Focus**: Intrinsic value, margin of safety, competitive moats
- **Time Horizon**: Years to decades

### 2. George Soros - Macro Trading & Reflexivity
- **Philosophy**: Markets are always biased, identify boom-bust cycles
- **Focus**: Macroeconomic imbalances, policy errors, sentiment extremes
- **Time Horizon**: Weeks to months

### 3. Ray Dalio - All-Weather Risk Parity
- **Philosophy**: Build balanced portfolios for any economic environment
- **Focus**: Risk diversification, correlation analysis, tail risks
- **Time Horizon**: Years with tactical adjustments

## How It Works

### Enhanced Cognitive Loop

```
1. Market Observation
   ↓
2. Perception Processing
   ↓
3. Memory Retrieval
   ↓
4. Master Consultation ⭐ NEW
   - Buffett's perspective
   - Soros's perspective
   - Dalio's perspective
   ↓
5. Cognitive Reasoning (synthesizing all inputs)
   ↓
6. Final Decision
```

### Technical Implementation

The system uses the `get_master_opinions()` function which:
1. Takes the current market scenario
2. Consults each investment master via their persona
3. Collects their recommendations
4. Formats and presents their insights
5. Feeds into the cognitive reasoning process

## Usage

### Basic Usage

```python
from examples.cognitive_trading_agent import run_trading_scenario

# With investment masters (default)
run_trading_scenario(enable_masters=True)

# Without investment masters
run_trading_scenario(enable_masters=False)
```

### Interactive Mode

```bash
python examples/cognitive_trading_agent.py
```

Then select option 1: "Full Trading Scenario (5-day campaign with Investment Masters)"

### Expected Output

```
🎩 Consulting investment masters...
  ✓ Warren Buffett (Value Investing)
  ✓ George Soros (Macro Trading & Reflexivity)
  ✓ Ray Dalio (All-Weather Risk Parity)

================================================================================
💭 INVESTMENT MASTERS' PERSPECTIVES
================================================================================

📊 Warren Buffett (Value Investing):
[Buffett's analysis based on value investing principles...]

📊 George Soros (Macro Trading & Reflexivity):
[Soros's analysis based on macro trends and reflexivity...]

📊 Ray Dalio (All-Weather Risk Parity):
[Dalio's analysis based on risk parity approach...]

================================================================================

🎯 FINAL DECISION (Incorporating Masters' Wisdom)
================================================================================
[Synthesized recommendation combining cognitive analysis and master insights]
```

## Benefits

1. **Multi-Perspective Analysis**: Combines different investment philosophies
2. **Wisdom Integration**: Leverages decades of investment expertise
3. **Flexible Control**: Can be enabled or disabled as needed
4. **Transparent Process**: Shows each master's reasoning clearly

## Customization

### Adding More Masters

To include additional masters (Jim Simons or Cathie Wood):

```python
# In get_master_opinions() function
master_names = ['buffett', 'soros', 'dalio', 'simons', 'wood']
```

Available masters:
- `'simons'` - Jim Simons (Quantitative Renaissance)
- `'wood'` - Cathie Wood (Disruptive Innovation)

### Adjusting Reasoning Depth

Modify the `max_iterations` parameter in the master consultation:

```python
graph = ReasoningGraph(max_iterations=2)  # Faster
graph = ReasoningGraph(max_iterations=3)  # More thorough
```

## Requirements

- OpenAI API key must be set: `export OPENAI_API_KEY='your-key'`
- The `strategy_comparison` module must be available
- Sufficient API quota for multiple LLM calls

## Performance Considerations

- Each master consultation adds ~2-3 seconds
- Total additional time: ~6-9 seconds for 3 masters
- Can be disabled if speed is critical

## Technical Details

### Core Functions

**`get_master_opinions(scenario_data: dict) -> str`**
- Input: Dictionary with 'day', 'observation', 'question'
- Output: Formatted string with all master opinions
- Handles errors gracefully (returns notification if masters unavailable)

**`run_trading_scenario(enable_masters: bool = True)`**
- Parameter: `enable_masters` controls master consultation
- Default: `True` (masters enabled)
- Backward compatible: Can disable for original behavior

### Dependencies

```python
from strategy_comparison.llm_crypto.master_strategies import PERSONAS
from core import ReasoningGraph
```

## Troubleshooting

### Masters Not Available

If you see "Investment Masters' Opinions: Module not available":
- Check that `strategy_comparison` directory exists
- Verify `master_strategies.py` is in the correct location

### API Errors

If consultations fail:
- Verify OPENAI_API_KEY is set
- Check API quota and rate limits
- Review error messages in console

## Example Scenario

```python
scenario = {
    'day': 2,
    'observation': """
        SPY: $455.20 (+1.2%)
        VIX: 14.8 (-0.4%)
        Position: Long from $450, +$5.20 profit
    """,
    'question': "Should we hold, add, or take profits?"
}

opinions = get_master_opinions(scenario)
# Returns formatted master perspectives
```

## Future Enhancements

- [ ] Consensus scoring across masters
- [ ] Position sizing based on master agreement
- [ ] Historical accuracy tracking per master
- [ ] Dynamic master selection based on market regime

---

**Happy Trading with the Masters! 🎩📈**
