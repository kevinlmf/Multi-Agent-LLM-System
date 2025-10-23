# Cognitive Multi-Agent System

A production-grade **Cognitive Framework** that integrates **Perception**, **Memory**, and **Multi-Agent Reasoning** for intelligent decision-making across multiple domains.

##  System Overview

This is an **extensible multi-agent reasoning framework** designed to be applied to various real-world scenarios. The core engine provides:

### Current Status
- ✅ **Core Engine**: Fully implemented (Reasoner → Critic → Refiner loop)
- ✅ **Graph-Based Memory**: LangGraph state management + dual memory system (episodic + semantic)
- ✅ **Applied Scenario**: Quantitative Trading Strategy Analysis with Investment Masters Integration 
-  **Future Scenarios**: Smart decision assistant, sentiment monitoring (see Roadmap)

## Cognitive Loop Architecture

The core reasoning engine implements a complete cognitive cycle that can be adapted to any domain:

| Phase | Module | Technical Implementation |
|------|---------|---------|
| 1. Receive | Input Adapter | Question Input / State Initialization |
| 2. Record | **Memory (Graph-Based)** | **LangGraph State** / Reasoning History |
| 3. Understand | Reasoning Core | **Reasoner Agent** (LLM-powered) |
| 4. Decide | Controller | **Critic Agent** (Confidence-based routing) |
| 5. Execute | Tool Layer | **Refiner Agent** (Answer synthesis) |
| 6. Reflect | Feedback Module | Multi-iteration loop with critique |

**Core Flow:** Input → Reasoner → Critic → (iterate if needed) → Refiner → Output

## Installation

```bash
git clone https://github.com/kevinlmf/Multi-Agent-LLM-System
cd Multi-Agent-LLM-System
pip install -r requirements.txt
export OPENAI_API_KEY='your-api-key-here'
```


## 📂 Project Structure

```bash
Cognitive_Multi_Agent_System/
├── core/                    # Core reasoning engine
│   ├── cognitive_state.py   # Manages agent memory and contextual state (LangGraph)
│   ├── cognitive_agents.py  # Defines Reasoner, Critic, Refiner agents
│   └── cognitive_graph.py   # LangGraph controller that orchestrates the cognitive loop
│
├── perception/              # Perception layer (entity extraction, world model)
├── memory/                  # Memory layer (episodic + semantic memory graphs)
│
├── examples/                # Demonstrations and reasoning experiments
│   └── cognitive_trading_agent.py  # Trading demo with Investment Masters integration
│
└── tests/                   # Unit tests for reasoning and workflow integrity
```

## Scenario 1 — Financial Strategy Analysis (Current)

A live demo showing how the cognitive loop improves multi-perspective reasoning in quantitative finance, **enhanced with Investment Masters consultation** 🎩.

### Enhanced Features
- **Graph-Based Memory**: LangGraph manages reasoning state and history
- **Investment Masters Integration**: Consults legendary investors (Buffett, Soros, Dalio) for multi-perspective analysis
- **Cognitive Loop**: Perception → Memory → Master Consultation → Reasoning → Decision

### Run Demo

```bash
# Launch interactive demo
python examples/cognitive_trading_agent.py
```

### What You Get

**Multi-Perspective Analysis:**
```
Observation → Perception → Memory → Master Consultation → Reasoning → Decision
                                           ↓
                            Buffett (Value Investing)
                            Soros (Macro Trading)
                            Dalio (Risk Parity)
```

**Example Output:**
```
Consulting investment masters...
  ✓ Warren Buffett (Value Investing)
  ✓ George Soros (Macro Trading & Reflexivity)
  ✓ Ray Dalio (All-Weather Risk Parity)

 FINAL DECISION (Incorporating Masters' Wisdom)
[Recommendation synthesizing cognitive analysis + master opinions]
```

See [INVESTMENT_MASTERS_INTEGRATION.md](INVESTMENT_MASTERS_INTEGRATION.md) for detailed documentation.

##  Roadmap

### Phase 1: Core Reasoning System + Mathematical Foundations
This phase aims at bridging **intelligent reasoning systems** with **formal quantitative modeling**, ensuring interpretability, stability, and verifiable improvement over time.

#### Core Agent Framework — Mathematical Foundations

| **Agent** | **Role** | **Theoretical Focus** | **Goal** |
|------------|-----------|-----------------------|-----------|
| **Reasoner Agent**<br>Representation & Probabilistic Understanding | Learns structured representations and performs hypothesis generation (LLM-powered). | • **Representation Learning:** Encode reasoning states into latent embeddings capturing semantic and structural dependencies.<br>• **Variational Inference:** Approximate latent reasoning dynamics to infer hidden cognitive variables. | Build reasoning as *representation learning + variational inference*, enabling interpretable latent cognition. |
| **Critic Agent**<br>Decision & Evaluation Framework | Evaluates reasoning confidence, selects optimal branches, and refines decision flow. | • **Decision Theory:** Choose actions maximizing expected utility.<br>• **Reinforcement Learning:** Model confidence and feedback as a value function guiding adaptive control. | Develop a confidence-driven controller grounded in Bayesian and decision-theoretic principles. |
| **Refiner Agent**<br>Synthesis through Optimization | Integrates multi-branch outputs into coherent, high-fidelity answers. | • **Optimal Transport:** Align and merge reasoning distributions via Wasserstein barycenters.<br>• **Multi-Objective Optimization:** Balance accuracy and stability during synthesis. | Define refinement as an optimization process achieving semantic and structural coherence. |

### Phase 2: Applied Scenarios

#### Scenario 2: Intelligent Decision Assistant
**Status**: Planned
- **Example Use Cases**:
  - "Should I take the subway or bike to work today considering weather and traffic?"
  - "Recommend the best time to leave for my 3 PM meeting downtown"
  - "Is today a good day for outdoor activities?"

#### Scenario 3: Public Sentiment Monitor
**Status**:  Planned
- **Example Use Cases**:
  - "Analyze how people perceive **Artificial Intelligence (AI)**."
  - "Understand public opinion on the **Federal Reserve's interest rate cuts**."
  - "Explore how people think about **LeBron James** — especially his decisions."

---

**Better reasoning, brighter tomorrow** ✨

