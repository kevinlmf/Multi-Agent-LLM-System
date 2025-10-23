"""
Multi-Agent Reasoning System
A production-grade cognitive system with perception, memory, and multi-agent reasoning.

Provides both:
1. Original reasoning-only system (backward compatible)
2. Enhanced cognitive system with perception and memory
"""

# Original system (backward compatible)
from .agents import ReasonerAgent, CriticAgent, RefinerAgent
from .graph import ReasoningGraph
from .state import ReasoningState, ReasoningStep

# Enhanced cognitive system
from .cognitive_state import (
    CognitiveState,
    PerceptionSnapshot,
    MemorySnapshot,
    create_initial_cognitive_state,
    get_full_context,
    create_reasoning_step
)

from .cognitive_agents import (
    PerceiverAgent,
    CognitiveReasonerAgent,
    CognitiveCriticAgent,
    CognitiveRefinerAgent,
    ConsolidatorAgent
)

from .cognitive_graph import CognitiveReasoningGraph

__version__ = "2.0.0"
__all__ = [
    # Original
    "ReasonerAgent",
    "CriticAgent",
    "RefinerAgent",
    "ReasoningGraph",
    "ReasoningState",
    "ReasoningStep",

    # Cognitive
    "CognitiveState",
    "PerceptionSnapshot",
    "MemorySnapshot",
    "create_initial_cognitive_state",
    "get_full_context",
    "create_reasoning_step",
    "PerceiverAgent",
    "CognitiveReasonerAgent",
    "CognitiveCriticAgent",
    "CognitiveRefinerAgent",
    "ConsolidatorAgent",
    "CognitiveReasoningGraph",
]
