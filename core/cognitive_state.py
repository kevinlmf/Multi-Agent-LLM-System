"""
Enhanced Cognitive State integrating Perception and Memory
Extends the original ReasoningState with perception and memory capabilities
"""

from typing import TypedDict, List, Optional, Annotated, Dict, Any
from operator import add
from datetime import datetime


class ReasoningStep(TypedDict):
    """Single step in the reasoning process."""
    role: str  # "perceiver", "reasoner", "critic", "refiner", "consolidator"
    content: str
    timestamp: Optional[str]
    metadata: Optional[dict]


class PerceptionSnapshot(TypedDict):
    """Snapshot of current world state from perception"""
    entities: List[str]
    relations: List[tuple]
    event_count: int
    graph_density: float
    timestamp: str


class MemorySnapshot(TypedDict):
    """Snapshot of memory state"""
    episodic_count: int
    semantic_count: int
    avg_importance: float
    recent_episodes: List[Dict[str, Any]]
    relevant_concepts: List[str]
    timestamp: str


class CognitiveState(TypedDict):
    """
    Enhanced state for cognitive multi-agent workflow.

    Extends ReasoningState with:
    - Perception: Current world model and observations
    - Memory: Episodic and semantic memory access
    - Learning: Experience consolidation and pattern extraction

    Attributes:
        # Original reasoning fields
        question: The original question/problem
        reasoning_history: List of all reasoning steps
        current_reasoning: Current reasoning output
        critique: Current critique output
        final_answer: Final answer after all iterations
        iteration: Current iteration number
        max_iterations: Maximum allowed iterations
        confidence_score: Quality metric (0.0-1.0)
        should_continue: Whether to continue refining

        # Perception fields
        perception_snapshot: Current world state
        world_context: Textual summary of world model
        raw_observation: Latest observation data

        # Memory fields
        memory_snapshot: Memory system state
        retrieved_memories: List of relevant past experiences
        memory_context: Textual summary of relevant memories
        semantic_knowledge: List of learned concepts and patterns

        # Action & Learning fields
        action_taken: Action decided by the system
        reward: Reward received from action
        experience_stored: Whether experience was stored to memory
    """
    # Original reasoning state
    question: str
    reasoning_history: Annotated[List[ReasoningStep], add]
    current_reasoning: str
    critique: str
    final_answer: str
    iteration: int
    max_iterations: int
    confidence_score: Optional[float]
    should_continue: bool

    # Perception state
    perception_snapshot: Optional[PerceptionSnapshot]
    world_context: str
    raw_observation: Optional[Dict[str, Any]]

    # Memory state
    memory_snapshot: Optional[MemorySnapshot]
    retrieved_memories: List[Dict[str, Any]]
    memory_context: str
    semantic_knowledge: List[str]

    # Action & Learning
    action_taken: Optional[Dict[str, Any]]
    reward: Optional[float]
    experience_stored: bool


def create_initial_cognitive_state(
    question: str,
    max_iterations: int = 5
) -> CognitiveState:
    """Create initial cognitive state with all fields initialized."""
    return CognitiveState(
        # Reasoning
        question=question,
        reasoning_history=[],
        current_reasoning="",
        critique="",
        final_answer="",
        iteration=0,
        max_iterations=max_iterations,
        confidence_score=0.0,
        should_continue=True,

        # Perception
        perception_snapshot=None,
        world_context="",
        raw_observation=None,

        # Memory
        memory_snapshot=None,
        retrieved_memories=[],
        memory_context="",
        semantic_knowledge=[],

        # Action
        action_taken=None,
        reward=None,
        experience_stored=False
    )


def add_perception_to_state(
    state: CognitiveState,
    snapshot: PerceptionSnapshot,
    context: str
) -> Dict[str, Any]:
    """Add perception information to state."""
    return {
        "perception_snapshot": snapshot,
        "world_context": context
    }


def add_memory_to_state(
    state: CognitiveState,
    snapshot: MemorySnapshot,
    retrieved: List[Dict[str, Any]],
    context: str,
    knowledge: List[str]
) -> Dict[str, Any]:
    """Add memory information to state."""
    return {
        "memory_snapshot": snapshot,
        "retrieved_memories": retrieved,
        "memory_context": context,
        "semantic_knowledge": knowledge
    }


def get_full_context(state: CognitiveState) -> str:
    """
    Get complete cognitive context for agent reasoning.
    Combines world state, memories, and reasoning history.
    """
    context_parts = []

    # World perception
    if state.get("world_context"):
        context_parts.append(f"## Current World State\n{state['world_context']}")

    # Memory context
    if state.get("memory_context"):
        context_parts.append(f"## Relevant Past Experiences\n{state['memory_context']}")

    # Semantic knowledge
    if state.get("semantic_knowledge"):
        knowledge_text = "\n".join([f"- {k}" for k in state["semantic_knowledge"][:5]])
        context_parts.append(f"## Learned Knowledge\n{knowledge_text}")

    # Recent reasoning
    history = state.get("reasoning_history", [])
    if history:
        recent_steps = history[-3:]  # Last 3 steps
        history_text = "\n".join([
            f"[{step['role']}]: {step['content'][:200]}..."
            for step in recent_steps
        ])
        context_parts.append(f"## Recent Reasoning Steps\n{history_text}")

    return "\n\n".join(context_parts)


def create_reasoning_step(
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> ReasoningStep:
    """Create a reasoning step with timestamp."""
    return ReasoningStep(
        role=role,
        content=content,
        timestamp=datetime.now().isoformat(),
        metadata=metadata or {}
    )
