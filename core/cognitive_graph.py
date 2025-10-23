"""
Cognitive Multi-Agent Graph Workflow
Orchestrates perception, memory, reasoning, and learning using LangGraph
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from .cognitive_state import CognitiveState, create_initial_cognitive_state
from .cognitive_agents import (
    PerceiverAgent,
    CognitiveReasonerAgent,
    CognitiveCriticAgent,
    CognitiveRefinerAgent,
    ConsolidatorAgent
)

# Import perception and memory systems
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from perception.perception_main import PerceptionEngine
    from memory.memory_main import MemoryEngine
    PERCEPTION_AVAILABLE = True
except ImportError:
    PERCEPTION_AVAILABLE = False
    print("Warning: Perception/Memory modules not available. Running in reasoning-only mode.")


class CognitiveReasoningGraph:
    """
    Cognitive Multi-Agent Workflow with Perception and Memory.

    Workflow:
    1. Perceive: Process observations, update world model
    2. Retrieve: Query relevant memories
    3. Reason: Generate reasoning with full context
    4. Critique: Evaluate reasoning quality
    5. Decide: Continue iterating or finalize
    6. Refine: Produce final answer
    7. Store: Save experience to memory
    8. Consolidate: Extract knowledge (periodic)
    """

    def __init__(
        self,
        max_iterations: int = 5,
        model_name: str = "gpt-4o-mini",
        enable_perception: bool = True,
        enable_memory: bool = True
    ):
        self.max_iterations = max_iterations
        self.model_name = model_name
        self.enable_perception = enable_perception and PERCEPTION_AVAILABLE
        self.enable_memory = enable_memory and PERCEPTION_AVAILABLE

        # Initialize agents
        self.perceiver = PerceiverAgent(model_name=model_name)
        self.reasoner = CognitiveReasonerAgent(model_name=model_name)
        self.critic = CognitiveCriticAgent(model_name=model_name)
        self.refiner = CognitiveRefinerAgent(model_name=model_name)
        self.consolidator = ConsolidatorAgent(model_name=model_name)

        # Initialize perception and memory engines
        if self.enable_perception:
            self.perception_engine = PerceptionEngine()
        else:
            self.perception_engine = None

        if self.enable_memory:
            self.memory_engine = MemoryEngine()
        else:
            self.memory_engine = None

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the cognitive workflow graph."""
        workflow = StateGraph(CognitiveState)

        # Add nodes
        if self.enable_perception:
            workflow.add_node("perceive", self._perceive_node)
        if self.enable_memory:
            workflow.add_node("retrieve", self._retrieve_node)

        workflow.add_node("reason", self._reason_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("refine", self._refine_node)

        if self.enable_memory:
            workflow.add_node("store", self._store_node)

        # Define edges
        if self.enable_perception:
            workflow.set_entry_point("perceive")
            if self.enable_memory:
                workflow.add_edge("perceive", "retrieve")
                workflow.add_edge("retrieve", "reason")
            else:
                workflow.add_edge("perceive", "reason")
        elif self.enable_memory:
            workflow.set_entry_point("retrieve")
            workflow.add_edge("retrieve", "reason")
        else:
            workflow.set_entry_point("reason")

        workflow.add_edge("reason", "critique")

        # Conditional edge: continue or finalize
        workflow.add_conditional_edges(
            "critique",
            self._should_continue,
            {
                "continue": "reason",  # Loop back to reasoner
                "finalize": "refine"   # Move to refiner
            }
        )

        if self.enable_memory:
            workflow.add_edge("refine", "store")
            workflow.add_edge("store", END)
        else:
            workflow.add_edge("refine", END)

        return workflow.compile()

    def _perceive_node(self, state: CognitiveState) -> Dict[str, Any]:
        """Process observation and update world model."""
        raw_obs = state.get("raw_observation")

        if raw_obs and self.perception_engine:
            obs_text = raw_obs.get("text", "")

            # Process perception
            self.perception_engine.perceive_text(obs_text)

            # Get world state summary
            world_stats = self.perception_engine.get_world_statistics()
            entities = list(self.perception_engine.world_graph.graph.nodes())[:10]
            relations = list(self.perception_engine.world_graph.graph.edges())[:10]

            # Create perception snapshot
            from .cognitive_state import PerceptionSnapshot
            snapshot = PerceptionSnapshot(
                entities=entities,
                relations=relations,
                event_count=world_stats.get("event_count", 0),
                graph_density=world_stats.get("graph_density", 0.0),
                timestamp=raw_obs.get("timestamp", "")
            )

            # Use perceiver agent to generate context
            perceiver_result = self.perceiver.perceive(state, obs_text)

            return {
                **perceiver_result,
                "perception_snapshot": snapshot
            }

        return {}

    def _retrieve_node(self, state: CognitiveState) -> Dict[str, Any]:
        """Retrieve relevant memories."""
        if not self.memory_engine:
            return {}

        question = state["question"]

        # Retrieve relevant memories
        memories = self.memory_engine.retrieve(
            query=question,
            strategy="hybrid",
            top_k=5
        )

        # Get memory statistics
        mem_stats = self.memory_engine.get_statistics()

        # Get semantic knowledge
        semantic_concepts = []
        if hasattr(self.memory_engine, 'semantic_memory'):
            concepts = list(self.memory_engine.semantic_memory.concepts.values())[:5]
            semantic_concepts = [f"{c.name}: {c.concept_type}" for c in concepts]

        # Format memory context
        memory_context = self._format_memory_context(memories)

        # Create memory snapshot
        from .cognitive_state import MemorySnapshot
        snapshot = MemorySnapshot(
            episodic_count=mem_stats.get("episodic", {}).get("total_episodes", 0),
            semantic_count=mem_stats.get("semantic", {}).get("total_concepts", 0),
            avg_importance=mem_stats.get("episodic", {}).get("avg_importance", 0.0),
            recent_episodes=[],
            relevant_concepts=semantic_concepts,
            timestamp=""
        )

        return {
            "memory_snapshot": snapshot,
            "retrieved_memories": memories,
            "memory_context": memory_context,
            "semantic_knowledge": semantic_concepts
        }

    def _reason_node(self, state: CognitiveState) -> Dict[str, Any]:
        """Generate reasoning with cognitive context."""
        result = self.reasoner.reason(state)

        # Increment iteration
        new_iteration = state["iteration"] + 1

        return {
            **result,
            "iteration": new_iteration
        }

    def _critique_node(self, state: CognitiveState) -> Dict[str, Any]:
        """Evaluate reasoning quality."""
        return self.critic.critique(state)

    def _should_continue(self, state: CognitiveState) -> str:
        """Determine whether to continue iterating."""
        if state.get("should_continue", False):
            return "continue"
        return "finalize"

    def _refine_node(self, state: CognitiveState) -> Dict[str, Any]:
        """Produce final answer."""
        return self.refiner.refine(state)

    def _store_node(self, state: CognitiveState) -> Dict[str, Any]:
        """Store experience to memory."""
        if not self.memory_engine:
            return {"experience_stored": False}

        # Create experience record
        from .cognitive_state import create_reasoning_step

        # Prepare world snapshot
        world_snapshot = {
            "world_graph": self.perception_engine.world_graph.to_dict() if hasattr(self.perception_engine, 'world_graph') else {},
            "timestamp": state.get("timestamp", "")
        }

        # Prepare perception result
        perception_result = {
            "summary": state.get("question", ""),
            "entities": [],
            "relations": [],
            "keywords": []
        }

        action = {
            "type": "reasoning",
            "decision": state.get("final_answer", ""),
            "confidence": state.get("confidence_score", 0.0)
        }

        # Store to memory
        self.memory_engine.store_experience(
            world_snapshot=world_snapshot,
            perception_result=perception_result,
            action=action,
            reward=state.get("reward", 0.0)
        )

        # Periodic consolidation
        mem_stats = self.memory_engine.get_statistics()
        episodic_count = mem_stats.get("episodic", {}).get("total_episodes", 0)

        if episodic_count > 0 and episodic_count % 10 == 0:
            # Consolidate memories
            self.memory_engine.consolidate_memories()

            # Extract knowledge using consolidator agent
            recent_memories = self.memory_engine.retrieve(
                query="recent experiences",
                strategy="recent",
                top_k=10
            )
            knowledge = self.consolidator.consolidate(recent_memories)

            return {
                "experience_stored": True,
                "semantic_knowledge": knowledge
            }

        return {"experience_stored": True}

    def _format_memory_context(self, memories: list) -> str:
        """Format retrieved memories into context string."""
        if not memories:
            return "No relevant past experiences found."

        context_parts = []
        for i, mem in enumerate(memories[:5], 1):
            mem_text = mem.get("content", {})
            if isinstance(mem_text, dict):
                obs = mem_text.get("observation", {})
                action = mem_text.get("action", {})
                reward = mem_text.get("reward", "N/A")
                context_parts.append(
                    f"{i}. Observation: {obs}\n   Action: {action}\n   Outcome: {reward}"
                )
            else:
                context_parts.append(f"{i}. {str(mem_text)[:200]}")

        return "\n\n".join(context_parts)

    def reason(
        self,
        question: str,
        observation: Optional[str] = None,
        reward: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute cognitive reasoning workflow.

        Args:
            question: The question or task to reason about
            observation: Optional observation to perceive
            reward: Optional reward signal from previous action

        Returns:
            Dict with final_answer, reasoning_history, confidence_score, etc.
        """
        # Create initial state
        initial_state = create_initial_cognitive_state(
            question=question,
            max_iterations=self.max_iterations
        )

        # Add observation if provided
        if observation:
            initial_state["raw_observation"] = {
                "text": observation,
                "timestamp": ""
            }

        # Add reward if provided
        if reward is not None:
            initial_state["reward"] = reward

        # Execute workflow
        final_state = self.graph.invoke(initial_state)

        # Extract key results
        return {
            "question": final_state["question"],
            "final_answer": final_state["final_answer"],
            "reasoning_history": final_state["reasoning_history"],
            "confidence_score": final_state.get("confidence_score"),
            "iterations": final_state["iteration"],
            "perception_snapshot": final_state.get("perception_snapshot"),
            "memory_snapshot": final_state.get("memory_snapshot"),
            "semantic_knowledge": final_state.get("semantic_knowledge", []),
            "experience_stored": final_state.get("experience_stored", False)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "max_iterations": self.max_iterations,
            "perception_enabled": self.enable_perception,
            "memory_enabled": self.enable_memory
        }

        if self.perception_engine:
            stats["perception"] = self.perception_engine.get_world_statistics()

        if self.memory_engine:
            stats["memory"] = self.memory_engine.get_statistics()

        return stats
