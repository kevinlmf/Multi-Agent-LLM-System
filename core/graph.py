"""
LangGraph Reasoning Workflow
Implements the multi-agent reasoning graph with state management.
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from .state import ReasoningState
from .agents import ReasonerAgent, CriticAgent, RefinerAgent


class ReasoningGraph:
    """
    Multi-Agent Reasoning Graph using LangGraph.

    Workflow:
    1. START → Reasoner (generates initial reasoning)
    2. Reasoner → Critic (evaluates reasoning)
    3. Critic → Decision:
       - If confidence >= 0.9 OR max_iterations reached → Refiner → END
       - Else → Reasoner (refine based on critique) → repeat
    4. Refiner → END (produces final answer)
    """

    def __init__(
        self,
        reasoner_model: str = "gpt-3.5-turbo",
        critic_model: str = "gpt-3.5-turbo",
        refiner_model: str = "gpt-3.5-turbo",
        max_iterations: int = 3
    ):
        """
        Initialize the reasoning graph.

        Args:
            reasoner_model: Model name for reasoner agent
            critic_model: Model name for critic agent
            refiner_model: Model name for refiner agent
            max_iterations: Maximum number of reasoning iterations
        """
        self.reasoner = ReasonerAgent(model_name=reasoner_model, temperature=0.7)
        self.critic = CriticAgent(model_name=critic_model, temperature=0.3)
        self.refiner = RefinerAgent(model_name=refiner_model, temperature=0.5)
        self.max_iterations = max_iterations

        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph workflow."""
        # Create state graph
        workflow = StateGraph(ReasoningState)

        # Add nodes
        workflow.add_node("reasoner", self._reasoner_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("refiner", self._refiner_node)

        # Define edges
        workflow.set_entry_point("reasoner")
        workflow.add_edge("reasoner", "critic")

        # Conditional edge: continue refining or finalize
        workflow.add_conditional_edges(
            "critic",
            self._should_continue,
            {
                "continue": "reasoner",
                "finalize": "refiner"
            }
        )

        workflow.add_edge("refiner", END)

        return workflow

    def _reasoner_node(self, state: ReasoningState) -> Dict[str, Any]:
        """Reasoner node: generate or refine reasoning."""
        print(f"\n🧠 [Iteration {state['iteration'] + 1}] Reasoner thinking...")
        result = self.reasoner.reason(state)
        result["iteration"] = state["iteration"] + 1
        return result

    def _critic_node(self, state: ReasoningState) -> Dict[str, Any]:
        """Critic node: evaluate reasoning."""
        print(f"🔍 [Iteration {state['iteration']}] Critic evaluating...")
        result = self.critic.critique(state)
        print(f"   Confidence: {result['confidence_score']:.2f}")
        return result

    def _refiner_node(self, state: ReasoningState) -> Dict[str, Any]:
        """Refiner node: produce final answer."""
        print(f"✨ Refiner synthesizing final answer...")
        return self.refiner.refine(state)

    def _should_continue(self, state: ReasoningState) -> Literal["continue", "finalize"]:
        """Decide whether to continue refining or finalize."""
        if state["should_continue"]:
            print(f"   → Continuing to next iteration\n")
            return "continue"
        else:
            reason = "High confidence" if state.get("confidence_score", 0) >= 0.9 else "Max iterations"
            print(f"   → Finalizing ({reason})\n")
            return "finalize"

    def reason(self, question: str, max_iterations: int = None) -> Dict[str, Any]:
        """
        Execute the reasoning workflow.

        Args:
            question: The question/problem to reason about
            max_iterations: Override default max iterations

        Returns:
            Dictionary containing:
                - final_answer: The final answer
                - reasoning_history: List of all reasoning steps
                - confidence_score: Final confidence score
                - iterations: Number of iterations performed
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        # Initialize state
        initial_state: ReasoningState = {
            "question": question,
            "reasoning_history": [],
            "current_reasoning": "",
            "critique": "",
            "refined_reasoning": "",
            "final_answer": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "should_continue": True,
            "confidence_score": None
        }

        print("=" * 80)
        print(f"🚀 Starting Multi-Agent Reasoning")
        print(f"📝 Question: {question}")
        print(f"🔄 Max iterations: {max_iterations}")
        print("=" * 80)

        # Run the graph
        final_state = self.app.invoke(initial_state)

        print("=" * 80)
        print("✅ Reasoning Complete!")
        print(f"📊 Total iterations: {final_state['iteration']}")
        print(f"🎯 Final confidence: {final_state.get('confidence_score', 0):.2f}")
        print("=" * 80)

        return {
            "final_answer": final_state["final_answer"],
            "reasoning_history": final_state["reasoning_history"],
            "confidence_score": final_state.get("confidence_score", 0.0),
            "iterations": final_state["iteration"],
            "question": question
        }

    def visualize(self, output_path: str = "reasoning_graph.png"):
        """
        Visualize the reasoning graph structure.

        Args:
            output_path: Path to save the visualization
        """
        try:
            from IPython.display import Image, display
            # Generate graph visualization
            graph_image = self.app.get_graph().draw_mermaid_png()

            with open(output_path, "wb") as f:
                f.write(graph_image)

            print(f"✅ Graph visualization saved to: {output_path}")

            # Try to display if in Jupyter
            try:
                display(Image(graph_image))
            except:
                pass

        except Exception as e:
            print(f"⚠️  Could not generate visualization: {e}")
            print("   Install required packages: pip install pygraphviz")
