"""
Reasoning State Management
Defines the state structure for the multi-agent reasoning workflow.
"""

from typing import TypedDict, List, Optional, Annotated
from operator import add


class ReasoningStep(TypedDict):
    """Single step in the reasoning process."""
    role: str  # "reasoner", "critic", "refiner"
    content: str
    timestamp: Optional[str]
    metadata: Optional[dict]


class ReasoningState(TypedDict):
    """
    State for the reasoning graph workflow.

    Attributes:
        question: The original question/problem
        reasoning_history: List of all reasoning steps
        current_reasoning: Current reasoning output
        critique: Current critique output
        refined_reasoning: Refined reasoning after critique
        final_answer: Final answer after all iterations
        iteration: Current iteration number
        max_iterations: Maximum allowed iterations
        should_continue: Whether to continue refining
    """
    question: str
    reasoning_history: Annotated[List[ReasoningStep], add]
    current_reasoning: str
    critique: str
    refined_reasoning: str
    final_answer: str
    iteration: int
    max_iterations: int
    should_continue: bool
    confidence_score: Optional[float]
