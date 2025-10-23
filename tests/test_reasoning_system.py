"""
Unit tests for the Multi-Agent Reasoning System.
"""

import pytest
import os
from core import ReasoningGraph, ReasoningState
from core.agents import ReasonerAgent, CriticAgent, RefinerAgent


class TestAgents:
    """Test individual agent functionality."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        return {
            "question": "What is 2+2?",
            "reasoning_history": [],
            "current_reasoning": "",
            "critique": "",
            "refined_reasoning": "",
            "final_answer": "",
            "iteration": 0,
            "max_iterations": 3,
            "should_continue": True,
            "confidence_score": None
        }

    def test_reasoner_initialization(self):
        """Test reasoner agent initialization."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        reasoner = ReasonerAgent()
        assert reasoner.role == "reasoner"
        assert reasoner.llm is not None

    def test_critic_initialization(self):
        """Test critic agent initialization."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        critic = CriticAgent()
        assert critic.role == "critic"
        assert critic.llm is not None

    def test_refiner_initialization(self):
        """Test refiner agent initialization."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        refiner = RefinerAgent()
        assert refiner.role == "refiner"
        assert refiner.llm is not None

    def test_confidence_parsing(self):
        """Test confidence score parsing."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        critic = CriticAgent()

        # Test valid parsing
        critique_text = "CONFIDENCE: 0.85\nCRITIQUE: This is good reasoning."
        score = critic._parse_confidence(critique_text)
        assert score == 0.85

        # Test default fallback
        critique_text = "No confidence mentioned"
        score = critic._parse_confidence(critique_text)
        assert score == 0.8  # Default


class TestReasoningGraph:
    """Test the reasoning graph workflow."""

    def test_graph_initialization(self):
        """Test graph initialization."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        graph = ReasoningGraph(max_iterations=3)
        assert graph.reasoner is not None
        assert graph.critic is not None
        assert graph.refiner is not None
        assert graph.max_iterations == 3

    def test_graph_structure(self):
        """Test that graph has correct structure."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        graph = ReasoningGraph()
        assert graph.graph is not None
        assert graph.app is not None


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.slow
    def test_simple_reasoning(self):
        """Test complete reasoning workflow with simple question."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        graph = ReasoningGraph(max_iterations=2)
        question = "What is 5 + 3?"

        result = graph.reason(question)

        # Verify result structure
        assert "final_answer" in result
        assert "reasoning_history" in result
        assert "confidence_score" in result
        assert "iterations" in result

        # Verify result content
        assert result["final_answer"] is not None
        assert len(result["reasoning_history"]) > 0
        assert result["confidence_score"] >= 0.0
        assert result["confidence_score"] <= 1.0
        assert result["iterations"] >= 1

    @pytest.mark.slow
    def test_complex_reasoning(self):
        """Test reasoning with a more complex question."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        graph = ReasoningGraph(max_iterations=3)
        question = """
        If a train travels 60 mph for 2 hours, then 40 mph for 3 hours,
        what is the average speed for the entire journey?
        """

        result = graph.reason(question)

        assert result["final_answer"] is not None
        assert len(result["reasoning_history"]) >= 3  # At least reasoner, critic, refiner
        assert result["iterations"] >= 1

    @pytest.mark.slow
    def test_iteration_limit(self):
        """Test that iteration limit is respected."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        graph = ReasoningGraph(max_iterations=2)
        question = "Explain quantum entanglement in detail."

        result = graph.reason(question)

        # Should not exceed max iterations
        assert result["iterations"] <= 2


class TestStateManagement:
    """Test state management."""

    def test_reasoning_step_creation(self):
        """Test creation of reasoning steps."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        agent = ReasonerAgent()
        step = agent._create_reasoning_step(
            content="Test reasoning",
            metadata={"iteration": 1}
        )

        assert step["role"] == "reasoner"
        assert step["content"] == "Test reasoning"
        assert "timestamp" in step
        assert step["metadata"]["iteration"] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_question(self):
        """Test handling of empty question."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        graph = ReasoningGraph()

        # Should handle gracefully or raise appropriate error
        try:
            result = graph.reason("")
            assert result is not None
        except Exception as e:
            assert isinstance(e, (ValueError, TypeError))

    def test_very_long_question(self):
        """Test handling of very long questions."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        graph = ReasoningGraph(max_iterations=1)
        question = "What is the answer? " * 1000  # Very long question

        result = graph.reason(question)
        assert result is not None

    def test_max_iterations_zero(self):
        """Test behavior with max_iterations=1."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        graph = ReasoningGraph()
        result = graph.reason("What is 2+2?", max_iterations=1)

        assert result["iterations"] == 1


# Run with: pytest tests/test_reasoning_system.py -v
# Run slow tests: pytest tests/test_reasoning_system.py -v --runslow
# Skip slow tests: pytest tests/test_reasoning_system.py -v -m "not slow"

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
