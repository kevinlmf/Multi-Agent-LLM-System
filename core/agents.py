"""
Agent Implementations
Three specialized agents for multi-round reasoning: Reasoner, Critic, Refiner.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime
import os


class BaseAgent:
    """Base class for all reasoning agents."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        self.role = "base"

    def _create_reasoning_step(self, content: str, metadata: dict = None) -> Dict[str, Any]:
        """Create a reasoning step with timestamp."""
        return {
            "role": self.role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }


class ReasonerAgent(BaseAgent):
    """
    Reasoner Agent: Generates initial reasoning or refines based on critique.

    Responsibilities:
    - Generate step-by-step reasoning for the problem
    - Consider previous critique (if any) to improve reasoning
    - Provide clear logical flow
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.role = "reasoner"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Reasoning Agent. Your task is to think through problems step-by-step with clarity and precision.

When reasoning:
1. Break down the problem into manageable steps
2. Show your thought process explicitly
3. Consider edge cases and alternative approaches
4. Be rigorous and logical

If you receive critique, carefully address each point and improve your reasoning accordingly."""),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def reason(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reasoning for the given question."""
        question = state["question"]
        iteration = state.get("iteration", 0)
        critique = state.get("critique", "")

        if iteration == 0:
            # First iteration: initial reasoning
            input_text = f"""Question: {question}

Please provide step-by-step reasoning to solve this problem."""
        else:
            # Subsequent iterations: refine based on critique
            previous_reasoning = state.get("current_reasoning", "")
            input_text = f"""Question: {question}

Your previous reasoning:
{previous_reasoning}

Critique received:
{critique}

Please refine your reasoning by addressing the critique while maintaining logical clarity."""

        response = self.chain.invoke({"input": input_text})
        reasoning = response.content

        # Create reasoning step
        reasoning_step = self._create_reasoning_step(
            content=reasoning,
            metadata={"iteration": iteration, "type": "initial" if iteration == 0 else "refined"}
        )

        return {
            "current_reasoning": reasoning,
            "reasoning_history": [reasoning_step]
        }


class CriticAgent(BaseAgent):
    """
    Critic Agent: Evaluates reasoning quality and provides constructive feedback.

    Responsibilities:
    - Identify logical flaws or gaps
    - Check for completeness and correctness
    - Provide specific, actionable feedback
    - Assign confidence score
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
        super().__init__(model_name, temperature)
        self.role = "critic"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Critical Thinking Expert. Your role is to evaluate reasoning quality rigorously.

Evaluation criteria:
1. **Logical Consistency**: Are there any logical flaws or contradictions?
2. **Completeness**: Are all aspects of the problem addressed?
3. **Correctness**: Is the reasoning factually accurate?
4. **Clarity**: Is the reasoning clear and easy to follow?

Provide:
- Specific issues found (if any)
- Suggestions for improvement
- A confidence score (0.0-1.0) where:
  - 0.9-1.0: Excellent, ready to finalize
  - 0.7-0.89: Good, minor improvements needed
  - Below 0.7: Significant issues, needs revision

Format your response as:
CONFIDENCE: [score]
CRITIQUE: [detailed feedback]"""),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def critique(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the current reasoning."""
        question = state["question"]
        current_reasoning = state["current_reasoning"]

        input_text = f"""Question: {question}

Reasoning to evaluate:
{current_reasoning}

Please evaluate this reasoning thoroughly."""

        response = self.chain.invoke({"input": input_text})
        critique_text = response.content

        # Parse confidence score
        confidence_score = self._parse_confidence(critique_text)

        # Create critique step
        critique_step = self._create_reasoning_step(
            content=critique_text,
            metadata={"confidence_score": confidence_score}
        )

        # Decide whether to continue based on confidence
        should_continue = confidence_score < 0.9 and state["iteration"] < state["max_iterations"] - 1

        return {
            "critique": critique_text,
            "confidence_score": confidence_score,
            "should_continue": should_continue,
            "reasoning_history": [critique_step]
        }

    def _parse_confidence(self, critique_text: str) -> float:
        """Extract confidence score from critique."""
        try:
            for line in critique_text.split("\n"):
                if line.startswith("CONFIDENCE:"):
                    score_str = line.split(":")[1].strip()
                    return float(score_str)
        except:
            pass
        return 0.8  # Default if parsing fails


class RefinerAgent(BaseAgent):
    """
    Refiner Agent: Produces final polished output.

    Responsibilities:
    - Synthesize reasoning and critique
    - Produce clear, concise final answer
    - Ensure all requirements are met
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.5):
        super().__init__(model_name, temperature)
        self.role = "refiner"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Synthesis Expert. Your role is to produce the final, polished answer.

Your responsibilities:
1. Synthesize the reasoning into a clear, concise answer
2. Ensure all aspects of the question are addressed
3. Present the answer in the most helpful format
4. Remove unnecessary verbosity while keeping essential details

Provide a final answer that is:
- Clear and direct
- Complete and accurate
- Well-structured and easy to understand"""),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def refine(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Produce final refined answer."""
        question = state["question"]
        current_reasoning = state["current_reasoning"]
        critique = state["critique"]

        input_text = f"""Question: {question}

Final reasoning:
{current_reasoning}

Critique:
{critique}

Please synthesize this into a clear, final answer."""

        response = self.chain.invoke({"input": input_text})
        final_answer = response.content

        # Create refiner step
        refiner_step = self._create_reasoning_step(
            content=final_answer,
            metadata={"is_final": True}
        )

        return {
            "final_answer": final_answer,
            "reasoning_history": [refiner_step]
        }
