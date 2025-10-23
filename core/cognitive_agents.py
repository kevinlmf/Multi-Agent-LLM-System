"""
Enhanced Cognitive Agents with Perception and Memory
Extends original agents to leverage world state and past experiences
"""

from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime
import os

from .cognitive_state import create_reasoning_step, get_full_context


class CognitiveBaseAgent:
    """Base class for cognitive agents with perception and memory access."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
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
        return create_reasoning_step(self.role, content, metadata)

    def _get_cognitive_context(self, state: Dict[str, Any]) -> str:
        """Get full cognitive context including perception and memory."""
        return get_full_context(state)


class PerceiverAgent(CognitiveBaseAgent):
    """
    Perceiver Agent: Processes observations and updates world model.

    Responsibilities:
    - Parse incoming observations
    - Extract entities and relations
    - Update world graph
    - Provide context summary
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.5):
        super().__init__(model_name, temperature)
        self.role = "perceiver"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Perception Agent. Your task is to analyze observations and extract structured information.

When processing observations:
1. Identify key entities (people, organizations, prices, quantities, events)
2. Extract relationships between entities
3. Detect important events or changes
4. Summarize the current situation

Provide clear, structured summaries that can inform decision-making."""),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def perceive(self, state: Dict[str, Any], observation: str) -> Dict[str, Any]:
        """Process observation and update perception state."""
        input_text = f"""New Observation:
{observation}

Please analyze this observation and provide:
1. Key entities mentioned
2. Important relationships or patterns
3. Notable events or changes
4. A brief summary of the current situation"""

        response = self.chain.invoke({"input": input_text})
        perception_summary = response.content

        # Create perception step
        perception_step = self._create_reasoning_step(
            content=perception_summary,
            metadata={"observation": observation[:200]}
        )

        return {
            "world_context": perception_summary,
            "raw_observation": {"text": observation, "timestamp": datetime.now().isoformat()},
            "reasoning_history": [perception_step]
        }


class CognitiveReasonerAgent(CognitiveBaseAgent):
    """
    Enhanced Reasoner Agent with perception and memory access.

    Responsibilities:
    - Generate reasoning informed by current world state
    - Leverage past experiences from memory
    - Consider learned patterns and knowledge
    - Refine based on critique
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.role = "reasoner"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Cognitive Reasoning Agent. You have access to:
- Current world state (what's happening now)
- Past experiences (what happened before)
- Learned knowledge (patterns and insights)

Your task is to reason step-by-step with clarity and precision, incorporating all available context.

When reasoning:
1. Consider the current situation (perception)
2. Draw on relevant past experiences (memory)
3. Apply learned knowledge and patterns
4. Think through the problem logically
5. If receiving critique, address each point carefully

Be rigorous, consider alternatives, and explain your thought process."""),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def reason(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reasoning informed by perception and memory."""
        question = state["question"]
        iteration = state.get("iteration", 0)
        critique = state.get("critique", "")

        # Get full cognitive context
        cognitive_context = self._get_cognitive_context(state)

        if iteration == 0:
            # First iteration: initial reasoning
            input_text = f"""Question/Task: {question}

{cognitive_context}

Please provide step-by-step reasoning to solve this problem, considering all available context."""
        else:
            # Subsequent iterations: refine based on critique
            previous_reasoning = state.get("current_reasoning", "")
            input_text = f"""Question/Task: {question}

{cognitive_context}

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
            metadata={
                "iteration": iteration,
                "type": "initial" if iteration == 0 else "refined",
                "has_memory": len(state.get("retrieved_memories", [])) > 0,
                "has_perception": state.get("world_context") is not None
            }
        )

        return {
            "current_reasoning": reasoning,
            "reasoning_history": [reasoning_step]
        }


class CognitiveCriticAgent(CognitiveBaseAgent):
    """
    Enhanced Critic Agent that evaluates reasoning quality.

    Responsibilities:
    - Evaluate reasoning against current reality (perception)
    - Check consistency with past experiences (memory)
    - Verify alignment with learned knowledge
    - Provide actionable feedback
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        super().__init__(model_name, temperature)
        self.role = "critic"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Critical Thinking Expert with access to comprehensive context.

Evaluation criteria:
1. **Reality Check**: Does reasoning align with current world state?
2. **Historical Consistency**: Is it consistent with past experiences?
3. **Knowledge Application**: Does it properly apply learned patterns?
4. **Logical Soundness**: Is the reasoning logically valid?
5. **Completeness**: Are all aspects addressed?

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
        """Evaluate reasoning with cognitive context."""
        question = state["question"]
        current_reasoning = state["current_reasoning"]

        # Get full cognitive context
        cognitive_context = self._get_cognitive_context(state)

        input_text = f"""Question/Task: {question}

{cognitive_context}

Reasoning to evaluate:
{current_reasoning}

Please evaluate this reasoning thoroughly, considering whether it properly leverages the available context."""

        response = self.chain.invoke({"input": input_text})
        critique_text = response.content

        # Parse confidence score
        confidence_score = self._parse_confidence(critique_text)

        # Create critique step
        critique_step = self._create_reasoning_step(
            content=critique_text,
            metadata={"confidence_score": confidence_score}
        )

        # Decide whether to continue
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


class CognitiveRefinerAgent(CognitiveBaseAgent):
    """
    Enhanced Refiner Agent that produces final output.

    Responsibilities:
    - Synthesize reasoning with full context
    - Produce actionable recommendations
    - Ensure alignment with world state and memories
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.5):
        super().__init__(model_name, temperature)
        self.role = "refiner"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Synthesis Expert. Your role is to produce the final, polished answer.

Your responsibilities:
1. Synthesize reasoning into clear, actionable output
2. Ensure alignment with current world state
3. Incorporate relevant past experiences
4. Apply learned knowledge appropriately
5. Present recommendations that are practical and well-justified

Provide a final answer that is:
- Clear and actionable
- Grounded in reality (perception)
- Informed by experience (memory)
- Complete and accurate
- Well-structured and practical"""),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def refine(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Produce final answer with full cognitive context."""
        question = state["question"]
        current_reasoning = state["current_reasoning"]
        critique = state["critique"]

        # Get full cognitive context
        cognitive_context = self._get_cognitive_context(state)

        input_text = f"""Question/Task: {question}

{cognitive_context}

Final reasoning:
{current_reasoning}

Critique:
{critique}

Please synthesize this into a clear, final answer that is actionable and well-grounded in the available context."""

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


class ConsolidatorAgent(CognitiveBaseAgent):
    """
    Consolidator Agent: Learns from experiences and extracts knowledge.

    Responsibilities:
    - Analyze recent experiences
    - Extract patterns and insights
    - Update semantic memory
    - Strengthen useful knowledge
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.5):
        super().__init__(model_name, temperature)
        self.role = "consolidator"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Knowledge Consolidation Agent. Your task is to learn from experiences.

When analyzing experiences:
1. Identify recurring patterns
2. Extract generalizable insights
3. Recognize cause-effect relationships
4. Formulate actionable knowledge

Provide clear, concise knowledge statements that can guide future decisions."""),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def consolidate(self, experiences: List[Dict[str, Any]]) -> List[str]:
        """Extract knowledge from experiences."""
        # Format experiences
        exp_text = "\n\n".join([
            f"Experience {i+1}:\n{exp.get('summary', str(exp))}"
            for i, exp in enumerate(experiences[:10])  # Last 10 experiences
        ])

        input_text = f"""Recent Experiences:
{exp_text}

Please analyze these experiences and extract:
1. Common patterns you observe
2. Insights about what works well
3. Lessons learned from outcomes
4. Actionable knowledge for future decisions

Provide 3-5 key insights as bullet points."""

        response = self.chain.invoke({"input": input_text})
        insights = response.content

        # Parse insights into list
        knowledge_list = []
        for line in insights.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                knowledge_list.append(line.lstrip("-•0123456789. "))

        return knowledge_list
