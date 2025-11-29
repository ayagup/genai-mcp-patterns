"""
Common Sense Reasoning MCP Pattern

This pattern demonstrates reasoning based on everyday knowledge, practical
understanding, and intuitive judgments about the world.

Pattern Type: Advanced Reasoning
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class CommonSenseReasoningState(TypedDict):
    """State for common sense reasoning workflow"""
    scenario: str
    question: str
    common_sense_knowledge: List[Dict[str, str]]
    implicit_assumptions: List[Dict[str, str]]
    practical_constraints: List[Dict[str, str]]
    typical_patterns: List[Dict[str, str]]
    reasoning_steps: List[Dict[str, Any]]
    answer: Dict[str, Any]
    confidence_assessment: Dict[str, Any]
    final_reasoning: str
    messages: Annotated[List, operator.add]
    current_step: str


class CommonSenseKnowledgeRetriever:
    """Retrieve relevant common sense knowledge"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def retrieve_knowledge(self, scenario: str, question: str) -> List[Dict[str, str]]:
        """Retrieve common sense knowledge"""
        prompt = f"""Identify relevant common sense knowledge:

Scenario: {scenario}
Question: {question}

List common sense knowledge that applies (things everyone knows):
- Physical laws (gravity, cause-effect)
- Human behavior patterns
- Object properties
- Social norms
- Practical facts

Return JSON array:
[
    {{
        "knowledge": "statement of common sense fact",
        "category": "physical|social|psychological|practical",
        "relevance": "why it's relevant to scenario"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at identifying relevant common sense knowledge."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class ImplicitAssumptionIdentifier:
    """Identify implicit assumptions"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def identify_assumptions(self, scenario: str) -> List[Dict[str, str]]:
        """Identify implicit assumptions in scenario"""
        prompt = f"""Identify implicit assumptions in this scenario:

Scenario: {scenario}

What assumptions are being made that aren't stated explicitly?
Consider:
- Normal conditions (not extraordinary circumstances)
- Typical behavior
- Standard capabilities
- Common contexts

Return JSON array:
[
    {{
        "assumption": "what is assumed",
        "justification": "why this is a reasonable assumption",
        "if_violated": "what would happen if this assumption is wrong"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at identifying implicit assumptions."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class PracticalConstraintAnalyzer:
    """Analyze practical constraints"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_constraints(self, scenario: str, question: str) -> List[Dict[str, str]]:
        """Analyze practical constraints"""
        prompt = f"""Identify practical constraints:

Scenario: {scenario}
Question: {question}

What practical constraints apply?
- Physical limitations
- Time constraints
- Resource constraints
- Capability constraints
- Safety considerations

Return JSON array:
[
    {{
        "constraint": "what the constraint is",
        "type": "physical|temporal|resource|capability|safety",
        "impact": "how it affects the situation"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at analyzing practical constraints."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class TypicalPatternRecognizer:
    """Recognize typical patterns and expectations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def recognize_patterns(self, scenario: str) -> List[Dict[str, str]]:
        """Recognize typical patterns"""
        prompt = f"""Recognize typical patterns in this scenario:

Scenario: {scenario}

What typical patterns or expectations apply?
- How things normally work
- Typical sequences of events
- Expected behaviors
- Standard procedures

Return JSON array:
[
    {{
        "pattern": "description of typical pattern",
        "expectation": "what we typically expect",
        "exceptions": "when this pattern might not hold"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at recognizing typical patterns."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class CommonSenseReasoner:
    """Apply common sense reasoning"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def reason(self, scenario: str, question: str,
              knowledge: List[Dict], assumptions: List[Dict],
              constraints: List[Dict], patterns: List[Dict]) -> List[Dict[str, Any]]:
        """Apply common sense reasoning"""
        prompt = f"""Apply common sense reasoning:

Scenario: {scenario}
Question: {question}

Common Sense Knowledge:
{json.dumps(knowledge, indent=2)}

Implicit Assumptions:
{json.dumps(assumptions, indent=2)}

Practical Constraints:
{json.dumps(constraints, indent=2)}

Typical Patterns:
{json.dumps(patterns, indent=2)}

Reason through the question step by step using common sense.

Return JSON array of reasoning steps:
[
    {{
        "step": 1,
        "reasoning": "what you're thinking",
        "uses": "which knowledge/assumptions/constraints are applied",
        "conclusion": "what this step concludes"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at common sense reasoning."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class AnswerSynthesizer:
    """Synthesize final answer"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def synthesize_answer(self, question: str,
                         reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize final answer"""
        prompt = f"""Synthesize the final answer:

Question: {question}

Reasoning Steps:
{json.dumps(reasoning_steps, indent=2)}

Provide a clear, common-sense answer.

Return JSON:
{{
    "answer": "the answer to the question",
    "confidence": 0.0-1.0,
    "rationale": "brief explanation of why this is the answer",
    "alternative_considerations": ["other things to consider"]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at synthesizing common sense answers."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return {
            "answer": "Unable to determine",
            "confidence": 0.0,
            "rationale": "Insufficient reasoning",
            "alternative_considerations": []
        }


class ConfidenceAssessor:
    """Assess confidence in common sense reasoning"""
    
    def assess_confidence(self, answer: Dict[str, Any],
                         knowledge: List[Dict],
                         assumptions: List[Dict]) -> Dict[str, Any]:
        """Assess confidence"""
        # Factors affecting confidence
        knowledge_strength = len(knowledge) / 10  # More knowledge = higher confidence
        assumption_risk = 1.0 - (len(assumptions) * 0.1)  # More assumptions = lower confidence
        
        answer_confidence = answer.get("confidence", 0.5)
        
        # Combined confidence
        overall = (
            knowledge_strength * 0.3 +
            assumption_risk * 0.3 +
            answer_confidence * 0.4
        )
        
        return {
            "overall_confidence": min(1.0, overall),
            "knowledge_based": knowledge_strength,
            "assumption_based": assumption_risk,
            "answer_confidence": answer_confidence,
            "assessment": self._assess(overall),
            "caveats": self._identify_caveats(assumptions)
        }
    
    def _assess(self, confidence: float) -> str:
        """Assess confidence level"""
        if confidence >= 0.8:
            return "High confidence - strong common sense basis"
        elif confidence >= 0.6:
            return "Moderate confidence - reasonable common sense basis"
        elif confidence >= 0.4:
            return "Low confidence - limited common sense basis"
        else:
            return "Very low confidence - weak common sense basis"
    
    def _identify_caveats(self, assumptions: List[Dict]) -> List[str]:
        """Identify caveats based on assumptions"""
        caveats = []
        
        for assumption in assumptions:
            violation = assumption.get("if_violated", "")
            if violation:
                caveats.append(f"If {assumption.get('assumption', 'unknown')}, then {violation}")
        
        return caveats


# Agent functions
def initialize_common_sense(state: CommonSenseReasoningState) -> CommonSenseReasoningState:
    """Initialize common sense reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing common sense reasoning for: {state['question']}"
    ))
    state["reasoning_steps"] = []
    state["current_step"] = "initialized"
    return state


def retrieve_knowledge(state: CommonSenseReasoningState) -> CommonSenseReasoningState:
    """Retrieve common sense knowledge"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    retriever = CommonSenseKnowledgeRetriever(llm)
    
    knowledge = retriever.retrieve_knowledge(
        state["scenario"],
        state["question"]
    )
    
    state["common_sense_knowledge"] = knowledge
    
    state["messages"].append(HumanMessage(
        content=f"Retrieved {len(knowledge)} pieces of common sense knowledge"
    ))
    state["current_step"] = "knowledge_retrieved"
    return state


def identify_assumptions(state: CommonSenseReasoningState) -> CommonSenseReasoningState:
    """Identify implicit assumptions"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    identifier = ImplicitAssumptionIdentifier(llm)
    
    assumptions = identifier.identify_assumptions(state["scenario"])
    
    state["implicit_assumptions"] = assumptions
    
    state["messages"].append(HumanMessage(
        content=f"Identified {len(assumptions)} implicit assumptions"
    ))
    state["current_step"] = "assumptions_identified"
    return state


def analyze_constraints(state: CommonSenseReasoningState) -> CommonSenseReasoningState:
    """Analyze practical constraints"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    analyzer = PracticalConstraintAnalyzer(llm)
    
    constraints = analyzer.analyze_constraints(
        state["scenario"],
        state["question"]
    )
    
    state["practical_constraints"] = constraints
    
    state["messages"].append(HumanMessage(
        content=f"Analyzed {len(constraints)} practical constraints"
    ))
    state["current_step"] = "constraints_analyzed"
    return state


def recognize_patterns(state: CommonSenseReasoningState) -> CommonSenseReasoningState:
    """Recognize typical patterns"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    recognizer = TypicalPatternRecognizer(llm)
    
    patterns = recognizer.recognize_patterns(state["scenario"])
    
    state["typical_patterns"] = patterns
    
    state["messages"].append(HumanMessage(
        content=f"Recognized {len(patterns)} typical patterns"
    ))
    state["current_step"] = "patterns_recognized"
    return state


def apply_reasoning(state: CommonSenseReasoningState) -> CommonSenseReasoningState:
    """Apply common sense reasoning"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    reasoner = CommonSenseReasoner(llm)
    
    steps = reasoner.reason(
        state["scenario"],
        state["question"],
        state["common_sense_knowledge"],
        state["implicit_assumptions"],
        state["practical_constraints"],
        state["typical_patterns"]
    )
    
    state["reasoning_steps"] = steps
    
    state["messages"].append(HumanMessage(
        content=f"Applied common sense reasoning in {len(steps)} steps"
    ))
    state["current_step"] = "reasoning_applied"
    return state


def synthesize_answer(state: CommonSenseReasoningState) -> CommonSenseReasoningState:
    """Synthesize final answer"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    synthesizer = AnswerSynthesizer(llm)
    
    answer = synthesizer.synthesize_answer(
        state["question"],
        state["reasoning_steps"]
    )
    
    state["answer"] = answer
    
    state["messages"].append(HumanMessage(
        content=f"Answer: {answer.get('answer', 'Unknown')[:100]}... (confidence: {answer.get('confidence', 0):.2f})"
    ))
    state["current_step"] = "answer_synthesized"
    return state


def assess_confidence(state: CommonSenseReasoningState) -> CommonSenseReasoningState:
    """Assess confidence"""
    assessor = ConfidenceAssessor()
    
    assessment = assessor.assess_confidence(
        state["answer"],
        state["common_sense_knowledge"],
        state["implicit_assumptions"]
    )
    
    state["confidence_assessment"] = assessment
    
    state["messages"].append(HumanMessage(
        content=f"Confidence: {assessment['overall_confidence']:.2f} - {assessment['assessment']}"
    ))
    state["current_step"] = "confidence_assessed"
    return state


def generate_report(state: CommonSenseReasoningState) -> CommonSenseReasoningState:
    """Generate final report"""
    report = f"""
COMMON SENSE REASONING REPORT
==============================

Scenario: {state['scenario']}
Question: {state['question']}

Common Sense Knowledge Applied ({len(state['common_sense_knowledge'])}):
"""
    
    for i, k in enumerate(state['common_sense_knowledge'], 1):
        report += f"{i}. {k.get('knowledge', 'Unknown')} ({k.get('category', 'unknown')})\n"
        report += f"   Relevance: {k.get('relevance', 'N/A')}\n"
    
    report += f"""
Implicit Assumptions ({len(state['implicit_assumptions'])}):
"""
    
    for i, a in enumerate(state['implicit_assumptions'], 1):
        report += f"{i}. {a.get('assumption', 'Unknown')}\n"
        report += f"   Justification: {a.get('justification', 'N/A')}\n"
    
    report += f"""
Practical Constraints ({len(state['practical_constraints'])}):
"""
    
    for c in state['practical_constraints']:
        report += f"- {c.get('constraint', 'Unknown')} ({c.get('type', 'unknown')})\n"
        report += f"  Impact: {c.get('impact', 'N/A')}\n"
    
    report += f"""
Typical Patterns ({len(state['typical_patterns'])}):
"""
    
    for p in state['typical_patterns']:
        report += f"- {p.get('pattern', 'Unknown')}\n"
        report += f"  Expectation: {p.get('expectation', 'N/A')}\n"
    
    report += f"""
Reasoning Steps ({len(state['reasoning_steps'])}):
"""
    
    for step in state['reasoning_steps']:
        report += f"\nStep {step.get('step', 0)}:\n"
        report += f"  Reasoning: {step.get('reasoning', 'Unknown')}\n"
        report += f"  Uses: {step.get('uses', 'Unknown')}\n"
        report += f"  Conclusion: {step.get('conclusion', 'Unknown')}\n"
    
    report += f"""
ANSWER:
{state['answer'].get('answer', 'No answer')}

Rationale: {state['answer'].get('rationale', 'N/A')}

Alternative Considerations:
{chr(10).join(f'- {alt}' for alt in state['answer'].get('alternative_considerations', []))}

Confidence Assessment:
- Overall Confidence: {state['confidence_assessment']['overall_confidence']:.2f}
- Assessment: {state['confidence_assessment']['assessment']}
- Knowledge-based: {state['confidence_assessment']['knowledge_based']:.2f}
- Assumption-based: {state['confidence_assessment']['assumption_based']:.2f}

Caveats:
{chr(10).join(f'- {cav}' for cav in state['confidence_assessment']['caveats'])}

Reasoning Summary:
- Common Sense Knowledge: {len(state['common_sense_knowledge'])}
- Implicit Assumptions: {len(state['implicit_assumptions'])}
- Practical Constraints: {len(state['practical_constraints'])}
- Reasoning Steps: {len(state['reasoning_steps'])}
- Confidence: {state['confidence_assessment']['overall_confidence']:.2%}
"""
    
    state["final_reasoning"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_common_sense_reasoning_graph():
    """Create the common sense reasoning workflow graph"""
    workflow = StateGraph(CommonSenseReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_common_sense)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("identify_assumptions", identify_assumptions)
    workflow.add_node("analyze_constraints", analyze_constraints)
    workflow.add_node("recognize_patterns", recognize_patterns)
    workflow.add_node("apply_reasoning", apply_reasoning)
    workflow.add_node("synthesize_answer", synthesize_answer)
    workflow.add_node("assess_confidence", assess_confidence)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "identify_assumptions")
    workflow.add_edge("identify_assumptions", "analyze_constraints")
    workflow.add_edge("analyze_constraints", "recognize_patterns")
    workflow.add_edge("recognize_patterns", "apply_reasoning")
    workflow.add_edge("apply_reasoning", "synthesize_answer")
    workflow.add_edge("synthesize_answer", "assess_confidence")
    workflow.add_edge("assess_confidence", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample common sense reasoning task
    initial_state = {
        "scenario": "A person walks into a restaurant at 2 PM on a Tuesday. The restaurant has a 'Closed' sign on the door but the lights are on inside and people can be seen through the window.",
        "question": "Should the person try to enter the restaurant?",
        "common_sense_knowledge": [],
        "implicit_assumptions": [],
        "practical_constraints": [],
        "typical_patterns": [],
        "reasoning_steps": [],
        "answer": {},
        "confidence_assessment": {},
        "final_reasoning": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_common_sense_reasoning_graph()
    
    print("Common Sense Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Answer: {result['answer'].get('answer', 'Unknown')}")
    print(f"Confidence: {result['confidence_assessment']['overall_confidence']:.2%}")
