"""
Deductive Reasoning MCP Pattern

This pattern demonstrates reasoning from general principles to specific
conclusions, applying rules and logical inference.

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
class DeductiveReasoningState(TypedDict):
    """State for deductive reasoning workflow"""
    premises: List[str]
    rules: List[Dict[str, str]]
    target_conclusion: str
    logical_chain: List[Dict[str, Any]]
    intermediate_conclusions: List[Dict[str, str]]
    final_conclusion: Dict[str, Any]
    validity_check: Dict[str, Any]
    soundness_check: Dict[str, Any]
    final_reasoning: str
    messages: Annotated[List, operator.add]
    current_step: str


class PremiseAnalyzer:
    """Analyze premises for deductive reasoning"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_premises(self, premises: List[str],
                        rules: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze premises"""
        prompt = f"""Analyze these premises for deductive reasoning:

Premises:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(premises))}

Rules:
{chr(10).join(f'{i+1}. {r.get("name", "Unknown")}: {r.get("rule", "Unknown")}' for i, r in enumerate(rules))}

Analyze:
1. Are premises consistent?
2. What type are they (universal, particular, conditional)?
3. What can be inferred directly?

Return JSON:
{{
    "consistency": "consistent|inconsistent|unknown",
    "premise_types": [{{"premise": "text", "type": "universal|particular|conditional"}}],
    "direct_inferences": ["what can be concluded immediately"],
    "applicable_rules": ["which rules can be applied"]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at logical premise analysis."),
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
            "consistency": "unknown",
            "premise_types": [],
            "direct_inferences": [],
            "applicable_rules": []
        }


class LogicalChainBuilder:
    """Build chain of logical inferences"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def build_chain(self, premises: List[str], rules: List[Dict[str, str]],
                   target: str, premise_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build logical inference chain"""
        prompt = f"""Build a logical chain from premises to conclusion:

Premises:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(premises))}

Rules Available:
{chr(10).join(f'{i+1}. {r.get("name", "Unknown")}: {r.get("rule", "Unknown")}' for i, r in enumerate(rules))}

Target Conclusion: {target}

Direct Inferences Available:
{chr(10).join(f'- {inf}' for inf in premise_analysis.get('direct_inferences', []))}

Build a step-by-step logical chain showing:
1. What premise(s) or prior conclusion(s) are used
2. What rule is applied
3. What conclusion follows

Return JSON array:
[
    {{
        "step": 1,
        "from": ["premise 1", "premise 2"],
        "rule": "rule name",
        "inference": "what is inferred",
        "type": "modus_ponens|modus_tollens|syllogism|transitive|other"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at building logical chains."),
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


class IntermediateConclusionExtractor:
    """Extract intermediate conclusions"""
    
    def extract_conclusions(self, logical_chain: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract intermediate conclusions from chain"""
        conclusions = []
        
        for step in logical_chain:
            conclusions.append({
                "step": step.get("step", 0),
                "conclusion": step.get("inference", "Unknown"),
                "based_on": ", ".join(step.get("from", [])),
                "rule_used": step.get("rule", "Unknown")
            })
        
        return conclusions


class ValidityChecker:
    """Check logical validity"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def check_validity(self, premises: List[str], logical_chain: List[Dict[str, Any]],
                      final_conclusion: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the argument is logically valid"""
        prompt = f"""Check the logical validity of this argument:

Premises:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(premises))}

Logical Chain:
{json.dumps(logical_chain, indent=2)}

Conclusion: {final_conclusion.get('conclusion', 'Unknown')}

Determine:
1. Is each step logically valid?
2. Does the conclusion follow necessarily from premises?
3. Are there any logical fallacies?
4. Is the overall argument valid?

Return JSON:
{{
    "overall_validity": "valid|invalid|uncertain",
    "step_validity": [{{"step": 1, "valid": true|false, "issue": "if invalid"}}],
    "fallacies_detected": ["fallacy1", "fallacy2"],
    "validity_explanation": "why it is or isn't valid"
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at checking logical validity."),
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
            "overall_validity": "uncertain",
            "step_validity": [],
            "fallacies_detected": [],
            "validity_explanation": "Unable to determine"
        }


class SoundnessChecker:
    """Check soundness (validity + true premises)"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def check_soundness(self, premises: List[str], validity: Dict[str, Any]) -> Dict[str, Any]:
        """Check if argument is sound"""
        prompt = f"""Check the soundness of this argument:

Premises:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(premises))}

Validity Status: {validity.get('overall_validity', 'unknown')}

For soundness, check:
1. Is the argument valid? (already assessed)
2. Are all premises actually true?

Return JSON:
{{
    "overall_soundness": "sound|unsound|uncertain",
    "premise_truth_assessment": [{{"premise": "text", "likely_true": true|false, "reasoning": "why"}}],
    "soundness_explanation": "overall assessment"
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at assessing argument soundness."),
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
            "overall_soundness": "uncertain",
            "premise_truth_assessment": [],
            "soundness_explanation": "Unable to determine"
        }


# Agent functions
def initialize_deductive_reasoning(state: DeductiveReasoningState) -> DeductiveReasoningState:
    """Initialize deductive reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing deductive reasoning with {len(state['premises'])} premises"
    ))
    state["logical_chain"] = []
    state["intermediate_conclusions"] = []
    state["current_step"] = "initialized"
    return state


def analyze_premises(state: DeductiveReasoningState) -> DeductiveReasoningState:
    """Analyze premises"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = PremiseAnalyzer(llm)
    
    analysis = analyzer.analyze_premises(
        state["premises"],
        state["rules"]
    )
    
    state["messages"].append(HumanMessage(
        content=f"Premise analysis: {analysis.get('consistency', 'unknown')}, "
                f"{len(analysis.get('direct_inferences', []))} direct inferences"
    ))
    state["current_step"] = "premises_analyzed"
    
    # Store analysis for next step
    state["_premise_analysis"] = analysis
    
    return state


def build_logical_chain(state: DeductiveReasoningState) -> DeductiveReasoningState:
    """Build logical chain"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    builder = LogicalChainBuilder(llm)
    
    premise_analysis = state.get("_premise_analysis", {})
    
    chain = builder.build_chain(
        state["premises"],
        state["rules"],
        state["target_conclusion"],
        premise_analysis
    )
    
    state["logical_chain"] = chain
    
    state["messages"].append(HumanMessage(
        content=f"Built logical chain with {len(chain)} steps"
    ))
    state["current_step"] = "chain_built"
    return state


def extract_intermediate_conclusions(state: DeductiveReasoningState) -> DeductiveReasoningState:
    """Extract intermediate conclusions"""
    extractor = IntermediateConclusionExtractor()
    
    conclusions = extractor.extract_conclusions(state["logical_chain"])
    
    state["intermediate_conclusions"] = conclusions
    
    state["messages"].append(HumanMessage(
        content=f"Extracted {len(conclusions)} intermediate conclusions"
    ))
    state["current_step"] = "intermediate_extracted"
    return state


def derive_final_conclusion(state: DeductiveReasoningState) -> DeductiveReasoningState:
    """Derive final conclusion"""
    if state["logical_chain"]:
        last_step = state["logical_chain"][-1]
        
        final = {
            "conclusion": last_step.get("inference", "No conclusion reached"),
            "based_on_steps": len(state["logical_chain"]),
            "inference_type": last_step.get("type", "unknown"),
            "final_step": last_step
        }
    else:
        final = {
            "conclusion": "No logical chain constructed",
            "based_on_steps": 0,
            "inference_type": "none",
            "final_step": {}
        }
    
    state["final_conclusion"] = final
    
    state["messages"].append(HumanMessage(
        content=f"Final conclusion: {final['conclusion'][:100]}..."
    ))
    state["current_step"] = "conclusion_derived"
    return state


def check_validity(state: DeductiveReasoningState) -> DeductiveReasoningState:
    """Check validity"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    checker = ValidityChecker(llm)
    
    validity = checker.check_validity(
        state["premises"],
        state["logical_chain"],
        state["final_conclusion"]
    )
    
    state["validity_check"] = validity
    
    state["messages"].append(HumanMessage(
        content=f"Validity: {validity.get('overall_validity', 'unknown')}, "
                f"{len(validity.get('fallacies_detected', []))} fallacies detected"
    ))
    state["current_step"] = "validity_checked"
    return state


def check_soundness(state: DeductiveReasoningState) -> DeductiveReasoningState:
    """Check soundness"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    checker = SoundnessChecker(llm)
    
    soundness = checker.check_soundness(
        state["premises"],
        state["validity_check"]
    )
    
    state["soundness_check"] = soundness
    
    state["messages"].append(HumanMessage(
        content=f"Soundness: {soundness.get('overall_soundness', 'unknown')}"
    ))
    state["current_step"] = "soundness_checked"
    return state


def generate_report(state: DeductiveReasoningState) -> DeductiveReasoningState:
    """Generate final report"""
    report = f"""
DEDUCTIVE REASONING REPORT
==========================

Target Conclusion: {state['target_conclusion']}

Premises ({len(state['premises'])}):
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(state['premises']))}

Rules Applied:
{chr(10).join(f'{i+1}. {r.get("name", "Unknown")}: {r.get("rule", "Unknown")}' for i, r in enumerate(state['rules']))}

Logical Chain ({len(state['logical_chain'])} steps):
"""
    
    for step in state['logical_chain']:
        report += f"\nStep {step.get('step', 0)}:\n"
        report += f"  From: {', '.join(step.get('from', []))}\n"
        report += f"  Rule: {step.get('rule', 'Unknown')}\n"
        report += f"  Type: {step.get('type', 'unknown')}\n"
        report += f"  Inference: {step.get('inference', 'Unknown')}\n"
    
    report += f"""
Intermediate Conclusions:
{chr(10).join(f'{i+1}. {c.get("conclusion", "Unknown")}' for i, c in enumerate(state['intermediate_conclusions']))}

FINAL CONCLUSION:
{state['final_conclusion'].get('conclusion', 'No conclusion')}

Based on: {state['final_conclusion'].get('based_on_steps', 0)} logical steps
Inference Type: {state['final_conclusion'].get('inference_type', 'unknown')}

Validity Analysis:
- Overall Validity: {state['validity_check'].get('overall_validity', 'unknown')}
- Explanation: {state['validity_check'].get('validity_explanation', 'N/A')}
"""
    
    if state['validity_check'].get('fallacies_detected'):
        report += f"\nFallacies Detected:\n"
        for fallacy in state['validity_check']['fallacies_detected']:
            report += f"- {fallacy}\n"
    
    report += f"""
Step-by-Step Validity:
"""
    
    for sv in state['validity_check'].get('step_validity', []):
        status = "✓" if sv.get('valid') else "✗"
        report += f"{status} Step {sv.get('step', 0)}"
        if not sv.get('valid'):
            report += f" - Issue: {sv.get('issue', 'Unknown')}"
        report += "\n"
    
    report += f"""
Soundness Analysis:
- Overall Soundness: {state['soundness_check'].get('overall_soundness', 'unknown')}
- Explanation: {state['soundness_check'].get('soundness_explanation', 'N/A')}

Premise Truth Assessment:
"""
    
    for pta in state['soundness_check'].get('premise_truth_assessment', []):
        likely = "✓" if pta.get('likely_true') else "?"
        report += f"{likely} {pta.get('premise', 'Unknown')}\n"
        report += f"   Reasoning: {pta.get('reasoning', 'N/A')}\n"
    
    report += f"""
Reasoning Summary:
- Premises: {len(state['premises'])}
- Logical Steps: {len(state['logical_chain'])}
- Validity: {state['validity_check'].get('overall_validity', 'unknown')}
- Soundness: {state['soundness_check'].get('overall_soundness', 'unknown')}
- Conclusion Reached: {'Yes' if state['final_conclusion'].get('conclusion') != 'No conclusion' else 'No'}
"""
    
    state["final_reasoning"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_deductive_reasoning_graph():
    """Create the deductive reasoning workflow graph"""
    workflow = StateGraph(DeductiveReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_deductive_reasoning)
    workflow.add_node("analyze_premises", analyze_premises)
    workflow.add_node("build_chain", build_logical_chain)
    workflow.add_node("extract_intermediate", extract_intermediate_conclusions)
    workflow.add_node("derive_conclusion", derive_final_conclusion)
    workflow.add_node("check_validity", check_validity)
    workflow.add_node("check_soundness", check_soundness)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze_premises")
    workflow.add_edge("analyze_premises", "build_chain")
    workflow.add_edge("build_chain", "extract_intermediate")
    workflow.add_edge("extract_intermediate", "derive_conclusion")
    workflow.add_edge("derive_conclusion", "check_validity")
    workflow.add_edge("check_validity", "check_soundness")
    workflow.add_edge("check_soundness", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample deductive reasoning task
    initial_state = {
        "premises": [
            "All software engineers need to understand algorithms",
            "Everyone who understands algorithms can solve complex problems",
            "Maria is a software engineer"
        ],
        "rules": [
            {"name": "Modus Ponens", "rule": "If P then Q, P is true, therefore Q is true"},
            {"name": "Universal Instantiation", "rule": "All X are Y, A is X, therefore A is Y"},
            {"name": "Hypothetical Syllogism", "rule": "If P then Q, if Q then R, therefore if P then R"}
        ],
        "target_conclusion": "Maria can solve complex problems",
        "logical_chain": [],
        "intermediate_conclusions": [],
        "final_conclusion": {},
        "validity_check": {},
        "soundness_check": {},
        "final_reasoning": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_deductive_reasoning_graph()
    
    print("Deductive Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Validity: {result['validity_check'].get('overall_validity', 'unknown')}")
    print(f"Soundness: {result['soundness_check'].get('overall_soundness', 'unknown')}")
