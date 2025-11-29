"""
Probabilistic Reasoning MCP Pattern

This pattern demonstrates reasoning about uncertainty, probabilities,
likelihoods, and risk assessment.

Pattern Type: Advanced Reasoning
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any, Tuple
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class ProbabilisticReasoningState(TypedDict):
    """State for probabilistic reasoning workflow"""
    scenario: str
    question: str
    events: List[Dict[str, Any]]
    probabilities: Dict[str, float]
    dependencies: List[Dict[str, str]]
    conditional_probs: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    expected_values: List[Dict[str, Any]]
    uncertainty_analysis: Dict[str, Any]
    recommendations: List[Dict[str, str]]
    final_answer: str
    messages: Annotated[List, operator.add]
    current_step: str


class EventIdentifier:
    """Identify probabilistic events"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def identify_events(self, scenario: str) -> List[Dict[str, Any]]:
        """Identify events in scenario"""
        prompt = f"""Identify probabilistic events in the scenario:

Scenario: {scenario}

Identify:
1. Key events or outcomes
2. Whether they are certain, uncertain, or random
3. Whether they are independent or dependent on other events

Return JSON array:
[
    {{
        "event_id": "unique_id",
        "event": "description of event",
        "outcome_type": "binary|categorical|continuous",
        "possible_outcomes": ["list of possible outcomes"],
        "certainty_level": "certain|uncertain|random"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at identifying probabilistic events."),
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


class ProbabilityEstimator:
    """Estimate probabilities"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def estimate_probabilities(self, events: List[Dict[str, Any]],
                              scenario: str) -> Dict[str, float]:
        """Estimate base probabilities"""
        prompt = f"""Estimate probabilities for events:

Scenario: {scenario}

Events:
{json.dumps(events, indent=2)}

For each event and outcome, estimate:
- P(outcome) - probability of outcome
- Based on: statistics, domain knowledge, or reasonable assumptions

Return JSON object:
{{
    "event_id:outcome": 0.0-1.0,
    ...
}}

Example:
{{
    "rain:yes": 0.3,
    "rain:no": 0.7,
    "traffic:heavy": 0.4,
    "traffic:moderate": 0.4,
    "traffic:light": 0.2
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at probability estimation."),
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
        
        return {}


class DependencyAnalyzer:
    """Analyze event dependencies"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_dependencies(self, events: List[Dict[str, Any]],
                            scenario: str) -> List[Dict[str, str]]:
        """Analyze dependencies between events"""
        prompt = f"""Analyze dependencies between events:

Scenario: {scenario}

Events:
{json.dumps(events, indent=2)}

Identify:
1. Which events depend on others
2. Whether independent or dependent
3. Type of dependency (causal, conditional, etc.)

Return JSON array:
[
    {{
        "event1_id": "id",
        "event2_id": "id",
        "dependency_type": "independent|dependent|causal|conditional",
        "description": "how they relate"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at analyzing probabilistic dependencies."),
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


class ConditionalProbCalculator:
    """Calculate conditional probabilities"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def calculate_conditionals(self, events: List[Dict[str, Any]],
                              dependencies: List[Dict[str, str]],
                              base_probs: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate conditional probabilities"""
        prompt = f"""Calculate conditional probabilities:

Base Probabilities:
{json.dumps(base_probs, indent=2)}

Dependencies:
{json.dumps(dependencies, indent=2)}

Calculate conditional probabilities like:
- P(A|B) - probability of A given B
- P(B|A) - probability of B given A

Use Bayes' theorem where applicable.

Return JSON array:
[
    {{
        "conditional": "P(A|B)",
        "event_a": "event_id",
        "event_b": "event_id",
        "probability": 0.0-1.0,
        "calculation_method": "bayes|direct|estimated"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at calculating conditional probabilities."),
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


class RiskAssessor:
    """Assess risk"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def assess_risk(self, events: List[Dict[str, Any]],
                   probabilities: Dict[str, float],
                   question: str) -> Dict[str, Any]:
        """Assess risk in scenario"""
        prompt = f"""Assess risk in this scenario:

Question: {question}

Events and Probabilities:
{json.dumps(probabilities, indent=2)}

Assess:
1. What are the risks/negative outcomes?
2. Likelihood of each risk
3. Severity/impact of each risk
4. Overall risk level (low/medium/high)

Return JSON:
{{
    "risks": [
        {{
            "risk": "description",
            "probability": 0.0-1.0,
            "severity": "low|medium|high",
            "impact_description": "what happens if occurs"
        }}
    ],
    "overall_risk_level": "low|medium|high",
    "key_risk_factors": ["factors contributing to risk"],
    "confidence": 0.0-1.0
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at risk assessment."),
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
        
        return {}


class ExpectedValueCalculator:
    """Calculate expected values"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def calculate_expected_values(self, events: List[Dict[str, Any]],
                                  probabilities: Dict[str, float],
                                  scenario: str) -> List[Dict[str, Any]]:
        """Calculate expected values for decisions"""
        prompt = f"""Calculate expected values:

Scenario: {scenario}

Events and Probabilities:
{json.dumps(probabilities, indent=2)}

For decisions/actions in scenario, calculate:
- Expected value = Σ(probability × outcome_value)
- For each possible decision or action

Return JSON array:
[
    {{
        "decision": "description of decision/action",
        "outcomes": [
            {{
                "outcome": "what happens",
                "probability": 0.0-1.0,
                "value": "numeric or qualitative value"
            }}
        ],
        "expected_value": "calculated or estimated",
        "recommendation": "take|avoid|neutral"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at expected value calculation."),
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


class UncertaintyAnalyzer:
    """Analyze uncertainty"""
    
    def analyze_uncertainty(self, events: List[Dict[str, Any]],
                           probabilities: Dict[str, float]) -> Dict[str, Any]:
        """Analyze sources and levels of uncertainty"""
        # Calculate entropy as measure of uncertainty
        uncertainties = {}
        
        for event in events:
            event_id = event.get("event_id", "")
            outcomes = event.get("possible_outcomes", [])
            
            # Calculate entropy for this event
            entropy = 0.0
            for outcome in outcomes:
                key = f"{event_id}:{outcome}"
                p = probabilities.get(key, 0.0)
                
                if p > 0:
                    entropy -= p * (p ** 0.5)  # Simplified entropy measure
            
            uncertainties[event_id] = entropy
        
        # Overall uncertainty
        avg_uncertainty = sum(uncertainties.values()) / len(uncertainties) if uncertainties else 0.0
        
        return {
            "event_uncertainties": uncertainties,
            "overall_uncertainty": avg_uncertainty,
            "uncertainty_level": "high" if avg_uncertainty > 0.7 else "medium" if avg_uncertainty > 0.4 else "low",
            "most_uncertain_events": sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)[:3]
        }


class ProbabilisticRecommender:
    """Generate recommendations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_recommendations(self, risk_assessment: Dict[str, Any],
                                expected_values: List[Dict[str, Any]],
                                uncertainty: Dict[str, Any],
                                question: str) -> List[Dict[str, str]]:
        """Generate probabilistic recommendations"""
        prompt = f"""Generate recommendations based on probabilistic analysis:

Question: {question}

Risk Assessment:
{json.dumps(risk_assessment, indent=2)}

Expected Values:
{json.dumps(expected_values, indent=2)}

Uncertainty Analysis:
{json.dumps(uncertainty, indent=2)}

Generate recommendations considering:
1. Risk levels
2. Expected values
3. Uncertainty
4. Trade-offs

Return JSON array:
[
    {{
        "recommendation": "what to do",
        "rationale": "why this recommendation",
        "confidence": 0.0-1.0,
        "risk_level": "low|medium|high"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at making decisions under uncertainty."),
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


# Agent functions
def initialize_probabilistic_reasoning(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Initialize probabilistic reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing probabilistic reasoning: {state['question']}"
    ))
    state["events"] = []
    state["probabilities"] = {}
    state["dependencies"] = []
    state["current_step"] = "initialized"
    return state


def identify_events(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Identify probabilistic events"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    identifier = EventIdentifier(llm)
    
    events = identifier.identify_events(state["scenario"])
    
    state["events"] = events
    
    state["messages"].append(HumanMessage(
        content=f"Identified {len(events)} probabilistic events"
    ))
    state["current_step"] = "events_identified"
    return state


def estimate_probabilities(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Estimate probabilities"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    estimator = ProbabilityEstimator(llm)
    
    probabilities = estimator.estimate_probabilities(
        state["events"],
        state["scenario"]
    )
    
    state["probabilities"] = probabilities
    
    state["messages"].append(HumanMessage(
        content=f"Estimated {len(probabilities)} probabilities"
    ))
    state["current_step"] = "probabilities_estimated"
    return state


def analyze_dependencies(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Analyze event dependencies"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = DependencyAnalyzer(llm)
    
    dependencies = analyzer.analyze_dependencies(
        state["events"],
        state["scenario"]
    )
    
    state["dependencies"] = dependencies
    
    state["messages"].append(HumanMessage(
        content=f"Analyzed {len(dependencies)} dependencies"
    ))
    state["current_step"] = "dependencies_analyzed"
    return state


def calculate_conditionals(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Calculate conditional probabilities"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    calculator = ConditionalProbCalculator(llm)
    
    conditional_probs = calculator.calculate_conditionals(
        state["events"],
        state["dependencies"],
        state["probabilities"]
    )
    
    state["conditional_probs"] = conditional_probs
    
    state["messages"].append(HumanMessage(
        content=f"Calculated {len(conditional_probs)} conditional probabilities"
    ))
    state["current_step"] = "conditionals_calculated"
    return state


def assess_risk(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Assess risk"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    assessor = RiskAssessor(llm)
    
    risk_assessment = assessor.assess_risk(
        state["events"],
        state["probabilities"],
        state["question"]
    )
    
    state["risk_assessment"] = risk_assessment
    
    state["messages"].append(HumanMessage(
        content=f"Risk assessment: {risk_assessment.get('overall_risk_level', 'unknown')} risk level"
    ))
    state["current_step"] = "risk_assessed"
    return state


def calculate_expected_values(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Calculate expected values"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    calculator = ExpectedValueCalculator(llm)
    
    expected_values = calculator.calculate_expected_values(
        state["events"],
        state["probabilities"],
        state["scenario"]
    )
    
    state["expected_values"] = expected_values
    
    state["messages"].append(HumanMessage(
        content=f"Calculated {len(expected_values)} expected values"
    ))
    state["current_step"] = "expected_values_calculated"
    return state


def analyze_uncertainty(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Analyze uncertainty"""
    analyzer = UncertaintyAnalyzer()
    
    uncertainty_analysis = analyzer.analyze_uncertainty(
        state["events"],
        state["probabilities"]
    )
    
    state["uncertainty_analysis"] = uncertainty_analysis
    
    state["messages"].append(HumanMessage(
        content=f"Uncertainty level: {uncertainty_analysis['uncertainty_level']}"
    ))
    state["current_step"] = "uncertainty_analyzed"
    return state


def generate_recommendations(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Generate recommendations"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    recommender = ProbabilisticRecommender(llm)
    
    recommendations = recommender.generate_recommendations(
        state["risk_assessment"],
        state["expected_values"],
        state["uncertainty_analysis"],
        state["question"]
    )
    
    state["recommendations"] = recommendations
    
    state["messages"].append(HumanMessage(
        content=f"Generated {len(recommendations)} recommendations"
    ))
    state["current_step"] = "recommendations_generated"
    return state


def generate_report(state: ProbabilisticReasoningState) -> ProbabilisticReasoningState:
    """Generate final report"""
    report = f"""
PROBABILISTIC REASONING REPORT
==============================

Scenario: {state['scenario']}
Question: {state['question']}

Probabilistic Events ({len(state['events'])}):
"""
    
    for event in state['events']:
        report += f"\n{event.get('event_id', 'unknown')}:\n"
        report += f"  Description: {event.get('event', 'N/A')}\n"
        report += f"  Type: {event.get('outcome_type', 'unknown')}\n"
        report += f"  Outcomes: {', '.join(event.get('possible_outcomes', []))}\n"
    
    report += f"""
Base Probabilities ({len(state['probabilities'])}):
"""
    
    for key, prob in sorted(state['probabilities'].items()):
        report += f"- P({key}) = {prob:.2%}\n"
    
    report += f"""
Event Dependencies ({len(state['dependencies'])}):
"""
    
    for dep in state['dependencies']:
        report += f"- {dep.get('event1_id', 'A')} ←→ {dep.get('event2_id', 'B')}: {dep.get('dependency_type', 'unknown')}\n"
        report += f"  {dep.get('description', 'N/A')}\n"
    
    report += f"""
Conditional Probabilities ({len(state['conditional_probs'])}):
"""
    
    for cond in state['conditional_probs']:
        report += f"- {cond.get('conditional', 'P(A|B)')} = {cond.get('probability', 0):.2%}\n"
        report += f"  Method: {cond.get('calculation_method', 'unknown')}\n"
    
    report += f"""
Risk Assessment:
- Overall Risk Level: {state['risk_assessment'].get('overall_risk_level', 'unknown').upper()}
- Key Risk Factors: {', '.join(state['risk_assessment'].get('key_risk_factors', []))}

Identified Risks:
"""
    
    for risk in state['risk_assessment'].get('risks', []):
        report += f"\n  {risk.get('risk', 'Unknown risk')}:\n"
        report += f"  - Probability: {risk.get('probability', 0):.2%}\n"
        report += f"  - Severity: {risk.get('severity', 'unknown')}\n"
        report += f"  - Impact: {risk.get('impact_description', 'N/A')}\n"
    
    report += f"""
Expected Values ({len(state['expected_values'])}):
"""
    
    for ev in state['expected_values']:
        report += f"\nDecision: {ev.get('decision', 'Unknown')}\n"
        report += f"  Expected Value: {ev.get('expected_value', 'N/A')}\n"
        report += f"  Recommendation: {ev.get('recommendation', 'neutral')}\n"
    
    report += f"""
Uncertainty Analysis:
- Overall Uncertainty: {state['uncertainty_analysis']['overall_uncertainty']:.2f}
- Uncertainty Level: {state['uncertainty_analysis']['uncertainty_level'].upper()}
- Most Uncertain Events: {', '.join([e[0] for e in state['uncertainty_analysis']['most_uncertain_events']])}

RECOMMENDATIONS ({len(state['recommendations'])}):
"""
    
    for i, rec in enumerate(state['recommendations'], 1):
        report += f"\n{i}. {rec.get('recommendation', 'No recommendation')}\n"
        report += f"   Rationale: {rec.get('rationale', 'N/A')}\n"
        report += f"   Confidence: {rec.get('confidence', 0):.2%}\n"
        report += f"   Risk Level: {rec.get('risk_level', 'unknown')}\n"
    
    report += f"""
Summary:
- Total Events: {len(state['events'])}
- Base Probabilities: {len(state['probabilities'])}
- Conditional Probabilities: {len(state['conditional_probs'])}
- Overall Risk: {state['risk_assessment'].get('overall_risk_level', 'unknown')}
- Uncertainty: {state['uncertainty_analysis']['uncertainty_level']}
- Recommendations: {len(state['recommendations'])}
"""
    
    state["final_answer"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_probabilistic_reasoning_graph():
    """Create the probabilistic reasoning workflow graph"""
    workflow = StateGraph(ProbabilisticReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_probabilistic_reasoning)
    workflow.add_node("identify_events", identify_events)
    workflow.add_node("estimate_probabilities", estimate_probabilities)
    workflow.add_node("analyze_dependencies", analyze_dependencies)
    workflow.add_node("calculate_conditionals", calculate_conditionals)
    workflow.add_node("assess_risk", assess_risk)
    workflow.add_node("calculate_expected_values", calculate_expected_values)
    workflow.add_node("analyze_uncertainty", analyze_uncertainty)
    workflow.add_node("generate_recommendations", generate_recommendations)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "identify_events")
    workflow.add_edge("identify_events", "estimate_probabilities")
    workflow.add_edge("estimate_probabilities", "analyze_dependencies")
    workflow.add_edge("analyze_dependencies", "calculate_conditionals")
    workflow.add_edge("calculate_conditionals", "assess_risk")
    workflow.add_edge("assess_risk", "calculate_expected_values")
    workflow.add_edge("calculate_expected_values", "analyze_uncertainty")
    workflow.add_edge("analyze_uncertainty", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample probabilistic reasoning task
    initial_state = {
        "scenario": "You are considering launching a new product. Market research suggests 60% chance of high demand, 30% moderate demand, and 10% low demand. Production costs are fixed at $100K. High demand generates $300K revenue, moderate generates $150K, and low generates $50K. There's also a 20% chance a competitor launches a similar product, which would reduce your revenue by 40% in any scenario.",
        "question": "Should you launch the product? What are the risks and expected returns?",
        "events": [],
        "probabilities": {},
        "dependencies": [],
        "conditional_probs": [],
        "risk_assessment": {},
        "expected_values": [],
        "uncertainty_analysis": {},
        "recommendations": [],
        "final_answer": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_probabilistic_reasoning_graph()
    
    print("Probabilistic Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
