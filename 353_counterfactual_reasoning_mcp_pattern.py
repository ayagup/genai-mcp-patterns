"""
Counterfactual Reasoning MCP Pattern

This pattern demonstrates reasoning about alternative scenarios, "what if"
questions, and exploring consequences of different choices or events.

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
class CounterfactualReasoningState(TypedDict):
    """State for counterfactual reasoning workflow"""
    actual_scenario: str
    counterfactual_question: str
    actual_state_analysis: Dict[str, Any]
    counterfactual_conditions: List[Dict[str, Any]]
    alternative_scenarios: List[Dict[str, Any]]
    consequence_predictions: List[Dict[str, Any]]
    causal_differences: List[Dict[str, str]]
    feasibility_assessment: Dict[str, Any]
    counterfactual_narrative: str
    final_reasoning: str
    messages: Annotated[List, operator.add]
    current_step: str


class ActualStateAnalyzer:
    """Analyze the actual/factual state"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_actual_state(self, actual_scenario: str) -> Dict[str, Any]:
        """Analyze the actual state of affairs"""
        prompt = f"""Analyze this actual scenario:

Scenario: {actual_scenario}

Extract and analyze:
1. Key facts and events
2. Actors and their decisions
3. Causal factors leading to current state
4. Outcomes and consequences
5. Critical decision points

Provide JSON:
{{
    "key_facts": ["fact1", "fact2"],
    "actors": [{{"name": "actor", "decision": "what they did", "impact": "result"}}],
    "causal_factors": ["factor1", "factor2"],
    "outcomes": ["outcome1", "outcome2"],
    "decision_points": [{{"point": "when", "decision": "what", "significance": "why important"}}]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at analyzing factual scenarios."),
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
            "key_facts": [],
            "actors": [],
            "causal_factors": [],
            "outcomes": [],
            "decision_points": []
        }


class CounterfactualConditionGenerator:
    """Generate counterfactual conditions"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_conditions(self, actual_analysis: Dict[str, Any],
                           counterfactual_question: str) -> List[Dict[str, Any]]:
        """Generate counterfactual conditions to explore"""
        prompt = f"""Generate counterfactual conditions based on:

Counterfactual Question: {counterfactual_question}

Actual State:
- Key Facts: {json.dumps(actual_analysis.get('key_facts', []))}
- Decision Points: {json.dumps(actual_analysis.get('decision_points', []))}

Generate 3-5 specific counterfactual conditions that would need to change.

Return JSON array:
[
    {{
        "condition_id": 1,
        "what_changes": "what would be different",
        "type": "decision|event|circumstance",
        "magnitude": "small|moderate|large",
        "plausibility": "highly_plausible|plausible|unlikely"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at formulating counterfactual conditions."),
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


class AlternativeScenarioBuilder:
    """Build alternative scenarios"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def build_scenarios(self, actual_scenario: str,
                       actual_analysis: Dict[str, Any],
                       conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build alternative scenarios based on counterfactual conditions"""
        scenarios = []
        
        for condition in conditions:
            scenario = self._build_single_scenario(
                actual_scenario,
                actual_analysis,
                condition
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _build_single_scenario(self, actual_scenario: str,
                              actual_analysis: Dict[str, Any],
                              condition: Dict[str, Any]) -> Dict[str, Any]:
        """Build a single alternative scenario"""
        prompt = f"""Build an alternative scenario:

Actual Scenario: {actual_scenario}

Counterfactual Condition:
{json.dumps(condition, indent=2)}

Actual Outcomes:
{json.dumps(actual_analysis.get('outcomes', []))}

Describe what would have happened if this condition were true.
Include:
1. How events would unfold differently
2. What outcomes would change
3. Ripple effects
4. Comparison to actual scenario

Return JSON:
{{
    "scenario_description": "narrative of alternative events",
    "altered_events": ["event1", "event2"],
    "new_outcomes": ["outcome1", "outcome2"],
    "unchanged_aspects": ["what stays same"],
    "divergence_point": "when scenarios diverge"
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at constructing alternative scenarios."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                scenario_data = json.loads(content[json_start:json_end])
                scenario_data["condition"] = condition
                return scenario_data
            except:
                pass
        
        return {
            "scenario_description": "Could not generate scenario",
            "altered_events": [],
            "new_outcomes": [],
            "unchanged_aspects": [],
            "divergence_point": "unknown",
            "condition": condition
        }


class ConsequencePredictor:
    """Predict consequences of counterfactual scenarios"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def predict_consequences(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict consequences for each scenario"""
        predictions = []
        
        for scenario in scenarios:
            prediction = self._predict_single_scenario(scenario)
            predictions.append(prediction)
        
        return predictions
    
    def _predict_single_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Predict consequences for a single scenario"""
        prompt = f"""Predict consequences of this counterfactual scenario:

Scenario: {scenario.get('scenario_description', 'Unknown')}
New Outcomes: {json.dumps(scenario.get('new_outcomes', []))}

Predict:
1. Short-term consequences (immediate)
2. Medium-term consequences (months/years)
3. Long-term consequences (years/decades)
4. Intended vs unintended consequences
5. Stakeholder impacts

Return JSON:
{{
    "short_term": ["consequence1", "consequence2"],
    "medium_term": ["consequence1", "consequence2"],
    "long_term": ["consequence1", "consequence2"],
    "unintended": ["consequence1", "consequence2"],
    "stakeholder_impacts": [{{"stakeholder": "name", "impact": "description"}}]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at predicting consequences."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                prediction = json.loads(content[json_start:json_end])
                prediction["scenario_id"] = scenario.get("condition", {}).get("condition_id", 0)
                return prediction
            except:
                pass
        
        return {
            "short_term": [],
            "medium_term": [],
            "long_term": [],
            "unintended": [],
            "stakeholder_impacts": [],
            "scenario_id": scenario.get("condition", {}).get("condition_id", 0)
        }


class CausalDifferenceAnalyzer:
    """Analyze causal differences between actual and counterfactual"""
    
    def analyze_differences(self, actual_analysis: Dict[str, Any],
                           scenarios: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Analyze what causes the differences"""
        differences = []
        
        actual_outcomes = set(actual_analysis.get("outcomes", []))
        
        for scenario in scenarios:
            new_outcomes = set(scenario.get("new_outcomes", []))
            
            # Find differences
            only_actual = actual_outcomes - new_outcomes
            only_counterfactual = new_outcomes - actual_outcomes
            
            if only_actual or only_counterfactual:
                differences.append({
                    "scenario_id": scenario.get("condition", {}).get("condition_id", 0),
                    "condition": scenario.get("condition", {}).get("what_changes", "Unknown"),
                    "outcomes_lost": list(only_actual),
                    "outcomes_gained": list(only_counterfactual),
                    "causal_mechanism": self._infer_mechanism(
                        scenario.get("condition", {}),
                        only_actual,
                        only_counterfactual
                    )
                })
        
        return differences
    
    def _infer_mechanism(self, condition: Dict, lost: set, gained: set) -> str:
        """Infer causal mechanism for the difference"""
        change = condition.get("what_changes", "unknown change")
        
        if lost and gained:
            return f"Changing {change} prevented {len(lost)} outcomes and enabled {len(gained)} new outcomes"
        elif lost:
            return f"Changing {change} would have prevented {len(lost)} outcomes"
        elif gained:
            return f"Changing {change} would have enabled {len(gained)} new outcomes"
        else:
            return f"Changing {change} has no clear outcome impact"


class FeasibilityAssessor:
    """Assess feasibility of counterfactual scenarios"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def assess_feasibility(self, conditions: List[Dict[str, Any]],
                          scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess feasibility of counterfactual scenarios"""
        # Aggregate plausibility ratings
        plausibility_counts = {}
        for condition in conditions:
            plaus = condition.get("plausibility", "unknown")
            plausibility_counts[plaus] = plausibility_counts.get(plaus, 0) + 1
        
        # Assess magnitude of changes
        magnitude_counts = {}
        for condition in conditions:
            mag = condition.get("magnitude", "unknown")
            magnitude_counts[mag] = magnitude_counts.get(mag, 0) + 1
        
        # Calculate overall feasibility score
        feasibility_score = self._calculate_feasibility(conditions)
        
        return {
            "overall_feasibility": feasibility_score,
            "plausibility_distribution": plausibility_counts,
            "magnitude_distribution": magnitude_counts,
            "most_feasible": max(conditions, key=lambda c: self._condition_feasibility(c))
            if conditions else None,
            "assessment": self._assess_score(feasibility_score)
        }
    
    def _calculate_feasibility(self, conditions: List[Dict[str, Any]]) -> float:
        """Calculate overall feasibility score"""
        if not conditions:
            return 0.0
        
        scores = [self._condition_feasibility(c) for c in conditions]
        return sum(scores) / len(scores)
    
    def _condition_feasibility(self, condition: Dict[str, Any]) -> float:
        """Calculate feasibility for a single condition"""
        plaus_scores = {
            "highly_plausible": 1.0,
            "plausible": 0.7,
            "unlikely": 0.3
        }
        
        mag_scores = {
            "small": 1.0,
            "moderate": 0.7,
            "large": 0.4
        }
        
        plaus_score = plaus_scores.get(condition.get("plausibility", "unlikely"), 0.5)
        mag_score = mag_scores.get(condition.get("magnitude", "large"), 0.5)
        
        return (plaus_score + mag_score) / 2
    
    def _assess_score(self, score: float) -> str:
        """Assess feasibility based on score"""
        if score >= 0.8:
            return "Highly feasible counterfactual"
        elif score >= 0.6:
            return "Moderately feasible"
        elif score >= 0.4:
            return "Somewhat feasible"
        else:
            return "Low feasibility"


# Agent functions
def initialize_counterfactual(state: CounterfactualReasoningState) -> CounterfactualReasoningState:
    """Initialize counterfactual reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing counterfactual reasoning: {state['counterfactual_question']}"
    ))
    state["counterfactual_conditions"] = []
    state["alternative_scenarios"] = []
    state["current_step"] = "initialized"
    return state


def analyze_actual_state(state: CounterfactualReasoningState) -> CounterfactualReasoningState:
    """Analyze actual/factual state"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = ActualStateAnalyzer(llm)
    
    analysis = analyzer.analyze_actual_state(state["actual_scenario"])
    state["actual_state_analysis"] = analysis
    
    state["messages"].append(HumanMessage(
        content=f"Analyzed actual state: {len(analysis.get('key_facts', []))} facts, "
                f"{len(analysis.get('decision_points', []))} decision points"
    ))
    state["current_step"] = "actual_analyzed"
    return state


def generate_counterfactual_conditions(state: CounterfactualReasoningState) -> CounterfactualReasoningState:
    """Generate counterfactual conditions"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.6)
    generator = CounterfactualConditionGenerator(llm)
    
    conditions = generator.generate_conditions(
        state["actual_state_analysis"],
        state["counterfactual_question"]
    )
    
    state["counterfactual_conditions"] = conditions
    
    state["messages"].append(HumanMessage(
        content=f"Generated {len(conditions)} counterfactual conditions"
    ))
    state["current_step"] = "conditions_generated"
    return state


def build_alternative_scenarios(state: CounterfactualReasoningState) -> CounterfactualReasoningState:
    """Build alternative scenarios"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    builder = AlternativeScenarioBuilder(llm)
    
    scenarios = builder.build_scenarios(
        state["actual_scenario"],
        state["actual_state_analysis"],
        state["counterfactual_conditions"]
    )
    
    state["alternative_scenarios"] = scenarios
    
    state["messages"].append(HumanMessage(
        content=f"Built {len(scenarios)} alternative scenarios"
    ))
    state["current_step"] = "scenarios_built"
    return state


def predict_consequences(state: CounterfactualReasoningState) -> CounterfactualReasoningState:
    """Predict consequences of scenarios"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.6)
    predictor = ConsequencePredictor(llm)
    
    predictions = predictor.predict_consequences(state["alternative_scenarios"])
    state["consequence_predictions"] = predictions
    
    total_consequences = sum(
        len(p.get("short_term", [])) + len(p.get("medium_term", [])) + len(p.get("long_term", []))
        for p in predictions
    )
    
    state["messages"].append(HumanMessage(
        content=f"Predicted {total_consequences} total consequences across {len(predictions)} scenarios"
    ))
    state["current_step"] = "consequences_predicted"
    return state


def analyze_causal_differences(state: CounterfactualReasoningState) -> CounterfactualReasoningState:
    """Analyze causal differences"""
    analyzer = CausalDifferenceAnalyzer()
    
    differences = analyzer.analyze_differences(
        state["actual_state_analysis"],
        state["alternative_scenarios"]
    )
    
    state["causal_differences"] = differences
    
    state["messages"].append(HumanMessage(
        content=f"Analyzed {len(differences)} causal differences"
    ))
    state["current_step"] = "differences_analyzed"
    return state


def assess_feasibility(state: CounterfactualReasoningState) -> CounterfactualReasoningState:
    """Assess feasibility of counterfactuals"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    assessor = FeasibilityAssessor(llm)
    
    assessment = assessor.assess_feasibility(
        state["counterfactual_conditions"],
        state["alternative_scenarios"]
    )
    
    state["feasibility_assessment"] = assessment
    
    state["messages"].append(HumanMessage(
        content=f"Feasibility assessment: {assessment['overall_feasibility']:.2f} - "
                f"{assessment['assessment']}"
    ))
    state["current_step"] = "feasibility_assessed"
    return state


def synthesize_counterfactual_narrative(state: CounterfactualReasoningState) -> CounterfactualReasoningState:
    """Synthesize counterfactual narrative"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.6)
    
    prompt = f"""Create a comprehensive counterfactual narrative:

Question: {state['counterfactual_question']}

Actual Scenario:
{state['actual_scenario']}

Alternative Scenarios Explored: {len(state['alternative_scenarios'])}

Best insights from scenarios and consequences.

Create a narrative that:
1. Addresses the counterfactual question
2. Explains what would have happened differently
3. Discusses consequences
4. Compares to actual outcomes
5. Assesses feasibility and plausibility"""
    
    messages = [
        SystemMessage(content="You are an expert at constructing counterfactual narratives."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    state["counterfactual_narrative"] = response.content
    
    state["messages"].append(HumanMessage(
        content=f"Synthesized counterfactual narrative ({len(response.content.split())} words)"
    ))
    state["current_step"] = "narrative_synthesized"
    return state


def generate_report(state: CounterfactualReasoningState) -> CounterfactualReasoningState:
    """Generate final report"""
    report = f"""
COUNTERFACTUAL REASONING REPORT
================================

Counterfactual Question: {state['counterfactual_question']}

Actual Scenario:
{state['actual_scenario']}

Actual State Analysis:
- Key Facts: {len(state['actual_state_analysis'].get('key_facts', []))}
- Decision Points: {len(state['actual_state_analysis'].get('decision_points', []))}
- Actual Outcomes: {len(state['actual_state_analysis'].get('outcomes', []))}

Counterfactual Conditions ({len(state['counterfactual_conditions'])}):
"""
    
    for i, condition in enumerate(state['counterfactual_conditions'], 1):
        report += f"{i}. {condition.get('what_changes', 'Unknown')}\n"
        report += f"   - Type: {condition.get('type', 'unknown')}\n"
        report += f"   - Magnitude: {condition.get('magnitude', 'unknown')}\n"
        report += f"   - Plausibility: {condition.get('plausibility', 'unknown')}\n"
    
    report += f"""
Alternative Scenarios Explored: {len(state['alternative_scenarios'])}

Consequence Predictions:
"""
    
    for prediction in state['consequence_predictions']:
        report += f"\nScenario {prediction.get('scenario_id', 0)}:\n"
        report += f"- Short-term: {len(prediction.get('short_term', []))} consequences\n"
        report += f"- Medium-term: {len(prediction.get('medium_term', []))} consequences\n"
        report += f"- Long-term: {len(prediction.get('long_term', []))} consequences\n"
        report += f"- Unintended: {len(prediction.get('unintended', []))} consequences\n"
    
    report += f"""
Causal Differences:
"""
    
    for diff in state['causal_differences']:
        report += f"\n{diff.get('condition', 'Unknown')}:\n"
        report += f"- Outcomes Lost: {len(diff.get('outcomes_lost', []))}\n"
        report += f"- Outcomes Gained: {len(diff.get('outcomes_gained', []))}\n"
        report += f"- Mechanism: {diff.get('causal_mechanism', 'Unknown')}\n"
    
    report += f"""
Feasibility Assessment:
- Overall Score: {state['feasibility_assessment']['overall_feasibility']:.2f}
- Assessment: {state['feasibility_assessment']['assessment']}
- Plausibility: {state['feasibility_assessment']['plausibility_distribution']}
- Magnitude: {state['feasibility_assessment']['magnitude_distribution']}

COUNTERFACTUAL NARRATIVE:
{'-' * 50}
{state['counterfactual_narrative']}
{'-' * 50}

Reasoning Summary:
- Counterfactual Conditions: {len(state['counterfactual_conditions'])}
- Alternative Scenarios: {len(state['alternative_scenarios'])}
- Total Consequences: {sum(len(p.get('short_term', [])) + len(p.get('medium_term', [])) + len(p.get('long_term', [])) for p in state['consequence_predictions'])}
- Feasibility: {state['feasibility_assessment']['overall_feasibility']:.2%}
"""
    
    state["final_reasoning"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_counterfactual_reasoning_graph():
    """Create the counterfactual reasoning workflow graph"""
    workflow = StateGraph(CounterfactualReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_counterfactual)
    workflow.add_node("analyze_actual", analyze_actual_state)
    workflow.add_node("generate_conditions", generate_counterfactual_conditions)
    workflow.add_node("build_scenarios", build_alternative_scenarios)
    workflow.add_node("predict_consequences", predict_consequences)
    workflow.add_node("analyze_differences", analyze_causal_differences)
    workflow.add_node("assess_feasibility", assess_feasibility)
    workflow.add_node("synthesize_narrative", synthesize_counterfactual_narrative)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze_actual")
    workflow.add_edge("analyze_actual", "generate_conditions")
    workflow.add_edge("generate_conditions", "build_scenarios")
    workflow.add_edge("build_scenarios", "predict_consequences")
    workflow.add_edge("predict_consequences", "analyze_differences")
    workflow.add_edge("analyze_differences", "assess_feasibility")
    workflow.add_edge("assess_feasibility", "synthesize_narrative")
    workflow.add_edge("synthesize_narrative", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample counterfactual reasoning task
    initial_state = {
        "actual_scenario": "Company X launched a new product without conducting user research, resulting in poor adoption and eventual discontinuation after 6 months",
        "counterfactual_question": "What if Company X had conducted thorough user research before launching the product?",
        "actual_state_analysis": {},
        "counterfactual_conditions": [],
        "alternative_scenarios": [],
        "consequence_predictions": [],
        "causal_differences": [],
        "feasibility_assessment": {},
        "counterfactual_narrative": "",
        "final_reasoning": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_counterfactual_reasoning_graph()
    
    print("Counterfactual Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Feasibility: {result['feasibility_assessment']['overall_feasibility']:.2%}")
