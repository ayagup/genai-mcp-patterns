"""
Pattern 253: Probabilistic Decision MCP Pattern

This pattern demonstrates probabilistic decision making - making decisions
based on probability distributions and likelihood calculations.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import random


class ProbabilisticDecisionState(TypedDict):
    """State for probabilistic decision workflow"""
    messages: Annotated[List[str], add]
    scenarios: List[Dict[str, Any]]
    probabilities: Dict[str, float]
    expected_values: Dict[str, float]
    decision: str


class ProbabilisticDecisionMaker:
    """Makes decisions based on probabilities"""
    
    def calculate_probability(self, scenario: Dict[str, Any]) -> float:
        """Calculate probability of scenario"""
        # Simplified Bayesian calculation
        prior = scenario.get("prior_probability", 0.5)
        evidence_strength = scenario.get("evidence_strength", 0.5)
        
        # Bayes' theorem simplified
        likelihood = evidence_strength
        probability = (likelihood * prior) / ((likelihood * prior) + (1 - likelihood) * (1 - prior))
        
        return round(probability, 3)
    
    def calculate_expected_value(self, scenario: Dict[str, Any], probability: float) -> float:
        """Calculate expected value"""
        value = scenario.get("value", 0)
        cost = scenario.get("cost", 0)
        
        expected_value = (value - cost) * probability
        return round(expected_value, 2)
    
    def decide(self, scenarios: List[Dict[str, Any]]) -> tuple:
        """Make decision based on expected values"""
        probabilities = {}
        expected_values = {}
        
        for scenario in scenarios:
            name = scenario["name"]
            prob = self.calculate_probability(scenario)
            ev = self.calculate_expected_value(scenario, prob)
            
            probabilities[name] = prob
            expected_values[name] = ev
        
        # Select scenario with highest expected value
        best_scenario = max(expected_values, key=expected_values.get)
        
        return best_scenario, probabilities, expected_values


def define_scenarios_agent(state: ProbabilisticDecisionState) -> ProbabilisticDecisionState:
    """Define decision scenarios"""
    print("\n📊 Defining Scenarios...")
    
    scenarios = [
        {
            "name": "Aggressive Expansion",
            "prior_probability": 0.4,
            "evidence_strength": 0.7,
            "value": 1000000,
            "cost": 300000,
            "description": "Rapid market expansion with high risk/reward"
        },
        {
            "name": "Conservative Growth",
            "prior_probability": 0.7,
            "evidence_strength": 0.8,
            "value": 500000,
            "cost": 100000,
            "description": "Steady, low-risk growth strategy"
        },
        {
            "name": "Innovation Focus",
            "prior_probability": 0.5,
            "evidence_strength": 0.6,
            "value": 800000,
            "cost": 250000,
            "description": "Invest heavily in R&D and innovation"
        },
        {
            "name": "Market Consolidation",
            "prior_probability": 0.6,
            "evidence_strength": 0.75,
            "value": 600000,
            "cost": 150000,
            "description": "Strengthen position in existing markets"
        }
    ]
    
    print(f"\n  Total Scenarios: {len(scenarios)}")
    for scenario in scenarios:
        print(f"\n    • {scenario['name']}")
        print(f"      {scenario['description']}")
        print(f"      Value: ${scenario['value']:,} | Cost: ${scenario['cost']:,}")
    
    return {
        **state,
        "scenarios": scenarios,
        "messages": [f"✓ Defined {len(scenarios)} scenarios"]
    }


def calculate_probabilities_agent(state: ProbabilisticDecisionState) -> ProbabilisticDecisionState:
    """Calculate probabilities and expected values"""
    print("\n🎲 Calculating Probabilities...")
    
    decision_maker = ProbabilisticDecisionMaker()
    decision, probabilities, expected_values = decision_maker.decide(state["scenarios"])
    
    print(f"\n  Scenario Analysis:")
    for scenario in state["scenarios"]:
        name = scenario["name"]
        prob = probabilities[name]
        ev = expected_values[name]
        
        print(f"\n    • {name}")
        print(f"      Probability: {prob:.1%}")
        print(f"      Expected Value: ${ev:,.2f}")
    
    print(f"\n  Best Option: {decision}")
    print(f"  Expected Value: ${expected_values[decision]:,.2f}")
    
    return {
        **state,
        "probabilities": probabilities,
        "expected_values": expected_values,
        "decision": decision,
        "messages": [f"✓ Selected: {decision}"]
    }


def generate_probabilistic_report_agent(state: ProbabilisticDecisionState) -> ProbabilisticDecisionState:
    """Generate probabilistic decision report"""
    print("\n" + "="*70)
    print("PROBABILISTIC DECISION REPORT")
    print("="*70)
    
    print(f"\n📊 Scenarios Evaluated: {len(state['scenarios'])}")
    
    print(f"\n🎲 Probability Analysis:")
    for scenario in state["scenarios"]:
        name = scenario["name"]
        print(f"\n  {name}:")
        print(f"    Prior Probability: {scenario['prior_probability']:.1%}")
        print(f"    Evidence Strength: {scenario['evidence_strength']:.1%}")
        print(f"    Calculated Probability: {state['probabilities'][name]:.1%}")
        print(f"    Expected Value: ${state['expected_values'][name]:,.2f}")
    
    print(f"\n✅ Decision:")
    print(f"  Selected Strategy: {state['decision']}")
    print(f"  Probability of Success: {state['probabilities'][state['decision']]:.1%}")
    print(f"  Expected Value: ${state['expected_values'][state['decision']]:,.2f}")
    
    # Rank all scenarios
    ranked = sorted(state['expected_values'].items(), key=lambda x: x[1], reverse=True)
    print(f"\n📈 Ranking by Expected Value:")
    for i, (name, ev) in enumerate(ranked, 1):
        print(f"  {i}. {name}: ${ev:,.2f} (P={state['probabilities'][name]:.1%})")
    
    print("\n💡 Probabilistic Decision Benefits:")
    print("  • Quantifies uncertainty")
    print("  • Data-driven decisions")
    print("  • Considers likelihood and impact")
    print("  • Enables risk assessment")
    print("  • Maximizes expected value")
    
    print("\n="*70)
    print("✅ Probabilistic Decision Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_probabilistic_decision_graph():
    workflow = StateGraph(ProbabilisticDecisionState)
    workflow.add_node("define", define_scenarios_agent)
    workflow.add_node("calculate", calculate_probabilities_agent)
    workflow.add_node("report", generate_probabilistic_report_agent)
    workflow.add_edge(START, "define")
    workflow.add_edge("define", "calculate")
    workflow.add_edge("calculate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 253: Probabilistic Decision MCP Pattern")
    print("="*70)
    
    app = create_probabilistic_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "scenarios": [],
        "probabilities": {},
        "expected_values": {},
        "decision": ""
    })
    print("\n✅ Probabilistic Decision Pattern Complete!")


if __name__ == "__main__":
    main()
