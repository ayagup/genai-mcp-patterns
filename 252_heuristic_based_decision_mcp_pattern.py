"""
Pattern 252: Heuristic-Based Decision MCP Pattern

This pattern demonstrates heuristic-based decision making - using practical
rules of thumb and shortcuts to make quick, reasonable decisions.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import random


class HeuristicBasedDecisionState(TypedDict):
    """State for heuristic-based decision workflow"""
    messages: Annotated[List[str], add]
    problem: Dict[str, Any]
    heuristics_applied: List[Dict[str, Any]]
    decision: str
    confidence: float


class Heuristic:
    """Represents a decision heuristic"""
    
    def __init__(self, name: str, evaluate_func, weight: float = 1.0):
        self.name = name
        self.evaluate = evaluate_func
        self.weight = weight


class HeuristicEngine:
    """Heuristic-based decision engine"""
    
    def __init__(self):
        self.heuristics = self._setup_heuristics()
    
    def _setup_heuristics(self) -> List[Heuristic]:
        """Setup common heuristics"""
        return [
            Heuristic(
                "Availability Heuristic",
                lambda p: {"score": 0.8 if p.get("recent_occurrences", 0) > 5 else 0.3,
                          "reason": "Recent occurrences make it seem more likely"},
                weight=0.7
            ),
            Heuristic(
                "Representativeness Heuristic",
                lambda p: {"score": 0.9 if p.get("matches_pattern", False) else 0.4,
                          "reason": "Matches expected pattern"},
                weight=0.8
            ),
            Heuristic(
                "Anchoring Heuristic",
                lambda p: {"score": 0.7 if p.get("initial_estimate", 0) > 50 else 0.5,
                          "reason": "Anchored to initial estimate"},
                weight=0.6
            ),
            Heuristic(
                "Satisficing Heuristic",
                lambda p: {"score": 0.85 if p.get("meets_threshold", False) else 0.2,
                          "reason": "Good enough solution found"},
                weight=0.9
            ),
            Heuristic(
                "Recognition Heuristic",
                lambda p: {"score": 0.75 if p.get("is_familiar", False) else 0.3,
                          "reason": "Familiarity breeds confidence"},
                weight=0.7
            )
        ]
    
    def decide(self, problem: Dict[str, Any]) -> tuple:
        """Make decision using heuristics"""
        results = []
        total_score = 0
        total_weight = 0
        
        for heuristic in self.heuristics:
            result = heuristic.evaluate(problem)
            weighted_score = result["score"] * heuristic.weight
            
            results.append({
                "heuristic": heuristic.name,
                "score": result["score"],
                "weight": heuristic.weight,
                "weighted_score": weighted_score,
                "reason": result["reason"]
            })
            
            total_score += weighted_score
            total_weight += heuristic.weight
        
        # Calculate overall confidence
        confidence = total_score / total_weight if total_weight > 0 else 0
        
        # Make decision based on confidence
        if confidence >= 0.7:
            decision = "PROCEED_WITH_HIGH_CONFIDENCE"
        elif confidence >= 0.5:
            decision = "PROCEED_WITH_CAUTION"
        else:
            decision = "SEEK_MORE_INFORMATION"
        
        return decision, confidence, results


def receive_problem_agent(state: HeuristicBasedDecisionState) -> HeuristicBasedDecisionState:
    """Receive problem for heuristic analysis"""
    print("\n📥 Receiving Problem...")
    
    problem = {
        "type": "investment_decision",
        "recent_occurrences": 8,
        "matches_pattern": True,
        "initial_estimate": 65,
        "meets_threshold": True,
        "is_familiar": True,
        "details": "Evaluate new market opportunity"
    }
    
    print(f"\n  Problem Type: {problem['type']}")
    print(f"  Details: {problem['details']}")
    print(f"\n  Problem Attributes:")
    for key, value in problem.items():
        if key not in ["type", "details"]:
            print(f"    {key}: {value}")
    
    return {
        **state,
        "problem": problem,
        "messages": ["✓ Problem received"]
    }


def apply_heuristics_agent(state: HeuristicBasedDecisionState) -> HeuristicBasedDecisionState:
    """Apply heuristics to make decision"""
    print("\n🧠 Applying Heuristics...")
    
    engine = HeuristicEngine()
    decision, confidence, heuristics_applied = engine.decide(state["problem"])
    
    print(f"\n  Heuristics Evaluated:")
    for h in heuristics_applied:
        print(f"\n    • {h['heuristic']}")
        print(f"      Score: {h['score']:.2f} | Weight: {h['weight']:.1f} | Weighted: {h['weighted_score']:.2f}")
        print(f"      Reason: {h['reason']}")
    
    print(f"\n  Overall Confidence: {confidence:.2%}")
    print(f"  Decision: {decision}")
    
    return {
        **state,
        "heuristics_applied": heuristics_applied,
        "decision": decision,
        "confidence": confidence,
        "messages": [f"✓ Decision: {decision} ({confidence:.0%} confidence)"]
    }


def generate_heuristic_report_agent(state: HeuristicBasedDecisionState) -> HeuristicBasedDecisionState:
    """Generate heuristic decision report"""
    print("\n" + "="*70)
    print("HEURISTIC-BASED DECISION REPORT")
    print("="*70)
    
    print(f"\n📊 Problem Analysis:")
    print(f"  Type: {state['problem']['type']}")
    print(f"  Details: {state['problem']['details']}")
    
    print(f"\n🧠 Heuristics Applied ({len(state['heuristics_applied'])}):")
    for h in state["heuristics_applied"]:
        print(f"\n  • {h['heuristic']}")
        print(f"    Raw Score: {h['score']:.2f}")
        print(f"    Weight: {h['weight']:.1f}")
        print(f"    Weighted Score: {h['weighted_score']:.2f}")
        print(f"    Rationale: {h['reason']}")
    
    print(f"\n✅ Final Decision:")
    print(f"  Decision: {state['decision']}")
    print(f"  Confidence: {state['confidence']:.1%}")
    
    # Categorize confidence level
    if state['confidence'] >= 0.7:
        level = "HIGH - Strong heuristic support"
    elif state['confidence'] >= 0.5:
        level = "MEDIUM - Moderate heuristic support"
    else:
        level = "LOW - Weak heuristic support"
    
    print(f"  Confidence Level: {level}")
    
    print("\n💡 Heuristic Decision Benefits:")
    print("  • Fast decision-making")
    print("  • Works with incomplete information")
    print("  • Leverages past experience")
    print("  • Practical and efficient")
    print("  • Good for time-sensitive decisions")
    
    print("\n⚠️ Considerations:")
    print("  • May have cognitive biases")
    print("  • Not always optimal")
    print("  • Context-dependent accuracy")
    print("  • Should be validated when possible")
    
    print("\n="*70)
    print("✅ Heuristic-Based Decision Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_heuristic_based_decision_graph():
    workflow = StateGraph(HeuristicBasedDecisionState)
    workflow.add_node("receive", receive_problem_agent)
    workflow.add_node("apply_heuristics", apply_heuristics_agent)
    workflow.add_node("report", generate_heuristic_report_agent)
    workflow.add_edge(START, "receive")
    workflow.add_edge("receive", "apply_heuristics")
    workflow.add_edge("apply_heuristics", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 252: Heuristic-Based Decision MCP Pattern")
    print("="*70)
    
    app = create_heuristic_based_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "problem": {},
        "heuristics_applied": [],
        "decision": "",
        "confidence": 0.0
    })
    print("\n✅ Heuristic-Based Decision Pattern Complete!")


if __name__ == "__main__":
    main()
