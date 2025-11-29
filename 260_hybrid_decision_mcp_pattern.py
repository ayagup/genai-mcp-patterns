"""
Pattern 260: Hybrid Decision MCP Pattern

This pattern demonstrates hybrid decision making - combining multiple
decision paradigms (rule-based, probabilistic, ML, heuristic) for
comprehensive decision making.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class HybridDecisionState(TypedDict):
    """State for hybrid decision workflow"""
    messages: Annotated[List[str], add]
    problem: Dict[str, Any]
    rule_decision: Dict[str, Any]
    probabilistic_decision: Dict[str, Any]
    ml_decision: Dict[str, Any]
    heuristic_decision: Dict[str, Any]
    hybrid_decision: Dict[str, Any]


class RuleBasedDecider:
    """Rule-based decision component"""
    
    def decide(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        amount = problem.get("amount", 0)
        
        if amount > 100000:
            return {
                "method": "Rule-Based",
                "decision": "REJECT",
                "confidence": 0.95,
                "reason": "Amount exceeds maximum threshold ($100k)"
            }
        elif amount < 1000:
            return {
                "method": "Rule-Based",
                "decision": "APPROVE",
                "confidence": 0.90,
                "reason": "Amount below minimum threshold ($1k)"
            }
        else:
            return {
                "method": "Rule-Based",
                "decision": "REVIEW",
                "confidence": 0.70,
                "reason": "Amount within review range"
            }


class ProbabilisticDecider:
    """Probabilistic decision component"""
    
    def decide(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        credit_score = problem.get("credit_score", 650)
        
        # Simplified probability calculation
        approval_prob = (credit_score - 300) / 550  # Scale 300-850 to 0-1
        approval_prob = max(0, min(1, approval_prob))
        
        if approval_prob > 0.75:
            decision = "APPROVE"
            confidence = approval_prob
        elif approval_prob < 0.40:
            decision = "REJECT"
            confidence = 1 - approval_prob
        else:
            decision = "REVIEW"
            confidence = 0.60
        
        return {
            "method": "Probabilistic",
            "decision": decision,
            "confidence": confidence,
            "reason": f"Approval probability: {approval_prob:.2%}"
        }


class MLDecider:
    """Machine learning decision component"""
    
    def decide(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Simplified ML model simulation
        features = {
            "income": problem.get("income", 50000) / 200000,  # Normalize
            "debt_ratio": 1 - problem.get("debt_ratio", 0.4),  # Invert
            "employment_years": min(problem.get("employment_years", 0) / 10, 1)
        }
        
        # Weighted sum as ML prediction
        score = (features["income"] * 0.4 + 
                features["debt_ratio"] * 0.35 + 
                features["employment_years"] * 0.25)
        
        if score > 0.70:
            decision = "APPROVE"
            confidence = score
        elif score < 0.45:
            decision = "REJECT"
            confidence = 1 - score
        else:
            decision = "REVIEW"
            confidence = 0.65
        
        return {
            "method": "ML Model",
            "decision": decision,
            "confidence": confidence,
            "reason": f"Model score: {score:.2f}"
        }


class HeuristicDecider:
    """Heuristic-based decision component"""
    
    def decide(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Use simple heuristics
        has_collateral = problem.get("has_collateral", False)
        history_months = problem.get("history_months", 0)
        
        if has_collateral and history_months > 24:
            decision = "APPROVE"
            confidence = 0.85
            reason = "Has collateral and long history"
        elif not has_collateral and history_months < 6:
            decision = "REJECT"
            confidence = 0.75
            reason = "No collateral and short history"
        else:
            decision = "REVIEW"
            confidence = 0.60
            reason = "Mixed indicators"
        
        return {
            "method": "Heuristic",
            "decision": decision,
            "confidence": confidence,
            "reason": reason
        }


class HybridDecisionMaker:
    """Combines multiple decision methods"""
    
    def __init__(self):
        self.rule_based = RuleBasedDecider()
        self.probabilistic = ProbabilisticDecider()
        self.ml_model = MLDecider()
        self.heuristic = HeuristicDecider()
    
    def combine_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine decisions from multiple methods"""
        
        # Weighted voting based on confidence
        votes = {}
        total_confidence = 0
        
        for decision_result in decisions:
            decision = decision_result["decision"]
            confidence = decision_result["confidence"]
            
            votes[decision] = votes.get(decision, 0) + confidence
            total_confidence += confidence
        
        # Select decision with highest weighted vote
        final_decision = max(votes, key=votes.get)
        final_confidence = votes[final_decision] / total_confidence if total_confidence > 0 else 0
        
        # Count agreement
        agreement_count = sum(1 for d in decisions if d["decision"] == final_decision)
        agreement_pct = agreement_count / len(decisions)
        
        return {
            "decision": final_decision,
            "confidence": final_confidence,
            "agreement": agreement_pct,
            "votes": votes,
            "methods_agreeing": agreement_count,
            "total_methods": len(decisions)
        }


def define_problem_agent(state: HybridDecisionState) -> HybridDecisionState:
    """Define problem for hybrid decision"""
    print("\n📋 Defining Problem...")
    
    problem = {
        "type": "loan_application",
        "amount": 45000,
        "credit_score": 720,
        "income": 85000,
        "debt_ratio": 0.35,
        "employment_years": 5,
        "has_collateral": True,
        "history_months": 36
    }
    
    print(f"\n  Problem Details:")
    for key, value in problem.items():
        print(f"    {key}: {value}")
    
    return {
        **state,
        "problem": problem,
        "messages": ["✓ Problem defined"]
    }


def apply_decision_methods_agent(state: HybridDecisionState) -> HybridDecisionState:
    """Apply all decision methods"""
    print("\n🔍 Applying Decision Methods...")
    
    maker = HybridDecisionMaker()
    
    # Get decisions from each method
    rule_decision = maker.rule_based.decide(state["problem"])
    prob_decision = maker.probabilistic.decide(state["problem"])
    ml_decision = maker.ml_model.decide(state["problem"])
    heur_decision = maker.heuristic.decide(state["problem"])
    
    print(f"\n  Method Results:")
    for decision in [rule_decision, prob_decision, ml_decision, heur_decision]:
        print(f"\n    {decision['method']}:")
        print(f"      Decision: {decision['decision']}")
        print(f"      Confidence: {decision['confidence']:.0%}")
        print(f"      Reason: {decision['reason']}")
    
    return {
        **state,
        "rule_decision": rule_decision,
        "probabilistic_decision": prob_decision,
        "ml_decision": ml_decision,
        "heuristic_decision": heur_decision,
        "messages": ["✓ All methods applied"]
    }


def combine_decisions_agent(state: HybridDecisionState) -> HybridDecisionState:
    """Combine decisions into hybrid decision"""
    print("\n🔄 Combining Decisions...")
    
    maker = HybridDecisionMaker()
    decisions = [
        state["rule_decision"],
        state["probabilistic_decision"],
        state["ml_decision"],
        state["heuristic_decision"]
    ]
    
    hybrid = maker.combine_decisions(decisions)
    
    print(f"\n  Hybrid Decision:")
    print(f"    Final Decision: {hybrid['decision']}")
    print(f"    Confidence: {hybrid['confidence']:.0%}")
    print(f"    Agreement: {hybrid['methods_agreeing']}/{hybrid['total_methods']} methods")
    print(f"    Agreement Rate: {hybrid['agreement']:.0%}")
    
    return {
        **state,
        "hybrid_decision": hybrid,
        "messages": [f"✓ Hybrid decision: {hybrid['decision']}"]
    }


def generate_hybrid_report_agent(state: HybridDecisionState) -> HybridDecisionState:
    """Generate hybrid decision report"""
    print("\n" + "="*70)
    print("HYBRID DECISION REPORT")
    print("="*70)
    
    print(f"\n📊 Problem:")
    for key, value in state["problem"].items():
        print(f"  {key}: {value}")
    
    print(f"\n🔍 Individual Method Results:")
    
    methods = [
        ("Rule-Based", state["rule_decision"]),
        ("Probabilistic", state["probabilistic_decision"]),
        ("ML Model", state["ml_decision"]),
        ("Heuristic", state["heuristic_decision"])
    ]
    
    for name, decision in methods:
        print(f"\n  {name}:")
        print(f"    Decision: {decision['decision']}")
        print(f"    Confidence: {decision['confidence']:.0%}")
        print(f"    Reason: {decision['reason']}")
    
    print(f"\n🔄 Hybrid Decision Synthesis:")
    hybrid = state["hybrid_decision"]
    print(f"  Final Decision: {hybrid['decision']}")
    print(f"  Confidence: {hybrid['confidence']:.0%}")
    print(f"  Agreement: {hybrid['methods_agreeing']}/{hybrid['total_methods']} methods ({hybrid['agreement']:.0%})")
    
    print(f"\n  Vote Distribution:")
    for decision, weight in sorted(hybrid['votes'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {decision}: {weight:.2f}")
    
    print("\n💡 Hybrid Decision Benefits:")
    print("  • Combines multiple perspectives")
    print("  • Leverages different paradigms")
    print("  • More robust than single method")
    print("  • Reduces individual method bias")
    print("  • Captures diverse insights")
    print("  • Better handles edge cases")
    print("  • Increased confidence")
    
    print(f"\n🎯 Decision Rationale:")
    if hybrid['agreement'] >= 0.75:
        print(f"  ✅ Strong consensus ({hybrid['agreement']:.0%} agreement)")
        print(f"  High confidence in {hybrid['decision']} decision")
    elif hybrid['agreement'] >= 0.50:
        print(f"  ⚠️ Moderate consensus ({hybrid['agreement']:.0%} agreement)")
        print(f"  Recommend additional review")
    else:
        print(f"  ❌ Low consensus ({hybrid['agreement']:.0%} agreement)")
        print(f"  Recommend human decision")
    
    print("\n="*70)
    print("✅ Hybrid Decision Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_hybrid_decision_graph():
    workflow = StateGraph(HybridDecisionState)
    workflow.add_node("define", define_problem_agent)
    workflow.add_node("apply", apply_decision_methods_agent)
    workflow.add_node("combine", combine_decisions_agent)
    workflow.add_node("report", generate_hybrid_report_agent)
    workflow.add_edge(START, "define")
    workflow.add_edge("define", "apply")
    workflow.add_edge("apply", "combine")
    workflow.add_edge("combine", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 260: Hybrid Decision MCP Pattern")
    print("="*70)
    
    app = create_hybrid_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "problem": {},
        "rule_decision": {},
        "probabilistic_decision": {},
        "ml_decision": {},
        "heuristic_decision": {},
        "hybrid_decision": {}
    })
    print("\n✅ Hybrid Decision Pattern Complete!")


if __name__ == "__main__":
    main()
