"""
Pattern 258: Weighted Decision MCP Pattern

This pattern demonstrates weighted decision making - assigning importance
weights to different factors and computing weighted scores for decisions.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class WeightedDecisionState(TypedDict):
    """State for weighted decision workflow"""
    messages: Annotated[List[str], add]
    factors: Dict[str, float]
    candidates: List[Dict[str, Any]]
    weighted_scores: List[Dict[str, Any]]
    selected_candidate: Dict[str, Any]


class WeightedDecisionMaker:
    """Makes decisions based on weighted factors"""
    
    def __init__(self, factors: Dict[str, float]):
        """
        Args:
            factors: Dict of factor_name -> weight (weights should sum to 1.0)
        """
        self.factors = factors
        # Normalize weights to sum to 1.0
        total_weight = sum(factors.values())
        self.normalized_factors = {k: v/total_weight for k, v in factors.items()}
    
    def evaluate_candidate(self, candidate: Dict[str, Any]) -> float:
        """Calculate weighted score for a candidate"""
        scores = candidate.get("scores", {})
        weighted_score = 0.0
        
        details = []
        for factor, weight in self.normalized_factors.items():
            score = scores.get(factor, 0)
            contribution = score * weight
            weighted_score += contribution
            details.append({
                "factor": factor,
                "score": score,
                "weight": weight,
                "contribution": contribution
            })
        
        return weighted_score, details
    
    def rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank all candidates by weighted score"""
        results = []
        
        for candidate in candidates:
            weighted_score, details = self.evaluate_candidate(candidate)
            results.append({
                "candidate": candidate,
                "weighted_score": weighted_score,
                "details": details
            })
        
        # Sort by weighted score (descending)
        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        return results


def define_decision_factors_agent(state: WeightedDecisionState) -> WeightedDecisionState:
    """Define decision factors and weights"""
    print("\n📊 Defining Decision Factors...")
    
    factors = {
        "technical_skills": 0.35,    # 35% weight
        "experience": 0.25,          # 25% weight
        "culture_fit": 0.20,         # 20% weight
        "communication": 0.15,       # 15% weight
        "leadership": 0.05           # 5% weight
    }
    
    print("\n  Factors and Weights:")
    for factor, weight in factors.items():
        print(f"    • {factor}: {weight:.0%}")
    
    total = sum(factors.values())
    print(f"\n  Total Weight: {total:.2f} (should be 1.0)")
    
    return {
        **state,
        "factors": factors,
        "messages": ["✓ Decision factors defined"]
    }


def evaluate_candidates_agent(state: WeightedDecisionState) -> WeightedDecisionState:
    """Evaluate candidates using weighted scoring"""
    print("\n👥 Evaluating Candidates...")
    
    # Sample candidates for job position
    candidates = [
        {
            "name": "Alice Johnson",
            "scores": {
                "technical_skills": 0.92,
                "experience": 0.78,
                "culture_fit": 0.85,
                "communication": 0.88,
                "leadership": 0.70
            }
        },
        {
            "name": "Bob Smith",
            "scores": {
                "technical_skills": 0.88,
                "experience": 0.95,
                "culture_fit": 0.75,
                "communication": 0.82,
                "leadership": 0.85
            }
        },
        {
            "name": "Carol Davis",
            "scores": {
                "technical_skills": 0.95,
                "experience": 0.65,
                "culture_fit": 0.90,
                "communication": 0.92,
                "leadership": 0.68
            }
        },
        {
            "name": "David Wilson",
            "scores": {
                "technical_skills": 0.82,
                "experience": 0.88,
                "culture_fit": 0.92,
                "communication": 0.85,
                "leadership": 0.78
            }
        }
    ]
    
    decision_maker = WeightedDecisionMaker(state["factors"])
    weighted_scores = decision_maker.rank_candidates(candidates)
    
    print(f"\n  Candidates Evaluated: {len(candidates)}")
    for i, result in enumerate(weighted_scores, 1):
        candidate = result["candidate"]
        print(f"\n  {i}. {candidate['name']}")
        print(f"     Weighted Score: {result['weighted_score']:.3f}")
    
    return {
        **state,
        "candidates": candidates,
        "weighted_scores": weighted_scores,
        "selected_candidate": weighted_scores[0]["candidate"] if weighted_scores else {},
        "messages": [f"✓ {len(candidates)} candidates evaluated"]
    }


def generate_weighted_report_agent(state: WeightedDecisionState) -> WeightedDecisionState:
    """Generate weighted decision report"""
    print("\n" + "="*70)
    print("WEIGHTED DECISION REPORT")
    print("="*70)
    
    print(f"\n🎯 Decision Factors and Weights:")
    for factor, weight in state["factors"].items():
        print(f"  {factor.replace('_', ' ').title()}: {weight:.0%}")
    
    print(f"\n📊 Candidate Rankings:")
    for i, result in enumerate(state["weighted_scores"], 1):
        candidate = result["candidate"]
        print(f"\n{i}. {candidate['name']} - Score: {result['weighted_score']:.3f}")
        print(f"   Factor Breakdown:")
        for detail in result["details"]:
            print(f"     • {detail['factor'].replace('_', ' ').title()}: "
                  f"{detail['score']:.2f} × {detail['weight']:.0%} = {detail['contribution']:.3f}")
    
    print(f"\n✅ Selected Candidate:")
    selected = state["selected_candidate"]
    print(f"  Name: {selected['name']}")
    print(f"  Final Score: {state['weighted_scores'][0]['weighted_score']:.3f}")
    
    # Show why they were selected
    top_result = state['weighted_scores'][0]
    print(f"\n💡 Selection Rationale:")
    top_contributions = sorted(top_result['details'], 
                               key=lambda x: x['contribution'], 
                               reverse=True)[:3]
    for detail in top_contributions:
        print(f"  • Strong {detail['factor'].replace('_', ' ')}: "
              f"{detail['score']:.2f} (contributed {detail['contribution']:.3f})")
    
    print("\n🎯 Weighted Decision Benefits:")
    print("  • Reflects true priorities")
    print("  • Transparent trade-offs")
    print("  • Consistent evaluation")
    print("  • Quantifiable comparison")
    print("  • Fair and objective")
    print("  • Aligned with goals")
    
    # Score distribution
    scores = [r['weighted_score'] for r in state['weighted_scores']]
    print(f"\n📈 Score Statistics:")
    print(f"  Highest: {max(scores):.3f}")
    print(f"  Lowest: {min(scores):.3f}")
    print(f"  Average: {sum(scores)/len(scores):.3f}")
    print(f"  Range: {max(scores) - min(scores):.3f}")
    
    print("\n="*70)
    print("✅ Weighted Decision Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_weighted_decision_graph():
    workflow = StateGraph(WeightedDecisionState)
    workflow.add_node("define_factors", define_decision_factors_agent)
    workflow.add_node("evaluate", evaluate_candidates_agent)
    workflow.add_node("report", generate_weighted_report_agent)
    workflow.add_edge(START, "define_factors")
    workflow.add_edge("define_factors", "evaluate")
    workflow.add_edge("evaluate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 258: Weighted Decision MCP Pattern")
    print("="*70)
    
    app = create_weighted_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "factors": {},
        "candidates": [],
        "weighted_scores": [],
        "selected_candidate": {}
    })
    print("\n✅ Weighted Decision Pattern Complete!")


if __name__ == "__main__":
    main()
