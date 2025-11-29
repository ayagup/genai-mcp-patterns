"""
Pattern 254: Multi-Criteria Decision MCP Pattern

This pattern demonstrates multi-criteria decision making - evaluating options
based on multiple conflicting criteria using structured analysis.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class MultiCriteriaDecisionState(TypedDict):
    """State for multi-criteria decision workflow"""
    messages: Annotated[List[str], add]
    alternatives: List[Dict[str, Any]]
    criteria: List[Dict[str, Any]]
    evaluation_matrix: List[List[float]]
    scores: Dict[str, float]
    decision: str


class MultiCriteriaAnalyzer:
    """Analyzes decisions using multiple criteria"""
    
    def normalize_scores(self, scores: List[float], is_benefit: bool = True) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores or all(s == 0 for s in scores):
            return [0.0] * len(scores)
        
        max_score = max(scores)
        min_score = min(scores)
        range_score = max_score - min_score
        
        if range_score == 0:
            return [1.0] * len(scores)
        
        if is_benefit:
            return [(s - min_score) / range_score for s in scores]
        else:  # Cost criterion (lower is better)
            return [(max_score - s) / range_score for s in scores]
    
    def weighted_sum_method(self, alternatives: List[Dict], criteria: List[Dict]) -> tuple:
        """Calculate weighted sum scores"""
        num_alternatives = len(alternatives)
        num_criteria = len(criteria)
        
        # Build evaluation matrix
        evaluation_matrix = []
        
        for criterion in criteria:
            criterion_scores = []
            for alternative in alternatives:
                score = alternative.get(criterion["name"], 0)
                criterion_scores.append(score)
            
            # Normalize scores
            normalized = self.normalize_scores(
                criterion_scores,
                is_benefit=criterion.get("type", "benefit") == "benefit"
            )
            evaluation_matrix.append(normalized)
        
        # Calculate weighted scores for each alternative
        final_scores = {}
        for i, alternative in enumerate(alternatives):
            weighted_score = 0
            for j, criterion in enumerate(criteria):
                weighted_score += evaluation_matrix[j][i] * criterion["weight"]
            final_scores[alternative["name"]] = round(weighted_score, 3)
        
        # Select best alternative
        best_alternative = max(final_scores, key=final_scores.get)
        
        # Transpose matrix for reporting
        transposed_matrix = [[evaluation_matrix[j][i] for j in range(num_criteria)] 
                            for i in range(num_alternatives)]
        
        return best_alternative, final_scores, transposed_matrix


def define_decision_problem_agent(state: MultiCriteriaDecisionState) -> MultiCriteriaDecisionState:
    """Define decision problem with alternatives and criteria"""
    print("\n📋 Defining Multi-Criteria Decision Problem...")
    
    # Define alternatives (e.g., selecting a cloud provider)
    alternatives = [
        {
            "name": "Provider A",
            "cost": 85,           # Lower is better
            "performance": 90,    # Higher is better
            "reliability": 95,    # Higher is better
            "support": 80,        # Higher is better
            "scalability": 85     # Higher is better
        },
        {
            "name": "Provider B",
            "cost": 70,
            "performance": 85,
            "reliability": 90,
            "support": 90,
            "scalability": 92
        },
        {
            "name": "Provider C",
            "cost": 95,
            "performance": 95,
            "reliability": 88,
            "support": 85,
            "scalability": 90
        },
        {
            "name": "Provider D",
            "cost": 80,
            "performance": 88,
            "reliability": 92,
            "support": 88,
            "scalability": 88
        }
    ]
    
    # Define criteria with weights
    criteria = [
        {"name": "cost", "weight": 0.25, "type": "cost"},         # Cost is minimize
        {"name": "performance", "weight": 0.30, "type": "benefit"},
        {"name": "reliability", "weight": 0.25, "type": "benefit"},
        {"name": "support", "weight": 0.10, "type": "benefit"},
        {"name": "scalability", "weight": 0.10, "type": "benefit"}
    ]
    
    print(f"\n  Alternatives: {len(alternatives)}")
    for alt in alternatives:
        print(f"    • {alt['name']}")
    
    print(f"\n  Criteria (Weights):")
    for crit in criteria:
        print(f"    • {crit['name']}: {crit['weight']:.0%} ({crit['type']})")
    
    return {
        **state,
        "alternatives": alternatives,
        "criteria": criteria,
        "messages": [f"✓ Defined {len(alternatives)} alternatives, {len(criteria)} criteria"]
    }


def evaluate_alternatives_agent(state: MultiCriteriaDecisionState) -> MultiCriteriaDecisionState:
    """Evaluate alternatives using multi-criteria analysis"""
    print("\n⚖️ Evaluating Alternatives...")
    
    analyzer = MultiCriteriaAnalyzer()
    decision, scores, evaluation_matrix = analyzer.weighted_sum_method(
        state["alternatives"],
        state["criteria"]
    )
    
    print(f"\n  Evaluation Matrix (Normalized):")
    print(f"  {'Alternative':<15}", end="")
    for criterion in state["criteria"]:
        print(f"{criterion['name']:<12}", end="")
    print("Score")
    
    for i, alternative in enumerate(state["alternatives"]):
        print(f"  {alternative['name']:<15}", end="")
        for j, value in enumerate(evaluation_matrix[i]):
            print(f"{value:<12.3f}", end="")
        print(f"{scores[alternative['name']]:.3f}")
    
    print(f"\n  Best Alternative: {decision}")
    print(f"  Score: {scores[decision]:.3f}")
    
    return {
        **state,
        "evaluation_matrix": evaluation_matrix,
        "scores": scores,
        "decision": decision,
        "messages": [f"✓ Selected: {decision}"]
    }


def generate_multicriteria_report_agent(state: MultiCriteriaDecisionState) -> MultiCriteriaDecisionState:
    """Generate multi-criteria decision report"""
    print("\n" + "="*70)
    print("MULTI-CRITERIA DECISION REPORT")
    print("="*70)
    
    print(f"\n📊 Decision Problem:")
    print(f"  Alternatives: {len(state['alternatives'])}")
    print(f"  Criteria: {len(state['criteria'])}")
    
    print(f"\n⚖️ Criteria Weights:")
    total_weight = sum(c["weight"] for c in state["criteria"])
    for criterion in state["criteria"]:
        print(f"  {criterion['name']}: {criterion['weight']:.1%} ({criterion['type']})")
    print(f"  Total: {total_weight:.1%}")
    
    print(f"\n📈 Alternative Scores:")
    sorted_scores = sorted(state["scores"].items(), key=lambda x: x[1], reverse=True)
    for rank, (alternative, score) in enumerate(sorted_scores, 1):
        marker = "⭐" if alternative == state["decision"] else "  "
        print(f"  {marker} {rank}. {alternative}: {score:.3f}")
    
    print(f"\n✅ Final Decision:")
    print(f"  Selected: {state['decision']}")
    print(f"  Score: {state['scores'][state['decision']]:.3f}")
    
    # Show detailed breakdown for winner
    winner = next(a for a in state["alternatives"] if a["name"] == state["decision"])
    print(f"\n  Detailed Breakdown for {state['decision']}:")
    for criterion in state["criteria"]:
        value = winner[criterion["name"]]
        weight = criterion["weight"]
        print(f"    {criterion['name']}: {value} (weight: {weight:.1%})")
    
    print("\n💡 Multi-Criteria Decision Benefits:")
    print("  • Structured decision framework")
    print("  • Multiple factors considered")
    print("  • Transparent weighting")
    print("  • Quantifiable comparison")
    print("  • Reduces bias")
    print("  • Handles trade-offs systematically")
    
    print("\n="*70)
    print("✅ Multi-Criteria Decision Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_multi_criteria_decision_graph():
    workflow = StateGraph(MultiCriteriaDecisionState)
    workflow.add_node("define", define_decision_problem_agent)
    workflow.add_node("evaluate", evaluate_alternatives_agent)
    workflow.add_node("report", generate_multicriteria_report_agent)
    workflow.add_edge(START, "define")
    workflow.add_edge("define", "evaluate")
    workflow.add_edge("evaluate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 254: Multi-Criteria Decision MCP Pattern")
    print("="*70)
    
    app = create_multi_criteria_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "alternatives": [],
        "criteria": [],
        "evaluation_matrix": [],
        "scores": {},
        "decision": ""
    })
    print("\n✅ Multi-Criteria Decision Pattern Complete!")


if __name__ == "__main__":
    main()
