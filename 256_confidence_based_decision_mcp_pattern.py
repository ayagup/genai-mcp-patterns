"""
Pattern 256: Confidence-Based Decision MCP Pattern

This pattern demonstrates confidence-based decision making - making decisions
based on confidence scores and uncertainty levels.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import random


class ConfidenceBasedDecisionState(TypedDict):
    """State for confidence-based decision workflow"""
    messages: Annotated[List[str], add]
    predictions: List[Dict[str, Any]]
    confidence_thresholds: Dict[str, float]
    decisions: List[Dict[str, Any]]


class ConfidenceEvaluator:
    """Evaluates decisions based on confidence scores"""
    
    def __init__(self):
        self.confidence_thresholds = {
            "high_confidence": 0.85,    # Proceed automatically
            "medium_confidence": 0.60,  # Proceed with human review
            "low_confidence": 0.40      # Requires human decision
        }
    
    def classify_confidence(self, confidence: float) -> str:
        """Classify confidence level"""
        if confidence >= self.confidence_thresholds["high_confidence"]:
            return "high"
        elif confidence >= self.confidence_thresholds["medium_confidence"]:
            return "medium"
        elif confidence >= self.confidence_thresholds["low_confidence"]:
            return "low"
        else:
            return "very_low"
    
    def decide(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision based on confidence"""
        confidence = prediction["confidence"]
        confidence_level = self.classify_confidence(confidence)
        
        # Decision logic based on confidence
        if confidence_level == "high":
            action = "AUTO_APPROVE"
            require_review = False
            risk_level = "low"
        elif confidence_level == "medium":
            action = "APPROVE_WITH_REVIEW"
            require_review = True
            risk_level = "medium"
        elif confidence_level == "low":
            action = "REQUIRE_HUMAN_DECISION"
            require_review = True
            risk_level = "high"
        else:  # very_low
            action = "REJECT_AUTO_DECISION"
            require_review = True
            risk_level = "very_high"
        
        return {
            "prediction_id": prediction["id"],
            "prediction": prediction["prediction"],
            "confidence": confidence,
            "confidence_level": confidence_level,
            "action": action,
            "require_review": require_review,
            "risk_level": risk_level
        }


def generate_predictions_agent(state: ConfidenceBasedDecisionState) -> ConfidenceBasedDecisionState:
    """Generate predictions with confidence scores"""
    print("\n🎯 Generating Predictions...")
    
    # Simulated ML model predictions with varying confidence
    predictions = [
        {
            "id": "P001",
            "prediction": "Approve Loan Application #12345",
            "confidence": 0.92,
            "factors": ["High credit score", "Stable income", "Low debt ratio"]
        },
        {
            "id": "P002",
            "prediction": "Detect Fraud in Transaction #67890",
            "confidence": 0.68,
            "factors": ["Unusual location", "Normal amount", "Known device"]
        },
        {
            "id": "P003",
            "prediction": "Recommend Product Bundle",
            "confidence": 0.45,
            "factors": ["Limited purchase history", "Ambiguous preferences"]
        },
        {
            "id": "P004",
            "prediction": "Classify Support Ticket as Priority",
            "confidence": 0.88,
            "factors": ["Critical keywords", "VIP customer", "SLA near breach"]
        },
        {
            "id": "P005",
            "prediction": "Auto-Resolve Customer Issue",
            "confidence": 0.35,
            "factors": ["Complex problem", "Multiple factors", "Unclear intent"]
        }
    ]
    
    print(f"\n  Generated Predictions: {len(predictions)}")
    for pred in predictions:
        print(f"\n    • {pred['id']}: {pred['prediction']}")
        print(f"      Confidence: {pred['confidence']:.0%}")
    
    return {
        **state,
        "predictions": predictions,
        "messages": [f"✓ Generated {len(predictions)} predictions"]
    }


def evaluate_confidence_agent(state: ConfidenceBasedDecisionState) -> ConfidenceBasedDecisionState:
    """Evaluate predictions based on confidence"""
    print("\n📊 Evaluating Confidence Levels...")
    
    evaluator = ConfidenceEvaluator()
    decisions = []
    
    for prediction in state["predictions"]:
        decision = evaluator.decide(prediction)
        decisions.append(decision)
        
        # Visual indicator
        if decision["confidence_level"] == "high":
            symbol = "🟢"
        elif decision["confidence_level"] == "medium":
            symbol = "🟡"
        elif decision["confidence_level"] == "low":
            symbol = "🟠"
        else:
            symbol = "🔴"
        
        print(f"\n    {symbol} {decision['prediction_id']}: {decision['prediction']}")
        print(f"      Confidence: {decision['confidence']:.0%} ({decision['confidence_level']})")
        print(f"      Action: {decision['action']}")
        print(f"      Review Required: {decision['require_review']}")
        print(f"      Risk Level: {decision['risk_level']}")
    
    return {
        **state,
        "confidence_thresholds": evaluator.confidence_thresholds,
        "decisions": decisions,
        "messages": [f"✓ Evaluated {len(decisions)} predictions"]
    }


def generate_confidence_report_agent(state: ConfidenceBasedDecisionState) -> ConfidenceBasedDecisionState:
    """Generate confidence-based decision report"""
    print("\n" + "="*70)
    print("CONFIDENCE-BASED DECISION REPORT")
    print("="*70)
    
    print(f"\n📊 Predictions Analyzed: {len(state['predictions'])}")
    
    print(f"\n🎯 Confidence Thresholds:")
    for level, threshold in state["confidence_thresholds"].items():
        print(f"  {level.replace('_', ' ').title()}: ≥ {threshold:.0%}")
    
    # Group decisions by confidence level
    by_level = {}
    for decision in state["decisions"]:
        level = decision["confidence_level"]
        by_level.setdefault(level, []).append(decision)
    
    print(f"\n📈 Decisions by Confidence Level:")
    
    for level in ["high", "medium", "low", "very_low"]:
        if level in by_level:
            count = len(by_level[level])
            print(f"\n  {level.upper()} Confidence ({count}):")
            for decision in by_level[level]:
                print(f"    • {decision['prediction_id']}: {decision['action']}")
                print(f"      Confidence: {decision['confidence']:.0%}")
    
    # Summary statistics
    auto_approve = sum(1 for d in state["decisions"] if d["action"] == "AUTO_APPROVE")
    need_review = sum(1 for d in state["decisions"] if d["require_review"])
    
    print(f"\n📋 Decision Summary:")
    print(f"  Auto-Approved: {auto_approve}")
    print(f"  Requiring Review: {need_review}")
    print(f"  Total Decisions: {len(state['decisions'])}")
    
    # Risk distribution
    print(f"\n⚠️ Risk Distribution:")
    risk_counts = {}
    for decision in state["decisions"]:
        risk = decision["risk_level"]
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    for risk, count in sorted(risk_counts.items()):
        print(f"  {risk.title()}: {count}")
    
    print("\n💡 Confidence-Based Decision Benefits:")
    print("  • Risk-aware automation")
    print("  • Appropriate human oversight")
    print("  • Quantified uncertainty")
    print("  • Adaptive decision-making")
    print("  • Quality assurance")
    print("  • Explainable decisions")
    
    print("\n="*70)
    print("✅ Confidence-Based Decision Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_confidence_based_decision_graph():
    workflow = StateGraph(ConfidenceBasedDecisionState)
    workflow.add_node("generate", generate_predictions_agent)
    workflow.add_node("evaluate", evaluate_confidence_agent)
    workflow.add_node("report", generate_confidence_report_agent)
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 256: Confidence-Based Decision MCP Pattern")
    print("="*70)
    
    app = create_confidence_based_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "predictions": [],
        "confidence_thresholds": {},
        "decisions": []
    })
    print("\n✅ Confidence-Based Decision Pattern Complete!")


if __name__ == "__main__":
    main()
