"""
Pattern 257: Ensemble Decision MCP Pattern

This pattern demonstrates ensemble decision making - combining multiple
decision models to achieve more robust and accurate decisions.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class EnsembleDecisionState(TypedDict):
    """State for ensemble decision workflow"""
    messages: Annotated[List[str], add]
    problem: Dict[str, Any]
    model_predictions: List[Dict[str, Any]]
    ensemble_decision: str
    confidence: float


class DecisionModel:
    """Base decision model"""
    
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
    
    def predict(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction"""
        raise NotImplementedError


class RuleBasedModel(DecisionModel):
    """Rule-based decision model"""
    
    def predict(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        score = problem.get("risk_score", 50)
        if score > 70:
            decision = "REJECT"
            confidence = 0.85
        elif score < 30:
            decision = "APPROVE"
            confidence = 0.90
        else:
            decision = "REVIEW"
            confidence = 0.70
        
        return {
            "model": self.name,
            "decision": decision,
            "confidence": confidence,
            "reasoning": f"Risk score {score} triggers {decision}"
        }


class MLModel(DecisionModel):
    """Machine learning decision model"""
    
    def predict(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Simplified ML prediction
        features_score = sum(problem.get("features", {}).values()) / max(len(problem.get("features", {})), 1)
        
        if features_score > 0.7:
            decision = "APPROVE"
            confidence = 0.88
        elif features_score < 0.4:
            decision = "REJECT"
            confidence = 0.82
        else:
            decision = "REVIEW"
            confidence = 0.65
        
        return {
            "model": self.name,
            "decision": decision,
            "confidence": confidence,
            "reasoning": f"Feature score {features_score:.2f} suggests {decision}"
        }


class HeuristicModel(DecisionModel):
    """Heuristic-based decision model"""
    
    def predict(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        history = problem.get("history_length", 0)
        
        if history > 50:
            decision = "APPROVE"
            confidence = 0.92
        elif history < 10:
            decision = "REJECT"
            confidence = 0.75
        else:
            decision = "REVIEW"
            confidence = 0.68
        
        return {
            "model": self.name,
            "decision": decision,
            "confidence": confidence,
            "reasoning": f"History length {history} indicates {decision}"
        }


class EnsembleDecisionMaker:
    """Combines multiple models for ensemble decision"""
    
    def __init__(self):
        self.models = [
            RuleBasedModel("Rule-Based Model", "Risk Assessment"),
            MLModel("ML Model", "Pattern Recognition"),
            HeuristicModel("Heuristic Model", "Experience-Based")
        ]
    
    def voting(self, predictions: List[Dict[str, Any]]) -> tuple:
        """Majority voting"""
        votes = {}
        for pred in predictions:
            decision = pred["decision"]
            votes[decision] = votes.get(decision, 0) + 1
        
        winner = max(votes, key=votes.get)
        return winner, votes
    
    def weighted_voting(self, predictions: List[Dict[str, Any]]) -> tuple:
        """Weighted voting by confidence"""
        weighted_votes = {}
        for pred in predictions:
            decision = pred["decision"]
            confidence = pred["confidence"]
            weighted_votes[decision] = weighted_votes.get(decision, 0) + confidence
        
        winner = max(weighted_votes, key=weighted_votes.get)
        total_confidence = sum(weighted_votes.values())
        normalized_confidence = weighted_votes[winner] / total_confidence if total_confidence > 0 else 0
        
        return winner, normalized_confidence
    
    def decide(self, problem: Dict[str, Any]) -> tuple:
        """Make ensemble decision"""
        predictions = []
        
        for model in self.models:
            prediction = model.predict(problem)
            predictions.append(prediction)
        
        # Use weighted voting for final decision
        decision, confidence = self.weighted_voting(predictions)
        
        return decision, confidence, predictions


def define_problem_agent(state: EnsembleDecisionState) -> EnsembleDecisionState:
    """Define problem for ensemble decision"""
    print("\n📋 Defining Problem...")
    
    problem = {
        "type": "loan_application",
        "risk_score": 55,
        "features": {
            "credit_score": 0.65,
            "income_stability": 0.72,
            "debt_ratio": 0.58
        },
        "history_length": 35,
        "amount": 50000
    }
    
    print(f"\n  Problem Type: {problem['type']}")
    print(f"  Risk Score: {problem['risk_score']}")
    print(f"  Features: {problem['features']}")
    print(f"  History Length: {problem['history_length']} months")
    
    return {
        **state,
        "problem": problem,
        "messages": ["✓ Problem defined"]
    }


def ensemble_prediction_agent(state: EnsembleDecisionState) -> EnsembleDecisionState:
    """Get ensemble prediction"""
    print("\n🤖 Running Ensemble Models...")
    
    ensemble = EnsembleDecisionMaker()
    decision, confidence, predictions = ensemble.decide(state["problem"])
    
    print(f"\n  Model Predictions:")
    for pred in predictions:
        print(f"\n    • {pred['model']}")
        print(f"      Decision: {pred['decision']}")
        print(f"      Confidence: {pred['confidence']:.0%}")
        print(f"      Reasoning: {pred['reasoning']}")
    
    print(f"\n  Ensemble Decision: {decision}")
    print(f"  Ensemble Confidence: {confidence:.0%}")
    
    return {
        **state,
        "model_predictions": predictions,
        "ensemble_decision": decision,
        "confidence": confidence,
        "messages": [f"✓ Ensemble decision: {decision}"]
    }


def generate_ensemble_report_agent(state: EnsembleDecisionState) -> EnsembleDecisionState:
    """Generate ensemble decision report"""
    print("\n" + "="*70)
    print("ENSEMBLE DECISION REPORT")
    print("="*70)
    
    print(f"\n📊 Problem:")
    for key, value in state["problem"].items():
        print(f"  {key}: {value}")
    
    print(f"\n🤖 Individual Model Predictions:")
    for pred in state["model_predictions"]:
        print(f"\n  {pred['model']}:")
        print(f"    Decision: {pred['decision']}")
        print(f"    Confidence: {pred['confidence']:.0%}")
        print(f"    Reasoning: {pred['reasoning']}")
    
    # Voting analysis
    votes = {}
    for pred in state["model_predictions"]:
        decision = pred["decision"]
        votes[decision] = votes.get(decision, 0) + 1
    
    print(f"\n🗳️ Voting Analysis:")
    for decision, count in sorted(votes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {decision}: {count} vote(s)")
    
    print(f"\n✅ Final Ensemble Decision:")
    print(f"  Decision: {state['ensemble_decision']}")
    print(f"  Confidence: {state['confidence']:.0%}")
    print(f"  Agreement: {votes.get(state['ensemble_decision'], 0)}/{len(state['model_predictions'])} models")
    
    print("\n💡 Ensemble Decision Benefits:")
    print("  • Combines multiple perspectives")
    print("  • Reduces individual model bias")
    print("  • More robust predictions")
    print("  • Higher accuracy")
    print("  • Better generalization")
    print("  • Risk mitigation")
    
    print("\n="*70)
    print("✅ Ensemble Decision Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_ensemble_decision_graph():
    workflow = StateGraph(EnsembleDecisionState)
    workflow.add_node("define", define_problem_agent)
    workflow.add_node("predict", ensemble_prediction_agent)
    workflow.add_node("report", generate_ensemble_report_agent)
    workflow.add_edge(START, "define")
    workflow.add_edge("define", "predict")
    workflow.add_edge("predict", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 257: Ensemble Decision MCP Pattern")
    print("="*70)
    
    app = create_ensemble_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "problem": {},
        "model_predictions": [],
        "ensemble_decision": "",
        "confidence": 0.0
    })
    print("\n✅ Ensemble Decision Pattern Complete!")


if __name__ == "__main__":
    main()
