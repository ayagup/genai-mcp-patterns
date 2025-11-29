"""
Pattern 278: Model Selection MCP Pattern

This pattern demonstrates intelligent model selection based on task requirements,
performance characteristics, and cost constraints.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class ModelSelectionState(TypedDict):
    """State for model selection workflow"""
    messages: Annotated[List[str], add]
    task_requirements: Dict[str, Any]
    available_models: List[Dict[str, Any]]
    model_evaluation: Dict[str, Any]
    selection_recommendation: Dict[str, Any]


class TaskAnalyzer:
    """Analyzes task requirements"""
    
    def __init__(self):
        self.task_characteristics = {
            "complexity": ["simple", "moderate", "complex", "very_complex"],
            "latency_sensitivity": ["low", "medium", "high", "critical"],
            "accuracy_requirement": ["low", "medium", "high", "critical"],
            "volume": ["low", "medium", "high", "very_high"]
        }
    
    def analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task requirements"""
        requirements = {
            "task_type": task.get("type", "general"),
            "complexity": task.get("complexity", "moderate"),
            "latency_requirement_ms": task.get("max_latency_ms", 1000),
            "accuracy_requirement": task.get("min_accuracy", 0.85),
            "volume_per_day": task.get("volume_per_day", 1000),
            "budget_per_1k": task.get("budget_per_1k", 0.10),
            "context_size": task.get("context_size", 4000),
            "output_size": task.get("output_size", 500),
            "special_capabilities": task.get("special_capabilities", [])
        }
        
        # Determine priority weights
        weights = {
            "accuracy": 1.0,
            "latency": 1.0,
            "cost": 1.0
        }
        
        if requirements["accuracy_requirement"] > 0.95:
            weights["accuracy"] = 2.0
        
        if requirements["latency_requirement_ms"] < 500:
            weights["latency"] = 2.0
        
        if requirements["volume_per_day"] > 10000:
            weights["cost"] = 2.0
        
        requirements["weights"] = weights
        
        return requirements


class ModelEvaluator:
    """Evaluates models against requirements"""
    
    def __init__(self):
        self.model_database = {
            "gpt-4": {
                "capabilities": ["reasoning", "code", "analysis", "creative"],
                "avg_latency_ms": 2000,
                "accuracy": 0.95,
                "cost_per_1k_tokens": 0.03,
                "max_context": 8192,
                "strengths": ["High accuracy", "Complex reasoning", "Versatile"],
                "weaknesses": ["Higher cost", "Slower response"]
            },
            "gpt-3.5-turbo": {
                "capabilities": ["general", "code", "conversation"],
                "avg_latency_ms": 800,
                "accuracy": 0.88,
                "cost_per_1k_tokens": 0.002,
                "max_context": 4096,
                "strengths": ["Fast", "Cost-effective", "Good general performance"],
                "weaknesses": ["Lower accuracy on complex tasks"]
            },
            "claude-3-opus": {
                "capabilities": ["reasoning", "analysis", "creative", "code"],
                "avg_latency_ms": 1800,
                "accuracy": 0.94,
                "cost_per_1k_tokens": 0.015,
                "max_context": 200000,
                "strengths": ["Very large context", "High accuracy", "Strong reasoning"],
                "weaknesses": ["Higher cost", "Moderate latency"]
            },
            "claude-3-sonnet": {
                "capabilities": ["general", "analysis", "code"],
                "avg_latency_ms": 1200,
                "accuracy": 0.90,
                "cost_per_1k_tokens": 0.003,
                "max_context": 200000,
                "strengths": ["Large context", "Balanced performance", "Good accuracy"],
                "weaknesses": ["Moderate cost"]
            },
            "claude-3-haiku": {
                "capabilities": ["general", "simple_tasks"],
                "avg_latency_ms": 400,
                "accuracy": 0.85,
                "cost_per_1k_tokens": 0.00025,
                "max_context": 200000,
                "strengths": ["Very fast", "Very cheap", "Large context"],
                "weaknesses": ["Lower accuracy on complex tasks"]
            },
            "llama-2-70b": {
                "capabilities": ["general", "conversation", "analysis"],
                "avg_latency_ms": 1500,
                "accuracy": 0.87,
                "cost_per_1k_tokens": 0.0009,
                "max_context": 4096,
                "strengths": ["Low cost", "Open source", "Good performance"],
                "weaknesses": ["Moderate latency", "Limited context"]
            }
        }
    
    def evaluate_models(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all models against requirements"""
        evaluations = []
        
        for model_name, model_specs in self.model_database.items():
            # Check basic compatibility
            if model_specs["max_context"] < requirements["context_size"]:
                continue  # Skip if context too small
            
            # Calculate scores
            accuracy_score = min(1.0, model_specs["accuracy"] / requirements["accuracy_requirement"])
            latency_score = min(1.0, requirements["latency_requirement_ms"] / model_specs["avg_latency_ms"])
            cost_score = min(1.0, requirements["budget_per_1k"] / model_specs["cost_per_1k_tokens"]) if model_specs["cost_per_1k_tokens"] > 0 else 1.0
            
            # Apply weights
            weights = requirements["weights"]
            weighted_score = (
                accuracy_score * weights["accuracy"] +
                latency_score * weights["latency"] +
                cost_score * weights["cost"]
            ) / sum(weights.values())
            
            # Calculate daily cost
            tokens_per_request = (requirements["context_size"] + requirements["output_size"]) / 1000
            daily_cost = tokens_per_request * model_specs["cost_per_1k_tokens"] * requirements["volume_per_day"]
            
            evaluations.append({
                "model_name": model_name,
                "overall_score": weighted_score * 100,
                "accuracy_score": accuracy_score * 100,
                "latency_score": latency_score * 100,
                "cost_score": cost_score * 100,
                "daily_cost": daily_cost,
                "avg_latency_ms": model_specs["avg_latency_ms"],
                "accuracy": model_specs["accuracy"],
                "cost_per_1k": model_specs["cost_per_1k_tokens"],
                "strengths": model_specs["strengths"],
                "weaknesses": model_specs["weaknesses"],
                "meets_requirements": accuracy_score >= 0.95 and latency_score >= 0.90 and cost_score >= 0.90
            })
        
        return sorted(evaluations, key=lambda x: x["overall_score"], reverse=True)


class ModelSelector:
    """Selects optimal model"""
    
    def select_model(self, evaluations: List[Dict[str, Any]], 
                    requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best model"""
        if not evaluations:
            return {"error": "No suitable models found"}
        
        # Primary recommendation
        primary = evaluations[0]
        
        # Alternative recommendations
        alternatives = evaluations[1:3] if len(evaluations) > 1 else []
        
        # Fallback for cost-sensitive
        cost_optimized = min(evaluations, key=lambda x: x["daily_cost"])
        
        # Fallback for latency-sensitive
        latency_optimized = min(evaluations, key=lambda x: x["avg_latency_ms"])
        
        # Fallback for accuracy-critical
        accuracy_optimized = max(evaluations, key=lambda x: x["accuracy"])
        
        return {
            "primary_recommendation": primary,
            "alternatives": alternatives,
            "specialized_recommendations": {
                "cost_optimized": cost_optimized if cost_optimized != primary else None,
                "latency_optimized": latency_optimized if latency_optimized != primary else None,
                "accuracy_optimized": accuracy_optimized if accuracy_optimized != primary else None
            },
            "recommendation_reason": self._generate_reason(primary, requirements)
        }
    
    def _generate_reason(self, model: Dict[str, Any], requirements: Dict[str, Any]) -> str:
        """Generate recommendation reason"""
        reasons = []
        
        if model["overall_score"] > 90:
            reasons.append("Excellent overall fit for requirements")
        elif model["overall_score"] > 75:
            reasons.append("Good fit for requirements")
        
        if model["accuracy_score"] > 95:
            reasons.append("meets accuracy requirements")
        
        if model["latency_score"] > 95:
            reasons.append("meets latency requirements")
        
        if model["cost_score"] > 95:
            reasons.append("within budget constraints")
        
        return ", ".join(reasons) if reasons else "Best available option"


def analyze_task_requirements_agent(state: ModelSelectionState) -> ModelSelectionState:
    """Analyze task requirements"""
    print("\n📋 Analyzing Task Requirements...")
    
    # Example task
    task = {
        "type": "code_generation",
        "complexity": "complex",
        "max_latency_ms": 2000,
        "min_accuracy": 0.90,
        "volume_per_day": 5000,
        "budget_per_1k": 0.01,
        "context_size": 6000,
        "output_size": 1000,
        "special_capabilities": ["code", "reasoning"]
    }
    
    analyzer = TaskAnalyzer()
    requirements = analyzer.analyze_task(task)
    
    print(f"\n  Task Type: {requirements['task_type']}")
    print(f"  Complexity: {requirements['complexity']}")
    print(f"  Max Latency: {requirements['latency_requirement_ms']}ms")
    print(f"  Min Accuracy: {requirements['accuracy_requirement']:.1%}")
    print(f"  Volume: {requirements['volume_per_day']:,} requests/day")
    print(f"  Budget: ${requirements['budget_per_1k']} per 1K tokens")
    print(f"  Context Size: {requirements['context_size']} tokens")
    
    print(f"\n  Priority Weights:")
    for factor, weight in requirements["weights"].items():
        print(f"    • {factor}: {weight}x")
    
    return {
        **state,
        "task_requirements": requirements,
        "messages": [f"✓ Analyzed task: {requirements['task_type']}"]
    }


def evaluate_models_agent(state: ModelSelectionState) -> ModelSelectionState:
    """Evaluate available models"""
    print("\n🔍 Evaluating Available Models...")
    
    evaluator = ModelEvaluator()
    evaluations = evaluator.evaluate_models(state["task_requirements"])
    
    print(f"\n  Models Evaluated: {len(evaluations)}")
    print(f"  Compatible Models: {sum(1 for e in evaluations if e['meets_requirements'])}")
    
    print(f"\n  Top 3 Models:")
    for i, eval in enumerate(evaluations[:3], 1):
        print(f"\n    {i}. {eval['model_name']}")
        print(f"       Overall Score: {eval['overall_score']:.1f}/100")
        print(f"       Accuracy: {eval['accuracy']:.1%} (score: {eval['accuracy_score']:.1f})")
        print(f"       Latency: {eval['avg_latency_ms']}ms (score: {eval['latency_score']:.1f})")
        print(f"       Cost: ${eval['cost_per_1k']}/1K (score: {eval['cost_score']:.1f})")
        print(f"       Daily Cost: ${eval['daily_cost']:.2f}")
    
    return {
        **state,
        "model_evaluation": {"evaluations": evaluations},
        "messages": [f"✓ Evaluated {len(evaluations)} models"]
    }


def select_model_agent(state: ModelSelectionState) -> ModelSelectionState:
    """Select optimal model"""
    print("\n🎯 Selecting Optimal Model...")
    
    selector = ModelSelector()
    selection = selector.select_model(
        state["model_evaluation"]["evaluations"],
        state["task_requirements"]
    )
    
    primary = selection["primary_recommendation"]
    print(f"\n  Primary Recommendation: {primary['model_name']}")
    print(f"  Overall Score: {primary['overall_score']:.1f}/100")
    print(f"  Reason: {selection['recommendation_reason']}")
    
    if selection["alternatives"]:
        print(f"\n  Alternative Options:")
        for alt in selection["alternatives"]:
            print(f"    • {alt['model_name']} (score: {alt['overall_score']:.1f})")
    
    return {
        **state,
        "selection_recommendation": selection,
        "messages": [f"✓ Selected model: {primary['model_name']}"]
    }


def generate_selection_report_agent(state: ModelSelectionState) -> ModelSelectionState:
    """Generate model selection report"""
    print("\n" + "="*70)
    print("MODEL SELECTION REPORT")
    print("="*70)
    
    requirements = state["task_requirements"]
    print(f"\n📋 Task Requirements:")
    print(f"  Task Type: {requirements['task_type']}")
    print(f"  Complexity: {requirements['complexity']}")
    print(f"  Maximum Latency: {requirements['latency_requirement_ms']}ms")
    print(f"  Minimum Accuracy: {requirements['accuracy_requirement']:.1%}")
    print(f"  Daily Volume: {requirements['volume_per_day']:,} requests")
    print(f"  Budget: ${requirements['budget_per_1k']} per 1K tokens")
    print(f"  Context Size: {requirements['context_size']} tokens")
    print(f"  Output Size: {requirements['output_size']} tokens")
    
    print(f"\n  Priority Weights:")
    for factor, weight in requirements["weights"].items():
        indicator = "⭐" * int(weight) + "☆" * (3 - int(weight))
        print(f"    {factor.capitalize()}: {indicator} ({weight}x)")
    
    print(f"\n🔍 Model Evaluation Results:")
    evaluations = state["model_evaluation"]["evaluations"]
    print(f"  Total Models Evaluated: {len(evaluations)}")
    print(f"  Models Meeting Requirements: {sum(1 for e in evaluations if e['meets_requirements'])}")
    
    print(f"\n  Model Comparison:")
    print(f"\n  {'Model':<20} {'Score':>8} {'Accuracy':>10} {'Latency':>10} {'Cost/Day':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
    for eval in evaluations[:5]:
        meets = "✓" if eval["meets_requirements"] else "✗"
        print(f"  {meets} {eval['model_name']:<18} {eval['overall_score']:>6.1f}/100 {eval['accuracy']:>9.1%} {eval['avg_latency_ms']:>8}ms ${eval['daily_cost']:>10.2f}")
    
    print(f"\n🎯 Recommendation:")
    selection = state["selection_recommendation"]
    primary = selection["primary_recommendation"]
    
    print(f"\n  PRIMARY: {primary['model_name']}")
    print(f"  Overall Score: {primary['overall_score']:.1f}/100")
    print(f"  Reason: {selection['recommendation_reason']}")
    
    print(f"\n  Performance Metrics:")
    print(f"    • Accuracy: {primary['accuracy']:.1%} (score: {primary['accuracy_score']:.1f}/100)")
    print(f"    • Latency: {primary['avg_latency_ms']}ms (score: {primary['latency_score']:.1f}/100)")
    print(f"    • Cost: ${primary['cost_per_1k']} per 1K tokens (score: {primary['cost_score']:.1f}/100)")
    print(f"    • Daily Cost: ${primary['daily_cost']:.2f}")
    print(f"    • Monthly Cost: ${primary['daily_cost'] * 30:.2f}")
    
    print(f"\n  Strengths:")
    for strength in primary["strengths"]:
        print(f"    ✓ {strength}")
    
    print(f"\n  Weaknesses:")
    for weakness in primary["weaknesses"]:
        print(f"    ✗ {weakness}")
    
    if selection["alternatives"]:
        print(f"\n  Alternative Options:")
        for i, alt in enumerate(selection["alternatives"], 1):
            print(f"\n    {i}. {alt['model_name']} (score: {alt['overall_score']:.1f}/100)")
            print(f"       Daily Cost: ${alt['daily_cost']:.2f}")
            print(f"       Best for: {', '.join(alt['strengths'][:2])}")
    
    specialized = selection["specialized_recommendations"]
    if any(specialized.values()):
        print(f"\n  Specialized Recommendations:")
        
        if specialized["cost_optimized"]:
            model = specialized["cost_optimized"]
            print(f"\n    💰 Cost-Optimized: {model['model_name']}")
            print(f"       Daily Cost: ${model['daily_cost']:.2f} (saves ${primary['daily_cost'] - model['daily_cost']:.2f}/day)")
        
        if specialized["latency_optimized"]:
            model = specialized["latency_optimized"]
            print(f"\n    ⚡ Latency-Optimized: {model['model_name']}")
            print(f"       Latency: {model['avg_latency_ms']}ms (saves {primary['avg_latency_ms'] - model['avg_latency_ms']}ms)")
        
        if specialized["accuracy_optimized"]:
            model = specialized["accuracy_optimized"]
            print(f"\n    🎯 Accuracy-Optimized: {model['model_name']}")
            print(f"       Accuracy: {model['accuracy']:.1%} (+{(model['accuracy'] - primary['accuracy']):.1%})")
    
    print(f"\n💡 Model Selection Benefits:")
    print("  • Optimized performance for specific task")
    print("  • Cost-effective solution")
    print("  • Balanced trade-offs")
    print("  • Meets all requirements")
    print("  • Alternative options available")
    print("  • Data-driven decision")
    
    print("\n="*70)
    print("✅ Model Selection Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_model_selection_graph():
    workflow = StateGraph(ModelSelectionState)
    workflow.add_node("analyze_task", analyze_task_requirements_agent)
    workflow.add_node("evaluate_models", evaluate_models_agent)
    workflow.add_node("select_model", select_model_agent)
    workflow.add_node("generate_report", generate_selection_report_agent)
    workflow.add_edge(START, "analyze_task")
    workflow.add_edge("analyze_task", "evaluate_models")
    workflow.add_edge("evaluate_models", "select_model")
    workflow.add_edge("select_model", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 278: Model Selection MCP Pattern")
    print("="*70)
    
    app = create_model_selection_graph()
    final_state = app.invoke({
        "messages": [],
        "task_requirements": {},
        "available_models": [],
        "model_evaluation": {},
        "selection_recommendation": {}
    })
    print("\n✅ Model Selection Pattern Complete!")


if __name__ == "__main__":
    main()
