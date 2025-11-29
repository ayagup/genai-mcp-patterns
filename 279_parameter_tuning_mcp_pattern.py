"""
Pattern 279: Parameter Tuning MCP Pattern

This pattern demonstrates hyperparameter optimization through systematic
exploration, grid search, and performance evaluation.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import itertools


class ParameterTuningState(TypedDict):
    """State for parameter tuning workflow"""
    messages: Annotated[List[str], add]
    parameter_space: Dict[str, Any]
    tuning_results: List[Dict[str, Any]]
    best_parameters: Dict[str, Any]
    optimization_report: Dict[str, Any]


class ParameterSpace:
    """Defines parameter search space"""
    
    def __init__(self):
        self.parameter_types = {
            "learning_rate": "continuous",
            "batch_size": "discrete",
            "temperature": "continuous",
            "max_tokens": "discrete",
            "top_p": "continuous",
            "epochs": "discrete"
        }
    
    def define_space(self, model_type: str) -> Dict[str, List]:
        """Define parameter search space based on model type"""
        if model_type == "llm":
            return {
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
                "top_p": [0.1, 0.5, 0.9, 0.95, 1.0],
                "max_tokens": [100, 256, 512, 1024, 2048],
                "frequency_penalty": [0.0, 0.2, 0.5, 0.8, 1.0],
                "presence_penalty": [0.0, 0.2, 0.5, 0.8, 1.0]
            }
        elif model_type == "ml_classifier":
            return {
                "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                "batch_size": [16, 32, 64, 128],
                "epochs": [10, 20, 50, 100],
                "dropout": [0.1, 0.2, 0.3, 0.5]
            }
        else:
            return {}
    
    def generate_combinations(self, space: Dict[str, List], strategy: str = "grid") -> List[Dict[str, Any]]:
        """Generate parameter combinations"""
        if strategy == "grid":
            # Grid search: all combinations
            keys = list(space.keys())
            values = list(space.values())
            combinations = list(itertools.product(*values))
            
            # Limit to reasonable number
            if len(combinations) > 100:
                # Sample evenly
                step = len(combinations) // 100
                combinations = combinations[::step]
            
            return [dict(zip(keys, combo)) for combo in combinations]
        
        elif strategy == "random":
            # Random search: sample combinations
            import random
            combinations = []
            for _ in range(min(50, len(list(itertools.product(*space.values()))))):
                combo = {k: random.choice(v) for k, v in space.items()}
                combinations.append(combo)
            return combinations
        
        return []


class ParameterEvaluator:
    """Evaluates parameter configurations"""
    
    def __init__(self):
        self.metrics = ["accuracy", "latency", "cost"]
    
    def evaluate_config(self, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Evaluate a parameter configuration"""
        # Simulated evaluation (in real scenario, would actually run model)
        
        if model_type == "llm":
            # Higher temperature = more creative but less consistent
            # Lower temperature = more deterministic
            base_accuracy = 0.85
            temp_factor = 1.0 - (config.get("temperature", 0.5) * 0.1)
            accuracy = base_accuracy * temp_factor
            
            # Latency affected by max_tokens
            base_latency = 500
            latency = base_latency * (config.get("max_tokens", 512) / 512)
            
            # Cost affected by max_tokens
            base_cost = 0.002
            cost = base_cost * (config.get("max_tokens", 512) / 1000)
            
            # Quality score
            quality = accuracy * 0.6 + (1.0 - min(config.get("temperature", 0.5), 0.9)) * 0.4
            
        else:  # ml_classifier
            # Learning rate and batch size affect accuracy
            lr = config.get("learning_rate", 0.01)
            batch_size = config.get("batch_size", 32)
            epochs = config.get("epochs", 20)
            
            # Simulate accuracy (inverted U-curve for learning rate)
            if lr < 0.001:
                accuracy = 0.75 + (lr / 0.001) * 0.05
            elif lr < 0.01:
                accuracy = 0.80 + ((0.01 - lr) / 0.009) * 0.08
            else:
                accuracy = 0.88 - ((lr - 0.01) / 0.09) * 0.15
            
            # Training time affected by batch size and epochs
            base_time = 100
            latency = base_time * (epochs / 20) * (64 / batch_size)
            
            # Cost proportional to training time
            cost = latency * 0.01
            
            quality = accuracy
        
        # Calculate composite score
        composite_score = (
            accuracy * 0.5 +
            (1000 / max(latency, 100)) * 0.3 +
            (0.01 / max(cost, 0.001)) * 0.2
        )
        
        return {
            "config": config,
            "accuracy": accuracy,
            "latency_ms": latency,
            "cost": cost,
            "quality": quality,
            "composite_score": composite_score
        }


class ParameterOptimizer:
    """Optimizes parameter selection"""
    
    def find_best_parameters(self, results: List[Dict[str, Any]], 
                            optimization_goal: str = "balanced") -> Dict[str, Any]:
        """Find best parameters based on goal"""
        
        if optimization_goal == "accuracy":
            best = max(results, key=lambda x: x["accuracy"])
        elif optimization_goal == "latency":
            best = min(results, key=lambda x: x["latency_ms"])
        elif optimization_goal == "cost":
            best = min(results, key=lambda x: x["cost"])
        else:  # balanced
            best = max(results, key=lambda x: x["composite_score"])
        
        # Find top 5 configurations
        top_configs = sorted(results, key=lambda x: x["composite_score"], reverse=True)[:5]
        
        # Analyze parameter importance
        parameter_importance = self._analyze_parameter_importance(results)
        
        return {
            "best_config": best,
            "top_configs": top_configs,
            "parameter_importance": parameter_importance,
            "total_evaluated": len(results)
        }
    
    def _analyze_parameter_importance(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze which parameters have most impact"""
        importance = {}
        
        if not results:
            return importance
        
        # Get all parameter names
        param_names = list(results[0]["config"].keys())
        
        for param in param_names:
            # Group results by parameter value
            by_value = {}
            for result in results:
                value = result["config"][param]
                if value not in by_value:
                    by_value[value] = []
                by_value[value].append(result["composite_score"])
            
            # Calculate variance in average scores
            avg_scores = [sum(scores) / len(scores) for scores in by_value.values()]
            if len(avg_scores) > 1:
                variance = sum((s - sum(avg_scores) / len(avg_scores)) ** 2 for s in avg_scores) / len(avg_scores)
                importance[param] = variance
            else:
                importance[param] = 0.0
        
        # Normalize to 0-100
        max_importance = max(importance.values()) if importance else 1.0
        return {k: (v / max_importance * 100) if max_importance > 0 else 0 
                for k, v in importance.items()}


def define_parameter_space_agent(state: ParameterTuningState) -> ParameterTuningState:
    """Define parameter search space"""
    print("\n🎯 Defining Parameter Search Space...")
    
    model_type = "llm"  # or "ml_classifier"
    strategy = "grid"
    
    param_space = ParameterSpace()
    space = param_space.define_space(model_type)
    combinations = param_space.generate_combinations(space, strategy)
    
    print(f"\n  Model Type: {model_type}")
    print(f"  Search Strategy: {strategy}")
    print(f"  Parameters to Tune: {len(space)}")
    
    print(f"\n  Parameter Ranges:")
    for param, values in space.items():
        print(f"    • {param}: {values[0]} to {values[-1]} ({len(values)} values)")
    
    print(f"\n  Total Configurations: {len(combinations)}")
    
    return {
        **state,
        "parameter_space": {
            "model_type": model_type,
            "space": space,
            "combinations": combinations,
            "strategy": strategy
        },
        "messages": [f"✓ Defined {len(combinations)} parameter configurations"]
    }


def evaluate_parameters_agent(state: ParameterTuningState) -> ParameterTuningState:
    """Evaluate parameter configurations"""
    print("\n🔬 Evaluating Parameter Configurations...")
    
    evaluator = ParameterEvaluator()
    combinations = state["parameter_space"]["combinations"]
    model_type = state["parameter_space"]["model_type"]
    
    results = []
    for config in combinations:
        result = evaluator.evaluate_config(config, model_type)
        results.append(result)
    
    # Calculate statistics
    avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    avg_cost = sum(r["cost"] for r in results) / len(results)
    
    print(f"\n  Configurations Evaluated: {len(results)}")
    print(f"\n  Average Metrics:")
    print(f"    • Accuracy: {avg_accuracy:.1%}")
    print(f"    • Latency: {avg_latency:.1f}ms")
    print(f"    • Cost: ${avg_cost:.4f}")
    
    print(f"\n  Top 3 Configurations:")
    top_3 = sorted(results, key=lambda x: x["composite_score"], reverse=True)[:3]
    for i, result in enumerate(top_3, 1):
        print(f"\n    {i}. Score: {result['composite_score']:.3f}")
        print(f"       Accuracy: {result['accuracy']:.1%}")
        print(f"       Config: {result['config']}")
    
    return {
        **state,
        "tuning_results": results,
        "messages": [f"✓ Evaluated {len(results)} configurations"]
    }


def optimize_parameters_agent(state: ParameterTuningState) -> ParameterTuningState:
    """Optimize parameter selection"""
    print("\n💡 Optimizing Parameter Selection...")
    
    optimizer = ParameterOptimizer()
    optimization = optimizer.find_best_parameters(
        state["tuning_results"],
        optimization_goal="balanced"
    )
    
    best = optimization["best_config"]
    
    print(f"\n  Best Configuration Found:")
    print(f"    Composite Score: {best['composite_score']:.3f}")
    print(f"    Accuracy: {best['accuracy']:.1%}")
    print(f"    Latency: {best['latency_ms']:.1f}ms")
    print(f"    Cost: ${best['cost']:.4f}")
    
    print(f"\n  Parameters:")
    for param, value in best["config"].items():
        print(f"    • {param}: {value}")
    
    print(f"\n  Parameter Importance:")
    for param, importance in sorted(optimization["parameter_importance"].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]:
        print(f"    • {param}: {importance:.1f}/100")
    
    return {
        **state,
        "best_parameters": optimization,
        "messages": [f"✓ Optimized parameters (score: {best['composite_score']:.3f})"]
    }


def generate_tuning_report_agent(state: ParameterTuningState) -> ParameterTuningState:
    """Generate parameter tuning report"""
    print("\n" + "="*70)
    print("PARAMETER TUNING REPORT")
    print("="*70)
    
    param_space = state["parameter_space"]
    print(f"\n🎯 Parameter Search Configuration:")
    print(f"  Model Type: {param_space['model_type']}")
    print(f"  Search Strategy: {param_space['strategy']}")
    print(f"  Configurations Evaluated: {len(state['tuning_results'])}")
    
    print(f"\n  Parameter Space:")
    for param, values in param_space["space"].items():
        print(f"    • {param}: [{values[0]}, ..., {values[-1]}] ({len(values)} values)")
    
    print(f"\n📊 Evaluation Results:")
    results = state["tuning_results"]
    
    print(f"  Performance Statistics:")
    accuracies = [r["accuracy"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    costs = [r["cost"] for r in results]
    
    print(f"\n    Accuracy:")
    print(f"      Mean: {sum(accuracies)/len(accuracies):.1%}")
    print(f"      Best: {max(accuracies):.1%}")
    print(f"      Worst: {min(accuracies):.1%}")
    
    print(f"\n    Latency (ms):")
    print(f"      Mean: {sum(latencies)/len(latencies):.1f}")
    print(f"      Best: {min(latencies):.1f}")
    print(f"      Worst: {max(latencies):.1f}")
    
    print(f"\n    Cost ($):")
    print(f"      Mean: ${sum(costs)/len(costs):.4f}")
    print(f"      Best: ${min(costs):.4f}")
    print(f"      Worst: ${max(costs):.4f}")
    
    print(f"\n🏆 Optimal Configuration:")
    best_params = state["best_parameters"]
    best = best_params["best_config"]
    
    print(f"\n  Overall Score: {best['composite_score']:.3f}")
    print(f"\n  Performance Metrics:")
    print(f"    • Accuracy: {best['accuracy']:.1%}")
    print(f"    • Latency: {best['latency_ms']:.1f}ms")
    print(f"    • Cost: ${best['cost']:.4f}")
    print(f"    • Quality: {best['quality']:.1%}")
    
    print(f"\n  Optimal Parameters:")
    for param, value in best["config"].items():
        print(f"    • {param}: {value}")
    
    print(f"\n📈 Top 5 Configurations:")
    for i, config in enumerate(best_params["top_configs"], 1):
        print(f"\n  {i}. Score: {config['composite_score']:.3f}")
        print(f"     Accuracy: {config['accuracy']:.1%}, Latency: {config['latency_ms']:.1f}ms, Cost: ${config['cost']:.4f}")
        print(f"     Config: {config['config']}")
    
    print(f"\n🔍 Parameter Importance Analysis:")
    print(f"  (Higher score = more impact on performance)\n")
    
    sorted_importance = sorted(best_params["parameter_importance"].items(), 
                              key=lambda x: x[1], reverse=True)
    
    for param, importance in sorted_importance:
        bar_length = int(importance / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {param:.<25} {bar} {importance:>5.1f}/100")
    
    print(f"\n💡 Recommendations:")
    top_params = sorted_importance[:2]
    print(f"  • Focus on tuning: {', '.join(p[0] for p in top_params)}")
    print(f"  • These parameters have the highest impact on performance")
    
    # Compare with baseline (median configuration)
    median_idx = len(results) // 2
    baseline = sorted(results, key=lambda x: x["composite_score"])[median_idx]
    improvement = ((best["composite_score"] - baseline["composite_score"]) / baseline["composite_score"] * 100)
    
    print(f"\n📈 Improvement Over Baseline:")
    print(f"  Baseline Score: {baseline['composite_score']:.3f}")
    print(f"  Optimized Score: {best['composite_score']:.3f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    print(f"\n💡 Parameter Tuning Benefits:")
    print("  • Optimized model performance")
    print("  • Data-driven parameter selection")
    print("  • Identified key parameters")
    print("  • Quantified trade-offs")
    print("  • Repeatable optimization process")
    print("  • Performance baselines established")
    
    print("\n="*70)
    print("✅ Parameter Tuning Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_parameter_tuning_graph():
    workflow = StateGraph(ParameterTuningState)
    workflow.add_node("define_space", define_parameter_space_agent)
    workflow.add_node("evaluate", evaluate_parameters_agent)
    workflow.add_node("optimize", optimize_parameters_agent)
    workflow.add_node("generate_report", generate_tuning_report_agent)
    workflow.add_edge(START, "define_space")
    workflow.add_edge("define_space", "evaluate")
    workflow.add_edge("evaluate", "optimize")
    workflow.add_edge("optimize", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 279: Parameter Tuning MCP Pattern")
    print("="*70)
    
    app = create_parameter_tuning_graph()
    final_state = app.invoke({
        "messages": [],
        "parameter_space": {},
        "tuning_results": [],
        "best_parameters": {},
        "optimization_report": {}
    })
    print("\n✅ Parameter Tuning Pattern Complete!")


if __name__ == "__main__":
    main()
