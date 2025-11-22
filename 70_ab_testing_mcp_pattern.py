"""
A/B Testing MCP Pattern

This pattern demonstrates agents comparing and optimizing alternative approaches
through controlled experimentation and statistical analysis.

Key Features:
- Controlled experiments
- Variant comparison
- Statistical analysis
- Performance optimization
- Data-driven decisions
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ABTestingState(TypedDict):
    """State for A/B testing pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    experiment_name: str
    variant_a: dict[str, str | float]
    variant_b: dict[str, str | float]
    test_results: dict[str, list[float]]
    sample_size: int
    confidence_level: float
    winning_variant: str
    performance_lift: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Experiment Designer
def experiment_designer(state: ABTestingState) -> ABTestingState:
    """Designs A/B test experiment"""
    experiment_name = state.get("experiment_name", "")
    variant_a = state.get("variant_a", {})
    variant_b = state.get("variant_b", {})
    
    system_message = SystemMessage(content="""You are an experiment designer. 
    Design a rigorous A/B test to compare two variants.""")
    
    user_message = HumanMessage(content=f"""Design A/B test:

Experiment: {experiment_name}

Variant A (Control): {variant_a.get('description', 'N/A')}
Variant B (Treatment): {variant_b.get('description', 'N/A')}

Design experiment methodology.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🔬 Experiment Designer:\n{response.content}\n\n✅ A/B test designed")]
    }


# Traffic Splitter
def traffic_splitter(state: ABTestingState) -> ABTestingState:
    """Splits traffic between variants"""
    experiment_name = state.get("experiment_name", "")
    sample_size = state.get("sample_size", 100)
    
    system_message = SystemMessage(content="""You are a traffic splitter. 
    Distribute test traffic evenly between variants.""")
    
    user_message = HumanMessage(content=f"""Split traffic:

Experiment: {experiment_name}
Sample Size: {sample_size}

Allocate traffic 50/50 to variants A and B.""")
    
    response = llm.invoke([system_message, user_message])
    
    split_size = sample_size // 2
    
    return {
        "messages": [AIMessage(content=f"📊 Traffic Splitter:\n{response.content}\n\n✅ Split: {split_size} samples per variant")]
    }


# Variant Tester
def variant_tester(state: ABTestingState) -> ABTestingState:
    """Tests both variants and collects performance data"""
    experiment_name = state.get("experiment_name", "")
    variant_a = state.get("variant_a", {})
    variant_b = state.get("variant_b", {})
    sample_size = state.get("sample_size", 100)
    
    system_message = SystemMessage(content="""You are a variant tester. 
    Run both variants and collect performance metrics.""")
    
    user_message = HumanMessage(content=f"""Test variants:

Experiment: {experiment_name}
Variant A: {variant_a.get('description', 'N/A')}
Variant B: {variant_b.get('description', 'N/A')}
Samples per variant: {sample_size // 2}

Execute test and collect metrics.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate test results
    # Variant A (baseline)
    import random
    random.seed(42)  # Consistent results
    
    baseline_mean = float(variant_a.get("expected_performance", 0.65))
    treatment_mean = float(variant_b.get("expected_performance", 0.72))
    
    results_a = [
        baseline_mean + random.gauss(0, 0.05)
        for _ in range(sample_size // 2)
    ]
    
    # Variant B (typically better)
    results_b = [
        treatment_mean + random.gauss(0, 0.05)
        for _ in range(sample_size // 2)
    ]
    
    test_results = {
        "variant_a": results_a,
        "variant_b": results_b
    }
    
    avg_a = sum(results_a) / len(results_a)
    avg_b = sum(results_b) / len(results_b)
    
    return {
        "messages": [AIMessage(content=f"🧪 Variant Tester:\n{response.content}\n\n✅ Results: A={avg_a:.2%}, B={avg_b:.2%}")],
        "test_results": test_results
    }


# Statistical Analyzer
def statistical_analyzer(state: ABTestingState) -> ABTestingState:
    """Analyzes results statistically"""
    experiment_name = state.get("experiment_name", "")
    test_results = state.get("test_results", {})
    
    system_message = SystemMessage(content="""You are a statistical analyzer. 
    Perform statistical analysis to determine significance.""")
    
    results_a = test_results.get("variant_a", [])
    results_b = test_results.get("variant_b", [])
    
    avg_a = sum(results_a) / len(results_a) if results_a else 0
    avg_b = sum(results_b) / len(results_b) if results_b else 0
    
    user_message = HumanMessage(content=f"""Analyze results:

Experiment: {experiment_name}

Variant A: {len(results_a)} samples, mean={avg_a:.2%}
Variant B: {len(results_b)} samples, mean={avg_b:.2%}

Perform statistical significance testing.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate basic statistics
    # Standard deviation
    def std_dev(values, mean):
        if not values:
            return 0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    std_a = std_dev(results_a, avg_a)
    std_b = std_dev(results_b, avg_b)
    
    # Simple significance test (simplified t-test approximation)
    # In reality would use scipy.stats.ttest_ind
    n_a = len(results_a)
    n_b = len(results_b)
    
    if n_a > 1 and n_b > 1:
        pooled_std = ((std_a ** 2 / n_a) + (std_b ** 2 / n_b)) ** 0.5
        t_stat = abs(avg_b - avg_a) / pooled_std if pooled_std > 0 else 0
        
        # Simplified: t_stat > 2 suggests significance at ~95% confidence
        confidence_level = min(99.9, 50 + (t_stat * 20))
    else:
        confidence_level = 0.0
    
    return {
        "messages": [AIMessage(content=f"📊 Statistical Analyzer:\n{response.content}\n\n✅ Confidence: {confidence_level:.1f}%")],
        "confidence_level": confidence_level
    }


# Winner Selector
def winner_selector(state: ABTestingState) -> ABTestingState:
    """Selects winning variant based on results"""
    experiment_name = state.get("experiment_name", "")
    test_results = state.get("test_results", {})
    confidence_level = state.get("confidence_level", 0.0)
    variant_a = state.get("variant_a", {})
    variant_b = state.get("variant_b", {})
    
    system_message = SystemMessage(content="""You are a winner selector. 
    Determine the winning variant based on statistical evidence.""")
    
    results_a = test_results.get("variant_a", [])
    results_b = test_results.get("variant_b", [])
    
    avg_a = sum(results_a) / len(results_a) if results_a else 0
    avg_b = sum(results_b) / len(results_b) if results_b else 0
    
    user_message = HumanMessage(content=f"""Select winner:

Experiment: {experiment_name}

Variant A: {avg_a:.2%}
Variant B: {avg_b:.2%}
Confidence: {confidence_level:.1f}%

Determine winning variant.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Determine winner
    if confidence_level >= 95.0:
        if avg_b > avg_a:
            winning_variant = "Variant B"
            performance_lift = ((avg_b - avg_a) / avg_a * 100) if avg_a > 0 else 0
        else:
            winning_variant = "Variant A"
            performance_lift = ((avg_a - avg_b) / avg_b * 100) if avg_b > 0 else 0
    else:
        winning_variant = "Inconclusive"
        performance_lift = 0.0
    
    return {
        "messages": [AIMessage(content=f"🏆 Winner Selector:\n{response.content}\n\n✅ Winner: {winning_variant} | Lift: {performance_lift:+.1f}%")],
        "winning_variant": winning_variant,
        "performance_lift": performance_lift
    }


# A/B Test Monitor
def ab_test_monitor(state: ABTestingState) -> ABTestingState:
    """Monitors overall A/B test results"""
    experiment_name = state.get("experiment_name", "")
    variant_a = state.get("variant_a", {})
    variant_b = state.get("variant_b", {})
    test_results = state.get("test_results", {})
    sample_size = state.get("sample_size", 0)
    confidence_level = state.get("confidence_level", 0.0)
    winning_variant = state.get("winning_variant", "")
    performance_lift = state.get("performance_lift", 0.0)
    
    results_a = test_results.get("variant_a", [])
    results_b = test_results.get("variant_b", [])
    
    avg_a = sum(results_a) / len(results_a) if results_a else 0
    avg_b = sum(results_b) / len(results_b) if results_b else 0
    
    # Determine recommendation
    if confidence_level >= 95.0:
        if winning_variant == "Variant B":
            recommendation = f"✅ Roll out Variant B ({performance_lift:+.1f}% improvement)"
        else:
            recommendation = "✅ Keep Variant A (control is better)"
    else:
        recommendation = "⚠️ Inconclusive - need more data or redesign test"
    
    summary = f"""
    ✅ A/B TESTING PATTERN COMPLETE
    
    Experiment Summary:
    • Name: {experiment_name}
    • Sample Size: {sample_size} ({sample_size // 2} per variant)
    • Confidence Level: {confidence_level:.1f}%
    • Winner: {winning_variant}
    • Performance Lift: {performance_lift:+.1f}%
    
    Variant A (Control):
    • Description: {variant_a.get('description', 'N/A')}
    • Performance: {avg_a:.2%}
    • Samples: {len(results_a)}
    
    Variant B (Treatment):
    • Description: {variant_b.get('description', 'N/A')}
    • Performance: {avg_b:.2%}
    • Samples: {len(results_b)}
    
    Statistical Analysis:
    • Confidence Level: {confidence_level:.1f}%
    • Significance Threshold: 95%
    • Result: {"Statistically Significant" if confidence_level >= 95 else "Not Significant"}
    
    A/B Testing Process:
    1. Design Experiment → 2. Split Traffic → 3. Test Variants → 
    4. Analyze Statistically → 5. Select Winner → 6. Implement
    
    A/B Testing Benefits:
    • Data-driven decisions
    • Controlled experimentation
    • Statistical rigor
    • Risk mitigation
    • Performance optimization
    • Continuous improvement
    
    Recommendation:
    {recommendation}
    
    Key Insights:
    • Variant B performed {performance_lift:+.1f}% better than Variant A
    • Results are {"statistically significant" if confidence_level >= 95 else "not yet significant"}
    • Confidence level: {confidence_level:.1f}% (target: 95%+)
    • Sample size: {sample_size} total observations
    """
    
    return {
        "messages": [AIMessage(content=f"📊 A/B Test Monitor:\n{summary}")]
    }


# Build the graph
def build_ab_testing_graph():
    """Build the A/B testing pattern graph"""
    workflow = StateGraph(ABTestingState)
    
    workflow.add_node("designer", experiment_designer)
    workflow.add_node("splitter", traffic_splitter)
    workflow.add_node("tester", variant_tester)
    workflow.add_node("analyzer", statistical_analyzer)
    workflow.add_node("selector", winner_selector)
    workflow.add_node("monitor", ab_test_monitor)
    
    workflow.add_edge(START, "designer")
    workflow.add_edge("designer", "splitter")
    workflow.add_edge("splitter", "tester")
    workflow.add_edge("tester", "analyzer")
    workflow.add_edge("analyzer", "selector")
    workflow.add_edge("selector", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_ab_testing_graph()
    
    print("=== A/B Testing MCP Pattern ===\n")
    
    # Example: Testing two recommendation algorithms
    state = {
        "messages": [],
        "experiment_name": "Recommendation Algorithm Optimization",
        "variant_a": {
            "description": "Collaborative Filtering (baseline)",
            "expected_performance": 0.65
        },
        "variant_b": {
            "description": "Hybrid (CF + Content-based)",
            "expected_performance": 0.72
        },
        "test_results": {},
        "sample_size": 200,
        "confidence_level": 0.0,
        "winning_variant": "",
        "performance_lift": 0.0
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("A/B TEST COMPLETE")
    print("=" * 70)
    print(f"\nExperiment: {state['experiment_name']}")
    print(f"Winner: {result['winning_variant']}")
    print(f"Performance Lift: {result['performance_lift']:+.1f}%")
    print(f"Confidence: {result['confidence_level']:.1f}%")
    print(f"Sample Size: {state['sample_size']}")
