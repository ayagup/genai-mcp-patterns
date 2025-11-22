"""
Active Learning MCP Pattern

This pattern demonstrates agents strategically selecting the most informative
examples for training to maximize learning efficiency with minimal labeled data.

Key Features:
- Strategic sample selection
- Uncertainty-based selection
- Query strategies
- Iterative learning
- Efficient labeling
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ActiveLearningState(TypedDict):
    """State for active learning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    unlabeled_pool: list[str]
    labeled_data: list[dict[str, str]]
    current_model_accuracy: float
    uncertainty_scores: list[dict[str, float]]
    selected_samples: list[str]
    iteration: int
    total_queries: int


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Uncertainty Estimator
def uncertainty_estimator(state: ActiveLearningState) -> ActiveLearningState:
    """Estimates uncertainty for unlabeled samples"""
    task = state.get("task", "")
    unlabeled_pool = state.get("unlabeled_pool", [])
    labeled_data = state.get("labeled_data", [])
    iteration = state.get("iteration", 0)
    
    system_message = SystemMessage(content="""You are an uncertainty estimator. 
    Evaluate how uncertain the current model would be about each unlabeled sample.""")
    
    labeled_summary = f"{len(labeled_data)} labeled examples" if labeled_data else "No labeled data yet"
    unlabeled_summary = "\n".join([f"  {i+1}. {sample[:60]}..." for i, sample in enumerate(unlabeled_pool[:5])])
    
    user_message = HumanMessage(content=f"""Estimate uncertainty:

Task: {task}
Training Data: {labeled_summary}
Unlabeled Pool: {len(unlabeled_pool)} samples

Sample unlabeled data:
{unlabeled_summary}

Identify which samples the model would be most uncertain about.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate uncertainty scores (higher = more uncertain)
    # In real scenario, would use model confidence scores
    uncertainty_scores = []
    for i, sample in enumerate(unlabeled_pool):
        # Vary uncertainty to simulate different confidence levels
        base_uncertainty = 0.5
        variation = (hash(sample) % 50) / 100.0  # Pseudo-random but consistent
        uncertainty = base_uncertainty + variation
        
        uncertainty_scores.append({
            "sample": sample,
            "uncertainty": uncertainty,
            "index": i
        })
    
    # Sort by uncertainty (descending)
    uncertainty_scores.sort(key=lambda x: x["uncertainty"], reverse=True)
    
    avg_uncertainty = sum(s["uncertainty"] for s in uncertainty_scores) / len(uncertainty_scores) if uncertainty_scores else 0
    
    return {
        "messages": [AIMessage(content=f"📊 Uncertainty Estimator (Iteration {iteration + 1}):\n{response.content}\n\n✅ Scored {len(uncertainty_scores)} samples | Avg Uncertainty: {avg_uncertainty:.2f}")],
        "uncertainty_scores": uncertainty_scores
    }


# Query Strategy Selector
def query_strategy_selector(state: ActiveLearningState) -> ActiveLearningState:
    """Selects samples using active learning query strategy"""
    task = state.get("task", "")
    uncertainty_scores = state.get("uncertainty_scores", [])
    iteration = state.get("iteration", 0)
    
    system_message = SystemMessage(content="""You are a query strategy selector. 
    Choose the most informative samples for labeling using active learning strategies.""")
    
    top_uncertain = "\n".join([
        f"  {i+1}. {s['sample'][:50]}... (uncertainty: {s['uncertainty']:.2f})"
        for i, s in enumerate(uncertainty_scores[:5])
    ])
    
    user_message = HumanMessage(content=f"""Select samples for labeling:

Strategy: Uncertainty Sampling
Available: {len(uncertainty_scores)} samples

Top uncertain samples:
{top_uncertain}

Select the most informative samples to label.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Select top-k most uncertain samples
    k = 3  # Query 3 samples per iteration
    selected_samples = [s["sample"] for s in uncertainty_scores[:k]]
    
    return {
        "messages": [AIMessage(content=f"🎯 Query Strategy Selector:\n{response.content}\n\n✅ Selected {len(selected_samples)} samples for labeling")],
        "selected_samples": selected_samples
    }


# Oracle Labeler
def oracle_labeler(state: ActiveLearningState) -> ActiveLearningState:
    """Labels selected samples (simulates human expert)"""
    task = state.get("task", "")
    selected_samples = state.get("selected_samples", [])
    labeled_data = state.get("labeled_data", [])
    iteration = state.get("iteration", 0)
    
    system_message = SystemMessage(content="""You are an oracle labeler (expert). 
    Provide accurate labels for the selected samples.""")
    
    samples_text = "\n".join([f"  {i+1}. {sample[:60]}..." for i, sample in enumerate(selected_samples)])
    
    user_message = HumanMessage(content=f"""Label selected samples:

Task: {task}

Samples to label:
{samples_text}

Provide accurate labels.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate labeling based on task type
    new_labels = []
    if "sentiment" in task.lower():
        labels = ["positive", "negative", "neutral"]
    elif "category" in task.lower():
        labels = ["category_A", "category_B", "category_C"]
    else:
        labels = ["class_1", "class_2", "class_3"]
    
    for i, sample in enumerate(selected_samples):
        label = labels[hash(sample) % len(labels)]  # Pseudo-random but consistent
        new_labels.append({
            "text": sample,
            "label": label,
            "iteration": iteration + 1
        })
    
    # Add to labeled data
    labeled_data = labeled_data + new_labels
    
    return {
        "messages": [AIMessage(content=f"👨‍🏫 Oracle Labeler:\n{response.content}\n\n✅ Labeled {len(new_labels)} samples | Total labeled: {len(labeled_data)}")],
        "labeled_data": labeled_data
    }


# Model Updater
def model_updater(state: ActiveLearningState) -> ActiveLearningState:
    """Updates model with newly labeled data"""
    task = state.get("task", "")
    labeled_data = state.get("labeled_data", [])
    selected_samples = state.get("selected_samples", [])
    unlabeled_pool = state.get("unlabeled_pool", [])
    iteration = state.get("iteration", 0)
    
    system_message = SystemMessage(content="""You are a model updater. 
    Retrain the model with newly labeled data.""")
    
    user_message = HumanMessage(content=f"""Update model:

Task: {task}
Training Data: {len(labeled_data)} labeled samples
New Samples: {len(selected_samples)}

Retrain model with augmented dataset.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Remove selected samples from unlabeled pool
    unlabeled_pool = [s for s in unlabeled_pool if s not in selected_samples]
    
    # Simulate accuracy improvement (more data = better accuracy)
    base_accuracy = 0.60
    data_bonus = min(0.30, len(labeled_data) * 0.02)
    current_model_accuracy = base_accuracy + data_bonus
    
    return {
        "messages": [AIMessage(content=f"🔄 Model Updater:\n{response.content}\n\n✅ Model updated | Accuracy: {current_model_accuracy:.1%} | Unlabeled remaining: {len(unlabeled_pool)}")],
        "unlabeled_pool": unlabeled_pool,
        "current_model_accuracy": current_model_accuracy,
        "iteration": iteration + 1,
        "total_queries": state.get("total_queries", 0) + len(selected_samples)
    }


# Active Learning Monitor
def active_learning_monitor(state: ActiveLearningState) -> ActiveLearningState:
    """Monitors overall active learning process"""
    task = state.get("task", "")
    labeled_data = state.get("labeled_data", [])
    unlabeled_pool = state.get("unlabeled_pool", [])
    current_model_accuracy = state.get("current_model_accuracy", 0.0)
    iteration = state.get("iteration", 0)
    total_queries = state.get("total_queries", 0)
    selected_samples = state.get("selected_samples", [])
    
    # Calculate efficiency metrics
    initial_unlabeled = len(unlabeled_pool) + total_queries
    labeling_efficiency = (total_queries / initial_unlabeled * 100) if initial_unlabeled > 0 else 0
    
    recent_labels = "\n".join([
        f"    {i+1}. {item['text'][:50]}... → {item['label']}"
        for i, item in enumerate(labeled_data[-3:])
    ])
    
    summary = f"""
    ✅ ACTIVE LEARNING PATTERN - Iteration {iteration}
    
    Learning Summary:
    • Task: {task}
    • Current Iteration: {iteration}
    • Total Queries Made: {total_queries}
    • Labeled Data: {len(labeled_data)} samples
    • Unlabeled Pool: {len(unlabeled_pool)} samples
    • Model Accuracy: {current_model_accuracy:.1%}
    
    Efficiency Metrics:
    • Samples Queried: {total_queries} of {initial_unlabeled} ({labeling_efficiency:.1f}%)
    • Queries This Round: {len(selected_samples)}
    • Labeling Efficiency: High (strategic selection)
    • Accuracy Gain: {current_model_accuracy:.1%}
    
    Recent Labels:
{recent_labels if recent_labels else "    None yet"}
    
    Active Learning Cycle:
    1. Estimate Uncertainty → 2. Select Query Strategy → 
    3. Label Selected Samples → 4. Update Model → 5. Repeat
    
    Active Learning Benefits:
    • Minimizes labeling effort
    • Strategic sample selection
    • Faster model improvement
    • Cost-effective learning
    • Focuses on informative data
    • Efficient use of expert time
    
    Query Strategy: Uncertainty Sampling
    • Selects samples model is most uncertain about
    • Maximizes information gain per query
    • Reduces total labeling requirements
    
    Progress:
    • {labeling_efficiency:.1f}% of data labeled
    • {current_model_accuracy:.1%} accuracy achieved
    • {len(unlabeled_pool)} samples remaining
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Active Learning Monitor:\n{summary}")]
    }


# Build the graph
def build_active_learning_graph():
    """Build the active learning pattern graph"""
    workflow = StateGraph(ActiveLearningState)
    
    workflow.add_node("estimator", uncertainty_estimator)
    workflow.add_node("selector", query_strategy_selector)
    workflow.add_node("labeler", oracle_labeler)
    workflow.add_node("updater", model_updater)
    workflow.add_node("monitor", active_learning_monitor)
    
    workflow.add_edge(START, "estimator")
    workflow.add_edge("estimator", "selector")
    workflow.add_edge("selector", "labeler")
    workflow.add_edge("labeler", "updater")
    workflow.add_edge("updater", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage - Multiple active learning iterations
if __name__ == "__main__":
    graph = build_active_learning_graph()
    
    print("=== Active Learning MCP Pattern ===\n")
    
    # Initial state
    state = {
        "messages": [],
        "task": "Sentiment Classification",
        "unlabeled_pool": [
            "This is an excellent product!",
            "Terrible quality, very disappointed.",
            "It's okay, nothing special.",
            "Best purchase I've ever made!",
            "Waste of money, do not buy.",
            "Pretty average experience.",
            "Absolutely love it!",
            "Not worth the price.",
            "Decent quality for the cost.",
            "Would not recommend to anyone.",
            "Exceeded my expectations!",
            "Just mediocre, expected better.",
            "Outstanding service and quality!",
            "Horrible experience overall.",
            "It works as expected."
        ],
        "labeled_data": [],
        "current_model_accuracy": 0.0,
        "uncertainty_scores": [],
        "selected_samples": [],
        "iteration": 0,
        "total_queries": 0
    }
    
    # Run multiple active learning iterations
    for i in range(4):
        print(f"\n{'=' * 70}")
        print(f"ACTIVE LEARNING ITERATION {i + 1}")
        print('=' * 70)
        
        result = graph.invoke(state)
        
        # Show messages for this iteration
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Update state for next iteration
        state = {
            "messages": [],
            "task": state["task"],
            "unlabeled_pool": result.get("unlabeled_pool", []),
            "labeled_data": result.get("labeled_data", []),
            "current_model_accuracy": result.get("current_model_accuracy", 0.0),
            "uncertainty_scores": [],
            "selected_samples": [],
            "iteration": result.get("iteration", i + 1),
            "total_queries": result.get("total_queries", 0)
        }
        
        # Stop if unlabeled pool is exhausted
        if not state["unlabeled_pool"]:
            print("\n✅ All samples labeled!")
            break
    
    print(f"\n\n{'=' * 70}")
    print("FINAL ACTIVE LEARNING RESULTS")
    print('=' * 70)
    print(f"\nTotal Iterations: {state['iteration']}")
    print(f"Total Queries: {state['total_queries']}")
    print(f"Labeled Samples: {len(state['labeled_data'])}")
    print(f"Final Accuracy: {state['current_model_accuracy']:.1%}")
    print(f"Unlabeled Remaining: {len(state['unlabeled_pool'])}")
