"""
Meta-Learning MCP Pattern

This pattern demonstrates agents learning how to learn more effectively,
adapting their learning strategies based on experience across multiple tasks.

Key Features:
- Learning strategy adaptation
- Cross-task learning
- Fast adaptation
- Learning efficiency optimization
- Meta-knowledge accumulation
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class MetaLearningState(TypedDict):
    """State for meta-learning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task_family: str
    tasks_completed: list[dict[str, str | float]]
    learning_strategies: dict[str, dict[str, str | float]]
    meta_knowledge: dict[str, list[str]]
    current_task: str
    adaptation_speed: float
    meta_performance: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Task Analyzer
def task_analyzer(state: MetaLearningState) -> MetaLearningState:
    """Analyzes new task to identify similarities with previous tasks"""
    task_family = state.get("task_family", "")
    current_task = state.get("current_task", "")
    tasks_completed = state.get("tasks_completed", [])
    
    system_message = SystemMessage(content="""You are a task analyzer. 
    Analyze the new task and identify similarities with previously learned tasks.""")
    
    previous_tasks = "\n".join([
        f"  • {task['name']}: Performance {task.get('performance', 0):.1%}"
        for task in tasks_completed[-3:]
    ]) if tasks_completed else "  • No previous tasks"
    
    user_message = HumanMessage(content=f"""Analyze new task:

Task Family: {task_family}
Current Task: {current_task}
Previous Experience: {len(tasks_completed)} tasks

Recent Tasks:
{previous_tasks}

Identify patterns and similarities.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🔍 Task Analyzer:\n{response.content}\n\n✅ Experience: {len(tasks_completed)} similar tasks")]
    }


# Strategy Selector
def strategy_selector(state: MetaLearningState) -> MetaLearningState:
    """Selects optimal learning strategy based on meta-knowledge"""
    task_family = state.get("task_family", "")
    current_task = state.get("current_task", "")
    learning_strategies = state.get("learning_strategies", {})
    meta_knowledge = state.get("meta_knowledge", {})
    
    system_message = SystemMessage(content="""You are a strategy selector. 
    Choose the most effective learning strategy based on past experience.""")
    
    strategies_summary = "\n".join([
        f"  • {name}: Success rate {info.get('success_rate', 0):.1%}"
        for name, info in learning_strategies.items()
    ]) if learning_strategies else "  • No strategies yet"
    
    knowledge_summary = "\n".join([
        f"  • {category}: {len(items)} insights"
        for category, items in meta_knowledge.items()
    ]) if meta_knowledge else "  • No meta-knowledge yet"
    
    user_message = HumanMessage(content=f"""Select learning strategy:

Task: {current_task}
Task Family: {task_family}

Available Strategies:
{strategies_summary}

Meta-Knowledge:
{knowledge_summary}

Choose optimal strategy.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Select or create strategy
    if learning_strategies:
        # Choose best performing strategy
        best_strategy = max(
            learning_strategies.keys(),
            key=lambda k: learning_strategies[k].get("success_rate", 0.0)
        )
    else:
        best_strategy = "adaptive_learning"
    
    return {
        "messages": [AIMessage(content=f"🎯 Strategy Selector:\n{response.content}\n\n✅ Selected: {best_strategy}")]
    }


# Fast Adapter
def fast_adapter(state: MetaLearningState) -> MetaLearningState:
    """Adapts quickly to new task using meta-learned knowledge"""
    current_task = state.get("current_task", "")
    tasks_completed = state.get("tasks_completed", [])
    meta_knowledge = state.get("meta_knowledge", {})
    
    system_message = SystemMessage(content="""You are a fast adapter. 
    Apply meta-learned knowledge to quickly adapt to the new task.""")
    
    relevant_knowledge = "\n".join([
        f"  • {item}"
        for items in meta_knowledge.values()
        for item in items[:2]
    ]) if meta_knowledge else "  • Learning from scratch"
    
    user_message = HumanMessage(content=f"""Adapt to task: {current_task}

Meta-Knowledge Available:
{relevant_knowledge}

Previous Tasks: {len(tasks_completed)}

Apply meta-knowledge for fast adaptation.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate adaptation speed (improves with experience)
    base_speed = 0.5
    experience_bonus = min(0.4, len(tasks_completed) * 0.05)
    knowledge_bonus = min(0.1, sum(len(items) for items in meta_knowledge.values()) * 0.01)
    adaptation_speed = base_speed + experience_bonus + knowledge_bonus
    
    return {
        "messages": [AIMessage(content=f"⚡ Fast Adapter:\n{response.content}\n\n✅ Adaptation Speed: {adaptation_speed:.1%} (faster with experience)")],
        "adaptation_speed": adaptation_speed
    }


# Performance Evaluator
def performance_evaluator(state: MetaLearningState) -> MetaLearningState:
    """Evaluates performance on current task"""
    current_task = state.get("current_task", "")
    tasks_completed = state.get("tasks_completed", [])
    adaptation_speed = state.get("adaptation_speed", 0.5)
    
    system_message = SystemMessage(content="""You are a performance evaluator. 
    Assess how well the agent performed on the current task.""")
    
    user_message = HumanMessage(content=f"""Evaluate performance:

Task: {current_task}
Adaptation Speed: {adaptation_speed:.1%}
Task Experience: {len(tasks_completed)} previous tasks

Assess task performance.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate task performance (improves with meta-learning)
    base_performance = 0.65
    meta_learning_bonus = min(0.25, len(tasks_completed) * 0.03)
    adaptation_bonus = adaptation_speed * 0.1
    task_performance = base_performance + meta_learning_bonus + adaptation_bonus
    
    # Record completed task
    task_record = {
        "name": current_task,
        "performance": task_performance,
        "adaptation_speed": adaptation_speed,
        "task_number": str(len(tasks_completed) + 1)
    }
    
    return {
        "messages": [AIMessage(content=f"📊 Performance Evaluator:\n{response.content}\n\n✅ Performance: {task_performance:.1%}")],
        "tasks_completed": [task_record]
    }


# Meta-Knowledge Updater
def meta_knowledge_updater(state: MetaLearningState) -> MetaLearningState:
    """Updates meta-knowledge based on learning experience"""
    task_family = state.get("task_family", "")
    tasks_completed = state.get("tasks_completed", [])
    meta_knowledge = state.get("meta_knowledge", {})
    learning_strategies = state.get("learning_strategies", {})
    
    system_message = SystemMessage(content="""You are a meta-knowledge updater. 
    Extract meta-level insights from the learning experience.""")
    
    latest_task = tasks_completed[-1] if tasks_completed else {}
    
    user_message = HumanMessage(content=f"""Update meta-knowledge:

Task Family: {task_family}
Latest Task: {latest_task.get('name', 'N/A')}
Performance: {latest_task.get('performance', 0):.1%}
Total Tasks: {len(tasks_completed)}

Extract meta-level learning insights.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Update meta-knowledge
    if "learning_patterns" not in meta_knowledge:
        meta_knowledge["learning_patterns"] = []
    if "adaptation_strategies" not in meta_knowledge:
        meta_knowledge["adaptation_strategies"] = []
    if "task_similarities" not in meta_knowledge:
        meta_knowledge["task_similarities"] = []
    
    # Add new meta-knowledge
    if len(tasks_completed) >= 2:
        meta_knowledge["learning_patterns"].append(
            f"Pattern {len(tasks_completed)}: Faster learning with more task experience"
        )
    if len(tasks_completed) >= 3:
        meta_knowledge["adaptation_strategies"].append(
            f"Strategy {len(tasks_completed)}: Reuse successful approaches from similar tasks"
        )
    if len(tasks_completed) >= 4:
        meta_knowledge["task_similarities"].append(
            f"Similarity {len(tasks_completed)}: {task_family} tasks share common structure"
        )
    
    # Update strategy success rates
    if "adaptive_learning" not in learning_strategies:
        learning_strategies["adaptive_learning"] = {
            "name": "adaptive_learning",
            "success_rate": 0.0,
            "uses": 0.0
        }
    
    strategy = learning_strategies["adaptive_learning"]
    current_uses = float(strategy.get("uses", 0))
    current_success = float(strategy.get("success_rate", 0))
    latest_perf = float(latest_task.get("performance", 0))
    
    new_uses = current_uses + 1
    new_success = (current_success * current_uses + latest_perf) / new_uses
    
    learning_strategies["adaptive_learning"] = {
        "name": "adaptive_learning",
        "success_rate": new_success,
        "uses": new_uses
    }
    
    total_meta = sum(len(items) for items in meta_knowledge.values())
    
    return {
        "messages": [AIMessage(content=f"🧠 Meta-Knowledge Updater:\n{response.content}\n\n✅ Meta-knowledge: {total_meta} insights | Strategies: {len(learning_strategies)}")],
        "meta_knowledge": meta_knowledge,
        "learning_strategies": learning_strategies
    }


# Meta-Learning Monitor
def meta_learning_monitor(state: MetaLearningState) -> MetaLearningState:
    """Monitors overall meta-learning progress"""
    task_family = state.get("task_family", "")
    current_task = state.get("current_task", "")
    tasks_completed = state.get("tasks_completed", [])
    meta_knowledge = state.get("meta_knowledge", {})
    learning_strategies = state.get("learning_strategies", {})
    adaptation_speed = state.get("adaptation_speed", 0.0)
    
    # Calculate meta-performance (improvement over tasks)
    if len(tasks_completed) >= 2:
        first_perf = float(tasks_completed[0].get("performance", 0))
        latest_perf = float(tasks_completed[-1].get("performance", 0))
        meta_performance = ((latest_perf - first_perf) / first_perf * 100) if first_perf > 0 else 0
    else:
        meta_performance = 0.0
    
    tasks_summary = "\n".join([
        f"    {i+1}. {task['name']}: {task.get('performance', 0):.1%} (speed: {task.get('adaptation_speed', 0):.1%})"
        for i, task in enumerate(tasks_completed[-5:])
    ])
    
    meta_knowledge_summary = "\n".join([
        f"    • {category.replace('_', ' ').title()}: {len(items)} insights"
        for category, items in meta_knowledge.items()
    ])
    
    strategies_summary = "\n".join([
        f"    • {name}: {info.get('success_rate', 0):.1%} success ({int(info.get('uses', 0))} uses)"
        for name, info in learning_strategies.items()
    ])
    
    avg_performance = sum(float(t.get("performance", 0)) for t in tasks_completed) / len(tasks_completed) if tasks_completed else 0
    
    summary = f"""
    ✅ META-LEARNING PATTERN COMPLETE
    
    Meta-Learning Summary:
    • Task Family: {task_family}
    • Tasks Completed: {len(tasks_completed)}
    • Meta-Performance Improvement: {meta_performance:+.1f}%
    • Average Performance: {avg_performance:.1%}
    • Current Adaptation Speed: {adaptation_speed:.1%}
    
    Recent Tasks:
{tasks_summary if tasks_summary else "    • No tasks yet"}
    
    Meta-Knowledge Acquired:
{meta_knowledge_summary if meta_knowledge_summary else "    • None yet"}
    
    Learning Strategies:
{strategies_summary if strategies_summary else "    • None yet"}
    
    Meta-Learning Process:
    1. Analyze Task → 2. Select Strategy → 3. Fast Adapt → 
    4. Evaluate Performance → 5. Update Meta-Knowledge → 6. Repeat
    
    Meta-Learning Benefits:
    • Learn how to learn
    • Fast task adaptation
    • Cross-task knowledge transfer
    • Improved learning efficiency
    • Strategy optimization
    • Cumulative improvement
    
    Key Insights:
    • Adaptation speed improved from ~50% to {adaptation_speed:.1%}
    • Performance improved {meta_performance:+.1f}% across tasks
    • {sum(len(items) for items in meta_knowledge.values())} meta-insights gained
    • {len(learning_strategies)} learning strategies optimized
    
    Meta-Learning Power:
    The more tasks completed, the faster and better the agent learns new tasks
    in the same family. This demonstrates learning-to-learn capability.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Meta-Learning Monitor:\n{summary}")],
        "meta_performance": meta_performance
    }


# Build the graph
def build_meta_learning_graph():
    """Build the meta-learning pattern graph"""
    workflow = StateGraph(MetaLearningState)
    
    workflow.add_node("analyzer", task_analyzer)
    workflow.add_node("selector", strategy_selector)
    workflow.add_node("adapter", fast_adapter)
    workflow.add_node("evaluator", performance_evaluator)
    workflow.add_node("updater", meta_knowledge_updater)
    workflow.add_node("monitor", meta_learning_monitor)
    
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "selector")
    workflow.add_edge("selector", "adapter")
    workflow.add_edge("adapter", "evaluator")
    workflow.add_edge("evaluator", "updater")
    workflow.add_edge("updater", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage - Multiple tasks to demonstrate meta-learning
if __name__ == "__main__":
    graph = build_meta_learning_graph()
    
    print("=== Meta-Learning MCP Pattern ===\n")
    
    task_family = "Text Classification"
    tasks = [
        "Sentiment Analysis",
        "Spam Detection",
        "Topic Classification",
        "Intent Recognition",
        "Emotion Detection"
    ]
    
    # Initial state
    state = {
        "messages": [],
        "task_family": task_family,
        "tasks_completed": [],
        "learning_strategies": {},
        "meta_knowledge": {},
        "current_task": "",
        "adaptation_speed": 0.0,
        "meta_performance": 0.0
    }
    
    # Learn multiple tasks
    for i, task in enumerate(tasks):
        print(f"\n{'=' * 70}")
        print(f"META-LEARNING TASK {i + 1}: {task}")
        print('=' * 70)
        
        state["current_task"] = task
        
        result = graph.invoke(state)
        
        # Show messages for this task
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Update state for next task
        state = {
            "messages": [],
            "task_family": task_family,
            "tasks_completed": result.get("tasks_completed", []),
            "learning_strategies": result.get("learning_strategies", {}),
            "meta_knowledge": result.get("meta_knowledge", {}),
            "current_task": "",
            "adaptation_speed": result.get("adaptation_speed", 0.0),
            "meta_performance": result.get("meta_performance", 0.0)
        }
    
    print(f"\n\n{'=' * 70}")
    print("FINAL META-LEARNING RESULTS")
    print('=' * 70)
    print(f"\nTask Family: {task_family}")
    print(f"Tasks Completed: {len(state['tasks_completed'])}")
    print(f"Meta-Performance Improvement: {state['meta_performance']:+.1f}%")
    print(f"Final Adaptation Speed: {state['adaptation_speed']:.1%}")
    print(f"Meta-Knowledge Items: {sum(len(items) for items in state['meta_knowledge'].values())}")
    print(f"Learning Strategies: {len(state['learning_strategies'])}")
