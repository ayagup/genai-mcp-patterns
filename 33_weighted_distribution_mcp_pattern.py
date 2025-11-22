"""
Weighted Distribution MCP Pattern

This pattern demonstrates distributing tasks based on agent capacity weights,
allocating more tasks to higher-capacity agents.

Key Features:
- Weight-based task allocation
- Capacity-aware distribution
- Proportional task assignment
- Optimizes resource utilization
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class WeightedDistributionState(TypedDict):
    """State for weighted distribution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: list[dict[str, str]]
    agent_weights: dict[str, int]  # agent -> capacity weight
    task_assignments: dict[str, list[str]]
    completed_tasks: list[str]
    results: dict[str, str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Weighted Distributor
def weighted_distributor(state: WeightedDistributionState) -> WeightedDistributionState:
    """Distributes tasks based on agent weights"""
    tasks = state.get("tasks", [])
    agent_weights = state.get("agent_weights", {})
    task_assignments = state.get("task_assignments", {name: [] for name in agent_weights.keys()})
    
    system_message = SystemMessage(content="""You are a weighted task distributor. Allocate tasks 
    to agents proportionally based on their capacity weights. Higher weight agents receive more tasks.""")
    
    tasks_info = "\n".join([f"Task {t['id']}: {t['description']}" for t in tasks])
    weights_info = "\n".join([f"{agent}: weight {weight}" for agent, weight in agent_weights.items()])
    
    user_message = HumanMessage(content=f"""Distribute these tasks:
    
{tasks_info}

Agent Capacities:
{weights_info}

Allocate tasks proportionally based on weights.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate total weight
    total_weight = sum(agent_weights.values())
    
    # Distribute tasks proportionally
    agent_list = list(agent_weights.keys())
    task_index = 0
    
    for agent in agent_list:
        # Calculate number of tasks for this agent
        agent_proportion = agent_weights[agent] / total_weight
        num_tasks = int(len(tasks) * agent_proportion)
        
        # Assign tasks
        for _ in range(num_tasks):
            if task_index < len(tasks):
                task_assignments[agent].append(tasks[task_index]['id'])
                task_index += 1
    
    # Distribute remaining tasks
    while task_index < len(tasks):
        for agent in agent_list:
            if task_index < len(tasks):
                task_assignments[agent].append(tasks[task_index]['id'])
                task_index += 1
    
    assignment_summary = "\n".join([
        f"{agent} (weight={agent_weights[agent]}): {len(tasks)} tasks"
        for agent, tasks in task_assignments.items() if tasks
    ])
    
    return {
        "messages": [AIMessage(content=f"⚖️ Weighted Distributor: {response.content}\n\nAssignments:\n{assignment_summary}")],
        "task_assignments": task_assignments
    }


# High-Capacity Worker
def high_capacity_worker(state: WeightedDistributionState) -> WeightedDistributionState:
    """High-capacity worker (weight=5)"""
    assignments = state.get("task_assignments", {}).get("high_capacity", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="🚀 High-Capacity Worker: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are a high-capacity worker with maximum processing 
    power. You can handle large workloads efficiently.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these tasks at high capacity:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"High-Capacity: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"🚀 High-Capacity Worker: {response.content}\n\nProcessed {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Medium-Capacity Worker
def medium_capacity_worker(state: WeightedDistributionState) -> WeightedDistributionState:
    """Medium-capacity worker (weight=3)"""
    assignments = state.get("task_assignments", {}).get("medium_capacity", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="⚡ Medium-Capacity Worker: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are a medium-capacity worker with balanced 
    processing power. You handle moderate workloads effectively.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these tasks at medium capacity:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Medium-Capacity: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"⚡ Medium-Capacity Worker: {response.content}\n\nProcessed {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Low-Capacity Worker
def low_capacity_worker(state: WeightedDistributionState) -> WeightedDistributionState:
    """Low-capacity worker (weight=2)"""
    assignments = state.get("task_assignments", {}).get("low_capacity", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="🔋 Low-Capacity Worker: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are a low-capacity worker with limited 
    processing power. You handle smaller workloads carefully.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these tasks at low capacity:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Low-Capacity: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"🔋 Low-Capacity Worker: {response.content}\n\nProcessed {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Result Analyzer
def result_analyzer(state: WeightedDistributionState) -> WeightedDistributionState:
    """Analyzes weighted distribution results"""
    task_assignments = state.get("task_assignments", {})
    agent_weights = state.get("agent_weights", {})
    
    distribution = "\n".join([
        f"  {agent} (weight={agent_weights.get(agent, 0)}): {len(tasks)} tasks ({len(tasks)/sum(len(t) for t in task_assignments.values())*100:.1f}%)"
        for agent, tasks in task_assignments.items()
    ])
    
    summary = f"""
    ✅ WEIGHTED DISTRIBUTION COMPLETE
    
    Total Tasks: {sum(len(t) for t in task_assignments.values())}
    Total Weight: {sum(agent_weights.values())}
    
    Task Allocation:
{distribution}
    
    Tasks allocated proportionally based on agent capacity weights.
    Higher capacity agents received more tasks as expected.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Result Analyzer:\n{summary}")]
    }


# Build the graph
def build_weighted_distribution_graph():
    """Build the weighted distribution MCP pattern graph"""
    workflow = StateGraph(WeightedDistributionState)
    
    workflow.add_node("distributor", weighted_distributor)
    workflow.add_node("high_capacity", high_capacity_worker)
    workflow.add_node("medium_capacity", medium_capacity_worker)
    workflow.add_node("low_capacity", low_capacity_worker)
    workflow.add_node("analyzer", result_analyzer)
    
    workflow.add_edge(START, "distributor")
    workflow.add_edge("distributor", "high_capacity")
    workflow.add_edge("distributor", "medium_capacity")
    workflow.add_edge("distributor", "low_capacity")
    workflow.add_edge("high_capacity", "analyzer")
    workflow.add_edge("medium_capacity", "analyzer")
    workflow.add_edge("low_capacity", "analyzer")
    workflow.add_edge("analyzer", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_weighted_distribution_graph()
    
    print("=== Weighted Distribution MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "tasks": [
            {"id": "task_1", "description": "Process large dataset"},
            {"id": "task_2", "description": "Analyze user behavior"},
            {"id": "task_3", "description": "Generate reports"},
            {"id": "task_4", "description": "Send notifications"},
            {"id": "task_5", "description": "Update database"},
            {"id": "task_6", "description": "Backup files"},
            {"id": "task_7", "description": "Optimize queries"},
            {"id": "task_8", "description": "Clean logs"},
            {"id": "task_9", "description": "Index data"},
            {"id": "task_10", "description": "Validate records"},
        ],
        "agent_weights": {
            "high_capacity": 5,    # 50% of tasks (5/10)
            "medium_capacity": 3,  # 30% of tasks (3/10)
            "low_capacity": 2      # 20% of tasks (2/10)
        },
        "task_assignments": {
            "high_capacity": [],
            "medium_capacity": [],
            "low_capacity": []
        },
        "completed_tasks": [],
        "results": {}
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Weighted Distribution Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
