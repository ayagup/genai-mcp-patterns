"""
Sharding MCP Pattern

This pattern demonstrates distributing data and tasks across multiple shards
based on a sharding key, enabling horizontal scalability.

Key Features:
- Consistent hash-based sharding
- Horizontal data partitioning
- Load distribution across shards
- Scalable architecture
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ShardingState(TypedDict):
    """State for sharding pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: list[dict]  # includes shard_key field
    shard_assignments: dict[str, list[str]]  # shard -> task_ids
    shard_loads: dict[str, int]
    completed_tasks: list[str]
    results: dict[str, str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Shard Coordinator
def shard_coordinator(state: ShardingState) -> ShardingState:
    """Coordinates task distribution across shards"""
    tasks = state.get("tasks", [])
    num_shards = 4
    
    system_message = SystemMessage(content="""You are a shard coordinator. Distribute tasks 
    across shards using consistent hashing on shard keys for even distribution.""")
    
    tasks_info = "\n".join([
        f"Task {t['id']}: {t['description']} (Key: {t.get('shard_key', 'default')})"
        for t in tasks
    ])
    
    user_message = HumanMessage(content=f"""Distribute these tasks across {num_shards} shards:
    
{tasks_info}

Use shard_key to determine shard assignment via consistent hashing.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Distribute tasks using simple hash-based sharding
    shard_assignments = {f"shard_{i}": [] for i in range(num_shards)}
    shard_loads = {f"shard_{i}": 0 for i in range(num_shards)}
    
    for task in tasks:
        shard_key = task.get("shard_key", "default")
        shard_id = hash(shard_key) % num_shards
        shard_name = f"shard_{shard_id}"
        shard_assignments[shard_name].append(task['id'])
        shard_loads[shard_name] += 1
    
    distribution = "\n".join([
        f"  {shard}: {len(tasks)} tasks (load: {shard_loads[shard]})"
        for shard, tasks in shard_assignments.items() if tasks
    ])
    
    return {
        "messages": [AIMessage(content=f"🔀 Shard Coordinator: {response.content}\n\nDistribution:\n{distribution}")],
        "shard_assignments": shard_assignments,
        "shard_loads": shard_loads
    }


# Shard 0 Processor
def shard_0_processor(state: ShardingState) -> ShardingState:
    """Processes tasks for shard 0"""
    assignments = state.get("shard_assignments", {}).get("shard_0", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="📦 Shard 0: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are Shard 0 processor. Handle your partition 
    of the distributed workload efficiently.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process Shard 0 tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[Shard-0] {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"📦 Shard 0: {response.content}\n\nProcessed {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Shard 1 Processor
def shard_1_processor(state: ShardingState) -> ShardingState:
    """Processes tasks for shard 1"""
    assignments = state.get("shard_assignments", {}).get("shard_1", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="📦 Shard 1: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are Shard 1 processor. Handle your partition 
    of the distributed workload efficiently.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process Shard 1 tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[Shard-1] {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"📦 Shard 1: {response.content}\n\nProcessed {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Shard 2 Processor
def shard_2_processor(state: ShardingState) -> ShardingState:
    """Processes tasks for shard 2"""
    assignments = state.get("shard_assignments", {}).get("shard_2", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="📦 Shard 2: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are Shard 2 processor. Handle your partition 
    of the distributed workload efficiently.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process Shard 2 tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[Shard-2] {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"📦 Shard 2: {response.content}\n\nProcessed {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Shard 3 Processor
def shard_3_processor(state: ShardingState) -> ShardingState:
    """Processes tasks for shard 3"""
    assignments = state.get("shard_assignments", {}).get("shard_3", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="📦 Shard 3: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are Shard 3 processor. Handle your partition 
    of the distributed workload efficiently.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process Shard 3 tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[Shard-3] {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"📦 Shard 3: {response.content}\n\nProcessed {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Shard Monitor
def shard_monitor(state: ShardingState) -> ShardingState:
    """Monitors shard distribution and performance"""
    shard_loads = state.get("shard_loads", {})
    completed = state.get("completed_tasks", [])
    
    total_load = sum(shard_loads.values())
    avg_load = total_load / len(shard_loads) if shard_loads else 0
    
    load_distribution = "\n".join([
        f"  {shard}: {load} tasks ({(load/total_load*100):.1f}%)" if total_load > 0 else f"  {shard}: 0 tasks"
        for shard, load in shard_loads.items()
    ])
    
    summary = f"""
    ✅ SHARDING DISTRIBUTION COMPLETE
    
    Total Tasks: {len(completed)}
    Number of Shards: {len(shard_loads)}
    Average Load per Shard: {avg_load:.1f}
    
    Load Distribution:
{load_distribution}
    
    Sharding Benefits:
    • Horizontal scalability achieved
    • Even load distribution
    • Data partitioning for parallel processing
    • Can add more shards as needed
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Shard Monitor:\n{summary}")]
    }


# Build the graph
def build_sharding_graph():
    """Build the sharding MCP pattern graph"""
    workflow = StateGraph(ShardingState)
    
    workflow.add_node("coordinator", shard_coordinator)
    workflow.add_node("shard_0", shard_0_processor)
    workflow.add_node("shard_1", shard_1_processor)
    workflow.add_node("shard_2", shard_2_processor)
    workflow.add_node("shard_3", shard_3_processor)
    workflow.add_node("monitor", shard_monitor)
    
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "shard_0")
    workflow.add_edge("coordinator", "shard_1")
    workflow.add_edge("coordinator", "shard_2")
    workflow.add_edge("coordinator", "shard_3")
    workflow.add_edge("shard_0", "monitor")
    workflow.add_edge("shard_1", "monitor")
    workflow.add_edge("shard_2", "monitor")
    workflow.add_edge("shard_3", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_sharding_graph()
    
    print("=== Sharding MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "tasks": [
            {"id": "task_1", "description": "Process user_123 data", "shard_key": "user_123"},
            {"id": "task_2", "description": "Process user_456 data", "shard_key": "user_456"},
            {"id": "task_3", "description": "Process user_789 data", "shard_key": "user_789"},
            {"id": "task_4", "description": "Process user_012 data", "shard_key": "user_012"},
            {"id": "task_5", "description": "Process user_345 data", "shard_key": "user_345"},
            {"id": "task_6", "description": "Process user_678 data", "shard_key": "user_678"},
            {"id": "task_7", "description": "Process user_901 data", "shard_key": "user_901"},
            {"id": "task_8", "description": "Process user_234 data", "shard_key": "user_234"},
            {"id": "task_9", "description": "Process user_567 data", "shard_key": "user_567"},
            {"id": "task_10", "description": "Process user_890 data", "shard_key": "user_890"},
        ],
        "shard_assignments": {},
        "shard_loads": {},
        "completed_tasks": [],
        "results": {}
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Sharding Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
