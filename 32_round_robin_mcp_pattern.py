"""
Round-Robin MCP Pattern

This pattern demonstrates distributing tasks in sequential circular order 
across worker agents, ensuring equal distribution over time.

Key Features:
- Sequential circular task distribution
- Fair allocation across workers
- Simple and predictable assignment
- No load tracking needed
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class RoundRobinState(TypedDict):
    """State for round-robin pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: list[dict[str, str]]
    current_worker_index: int
    worker_names: list[str]
    task_assignments: dict[str, list[str]]
    completed_tasks: list[str]
    results: dict[str, str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Round-Robin Dispatcher
def round_robin_dispatcher(state: RoundRobinState) -> RoundRobinState:
    """Distributes tasks in round-robin fashion"""
    tasks = state.get("tasks", [])
    worker_names = state.get("worker_names", ["worker_a", "worker_b", "worker_c"])
    current_index = state.get("current_worker_index", 0)
    task_assignments = state.get("task_assignments", {name: [] for name in worker_names})
    
    system_message = SystemMessage(content="""You are a round-robin dispatcher. Distribute tasks 
    in circular sequential order to ensure equal distribution across workers.""")
    
    tasks_info = "\n".join([f"Task {t['id']}: {t['description']}" for t in tasks])
    
    user_message = HumanMessage(content=f"""Distribute these tasks in round-robin order:
    
{tasks_info}

Workers: {', '.join(worker_names)}

Assign tasks sequentially in circular order.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Distribute tasks in round-robin
    for task in tasks:
        worker = worker_names[current_index % len(worker_names)]
        task_assignments[worker].append(task['id'])
        current_index += 1
    
    assignment_summary = "\n".join([
        f"{worker}: {len(tasks)} tasks - {', '.join(tasks)}"
        for worker, tasks in task_assignments.items() if tasks
    ])
    
    return {
        "messages": [AIMessage(content=f"🔄 Round-Robin Dispatcher: {response.content}\n\nAssignments:\n{assignment_summary}")],
        "current_worker_index": current_index,
        "task_assignments": task_assignments
    }


# Worker A
def worker_a_agent(state: RoundRobinState) -> RoundRobinState:
    """Worker A processes assigned tasks"""
    assignments = state.get("task_assignments", {}).get("worker_a", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="👷 Worker A: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    
    system_message = SystemMessage(content="""You are Worker A in a round-robin system. 
    Process your assigned tasks efficiently.""")
    
    user_message = HumanMessage(content=f"""Process these tasks:\n{tasks_desc}""")
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Worker A: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"👷 Worker A: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Worker B
def worker_b_agent(state: RoundRobinState) -> RoundRobinState:
    """Worker B processes assigned tasks"""
    assignments = state.get("task_assignments", {}).get("worker_b", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="👷 Worker B: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    
    system_message = SystemMessage(content="""You are Worker B in a round-robin system. 
    Process your assigned tasks efficiently.""")
    
    user_message = HumanMessage(content=f"""Process these tasks:\n{tasks_desc}""")
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Worker B: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"👷 Worker B: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Worker C
def worker_c_agent(state: RoundRobinState) -> RoundRobinState:
    """Worker C processes assigned tasks"""
    assignments = state.get("task_assignments", {}).get("worker_c", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="👷 Worker C: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    
    system_message = SystemMessage(content="""You are Worker C in a round-robin system. 
    Process your assigned tasks efficiently.""")
    
    user_message = HumanMessage(content=f"""Process these tasks:\n{tasks_desc}""")
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Worker C: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"👷 Worker C: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Result Collector
def result_collector(state: RoundRobinState) -> RoundRobinState:
    """Collects and summarizes results"""
    task_assignments = state.get("task_assignments", {})
    completed = state.get("completed_tasks", [])
    
    distribution = "\n".join([
        f"  {worker}: {len(tasks)} tasks"
        for worker, tasks in task_assignments.items()
    ])
    
    summary = f"""
    ✅ ROUND-ROBIN DISTRIBUTION COMPLETE
    
    Total Tasks: {len(completed)}
    
    Task Distribution:
{distribution}
    
    Tasks distributed evenly using round-robin algorithm.
    Each worker received an equal share of tasks in sequential order.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Result Collector:\n{summary}")]
    }


# Build the graph
def build_round_robin_graph():
    """Build the round-robin MCP pattern graph"""
    workflow = StateGraph(RoundRobinState)
    
    workflow.add_node("dispatcher", round_robin_dispatcher)
    workflow.add_node("worker_a", worker_a_agent)
    workflow.add_node("worker_b", worker_b_agent)
    workflow.add_node("worker_c", worker_c_agent)
    workflow.add_node("collector", result_collector)
    
    workflow.add_edge(START, "dispatcher")
    workflow.add_edge("dispatcher", "worker_a")
    workflow.add_edge("dispatcher", "worker_b")
    workflow.add_edge("dispatcher", "worker_c")
    workflow.add_edge("worker_a", "collector")
    workflow.add_edge("worker_b", "collector")
    workflow.add_edge("worker_c", "collector")
    workflow.add_edge("collector", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_round_robin_graph()
    
    print("=== Round-Robin MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "tasks": [
            {"id": "task_1", "description": "Process order #1001"},
            {"id": "task_2", "description": "Send email notification"},
            {"id": "task_3", "description": "Update database"},
            {"id": "task_4", "description": "Generate report"},
            {"id": "task_5", "description": "Backup data"},
            {"id": "task_6", "description": "Analyze logs"},
            {"id": "task_7", "description": "Optimize query"},
            {"id": "task_8", "description": "Clean cache"},
            {"id": "task_9", "description": "Index search"},
        ],
        "current_worker_index": 0,
        "worker_names": ["worker_a", "worker_b", "worker_c"],
        "task_assignments": {"worker_a": [], "worker_b": [], "worker_c": []},
        "completed_tasks": [],
        "results": {}
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Round-Robin Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Distribution Summary ===")
    for worker, tasks in result['task_assignments'].items():
        print(f"{worker}: {len(tasks)} tasks - {', '.join(tasks)}")
