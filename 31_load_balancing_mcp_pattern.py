"""
Load Balancing MCP Pattern

This pattern demonstrates distributing tasks across multiple worker agents to 
balance workload and optimize resource utilization.

Key Features:
- Dynamic load distribution across workers
- Load tracking per agent
- Least-loaded agent selection
- Concurrent task processing
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class LoadBalancingState(TypedDict):
    """State for load balancing pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: list[dict[str, str]]  # List of tasks to process
    agent_loads: dict[str, int]  # Current load per agent
    task_assignments: dict[str, list[str]]  # agent -> [tasks]
    completed_tasks: list[str]
    results: dict[str, str]  # task_id -> result


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Load Balancer
def load_balancer(state: LoadBalancingState) -> LoadBalancingState:
    """Distributes tasks to least-loaded agents"""
    tasks = state.get("tasks", [])
    agent_loads = state.get("agent_loads", {"worker_1": 0, "worker_2": 0, "worker_3": 0})
    task_assignments = state.get("task_assignments", {"worker_1": [], "worker_2": [], "worker_3": []})
    
    system_message = SystemMessage(content="""You are a load balancer. Distribute incoming tasks 
    to worker agents based on their current load. Always assign to the least-loaded agent.""")
    
    tasks_info = "\n".join([f"Task {t['id']}: {t['description']}" for t in tasks])
    loads_info = "\n".join([f"{agent}: {load} tasks" for agent, load in agent_loads.items()])
    
    user_message = HumanMessage(content=f"""Distribute these tasks:
    
{tasks_info}

Current agent loads:
{loads_info}

Assign tasks to balance load across workers.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Distribute tasks to least-loaded agents
    for task in tasks:
        # Find least-loaded agent
        least_loaded = min(agent_loads.items(), key=lambda x: x[1])[0]
        
        # Assign task
        task_assignments[least_loaded].append(task['id'])
        agent_loads[least_loaded] += 1
    
    assignment_summary = "\n".join([
        f"{agent}: {len(tasks)} tasks - {', '.join(tasks)}"
        for agent, tasks in task_assignments.items()
    ])
    
    return {
        "messages": [AIMessage(content=f"⚖️ Load Balancer: {response.content}\n\nAssignments:\n{assignment_summary}")],
        "agent_loads": agent_loads,
        "task_assignments": task_assignments
    }


# Worker 1
def worker_1_agent(state: LoadBalancingState) -> LoadBalancingState:
    """Worker 1 processes assigned tasks"""
    assignments = state.get("task_assignments", {}).get("worker_1", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="👷 Worker 1: No tasks assigned")]}
    
    # Get task details
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are Worker 1. Process your assigned tasks 
    efficiently and provide results.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these assigned tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Worker 1 completed: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"👷 Worker 1: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Worker 2
def worker_2_agent(state: LoadBalancingState) -> LoadBalancingState:
    """Worker 2 processes assigned tasks"""
    assignments = state.get("task_assignments", {}).get("worker_2", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="👷 Worker 2: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are Worker 2. Process your assigned tasks 
    efficiently and provide results.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these assigned tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Worker 2 completed: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"👷 Worker 2: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Worker 3
def worker_3_agent(state: LoadBalancingState) -> LoadBalancingState:
    """Worker 3 processes assigned tasks"""
    assignments = state.get("task_assignments", {}).get("worker_3", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="👷 Worker 3: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are Worker 3. Process your assigned tasks 
    efficiently and provide results.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these assigned tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Worker 3 completed: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"👷 Worker 3: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Result Aggregator
def result_aggregator(state: LoadBalancingState) -> LoadBalancingState:
    """Aggregates results from all workers"""
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    agent_loads = state.get("agent_loads", {})
    
    summary = f"""
    ✅ LOAD BALANCING COMPLETE
    
    Total Tasks: {len(completed)}
    
    Load Distribution:
    {chr(10).join([f'  {agent}: {load} tasks' for agent, load in agent_loads.items()])}
    
    All tasks processed successfully with balanced load distribution.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Result Aggregator:\n{summary}")]
    }


# Build the graph
def build_load_balancing_graph():
    """Build the load balancing MCP pattern graph"""
    workflow = StateGraph(LoadBalancingState)
    
    # Add nodes
    workflow.add_node("balancer", load_balancer)
    workflow.add_node("worker_1", worker_1_agent)
    workflow.add_node("worker_2", worker_2_agent)
    workflow.add_node("worker_3", worker_3_agent)
    workflow.add_node("aggregator", result_aggregator)
    
    # Define edges
    workflow.add_edge(START, "balancer")
    
    # Workers process in parallel after balancer
    workflow.add_edge("balancer", "worker_1")
    workflow.add_edge("balancer", "worker_2")
    workflow.add_edge("balancer", "worker_3")
    
    # All workers feed into aggregator
    workflow.add_edge("worker_1", "aggregator")
    workflow.add_edge("worker_2", "aggregator")
    workflow.add_edge("worker_3", "aggregator")
    
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_load_balancing_graph()
    
    print("=== Load Balancing MCP Pattern ===\n")
    
    # Create tasks to distribute
    initial_state = {
        "messages": [],
        "tasks": [
            {"id": "task_1", "description": "Process customer order #1001"},
            {"id": "task_2", "description": "Generate monthly report"},
            {"id": "task_3", "description": "Send notification emails"},
            {"id": "task_4", "description": "Update inventory database"},
            {"id": "task_5", "description": "Analyze user behavior data"},
            {"id": "task_6", "description": "Backup system logs"},
            {"id": "task_7", "description": "Optimize search indices"},
            {"id": "task_8", "description": "Process refund request"},
            {"id": "task_9", "description": "Generate analytics dashboard"},
        ],
        "agent_loads": {"worker_1": 0, "worker_2": 0, "worker_3": 0},
        "task_assignments": {"worker_1": [], "worker_2": [], "worker_3": []},
        "completed_tasks": [],
        "results": {}
    }
    
    # Run load balancing
    result = graph.invoke(initial_state)
    
    print("\n=== Load Balancing Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Statistics ===")
    print(f"Total Tasks: {len(result['tasks'])}")
    print(f"Completed: {len(result['completed_tasks'])}")
    print(f"\nLoad Distribution:")
    for agent, load in result['agent_loads'].items():
        print(f"  {agent}: {load} tasks")
