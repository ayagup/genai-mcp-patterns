"""
Priority-Based Distribution MCP Pattern

This pattern demonstrates distributing tasks based on priority levels,
ensuring high-priority tasks are processed first.

Key Features:
- Priority-based task allocation
- High-priority tasks processed first
- Multiple priority levels
- Efficient critical task handling
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class PriorityDistributionState(TypedDict):
    """State for priority-based distribution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: list[dict[str, any]]  # includes priority field
    task_assignments: dict[str, list[str]]  # priority_level -> task_ids
    completed_tasks: dict[str, list[str]]  # priority_level -> completed_task_ids
    results: dict[str, str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Priority Analyzer
def priority_analyzer(state: PriorityDistributionState) -> PriorityDistributionState:
    """Analyzes and categorizes tasks by priority"""
    tasks = state.get("tasks", [])
    
    system_message = SystemMessage(content="""You are a priority analyzer. Categorize tasks 
    based on their priority levels (critical, high, medium, low) to ensure proper processing order.""")
    
    tasks_info = "\n".join([f"Task {t['id']}: {t['description']} (Priority: {t['priority']})" for t in tasks])
    
    user_message = HumanMessage(content=f"""Analyze and categorize these tasks by priority:
    
{tasks_info}

Organize by: CRITICAL > HIGH > MEDIUM > LOW""")
    
    response = llm.invoke([system_message, user_message])
    
    # Categorize tasks by priority
    task_assignments = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": []
    }
    
    for task in tasks:
        priority = task.get("priority", "medium").lower()
        if priority in task_assignments:
            task_assignments[priority].append(task['id'])
    
    summary = "\n".join([
        f"  {priority.upper()}: {len(tasks)} tasks"
        for priority, tasks in task_assignments.items() if tasks
    ])
    
    return {
        "messages": [AIMessage(content=f"🔍 Priority Analyzer: {response.content}\n\nCategorization:\n{summary}")],
        "task_assignments": task_assignments
    }


# Critical Priority Handler
def critical_priority_handler(state: PriorityDistributionState) -> PriorityDistributionState:
    """Handles critical priority tasks immediately"""
    critical_tasks = state.get("task_assignments", {}).get("critical", [])
    tasks = state.get("tasks", [])
    
    if not critical_tasks:
        return {"messages": [AIMessage(content="🚨 Critical Handler: No critical tasks")]}
    
    my_tasks = [t for t in tasks if t['id'] in critical_tasks]
    
    system_message = SystemMessage(content="""You are the critical priority handler. Process 
    critical tasks immediately with highest urgency. These tasks cannot be delayed.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""URGENT - Process these critical tasks immediately:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", {"critical": [], "high": [], "medium": [], "low": []})
    results = state.get("results", {})
    
    for task in my_tasks:
        completed["critical"].append(task['id'])
        results[task['id']] = f"[CRITICAL] {task['description']} - Processed immediately"
    
    return {
        "messages": [AIMessage(content=f"🚨 Critical Handler: {response.content}\n\nProcessed {len(my_tasks)} critical tasks")],
        "completed_tasks": completed,
        "results": results
    }


# High Priority Handler
def high_priority_handler(state: PriorityDistributionState) -> PriorityDistributionState:
    """Handles high priority tasks"""
    high_tasks = state.get("task_assignments", {}).get("high", [])
    tasks = state.get("tasks", [])
    
    if not high_tasks:
        return {"messages": [AIMessage(content="⚡ High Priority Handler: No high priority tasks")]}
    
    my_tasks = [t for t in tasks if t['id'] in high_tasks]
    
    system_message = SystemMessage(content="""You are the high priority handler. Process 
    high priority tasks promptly after critical tasks.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these high priority tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", {"critical": [], "high": [], "medium": [], "low": []})
    results = state.get("results", {})
    
    for task in my_tasks:
        completed["high"].append(task['id'])
        results[task['id']] = f"[HIGH] {task['description']} - Processed promptly"
    
    return {
        "messages": [AIMessage(content=f"⚡ High Priority Handler: {response.content}\n\nProcessed {len(my_tasks)} high priority tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Medium Priority Handler
def medium_priority_handler(state: PriorityDistributionState) -> PriorityDistributionState:
    """Handles medium priority tasks"""
    medium_tasks = state.get("task_assignments", {}).get("medium", [])
    tasks = state.get("tasks", [])
    
    if not medium_tasks:
        return {"messages": [AIMessage(content="📋 Medium Priority Handler: No medium priority tasks")]}
    
    my_tasks = [t for t in tasks if t['id'] in medium_tasks]
    
    system_message = SystemMessage(content="""You are the medium priority handler. Process 
    medium priority tasks in normal course after higher priority tasks.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these medium priority tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", {"critical": [], "high": [], "medium": [], "low": []})
    results = state.get("results", {})
    
    for task in my_tasks:
        completed["medium"].append(task['id'])
        results[task['id']] = f"[MEDIUM] {task['description']} - Processed normally"
    
    return {
        "messages": [AIMessage(content=f"📋 Medium Priority Handler: {response.content}\n\nProcessed {len(my_tasks)} medium priority tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Low Priority Handler
def low_priority_handler(state: PriorityDistributionState) -> PriorityDistributionState:
    """Handles low priority tasks"""
    low_tasks = state.get("task_assignments", {}).get("low", [])
    tasks = state.get("tasks", [])
    
    if not low_tasks:
        return {"messages": [AIMessage(content="🔹 Low Priority Handler: No low priority tasks")]}
    
    my_tasks = [t for t in tasks if t['id'] in low_tasks]
    
    system_message = SystemMessage(content="""You are the low priority handler. Process 
    low priority tasks when higher priority tasks are complete.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these low priority tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", {"critical": [], "high": [], "medium": [], "low": []})
    results = state.get("results", {})
    
    for task in my_tasks:
        completed["low"].append(task['id'])
        results[task['id']] = f"[LOW] {task['description']} - Processed when available"
    
    return {
        "messages": [AIMessage(content=f"🔹 Low Priority Handler: {response.content}\n\nProcessed {len(my_tasks)} low priority tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Priority Coordinator
def priority_coordinator(state: PriorityDistributionState) -> PriorityDistributionState:
    """Coordinates priority-based processing results"""
    completed = state.get("completed_tasks", {})
    
    total_tasks = sum(len(tasks) for tasks in completed.values())
    
    summary = "\n".join([
        f"  {priority.upper()}: {len(tasks)} tasks completed"
        for priority, tasks in completed.items() if tasks
    ])
    
    report = f"""
    ✅ PRIORITY-BASED DISTRIBUTION COMPLETE
    
    Total Tasks: {total_tasks}
    
    Processing Summary:
{summary}
    
    Tasks processed in priority order:
    1. CRITICAL tasks (immediate)
    2. HIGH priority tasks (prompt)
    3. MEDIUM priority tasks (normal)
    4. LOW priority tasks (when available)
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Priority Coordinator:\n{report}")]
    }


# Build the graph
def build_priority_distribution_graph():
    """Build the priority-based distribution MCP pattern graph"""
    workflow = StateGraph(PriorityDistributionState)
    
    workflow.add_node("analyzer", priority_analyzer)
    workflow.add_node("critical_handler", critical_priority_handler)
    workflow.add_node("high_handler", high_priority_handler)
    workflow.add_node("medium_handler", medium_priority_handler)
    workflow.add_node("low_handler", low_priority_handler)
    workflow.add_node("coordinator", priority_coordinator)
    
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "critical_handler")
    workflow.add_edge("analyzer", "high_handler")
    workflow.add_edge("analyzer", "medium_handler")
    workflow.add_edge("analyzer", "low_handler")
    workflow.add_edge("critical_handler", "coordinator")
    workflow.add_edge("high_handler", "coordinator")
    workflow.add_edge("medium_handler", "coordinator")
    workflow.add_edge("low_handler", "coordinator")
    workflow.add_edge("coordinator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_priority_distribution_graph()
    
    print("=== Priority-Based Distribution MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "tasks": [
            {"id": "task_1", "description": "System crash recovery", "priority": "critical"},
            {"id": "task_2", "description": "Security vulnerability patch", "priority": "critical"},
            {"id": "task_3", "description": "Customer escalation", "priority": "high"},
            {"id": "task_4", "description": "Feature deployment", "priority": "high"},
            {"id": "task_5", "description": "Performance optimization", "priority": "high"},
            {"id": "task_6", "description": "UI enhancement", "priority": "medium"},
            {"id": "task_7", "description": "Documentation update", "priority": "medium"},
            {"id": "task_8", "description": "Code refactoring", "priority": "low"},
            {"id": "task_9", "description": "Test coverage improvement", "priority": "low"},
        ],
        "task_assignments": {"critical": [], "high": [], "medium": [], "low": []},
        "completed_tasks": {"critical": [], "high": [], "medium": [], "low": []},
        "results": {}
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Priority-Based Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
