"""
Geographic Distribution MCP Pattern

This pattern demonstrates distributing tasks based on geographic location,
ensuring locality-aware processing and minimal latency.

Key Features:
- Location-based task routing
- Regional data processing
- Latency optimization
- Geographic redundancy
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class GeographicDistributionState(TypedDict):
    """State for geographic distribution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: list[dict]  # includes region field
    region_assignments: dict[str, list[str]]  # region -> task_ids
    completed_tasks: list[str]
    results: dict[str, str]
    processing_latency: dict[str, float]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Geographic Router
def geographic_router(state: GeographicDistributionState) -> GeographicDistributionState:
    """Routes tasks to appropriate geographic regions"""
    tasks = state.get("tasks", [])
    
    system_message = SystemMessage(content="""You are a geographic router. Distribute tasks 
    to regional processors based on location for optimal latency and data locality.""")
    
    tasks_info = "\n".join([
        f"Task {t['id']}: {t['description']} (Region: {t.get('region', 'global')})"
        for t in tasks
    ])
    
    user_message = HumanMessage(content=f"""Route these tasks to regional processors:
    
{tasks_info}

Available regions: North America, Europe, Asia-Pacific

Route based on task origin for minimal latency.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Route tasks by region
    region_assignments = {
        "north_america": [],
        "europe": [],
        "asia_pacific": []
    }
    
    for task in tasks:
        region = task.get("region", "north_america").lower().replace(" ", "_").replace("-", "_")
        if region in region_assignments:
            region_assignments[region].append(task['id'])
    
    routing_summary = "\n".join([
        f"  {region.replace('_', ' ').title()}: {len(tasks)} tasks"
        for region, tasks in region_assignments.items() if tasks
    ])
    
    return {
        "messages": [AIMessage(content=f"🌍 Geographic Router: {response.content}\n\nRouting:\n{routing_summary}")],
        "region_assignments": region_assignments
    }


# North America Processor
def north_america_processor(state: GeographicDistributionState) -> GeographicDistributionState:
    """Processes tasks in North America region"""
    assignments = state.get("region_assignments", {}).get("north_america", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="🇺🇸 North America: No tasks for this region")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are the North America regional processor. 
    Process tasks with low latency for US/Canada users. Data center: Virginia, USA.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these North American tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    latency = state.get("processing_latency", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[NA-Virginia] {task['description']}"
        latency[task['id']] = 15.0  # Low latency for regional processing
    
    return {
        "messages": [AIMessage(content=f"🇺🇸 North America: {response.content}\n\nProcessed {len(my_tasks)} tasks (avg latency: 15ms)")],
        "completed_tasks": completed,
        "results": results,
        "processing_latency": latency
    }


# Europe Processor
def europe_processor(state: GeographicDistributionState) -> GeographicDistributionState:
    """Processes tasks in Europe region"""
    assignments = state.get("region_assignments", {}).get("europe", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="🇪🇺 Europe: No tasks for this region")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are the Europe regional processor. 
    Process tasks with low latency for European users. Data center: Frankfurt, Germany.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these European tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    latency = state.get("processing_latency", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[EU-Frankfurt] {task['description']}"
        latency[task['id']] = 18.0  # Low latency for regional processing
    
    return {
        "messages": [AIMessage(content=f"🇪🇺 Europe: {response.content}\n\nProcessed {len(my_tasks)} tasks (avg latency: 18ms)")],
        "completed_tasks": completed,
        "results": results,
        "processing_latency": latency
    }


# Asia-Pacific Processor
def asia_pacific_processor(state: GeographicDistributionState) -> GeographicDistributionState:
    """Processes tasks in Asia-Pacific region"""
    assignments = state.get("region_assignments", {}).get("asia_pacific", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="🌏 Asia-Pacific: No tasks for this region")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are the Asia-Pacific regional processor. 
    Process tasks with low latency for APAC users. Data center: Singapore.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these Asia-Pacific tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    latency = state.get("processing_latency", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[APAC-Singapore] {task['description']}"
        latency[task['id']] = 20.0  # Low latency for regional processing
    
    return {
        "messages": [AIMessage(content=f"🌏 Asia-Pacific: {response.content}\n\nProcessed {len(my_tasks)} tasks (avg latency: 20ms)")],
        "completed_tasks": completed,
        "results": results,
        "processing_latency": latency
    }


# Latency Monitor
def latency_monitor(state: GeographicDistributionState) -> GeographicDistributionState:
    """Monitors and reports geographic distribution performance"""
    region_assignments = state.get("region_assignments", {})
    latency = state.get("processing_latency", {})
    
    avg_latency = sum(latency.values()) / len(latency) if latency else 0
    
    distribution = "\n".join([
        f"  {region.replace('_', ' ').title()}: {len(tasks)} tasks"
        for region, tasks in region_assignments.items() if tasks
    ])
    
    summary = f"""
    ✅ GEOGRAPHIC DISTRIBUTION COMPLETE
    
    Total Tasks: {sum(len(t) for t in region_assignments.values())}
    Average Latency: {avg_latency:.1f}ms
    
    Regional Distribution:
{distribution}
    
    Benefits achieved:
    • Reduced latency through regional processing
    • Data locality compliance
    • Geographic redundancy
    • Improved user experience
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Latency Monitor:\n{summary}")]
    }


# Build the graph
def build_geographic_distribution_graph():
    """Build the geographic distribution MCP pattern graph"""
    workflow = StateGraph(GeographicDistributionState)
    
    workflow.add_node("router", geographic_router)
    workflow.add_node("north_america", north_america_processor)
    workflow.add_node("europe", europe_processor)
    workflow.add_node("asia_pacific", asia_pacific_processor)
    workflow.add_node("monitor", latency_monitor)
    
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "north_america")
    workflow.add_edge("router", "europe")
    workflow.add_edge("router", "asia_pacific")
    workflow.add_edge("north_america", "monitor")
    workflow.add_edge("europe", "monitor")
    workflow.add_edge("asia_pacific", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_geographic_distribution_graph()
    
    print("=== Geographic Distribution MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "tasks": [
            {"id": "task_1", "description": "Process US customer order", "region": "north_america"},
            {"id": "task_2", "description": "Update Canadian inventory", "region": "north_america"},
            {"id": "task_3", "description": "Process German payment", "region": "europe"},
            {"id": "task_4", "description": "UK user registration", "region": "europe"},
            {"id": "task_5", "description": "French data analysis", "region": "europe"},
            {"id": "task_6", "description": "Singapore transaction", "region": "asia_pacific"},
            {"id": "task_7", "description": "Australian report generation", "region": "asia_pacific"},
            {"id": "task_8", "description": "Japanese user update", "region": "asia_pacific"},
        ],
        "region_assignments": {"north_america": [], "europe": [], "asia_pacific": []},
        "completed_tasks": [],
        "results": {},
        "processing_latency": {}
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Geographic Distribution Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
