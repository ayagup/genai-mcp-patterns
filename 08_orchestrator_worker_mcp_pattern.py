"""
Orchestrator-Worker MCP Pattern
================================
An orchestrator agent dynamically assigns tasks to a pool of worker agents.
The orchestrator manages task distribution, monitors progress, and aggregates results.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict, List
import operator
import json


# Define the state
class OrchestratorState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    task_queue: List[Dict[str, str]]  # Tasks to be assigned
    worker_assignments: Dict[str, str]  # Which worker is doing what
    worker_results: Dict[str, str]  # Results from each worker
    orchestration_plan: str
    final_result: str


# Orchestrator Agent
def orchestrator_agent(state: OrchestratorState):
    """Orchestrator that plans and assigns tasks to workers."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are an Orchestrator. Break down the main task into subtasks for workers.\n"
                "Available workers:\n"
                "- Worker 1: Research and data gathering\n"
                "- Worker 2: Analysis and processing\n"
                "- Worker 3: Content generation\n"
                "Create a JSON plan: {\"tasks\": [{\"worker\": \"worker_1\", \"task\": \"description\"}]}"
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Parse orchestration plan
    try:
        plan = json.loads(response.content)
        tasks = plan.get("tasks", [])
    except:
        tasks = [
            {"worker": "worker_1", "task": "Research the topic"},
            {"worker": "worker_2", "task": "Analyze the data"},
            {"worker": "worker_3", "task": "Generate content"}
        ]
    
    # Create task queue and assignments
    task_queue = tasks
    assignments = {task["worker"]: task["task"] for task in tasks}
    
    return {
        "messages": [AIMessage(content=f"Orchestrator: Created plan with {len(tasks)} tasks - {response.content}")],
        "task_queue": task_queue,
        "worker_assignments": assignments,
        "orchestration_plan": response.content
    }


# Worker 1 - Research Worker
def worker_1_agent(state: OrchestratorState):
    """Worker specializing in research and data gathering."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    assignment = state.get("worker_assignments", {}).get("worker_1", "No assignment")
    
    system_msg = SystemMessage(
        content=f"You are Worker 1 - Research Specialist.\n"
                f"Assignment from Orchestrator: {assignment}\n"
                "Complete your assigned task and report results."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Worker 1 (Research): {response.content}")],
        "worker_results": {"worker_1": response.content}
    }


# Worker 2 - Analysis Worker
def worker_2_agent(state: OrchestratorState):
    """Worker specializing in analysis and processing."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    assignment = state.get("worker_assignments", {}).get("worker_2", "No assignment")
    research_data = state.get("worker_results", {}).get("worker_1", "No research data")
    
    system_msg = SystemMessage(
        content=f"You are Worker 2 - Analysis Specialist.\n"
                f"Assignment from Orchestrator: {assignment}\n"
                f"Research data from Worker 1: {research_data}\n"
                "Complete your assigned task and report results."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Worker 2 (Analysis): {response.content}")],
        "worker_results": {"worker_2": response.content}
    }


# Worker 3 - Content Worker
def worker_3_agent(state: OrchestratorState):
    """Worker specializing in content generation."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    assignment = state.get("worker_assignments", {}).get("worker_3", "No assignment")
    analysis_data = state.get("worker_results", {}).get("worker_2", "No analysis data")
    
    system_msg = SystemMessage(
        content=f"You are Worker 3 - Content Generation Specialist.\n"
                f"Assignment from Orchestrator: {assignment}\n"
                f"Analysis data from Worker 2: {analysis_data}\n"
                "Complete your assigned task and report results."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Worker 3 (Content): {response.content}")],
        "worker_results": {"worker_3": response.content}
    }


# Orchestrator Aggregator
def orchestrator_aggregator(state: OrchestratorState):
    """Orchestrator collects and aggregates worker results."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    worker_results = state.get("worker_results", {})
    plan = state.get("orchestration_plan", "No plan")
    
    system_msg = SystemMessage(
        content=f"You are the Orchestrator. Aggregate worker results:\n"
                f"Original Plan: {plan}\n"
                f"Worker 1 Results: {worker_results.get('worker_1', 'N/A')}\n"
                f"Worker 2 Results: {worker_results.get('worker_2', 'N/A')}\n"
                f"Worker 3 Results: {worker_results.get('worker_3', 'N/A')}\n"
                "Synthesize all results into a final comprehensive output."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Orchestrator (Final): {response.content}")],
        "final_result": response.content
    }


# Build the orchestrator-worker graph
def create_orchestrator_worker_graph():
    """Create an orchestrator-worker workflow graph."""
    workflow = StateGraph(OrchestratorState)
    
    # Add orchestrator and worker nodes
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("worker_1", worker_1_agent)
    workflow.add_node("worker_2", worker_2_agent)
    workflow.add_node("worker_3", worker_3_agent)
    workflow.add_node("aggregator", orchestrator_aggregator)
    
    # Orchestration flow: Orchestrator -> Workers -> Aggregator
    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "worker_1")
    workflow.add_edge("worker_1", "worker_2")
    workflow.add_edge("worker_2", "worker_3")
    workflow.add_edge("worker_3", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the orchestrator-worker system
    graph = create_orchestrator_worker_graph()
    
    print("=" * 60)
    print("ORCHESTRATOR-WORKER MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Complex task requiring orchestration
    print("\n[Task: Create comprehensive business report]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Create a comprehensive business report on AI adoption in healthcare")],
        "task_queue": [],
        "worker_assignments": {},
        "worker_results": {},
        "orchestration_plan": "",
        "final_result": ""
    })
    
    print("\n--- Orchestration Workflow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Task Assignments ---")
    for worker, task in result.get("worker_assignments", {}).items():
        print(f"{worker}: {task[:100]}...")
    
    print(f"\n--- Final Orchestrated Result ---")
    print(f"{result.get('final_result', 'N/A')[:300]}...")
    
    print("\n" + "=" * 60)
