"""
Fan-Out/Fan-In MCP Pattern

This pattern demonstrates distributing a task to multiple agents (fan-out)
and then collecting and aggregating their results (fan-in).

Key Features:
- Parallel task distribution (fan-out)
- Multiple concurrent processors
- Result collection and aggregation (fan-in)
- Scatter-gather pattern
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class FanOutFanInState(TypedDict):
    """State for fan-out/fan-in pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    main_task: str
    worker_assignments: dict[str, str]  # worker -> subtask
    worker_results: dict[str, str]
    aggregated_result: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Task Splitter (Fan-Out)
def task_splitter(state: FanOutFanInState) -> FanOutFanInState:
    """Splits main task into subtasks for parallel processing (Fan-Out)"""
    main_task = state.get("main_task", "")
    
    system_message = SystemMessage(content="""You are a task splitter. Break down complex tasks 
    into smaller subtasks that can be processed in parallel by multiple workers.""")
    
    user_message = HumanMessage(content=f"""Split this task for parallel processing:
    
Main Task: {main_task}

Create subtasks for:
1. Research Worker
2. Analysis Worker
3. Synthesis Worker
4. Validation Worker

Assign specific aspects to each worker.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Define worker assignments
    worker_assignments = {
        "research": f"Research phase of: {main_task}",
        "analysis": f"Analysis phase of: {main_task}",
        "synthesis": f"Synthesis phase of: {main_task}",
        "validation": f"Validation phase of: {main_task}"
    }
    
    assignments_summary = "\n".join([
        f"  {worker}: {subtask}"
        for worker, subtask in worker_assignments.items()
    ])
    
    return {
        "messages": [AIMessage(content=f"📤 Task Splitter (Fan-Out): {response.content}\n\nAssignments:\n{assignments_summary}")],
        "worker_assignments": worker_assignments
    }


# Research Worker
def research_worker(state: FanOutFanInState) -> FanOutFanInState:
    """Handles research subtask"""
    subtask = state.get("worker_assignments", {}).get("research", "")
    
    system_message = SystemMessage(content="""You are a research worker. Gather information, 
    facts, and data related to your assigned subtask.""")
    
    user_message = HumanMessage(content=f"""Research: {subtask}""")
    
    response = llm.invoke([system_message, user_message])
    
    worker_results = state.get("worker_results", {})
    worker_results["research"] = response.content
    
    return {
        "messages": [AIMessage(content=f"🔍 Research Worker: Completed research phase")],
        "worker_results": worker_results
    }


# Analysis Worker
def analysis_worker(state: FanOutFanInState) -> FanOutFanInState:
    """Handles analysis subtask"""
    subtask = state.get("worker_assignments", {}).get("analysis", "")
    
    system_message = SystemMessage(content="""You are an analysis worker. Analyze data, 
    identify patterns, and extract insights from your assigned subtask.""")
    
    user_message = HumanMessage(content=f"""Analyze: {subtask}""")
    
    response = llm.invoke([system_message, user_message])
    
    worker_results = state.get("worker_results", {})
    worker_results["analysis"] = response.content
    
    return {
        "messages": [AIMessage(content=f"📊 Analysis Worker: Completed analysis phase")],
        "worker_results": worker_results
    }


# Synthesis Worker
def synthesis_worker(state: FanOutFanInState) -> FanOutFanInState:
    """Handles synthesis subtask"""
    subtask = state.get("worker_assignments", {}).get("synthesis", "")
    
    system_message = SystemMessage(content="""You are a synthesis worker. Combine information, 
    create connections, and formulate comprehensive conclusions.""")
    
    user_message = HumanMessage(content=f"""Synthesize: {subtask}""")
    
    response = llm.invoke([system_message, user_message])
    
    worker_results = state.get("worker_results", {})
    worker_results["synthesis"] = response.content
    
    return {
        "messages": [AIMessage(content=f"🔬 Synthesis Worker: Completed synthesis phase")],
        "worker_results": worker_results
    }


# Validation Worker
def validation_worker(state: FanOutFanInState) -> FanOutFanInState:
    """Handles validation subtask"""
    subtask = state.get("worker_assignments", {}).get("validation", "")
    
    system_message = SystemMessage(content="""You are a validation worker. Verify accuracy, 
    check for errors, and ensure quality of results.""")
    
    user_message = HumanMessage(content=f"""Validate: {subtask}""")
    
    response = llm.invoke([system_message, user_message])
    
    worker_results = state.get("worker_results", {})
    worker_results["validation"] = response.content
    
    return {
        "messages": [AIMessage(content=f"✅ Validation Worker: Completed validation phase")],
        "worker_results": worker_results
    }


# Result Aggregator (Fan-In)
def result_aggregator(state: FanOutFanInState) -> FanOutFanInState:
    """Aggregates results from all workers (Fan-In)"""
    worker_results = state.get("worker_results", {})
    main_task = state.get("main_task", "")
    
    system_message = SystemMessage(content="""You are a result aggregator. Combine results 
    from parallel workers into a comprehensive final output.""")
    
    results_summary = "\n\n".join([
        f"{worker.upper()} RESULTS:\n{result}"
        for worker, result in worker_results.items()
    ])
    
    user_message = HumanMessage(content=f"""Aggregate these parallel worker results for task: {main_task}

{results_summary}

Create a unified, comprehensive final result.""")
    
    response = llm.invoke([system_message, user_message])
    
    final_summary = f"""
    ✅ FAN-OUT/FAN-IN COMPLETE
    
    Main Task: {main_task}
    Workers Involved: {len(worker_results)}
    
    Processing Flow:
    1. Fan-Out: Task split into {len(worker_results)} subtasks
    2. Parallel Processing: All workers executed concurrently
    3. Fan-In: Results aggregated into final output
    
    Final Aggregated Result:
    {response.content}
    """
    
    return {
        "messages": [AIMessage(content=f"📥 Result Aggregator (Fan-In):\n{final_summary}")],
        "aggregated_result": response.content
    }


# Build the graph
def build_fan_out_fan_in_graph():
    """Build the fan-out/fan-in MCP pattern graph"""
    workflow = StateGraph(FanOutFanInState)
    
    # Add nodes
    workflow.add_node("splitter", task_splitter)
    workflow.add_node("research", research_worker)
    workflow.add_node("analysis", analysis_worker)
    workflow.add_node("synthesis", synthesis_worker)
    workflow.add_node("validation", validation_worker)
    workflow.add_node("aggregator", result_aggregator)
    
    # Fan-out phase
    workflow.add_edge(START, "splitter")
    workflow.add_edge("splitter", "research")
    workflow.add_edge("splitter", "analysis")
    workflow.add_edge("splitter", "synthesis")
    workflow.add_edge("splitter", "validation")
    
    # Fan-in phase
    workflow.add_edge("research", "aggregator")
    workflow.add_edge("analysis", "aggregator")
    workflow.add_edge("synthesis", "aggregator")
    workflow.add_edge("validation", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_fan_out_fan_in_graph()
    
    print("=== Fan-Out/Fan-In MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "main_task": "Develop a comprehensive marketing strategy for launching a new AI-powered product in the healthcare sector",
        "worker_assignments": {},
        "worker_results": {},
        "aggregated_result": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Fan-Out/Fan-In Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Aggregated Result ===")
    print(result.get("aggregated_result", "No result"))
