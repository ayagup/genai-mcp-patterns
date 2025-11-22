"""
Hierarchical Agent MCP Pattern
===============================
A supervisor agent manages and coordinates subordinate agents in a tree structure.
The supervisor delegates tasks to specialized workers and aggregates their results.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json


# Define the state
class HierarchicalState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    task: str
    delegated_tasks: dict
    results: dict
    next_worker: str


# Supervisor Agent
def supervisor_agent(state: HierarchicalState):
    """Supervisor that delegates tasks to workers."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_prompt = """You are a supervisor managing a team of workers.
    Available workers:
    - researcher: Gathers information and data
    - analyzer: Analyzes data and provides insights
    - writer: Creates written content and reports
    
    Analyze the task and decide which worker should handle it.
    Respond with JSON: {"worker": "researcher|analyzer|writer", "subtask": "specific instruction"}
    """
    
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    response = llm.invoke(prompt.format_messages(messages=messages))
    
    try:
        decision = json.loads(response.content)
        worker = decision.get("worker", "researcher")
        subtask = decision.get("subtask", "Process the task")
    except:
        worker = "researcher"
        subtask = "Process the task"
    
    return {
        "messages": [AIMessage(content=f"Supervisor: Delegating to {worker} - {subtask}")],
        "delegated_tasks": {worker: subtask},
        "next_worker": worker
    }


# Researcher Worker
def researcher_worker(state: HierarchicalState):
    """Worker specialized in research."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    subtask = state.get("delegated_tasks", {}).get("researcher", "Research the topic")
    
    system_msg = SystemMessage(content=f"You are a research specialist. Task: {subtask}")
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Researcher: {response.content}")],
        "results": {"researcher": response.content},
        "next_worker": "analyzer"
    }


# Analyzer Worker
def analyzer_worker(state: HierarchicalState):
    """Worker specialized in analysis."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    subtask = state.get("delegated_tasks", {}).get("analyzer", "Analyze the data")
    research_data = state.get("results", {}).get("researcher", "No research data")
    
    system_msg = SystemMessage(
        content=f"You are an analysis specialist. Task: {subtask}\nResearch data: {research_data}"
    )
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Analyzer: {response.content}")],
        "results": {"analyzer": response.content},
        "next_worker": "writer"
    }


# Writer Worker
def writer_worker(state: HierarchicalState):
    """Worker specialized in writing."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    subtask = state.get("delegated_tasks", {}).get("writer", "Write the report")
    research_data = state.get("results", {}).get("researcher", "No research")
    analysis_data = state.get("results", {}).get("analyzer", "No analysis")
    
    system_msg = SystemMessage(
        content=f"You are a writing specialist. Task: {subtask}\n"
                f"Research: {research_data}\nAnalysis: {analysis_data}"
    )
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Writer: {response.content}")],
        "results": {"writer": response.content},
        "next_worker": "supervisor_review"
    }


# Supervisor Review
def supervisor_review(state: HierarchicalState):
    """Supervisor reviews and aggregates worker results."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    results = state.get("results", {})
    
    system_msg = SystemMessage(
        content=f"You are a supervisor. Review and synthesize these worker results:\n"
                f"Researcher: {results.get('researcher', 'N/A')}\n"
                f"Analyzer: {results.get('analyzer', 'N/A')}\n"
                f"Writer: {results.get('writer', 'N/A')}\n"
                "Provide a final summary."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Supervisor Review: {response.content}")],
        "next_worker": "end"
    }


# Router
def route_worker(state: HierarchicalState) -> str:
    """Route to the appropriate worker."""
    next_worker = state.get("next_worker", "researcher")
    
    routing = {
        "researcher": "researcher",
        "analyzer": "analyzer",
        "writer": "writer",
        "supervisor_review": "supervisor_review",
        "end": "end"
    }
    
    return routing.get(next_worker, "end")


# Build the hierarchical graph
def create_hierarchical_agent_graph():
    """Create a hierarchical agent workflow graph."""
    workflow = StateGraph(HierarchicalState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("researcher", researcher_worker)
    workflow.add_node("analyzer", analyzer_worker)
    workflow.add_node("writer", writer_worker)
    workflow.add_node("supervisor_review", supervisor_review)
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "researcher")
    workflow.add_edge("researcher", "analyzer")
    workflow.add_edge("analyzer", "writer")
    workflow.add_edge("writer", "supervisor_review")
    workflow.add_edge("supervisor_review", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the hierarchical agent system
    graph = create_hierarchical_agent_graph()
    
    print("=" * 60)
    print("HIERARCHICAL AGENT MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Task delegated through hierarchy
    print("\n[Task: Create a market analysis report]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Create a comprehensive market analysis report on electric vehicles")],
        "task": "Market analysis report",
        "delegated_tasks": {},
        "results": {},
        "next_worker": "researcher"
    })
    
    print("\n--- Hierarchical Agent Workflow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print("\n" + "=" * 60)
