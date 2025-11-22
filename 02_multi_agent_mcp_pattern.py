"""
Multi-Agent MCP Pattern
========================
Multiple independent agents working collaboratively on different aspects of a task.
Each agent has specialized capabilities and they communicate through a shared state.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
import operator


# Define the state
class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    research_result: str
    analysis_result: str
    summary_result: str
    next_agent: str


# Tools for Research Agent
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Web search results for: {query}"


@tool
def database_query(query: str) -> str:
    """Query internal database."""
    return f"Database results for: {query}"


# Research Agent
def research_agent(state: MultiAgentState):
    """Agent specialized in gathering information."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [web_search, database_query]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state["messages"]
    system_msg = SystemMessage(content="You are a research agent. Gather relevant information.")
    
    response = llm_with_tools.invoke([system_msg] + list(messages))
    
    return {
        "messages": [AIMessage(content=f"Research Agent: Gathered information - {response.content}")],
        "research_result": response.content,
        "next_agent": "analysis"
    }


# Analysis Agent
def analysis_agent(state: MultiAgentState):
    """Agent specialized in analyzing data."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(content=f"You are an analysis agent. Analyze this research data: {state.get('research_result', 'No data')}")
    messages = state["messages"]
    
    response = llm.invoke([system_msg] + list(messages))
    
    return {
        "messages": [AIMessage(content=f"Analysis Agent: Analysis complete - {response.content}")],
        "analysis_result": response.content,
        "next_agent": "summary"
    }


# Summary Agent
def summary_agent(state: MultiAgentState):
    """Agent specialized in creating summaries."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content=f"You are a summary agent. Create a concise summary.\n"
                f"Research: {state.get('research_result', 'N/A')}\n"
                f"Analysis: {state.get('analysis_result', 'N/A')}"
    )
    messages = state["messages"]
    
    response = llm.invoke([system_msg] + list(messages))
    
    return {
        "messages": [AIMessage(content=f"Summary Agent: {response.content}")],
        "summary_result": response.content,
        "next_agent": "end"
    }


# Router function
def route_agent(state: MultiAgentState) -> str:
    """Route to the next agent based on state."""
    next_agent = state.get("next_agent", "research")
    
    if next_agent == "research":
        return "research"
    elif next_agent == "analysis":
        return "analysis"
    elif next_agent == "summary":
        return "summary"
    else:
        return "end"


# Build the multi-agent graph
def create_multi_agent_graph():
    """Create a multi-agent workflow graph."""
    workflow = StateGraph(MultiAgentState)
    
    # Add agent nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("summary", summary_agent)
    
    # Add edges
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "summary")
    workflow.add_edge("summary", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the multi-agent system
    graph = create_multi_agent_graph()
    
    print("=" * 60)
    print("MULTI-AGENT MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Complex task requiring multiple agents
    print("\n[Task: Analyze AI trends]")
    result = graph.invoke({
        "messages": [HumanMessage(content="What are the latest trends in artificial intelligence?")],
        "research_result": "",
        "analysis_result": "",
        "summary_result": "",
        "next_agent": "research"
    })
    
    print("\n--- Agent Outputs ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content}")
    
    print(f"\n--- Final Summary ---")
    print(f"{result.get('summary_result', 'N/A')}")
    
    print("\n" + "=" * 60)
