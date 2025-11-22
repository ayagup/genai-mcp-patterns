"""
Single-Agent MCP Pattern
========================
A single autonomous agent that processes tasks independently.
This pattern demonstrates a basic agent with tools and decision-making capabilities.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from typing import Literal


# Define tools for the agent
@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for information."""
    return f"Knowledge result for: {query}"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_info(topic: str) -> str:
    """Get current information about a topic."""
    return f"Current information about: {topic}"


# Define the agent node
def agent_node(state: MessagesState):
    """Single agent that processes messages and uses tools."""
    tools = [search_knowledge, calculate, get_current_info]
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Define routing logic
def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    """Determine if the agent should continue or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


# Build the graph
def create_single_agent_graph():
    """Create a single-agent workflow graph."""
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode([search_knowledge, calculate, get_current_info]))
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the single agent
    graph = create_single_agent_graph()
    
    # Test the agent
    print("=" * 60)
    print("SINGLE-AGENT MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example 1: Simple query
    print("\n[Example 1: Information Query]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Search for information about artificial intelligence")]
    })
    print(f"Response: {result['messages'][-1].content}")
    
    # Example 2: Calculation
    print("\n[Example 2: Calculation]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Calculate 125 * 48")]
    })
    print(f"Response: {result['messages'][-1].content}")
    
    # Example 3: Multi-step task
    print("\n[Example 3: Complex Query]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Get current information about climate change and calculate 2024 - 1990")]
    })
    print(f"Response: {result['messages'][-1].content}")
    
    print("\n" + "=" * 60)
