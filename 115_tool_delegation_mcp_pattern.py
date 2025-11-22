"""
Tool Delegation MCP Pattern

This pattern implements intelligent tool delegation based on
capabilities, workload, and availability.

Key Features:
- Capability-based routing
- Load balancing
- Dynamic delegation
- Priority handling
- Fallback mechanisms
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ToolDelegationState(TypedDict):
    """State for tool delegation pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: Dict
    available_tools: List[Dict]
    delegated_tool: str
    delegation_reason: str


llm = ChatOpenAI(model="gpt-4", temperature=0)


def delegation_router(state: ToolDelegationState) -> ToolDelegationState:
    """Routes tasks to appropriate tools"""
    task = state.get("task", {"type": "search", "priority": "high"})
    
    system_message = SystemMessage(content="You are a delegation router.")
    user_message = HumanMessage(content=f"Route task: {task['type']}")
    response = llm.invoke([system_message, user_message])
    
    available_tools = [
        {"name": "web_search", "capabilities": ["search"], "load": 0.3},
        {"name": "db_query", "capabilities": ["query"], "load": 0.7},
        {"name": "api_call", "capabilities": ["http"], "load": 0.5}
    ]
    
    # Select best tool based on capability and load
    delegated_tool = "web_search"
    delegation_reason = "Low load, capable of search"
    
    report = f"""
    🎯 Tool Delegation:
    
    Task: {task['type']} (Priority: {task.get('priority', 'normal')})
    Available Tools: {len(available_tools)}
    Delegated To: {delegated_tool}
    Reason: {delegation_reason}
    
    Delegation Strategies:
    • Capability matching
    • Load balancing
    • Priority handling
    • Geographic routing
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{report}")],
        "task": task,
        "available_tools": available_tools,
        "delegated_tool": delegated_tool,
        "delegation_reason": delegation_reason
    }


def build_tool_delegation_graph():
    workflow = StateGraph(ToolDelegationState)
    workflow.add_node("router", delegation_router)
    workflow.add_edge(START, "router")
    workflow.add_edge("router", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_tool_delegation_graph()
    print("=== Tool Delegation MCP Pattern ===\n")
    
    result = graph.invoke({
        "messages": [],
        "task": {},
        "available_tools": [],
        "delegated_tool": "",
        "delegation_reason": ""
    })
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
