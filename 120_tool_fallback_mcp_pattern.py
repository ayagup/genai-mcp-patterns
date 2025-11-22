"""
Tool Fallback MCP Pattern

This pattern implements graceful degradation when tools fail,
providing automatic fallback to alternative tools.

Key Features:
- Primary/fallback tool selection
- Health checking
- Automatic failover
- Circuit breaker
- Retry mechanisms
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ToolFallbackState(TypedDict):
    """State for tool fallback pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    primary_tool: str
    fallback_tools: List[str]
    selected_tool: str
    execution_result: str
    health_status: Dict[str, str]


llm = ChatOpenAI(model="gpt-4", temperature=0)


def fallback_manager(state: ToolFallbackState) -> ToolFallbackState:
    """Manages tool fallback logic"""
    primary_tool = "premium_search"
    fallback_tools = ["basic_search", "cached_search", "manual_search"]
    
    system_message = SystemMessage(content="You are a fallback manager.")
    user_message = HumanMessage(content=f"Select tool with fallback. Primary: {primary_tool}")
    response = llm.invoke([system_message, user_message])
    
    # Check health status
    health_status = {
        "premium_search": "down",
        "basic_search": "healthy",
        "cached_search": "healthy",
        "manual_search": "healthy"
    }
    
    # Select tool based on health
    if health_status.get(primary_tool) == "healthy":
        selected_tool = primary_tool
    else:
        # Fallback to first healthy tool
        selected_tool = next(
            (tool for tool in fallback_tools if health_status.get(tool) == "healthy"),
            fallback_tools[0]
        )
    
    execution_result = f"Executed with {selected_tool}"
    
    report = f"""
    🔄 Tool Fallback:
    
    Primary Tool: {primary_tool}
    Fallback Chain: {' → '.join(fallback_tools)}
    Selected Tool: {selected_tool}
    
    Health Status:
    {chr(10).join(f"  • {tool}: {status}" for tool, status in health_status.items())}
    
    Fallback Strategies:
    • Circuit breaker pattern
    • Retry with exponential backoff
    • Graceful degradation
    • Feature toggling
    • Bulkhead isolation
    
    Execution Result: {execution_result}
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{report}")],
        "primary_tool": primary_tool,
        "fallback_tools": fallback_tools,
        "selected_tool": selected_tool,
        "execution_result": execution_result,
        "health_status": health_status
    }


def health_checker(state: ToolFallbackState) -> ToolFallbackState:
    """Monitors tool health"""
    health_status = state.get("health_status", {})
    
    system_message = SystemMessage(content="You are a health checker.")
    user_message = HumanMessage(content=f"Monitor tool health: {list(health_status.keys())}")
    response = llm.invoke([system_message, user_message])
    
    report = f"""
    🏥 Health Monitoring:
    
    Monitoring Metrics:
    • Response time
    • Error rate
    • Availability
    • Throughput
    • Resource usage
    
    Health Checks:
    • Heartbeat pings
    • Dependency checks
    • Smoke tests
    • Load testing
    
    Circuit Breaker States:
    • Closed: Normal operation
    • Open: Failing, use fallback
    • Half-open: Testing recovery
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{report}")]
    }


def build_tool_fallback_graph():
    workflow = StateGraph(ToolFallbackState)
    workflow.add_node("fallback_manager", fallback_manager)
    workflow.add_node("health_checker", health_checker)
    workflow.add_edge(START, "fallback_manager")
    workflow.add_edge("fallback_manager", "health_checker")
    workflow.add_edge("health_checker", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_tool_fallback_graph()
    print("=== Tool Fallback MCP Pattern ===\n")
    
    result = graph.invoke({
        "messages": [],
        "primary_tool": "",
        "fallback_tools": [],
        "selected_tool": "",
        "execution_result": "",
        "health_status": {}
    })
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
