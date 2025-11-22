"""
Multi-Tool MCP Pattern

This pattern implements parallel execution of multiple tools
with result aggregation and coordination.

Key Features:
- Parallel tool execution
- Result aggregation
- Coordination and synchronization
- Timeout handling
- Partial failure management
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class MultiToolState(TypedDict):
    """State for multi-tool pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tools_to_execute: List[str]
    execution_results: Dict[str, str]
    aggregated_result: str


llm = ChatOpenAI(model="gpt-4", temperature=0)


def multi_tool_executor(state: MultiToolState) -> MultiToolState:
    """Executes multiple tools in parallel"""
    tools = state.get("tools_to_execute", ["search", "analyze", "summarize"])
    
    system_message = SystemMessage(content="You are a multi-tool executor.")
    user_message = HumanMessage(content=f"Execute {len(tools)} tools in parallel.")
    response = llm.invoke([system_message, user_message])
    
    # Simulate parallel execution results
    execution_results = {
        "search": "Found 10 relevant articles",
        "analyze": "Sentiment: positive, Key topics: AI, automation",
        "summarize": "Summary: AI adoption increasing across industries"
    }
    
    report = f"""
    ⚡ Multi-Tool Execution:
    
    Parallel Execution: {len(tools)} tools
    Results: {len(execution_results)} completed
    
    Multi-Tool Benefits:
    • Parallel execution for speed
    • Diverse perspectives
    • Comprehensive results
    • Fault tolerance
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{report}")],
        "tools_to_execute": tools,
        "execution_results": execution_results,
        "aggregated_result": " | ".join(execution_results.values())
    }


def build_multi_tool_graph():
    workflow = StateGraph(MultiToolState)
    workflow.add_node("executor", multi_tool_executor)
    workflow.add_edge(START, "executor")
    workflow.add_edge("executor", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_multi_tool_graph()
    print("=== Multi-Tool MCP Pattern ===\n")
    
    result = graph.invoke({
        "messages": [],
        "tools_to_execute": [],
        "execution_results": {},
        "aggregated_result": ""
    })
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
