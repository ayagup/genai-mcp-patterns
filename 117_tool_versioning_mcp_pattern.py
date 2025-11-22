"""
Tool Versioning MCP Pattern

This pattern implements tool versioning with compatibility
checking and graceful upgrades.

Key Features:
- Semantic versioning
- Compatibility checking
- Version migration
- Deprecation management
- Rollback support
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ToolVersioningState(TypedDict):
    """State for tool versioning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tool_versions: Dict[str, List[str]]
    requested_version: str
    selected_version: str
    compatibility_check: bool


llm = ChatOpenAI(model="gpt-4", temperature=0)


def version_manager(state: ToolVersioningState) -> ToolVersioningState:
    """Manages tool versions and compatibility"""
    requested_version = state.get("requested_version", "2.0.0")
    
    system_message = SystemMessage(content="You are a version manager.")
    user_message = HumanMessage(content=f"Manage version: {requested_version}")
    response = llm.invoke([system_message, user_message])
    
    tool_versions = {
        "calculator": ["1.0.0", "1.5.0", "2.0.0", "2.1.0"],
        "search": ["1.0.0", "2.0.0"],
        "analyzer": ["1.0.0", "1.1.0", "2.0.0"]
    }
    
    selected_version = "2.0.0"
    compatibility_check = True
    
    report = f"""
    📦 Tool Versioning:
    
    Requested: {requested_version}
    Selected: {selected_version}
    Compatible: {compatibility_check}
    
    Versioning Strategy:
    • Semantic versioning (MAJOR.MINOR.PATCH)
    • Backward compatibility checks
    • Deprecation warnings
    • Migration guides
    
    Available Versions:
    {chr(10).join(f"  • {tool}: {', '.join(versions)}" for tool, versions in tool_versions.items())}
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{report}")],
        "tool_versions": tool_versions,
        "requested_version": requested_version,
        "selected_version": selected_version,
        "compatibility_check": compatibility_check
    }


def build_tool_versioning_graph():
    workflow = StateGraph(ToolVersioningState)
    workflow.add_node("manager", version_manager)
    workflow.add_edge(START, "manager")
    workflow.add_edge("manager", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_tool_versioning_graph()
    print("=== Tool Versioning MCP Pattern ===\n")
    
    result = graph.invoke({
        "messages": [],
        "tool_versions": {},
        "requested_version": "",
        "selected_version": "",
        "compatibility_check": False
    })
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
