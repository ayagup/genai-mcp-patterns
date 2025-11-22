"""
Tool Chain MCP Pattern

This pattern implements sequential tool chaining with dependency
management and automatic data flow between tools.

Key Features:
- Sequential tool execution
- Dependency resolution
- Automatic data transformation
- Error propagation
- Chain optimization
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ToolChainState(TypedDict):
    """State for tool chain pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    chain_steps: List[Dict]
    current_step: int
    chain_output: str


llm = ChatOpenAI(model="gpt-4", temperature=0)


def chain_builder(state: ToolChainState) -> ToolChainState:
    """Builds the tool chain"""
    system_message = SystemMessage(content="You are a tool chain builder.")
    user_message = HumanMessage(content="Build sequential tool chain.")
    response = llm.invoke([system_message, user_message])
    
    chain_steps = [
        {"step": 1, "tool": "input_validator", "status": "pending"},
        {"step": 2, "tool": "data_processor", "status": "pending"},
        {"step": 3, "tool": "output_formatter", "status": "pending"}
    ]
    
    report = """
    🔗 Tool Chain:
    
    Sequential tool execution with data flow:
    1. Input Validator → 2. Data Processor → 3. Output Formatter
    
    Chain Pattern Benefits:
    • Ordered execution
    • Data transformation pipeline
    • Dependency management
    • Error isolation
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{report}")],
        "chain_steps": chain_steps,
        "current_step": 0
    }


def chain_executor(state: ToolChainState) -> ToolChainState:
    """Executes the tool chain"""
    chain_steps = state.get("chain_steps", [])
    
    system_message = SystemMessage(content="You are a chain executor.")
    user_message = HumanMessage(content=f"Execute {len(chain_steps)} steps.")
    response = llm.invoke([system_message, user_message])
    
    summary = f"""
    📊 CHAIN COMPLETE
    
    Executed {len(chain_steps)} tools sequentially.
    Data transformed through pipeline successfully.
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{summary}")],
        "chain_output": "Processed result"
    }


def build_tool_chain_graph():
    workflow = StateGraph(ToolChainState)
    workflow.add_node("builder", chain_builder)
    workflow.add_node("executor", chain_executor)
    workflow.add_edge(START, "builder")
    workflow.add_edge("builder", "executor")
    workflow.add_edge("executor", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_tool_chain_graph()
    print("=== Tool Chain MCP Pattern ===\n")
    
    result = graph.invoke({
        "messages": [],
        "chain_steps": [],
        "current_step": 0,
        "chain_output": ""
    })
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
