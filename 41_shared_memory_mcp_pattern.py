"""
Shared Memory MCP Pattern

This pattern demonstrates agents sharing a common memory space to exchange
information and coordinate their activities.

Key Features:
- Common memory accessible by all agents
- Read/write operations to shared state
- Information sharing across agents
- Coordinated state management
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state with shared memory
class SharedMemoryState(TypedDict):
    """State with shared memory for all agents"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    shared_memory: dict[str, any]  # Shared memory accessible by all agents
    task: str
    final_result: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Data Collector Agent
def data_collector(state: SharedMemoryState) -> SharedMemoryState:
    """Collects data and writes to shared memory"""
    task = state.get("task", "")
    shared_memory = state.get("shared_memory", {})
    
    system_message = SystemMessage(content="""You are a data collector. Gather information 
    and store it in shared memory for other agents to use.""")
    
    user_message = HumanMessage(content=f"""Collect data for: {task}
    
Store your findings in shared memory under the key 'collected_data'.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Write to shared memory
    shared_memory["collected_data"] = {
        "timestamp": "2025-11-09T10:00:00",
        "source": "data_collector",
        "data": response.content,
        "status": "collected"
    }
    
    return {
        "messages": [AIMessage(content=f"📥 Data Collector: {response.content}\n\n✅ Data written to shared memory")],
        "shared_memory": shared_memory
    }


# Analyzer Agent
def analyzer(state: SharedMemoryState) -> SharedMemoryState:
    """Reads from shared memory, analyzes data, and writes results"""
    shared_memory = state.get("shared_memory", {})
    
    # Read from shared memory
    collected_data = shared_memory.get("collected_data", {})
    
    system_message = SystemMessage(content="""You are an analyzer. Read data from shared memory, 
    analyze it, and write your analysis back to shared memory.""")
    
    user_message = HumanMessage(content=f"""Analyze this data from shared memory:

{collected_data.get('data', 'No data available')}

Store your analysis in shared memory under the key 'analysis'.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Write analysis to shared memory
    shared_memory["analysis"] = {
        "timestamp": "2025-11-09T10:05:00",
        "source": "analyzer",
        "analysis": response.content,
        "based_on": "collected_data"
    }
    
    return {
        "messages": [AIMessage(content=f"🔍 Analyzer: {response.content}\n\n✅ Analysis written to shared memory")],
        "shared_memory": shared_memory
    }


# Decision Maker Agent
def decision_maker(state: SharedMemoryState) -> SharedMemoryState:
    """Reads analysis from shared memory and makes decisions"""
    shared_memory = state.get("shared_memory", {})
    
    # Read from shared memory
    collected_data = shared_memory.get("collected_data", {})
    analysis = shared_memory.get("analysis", {})
    
    system_message = SystemMessage(content="""You are a decision maker. Read data and analysis 
    from shared memory, then make informed decisions.""")
    
    user_message = HumanMessage(content=f"""Make decisions based on:

Data: {collected_data.get('data', 'No data')}
Analysis: {analysis.get('analysis', 'No analysis')}

Store your decision in shared memory under the key 'decision'.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Write decision to shared memory
    shared_memory["decision"] = {
        "timestamp": "2025-11-09T10:10:00",
        "source": "decision_maker",
        "decision": response.content,
        "based_on": ["collected_data", "analysis"]
    }
    
    return {
        "messages": [AIMessage(content=f"🎯 Decision Maker: {response.content}\n\n✅ Decision written to shared memory")],
        "shared_memory": shared_memory
    }


# Action Executor Agent
def action_executor(state: SharedMemoryState) -> SharedMemoryState:
    """Reads decision from shared memory and executes actions"""
    shared_memory = state.get("shared_memory", {})
    
    # Read from shared memory
    decision = shared_memory.get("decision", {})
    
    system_message = SystemMessage(content="""You are an action executor. Read the decision 
    from shared memory and execute appropriate actions.""")
    
    user_message = HumanMessage(content=f"""Execute actions based on this decision:

{decision.get('decision', 'No decision available')}

Store execution results in shared memory under the key 'execution'.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Write execution results to shared memory
    shared_memory["execution"] = {
        "timestamp": "2025-11-09T10:15:00",
        "source": "action_executor",
        "results": response.content,
        "status": "completed"
    }
    
    return {
        "messages": [AIMessage(content=f"⚡ Action Executor: {response.content}\n\n✅ Execution results written to shared memory")],
        "shared_memory": shared_memory
    }


# Memory Monitor
def memory_monitor(state: SharedMemoryState) -> SharedMemoryState:
    """Monitors and reports on shared memory contents"""
    shared_memory = state.get("shared_memory", {})
    
    memory_summary = "📋 SHARED MEMORY CONTENTS:\n\n"
    for key, value in shared_memory.items():
        memory_summary += f"  {key}:\n"
        if isinstance(value, dict):
            for k, v in value.items():
                memory_summary += f"    - {k}: {v}\n"
        else:
            memory_summary += f"    {value}\n"
        memory_summary += "\n"
    
    final_summary = f"""
    ✅ SHARED MEMORY PATTERN COMPLETE
    
    Total Memory Entries: {len(shared_memory)}
    Agents Participated: 4 (Data Collector, Analyzer, Decision Maker, Action Executor)
    
{memory_summary}
    
    Shared Memory Benefits:
    • All agents can access common information
    • Sequential information building
    • No direct agent-to-agent communication needed
    • Centralized state management
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Memory Monitor:\n{final_summary}")],
        "final_result": shared_memory.get("execution", {}).get("results", "No results")
    }


# Build the graph
def build_shared_memory_graph():
    """Build the shared memory MCP pattern graph"""
    workflow = StateGraph(SharedMemoryState)
    
    workflow.add_node("collector", data_collector)
    workflow.add_node("analyzer", analyzer)
    workflow.add_node("decision_maker", decision_maker)
    workflow.add_node("executor", action_executor)
    workflow.add_node("monitor", memory_monitor)
    
    # Sequential execution with shared memory access
    workflow.add_edge(START, "collector")
    workflow.add_edge("collector", "analyzer")
    workflow.add_edge("analyzer", "decision_maker")
    workflow.add_edge("decision_maker", "executor")
    workflow.add_edge("executor", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_shared_memory_graph()
    
    print("=== Shared Memory MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "shared_memory": {},
        "task": "Analyze customer feedback and recommend product improvements",
        "final_result": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Shared Memory Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Shared Memory State ===")
    for key, value in result["shared_memory"].items():
        print(f"\n{key}:")
        print(f"  {value}")
