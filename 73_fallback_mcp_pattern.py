"""
Fallback MCP Pattern

This pattern provides alternative execution paths when primary operations fail,
ensuring graceful handling of errors with backup strategies.

Key Features:
- Primary execution path
- Multiple fallback options
- Cascading fallbacks
- Fallback prioritization
- Error recovery
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class FallbackState(TypedDict):
    """State for fallback pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    primary_result: str
    fallback_level: int
    execution_path: list[str]
    final_result: str
    success: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Primary Executor
def primary_executor(state: FallbackState) -> FallbackState:
    """Attempts primary execution path"""
    task = state.get("task", "")
    
    system_message = SystemMessage(content="""You are a primary executor. 
    Execute the task using the primary, preferred method.""")
    
    user_message = HumanMessage(content=f"""Execute task (Primary Path):

Task: {task}

Use primary execution method.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate primary failure (for demonstration)
    primary_success = False
    primary_result = "PRIMARY_FAILED: Service unavailable"
    
    execution_path = ["Primary (Failed)"]
    
    status = "❌ Failed" if not primary_success else "✅ Success"
    
    return {
        "messages": [AIMessage(content=f"🎯 Primary Executor:\n{response.content}\n\n{status}: {primary_result}")],
        "primary_result": primary_result,
        "execution_path": execution_path,
        "success": primary_success
    }


# Fallback Level 1 - Cache
def fallback_cache(state: FallbackState) -> FallbackState:
    """First fallback: Use cached data"""
    task = state.get("task", "")
    primary_result = state.get("primary_result", "")
    execution_path = state.get("execution_path", [])
    
    system_message = SystemMessage(content="""You are a cache fallback handler. 
    Attempt to serve request from cache when primary fails.""")
    
    user_message = HumanMessage(content=f"""Fallback to cache:

Task: {task}
Primary Result: {primary_result}

Check cache for cached response.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate cache miss
    cache_success = False
    cache_result = "CACHE_MISS: No cached data available"
    
    execution_path = execution_path + ["Fallback L1: Cache (Miss)"]
    
    status = "❌ Cache Miss" if not cache_success else "✅ Cache Hit"
    
    return {
        "messages": [AIMessage(content=f"💾 Fallback Level 1 (Cache):\n{response.content}\n\n{status}")],
        "fallback_level": 1,
        "execution_path": execution_path,
        "final_result": cache_result,
        "success": cache_success
    }


# Fallback Level 2 - Secondary Service
def fallback_secondary_service(state: FallbackState) -> FallbackState:
    """Second fallback: Use secondary service"""
    task = state.get("task", "")
    execution_path = state.get("execution_path", [])
    
    system_message = SystemMessage(content="""You are a secondary service fallback handler. 
    Route request to backup service when primary and cache fail.""")
    
    user_message = HumanMessage(content=f"""Fallback to secondary service:

Task: {task}

Use backup service provider.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate secondary service success
    secondary_success = True
    secondary_result = "SUCCESS: Data retrieved from secondary service (slower but available)"
    
    execution_path = execution_path + ["Fallback L2: Secondary Service (Success)"]
    
    status = "✅ Success" if secondary_success else "❌ Failed"
    
    return {
        "messages": [AIMessage(content=f"🔄 Fallback Level 2 (Secondary Service):\n{response.content}\n\n{status}")],
        "fallback_level": 2,
        "execution_path": execution_path,
        "final_result": secondary_result,
        "success": secondary_success
    }


# Fallback Level 3 - Degraded Response
def fallback_degraded(state: FallbackState) -> FallbackState:
    """Third fallback: Return degraded/minimal response"""
    task = state.get("task", "")
    execution_path = state.get("execution_path", [])
    
    system_message = SystemMessage(content="""You are a degraded response fallback handler. 
    Provide minimal/default response when all services fail.""")
    
    user_message = HumanMessage(content=f"""Fallback to degraded response:

Task: {task}

Return minimal safe default response.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Always succeeds with degraded response
    degraded_success = True
    degraded_result = "DEGRADED: Using default/minimal response (limited functionality)"
    
    execution_path = execution_path + ["Fallback L3: Degraded Response (Success)"]
    
    return {
        "messages": [AIMessage(content=f"⚠️ Fallback Level 3 (Degraded):\n{response.content}\n\n✅ Degraded response provided")],
        "fallback_level": 3,
        "execution_path": execution_path,
        "final_result": degraded_result,
        "success": degraded_success
    }


# Fallback Monitor
def fallback_monitor(state: FallbackState) -> FallbackState:
    """Monitors fallback execution and final outcome"""
    task = state.get("task", "")
    primary_result = state.get("primary_result", "")
    fallback_level = state.get("fallback_level", 0)
    execution_path = state.get("execution_path", [])
    final_result = state.get("final_result", "")
    success = state.get("success", False)
    
    execution_summary = "\n".join([
        f"    {i+1}. {path}"
        for i, path in enumerate(execution_path)
    ])
    
    fallback_names = {
        0: "Primary only",
        1: "Cache fallback",
        2: "Secondary service fallback",
        3: "Degraded response fallback"
    }
    
    outcome_emoji = "✅" if success else "❌"
    
    summary = f"""
    ✅ FALLBACK PATTERN COMPLETE
    
    Execution Summary:
    • Task: {task[:80]}...
    • Final Outcome: {outcome_emoji} {'Success' if success else 'All fallbacks exhausted'}
    • Fallback Level: {fallback_level} ({fallback_names.get(fallback_level, 'Unknown')})
    • Total Attempts: {len(execution_path)}
    
    Execution Path:
{execution_summary}
    
    Final Result:
    {final_result}
    
    Fallback Hierarchy:
    1. Primary Path: Preferred, full-featured service
    2. L1 - Cache: Fast, potentially stale data
    3. L2 - Secondary Service: Slower backup service
    4. L3 - Degraded: Minimal/default response
    
    Fallback Pattern Process:
    1. Try Primary → 2. If Fail, Try Cache → 3. If Fail, Try Secondary → 
    4. If Fail, Use Degraded → 5. Return Best Available Result
    
    Fallback Pattern Benefits:
    • Graceful degradation
    • Multiple recovery options
    • Cascading fallback strategy
    • Always provides some response
    • Improved availability
    • Better user experience
    
    Performance Characteristics:
    • Primary: Low latency, full features
    • Cache: Very low latency, may be stale
    • Secondary: Medium latency, full features
    • Degraded: Low latency, limited features
    
    Key Insight:
    Fallback pattern ensures system keeps functioning even when primary
    services fail, trading off features or performance for availability.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Fallback Monitor:\n{summary}")]
    }


# Build the graph with conditional fallback paths
def build_fallback_graph():
    """Build the fallback pattern graph"""
    workflow = StateGraph(FallbackState)
    
    workflow.add_node("primary", primary_executor)
    workflow.add_node("cache", fallback_cache)
    workflow.add_node("secondary", fallback_secondary_service)
    workflow.add_node("degraded", fallback_degraded)
    workflow.add_node("monitor", fallback_monitor)
    
    # Try primary first
    workflow.add_edge(START, "primary")
    
    # If primary fails, try cache
    workflow.add_edge("primary", "cache")
    
    # If cache fails, try secondary
    workflow.add_edge("cache", "secondary")
    
    # If secondary fails, use degraded (always succeeds)
    workflow.add_edge("secondary", "degraded")
    
    # Monitor results
    workflow.add_edge("degraded", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_fallback_graph()
    
    print("=== Fallback MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "task": "Fetch user profile data from user service",
        "primary_result": "",
        "fallback_level": 0,
        "execution_path": [],
        "final_result": "",
        "success": False
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("FALLBACK PATTERN COMPLETE")
    print("=" * 70)
    print(f"\nTask: {state['task']}")
    print(f"Fallback Level Used: {result.get('fallback_level', 0)}")
    print(f"Success: {result.get('success', False)}")
    print(f"Execution Attempts: {len(result.get('execution_path', []))}")
