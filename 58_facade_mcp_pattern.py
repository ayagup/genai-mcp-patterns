"""
Facade MCP Pattern

This pattern demonstrates providing a simplified, unified interface to a
complex subsystem of multiple agents.

Key Features:
- Simplified interface
- Subsystem encapsulation
- Coordinated operations
- Complexity hiding
- Unified access point
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class FacadeState(TypedDict):
    """State for facade pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    user_request: str
    orchestration_plan: list[str]
    auth_result: str
    db_result: str
    cache_result: str
    notification_result: str
    final_response: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Facade - Simplified Interface
def facade_controller(state: FacadeState) -> FacadeState:
    """Provides simplified interface to complex subsystem"""
    user_request = state.get("user_request", "")
    
    system_message = SystemMessage(content="""You are a facade controller. Provide 
    a simple, unified interface that hides the complexity of multiple subsystems.""")
    
    user_message = HumanMessage(content=f"""Handle user request: {user_request}

Coordinate subsystems:
- Authentication
- Database
- Cache
- Notification

Create orchestration plan.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Create orchestration plan
    orchestration_plan = [
        "1. Authenticate user",
        "2. Query database",
        "3. Update cache",
        "4. Send notification",
        "5. Return response"
    ]
    
    return {
        "messages": [AIMessage(content=f"🎭 Facade Controller: {response.content}\n\n✅ Created orchestration plan")],
        "orchestration_plan": orchestration_plan
    }


# Authentication Subsystem
def authentication_subsystem(state: FacadeState) -> FacadeState:
    """Complex authentication subsystem"""
    user_request = state.get("user_request", "")
    
    system_message = SystemMessage(content="""You are the authentication subsystem. 
    Handle complex authentication logic including verification, session management.""")
    
    user_message = HumanMessage(content=f"""Authenticate request: {user_request}

Perform:
- User verification
- Session validation
- Permission check""")
    
    response = llm.invoke([system_message, user_message])
    
    auth_result = "Authentication successful - Session ID: abc123, Permissions: read,write"
    
    return {
        "messages": [AIMessage(content=f"🔐 Authentication Subsystem: {response.content}")],
        "auth_result": auth_result
    }


# Database Subsystem
def database_subsystem(state: FacadeState) -> FacadeState:
    """Complex database subsystem"""
    user_request = state.get("user_request", "")
    auth_result = state.get("auth_result", "")
    
    system_message = SystemMessage(content="""You are the database subsystem. Handle 
    complex database operations including queries, transactions, and consistency.""")
    
    user_message = HumanMessage(content=f"""Execute database operations:

Request: {user_request}
Auth: {auth_result[:50]}...

Perform:
- Query execution
- Transaction management
- Data retrieval""")
    
    response = llm.invoke([system_message, user_message])
    
    db_result = "Database query successful - Retrieved 42 records, Execution time: 0.15s"
    
    return {
        "messages": [AIMessage(content=f"🗄️ Database Subsystem: {response.content}")],
        "db_result": db_result
    }


# Cache Subsystem
def cache_subsystem(state: FacadeState) -> FacadeState:
    """Complex cache subsystem"""
    db_result = state.get("db_result", "")
    
    system_message = SystemMessage(content="""You are the cache subsystem. Handle 
    complex caching logic including invalidation, distribution, and consistency.""")
    
    user_message = HumanMessage(content=f"""Update cache:

DB Result: {db_result}

Perform:
- Cache update
- Invalidation
- Distribution""")
    
    response = llm.invoke([system_message, user_message])
    
    cache_result = "Cache updated - TTL: 3600s, Distributed to 3 nodes"
    
    return {
        "messages": [AIMessage(content=f"💾 Cache Subsystem: {response.content}")],
        "cache_result": cache_result
    }


# Notification Subsystem
def notification_subsystem(state: FacadeState) -> FacadeState:
    """Complex notification subsystem"""
    user_request = state.get("user_request", "")
    
    system_message = SystemMessage(content="""You are the notification subsystem. Handle 
    complex notification logic including routing, templating, and delivery.""")
    
    user_message = HumanMessage(content=f"""Send notifications:

Request: {user_request}

Perform:
- Template selection
- Message formatting
- Multi-channel delivery""")
    
    response = llm.invoke([system_message, user_message])
    
    notification_result = "Notifications sent - Email: sent, SMS: sent, Push: sent"
    
    return {
        "messages": [AIMessage(content=f"📧 Notification Subsystem: {response.content}")],
        "notification_result": notification_result
    }


# Response Aggregator
def response_aggregator(state: FacadeState) -> FacadeState:
    """Aggregates responses from all subsystems"""
    user_request = state.get("user_request", "")
    auth_result = state.get("auth_result", "")
    db_result = state.get("db_result", "")
    cache_result = state.get("cache_result", "")
    notification_result = state.get("notification_result", "")
    
    system_message = SystemMessage(content="""You are a response aggregator. Combine 
    results from all subsystems into a simple, user-friendly response.""")
    
    user_message = HumanMessage(content=f"""Aggregate subsystem results:

Original Request: {user_request}

Subsystem Results:
- Auth: {auth_result[:50]}...
- DB: {db_result[:50]}...
- Cache: {cache_result[:50]}...
- Notification: {notification_result[:50]}...

Create simple user response.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📦 Response Aggregator: {response.content}")],
        "final_response": response.content
    }


# Facade Monitor
def facade_monitor(state: FacadeState) -> FacadeState:
    """Monitors facade operations"""
    user_request = state.get("user_request", "")
    orchestration_plan = state.get("orchestration_plan", [])
    final_response = state.get("final_response", "")
    
    plan_text = "\n".join([f"  {step}" for step in orchestration_plan])
    
    summary = f"""
    ✅ FACADE PATTERN COMPLETE
    
    Facade Summary:
    • User Request: {user_request[:80]}...
    • Subsystems Coordinated: 4
    • Orchestration Steps: {len(orchestration_plan)}
    
    Orchestration Plan:
{plan_text}
    
    Subsystem Operations:
    • ✅ Authentication: Completed
    • ✅ Database: Completed
    • ✅ Cache: Completed
    • ✅ Notification: Completed
    
    Facade Benefits:
    • Simple interface to complex subsystem
    • Hidden complexity
    • Coordinated operations
    • Reduced coupling
    • Easier to use and maintain
    • Subsystem independence
    
    Final Response:
    {final_response[:300]}...
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Facade Monitor:\n{summary}")]
    }


# Build the graph
def build_facade_graph():
    """Build the facade pattern graph"""
    workflow = StateGraph(FacadeState)
    
    workflow.add_node("facade", facade_controller)
    workflow.add_node("auth", authentication_subsystem)
    workflow.add_node("database", database_subsystem)
    workflow.add_node("cache", cache_subsystem)
    workflow.add_node("notification", notification_subsystem)
    workflow.add_node("aggregator", response_aggregator)
    workflow.add_node("monitor", facade_monitor)
    
    workflow.add_edge(START, "facade")
    workflow.add_edge("facade", "auth")
    workflow.add_edge("auth", "database")
    workflow.add_edge("database", "cache")
    workflow.add_edge("cache", "notification")
    workflow.add_edge("notification", "aggregator")
    workflow.add_edge("aggregator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_facade_graph()
    
    print("=== Facade MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "user_request": "Get my order history for the last 30 days",
        "orchestration_plan": [],
        "auth_result": "",
        "db_result": "",
        "cache_result": "",
        "notification_result": "",
        "final_response": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Facade Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== User Response ===")
    print(result.get("final_response", "No response generated"))
