"""
Proxy Agent MCP Pattern

This pattern demonstrates using a proxy agent as an intermediary that controls
access to another agent, providing additional functionality like caching,
access control, logging, etc.

Key Features:
- Access control and validation
- Request/response caching
- Logging and monitoring
- Lazy initialization
- Protection and security
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ProxyAgentState(TypedDict):
    """State for proxy agent pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    request: str
    user_id: str
    access_granted: bool
    cache_hit: bool
    real_agent_response: str
    proxy_response: str
    access_logs: list[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Simple cache (in-memory)
response_cache = {}


# Access Control Proxy
def access_control_proxy(state: ProxyAgentState) -> ProxyAgentState:
    """Proxy agent that validates access permissions"""
    user_id = state.get("user_id", "")
    request = state.get("request", "")
    
    system_message = SystemMessage(content="""You are an access control proxy. 
    Validate user permissions before allowing access to the real agent.""")
    
    user_message = HumanMessage(content=f"""Validate access for user: {user_id}
Request: {request}

Check permissions and authorize or deny access.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simple access control logic
    authorized_users = ["user_001", "user_002", "admin_001"]
    access_granted = user_id in authorized_users
    
    log_entry = f"[ACCESS] User {user_id} - {'GRANTED' if access_granted else 'DENIED'}"
    
    return {
        "messages": [AIMessage(content=f"🔐 Access Control Proxy: {response.content}\n\n{'✅ Access granted' if access_granted else '❌ Access denied'}")],
        "access_granted": access_granted,
        "access_logs": [log_entry]
    }


# Caching Proxy
def caching_proxy(state: ProxyAgentState) -> ProxyAgentState:
    """Proxy agent that caches responses"""
    request = state.get("request", "")
    access_granted = state.get("access_granted", False)
    
    if not access_granted:
        return {
            "messages": [AIMessage(content="⛔ Caching Proxy: Access denied, skipping cache lookup")],
            "cache_hit": False
        }
    
    system_message = SystemMessage(content="""You are a caching proxy. Check if the 
    request is cached and return cached response if available.""")
    
    user_message = HumanMessage(content=f"""Check cache for request: {request}

Look for cached response.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Check cache
    cache_key = request.lower().strip()
    cache_hit = cache_key in response_cache
    
    if cache_hit:
        cached_response = response_cache[cache_key]
        return {
            "messages": [AIMessage(content=f"💾 Caching Proxy: {response.content}\n\n✅ Cache HIT - Returning cached response")],
            "cache_hit": True,
            "proxy_response": cached_response
        }
    else:
        return {
            "messages": [AIMessage(content=f"💾 Caching Proxy: {response.content}\n\n❌ Cache MISS - Forwarding to real agent")],
            "cache_hit": False
        }


# Real Agent (Protected Resource)
def real_agent(state: ProxyAgentState) -> ProxyAgentState:
    """The actual agent that performs the work"""
    request = state.get("request", "")
    cache_hit = state.get("cache_hit", False)
    
    if cache_hit:
        return {
            "messages": [AIMessage(content="⏭️ Real Agent: Skipped (cache hit)")]
        }
    
    system_message = SystemMessage(content="""You are the real agent that processes 
    requests. Provide comprehensive and helpful responses.""")
    
    user_message = HumanMessage(content=f"""Process request: {request}

Provide detailed response.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Cache the response
    cache_key = request.lower().strip()
    response_cache[cache_key] = response.content
    
    return {
        "messages": [AIMessage(content=f"🤖 Real Agent: {response.content}")],
        "real_agent_response": response.content
    }


# Response Proxy
def response_proxy(state: ProxyAgentState) -> ProxyAgentState:
    """Proxy that formats and delivers the final response"""
    cache_hit = state.get("cache_hit", False)
    proxy_response = state.get("proxy_response", "")
    real_agent_response = state.get("real_agent_response", "")
    user_id = state.get("user_id", "")
    
    final_response = proxy_response if cache_hit else real_agent_response
    
    system_message = SystemMessage(content="""You are a response proxy. Format and 
    deliver the final response to the user.""")
    
    user_message = HumanMessage(content=f"""Format response for user {user_id}:

Response: {final_response[:200]}...

Add response metadata.""")
    
    response = llm.invoke([system_message, user_message])
    
    log_entry = f"[RESPONSE] User {user_id} - {'Cache' if cache_hit else 'Fresh'}"
    
    return {
        "messages": [AIMessage(content=f"📤 Response Proxy: {response.content}\n\n✅ Response delivered")],
        "proxy_response": final_response,
        "access_logs": [log_entry]
    }


# Audit Logger Proxy
def audit_logger_proxy(state: ProxyAgentState) -> ProxyAgentState:
    """Proxy that logs all interactions"""
    user_id = state.get("user_id", "")
    request = state.get("request", "")
    access_granted = state.get("access_granted", False)
    cache_hit = state.get("cache_hit", False)
    access_logs = state.get("access_logs", [])
    
    summary = f"""
    ✅ PROXY AGENT PATTERN COMPLETE
    
    Audit Trail:
    • User: {user_id}
    • Request: {request[:80]}...
    • Access: {'Granted' if access_granted else 'Denied'}
    • Cache: {'Hit' if cache_hit else 'Miss'}
    • Logs: {len(access_logs)} entries
    
    Access Logs:
    {'  ' + chr(10).join(access_logs)}
    
    Proxy Benefits:
    • Controlled access to protected resources
    • Response caching for performance
    • Comprehensive audit logging
    • Lazy initialization support
    • Security and validation layer
    • Transparent to client
    
    Cache Status:
    • Total cached responses: {len(response_cache)}
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Audit Logger:\n{summary}")]
    }


# Build the graph
def build_proxy_agent_graph():
    """Build the proxy agent pattern graph"""
    workflow = StateGraph(ProxyAgentState)
    
    workflow.add_node("access_control", access_control_proxy)
    workflow.add_node("caching", caching_proxy)
    workflow.add_node("real_agent", real_agent)
    workflow.add_node("response", response_proxy)
    workflow.add_node("audit", audit_logger_proxy)
    
    workflow.add_edge(START, "access_control")
    workflow.add_edge("access_control", "caching")
    workflow.add_edge("caching", "real_agent")
    workflow.add_edge("real_agent", "response")
    workflow.add_edge("response", "audit")
    workflow.add_edge("audit", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_proxy_agent_graph()
    
    print("=== Proxy Agent MCP Pattern ===\n")
    
    # First request (cache miss)
    print("\n--- Request 1: First time (cache miss) ---")
    initial_state = {
        "messages": [],
        "request": "What is the capital of France?",
        "user_id": "user_001",
        "access_granted": False,
        "cache_hit": False,
        "real_agent_response": "",
        "proxy_response": "",
        "access_logs": []
    }
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    # Second request (cache hit)
    print("\n\n--- Request 2: Same query (cache hit) ---")
    initial_state["messages"] = []
    initial_state["access_logs"] = []
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    # Unauthorized request
    print("\n\n--- Request 3: Unauthorized user ---")
    initial_state["messages"] = []
    initial_state["user_id"] = "unauthorized_user"
    initial_state["access_logs"] = []
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
