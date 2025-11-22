"""
Gateway MCP Pattern

This pattern demonstrates a gateway as the single entry point for
external requests, handling protocol translation, authentication,
rate limiting, and request forwarding.

Key Features:
- Single entry point
- Protocol translation
- Authentication and authorization
- Rate limiting
- Request/response transformation
"""

from typing import TypedDict, Sequence, Annotated
import operator
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class GatewayState(TypedDict):
    """State for gateway pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    external_request: str
    api_key: str
    authenticated: bool
    rate_limit_ok: bool
    internal_request: str
    backend_response: str
    external_response: str
    gateway_logs: list[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Simple rate limiting tracker
request_counts = {}


# Authentication Gateway
def authentication_gateway(state: GatewayState) -> GatewayState:
    """Authenticates incoming requests"""
    external_request = state.get("external_request", "")
    api_key = state.get("api_key", "")
    
    system_message = SystemMessage(content="""You are an authentication gateway. 
    Validate API keys and authenticate external requests.""")
    
    user_message = HumanMessage(content=f"""Authenticate request:
API Key: {api_key}
Request: {external_request}

Validate credentials.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simple authentication
    valid_keys = ["sk-123abc", "sk-456def", "sk-789ghi"]
    authenticated = api_key in valid_keys
    
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] AUTH - Key: {api_key[:8]}... - {'SUCCESS' if authenticated else 'FAILED'}"
    
    return {
        "messages": [AIMessage(content=f"🔐 Authentication Gateway: {response.content}\n\n{'✅ Authenticated' if authenticated else '❌ Authentication failed'}")],
        "authenticated": authenticated,
        "gateway_logs": [log_entry]
    }


# Rate Limiting Gateway
def rate_limiting_gateway(state: GatewayState) -> GatewayState:
    """Enforces rate limits"""
    api_key = state.get("api_key", "")
    authenticated = state.get("authenticated", False)
    
    if not authenticated:
        return {
            "messages": [AIMessage(content="⏭️ Rate Limiting Gateway: Skipped (not authenticated)")],
            "rate_limit_ok": False
        }
    
    system_message = SystemMessage(content="""You are a rate limiting gateway. 
    Enforce request limits to prevent abuse.""")
    
    user_message = HumanMessage(content=f"""Check rate limit for API key: {api_key}

Limit: 10 requests per minute

Verify if request is allowed.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simple rate limiting
    current_count = request_counts.get(api_key, 0)
    rate_limit_ok = current_count < 10
    
    if rate_limit_ok:
        request_counts[api_key] = current_count + 1
    
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] RATE - Key: {api_key[:8]}... - Count: {current_count + 1} - {'OK' if rate_limit_ok else 'EXCEEDED'}"
    
    return {
        "messages": [AIMessage(content=f"⏱️ Rate Limiting Gateway: {response.content}\n\n{'✅ Within limits' if rate_limit_ok else '❌ Rate limit exceeded'}")],
        "rate_limit_ok": rate_limit_ok,
        "gateway_logs": [log_entry]
    }


# Protocol Translator
def protocol_translator(state: GatewayState) -> GatewayState:
    """Translates external protocol to internal format"""
    external_request = state.get("external_request", "")
    rate_limit_ok = state.get("rate_limit_ok", False)
    
    if not rate_limit_ok:
        return {"messages": [AIMessage(content="⏭️ Protocol Translator: Skipped (rate limit exceeded)")]}
    
    system_message = SystemMessage(content="""You are a protocol translator. Convert 
    external API requests to internal service format.""")
    
    user_message = HumanMessage(content=f"""Translate external request to internal format:

External: {external_request}

Convert to internal service protocol.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simple translation
    internal_request = f"INTERNAL_SERVICE_CALL({external_request})"
    
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] TRANSLATE - External → Internal"
    
    return {
        "messages": [AIMessage(content=f"🔄 Protocol Translator: {response.content}\n\n✅ Translated to internal format")],
        "internal_request": internal_request,
        "gateway_logs": [log_entry]
    }


# Backend Forwarder
def backend_forwarder(state: GatewayState) -> GatewayState:
    """Forwards request to backend services"""
    internal_request = state.get("internal_request", "")
    
    if not internal_request:
        return {"messages": [AIMessage(content="⏭️ Backend Forwarder: Skipped")]}
    
    system_message = SystemMessage(content="""You are a backend forwarder. Route 
    requests to appropriate backend services and handle responses.""")
    
    user_message = HumanMessage(content=f"""Forward request to backend:

Request: {internal_request}

Process and return response.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate backend response
    backend_response = f"BACKEND_RESPONSE: {response.content}"
    
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] BACKEND - Request forwarded"
    
    return {
        "messages": [AIMessage(content=f"🔗 Backend Forwarder: {response.content}")],
        "backend_response": backend_response,
        "gateway_logs": [log_entry]
    }


# Response Transformer
def response_transformer(state: GatewayState) -> GatewayState:
    """Transforms backend response to external format"""
    backend_response = state.get("backend_response", "")
    
    if not backend_response:
        return {"messages": [AIMessage(content="⏭️ Response Transformer: Skipped")]}
    
    system_message = SystemMessage(content="""You are a response transformer. Convert 
    internal backend responses to external API format.""")
    
    user_message = HumanMessage(content=f"""Transform response to external format:

Internal: {backend_response[:150]}...

Format for external API.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simple transformation
    external_response = {
        "status": "success",
        "data": backend_response,
        "timestamp": datetime.now().isoformat()
    }
    
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] TRANSFORM - Internal → External"
    
    return {
        "messages": [AIMessage(content=f"🔄 Response Transformer: {response.content}\n\n✅ Transformed to external format")],
        "external_response": str(external_response),
        "gateway_logs": [log_entry]
    }


# Gateway Monitor
def gateway_monitor(state: GatewayState) -> GatewayState:
    """Monitors gateway operations"""
    external_request = state.get("external_request", "")
    api_key = state.get("api_key", "")
    authenticated = state.get("authenticated", False)
    rate_limit_ok = state.get("rate_limit_ok", False)
    gateway_logs = state.get("gateway_logs", [])
    external_response = state.get("external_response", "")
    
    logs_text = "\n".join([f"  {log}" for log in gateway_logs])
    
    summary = f"""
    ✅ GATEWAY PATTERN COMPLETE
    
    Gateway Summary:
    • External Request: {external_request[:80]}...
    • API Key: {api_key[:12]}...
    • Authentication: {'SUCCESS' if authenticated else 'FAILED'}
    • Rate Limit: {'OK' if rate_limit_ok else 'EXCEEDED'}
    • Log Entries: {len(gateway_logs)}
    
    Gateway Logs:
{logs_text}
    
    Gateway Benefits:
    • Single entry point for all requests
    • Centralized authentication
    • Rate limiting protection
    • Protocol translation
    • Request/response transformation
    • Comprehensive logging
    • Backend abstraction
    
    External Response:
    {external_response[:300] if external_response else 'Request blocked'}...
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Gateway Monitor:\n{summary}")]
    }


# Build the graph
def build_gateway_graph():
    """Build the gateway pattern graph"""
    workflow = StateGraph(GatewayState)
    
    workflow.add_node("auth", authentication_gateway)
    workflow.add_node("rate_limit", rate_limiting_gateway)
    workflow.add_node("translator", protocol_translator)
    workflow.add_node("forwarder", backend_forwarder)
    workflow.add_node("transformer", response_transformer)
    workflow.add_node("monitor", gateway_monitor)
    
    workflow.add_edge(START, "auth")
    workflow.add_edge("auth", "rate_limit")
    workflow.add_edge("rate_limit", "translator")
    workflow.add_edge("translator", "forwarder")
    workflow.add_edge("forwarder", "transformer")
    workflow.add_edge("transformer", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_gateway_graph()
    
    print("=== Gateway MCP Pattern ===\n")
    
    # Successful request
    print("\n--- Example 1: Valid Request ---")
    initial_state = {
        "messages": [],
        "external_request": "GET /api/users?limit=10",
        "api_key": "sk-123abc",
        "authenticated": False,
        "rate_limit_ok": False,
        "internal_request": "",
        "backend_response": "",
        "external_response": "",
        "gateway_logs": []
    }
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    # Unauthenticated request
    print("\n\n--- Example 2: Invalid API Key ---")
    initial_state["messages"] = []
    initial_state["api_key"] = "invalid-key"
    initial_state["gateway_logs"] = []
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
