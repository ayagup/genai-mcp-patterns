"""
Router Agent MCP Pattern

This pattern demonstrates intelligent routing of requests to appropriate
handlers based on content, priority, and capabilities.

Key Features:
- Content-based routing
- Priority-based routing
- Load-aware routing
- Dynamic route selection
- Routing metrics
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class RouterState(TypedDict):
    """State for router pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    request: str
    request_type: str
    priority: str
    route_destination: str
    handler_response: str
    routing_metrics: dict[str, int]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Request Classifier
def request_classifier(state: RouterState) -> RouterState:
    """Classifies incoming requests"""
    request = state.get("request", "")
    
    system_message = SystemMessage(content="""You are a request classifier. Analyze 
    requests to determine their type and priority for proper routing.""")
    
    user_message = HumanMessage(content=f"""Classify request: {request}

Determine:
- Request type (query, transaction, analysis, support)
- Priority (high, medium, low)""")
    
    response = llm.invoke([system_message, user_message])
    
    # Classify request
    request_lower = request.lower()
    
    if any(word in request_lower for word in ["urgent", "critical", "emergency"]):
        priority = "high"
    elif any(word in request_lower for word in ["soon", "important"]):
        priority = "medium"
    else:
        priority = "low"
    
    if any(word in request_lower for word in ["buy", "purchase", "payment", "order"]):
        request_type = "transaction"
    elif any(word in request_lower for word in ["analyze", "report", "metrics", "data"]):
        request_type = "analysis"
    elif any(word in request_lower for word in ["help", "support", "issue", "problem"]):
        request_type = "support"
    else:
        request_type = "query"
    
    return {
        "messages": [AIMessage(content=f"🔍 Request Classifier: {response.content}\n\n✅ Type: {request_type}, Priority: {priority}")],
        "request_type": request_type,
        "priority": priority
    }


# Route Selector
def route_selector(state: RouterState) -> RouterState:
    """Selects the appropriate route/handler"""
    request_type = state.get("request_type", "")
    priority = state.get("priority", "")
    
    system_message = SystemMessage(content="""You are a route selector. Choose the 
    best handler based on request type and priority.""")
    
    user_message = HumanMessage(content=f"""Select route for:
Type: {request_type}
Priority: {priority}

Available handlers:
- query_handler
- transaction_handler
- analysis_handler
- support_handler

Choose optimal route.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Route mapping
    route_map = {
        "query": "query_handler",
        "transaction": "transaction_handler",
        "analysis": "analysis_handler",
        "support": "support_handler"
    }
    
    route_destination = route_map.get(request_type, "query_handler")
    
    return {
        "messages": [AIMessage(content=f"🎯 Route Selector: {response.content}\n\n✅ Routing to: {route_destination}")],
        "route_destination": route_destination
    }


# Query Handler
def query_handler(state: RouterState) -> RouterState:
    """Handles query requests"""
    request = state.get("request", "")
    route_destination = state.get("route_destination", "")
    
    if route_destination != "query_handler":
        return {"messages": [AIMessage(content="⏭️ Query Handler: Skipped")]}
    
    system_message = SystemMessage(content="""You are a query handler. Process 
    information requests and provide accurate responses.""")
    
    user_message = HumanMessage(content=f"""Handle query: {request}

Provide comprehensive answer.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"❓ Query Handler: {response.content}")],
        "handler_response": response.content,
        "routing_metrics": {"query_count": 1}
    }


# Transaction Handler
def transaction_handler(state: RouterState) -> RouterState:
    """Handles transaction requests"""
    request = state.get("request", "")
    route_destination = state.get("route_destination", "")
    
    if route_destination != "transaction_handler":
        return {"messages": [AIMessage(content="⏭️ Transaction Handler: Skipped")]}
    
    system_message = SystemMessage(content="""You are a transaction handler. Process 
    purchases, payments, and orders securely.""")
    
    user_message = HumanMessage(content=f"""Handle transaction: {request}

Process securely.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"💳 Transaction Handler: {response.content}")],
        "handler_response": response.content,
        "routing_metrics": {"transaction_count": 1}
    }


# Analysis Handler
def analysis_handler(state: RouterState) -> RouterState:
    """Handles analysis requests"""
    request = state.get("request", "")
    route_destination = state.get("route_destination", "")
    
    if route_destination != "analysis_handler":
        return {"messages": [AIMessage(content="⏭️ Analysis Handler: Skipped")]}
    
    system_message = SystemMessage(content="""You are an analysis handler. Perform 
    data analysis and generate insights.""")
    
    user_message = HumanMessage(content=f"""Handle analysis: {request}

Provide data-driven insights.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📊 Analysis Handler: {response.content}")],
        "handler_response": response.content,
        "routing_metrics": {"analysis_count": 1}
    }


# Support Handler
def support_handler(state: RouterState) -> RouterState:
    """Handles support requests"""
    request = state.get("request", "")
    route_destination = state.get("route_destination", "")
    
    if route_destination != "support_handler":
        return {"messages": [AIMessage(content="⏭️ Support Handler: Skipped")]}
    
    system_message = SystemMessage(content="""You are a support handler. Help users 
    resolve issues and provide assistance.""")
    
    user_message = HumanMessage(content=f"""Handle support request: {request}

Provide helpful solution.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🆘 Support Handler: {response.content}")],
        "handler_response": response.content,
        "routing_metrics": {"support_count": 1}
    }


# Router Monitor
def router_monitor(state: RouterState) -> RouterState:
    """Monitors routing performance"""
    request = state.get("request", "")
    request_type = state.get("request_type", "")
    priority = state.get("priority", "")
    route_destination = state.get("route_destination", "")
    routing_metrics = state.get("routing_metrics", {})
    handler_response = state.get("handler_response", "")
    
    metrics_text = "\n".join([f"  • {k}: {v}" for k, v in routing_metrics.items()])
    
    summary = f"""
    ✅ ROUTER AGENT PATTERN COMPLETE
    
    Routing Summary:
    • Request: {request[:80]}...
    • Type: {request_type.upper()}
    • Priority: {priority.upper()}
    • Route: {route_destination.upper()}
    
    Routing Metrics:
{metrics_text if metrics_text else "  • No metrics"}
    
    Router Benefits:
    • Intelligent request classification
    • Dynamic routing decisions
    • Priority-based handling
    • Specialized handlers
    • Routing metrics tracking
    • Scalable architecture
    
    Handler Response:
    {handler_response[:300]}...
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Router Monitor:\n{summary}")]
    }


# Build the graph
def build_router_graph():
    """Build the router agent pattern graph"""
    workflow = StateGraph(RouterState)
    
    workflow.add_node("classifier", request_classifier)
    workflow.add_node("selector", route_selector)
    workflow.add_node("query", query_handler)
    workflow.add_node("transaction", transaction_handler)
    workflow.add_node("analysis", analysis_handler)
    workflow.add_node("support", support_handler)
    workflow.add_node("monitor", router_monitor)
    
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", "selector")
    workflow.add_edge("selector", "query")
    workflow.add_edge("query", "transaction")
    workflow.add_edge("transaction", "analysis")
    workflow.add_edge("analysis", "support")
    workflow.add_edge("support", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_router_graph()
    
    print("=== Router Agent MCP Pattern ===\n")
    
    # Query request
    print("\n--- Example 1: Query Request ---")
    initial_state = {
        "messages": [],
        "request": "What are the features of your product?",
        "request_type": "",
        "priority": "",
        "route_destination": "",
        "handler_response": "",
        "routing_metrics": {}
    }
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    # Transaction request
    print("\n\n--- Example 2: Transaction Request ---")
    initial_state["messages"] = []
    initial_state["request"] = "I want to purchase the premium plan urgently"
    initial_state["routing_metrics"] = {}
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
