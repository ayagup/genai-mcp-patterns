"""
Broker MCP Pattern

This pattern demonstrates a broker that handles service discovery,
registration, and intelligent routing between service providers and consumers.

Key Features:
- Service registration
- Service discovery
- Load balancing
- Health monitoring
- Dynamic routing
"""

from typing import TypedDict, Sequence, Annotated
import operator
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class BrokerState(TypedDict):
    """State for broker pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    service_request: str
    request_type: str
    available_services: list[dict[str, str]]
    selected_service: str
    service_response: str
    broker_metrics: dict[str, int]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Service Registry
SERVICE_REGISTRY = {
    "payment_processor": [
        {"id": "payment_svc_1", "endpoint": "https://payment1.api", "health": "healthy", "load": 45},
        {"id": "payment_svc_2", "endpoint": "https://payment2.api", "health": "healthy", "load": 30},
    ],
    "notification_service": [
        {"id": "notification_svc_1", "endpoint": "https://notify1.api", "health": "healthy", "load": 60},
        {"id": "notification_svc_2", "endpoint": "https://notify2.api", "health": "degraded", "load": 85},
    ],
    "analytics_service": [
        {"id": "analytics_svc_1", "endpoint": "https://analytics1.api", "health": "healthy", "load": 25},
        {"id": "analytics_svc_2", "endpoint": "https://analytics2.api", "health": "healthy", "load": 40},
    ],
    "storage_service": [
        {"id": "storage_svc_1", "endpoint": "https://storage1.api", "health": "healthy", "load": 50},
    ]
}


# Broker - Service Classifier
def service_classifier(state: BrokerState) -> BrokerState:
    """Classifies service request"""
    service_request = state.get("service_request", "")
    
    system_message = SystemMessage(content="""You are a service classifier in the broker. 
    Identify the type of service needed for the request.""")
    
    user_message = HumanMessage(content=f"""Classify service request: {service_request}

Available service types:
- payment_processor
- notification_service
- analytics_service
- storage_service

Determine required service type.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Classify request
    request_lower = service_request.lower()
    if any(word in request_lower for word in ["pay", "transaction", "checkout"]):
        request_type = "payment_processor"
    elif any(word in request_lower for word in ["notify", "email", "alert"]):
        request_type = "notification_service"
    elif any(word in request_lower for word in ["analyze", "metrics", "report"]):
        request_type = "analytics_service"
    elif any(word in request_lower for word in ["store", "save", "upload"]):
        request_type = "storage_service"
    else:
        request_type = "unknown"
    
    return {
        "messages": [AIMessage(content=f"🔍 Service Classifier: {response.content}\n\n✅ Classified as: {request_type}")],
        "request_type": request_type
    }


# Broker - Service Discovery
def service_discovery(state: BrokerState) -> BrokerState:
    """Discovers available services"""
    request_type = state.get("request_type", "")
    
    system_message = SystemMessage(content="""You are the service discovery component. 
    Find all available instances of the required service type.""")
    
    user_message = HumanMessage(content=f"""Discover services for: {request_type}

Query service registry and return available instances.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Get available services
    available_services = SERVICE_REGISTRY.get(request_type, [])
    
    return {
        "messages": [AIMessage(content=f"📋 Service Discovery: {response.content}\n\n✅ Found {len(available_services)} service instances")],
        "available_services": available_services
    }


# Broker - Health Checker
def health_checker(state: BrokerState) -> BrokerState:
    """Checks health of discovered services"""
    available_services = state.get("available_services", [])
    
    system_message = SystemMessage(content="""You are the health checker. Verify the 
    health status of discovered services before routing.""")
    
    services_status = "\n".join([
        f"  • {svc['id']}: {svc['health']} (load: {svc['load']}%)"
        for svc in available_services
    ])
    
    user_message = HumanMessage(content=f"""Check health of services:

{services_status}

Filter healthy services.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Filter healthy services
    healthy_services = [svc for svc in available_services if svc["health"] == "healthy"]
    
    return {
        "messages": [AIMessage(content=f"🏥 Health Checker: {response.content}\n\n✅ {len(healthy_services)}/{len(available_services)} services healthy")],
        "available_services": healthy_services
    }


# Broker - Load Balancer
def load_balancer(state: BrokerState) -> BrokerState:
    """Selects service based on load"""
    available_services = state.get("available_services", [])
    
    system_message = SystemMessage(content="""You are the load balancer. Select the 
    best service instance based on current load.""")
    
    services_load = "\n".join([
        f"  • {svc['id']}: load {svc['load']}%"
        for svc in available_services
    ])
    
    user_message = HumanMessage(content=f"""Select optimal service:

Available services:
{services_load}

Choose service with lowest load.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Select service with lowest load
    if available_services:
        selected = min(available_services, key=lambda x: x["load"])
        selected_service = selected["id"]
    else:
        selected_service = "none_available"
    
    return {
        "messages": [AIMessage(content=f"⚖️ Load Balancer: {response.content}\n\n✅ Selected: {selected_service}")],
        "selected_service": selected_service
    }


# Broker - Request Router
def request_router(state: BrokerState) -> BrokerState:
    """Routes request to selected service"""
    service_request = state.get("service_request", "")
    selected_service = state.get("selected_service", "")
    
    system_message = SystemMessage(content="""You are the request router. Forward 
    the request to the selected service and handle the response.""")
    
    user_message = HumanMessage(content=f"""Route request to service:

Request: {service_request}
Target Service: {selected_service}

Forward request and get response.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate service response
    service_response = f"SUCCESS: Request processed by {selected_service} at {datetime.now().strftime('%H:%M:%S')}"
    
    return {
        "messages": [AIMessage(content=f"🔀 Request Router: {response.content}\n\n✅ Response received")],
        "service_response": service_response
    }


# Broker - Metrics Collector
def metrics_collector(state: BrokerState) -> BrokerState:
    """Collects broker metrics"""
    selected_service = state.get("selected_service", "")
    
    system_message = SystemMessage(content="""You are the metrics collector. Track 
    broker operations and service usage.""")
    
    user_message = HumanMessage(content=f"""Collect metrics:

Service Used: {selected_service}

Update broker statistics.""")
    
    response = llm.invoke([system_message, user_message])
    
    broker_metrics = {
        "total_requests": 1,
        "successful_routes": 1,
        "failed_routes": 0,
        "avg_response_time_ms": 45
    }
    
    return {
        "messages": [AIMessage(content=f"📊 Metrics Collector: {response.content}\n\n✅ Metrics updated")],
        "broker_metrics": broker_metrics
    }


# Broker Monitor
def broker_monitor(state: BrokerState) -> BrokerState:
    """Monitors broker operations"""
    service_request = state.get("service_request", "")
    request_type = state.get("request_type", "")
    available_services = state.get("available_services", [])
    selected_service = state.get("selected_service", "")
    service_response = state.get("service_response", "")
    broker_metrics = state.get("broker_metrics", {})
    
    services_text = "\n".join([
        f"  • {svc['id']}: {svc['health']} (load: {svc['load']}%)"
        for svc in available_services
    ])
    
    metrics_text = "\n".join([
        f"  • {k}: {v}"
        for k, v in broker_metrics.items()
    ])
    
    summary = f"""
    ✅ BROKER PATTERN COMPLETE
    
    Broker Summary:
    • Request: {service_request[:80]}...
    • Service Type: {request_type}
    • Available Instances: {len(available_services)}
    • Selected Service: {selected_service}
    
    Available Services:
{services_text if services_text else "  • None"}
    
    Broker Workflow:
    1. ✅ Service Classification
    2. ✅ Service Discovery  
    3. ✅ Health Checking
    4. ✅ Load Balancing
    5. ✅ Request Routing
    6. ✅ Metrics Collection
    
    Broker Metrics:
{metrics_text}
    
    Broker Benefits:
    • Dynamic service discovery
    • Automatic load balancing
    • Health-aware routing
    • Service abstraction
    • Centralized metrics
    • Scalable architecture
    
    Service Response:
    {service_response}
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Broker Monitor:\n{summary}")]
    }


# Build the graph
def build_broker_graph():
    """Build the broker pattern graph"""
    workflow = StateGraph(BrokerState)
    
    workflow.add_node("classifier", service_classifier)
    workflow.add_node("discovery", service_discovery)
    workflow.add_node("health", health_checker)
    workflow.add_node("load_balancer", load_balancer)
    workflow.add_node("router", request_router)
    workflow.add_node("metrics", metrics_collector)
    workflow.add_node("monitor", broker_monitor)
    
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", "discovery")
    workflow.add_edge("discovery", "health")
    workflow.add_edge("health", "load_balancer")
    workflow.add_edge("load_balancer", "router")
    workflow.add_edge("router", "metrics")
    workflow.add_edge("metrics", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_broker_graph()
    
    print("=== Broker MCP Pattern ===\n")
    
    # Payment request
    print("\n--- Example 1: Payment Request ---")
    initial_state = {
        "messages": [],
        "service_request": "Process payment of $99.99 for order #12345",
        "request_type": "",
        "available_services": [],
        "selected_service": "",
        "service_response": "",
        "broker_metrics": {}
    }
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    # Analytics request
    print("\n\n--- Example 2: Analytics Request ---")
    initial_state["messages"] = []
    initial_state["service_request"] = "Generate analytics report for Q4 sales data"
    initial_state["available_services"] = []
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
