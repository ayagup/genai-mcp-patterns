"""
Bulkhead MCP Pattern

This pattern isolates resources into separate pools to prevent failures in one
area from affecting others, similar to bulkheads in a ship preventing flooding.

Key Features:
- Resource pool isolation
- Concurrency limits per pool
- Independent failure zones
- Resource allocation
- Overflow handling
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class BulkheadState(TypedDict):
    """State for bulkhead pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    service_name: str
    pool_limits: dict[str, int]  # Pool name -> max concurrent
    pool_usage: dict[str, int]   # Pool name -> current usage
    request_type: str
    pool_assigned: str
    request_accepted: bool
    rejection_reason: str
    metrics: dict[str, dict[str, int]]  # Pool -> {accepted, rejected}


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Request Classifier
def request_classifier(state: BulkheadState) -> BulkheadState:
    """Classifies requests to appropriate resource pool"""
    service_name = state.get("service_name", "")
    request_type = state.get("request_type", "")
    pool_limits = state.get("pool_limits", {})
    
    system_message = SystemMessage(content="""You are a request classifier. 
    Route requests to appropriate resource pools based on type and priority.""")
    
    user_message = HumanMessage(content=f"""Classify request:

Service: {service_name}
Request Type: {request_type}
Available Pools: {', '.join(pool_limits.keys())}

Determine which resource pool should handle this request.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Route to appropriate pool
    if "critical" in request_type.lower():
        pool_assigned = "critical_pool"
    elif "premium" in request_type.lower():
        pool_assigned = "premium_pool"
    else:
        pool_assigned = "standard_pool"
    
    classification = f"""
    📋 Request Classification:
    • Request Type: {request_type}
    • Assigned Pool: {pool_assigned}
    • Pool Limit: {pool_limits.get(pool_assigned, 0)} concurrent
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Request Classifier:\n{response.content}\n{classification}")],
        "pool_assigned": pool_assigned
    }


# Bulkhead Controller
def bulkhead_controller(state: BulkheadState) -> BulkheadState:
    """Controls access to resource pools with bulkhead isolation"""
    pool_assigned = state.get("pool_assigned", "")
    pool_limits = state.get("pool_limits", {})
    pool_usage = state.get("pool_usage", {})
    request_type = state.get("request_type", "")
    metrics = state.get("metrics", {})
    
    system_message = SystemMessage(content="""You are a bulkhead controller. 
    Enforce resource limits and maintain isolation between pools.""")
    
    user_message = HumanMessage(content=f"""Control access to pool:

Pool: {pool_assigned}
Current Usage: {pool_usage.get(pool_assigned, 0)}
Pool Limit: {pool_limits.get(pool_assigned, 0)}
Request: {request_type}

Decide whether to accept or reject the request.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Check if pool has capacity
    current_usage = pool_usage.get(pool_assigned, 0)
    limit = pool_limits.get(pool_assigned, 0)
    
    if current_usage < limit:
        request_accepted = True
        new_usage = current_usage + 1
        pool_usage[pool_assigned] = new_usage
        rejection_reason = ""
        decision = "✅ ACCEPTED"
        
        # Update metrics
        if pool_assigned not in metrics:
            metrics[pool_assigned] = {"accepted": 0, "rejected": 0}
        metrics[pool_assigned]["accepted"] += 1
    else:
        request_accepted = False
        new_usage = current_usage
        rejection_reason = f"Pool '{pool_assigned}' at capacity ({current_usage}/{limit})"
        decision = "❌ REJECTED"
        
        # Update metrics
        if pool_assigned not in metrics:
            metrics[pool_assigned] = {"accepted": 0, "rejected": 0}
        metrics[pool_assigned]["rejected"] += 1
    
    control_decision = f"""
    {decision}
    
    Bulkhead Status:
    • Pool: {pool_assigned}
    • Capacity: {new_usage}/{limit} ({(new_usage/limit*100):.0f}% utilized)
    • Decision: {decision}
    {'• Reason: ' + rejection_reason if rejection_reason else ''}
    
    Pool Isolation:
    Other pools remain unaffected by this decision.
    """
    
    return {
        "messages": [AIMessage(content=f"🚧 Bulkhead Controller:\n{response.content}\n{control_decision}")],
        "pool_usage": pool_usage,
        "request_accepted": request_accepted,
        "rejection_reason": rejection_reason,
        "metrics": metrics
    }


# Resource Allocator
def resource_allocator(state: BulkheadState) -> BulkheadState:
    """Allocates resources from the assigned pool"""
    request_accepted = state.get("request_accepted", False)
    pool_assigned = state.get("pool_assigned", "")
    request_type = state.get("request_type", "")
    rejection_reason = state.get("rejection_reason", "")
    
    if not request_accepted:
        return {
            "messages": [AIMessage(content=f"⚠️ Resource Allocator: Request rejected - {rejection_reason}")]
        }
    
    system_message = SystemMessage(content="""You are a resource allocator. 
    Allocate and manage resources from isolated pools.""")
    
    user_message = HumanMessage(content=f"""Allocate resources:

Pool: {pool_assigned}
Request Type: {request_type}

Allocate necessary resources from the pool.""")
    
    response = llm.invoke([system_message, user_message])
    
    allocation = f"""
    ✅ Resource Allocation Successful
    
    Allocated Resources:
    • Pool: {pool_assigned}
    • Request: {request_type}
    • Status: Processing in isolated pool
    
    Isolation Benefits:
    • Failure in this pool won't affect other pools
    • Dedicated resources prevent resource starvation
    • Independent scaling and monitoring
    • Blast radius limited to single pool
    """
    
    return {
        "messages": [AIMessage(content=f"📦 Resource Allocator:\n{response.content}\n{allocation}")]
    }


# Bulkhead Monitor
def bulkhead_monitor(state: BulkheadState) -> BulkheadState:
    """Monitors bulkhead health and resource utilization"""
    service_name = state.get("service_name", "")
    pool_limits = state.get("pool_limits", {})
    pool_usage = state.get("pool_usage", {})
    metrics = state.get("metrics", {})
    request_accepted = state.get("request_accepted", False)
    pool_assigned = state.get("pool_assigned", "")
    
    pool_status = []
    for pool_name, limit in pool_limits.items():
        usage = pool_usage.get(pool_name, 0)
        utilization = (usage / limit * 100) if limit > 0 else 0
        accepted = metrics.get(pool_name, {}).get("accepted", 0)
        rejected = metrics.get(pool_name, {}).get("rejected", 0)
        total = accepted + rejected
        rejection_rate = (rejected / total * 100) if total > 0 else 0
        
        status_icon = "🟢" if utilization < 70 else "🟡" if utilization < 90 else "🔴"
        
        pool_status.append(f"""
    {status_icon} {pool_name}:
       Capacity: {usage}/{limit} ({utilization:.0f}% utilized)
       Accepted: {accepted} | Rejected: {rejected}
       Rejection Rate: {rejection_rate:.1f}%""")
    
    pools_summary = "".join(pool_status)
    
    outcome = "✅ Request Accepted" if request_accepted else "❌ Request Rejected"
    
    summary = f"""
    {outcome}
    
    Bulkhead Health Report:
    
    Service: {service_name}
    Current Request Pool: {pool_assigned}
    {pools_summary}
    
    Bulkhead Pattern Process:
    1. Classify request → Assign to pool
    2. Check pool capacity → Accept/Reject
    3. Allocate from pool → Process request
    4. Monitor pool health → Adjust limits
    5. Release resources → Update capacity
    
    Bulkhead Pattern Benefits:
    • Fault isolation (failures don't cascade)
    • Resource protection (prevents starvation)
    • Independent scaling per pool
    • Priority-based allocation
    • Predictable performance
    • Limited blast radius
    
    Pool Design Guidelines:
    • Critical operations: Dedicated small pool
    • Premium users: Medium dedicated pool
    • Standard operations: Large shared pool
    • Isolate by: Priority, SLA, tenant, function
    
    Capacity Planning:
    • Monitor utilization trends
    • Adjust limits based on demand
    • Reserve capacity for critical operations
    • Plan for peak load scenarios
    
    Key Insight:
    Bulkhead pattern prevents cascading failures by isolating resources
    into separate pools, ensuring failures in one area don't affect others.
    Like ship bulkheads prevent entire vessel from sinking.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Bulkhead Monitor:\n{summary}")]
    }


# Build the graph
def build_bulkhead_graph():
    """Build the bulkhead pattern graph"""
    workflow = StateGraph(BulkheadState)
    
    workflow.add_node("classifier", request_classifier)
    workflow.add_node("controller", bulkhead_controller)
    workflow.add_node("allocator", resource_allocator)
    workflow.add_node("monitor", bulkhead_monitor)
    
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", "controller")
    workflow.add_edge("controller", "allocator")
    workflow.add_edge("allocator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_bulkhead_graph()
    
    print("=== Bulkhead MCP Pattern ===\n")
    
    # Initialize pools
    initial_pool_limits = {
        "critical_pool": 2,   # Small, dedicated for critical operations
        "premium_pool": 5,    # Medium, for premium users
        "standard_pool": 10   # Large, for standard operations
    }
    
    initial_pool_usage = {
        "critical_pool": 1,   # Already has 1 request
        "premium_pool": 3,    # Already has 3 requests
        "standard_pool": 8    # Already has 8 requests
    }
    
    requests = [
        "critical_user_authentication",
        "premium_data_processing",
        "standard_report_generation",
        "critical_payment_processing",  # This will be rejected (critical pool full)
        "standard_user_query"
    ]
    
    for i, request in enumerate(requests):
        print(f"\n{'='*70}")
        print(f"REQUEST {i+1}: {request}")
        print('='*70)
        
        state = {
            "messages": [],
            "service_name": "OrderProcessingService",
            "pool_limits": initial_pool_limits.copy(),
            "pool_usage": initial_pool_usage.copy(),
            "request_type": request,
            "pool_assigned": "",
            "request_accepted": False,
            "rejection_reason": "",
            "metrics": {}
        }
        
        result = graph.invoke(state)
        
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Update usage for next iteration
        if result.get("request_accepted", False):
            pool = result.get("pool_assigned", "")
            initial_pool_usage[pool] = result.get("pool_usage", {}).get(pool, 0)
