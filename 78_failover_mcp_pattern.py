"""
Failover MCP Pattern

This pattern automatically switches to backup resources when primary resources
fail, ensuring continuous service availability.

Key Features:
- Primary/backup configuration
- Automatic failover detection
- Seamless transition
- Health monitoring
- Failback capability
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class FailoverState(TypedDict):
    """State for failover pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    service_name: str
    primary_status: str  # "healthy", "degraded", "failed"
    backup_status: str
    active_instance: str  # "primary" or "backup"
    failover_occurred: bool
    failover_reason: str
    failback_ready: bool
    request: str
    result: str
    response_time: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Health Monitor
def health_monitor(state: FailoverState) -> FailoverState:
    """Monitors health of primary and backup instances"""
    service_name = state.get("service_name", "")
    
    system_message = SystemMessage(content="""You are a health monitor. 
    Monitor health of primary and backup instances for failover decisions.""")
    
    user_message = HumanMessage(content=f"""Monitor health:

Service: {service_name}

Check health status of primary and backup instances.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate health check
    primary_status = "failed"  # Primary is down
    backup_status = "healthy"  # Backup is healthy
    
    primary_icon = {"healthy": "🟢", "degraded": "🟡", "failed": "🔴"}.get(primary_status, "⚪")
    backup_icon = {"healthy": "🟢", "degraded": "🟡", "failed": "🔴"}.get(backup_status, "⚪")
    
    health_report = f"""
    Health Status Report:
    
    {primary_icon} Primary Instance: {primary_status.upper()}
    {backup_icon} Backup Instance: {backup_status.upper()}
    
    {'⚠️ PRIMARY FAILURE DETECTED - Failover may be required' if primary_status == 'failed' else ''}
    {'✅ Both instances healthy' if primary_status == 'healthy' and backup_status == 'healthy' else ''}
    """
    
    return {
        "messages": [AIMessage(content=f"🏥 Health Monitor:\n{response.content}\n{health_report}")],
        "primary_status": primary_status,
        "backup_status": backup_status
    }


# Failover Controller
def failover_controller(state: FailoverState) -> FailoverState:
    """Controls failover decision and execution"""
    service_name = state.get("service_name", "")
    primary_status = state.get("primary_status", "healthy")
    backup_status = state.get("backup_status", "healthy")
    active_instance = state.get("active_instance", "primary")
    
    system_message = SystemMessage(content="""You are a failover controller. 
    Make intelligent failover decisions based on health status.""")
    
    user_message = HumanMessage(content=f"""Control failover:

Service: {service_name}
Primary Status: {primary_status}
Backup Status: {backup_status}
Currently Active: {active_instance}

Determine if failover is needed.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Failover decision logic
    failover_occurred = False
    failover_reason = ""
    new_active_instance = active_instance
    
    if active_instance == "primary" and primary_status == "failed":
        if backup_status == "healthy":
            failover_occurred = True
            failover_reason = "Primary instance failed, switching to backup"
            new_active_instance = "backup"
        else:
            failover_reason = "Primary failed but backup unavailable - SERVICE OUTAGE"
    elif active_instance == "primary" and primary_status == "degraded":
        if backup_status == "healthy":
            failover_occurred = True
            failover_reason = "Primary degraded, proactively switching to backup"
            new_active_instance = "backup"
    
    decision_icon = "🔄" if failover_occurred else "✅"
    
    failover_decision = f"""
    {decision_icon} Failover Decision:
    
    • Current Active: {active_instance}
    • Primary: {primary_status}
    • Backup: {backup_status}
    • Failover Needed: {'YES' if failover_occurred else 'NO'}
    • New Active: {new_active_instance}
    {f'• Reason: {failover_reason}' if failover_reason else ''}
    
    {'🔄 EXECUTING FAILOVER...' if failover_occurred else '✅ No action needed'}
    """
    
    return {
        "messages": [AIMessage(content=f"🎛️ Failover Controller:\n{response.content}\n{failover_decision}")],
        "active_instance": new_active_instance,
        "failover_occurred": failover_occurred,
        "failover_reason": failover_reason
    }


# Request Router
def request_router(state: FailoverState) -> FailoverState:
    """Routes requests to active instance (primary or backup)"""
    request = state.get("request", "")
    active_instance = state.get("active_instance", "primary")
    primary_status = state.get("primary_status", "healthy")
    backup_status = state.get("backup_status", "healthy")
    
    system_message = SystemMessage(content="""You are a request router. 
    Route requests to the currently active instance.""")
    
    user_message = HumanMessage(content=f"""Route request:

Request: {request}
Active Instance: {active_instance}
Primary Status: {primary_status}
Backup Status: {backup_status}

Route to appropriate instance.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Route to active instance
    if active_instance == "primary" and primary_status in ["healthy", "degraded"]:
        result = f"SUCCESS: Request processed by PRIMARY instance"
        response_time = 0.15
    elif active_instance == "backup" and backup_status in ["healthy", "degraded"]:
        result = f"SUCCESS: Request processed by BACKUP instance (failover)"
        response_time = 0.18
    else:
        result = f"FAILED: No healthy instance available"
        response_time = 0.0
    
    routing_info = f"""
    Request Routing:
    
    • Request: {request}
    • Routed To: {active_instance.upper()} instance
    • Result: {result}
    • Response Time: {response_time*1000:.0f}ms
    
    {'✅ Request successful' if 'SUCCESS' in result else '❌ Request failed'}
    """
    
    return {
        "messages": [AIMessage(content=f"🔀 Request Router:\n{response.content}\n{routing_info}")],
        "result": result,
        "response_time": response_time
    }


# Failback Manager
def failback_manager(state: FailoverState) -> FailoverState:
    """Manages failback to primary when it recovers"""
    primary_status = state.get("primary_status", "healthy")
    active_instance = state.get("active_instance", "primary")
    failover_occurred = state.get("failover_occurred", False)
    
    system_message = SystemMessage(content="""You are a failback manager. 
    Monitor primary instance and manage failback when it recovers.""")
    
    user_message = HumanMessage(content=f"""Manage failback:

Primary Status: {primary_status}
Active Instance: {active_instance}
Failover Occurred: {failover_occurred}

Determine if failback to primary is appropriate.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Failback logic
    failback_ready = False
    
    if active_instance == "backup" and primary_status == "healthy":
        failback_ready = True
        failback_status = "✅ READY - Primary recovered, failback possible"
    elif active_instance == "backup" and primary_status == "degraded":
        failback_ready = False
        failback_status = "⏳ WAIT - Primary degraded, waiting for full recovery"
    elif active_instance == "backup" and primary_status == "failed":
        failback_ready = False
        failback_status = "❌ NOT READY - Primary still failed"
    else:
        failback_status = "✅ Already on primary - no failback needed"
    
    failback_info = f"""
    Failback Assessment:
    
    • Active Instance: {active_instance}
    • Primary Status: {primary_status}
    • Failback Ready: {'Yes' if failback_ready else 'No'}
    • Status: {failback_status}
    
    Failback Strategy:
    • Monitor primary health continuously
    • Wait for stable recovery (not just momentary)
    • Perform failback during low-traffic window
    • Gradual traffic shift (optional)
    • Keep backup ready for quick re-failover
    """
    
    return {
        "messages": [AIMessage(content=f"⏮️ Failback Manager:\n{response.content}\n{failback_info}")],
        "failback_ready": failback_ready
    }


# Failover Monitor
def failover_monitor(state: FailoverState) -> FailoverState:
    """Monitors overall failover status and provides insights"""
    service_name = state.get("service_name", "")
    primary_status = state.get("primary_status", "healthy")
    backup_status = state.get("backup_status", "healthy")
    active_instance = state.get("active_instance", "primary")
    failover_occurred = state.get("failover_occurred", False)
    failover_reason = state.get("failover_reason", "")
    failback_ready = state.get("failback_ready", False)
    request = state.get("request", "")
    result = state.get("result", "")
    response_time = state.get("response_time", 0.0)
    
    status_icon = "🟢" if "SUCCESS" in result else "🔴"
    
    summary = f"""
    {status_icon} FAILOVER PATTERN COMPLETE
    
    Service: {service_name}
    Request: {request}
    Result: {result}
    
    Instance Status:
    • Primary: {primary_status.upper()} {'✅' if primary_status == 'healthy' else '🟡' if primary_status == 'degraded' else '❌'}
    • Backup: {backup_status.upper()} {'✅' if backup_status == 'healthy' else '🟡' if backup_status == 'degraded' else '❌'}
    • Active: {active_instance.upper()}
    
    Failover Information:
    • Failover Occurred: {'Yes 🔄' if failover_occurred else 'No ✅'}
    {f'• Reason: {failover_reason}' if failover_reason else ''}
    • Failback Ready: {'Yes ✅' if failback_ready else 'No ⏳'}
    
    Performance:
    • Response Time: {response_time*1000:.0f}ms
    • Service Continuity: {'✅ Maintained' if 'SUCCESS' in result else '❌ Interrupted'}
    
    Failover Pattern Process:
    1. Health Monitor → Check primary and backup health
    2. Failover Controller → Decide if failover needed
    3. Request Router → Route to active instance
    4. Failback Manager → Monitor for primary recovery
    5. Execute Failback → Return to primary when ready
    
    Failover Triggers:
    • Primary instance failure (complete outage)
    • Primary degradation (performance issues)
    • Scheduled maintenance
    • Geographic/network issues
    • Resource exhaustion
    
    Failover Types:
    • Automatic: System detects and switches automatically
    • Manual: Operator initiates failover
    • Graceful: Planned, controlled switchover
    • Emergency: Immediate switch due to critical failure
    
    Failover Pattern Benefits:
    • High availability (minimal downtime)
    • Automatic recovery
    • Service continuity
    • No single point of failure
    • Transparent to users
    • Disaster recovery
    
    Implementation Strategies:
    
    Active-Passive:
    • Primary: Handles all traffic
    • Backup: Standby, ready to take over
    • Pro: Simple, cost-effective
    • Con: Backup resources idle
    
    Active-Active:
    • Both: Handle traffic simultaneously
    • Pro: Better resource utilization
    • Con: More complex, data sync challenges
    
    Best Practices:
    • Regular health checks (every few seconds)
    • Automated failover (no human intervention)
    • Data replication to backup
    • Test failover regularly
    • Monitor failover metrics
    • Gradual failback (not immediate)
    • Geographic diversity
    • Documentation and runbooks
    
    Failover Metrics to Monitor:
    • Detection time (time to detect failure)
    • Failover time (time to switch)
    • Recovery time (total downtime)
    • Failover frequency
    • Failback success rate
    • Data consistency
    
    Common Pitfalls:
    ❌ Split-brain: Both instances think they're primary
    ❌ Data loss: Incomplete replication
    ❌ False positives: Unnecessary failovers
    ❌ Slow detection: Long downtime
    ❌ No testing: Failover fails when needed
    
    Key Insight:
    Failover pattern ensures continuous service availability by automatically
    switching to backup resources when primary fails. Essential for high-
    availability systems and disaster recovery.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Failover Monitor:\n{summary}")]
    }


# Build the graph
def build_failover_graph():
    """Build the failover pattern graph"""
    workflow = StateGraph(FailoverState)
    
    workflow.add_node("health", health_monitor)
    workflow.add_node("controller", failover_controller)
    workflow.add_node("router", request_router)
    workflow.add_node("failback", failback_manager)
    workflow.add_node("monitor", failover_monitor)
    
    workflow.add_edge(START, "health")
    workflow.add_edge("health", "controller")
    workflow.add_edge("controller", "router")
    workflow.add_edge("router", "failback")
    workflow.add_edge("failback", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_failover_graph()
    
    print("=== Failover MCP Pattern ===\n")
    
    # Test Case 1: Normal operation (primary healthy)
    print("\n" + "="*70)
    print("TEST CASE 1: Primary Healthy - No Failover")
    print("="*70)
    
    state1 = {
        "messages": [],
        "service_name": "OrderProcessingService",
        "primary_status": "healthy",
        "backup_status": "healthy",
        "active_instance": "primary",
        "failover_occurred": False,
        "failover_reason": "",
        "failback_ready": False,
        "request": "Process order #12345",
        "result": "",
        "response_time": 0.0
    }
    
    # Modify state to show primary failed
    state1["primary_status"] = "failed"
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("FAILOVER DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"\nFailover Occurred: {result1.get('failover_occurred', False)}")
    print(f"Active Instance: {result1.get('active_instance', 'N/A')}")
    print(f"Service Continuity: {'✅ Maintained' if 'SUCCESS' in result1.get('result', '') else '❌ Interrupted'}")
