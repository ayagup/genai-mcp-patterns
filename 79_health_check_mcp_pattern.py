"""
Health Check MCP Pattern

This pattern continuously monitors system health and component status,
providing early detection of issues and supporting proactive responses.

Key Features:
- Continuous health monitoring
- Multi-component checks
- Health score calculation
- Issue detection
- Alerting and reporting
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class HealthCheckState(TypedDict):
    """State for health check pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    system_name: str
    components: dict[str, dict[str, any]]  # component -> {status, latency, error_rate, last_check}
    overall_health: str  # "healthy", "degraded", "critical", "down"
    health_score: float  # 0-100
    issues: list[str]
    alerts: list[str]
    check_timestamp: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Component Health Checker
def component_health_checker(state: HealthCheckState) -> HealthCheckState:
    """Checks health of individual components"""
    system_name = state.get("system_name", "")
    
    system_message = SystemMessage(content="""You are a component health checker. 
    Monitor individual system components and report their health status.""")
    
    user_message = HumanMessage(content=f"""Check component health:

System: {system_name}

Perform health checks on all system components.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate health checks for various components
    components = {
        "database": {
            "status": "healthy",
            "latency": 15.0,  # ms
            "error_rate": 0.1,  # %
            "last_check": "2024-01-01 10:00:00",
            "cpu_usage": 45.0,  # %
            "memory_usage": 60.0  # %
        },
        "api_gateway": {
            "status": "healthy",
            "latency": 25.0,
            "error_rate": 0.5,
            "last_check": "2024-01-01 10:00:00",
            "request_rate": 1500  # req/min
        },
        "cache": {
            "status": "degraded",
            "latency": 5.0,
            "error_rate": 2.5,  # Slightly elevated
            "last_check": "2024-01-01 10:00:00",
            "hit_rate": 75.0  # %
        },
        "message_queue": {
            "status": "healthy",
            "latency": 10.0,
            "error_rate": 0.0,
            "last_check": "2024-01-01 10:00:00",
            "queue_depth": 150
        },
        "search_service": {
            "status": "critical",
            "latency": 450.0,  # Very slow
            "error_rate": 8.5,  # High error rate
            "last_check": "2024-01-01 10:00:00",
            "index_size": 1000000
        }
    }
    
    component_summary = []
    for name, health in components.items():
        status_icon = {
            "healthy": "🟢",
            "degraded": "🟡",
            "critical": "🔴",
            "down": "⚫"
        }.get(health["status"], "⚪")
        
        component_summary.append(f"""
    {status_icon} {name.upper()}: {health['status'].upper()}
       Latency: {health['latency']}ms | Error Rate: {health['error_rate']}%""")
    
    components_report = "".join(component_summary)
    
    health_report = f"""
    Component Health Check Results:
{components_report}
    
    Total Components Checked: {len(components)}
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Component Health Checker:\n{response.content}\n{health_report}")],
        "components": components,
        "check_timestamp": "2024-01-01 10:00:00"
    }


# Health Analyzer
def health_analyzer(state: HealthCheckState) -> HealthCheckState:
    """Analyzes component health to determine overall system health"""
    components = state.get("components", {})
    system_name = state.get("system_name", "")
    
    system_message = SystemMessage(content="""You are a health analyzer. 
    Analyze component health data to determine overall system health.""")
    
    component_statuses = {name: comp["status"] for name, comp in components.items()}
    
    user_message = HumanMessage(content=f"""Analyze system health:

System: {system_name}
Component Statuses: {component_statuses}

Determine overall system health and calculate health score.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate health score
    status_scores = {
        "healthy": 100,
        "degraded": 70,
        "critical": 30,
        "down": 0
    }
    
    component_scores = [status_scores.get(comp["status"], 0) for comp in components.values()]
    health_score = sum(component_scores) / len(component_scores) if component_scores else 0
    
    # Determine overall health
    healthy_count = sum(1 for comp in components.values() if comp["status"] == "healthy")
    degraded_count = sum(1 for comp in components.values() if comp["status"] == "degraded")
    critical_count = sum(1 for comp in components.values() if comp["status"] == "critical")
    down_count = sum(1 for comp in components.values() if comp["status"] == "down")
    
    if down_count > 0 or critical_count >= len(components) * 0.5:
        overall_health = "critical"
    elif critical_count > 0 or degraded_count >= len(components) * 0.5:
        overall_health = "degraded"
    elif healthy_count == len(components):
        overall_health = "healthy"
    else:
        overall_health = "degraded"
    
    health_icon = {
        "healthy": "🟢",
        "degraded": "🟡",
        "critical": "🔴",
        "down": "⚫"
    }.get(overall_health, "⚪")
    
    analysis = f"""
    {health_icon} Overall System Health: {overall_health.upper()}
    
    Health Score: {health_score:.1f}/100
    
    Component Breakdown:
    • Healthy: {healthy_count} ({healthy_count/len(components)*100:.0f}%)
    • Degraded: {degraded_count} ({degraded_count/len(components)*100:.0f}%)
    • Critical: {critical_count} ({critical_count/len(components)*100:.0f}%)
    • Down: {down_count} ({down_count/len(components)*100:.0f}%)
    
    Total Components: {len(components)}
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Health Analyzer:\n{response.content}\n{analysis}")],
        "overall_health": overall_health,
        "health_score": health_score
    }


# Issue Detector
def issue_detector(state: HealthCheckState) -> HealthCheckState:
    """Detects specific issues from health check data"""
    components = state.get("components", {})
    overall_health = state.get("overall_health", "healthy")
    
    system_message = SystemMessage(content="""You are an issue detector. 
    Identify specific problems from health check data.""")
    
    user_message = HumanMessage(content=f"""Detect issues:

Overall Health: {overall_health}
Components: {len(components)}

Identify specific issues requiring attention.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Detect issues based on metrics
    issues = []
    
    for name, comp in components.items():
        if comp["status"] == "down":
            issues.append(f"CRITICAL: {name} is DOWN")
        elif comp["status"] == "critical":
            issues.append(f"CRITICAL: {name} in critical state")
        elif comp["status"] == "degraded":
            issues.append(f"WARNING: {name} degraded performance")
        
        if comp.get("latency", 0) > 200:
            issues.append(f"HIGH LATENCY: {name} ({comp['latency']}ms)")
        
        if comp.get("error_rate", 0) > 5:
            issues.append(f"HIGH ERROR RATE: {name} ({comp['error_rate']}%)")
        
        if comp.get("cpu_usage", 0) > 80:
            issues.append(f"HIGH CPU: {name} ({comp['cpu_usage']}%)")
        
        if comp.get("memory_usage", 0) > 85:
            issues.append(f"HIGH MEMORY: {name} ({comp['memory_usage']}%)")
    
    issue_count = len(issues)
    issue_icon = "❌" if issue_count > 0 else "✅"
    
    issues_report = f"""
    {issue_icon} Issues Detected: {issue_count}
    
    {'Issues:' if issues else 'No issues detected'}
{chr(10).join(f'    • {issue}' for issue in issues) if issues else ''}
    """
    
    return {
        "messages": [AIMessage(content=f"🔎 Issue Detector:\n{response.content}\n{issues_report}")],
        "issues": issues
    }


# Alert Manager
def alert_manager(state: HealthCheckState) -> HealthCheckState:
    """Manages alerts based on detected issues"""
    issues = state.get("issues", [])
    overall_health = state.get("overall_health", "healthy")
    health_score = state.get("health_score", 100.0)
    
    system_message = SystemMessage(content="""You are an alert manager. 
    Generate appropriate alerts based on system health and issues.""")
    
    user_message = HumanMessage(content=f"""Manage alerts:

Overall Health: {overall_health}
Health Score: {health_score}
Issues: {len(issues)}

Generate appropriate alerts.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate alerts based on severity
    alerts = []
    
    if overall_health == "critical":
        alerts.append("🚨 CRITICAL: System in critical state - immediate action required")
    elif overall_health == "degraded":
        alerts.append("⚠️ WARNING: System degraded - investigation recommended")
    
    if health_score < 50:
        alerts.append(f"🚨 CRITICAL: Health score critically low ({health_score:.0f}/100)")
    elif health_score < 75:
        alerts.append(f"⚠️ WARNING: Health score below threshold ({health_score:.0f}/100)")
    
    # Alert for critical issues
    critical_issues = [i for i in issues if "CRITICAL" in i]
    if critical_issues:
        alerts.append(f"🚨 CRITICAL: {len(critical_issues)} critical issue(s) detected")
    
    # Alert for high error rates
    error_issues = [i for i in issues if "ERROR RATE" in i]
    if error_issues:
        alerts.append(f"⚠️ WARNING: High error rate detected in {len(error_issues)} component(s)")
    
    alert_summary = f"""
    Alert Status: {len(alerts)} alert(s) generated
    
    {'Alerts:' if alerts else '✅ No alerts - system healthy'}
{chr(10).join(f'    {alert}' for alert in alerts) if alerts else ''}
    
    Alert Channels:
    • Email: ops-team@company.com
    • Slack: #system-alerts
    • PagerDuty: On-call engineer
    • Dashboard: Real-time monitoring
    """
    
    return {
        "messages": [AIMessage(content=f"🚨 Alert Manager:\n{response.content}\n{alert_summary}")],
        "alerts": alerts
    }


# Health Reporter
def health_reporter(state: HealthCheckState) -> HealthCheckState:
    """Generates comprehensive health report"""
    system_name = state.get("system_name", "")
    components = state.get("components", {})
    overall_health = state.get("overall_health", "healthy")
    health_score = state.get("health_score", 100.0)
    issues = state.get("issues", [])
    alerts = state.get("alerts", [])
    check_timestamp = state.get("check_timestamp", "")
    
    health_icon = {
        "healthy": "🟢",
        "degraded": "🟡",
        "critical": "🔴",
        "down": "⚫"
    }.get(overall_health, "⚪")
    
    summary = f"""
    {health_icon} HEALTH CHECK COMPLETE
    
    System: {system_name}
    Timestamp: {check_timestamp}
    Overall Health: {overall_health.upper()}
    Health Score: {health_score:.1f}/100
    
    Component Status ({len(components)} total):
{chr(10).join(f'    {{"healthy": "🟢", "degraded": "🟡", "critical": "🔴", "down": "⚫"}}.get(comp["status"], "⚪") {name}: {comp["status"].upper()} | {comp["latency"]}ms | {comp["error_rate"]}% errors' for name, comp in components.items())}
    
    Issues Detected: {len(issues)}
    Alerts Generated: {len(alerts)}
    
    Health Check Pattern Process:
    1. Component Checker → Check each component individually
    2. Health Analyzer → Calculate overall health score
    3. Issue Detector → Identify specific problems
    4. Alert Manager → Generate alerts for issues
    5. Health Reporter → Compile comprehensive report
    
    Health Check Types:
    
    Liveness Check:
    • Is service running?
    • Can it accept requests?
    • Basic: HTTP 200 response
    
    Readiness Check:
    • Is service ready to handle traffic?
    • Dependencies available?
    • Example: DB connection, cache ready
    
    Deep Health Check:
    • Comprehensive component checks
    • Performance metrics
    • Resource utilization
    • Dependency health
    
    Health Check Metrics:
    • Response time/latency
    • Error rates
    • Throughput
    • Resource usage (CPU, memory)
    • Connection pool status
    • Queue depths
    • Cache hit rates
    
    Health Check Benefits:
    • Early issue detection
    • Proactive response
    • System visibility
    • Automated recovery triggers
    • Load balancer integration
    • Service discovery support
    
    Implementation Best Practices:
    • Frequent checks (every 5-30 seconds)
    • Lightweight checks (< 100ms)
    • Independent of business logic
    • Don't impact performance
    • Multiple check levels
    • Clear pass/fail criteria
    • Versioned health endpoints
    • Secure endpoints
    
    Health Check Patterns:
    • Shallow: Quick basic checks
    • Deep: Comprehensive validation
    • Dependency: Check external services
    • Self: Internal component checks
    • Passive: Monitor existing traffic
    • Active: Send test requests
    
    Health Check States:
    • Healthy (🟢): All systems normal
    • Degraded (🟡): Partial issues, still functional
    • Critical (🔴): Major issues, limited functionality
    • Down (⚫): Service unavailable
    
    Integration Points:
    • Load Balancers: Remove unhealthy instances
    • Service Mesh: Circuit breaker activation
    • Orchestrators: Container restart
    • Monitoring: Alert triggers
    • Auto-scaling: Capacity decisions
    
    Common Health Check Endpoints:
    • /health - Basic liveness
    • /health/ready - Readiness check
    • /health/live - Liveness check
    • /health/deep - Comprehensive check
    • /metrics - Prometheus metrics
    
    Response Format:
    {{
      "status": "healthy|degraded|critical|down",
      "timestamp": "ISO-8601",
      "components": {{
        "database": {{"status": "healthy", "latency": 15}},
        ...
      }},
      "version": "1.0.0"
    }}
    
    Key Insight:
    Health checks enable proactive system monitoring and early issue detection,
    supporting automated recovery, load balancing, and maintaining high
    availability through continuous health assessment.
    """
    
    return {
        "messages": [AIMessage(content=f"📋 Health Reporter:\n{summary}")]
    }


# Build the graph
def build_health_check_graph():
    """Build the health check pattern graph"""
    workflow = StateGraph(HealthCheckState)
    
    workflow.add_node("checker", component_health_checker)
    workflow.add_node("analyzer", health_analyzer)
    workflow.add_node("detector", issue_detector)
    workflow.add_node("alerts", alert_manager)
    workflow.add_node("reporter", health_reporter)
    
    workflow.add_edge(START, "checker")
    workflow.add_edge("checker", "analyzer")
    workflow.add_edge("analyzer", "detector")
    workflow.add_edge("detector", "alerts")
    workflow.add_edge("alerts", "reporter")
    workflow.add_edge("reporter", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_health_check_graph()
    
    print("=== Health Check MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "system_name": "E-Commerce Platform",
        "components": {},
        "overall_health": "",
        "health_score": 0.0,
        "issues": [],
        "alerts": [],
        "check_timestamp": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("HEALTH CHECK COMPLETE")
    print("="*70)
    print(f"\nSystem: {state['system_name']}")
    print(f"Overall Health: {result.get('overall_health', 'N/A').upper()}")
    print(f"Health Score: {result.get('health_score', 0):.1f}/100")
    print(f"Issues: {len(result.get('issues', []))}")
    print(f"Alerts: {len(result.get('alerts', []))}")
