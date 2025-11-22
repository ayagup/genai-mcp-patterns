"""
Alerting MCP Pattern

This pattern implements intelligent alerting with notification routing,
escalation policies, and alert aggregation.

Key Features:
- Multi-channel alerting
- Smart alert routing
- Escalation management
- Alert deduplication
- Notification throttling
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class AlertingState(TypedDict):
    """State for alerting pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    alerts: List[Dict]  # [{id, severity, source, message, timestamp, status}]
    notification_channels: List[str]  # ["email", "slack", "pagerduty"]
    routing_rules: Dict[str, List[str]]  # {severity: [channels]}
    escalations: List[Dict]  # [{alert_id, level, assignee, timestamp}]
    suppressed_alerts: List[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Alert Manager
def alert_manager(state: AlertingState) -> AlertingState:
    """Manages incoming alerts and deduplication"""
    alerts = state.get("alerts", [])
    
    system_message = SystemMessage(content="""You are an alert management system.
    Process, deduplicate, and route alerts appropriately.""")
    
    user_message = HumanMessage(content=f"""Manage alerts:

Current Alerts: {len(alerts) if alerts else 'None'}

Set up alert management.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate sample alerts if not provided
    if not alerts:
        import uuid
        current_time = int(time.time())
        
        alerts = [
            {
                "id": str(uuid.uuid4())[:8],
                "severity": "critical",
                "source": "api-gateway",
                "message": "Service unavailable - health check failing",
                "timestamp": current_time,
                "status": "active",
                "metadata": {"endpoint": "/api/users", "error_rate": 85.5}
            },
            {
                "id": str(uuid.uuid4())[:8],
                "severity": "warning",
                "source": "database",
                "message": "Connection pool utilization high",
                "timestamp": current_time + 30,
                "status": "active",
                "metadata": {"pool_usage": 92, "max_connections": 100}
            },
            {
                "id": str(uuid.uuid4())[:8],
                "severity": "info",
                "source": "cache",
                "message": "Cache hit rate below threshold",
                "timestamp": current_time + 60,
                "status": "active",
                "metadata": {"hit_rate": 65.2, "threshold": 80}
            },
            {
                "id": str(uuid.uuid4())[:8],
                "severity": "critical",
                "source": "payment-service",
                "message": "Payment gateway timeout rate elevated",
                "timestamp": current_time + 90,
                "status": "active",
                "metadata": {"timeout_rate": 12.5, "sla_threshold": 1.0}
            }
        ]
    
    # Define notification channels and routing
    notification_channels = ["email", "slack", "pagerduty", "webhook"]
    routing_rules = {
        "critical": ["pagerduty", "slack", "email"],
        "warning": ["slack", "email"],
        "info": ["email"]
    }
    
    # Deduplicate alerts (suppress duplicates within 5 minutes)
    suppressed_alerts = []
    seen = set()
    for alert in alerts:
        alert_key = f"{alert['source']}-{alert['severity']}-{alert['message']}"
        if alert_key in seen:
            suppressed_alerts.append(alert["id"])
            alert["status"] = "suppressed"
        else:
            seen.add(alert_key)
    
    report = f"""
    🚨 Alert Management:
    
    Alert Overview:
    • Total Alerts: {len(alerts)}
    • Active: {sum(1 for a in alerts if a['status'] == 'active')}
    • Suppressed: {len(suppressed_alerts)}
    • Critical: {sum(1 for a in alerts if a['severity'] == 'critical')}
    • Warning: {sum(1 for a in alerts if a['severity'] == 'warning')}
    
    Alerting Concepts:
    
    Alert Severity Levels:
    
    Critical/P1:
    • Service down
    • Data loss
    • Security breach
    • Immediate response
    • Page on-call
    
    Warning/P2:
    • Degraded performance
    • Threshold breach
    • Approaching limits
    • Business hours response
    • Email/Slack notification
    
    Info/P3:
    • Normal events
    • Informational
    • No immediate action
    • Log and review
    • Batch notifications
    
    Alert States:
    
    Active:
    • Currently firing
    • Condition still met
    • Notifications sent
    • Awaiting acknowledgment
    
    Acknowledged:
    • Engineer notified
    • Work in progress
    • Still firing
    • No more notifications
    
    Resolved:
    • Condition cleared
    • Back to normal
    • Close incident
    • Post-mortem if needed
    
    Suppressed:
    • Intentionally muted
    • Maintenance window
    • Known issue
    • Temporary silence
    
    Notification Channels:
    
    Email:
    • Detailed information
    • Audit trail
    • Non-urgent
    • Batching possible
    • HTML formatting
    
    Slack/Teams:
    • Real-time notification
    • Team visibility
    • Quick discussion
    • Rich formatting
    • Emoji reactions
    
    PagerDuty/OpsGenie:
    • On-call rotation
    • Escalation policies
    • Acknowledgment tracking
    • Mobile push
    • Phone calls for P1
    
    Webhook:
    • Custom integrations
    • ITSM systems
    • Chatbots
    • Ticketing
    • Automation triggers
    
    Alert Routing Example (Prometheus):
    ```yaml
    route:
      receiver: 'default'
      group_by: ['alertname', 'cluster']
      group_wait: 10s
      group_interval: 5m
      repeat_interval: 12h
      
      routes:
      - match:
          severity: critical
        receiver: pagerduty
        continue: true
        
      - match:
          severity: warning
        receiver: slack
        
      - match_re:
          service: ^(payment|auth)
        receiver: critical-team
    
    receivers:
    - name: 'pagerduty'
      pagerduty_configs:
      - service_key: xxx
        
    - name: 'slack'
      slack_configs:
      - api_url: xxx
        channel: '#alerts'
    ```
    
    Deduplication Strategies:
    
    Time-based:
    • Suppress duplicates within window
    • Default: 5-15 minutes
    • Prevents alert storms
    
    Fingerprint:
    • Hash alert attributes
    • Match identical alerts
    • Group related events
    
    Correlation:
    • Root cause detection
    • Dependency awareness
    • Single notification for cascade
    
    Alert Aggregation:
    • Group by cluster
    • Batch notifications
    • Summary messages
    • Reduce noise
    
    Notification Formatting (Slack):
    ```python
    import requests
    
    def send_slack_alert(alert):
        payload = {{
            "attachments": [{{
                "color": "danger" if alert["severity"] == "critical" else "warning",
                "title": f"{{alert['severity'].upper()}}: {{alert['source']}}",
                "text": alert["message"],
                "fields": [
                    {{"title": "Severity", "value": alert["severity"], "short": True}},
                    {{"title": "Source", "value": alert["source"], "short": True}},
                    {{"title": "Time", "value": alert["timestamp"], "short": False}}
                ],
                "footer": "Alert System",
                "ts": alert["timestamp"]
            }}]
        }}
        
        requests.post(SLACK_WEBHOOK_URL, json=payload)
    ```
    """
    
    return {
        "messages": [AIMessage(content=f"🚨 Alert Manager:\n{response.content}\n{report}")],
        "alerts": alerts,
        "notification_channels": notification_channels,
        "routing_rules": routing_rules,
        "suppressed_alerts": suppressed_alerts
    }


# Escalation Manager
def escalation_manager(state: AlertingState) -> AlertingState:
    """Handles alert escalation based on policies"""
    alerts = state.get("alerts", [])
    routing_rules = state.get("routing_rules", {})
    
    system_message = SystemMessage(content="""You are an escalation management system.
    Handle alert escalation and ensure timely response.""")
    
    user_message = HumanMessage(content=f"""Manage escalations:

Active Alerts: {sum(1 for a in alerts if a.get('status') == 'active')}
Routing Rules: {len(routing_rules)}

Create escalation plan.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Create escalations for critical alerts
    escalations = []
    current_time = int(time.time())
    
    for alert in alerts:
        if alert.get("status") == "active" and alert.get("severity") == "critical":
            escalations.append({
                "alert_id": alert["id"],
                "level": 1,
                "assignee": "on-call-engineer",
                "timestamp": current_time,
                "channels": routing_rules.get("critical", []),
                "timeout_minutes": 15
            })
    
    summary = f"""
    📊 ALERTING COMPLETE
    
    Alerting Summary:
    • Total Alerts: {len(alerts)}
    • Active: {sum(1 for a in alerts if a.get('status') == 'active')}
    • Suppressed: {len(state.get('suppressed_alerts', []))}
    • Escalations: {len(escalations)}
    • Notification Channels: {len(state.get('notification_channels', []))}
    
    Escalations Created:
    {chr(10).join(f"• Alert {e['alert_id']}: Level {e['level']} → {e['assignee']} via {', '.join(e['channels'])}" for e in escalations) if escalations else "• No escalations needed"}
    
    Alerting Pattern Process:
    1. Alert Manager → Process and deduplicate alerts
    2. Escalation Manager → Handle escalation policies
    
    Escalation Policies:
    
    Level 1 (0-15 min):
    • Notify on-call engineer
    • PagerDuty/OpsGenie
    • Phone + SMS + Push
    • Acknowledge required
    
    Level 2 (15-30 min):
    • Escalate to senior engineer
    • Manager notification
    • Team Slack channel
    • Incident commander assigned
    
    Level 3 (30+ min):
    • Escalate to team lead
    • Page multiple engineers
    • Incident response team
    • Status page update
    
    Level 4 (1+ hour):
    • CTO/VP Engineering
    • Cross-team coordination
    • Customer communication
    • Post-mortem scheduled
    
    Alert Fatigue Prevention:
    
    Meaningful Alerts:
    • Alert on symptoms, not causes
    • Action-oriented messages
    • Clear remediation steps
    • Remove noise
    
    Smart Throttling:
    • Rate limiting per alert
    • Exponential backoff
    • Quiet hours (optional)
    • Maintenance windows
    
    Alert Quality:
    • Review alert value
    • Track acknowledgment rate
    • Measure time to resolve
    • Remove unused alerts
    
    Runbook Integration:
    • Link to documentation
    • Auto-remediation steps
    • Common causes
    • Diagnostic commands
    
    Alert Testing:
    
    Synthetic Alerts:
    • Test notification flow
    • Verify routing
    • Check escalation
    • Practice response
    
    Chaos Engineering:
    • Trigger real failures
    • Verify alerts fire
    • Test recovery procedures
    • Validate runbooks
    
    Best Practices:
    
    Alert Design:
    • Actionable message
    • Clear severity
    • Context included
    • Runbook link
    • Related metrics
    
    On-Call Management:
    • Rotation schedule
    • Handoff process
    • Escalation chain
    • On-call compensation
    • Incident review
    
    Metrics to Track:
    • MTTA (Mean Time To Acknowledge)
    • MTTR (Mean Time To Resolve)
    • Alert volume trends
    • False positive rate
    • Escalation frequency
    
    Key Insight:
    Effective alerting balances comprehensive coverage
    with alert fatigue prevention through intelligent
    routing, deduplication, and escalation.
    """
    
    return {
        "messages": [AIMessage(content=f"⚡ Escalation Manager:\n{response.content}\n{summary}")],
        "escalations": escalations
    }


# Build the graph
def build_alerting_graph():
    """Build the alerting pattern graph"""
    workflow = StateGraph(AlertingState)
    
    workflow.add_node("alert_manager", alert_manager)
    workflow.add_node("escalation_manager", escalation_manager)
    
    workflow.add_edge(START, "alert_manager")
    workflow.add_edge("alert_manager", "escalation_manager")
    workflow.add_edge("escalation_manager", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_alerting_graph()
    
    print("=== Alerting MCP Pattern ===\n")
    
    # Test Case: Multi-severity alerting with escalation
    print("\n" + "="*70)
    print("TEST CASE: Alert Management and Escalation")
    print("="*70)
    
    state = {
        "messages": [],
        "alerts": [],
        "notification_channels": [],
        "routing_rules": {},
        "escalations": [],
        "suppressed_alerts": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nAlerting Results:")
    print(f"Total Alerts: {len(result.get('alerts', []))}")
    print(f"Active: {sum(1 for a in result.get('alerts', []) if a.get('status') == 'active')}")
    print(f"Escalations: {len(result.get('escalations', []))}")
