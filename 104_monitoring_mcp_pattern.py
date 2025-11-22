"""
Monitoring MCP Pattern

This pattern implements comprehensive system monitoring with real-time
health checks, anomaly detection, and alerting.

Key Features:
- Multi-target health monitoring
- Real-time metric collection
- Anomaly detection
- Service availability tracking
- Resource utilization monitoring
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class MonitoringState(TypedDict):
    """State for monitoring pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    targets: List[Dict]  # [{name, type, endpoint, status, latency_ms, metrics}]
    health_checks: List[Dict]  # [{target, timestamp, status, response_time_ms}]
    anomalies: List[Dict]  # [{target, metric, current_value, expected_range, severity}]
    uptime_percentage: float
    alerts_triggered: List[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Health Monitor
def health_monitor(state: MonitoringState) -> MonitoringState:
    """Performs health checks on monitored targets"""
    targets = state.get("targets", [])
    
    system_message = SystemMessage(content="""You are a health monitoring system.
    Check the status and availability of monitored services.""")
    
    user_message = HumanMessage(content=f"""Monitor system health:

Targets: {len(targets) if targets else 'None defined'}

Perform health checks.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Define monitoring targets if not provided
    if not targets:
        targets = [
            {
                "name": "api-gateway",
                "type": "service",
                "endpoint": "http://api-gateway:8080/health",
                "expected_response_time_ms": 50,
                "critical": True
            },
            {
                "name": "database",
                "type": "database",
                "endpoint": "postgresql://db:5432",
                "expected_response_time_ms": 100,
                "critical": True
            },
            {
                "name": "cache",
                "type": "cache",
                "endpoint": "redis://cache:6379",
                "expected_response_time_ms": 10,
                "critical": False
            },
            {
                "name": "worker-service",
                "type": "service",
                "endpoint": "http://worker:8081/health",
                "expected_response_time_ms": 100,
                "critical": False
            }
        ]
    
    # Simulate health checks
    import random
    health_checks = []
    current_time = int(time.time())
    
    for target in targets:
        status = "healthy" if random.random() > 0.1 else "unhealthy"
        latency = target["expected_response_time_ms"] * (1 + random.uniform(-0.2, 0.4))
        
        health_checks.append({
            "target": target["name"],
            "timestamp": current_time,
            "status": status,
            "response_time_ms": round(latency, 2),
            "checks_performed": ["connectivity", "response_time", "status_code"]
        })
        
        # Update target status
        target["status"] = status
        target["latency_ms"] = round(latency, 2)
        target["last_check"] = current_time
    
    healthy_count = sum(1 for hc in health_checks if hc["status"] == "healthy")
    uptime_percentage = (healthy_count / len(health_checks)) * 100 if health_checks else 0.0
    
    report = f"""
    🏥 Health Monitoring:
    
    Health Check Results:
    • Total Targets: {len(targets)}
    • Healthy: {healthy_count}
    • Unhealthy: {len(health_checks) - healthy_count}
    • Uptime: {uptime_percentage:.2f}%
    
    Monitoring Concepts:
    
    Health Checks:
    
    Active Checks:
    • Periodic probes
    • HTTP /health endpoint
    • TCP connection test
    • Database ping
    • Custom checks
    
    Passive Checks:
    • Heartbeat monitoring
    • Activity tracking
    • Event-based
    • Watchdog timers
    
    Health Check Types:
    
    Liveness:
    • Is process running?
    • Can it accept requests?
    • Kubernetes liveness probe
    • Restart on failure
    
    Readiness:
    • Ready to serve traffic?
    • Dependencies available?
    • Kubernetes readiness probe
    • Remove from load balancer
    
    Startup:
    • Initial startup complete?
    • Longer grace period
    • Kubernetes startup probe
    • Prevent premature restarts
    
    Deep Health Check:
    • Check dependencies
    • Database connectivity
    • Cache availability
    • External APIs
    • Disk space
    • Memory available
    
    Monitoring Targets:
    
    Services:
    • HTTP endpoints
    • gRPC services
    • WebSocket connections
    • Message queues
    
    Infrastructure:
    • CPU usage
    • Memory usage
    • Disk I/O
    • Network bandwidth
    
    Databases:
    • Connection pool
    • Query latency
    • Replication lag
    • Lock contention
    
    External Dependencies:
    • Third-party APIs
    • CDN availability
    • DNS resolution
    • SSL certificates
    
    Health Check Implementation (Python):
    ```python
    from flask import Flask, jsonify
    import psycopg2
    import redis
    
    app = Flask(__name__)
    
    @app.route('/health')
    def health_check():
        checks = {{
            "status": "healthy",
            "checks": {{}}
        }}
        
        # Database check
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.close()
            checks["checks"]["database"] = "ok"
        except Exception as e:
            checks["status"] = "unhealthy"
            checks["checks"]["database"] = f"error: {{str(e)}}"
        
        # Cache check
        try:
            r = redis.Redis.from_url(REDIS_URL)
            r.ping()
            checks["checks"]["cache"] = "ok"
        except Exception as e:
            checks["checks"]["cache"] = f"error: {{str(e)}}"
        
        status_code = 200 if checks["status"] == "healthy" else 503
        return jsonify(checks), status_code
    ```
    
    Kubernetes Health Checks:
    ```yaml
    apiVersion: v1
    kind: Pod
    spec:
      containers:
      - name: app
        image: myapp:latest
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8080
          failureThreshold: 30
          periodSeconds: 10
    ```
    
    Monitoring Tools:
    
    Prometheus:
    • Time-series metrics
    • Pull-based collection
    • Alerting rules
    • Service discovery
    
    Nagios:
    • Classic monitoring
    • Plugin ecosystem
    • Host and service checks
    • Notification system
    
    Datadog:
    • Cloud-native
    • APM integration
    • Custom metrics
    • Real-time dashboards
    
    New Relic:
    • Full-stack monitoring
    • Synthetic monitoring
    • Mobile monitoring
    • Browser monitoring
    """
    
    return {
        "messages": [AIMessage(content=f"🏥 Health Monitor:\n{response.content}\n{report}")],
        "targets": targets,
        "health_checks": health_checks,
        "uptime_percentage": uptime_percentage
    }


# Anomaly Detector
def anomaly_detector(state: MonitoringState) -> MonitoringState:
    """Detects anomalies in monitored metrics"""
    health_checks = state.get("health_checks", [])
    targets = state.get("targets", [])
    
    system_message = SystemMessage(content="""You are an anomaly detection system.
    Identify unusual patterns and deviations from normal behavior.""")
    
    user_message = HumanMessage(content=f"""Analyze health data:

Health Checks: {len(health_checks)}
Targets: {len(targets)}

Detect anomalies.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Detect anomalies
    anomalies = []
    alerts_triggered = []
    
    for i, target in enumerate(targets):
        if target.get("status") == "unhealthy":
            anomalies.append({
                "target": target["name"],
                "metric": "health_status",
                "current_value": "unhealthy",
                "expected_value": "healthy",
                "severity": "critical" if target.get("critical") else "warning"
            })
            
            if target.get("critical"):
                alerts_triggered.append(f"CRITICAL: {target['name']} is unhealthy")
        
        # Check response time
        if target.get("latency_ms", 0) > target.get("expected_response_time_ms", 0) * 1.5:
            anomalies.append({
                "target": target["name"],
                "metric": "response_time",
                "current_value": f"{target['latency_ms']:.2f}ms",
                "expected_range": f"<{target['expected_response_time_ms'] * 1.5:.2f}ms",
                "severity": "warning"
            })
            alerts_triggered.append(f"WARNING: {target['name']} high latency ({target['latency_ms']:.2f}ms)")
    
    summary = f"""
    📊 MONITORING COMPLETE
    
    Monitoring Summary:
    • Targets Monitored: {len(targets)}
    • Health Checks: {len(health_checks)}
    • Uptime: {state.get('uptime_percentage', 0.0):.2f}%
    • Anomalies Detected: {len(anomalies)}
    • Alerts Triggered: {len(alerts_triggered)}
    
    {chr(10).join(f"• {alert}" for alert in alerts_triggered) if alerts_triggered else "• No alerts triggered"}
    
    Monitoring Pattern Process:
    1. Health Monitor → Perform health checks
    2. Anomaly Detector → Identify issues
    
    Anomaly Detection Methods:
    
    Statistical:
    • Standard deviation
    • Moving averages
    • Percentile thresholds
    • Z-score analysis
    
    Machine Learning:
    • Isolation Forest
    • LSTM autoencoders
    • One-class SVM
    • Clustering (DBSCAN)
    
    Rule-Based:
    • Threshold alerts
    • Rate of change
    • Correlation rules
    • Pattern matching
    
    Time-Series:
    • Seasonal decomposition
    • ARIMA forecasting
    • Prophet (Facebook)
    • Holt-Winters
    
    SLA Monitoring:
    
    Availability SLA:
    • 99.9% uptime = 43 min/month downtime
    • 99.99% = 4.3 min/month
    • 99.999% = 26 sec/month
    • Track error budget
    
    Performance SLA:
    • P95 latency < 200ms
    • P99 latency < 500ms
    • Throughput > 1000 req/s
    • Error rate < 0.1%
    
    Best Practices:
    
    Check Frequency:
    • Critical: 10-30 seconds
    • Important: 1-5 minutes
    • Normal: 5-15 minutes
    • Low priority: 15-60 minutes
    
    Alert Fatigue Prevention:
    • Meaningful thresholds
    • Alert aggregation
    • Smart deduplication
    • Escalation policies
    
    Monitoring as Code:
    • Version controlled
    • Infrastructure as Code
    • Terraform/Pulumi
    • GitOps workflow
    
    Key Insight:
    Proactive monitoring with anomaly detection enables
    early issue identification and prevents outages
    before they impact users.
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Anomaly Detector:\n{response.content}\n{summary}")],
        "anomalies": anomalies,
        "alerts_triggered": alerts_triggered
    }


# Build the graph
def build_monitoring_graph():
    """Build the monitoring pattern graph"""
    workflow = StateGraph(MonitoringState)
    
    workflow.add_node("health_monitor", health_monitor)
    workflow.add_node("anomaly_detector", anomaly_detector)
    
    workflow.add_edge(START, "health_monitor")
    workflow.add_edge("health_monitor", "anomaly_detector")
    workflow.add_edge("anomaly_detector", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_monitoring_graph()
    
    print("=== Monitoring MCP Pattern ===\n")
    
    # Test Case: Multi-target health monitoring
    print("\n" + "="*70)
    print("TEST CASE: System Health Monitoring")
    print("="*70)
    
    state = {
        "messages": [],
        "targets": [],
        "health_checks": [],
        "anomalies": [],
        "uptime_percentage": 0.0,
        "alerts_triggered": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nMonitoring Results:")
    print(f"Targets: {len(result.get('targets', []))}")
    print(f"Uptime: {result.get('uptime_percentage', 0.0):.2f}%")
    print(f"Anomalies: {len(result.get('anomalies', []))}")
    print(f"Alerts: {len(result.get('alerts_triggered', []))}")
