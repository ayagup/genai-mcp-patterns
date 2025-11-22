"""
Lifecycle Management MCP Pattern

This pattern manages the complete lifecycle of resources from creation
through initialization, usage, maintenance, and eventual disposal.

Key Features:
- Resource creation and initialization
- State transitions and validation
- Maintenance and health checks
- Graceful shutdown and cleanup
- Lifecycle event tracking
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class LifecycleState(TypedDict):
    """State for lifecycle management pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    resource_type: str
    lifecycle_stage: str  # "creation", "initialization", "active", "maintenance", "shutdown", "disposed"
    resources: Dict[str, Dict]  # resource_id -> {state, health, created_at, last_check}
    transition_history: List[Dict]  # {resource_id, from_state, to_state, timestamp}
    health_checks_passed: int
    health_checks_failed: int


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Lifecycle Manager
def lifecycle_manager(state: LifecycleState) -> LifecycleState:
    """Manages resource lifecycle stages"""
    resource_type = state.get("resource_type", "")
    
    system_message = SystemMessage(content="""You are a lifecycle manager.
    Manage the complete lifecycle of resources from creation to disposal.""")
    
    user_message = HumanMessage(content=f"""Manage resource lifecycle:

Resource Type: {resource_type}

Initialize lifecycle management.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Create sample resources in different lifecycle stages
    import time
    resources = {
        "resource_1": {"state": "active", "health": "healthy", "created_at": int(time.time()) - 3600, "last_check": int(time.time())},
        "resource_2": {"state": "initialization", "health": "unknown", "created_at": int(time.time()) - 60, "last_check": int(time.time())},
        "resource_3": {"state": "maintenance", "health": "degraded", "created_at": int(time.time()) - 7200, "last_check": int(time.time())}
    }
    
    transition_history = [
        {"resource_id": "resource_1", "from_state": "creation", "to_state": "initialization", "timestamp": int(time.time()) - 3600},
        {"resource_id": "resource_1", "from_state": "initialization", "to_state": "active", "timestamp": int(time.time()) - 3500},
    ]
    
    report = f"""
    🔄 Lifecycle Management:
    
    Resource Overview:
    • Type: {resource_type.upper()}
    • Total Resources: {len(resources)}
    • Active: {sum(1 for r in resources.values() if r['state'] == 'active')}
    • In Transition: {sum(1 for r in resources.values() if r['state'] in ['creation', 'initialization'])}
    • Maintenance: {sum(1 for r in resources.values() if r['state'] == 'maintenance')}
    
    Lifecycle Stages:
    
    1. Creation:
       • Allocate resources
       • Assign identifier
       • Set initial configuration
       • Register in inventory
       • Log creation event
    
    2. Initialization:
       • Load configuration
       • Establish connections
       • Warm up caches
       • Validate dependencies
       • Run startup checks
    
    3. Active (Running):
       • Handle requests
       • Monitor performance
       • Collect metrics
       • Respond to health checks
       • Process workload
    
    4. Maintenance:
       • Scheduled updates
       • Configuration changes
       • Performance tuning
       • Backup operations
       • Health recovery
    
    5. Shutdown (Graceful):
       • Stop accepting new work
       • Complete in-flight requests
       • Save state if needed
       • Close connections
       • Release resources
    
    6. Disposed:
       • Cleanup completed
       • Resources released
       • Logs archived
       • Metrics finalized
       • Deregistered
    
    State Transition Rules:
    
    Valid Transitions:
    • creation → initialization
    • initialization → active
    • active → maintenance
    • maintenance → active
    • active → shutdown
    • shutdown → disposed
    
    Invalid Transitions:
    • disposed → any (final state)
    • initialization → shutdown (must reach active first)
    • creation → active (must initialize)
    
    Lifecycle Patterns by Resource Type:
    
    Database Connection:
    ```python
    class DatabaseConnection:
        def __init__(self, config):
            self.state = 'creation'
            self.config = config
            self.connection = None
        
        def initialize(self):
            self.state = 'initialization'
            self.connection = connect(self.config)
            self.connection.ping()
            self.state = 'active'
        
        def use(self, query):
            if self.state != 'active':
                raise Exception("Not active")
            return self.connection.execute(query)
        
        def shutdown(self):
            self.state = 'shutdown'
            if self.connection:
                self.connection.close()
            self.state = 'disposed'
    ```
    
    Web Server:
    ```python
    class WebServer:
        def __init__(self, port):
            self.state = 'creation'
            self.port = port
            self.server = None
        
        def start(self):
            self.state = 'initialization'
            self.server = HTTPServer(('', self.port))
            self.server.bind()
            self.state = 'active'
            self.server.serve_forever()
        
        def shutdown(self):
            self.state = 'shutdown'
            # Graceful shutdown
            self.server.shutdown()
            self.server.close()
            self.state = 'disposed'
    ```
    
    Container (Docker):
    • Creating: Image pull, layer download
    • Starting: Container creation, entrypoint
    • Running: Application active
    • Pausing: Freeze processes
    • Stopping: SIGTERM, wait, SIGKILL
    • Removed: Cleanup filesystem
    
    Kubernetes Pod:
    • Pending: Scheduling, image pull
    • Running: All containers running
    • Succeeded: Completed successfully
    • Failed: Container error
    • Unknown: Node communication issue
    
    Lifecycle Hooks:
    
    Pre-Start:
    • Validate configuration
    • Check prerequisites
    • Prepare environment
    • Initialize logging
    
    Post-Start:
    • Health check passed
    • Ready to serve
    • Register with discovery
    • Emit ready event
    
    Pre-Stop:
    • Deregister from LB
    • Drain connections
    • Stop accepting requests
    • Notify dependents
    
    Post-Stop:
    • Cleanup temp files
    • Archive logs
    • Release locks
    • Update inventory
    """
    
    return {
        "messages": [AIMessage(content=f"🔄 Lifecycle Manager:\n{response.content}\n{report}")],
        "resources": resources,
        "transition_history": transition_history,
        "lifecycle_stage": "active"
    }


# Health Monitor
def health_monitor(state: LifecycleState) -> LifecycleState:
    """Monitors resource health throughout lifecycle"""
    resources = state.get("resources", {})
    
    system_message = SystemMessage(content="""You are a health monitor.
    Check resource health and trigger maintenance when needed.""")
    
    user_message = HumanMessage(content=f"""Monitor resource health:

Resources: {len(resources)}

Perform health checks.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate health checks
    health_checks_passed = 0
    health_checks_failed = 0
    
    for resource_id, resource_data in resources.items():
        # Simulate health check
        if resource_data["health"] in ["healthy", "unknown"]:
            health_checks_passed += 1
        else:
            health_checks_failed += 1
            # Trigger maintenance for degraded resources
            if resource_data["state"] == "active":
                resource_data["state"] = "maintenance"
    
    report = f"""
    🏥 Health Monitoring:
    
    Health Check Results:
    • Total Checks: {len(resources)}
    • Passed: {health_checks_passed}
    • Failed: {health_checks_failed}
    • Success Rate: {(health_checks_passed/len(resources)*100) if resources else 0:.1f}%
    
    Health Check Types:
    
    1. Liveness Probe:
       • Is process running?
       • Responds to ping?
       • Restart if failed
       • Example: HTTP /healthz
    
    2. Readiness Probe:
       • Can accept traffic?
       • Dependencies ready?
       • Remove from LB if not ready
       • Example: HTTP /ready
    
    3. Startup Probe:
       • Initial startup complete?
       • Slow-starting apps
       • Disable other probes until passed
       • Example: HTTP /started
    
    4. Performance Check:
       • Response time acceptable?
       • Resource usage normal?
       • Throughput adequate?
       • Quality of service
    
    Health Check Implementation:
    
    HTTP Endpoint:
    ```python
    @app.route('/health')
    def health_check():
        checks = {{
            'database': check_database(),
            'cache': check_cache(),
            'disk': check_disk_space(),
            'memory': check_memory()
        }}
        
        all_healthy = all(checks.values())
        status_code = 200 if all_healthy else 503
        
        return jsonify({{
            'status': 'healthy' if all_healthy else 'unhealthy',
            'checks': checks
        }}), status_code
    ```
    
    Kubernetes Probes:
    ```yaml
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
      failureThreshold: 3
    
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
      successThreshold: 1
    ```
    
    Health States:
    
    Healthy:
    • All checks pass
    • Normal performance
    • No errors
    • Ready for traffic
    
    Degraded:
    • Some checks fail
    • Reduced capacity
    • Increased latency
    • May continue serving
    
    Unhealthy:
    • Critical checks fail
    • Cannot serve traffic
    • Requires intervention
    • Remove from rotation
    
    Unknown:
    • Cannot determine health
    • Check timeout
    • Communication error
    • Assume unhealthy
    
    Recovery Actions:
    
    Automatic:
    • Restart process
    • Clear cache
    • Reconnect services
    • Scale resources
    
    Manual:
    • Investigate logs
    • Debug issues
    • Apply fixes
    • Manual restart
    
    Preventive:
    • Regular maintenance
    • Update dependencies
    • Optimize performance
    • Capacity planning
    """
    
    return {
        "messages": [AIMessage(content=f"🏥 Health Monitor:\n{response.content}\n{report}")],
        "resources": resources,
        "health_checks_passed": health_checks_passed,
        "health_checks_failed": health_checks_failed
    }


# Lifecycle Monitor
def lifecycle_monitor(state: LifecycleState) -> LifecycleState:
    """Monitors overall lifecycle management"""
    resource_type = state.get("resource_type", "")
    lifecycle_stage = state.get("lifecycle_stage", "")
    resources = state.get("resources", {})
    transition_history = state.get("transition_history", [])
    health_checks_passed = state.get("health_checks_passed", 0)
    health_checks_failed = state.get("health_checks_failed", 0)
    
    summary = f"""
    📊 LIFECYCLE MANAGEMENT COMPLETE
    
    Lifecycle Status:
    • Resource Type: {resource_type.upper()}
    • Current Stage: {lifecycle_stage.upper()}
    • Total Resources: {len(resources)}
    • Transitions: {len(transition_history)}
    • Health Checks Passed: {health_checks_passed}
    • Health Checks Failed: {health_checks_failed}
    
    Resource States:
    {chr(10).join(f"• {rid}: {data['state']} ({data['health']})" for rid, data in resources.items())}
    
    Lifecycle Management Pattern Process:
    1. Lifecycle Manager → Manage state transitions
    2. Health Monitor → Check resource health
    3. Monitor → Track lifecycle metrics
    
    Best Practices:
    
    Design:
    • Define clear lifecycle stages
    • Document valid transitions
    • Implement lifecycle hooks
    • Handle edge cases
    • Plan for failures
    
    Implementation:
    • State machine pattern
    • Event-driven transitions
    • Idempotent operations
    • Atomic state changes
    • Transaction support
    
    Monitoring:
    • Track state distribution
    • Monitor transition times
    • Alert on stuck resources
    • Health check metrics
    • Lifecycle duration
    
    Graceful Shutdown:
    • Signal handling (SIGTERM)
    • Drain connections
    • Complete in-flight work
    • Save state
    • Release resources
    
    Real-World Examples:
    
    AWS EC2 Instance:
    • pending → running
    • running → stopping
    • stopping → stopped
    • stopped → terminated
    • Can start stopped instances
    
    Kubernetes Pod:
    • Pending → Running
    • Running → Succeeded/Failed
    • Lifecycle hooks: postStart, preStop
    • Init containers
    • Sidecar containers
    
    Database Connection:
    • Closed → Connecting
    • Connecting → Open
    • Open → Executing
    • Executing → Open
    • Open → Closing
    • Closing → Closed
    
    Application Server:
    • Stopped → Starting
    • Starting → Ready
    • Ready → Serving
    • Serving → Draining
    • Draining → Stopped
    
    Lifecycle Metrics:
    
    Duration Metrics:
    • Time to active (startup)
    • Active duration (uptime)
    • Shutdown duration
    • Total lifecycle time
    
    Transition Metrics:
    • Transitions per hour
    • Failed transitions
    • Rollback rate
    • State distribution
    
    Health Metrics:
    • Health check success rate
    • Time in degraded state
    • Recovery time
    • MTBF, MTTR
    
    Common Patterns:
    
    Circuit Breaker Integration:
    • Monitor → Degraded → Shutdown
    • Automatic recovery attempts
    • Open circuit on repeated failures
    
    Auto-Scaling:
    • Monitor demand
    • Create new resources
    • Initialize and activate
    • Shutdown idle resources
    
    Blue-Green Deployment:
    • Create green environment
    • Initialize and warm up
    • Switch traffic
    • Shutdown blue environment
    
    Key Insight:
    Proper lifecycle management ensures resources are
    created, maintained, and disposed of correctly,
    preventing leaks and ensuring system reliability.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Lifecycle Monitor:\n{summary}")]
    }


# Build the graph
def build_lifecycle_graph():
    """Build the lifecycle management pattern graph"""
    workflow = StateGraph(LifecycleState)
    
    workflow.add_node("manager", lifecycle_manager)
    workflow.add_node("health", health_monitor)
    workflow.add_node("monitor", lifecycle_monitor)
    
    workflow.add_edge(START, "manager")
    workflow.add_edge("manager", "health")
    workflow.add_edge("health", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_lifecycle_graph()
    
    print("=== Lifecycle Management MCP Pattern ===\n")
    
    # Test Case: Application server lifecycle
    print("\n" + "="*70)
    print("TEST CASE: Application Server Lifecycle")
    print("="*70)
    
    state = {
        "messages": [],
        "resource_type": "application_server",
        "lifecycle_stage": "creation",
        "resources": {},
        "transition_history": [],
        "health_checks_passed": 0,
        "health_checks_failed": 0
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    resources = result.get('resources', {})
    print(f"\nTotal Resources: {len(resources)}")
    print(f"Health Checks Passed: {result.get('health_checks_passed', 0)}")
    print(f"Health Checks Failed: {result.get('health_checks_failed', 0)}")
