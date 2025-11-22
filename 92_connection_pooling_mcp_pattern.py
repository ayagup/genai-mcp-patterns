"""
Connection Pooling MCP Pattern

This pattern manages a pool of database/network connections to reduce
connection overhead and improve application performance.

Key Features:
- Connection pool management
- Connection validation and testing
- Connection lifecycle handling
- Automatic reconnection
- Connection leak detection
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ConnectionPoolState(TypedDict):
    """State for connection pooling pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    pool_name: str
    connection_type: str  # "database", "http", "redis", "message_queue"
    min_connections: int
    max_connections: int
    active_connections: int
    idle_connections: int
    connection_timeout: float  # seconds
    idle_timeout: float  # seconds
    max_lifetime: float  # seconds
    validation_query: str
    connection_stats: Dict[str, int]  # "created", "closed", "reused", "failed"
    leak_detected: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Connection Pool Manager
def connection_pool_manager(state: ConnectionPoolState) -> ConnectionPoolState:
    """Manages connection pool initialization"""
    pool_name = state.get("pool_name", "")
    connection_type = state.get("connection_type", "database")
    min_connections = state.get("min_connections", 5)
    max_connections = state.get("max_connections", 20)
    
    system_message = SystemMessage(content="""You are a connection pool manager.
    Initialize and maintain a pool of reusable connections.""")
    
    user_message = HumanMessage(content=f"""Initialize connection pool:

Pool: {pool_name}
Type: {connection_type}
Min: {min_connections}
Max: {max_connections}

Set up connection pool.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Initialize pool
    idle_connections = min_connections
    active_connections = 0
    
    connection_stats = {
        "created": min_connections,
        "closed": 0,
        "reused": 0,
        "failed": 0
    }
    
    report = f"""
    🔌 Connection Pool Management:
    
    Pool Configuration:
    • Pool Name: {pool_name}
    • Connection Type: {connection_type.upper()}
    • Min Connections: {min_connections}
    • Max Connections: {max_connections}
    • Idle Connections: {idle_connections}
    • Active Connections: {active_connections}
    
    Connection Pool Benefits:
    • Faster connection acquisition
    • Reduced connection overhead
    • Better resource utilization
    • Connection reuse
    • Controlled connection count
    
    Popular Connection Pools:
    
    Database:
    • HikariCP (Java - fastest)
    • Apache DBCP (Java)
    • C3P0 (Java)
    • Psycopg2 Pool (Python)
    • SQLAlchemy (Python)
    • node-postgres (Node.js)
    
    HTTP:
    • Apache HttpClient
    • OkHttp (Android/Java)
    • Requests Session (Python)
    • urllib3 (Python)
    • Axios (Node.js)
    
    Redis:
    • Jedis Pool (Java)
    • redis-py ConnectionPool
    • node-redis
    • StackExchange.Redis
    
    Configuration Best Practices:
    • Set appropriate min/max sizes
    • Configure timeouts properly
    • Enable connection validation
    • Monitor pool metrics
    • Handle connection leaks
    • Use prepared statements
    • Implement retry logic
    """
    
    return {
        "messages": [AIMessage(content=f"🔌 Connection Pool Manager:\n{response.content}\n{report}")],
        "idle_connections": idle_connections,
        "active_connections": active_connections,
        "connection_stats": connection_stats
    }


# Connection Validator
def connection_validator(state: ConnectionPoolState) -> ConnectionPoolState:
    """Validates connections before use"""
    pool_name = state.get("pool_name", "")
    connection_type = state.get("connection_type", "")
    validation_query = state.get("validation_query", "SELECT 1")
    idle_connections = state.get("idle_connections", 0)
    
    system_message = SystemMessage(content="""You are a connection validator.
    Test connections to ensure they're healthy before use.""")
    
    user_message = HumanMessage(content=f"""Validate connections:

Pool: {pool_name}
Type: {connection_type}
Validation Query: {validation_query}
Idle Connections: {idle_connections}

Perform validation checks.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate validation
    validation_results = {
        "total_validated": idle_connections,
        "passed": idle_connections,  # All pass in simulation
        "failed": 0,
        "validation_time_ms": 15.3
    }
    
    report = f"""
    ✅ Connection Validation:
    
    Validation Results:
    • Total Validated: {validation_results['total_validated']}
    • Passed: {validation_results['passed']}
    • Failed: {validation_results['failed']}
    • Average Time: {validation_results['validation_time_ms']:.1f}ms
    
    Validation Strategies:
    
    Test On Borrow:
    • Validate before returning to application
    • Ensures connection is valid
    • Adds latency to acquisition
    • Most reliable approach
    
    Test On Return:
    • Validate when returned to pool
    • Detect failures during use
    • No impact on acquisition
    • Cleanup invalid connections
    
    Test While Idle:
    • Background validation
    • Periodic health checks
    • No user impact
    • Proactive detection
    • Configured interval
    
    Validation Methods:
    
    Database:
    • Execute simple query (SELECT 1)
    • Check connection status
    • Verify transaction state
    • Test response time
    • Ping connection
    
    HTTP:
    • Send HEAD request
    • Check socket state
    • Verify SSL certificate
    • Test connection pool
    • Measure latency
    
    Redis:
    • PING command
    • Check connection state
    • Verify authentication
    • Test response time
    
    Validation Query Examples:
    • MySQL: SELECT 1
    • PostgreSQL: SELECT 1
    • Oracle: SELECT 1 FROM DUAL
    • SQL Server: SELECT 1
    • Redis: PING
    """
    
    return {
        "messages": [AIMessage(content=f"✅ Connection Validator:\n{response.content}\n{report}")]
    }


# Connection Lifecycle Manager
def connection_lifecycle_manager(state: ConnectionPoolState) -> ConnectionPoolState:
    """Manages connection lifecycle (creation, reuse, closure)"""
    connection_stats = state.get("connection_stats", {})
    idle_timeout = state.get("idle_timeout", 600.0)
    max_lifetime = state.get("max_lifetime", 1800.0)
    
    system_message = SystemMessage(content="""You are a connection lifecycle manager.
    Manage connection creation, reuse, and proper cleanup.""")
    
    user_message = HumanMessage(content=f"""Manage connection lifecycle:

Stats: {connection_stats}
Idle Timeout: {idle_timeout}s
Max Lifetime: {max_lifetime}s

Handle connection lifecycle.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate lifecycle events
    connection_stats["reused"] = connection_stats.get("reused", 0) + 5
    
    report = f"""
    ♻️ Connection Lifecycle:
    
    Lifecycle Statistics:
    • Created: {connection_stats.get('created', 0)}
    • Reused: {connection_stats.get('reused', 0)}
    • Closed: {connection_stats.get('closed', 0)}
    • Failed: {connection_stats.get('failed', 0)}
    • Reuse Ratio: {(connection_stats.get('reused', 0) / max(connection_stats.get('created', 1), 1)):.2f}
    
    Lifecycle Stages:
    
    1. Creation:
       • Establish connection
       • Authenticate
       • Set connection properties
       • Add to pool
       • Increment created counter
    
    2. Acquisition:
       • Get from idle pool
       • Validate if configured
       • Mark as active
       • Return to application
       • Track acquisition time
    
    3. Usage:
       • Execute operations
       • Monitor connection health
       • Track usage metrics
       • Detect errors
    
    4. Release:
       • Return to pool
       • Reset connection state
       • Clear session variables
       • Mark as idle
       • Increment reused counter
    
    5. Eviction:
       • Check idle timeout
       • Check max lifetime
       • Validation failure
       • Close connection
       • Remove from pool
       • Create replacement if needed
    
    Connection Properties:
    
    Timeouts:
    • Connection Timeout: {state.get('connection_timeout', 30)}s
    • Idle Timeout: {idle_timeout}s ({idle_timeout/60:.1f} minutes)
    • Max Lifetime: {max_lifetime}s ({max_lifetime/60:.1f} minutes)
    
    Why Timeouts Matter:
    • Idle Timeout: Free up unused connections
    • Max Lifetime: Prevent connection aging
    • Connection Timeout: Fail fast on unavailable servers
    • Statement Timeout: Prevent long-running queries
    
    Leak Detection:
    • Track connection acquisition time
    • Alert on long-held connections
    • Force close leaked connections
    • Log leak stack traces
    • Prevent pool exhaustion
    """
    
    return {
        "messages": [AIMessage(content=f"♻️ Connection Lifecycle Manager:\n{response.content}\n{report}")],
        "connection_stats": connection_stats
    }


# Pool Monitor
def pool_monitor(state: ConnectionPoolState) -> ConnectionPoolState:
    """Monitors pool performance and health"""
    pool_name = state.get("pool_name", "")
    connection_type = state.get("connection_type", "")
    min_connections = state.get("min_connections", 0)
    max_connections = state.get("max_connections", 0)
    active_connections = state.get("active_connections", 0)
    idle_connections = state.get("idle_connections", 0)
    connection_stats = state.get("connection_stats", {})
    leak_detected = state.get("leak_detected", False)
    
    total_connections = active_connections + idle_connections
    utilization = (active_connections / max_connections * 100) if max_connections > 0 else 0
    
    summary = f"""
    📊 CONNECTION POOLING COMPLETE
    
    Pool Status:
    • Pool Name: {pool_name}
    • Connection Type: {connection_type.upper()}
    • Total Connections: {total_connections}/{max_connections}
    • Active: {active_connections}
    • Idle: {idle_connections}
    • Utilization: {utilization:.1f}%
    
    Statistics:
    • Created: {connection_stats.get('created', 0)}
    • Reused: {connection_stats.get('reused', 0)}
    • Closed: {connection_stats.get('closed', 0)}
    • Failed: {connection_stats.get('failed', 0)}
    • Leak Detected: {'Yes ⚠️' if leak_detected else 'No ✅'}
    
    Connection Pooling Pattern Process:
    1. Pool Manager → Initialize connection pool
    2. Validator → Ensure connection health
    3. Lifecycle Manager → Manage connection states
    4. Monitor → Track performance and detect issues
    
    Connection Pool Metrics:
    
    Performance Metrics:
    • Acquisition time (p50, p95, p99)
    • Connection creation rate
    • Connection reuse rate
    • Active connection count
    • Idle connection count
    • Wait queue length
    
    Health Metrics:
    • Validation success rate
    • Connection error rate
    • Timeout rate
    • Leak detection rate
    • Pool exhaustion events
    
    Resource Metrics:
    • Memory per connection
    • Total pool memory
    • Thread count
    • File descriptor count
    
    Common Issues and Solutions:
    
    Pool Exhaustion:
    • Problem: All connections in use
    • Cause: Insufficient max pool size
    • Solution: Increase max_connections
    • Prevention: Monitor utilization trends
    
    Connection Leaks:
    • Problem: Connections not returned
    • Cause: Missing close() in finally block
    • Solution: Use try-with-resources / context managers
    • Detection: Leak detection threshold
    
    Stale Connections:
    • Problem: Connections become invalid
    • Cause: Server timeout, network issues
    • Solution: Connection validation + max lifetime
    • Prevention: keepAliveTime setting
    
    Performance Degradation:
    • Problem: Slow connection acquisition
    • Cause: Validation overhead, pool contention
    • Solution: Tune validation, increase pool size
    • Optimization: Disable test-on-borrow
    
    Sizing Recommendations:
    
    Database Connection Pool:
    • Formula: connections = ((core_count * 2) + effective_spindle_count)
    • Example: 8 cores + 2 spindles = 18 connections
    • Consider: Query duration, transaction time
    • Start conservative, increase based on metrics
    
    HTTP Connection Pool:
    • Per-route max: 20-50 connections
    • Total max: 200-500 connections
    • Keep-alive timeout: 30-60 seconds
    • Consider: Latency, throughput requirements
    
    Best Practices:
    
    Configuration:
    • Set min = expected baseline load
    • Set max = peak load capacity
    • Enable connection validation
    • Configure appropriate timeouts
    • Use prepared statement caching
    
    Code Practices:
    • Always close connections (try-finally)
    • Use connection pooling libraries
    • Don't hold connections long
    • Handle exceptions properly
    • Use transactions appropriately
    
    Monitoring:
    • Track pool utilization
    • Monitor acquisition time
    • Alert on pool exhaustion
    • Detect connection leaks
    • Review error logs
    
    Testing:
    • Load test with realistic traffic
    • Test pool exhaustion scenarios
    • Simulate network failures
    • Test connection recovery
    • Verify leak detection
    
    Key Insight:
    Connection pooling dramatically improves performance by
    reusing expensive connections. Proper configuration and
    monitoring are essential for reliability and efficiency.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Pool Monitor:\n{summary}")]
    }


# Build the graph
def build_connection_pool_graph():
    """Build the connection pooling pattern graph"""
    workflow = StateGraph(ConnectionPoolState)
    
    workflow.add_node("pool_mgr", connection_pool_manager)
    workflow.add_node("validator", connection_validator)
    workflow.add_node("lifecycle", connection_lifecycle_manager)
    workflow.add_node("monitor", pool_monitor)
    
    workflow.add_edge(START, "pool_mgr")
    workflow.add_edge("pool_mgr", "validator")
    workflow.add_edge("validator", "lifecycle")
    workflow.add_edge("lifecycle", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_connection_pool_graph()
    
    print("=== Connection Pooling MCP Pattern ===\n")
    
    # Test Case: Database connection pool
    print("\n" + "="*70)
    print("TEST CASE: PostgreSQL Connection Pool")
    print("="*70)
    
    state = {
        "messages": [],
        "pool_name": "postgres_pool",
        "connection_type": "database",
        "min_connections": 5,
        "max_connections": 20,
        "active_connections": 0,
        "idle_connections": 0,
        "connection_timeout": 30.0,
        "idle_timeout": 600.0,
        "max_lifetime": 1800.0,
        "validation_query": "SELECT 1",
        "connection_stats": {},
        "leak_detected": False
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nActive Connections: {result.get('active_connections', 0)}")
    print(f"Idle Connections: {result.get('idle_connections', 0)}")
    stats = result.get('connection_stats', {})
    print(f"Reuse Ratio: {(stats.get('reused', 0) / max(stats.get('created', 1), 1)):.2f}")
