"""
Resource Pooling MCP Pattern

This pattern manages a pool of reusable resources to improve performance and
reduce overhead by reusing existing resources instead of creating new ones.

Key Features:
- Resource pool management
- Efficient allocation and deallocation
- Resource lifecycle tracking
- Pool size management
- Resource health monitoring
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ResourcePoolState(TypedDict):
    """State for resource pooling pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    pool_name: str
    resource_type: str  # "database", "thread", "connection", "worker"
    pool_size: int
    min_pool_size: int
    max_pool_size: int
    available_resources: int
    in_use_resources: int
    resource_requests: int
    pool_utilization: float  # 0.0 to 1.0
    resource_health: Dict[str, str]  # resource_id -> "healthy", "degraded", "failed"
    wait_time: float  # average wait time for resource acquisition
    allocation_strategy: str  # "fifo", "lifo", "least_used", "round_robin"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Pool Manager
def pool_manager(state: ResourcePoolState) -> ResourcePoolState:
    """Manages resource pool initialization and sizing"""
    pool_name = state.get("pool_name", "")
    resource_type = state.get("resource_type", "")
    min_pool_size = state.get("min_pool_size", 5)
    max_pool_size = state.get("max_pool_size", 20)
    
    system_message = SystemMessage(content="""You are a resource pool manager. 
    Initialize and manage pools of reusable resources for optimal performance.""")
    
    user_message = HumanMessage(content=f"""Manage resource pool:

Pool Name: {pool_name}
Resource Type: {resource_type}
Min Pool Size: {min_pool_size}
Max Pool Size: {max_pool_size}

Initialize and configure resource pool.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Initialize pool with minimum resources
    pool_size = min_pool_size
    available_resources = min_pool_size
    in_use_resources = 0
    
    # Create initial resource health tracking
    resource_health = {}
    for i in range(pool_size):
        resource_id = f"{pool_name}_resource_{i+1}"
        resource_health[resource_id] = "healthy"
    
    pool_utilization = 0.0  # No resources in use initially
    
    pool_report = f"""
    🏊 Resource Pool Management:
    
    Pool Configuration:
    • Pool Name: {pool_name}
    • Resource Type: {resource_type}
    • Current Size: {pool_size}
    • Min Size: {min_pool_size}
    • Max Size: {max_pool_size}
    
    Resource Status:
    • Available: {available_resources}
    • In Use: {in_use_resources}
    • Utilization: {pool_utilization:.1%}
    • Health Status: {len([h for h in resource_health.values() if h == 'healthy'])} healthy
    
    Resource Pooling Benefits:
    
    Performance:
    • Reduced creation overhead
    • Faster resource acquisition
    • Improved response times
    • Lower latency
    • Better throughput
    
    Resource Efficiency:
    • Reuse existing resources
    • Limit resource creation
    • Controlled resource count
    • Prevent resource exhaustion
    • Optimize resource usage
    
    Scalability:
    • Handle traffic spikes
    • Dynamic pool sizing
    • Load distribution
    • Graceful degradation
    • Elastic scaling
    
    Pool Management Strategies:
    
    Sizing:
    • Min pool size: Always available resources
    • Max pool size: Resource cap limit
    • Dynamic sizing: Grow/shrink based on demand
    • Idle timeout: Return unused resources
    
    Allocation:
    • FIFO: First in, first out
    • LIFO: Last in, first out
    • Least Used: Balance wear
    • Round Robin: Even distribution
    
    Health Management:
    • Health checks before allocation
    • Automatic resource replacement
    • Graceful degradation
    • Circuit breaker integration
    
    Pool Types:
    
    Database Connection Pool:
    • Reuse DB connections
    • Reduce connection overhead
    • Limit concurrent connections
    • Connection validation
    • Statement caching
    
    Thread Pool:
    • Worker thread reuse
    • Task queue processing
    • Thread lifecycle management
    • Work stealing
    • Priority queues
    
    Object Pool:
    • Expensive object reuse
    • Memory allocation reduction
    • Object lifecycle management
    • State reset between uses
    
    HTTP Connection Pool:
    • Keep-alive connections
    • SSL/TLS session reuse
    • DNS caching
    • Connection pipelining
    
    Resource Pool Lifecycle:
    
    1. Initialization:
       • Create minimum resources
       • Validate resources
       • Mark as available
    
    2. Acquisition:
       • Check for available resource
       • Validate resource health
       • Mark as in-use
       • Return to requester
    
    3. Usage:
       • Resource performs work
       • Monitor resource health
       • Track usage metrics
    
    4. Release:
       • Return to pool
       • Reset resource state
       • Mark as available
       • Health check
    
    5. Cleanup:
       • Remove unhealthy resources
       • Shrink pool if needed
       • Close idle resources
       • Free memory
    
    Configuration Best Practices:
    • Set appropriate min/max sizes
    • Monitor pool metrics
    • Tune based on workload
    • Implement health checks
    • Use timeout mechanisms
    • Handle resource exhaustion
    • Log pool statistics
    """
    
    return {
        "messages": [AIMessage(content=f"🏊 Pool Manager:\n{response.content}\n{pool_report}")],
        "pool_size": pool_size,
        "available_resources": available_resources,
        "in_use_resources": in_use_resources,
        "pool_utilization": pool_utilization,
        "resource_health": resource_health
    }


# Resource Allocator
def resource_allocator(state: ResourcePoolState) -> ResourcePoolState:
    """Allocates resources from the pool"""
    pool_name = state.get("pool_name", "")
    available_resources = state.get("available_resources", 0)
    pool_size = state.get("pool_size", 0)
    max_pool_size = state.get("max_pool_size", 20)
    resource_requests = state.get("resource_requests", 5)  # Simulated requests
    allocation_strategy = state.get("allocation_strategy", "fifo")
    
    system_message = SystemMessage(content="""You are a resource allocator. 
    Efficiently allocate resources from the pool to satisfy requests.""")
    
    user_message = HumanMessage(content=f"""Allocate resources:

Pool: {pool_name}
Available Resources: {available_resources}
Pool Size: {pool_size}
Max Pool Size: {max_pool_size}
Pending Requests: {resource_requests}
Strategy: {allocation_strategy}

Allocate resources to requests.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate resource allocation
    resources_to_allocate = min(resource_requests, available_resources)
    
    # If no resources available but can grow pool
    if resources_to_allocate == 0 and pool_size < max_pool_size:
        # Grow pool
        growth = min(resource_requests, max_pool_size - pool_size)
        pool_size += growth
        available_resources += growth
        resources_to_allocate = min(resource_requests, available_resources)
        pool_grown = True
    else:
        pool_grown = False
    
    # Allocate resources
    allocated = resources_to_allocate
    available_resources -= allocated
    in_use_resources = state.get("in_use_resources", 0) + allocated
    
    # Calculate wait time (if requests > available)
    wait_time = 0.0
    if resource_requests > resources_to_allocate:
        # Some requests had to wait
        wait_time = (resource_requests - resources_to_allocate) * 0.1  # Simulated
    
    # Calculate pool utilization
    pool_utilization = in_use_resources / pool_size if pool_size > 0 else 0.0
    
    allocation_report = f"""
    📤 Resource Allocation:
    
    Allocation Results:
    • Requests: {resource_requests}
    • Allocated: {allocated}
    • Queued/Rejected: {resource_requests - allocated}
    • Success Rate: {(allocated/resource_requests*100) if resource_requests > 0 else 0:.1f}%
    
    Pool Status After Allocation:
    • Total Pool Size: {pool_size}
    • Available: {available_resources}
    • In Use: {in_use_resources}
    • Utilization: {pool_utilization:.1%}
    {f'• Pool Grown: +{growth} resources' if pool_grown else ''}
    
    Performance Metrics:
    • Average Wait Time: {wait_time:.3f}s
    • Allocation Strategy: {allocation_strategy.upper()}
    
    Allocation Strategies:
    
    FIFO (First-In-First-Out):
    • Fair allocation order
    • Simple to implement
    • No resource starvation
    • Predictable behavior
    
    LIFO (Last-In-First-Out):
    • Hot resource reuse
    • Better cache locality
    • May cause starvation
    • Stack-based allocation
    
    Least Used:
    • Balance resource wear
    • Extend resource lifetime
    • Even usage distribution
    • Health-aware allocation
    
    Round Robin:
    • Circular allocation
    • Even distribution
    • Simple load balancing
    • Predictable pattern
    
    Priority-Based:
    • VIP request handling
    • Critical workload first
    • Service level objectives
    • Multi-tier allocation
    
    Allocation Scenarios:
    
    High Demand (Requests > Available):
    • Queue excess requests
    • Grow pool if possible
    • Apply backpressure
    • Reject with timeout
    • Shed load if needed
    
    Normal Load:
    • Direct allocation
    • Minimal wait time
    • Optimal utilization
    • Stable performance
    
    Low Demand (Available > In-Use):
    • Immediate allocation
    • Consider pool shrinking
    • Idle timeout cleanup
    • Resource conservation
    
    Resource Exhaustion:
    • All resources in use
    • Pool at maximum
    • Queue requests
    • Apply timeout
    • Alternative: rejection
    
    Dynamic Pool Sizing:
    
    Growing Conditions:
    • High utilization (>80%)
    • Frequent queuing
    • Sustained demand
    • Below max pool size
    
    Shrinking Conditions:
    • Low utilization (<20%)
    • Idle resources
    • Above min pool size
    • Resource cost reduction
    
    Allocation Failure Handling:
    • Request queuing
    • Timeout mechanisms
    • Graceful degradation
    • Error responses
    • Retry logic
    • Circuit breaker
    """
    
    return {
        "messages": [AIMessage(content=f"📤 Resource Allocator:\n{response.content}\n{allocation_report}")],
        "pool_size": pool_size,
        "available_resources": available_resources,
        "in_use_resources": in_use_resources,
        "pool_utilization": pool_utilization,
        "wait_time": wait_time
    }


# Health Monitor
def health_monitor(state: ResourcePoolState) -> ResourcePoolState:
    """Monitors resource health and performs maintenance"""
    pool_name = state.get("pool_name", "")
    resource_health = state.get("resource_health", {})
    pool_size = state.get("pool_size", 0)
    
    system_message = SystemMessage(content="""You are a resource health monitor. 
    Monitor resource health, detect failures, and maintain pool quality.""")
    
    user_message = HumanMessage(content=f"""Monitor resource health:

Pool: {pool_name}
Pool Size: {pool_size}
Resources Tracked: {len(resource_health)}

Perform health checks and maintenance.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate health checks
    healthy_count = 0
    degraded_count = 0
    failed_count = 0
    
    for resource_id, health in resource_health.items():
        if health == "healthy":
            healthy_count += 1
        elif health == "degraded":
            degraded_count += 1
        else:
            failed_count += 1
    
    # Simulate some resources becoming degraded (random simulation)
    # In production, this would be actual health checks
    import random
    if healthy_count > 0 and random.random() < 0.1:  # 10% chance
        # Mark one resource as degraded
        for resource_id, health in resource_health.items():
            if health == "healthy":
                resource_health[resource_id] = "degraded"
                healthy_count -= 1
                degraded_count += 1
                break
    
    health_percentage = (healthy_count / len(resource_health) * 100) if resource_health else 0
    
    health_report = f"""
    🏥 Resource Health Monitoring:
    
    Health Status:
    • Healthy: {healthy_count} ({healthy_count/len(resource_health)*100 if resource_health else 0:.1f}%)
    • Degraded: {degraded_count}
    • Failed: {failed_count}
    • Overall Health: {health_percentage:.1f}%
    
    Pool Health Metrics:
    • Total Resources: {len(resource_health)}
    • Health Check Interval: 30 seconds
    • Auto-Replacement: Enabled
    
    Health Monitoring:
    
    Health Check Types:
    
    Passive Checks:
    • On resource acquisition
    • Before returning to pool
    • Lightweight validation
    • No overhead when idle
    
    Active Checks:
    • Periodic background checks
    • Proactive detection
    • All resources checked
    • Scheduled intervals
    
    Deep Checks:
    • Comprehensive validation
    • Resource stress testing
    • Connection verification
    • Performance benchmarking
    
    Health Indicators:
    
    Database Connections:
    • Connection validity
    • Query execution
    • Response time
    • Error rate
    • Transaction state
    
    Thread Pool:
    • Thread state
    • CPU usage
    • Memory consumption
    • Queue depth
    • Task completion rate
    
    HTTP Connections:
    • Connection state
    • Latency
    • Error rate
    • SSL certificate validity
    • DNS resolution
    
    Health States:
    
    Healthy (100%):
    • All checks pass
    • Normal performance
    • Ready for allocation
    • No intervention needed
    
    Degraded (50-99%):
    • Some checks fail
    • Reduced performance
    • Still functional
    • Monitor closely
    • May self-recover
    
    Failed (0-49%):
    • Critical checks fail
    • Unusable resource
    • Remove from pool
    • Create replacement
    • Investigate cause
    
    Maintenance Actions:
    
    Resource Replacement:
    • Remove failed resources
    • Create new resources
    • Validate new resources
    • Add to pool
    • Maintain pool size
    
    Resource Repair:
    • Attempt recovery
    • Reset connections
    • Clear state
    • Re-validate
    • Return to pool
    
    Pool Optimization:
    • Remove excess resources
    • Balance resource usage
    • Defragment pool
    • Update configuration
    
    Preventive Maintenance:
    • Rotate resources
    • Prevent resource aging
    • Update credentials
    • Refresh certificates
    • Clear caches
    
    Health Monitoring Best Practices:
    • Regular health checks
    • Multiple health indicators
    • Automated remediation
    • Health metrics logging
    • Alert on degradation
    • Trend analysis
    • Capacity planning
    """
    
    return {
        "messages": [AIMessage(content=f"🏥 Health Monitor:\n{response.content}\n{health_report}")],
        "resource_health": resource_health
    }


# Pool Monitor
def pool_monitor(state: ResourcePoolState) -> ResourcePoolState:
    """Monitors overall pool performance and provides insights"""
    pool_name = state.get("pool_name", "")
    resource_type = state.get("resource_type", "")
    pool_size = state.get("pool_size", 0)
    min_pool_size = state.get("min_pool_size", 0)
    max_pool_size = state.get("max_pool_size", 0)
    available_resources = state.get("available_resources", 0)
    in_use_resources = state.get("in_use_resources", 0)
    pool_utilization = state.get("pool_utilization", 0.0)
    wait_time = state.get("wait_time", 0.0)
    resource_health = state.get("resource_health", {})
    
    healthy_count = len([h for h in resource_health.values() if h == "healthy"])
    
    summary = f"""
    📊 RESOURCE POOLING COMPLETE
    
    Pool Summary:
    • Pool Name: {pool_name}
    • Resource Type: {resource_type}
    • Pool Size: {pool_size} (min: {min_pool_size}, max: {max_pool_size})
    
    Resource Status:
    • Available: {available_resources}
    • In Use: {in_use_resources}
    • Utilization: {pool_utilization:.1%}
    • Healthy: {healthy_count}/{len(resource_health)}
    
    Performance:
    • Average Wait Time: {wait_time:.3f}s
    • Health Status: {(healthy_count/len(resource_health)*100) if resource_health else 0:.1f}%
    
    Resource Pooling Pattern Process:
    1. Pool Manager → Initialize and configure resource pool
    2. Resource Allocator → Allocate resources to requests
    3. Health Monitor → Monitor resource health and maintenance
    4. Pool Monitor → Track metrics and optimize performance
    
    Resource Pooling Patterns:
    
    Common Pool Implementations:
    
    Apache Commons Pool:
    • Java object pooling
    • Generic pool framework
    • Configurable behaviors
    • Factory pattern
    • Validation support
    
    HikariCP:
    • Fast JDBC connection pool
    • Lightweight
    • High performance
    • JMX monitoring
    • Leak detection
    
    C3P0:
    • JDBC connection pooling
    • Statement caching
    • PreparedStatement pooling
    • Automatic testing
    • Recovery mechanisms
    
    Thread Pool Executor:
    • Java concurrent utilities
    • Worker thread pool
    • Task queue
    • Rejection policies
    • Thread factory
    
    Pool Configuration Parameters:
    
    Size Configuration:
    • minPoolSize: Minimum resources
    • maxPoolSize: Maximum resources
    • initialPoolSize: Starting resources
    • maxIdleTime: Idle resource timeout
    • acquireIncrement: Growth increment
    
    Timeout Configuration:
    • connectionTimeout: Acquisition timeout
    • idleTimeout: Idle before removal
    • maxLifetime: Resource lifetime
    • keepAliveTime: Thread keep-alive
    
    Validation Configuration:
    • testOnBorrow: Validate on acquisition
    • testOnReturn: Validate on release
    • testWhileIdle: Background validation
    • validationQuery: Health check query
    • validationTimeout: Check timeout
    
    Pool Metrics to Monitor:
    
    Utilization Metrics:
    • Active connections
    • Idle connections
    • Pool utilization %
    • Wait queue length
    • Request rate
    
    Performance Metrics:
    • Acquisition time
    • Wait time
    • Throughput
    • Error rate
    • Timeout rate
    
    Health Metrics:
    • Healthy resources
    • Failed resources
    • Replacement rate
    • Validation failures
    
    Resource Metrics:
    • Memory usage
    • Thread count
    • Connection count
    • Pool size
    • Growth/shrink events
    
    Common Pool Problems:
    
    Pool Exhaustion:
    • Symptom: All resources in use
    • Cause: Insufficient pool size
    • Solution: Increase max pool size
    • Prevention: Monitor utilization
    
    Resource Leaks:
    • Symptom: Resources not returned
    • Cause: Missing finally blocks
    • Solution: Automatic cleanup
    • Prevention: Proper resource management
    
    Stale Resources:
    • Symptom: Failed health checks
    • Cause: Resource aging
    • Solution: Refresh/replacement
    • Prevention: MaxLifetime setting
    
    Thrashing:
    • Symptom: Frequent grow/shrink
    • Cause: Improper sizing
    • Solution: Tune min/max sizes
    • Prevention: Workload analysis
    
    Pool Tuning Guidelines:
    
    Database Connection Pool:
    • Size: (core_count * 2) + effective_spindle_count
    • HikariCP formula for optimal sizing
    • Consider query duration
    • Monitor active connections
    
    Thread Pool:
    • CPU-bound: core_count + 1
    • I/O-bound: core_count * (1 + wait_time/service_time)
    • Consider task characteristics
    • Use fixed or cached pool
    
    HTTP Connection Pool:
    • Per-route max: 20-50
    • Total max: 200-500
    • Keep-alive: 30-60 seconds
    • Timeout: 5-30 seconds
    
    Best Practices:
    
    Design:
    • Use proven pool libraries
    • Configure appropriate sizes
    • Implement health checks
    • Handle resource cleanup
    • Use try-with-resources
    
    Monitoring:
    • Track pool metrics
    • Set up alerts
    • Monitor growth patterns
    • Analyze wait times
    • Review error logs
    
    Maintenance:
    • Regular health checks
    • Automatic resource refresh
    • Connection validation
    • Leak detection
    • Performance profiling
    
    Testing:
    • Load testing
    • Stress testing
    • Resource leak detection
    • Failure scenario testing
    • Timeout testing
    
    When to Use Resource Pooling:
    
    ✅ Good Fit:
    • Expensive resource creation
    • Frequent resource usage
    • Limited resource availability
    • Performance critical paths
    • High concurrency
    
    ❌ Not Recommended:
    • Cheap resource creation
    • Infrequent usage
    • No resource constraints
    • Simple use cases
    • Low concurrency
    
    Key Insight:
    Resource pooling improves performance by reusing expensive
    resources instead of creating new ones. Essential for database
    connections, threads, and network connections. Proper sizing
    and monitoring are critical for optimal performance.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Pool Monitor:\n{summary}")]
    }


# Build the graph
def build_resource_pool_graph():
    """Build the resource pooling pattern graph"""
    workflow = StateGraph(ResourcePoolState)
    
    workflow.add_node("pool_mgr", pool_manager)
    workflow.add_node("allocator", resource_allocator)
    workflow.add_node("health", health_monitor)
    workflow.add_node("monitor", pool_monitor)
    
    workflow.add_edge(START, "pool_mgr")
    workflow.add_edge("pool_mgr", "allocator")
    workflow.add_edge("allocator", "health")
    workflow.add_edge("health", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_resource_pool_graph()
    
    print("=== Resource Pooling MCP Pattern ===\n")
    
    # Test Case 1: Database connection pool
    print("\n" + "="*70)
    print("TEST CASE 1: Database Connection Pool")
    print("="*70)
    
    state1 = {
        "messages": [],
        "pool_name": "db_connection_pool",
        "resource_type": "database",
        "pool_size": 0,
        "min_pool_size": 5,
        "max_pool_size": 20,
        "available_resources": 0,
        "in_use_resources": 0,
        "resource_requests": 8,
        "pool_utilization": 0.0,
        "resource_health": {},
        "wait_time": 0.0,
        "allocation_strategy": "fifo"
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nPool Size: {result1.get('pool_size')}")
    print(f"Available: {result1.get('available_resources')}")
    print(f"In Use: {result1.get('in_use_resources')}")
    print(f"Utilization: {result1.get('pool_utilization', 0):.1%}")
    
    # Test Case 2: Thread pool with high demand
    print("\n\n" + "="*70)
    print("TEST CASE 2: Thread Pool (High Demand)")
    print("="*70)
    
    state2 = {
        "messages": [],
        "pool_name": "worker_thread_pool",
        "resource_type": "thread",
        "pool_size": 0,
        "min_pool_size": 10,
        "max_pool_size": 50,
        "available_resources": 0,
        "in_use_resources": 0,
        "resource_requests": 45,  # High demand
        "pool_utilization": 0.0,
        "resource_health": {},
        "wait_time": 0.0,
        "allocation_strategy": "least_used"
    }
    
    result2 = graph.invoke(state2)
    
    print(f"\nPool: {state2['pool_name']}")
    print(f"Resource Type: {state2['resource_type']}")
    print(f"Requests: {state2['resource_requests']}")
    print(f"Pool Size: {result2.get('pool_size')}")
    print(f"Utilization: {result2.get('pool_utilization', 0):.1%}")
    print(f"Wait Time: {result2.get('wait_time', 0):.3f}s")
