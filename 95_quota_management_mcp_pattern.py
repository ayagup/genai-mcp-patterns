"""
Quota Management MCP Pattern

This pattern manages resource quotas and usage limits to ensure fair
allocation and prevent resource exhaustion.

Key Features:
- Resource quota allocation
- Usage tracking and enforcement
- Quota renewal and expiration
- Multi-tier quota management
- Overage handling
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class QuotaState(TypedDict):
    """State for quota management pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    service_name: str
    quota_type: str  # "api_calls", "storage", "bandwidth", "compute"
    quota_period: str  # "hourly", "daily", "monthly"
    user_quotas: Dict[str, Dict[str, int]]  # user_id -> {limit, used, remaining}
    tier_quotas: Dict[str, int]  # tier -> limit
    quota_exceeded: List[str]  # user_ids that exceeded quota
    overage_policy: str  # "hard_limit", "soft_limit", "pay_per_use"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Quota Allocator
def quota_allocator(state: QuotaState) -> QuotaState:
    """Allocates quotas based on user tiers"""
    service_name = state.get("service_name", "")
    quota_type = state.get("quota_type", "api_calls")
    quota_period = state.get("quota_period", "daily")
    tier_quotas = state.get("tier_quotas", {"free": 1000, "pro": 10000, "enterprise": 100000})
    
    system_message = SystemMessage(content="""You are a quota allocator.
    Assign resource quotas based on user tiers and policies.""")
    
    user_message = HumanMessage(content=f"""Allocate quotas:

Service: {service_name}
Type: {quota_type}
Period: {quota_period}
Tiers: {tier_quotas}

Set up quota allocation.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Initialize user quotas
    user_quotas = {
        "user_free_1": {"tier": "free", "limit": tier_quotas["free"], "used": 0, "remaining": tier_quotas["free"]},
        "user_pro_1": {"tier": "pro", "limit": tier_quotas["pro"], "used": 0, "remaining": tier_quotas["pro"]},
        "user_ent_1": {"tier": "enterprise", "limit": tier_quotas["enterprise"], "used": 0, "remaining": tier_quotas["enterprise"]}
    }
    
    report = f"""
    📋 Quota Allocation:
    
    Configuration:
    • Service: {service_name}
    • Quota Type: {quota_type.upper()}
    • Period: {quota_period.upper()}
    
    Tier Quotas:
    • Free: {tier_quotas.get('free', 0):,}
    • Pro: {tier_quotas.get('pro', 0):,}
    • Enterprise: {tier_quotas.get('enterprise', 0):,}
    
    Quota Types:
    
    API Calls:
    • Request count limits
    • Per-endpoint quotas
    • Method-specific limits
    • Example: 1000 requests/day
    
    Storage:
    • Space allocation (GB/TB)
    • File count limits
    • Object storage quotas
    • Example: 100 GB storage
    
    Bandwidth:
    • Data transfer limits
    • Ingress/egress quotas
    • Network traffic caps
    • Example: 1 TB/month
    
    Compute:
    • CPU time quotas
    • Memory allocation
    • Execution duration
    • Example: 1000 compute hours
    
    Database:
    • Read/write operations
    • Connection limits
    • Query complexity
    • Example: 10M reads/day
    
    Quota Periods:
    
    Per-Second:
    • Burst protection
    • Rate limiting
    • Real-time control
    • Short-lived quotas
    
    Per-Hour:
    • Intermediate limits
    • Hourly budgets
    • Traffic shaping
    • Common for APIs
    
    Per-Day:
    • Daily budgets
    • Most common period
    • User-friendly
    • Easy to understand
    
    Per-Month:
    • Billing cycle alignment
    • Long-term limits
    • Subscription models
    • Storage quotas
    
    Multi-Tier Examples:
    
    GitHub API:
    • Free: 60 req/hour
    • Authenticated: 5,000 req/hour
    • Enterprise: Custom limits
    
    AWS Lambda:
    • Free: 1M requests/month
    • Concurrent: 1,000 executions
    • Duration: 1M seconds/month
    
    Google Maps API:
    • Free: $200 credit/month
    • Maps: 28,000 loads/month
    • Geocoding: 40,000 req/month
    
    Stripe API:
    • Default: 100 req/sec
    • Burst: 150 req/sec
    • Custom for high volume
    
    Quota Allocation Strategies:
    
    Fixed Quotas:
    • Simple tier-based
    • Predictable limits
    • Easy to communicate
    • Most common approach
    
    Dynamic Quotas:
    • Based on usage patterns
    • Time-of-day variations
    • Seasonal adjustments
    • Machine learning-driven
    
    Shared Quotas:
    • Organization-level
    • Team pooling
    • Resource sharing
    • Enterprise features
    
    Reserved Quotas:
    • Guaranteed capacity
    • Pre-allocated resources
    • SLA compliance
    • Priority access
    """
    
    return {
        "messages": [AIMessage(content=f"📋 Quota Allocator:\n{response.content}\n{report}")],
        "user_quotas": user_quotas
    }


# Usage Tracker
def usage_tracker(state: QuotaState) -> QuotaState:
    """Tracks resource usage against quotas"""
    user_quotas = state.get("user_quotas", {})
    quota_type = state.get("quota_type", "api_calls")
    
    system_message = SystemMessage(content="""You are a usage tracker.
    Monitor resource consumption and update quota usage.""")
    
    # Simulate usage
    usage_events = {
        "user_free_1": 250,
        "user_pro_1": 1500,
        "user_ent_1": 5000
    }
    
    user_message = HumanMessage(content=f"""Track usage:

Quota Type: {quota_type}
Usage Events: {usage_events}

Update quota consumption.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Update quotas
    quota_exceeded = []
    for user_id, usage in usage_events.items():
        if user_id in user_quotas:
            user_quotas[user_id]["used"] += usage
            user_quotas[user_id]["remaining"] = user_quotas[user_id]["limit"] - user_quotas[user_id]["used"]
            
            if user_quotas[user_id]["remaining"] < 0:
                quota_exceeded.append(user_id)
    
    report = f"""
    📊 Usage Tracking:
    
    Current Usage:
    {chr(10).join(f"• {uid}: {data['used']:,}/{data['limit']:,} ({data['tier']}) - {data['remaining']:,} remaining" 
                   for uid, data in user_quotas.items())}
    
    Quota Exceeded: {len(quota_exceeded)} users
    
    Usage Tracking Methods:
    
    Counter-Based:
    • Increment on each operation
    • Redis INCR command
    • Atomic operations
    • Fast and simple
    
    Time-Series:
    • Track usage over time
    • Granular analytics
    • Trend analysis
    • InfluxDB, Prometheus
    
    Event-Driven:
    • Publish usage events
    • Asynchronous tracking
    • Event sourcing
    • Kafka, Event Hub
    
    Batch Processing:
    • Aggregate periodically
    • Reduce write load
    • Eventually consistent
    • Cost-effective
    
    Implementation Example:
    ```python
    class QuotaTracker:
        def __init__(self, redis_client):
            self.redis = redis_client
        
        def track_usage(self, user_id, amount=1):
            # Increment usage counter
            key = f"quota:{user_id}:daily"
            used = self.redis.incr(key, amount)
            
            # Set expiry on first use
            if used == amount:
                self.redis.expire(key, 86400)  # 24h
            
            # Get limit
            limit = self.get_limit(user_id)
            
            # Check quota
            if used > limit:
                raise QuotaExceeded(
                    f"Quota exceeded: {used}/{limit}"
                )
            
            return {
                'used': used,
                'limit': limit,
                'remaining': limit - used
            }
        
        def reset_quota(self, user_id):
            key = f"quota:{user_id}:daily"
            self.redis.delete(key)
    ```
    
    Distributed Tracking:
    
    Redis:
    • INCR for atomicity
    • EXPIRE for auto-reset
    • Lua scripts for complex logic
    • Pub/Sub for notifications
    
    Database:
    • Transactional updates
    • Row-level locking
    • Durable storage
    • Audit trail
    
    In-Memory:
    • ConcurrentHashMap (Java)
    • Thread-safe counters
    • Fast access
    • Limited to single instance
    
    Hybrid:
    • Cache + persistent store
    • Write-through/Write-back
    • Best of both worlds
    • Complexity tradeoff
    
    Monitoring Metrics:
    • Quota utilization %
    • Time to quota exhaustion
    • Quota exceeded events
    • Peak usage times
    • Usage trends
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Usage Tracker:\n{response.content}\n{report}")],
        "user_quotas": user_quotas,
        "quota_exceeded": quota_exceeded
    }


# Quota Monitor
def quota_monitor(state: QuotaState) -> QuotaState:
    """Monitors quota status and handles overages"""
    service_name = state.get("service_name", "")
    quota_type = state.get("quota_type", "")
    user_quotas = state.get("user_quotas", {})
    quota_exceeded = state.get("quota_exceeded", [])
    overage_policy = state.get("overage_policy", "hard_limit")
    
    total_users = len(user_quotas)
    users_over_quota = len(quota_exceeded)
    
    summary = f"""
    📊 QUOTA MANAGEMENT COMPLETE
    
    Service Status:
    • Service: {service_name}
    • Quota Type: {quota_type.upper()}
    • Total Users: {total_users}
    • Users Over Quota: {users_over_quota}
    • Overage Policy: {overage_policy.upper()}
    
    User Quota Status:
    {chr(10).join(f"• {uid} ({data['tier']}): {data['used']:,}/{data['limit']:,} - {(data['used']/data['limit']*100):.1f}% used" 
                   for uid, data in user_quotas.items())}
    
    Quota Management Pattern Process:
    1. Quota Allocator → Assign quotas by tier
    2. Usage Tracker → Monitor consumption
    3. Monitor → Enforce limits and handle overages
    
    Overage Handling Policies:
    
    Hard Limit:
    • Reject requests immediately
    • HTTP 429 Too Many Requests
    • No overage allowed
    • Clear user communication
    • Most common for free tiers
    
    Soft Limit:
    • Allow temporary overage
    • Warning notifications
    • Grace period
    • Throttled performance
    • Upgrade prompts
    
    Pay-Per-Use:
    • Charge for overages
    • Automatic billing
    • No service interruption
    • Metered pricing
    • AWS Lambda model
    
    Throttling:
    • Reduce service quality
    • Lower priority
    • Slower response times
    • Degraded features
    • YouTube video quality
    
    Queuing:
    • Queue excess requests
    • Process when quota renews
    • Best-effort delivery
    • Email services
    
    Quota Reset Strategies:
    
    Fixed Schedule:
    • Reset at specific time
    • Midnight UTC common
    • Predictable behavior
    • Simple implementation
    
    Rolling Window:
    • Continuous sliding window
    • More fair allocation
    • Complex tracking
    • Better user experience
    
    On-Demand:
    • User-triggered reset
    • Subscription upgrade
    • Payment-based
    • Flexible but complex
    
    Real-World Examples:
    
    Twitter API:
    • App-based rate limits
    • User-based rate limits
    • 15-minute windows
    • Endpoint-specific limits
    • Different per tier
    
    AWS:
    • Service quotas (limits)
    • Soft limits (adjustable)
    • Hard limits (fixed)
    • Request increases
    • Automated monitoring
    
    Google Cloud:
    • Rate quotas (per 100 sec)
    • Daily quotas
    • Per-user quotas
    • Burst capacity
    • Quota increase requests
    
    SendGrid:
    • Free: 100 emails/day
    • Essentials: 50K/month
    • Pro: 1.5M/month
    • Overage: $0.0006/email
    
    Best Practices:
    
    Design:
    • Choose appropriate periods
    • Set reasonable defaults
    • Implement grace periods
    • Provide upgrade paths
    • Document clearly
    
    Communication:
    • Show quota usage in UI
    • Send warning emails
    • Provide quota API
    • Clear error messages
    • Usage dashboards
    
    Monitoring:
    • Track utilization trends
    • Alert on high usage
    • Detect abuse patterns
    • Capacity planning
    • Usage forecasting
    
    Implementation:
    • Atomic operations
    • Distributed tracking
    • Eventual consistency
    • Idempotent updates
    • Audit logging
    
    Testing:
    • Test quota enforcement
    • Test reset logic
    • Test edge cases
    • Load testing
    • Quota increase scenarios
    
    User Experience:
    • Clear quota visibility
    • Usage notifications
    • Upgrade prompts
    • Quota increase requests
    • Usage analytics
    
    Key Insight:
    Quota management ensures fair resource allocation and
    prevents abuse. Implement clear policies, communicate
    limits effectively, and provide easy upgrade paths.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Quota Monitor:\n{summary}")]
    }


# Build the graph
def build_quota_graph():
    """Build the quota management pattern graph"""
    workflow = StateGraph(QuotaState)
    
    workflow.add_node("allocator", quota_allocator)
    workflow.add_node("tracker", usage_tracker)
    workflow.add_node("monitor", quota_monitor)
    
    workflow.add_edge(START, "allocator")
    workflow.add_edge("allocator", "tracker")
    workflow.add_edge("tracker", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_quota_graph()
    
    print("=== Quota Management MCP Pattern ===\n")
    
    # Test Case: API quota management
    print("\n" + "="*70)
    print("TEST CASE: API Quota Management")
    print("="*70)
    
    state = {
        "messages": [],
        "service_name": "api_service",
        "quota_type": "api_calls",
        "quota_period": "daily",
        "user_quotas": {},
        "tier_quotas": {
            "free": 1000,
            "pro": 10000,
            "enterprise": 100000
        },
        "quota_exceeded": [],
        "overage_policy": "hard_limit"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    user_quotas = result.get('user_quotas', {})
    print(f"\nTotal Users: {len(user_quotas)}")
    print(f"Users Over Quota: {len(result.get('quota_exceeded', []))}")
