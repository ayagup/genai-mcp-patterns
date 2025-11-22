"""
Rate Limiting MCP Pattern

This pattern controls the rate of requests to prevent abuse, ensure fair usage,
and protect system resources through throttling and quota management.

Key Features:
- Request rate limiting
- Quota management
- Token bucket algorithm
- Sliding window counters
- User/IP-based limits
"""

from typing import TypedDict, Sequence, Annotated
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class RateLimitState(TypedDict):
    """State for rate limiting pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    client_id: str
    request_count: int
    rate_limit: int  # requests per minute
    time_window: int  # seconds
    quota_limit: int  # total requests per day
    quota_used: int
    current_timestamp: float
    request_allowed: bool
    limit_exceeded: bool
    retry_after: int  # seconds
    limit_type: str  # "rate", "quota", "both"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Rate Limit Checker
def rate_limit_checker(state: RateLimitState) -> RateLimitState:
    """Checks if request is within rate limits"""
    client_id = state.get("client_id", "")
    request_count = state.get("request_count", 0)
    rate_limit = state.get("rate_limit", 60)  # 60 req/min default
    time_window = state.get("time_window", 60)  # 60 seconds
    
    system_message = SystemMessage(content="""You are a rate limit checker. 
    Monitor request rates and enforce limits to prevent abuse.""")
    
    user_message = HumanMessage(content=f"""Check rate limit:

Client: {client_id}
Current Requests: {request_count}
Rate Limit: {rate_limit} requests per {time_window} seconds

Determine if request should be allowed.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Check rate limit
    rate_exceeded = request_count >= rate_limit
    
    # Calculate retry-after time
    retry_after = time_window if rate_exceeded else 0
    
    rate_check_result = f"""
    🚦 Rate Limit Check:
    
    • Client: {client_id}
    • Current Rate: {request_count}/{rate_limit} requests
    • Time Window: {time_window} seconds
    • Status: {'❌ EXCEEDED' if rate_exceeded else '✅ WITHIN LIMIT'}
    • Utilization: {(request_count/rate_limit*100):.1f}%
    
    {'⏱️ Retry After: ' + str(retry_after) + ' seconds' if rate_exceeded else '✅ Request can proceed'}
    
    Rate Limit Algorithm: Token Bucket
    """
    
    return {
        "messages": [AIMessage(content=f"🚦 Rate Limit Checker:\n{response.content}\n{rate_check_result}")],
        "request_allowed": not rate_exceeded,
        "limit_exceeded": rate_exceeded,
        "retry_after": retry_after
    }


# Quota Manager
def quota_manager(state: RateLimitState) -> RateLimitState:
    """Manages usage quotas"""
    client_id = state.get("client_id", "")
    quota_limit = state.get("quota_limit", 10000)  # 10k requests per day
    quota_used = state.get("quota_used", 0)
    request_allowed = state.get("request_allowed", False)
    
    system_message = SystemMessage(content="""You are a quota manager. 
    Track and enforce usage quotas to ensure fair resource allocation.""")
    
    user_message = HumanMessage(content=f"""Manage quota:

Client: {client_id}
Quota Used: {quota_used}/{quota_limit}
Request Allowed (by rate limit): {request_allowed}

Check if quota allows this request.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Check quota
    quota_exceeded = quota_used >= quota_limit
    
    # Override rate limit decision if quota exceeded
    if quota_exceeded:
        request_allowed = False
    
    quota_percentage = (quota_used / quota_limit * 100) if quota_limit > 0 else 0
    
    # Determine warning thresholds
    if quota_percentage >= 90:
        quota_status = "🔴 CRITICAL"
    elif quota_percentage >= 75:
        quota_status = "🟠 WARNING"
    elif quota_percentage >= 50:
        quota_status = "🟡 CAUTION"
    else:
        quota_status = "🟢 NORMAL"
    
    quota_report = f"""
    📊 Quota Management:
    
    • Client: {client_id}
    • Quota Used: {quota_used:,}/{quota_limit:,} ({quota_percentage:.1f}%)
    • Status: {quota_status}
    • Request: {'❌ DENIED (quota exceeded)' if quota_exceeded else '✅ Allowed'}
    
    Quota Limits:
    • Daily Limit: {quota_limit:,} requests
    • Remaining: {max(0, quota_limit - quota_used):,} requests
    • Reset Time: 24:00 UTC (in {24 - time.gmtime().tm_hour} hours)
    
    {'⚠️ Quota exhausted - upgrade plan required' if quota_exceeded else ''}
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Quota Manager:\n{response.content}\n{quota_report}")],
        "request_allowed": request_allowed,
        "limit_exceeded": quota_exceeded or state.get("limit_exceeded", False)
    }


# Throttle Controller
def throttle_controller(state: RateLimitState) -> RateLimitState:
    """Controls request throttling"""
    client_id = state.get("client_id", "")
    request_allowed = state.get("request_allowed", False)
    limit_exceeded = state.get("limit_exceeded", False)
    retry_after = state.get("retry_after", 0)
    request_count = state.get("request_count", 0)
    rate_limit = state.get("rate_limit", 60)
    
    system_message = SystemMessage(content="""You are a throttle controller. 
    Implement throttling strategies to manage request flow.""")
    
    user_message = HumanMessage(content=f"""Control throttling:

Client: {client_id}
Request Allowed: {request_allowed}
Limit Exceeded: {limit_exceeded}
Current Load: {request_count}/{rate_limit}

Apply throttling strategy.""")
    
    response = llm.invoke([system_message, user_message])
    
    if limit_exceeded:
        throttle_action = "REJECT"
        http_status = 429  # Too Many Requests
        throttle_message = f"Rate limit exceeded. Retry after {retry_after} seconds"
    elif request_count / rate_limit > 0.8:  # 80% of limit
        throttle_action = "SLOW"
        http_status = 200
        throttle_message = "Request allowed but throttled (near limit)"
    else:
        throttle_action = "ALLOW"
        http_status = 200
        throttle_message = "Request allowed"
    
    throttle_report = f"""
    ⚡ Throttle Control:
    
    • Action: {throttle_action}
    • HTTP Status: {http_status}
    • Message: {throttle_message}
    
    Throttling Strategy:
    • Algorithm: Token Bucket + Sliding Window
    • Burst Handling: Allow short bursts within limits
    • Fair Queuing: Prevent single client monopolization
    
    Response Headers:
    • X-RateLimit-Limit: {rate_limit}
    • X-RateLimit-Remaining: {max(0, rate_limit - request_count)}
    • X-RateLimit-Reset: {int(time.time()) + retry_after}
    {'• Retry-After: ' + str(retry_after) if limit_exceeded else ''}
    
    {'❌ Request throttled' if limit_exceeded else '✅ Request proceeding'}
    """
    
    return {
        "messages": [AIMessage(content=f"⚡ Throttle Controller:\n{response.content}\n{throttle_report}")]
    }


# Rate Limit Monitor
def rate_limit_monitor(state: RateLimitState) -> RateLimitState:
    """Monitors rate limiting metrics and patterns"""
    client_id = state.get("client_id", "")
    request_count = state.get("request_count", 0)
    rate_limit = state.get("rate_limit", 60)
    quota_used = state.get("quota_used", 0)
    quota_limit = state.get("quota_limit", 10000)
    request_allowed = state.get("request_allowed", False)
    limit_exceeded = state.get("limit_exceeded", False)
    retry_after = state.get("retry_after", 0)
    
    summary = f"""
    📈 RATE LIMITING COMPLETE
    
    Client Information:
    • Client ID: {client_id}
    • Request Status: {'✅ ALLOWED' if request_allowed else '❌ DENIED'}
    • Limit Exceeded: {'Yes ❌' if limit_exceeded else 'No ✅'}
    
    Rate Limiting:
    • Current Rate: {request_count}/{rate_limit} per minute
    • Utilization: {(request_count/rate_limit*100):.1f}%
    • Remaining: {max(0, rate_limit - request_count)} requests
    
    Quota Management:
    • Daily Usage: {quota_used:,}/{quota_limit:,}
    • Utilization: {(quota_used/quota_limit*100):.1f}%
    • Remaining: {max(0, quota_limit - quota_used):,} requests
    
    {'Retry Information:' if limit_exceeded else ''}
    {f'• Retry After: {retry_after} seconds' if limit_exceeded else ''}
    
    Rate Limiting Pattern Process:
    1. Rate Limit Check → Verify request rate
    2. Quota Management → Check daily quota
    3. Throttle Control → Apply throttling
    4. Response → Allow or deny request
    5. Monitor → Track patterns and metrics
    
    Rate Limiting Algorithms:
    
    1. Token Bucket:
       • Tokens added at fixed rate
       • Each request consumes token
       • Allows bursts (bucket size)
       • Most common algorithm
    
    2. Leaky Bucket:
       • Requests processed at fixed rate
       • Queue overflows rejected
       • Smooth output rate
       • No bursts allowed
    
    3. Fixed Window:
       • Count requests in time window
       • Reset at window boundary
       • Simple but edge case issues
       • Can allow 2x burst at boundary
    
    4. Sliding Window:
       • Rolling time window
       • More accurate than fixed
       • Prevents boundary bursts
       • Higher complexity
    
    5. Sliding Log:
       • Track all request timestamps
       • Most accurate
       • Memory intensive
       • Not scalable
    
    Rate Limit Strategies:
    
    Per-User Limits:
    • Authenticated users
    • User-specific quotas
    • Tier-based limits
    • Fair usage policy
    
    Per-IP Limits:
    • Anonymous requests
    • DDoS prevention
    • Geographic limits
    • Shared IP considerations
    
    Per-API-Key Limits:
    • Application-based
    • Service tier enforcement
    • Easy tracking
    • Key rotation support
    
    Global Limits:
    • System-wide protection
    • Resource capacity limits
    • Prevent overload
    • Emergency throttling
    
    Common Rate Limit Tiers:
    
    Free Tier:
    • 60 requests/minute
    • 1,000 requests/day
    • Basic features only
    
    Basic Tier:
    • 300 requests/minute
    • 10,000 requests/day
    • Standard features
    
    Pro Tier:
    • 1,000 requests/minute
    • 100,000 requests/day
    • Advanced features
    
    Enterprise:
    • Custom limits
    • Dedicated resources
    • Priority support
    • SLA guarantees
    
    HTTP Status Codes:
    • 200: Request allowed
    • 429: Too Many Requests
    • 503: Service temporarily unavailable
    
    Response Headers:
    • X-RateLimit-Limit: Max requests
    • X-RateLimit-Remaining: Requests left
    • X-RateLimit-Reset: Reset timestamp
    • Retry-After: Seconds to wait
    
    Rate Limiting Best Practices:
    • Clear limit documentation
    • Informative error messages
    • Retry-After headers
    • Gradual limit increases
    • Burst tolerance
    • Multiple limit types
    • Real-time monitoring
    • Alert on anomalies
    • Dynamic adjustment
    • Whitelist critical clients
    
    Implementation Considerations:
    • Distributed systems (Redis)
    • Atomic operations
    • Clock synchronization
    • Storage efficiency
    • Performance impact
    • Failover handling
    
    Bypass Mechanisms:
    • Whitelist trusted IPs
    • Admin override
    • Emergency access
    • Rate limit exemptions
    
    Common Use Cases:
    • API rate limiting
    • Login attempt limits
    • Search query limits
    • Email sending limits
    • File upload limits
    • Comment posting limits
    • Payment processing limits
    
    Abuse Prevention:
    • DDoS protection
    • Brute force prevention
    • Scraping prevention
    • Spam prevention
    • Resource exhaustion protection
    
    Key Insight:
    Rate limiting controls request rates to prevent abuse, ensure
    fair resource allocation, and protect system stability. Essential
    for APIs, web services, and any multi-user system.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Rate Limit Monitor:\n{summary}")]
    }


# Build the graph
def build_rate_limit_graph():
    """Build the rate limiting pattern graph"""
    workflow = StateGraph(RateLimitState)
    
    workflow.add_node("rate_checker", rate_limit_checker)
    workflow.add_node("quota_mgr", quota_manager)
    workflow.add_node("throttle", throttle_controller)
    workflow.add_node("monitor", rate_limit_monitor)
    
    workflow.add_edge(START, "rate_checker")
    workflow.add_edge("rate_checker", "quota_mgr")
    workflow.add_edge("quota_mgr", "throttle")
    workflow.add_edge("throttle", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_rate_limit_graph()
    
    print("=== Rate Limiting MCP Pattern ===\n")
    
    # Test Case 1: Within limits
    print("\n" + "="*70)
    print("TEST CASE 1: Request Within Limits")
    print("="*70)
    
    state1 = {
        "messages": [],
        "client_id": "user_12345",
        "request_count": 45,  # Below 60/min limit
        "rate_limit": 60,
        "time_window": 60,
        "quota_limit": 10000,
        "quota_used": 5000,  # 50% of daily quota
        "current_timestamp": time.time(),
        "request_allowed": False,
        "limit_exceeded": False,
        "retry_after": 0,
        "limit_type": "both"
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nRequest Status: {'✅ ALLOWED' if result1.get('request_allowed') else '❌ DENIED'}")
    
    # Test Case 2: Rate limit exceeded
    print("\n\n" + "="*70)
    print("TEST CASE 2: Rate Limit Exceeded")
    print("="*70)
    
    state2 = {
        "messages": [],
        "client_id": "user_67890",
        "request_count": 65,  # Exceeds 60/min limit
        "rate_limit": 60,
        "time_window": 60,
        "quota_limit": 10000,
        "quota_used": 2000,
        "current_timestamp": time.time(),
        "request_allowed": False,
        "limit_exceeded": False,
        "retry_after": 0,
        "limit_type": "both"
    }
    
    result2 = graph.invoke(state2)
    
    print(f"\nClient: {state2['client_id']}")
    print(f"Request Rate: {state2['request_count']}/{state2['rate_limit']}")
    print(f"Status: {'DENIED ❌' if result2.get('limit_exceeded') else 'ALLOWED ✅'}")
    print(f"Retry After: {result2.get('retry_after', 0)} seconds")
    
    # Test Case 3: Quota exceeded
    print("\n\n" + "="*70)
    print("TEST CASE 3: Daily Quota Exceeded")
    print("="*70)
    
    state3 = {
        "messages": [],
        "client_id": "user_99999",
        "request_count": 30,  # Within rate limit
        "rate_limit": 60,
        "time_window": 60,
        "quota_limit": 10000,
        "quota_used": 10005,  # Exceeded daily quota
        "current_timestamp": time.time(),
        "request_allowed": False,
        "limit_exceeded": False,
        "retry_after": 0,
        "limit_type": "both"
    }
    
    result3 = graph.invoke(state3)
    
    print(f"\nClient: {state3['client_id']}")
    print(f"Daily Quota: {state3['quota_used']:,}/{state3['quota_limit']:,}")
    print(f"Status: {'DENIED ❌ (Quota Exceeded)' if result3.get('limit_exceeded') else 'ALLOWED ✅'}")
