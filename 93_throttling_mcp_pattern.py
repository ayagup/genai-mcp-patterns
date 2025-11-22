"""
Throttling MCP Pattern

This pattern controls the rate of operations to prevent system overload
and ensure fair resource usage across multiple clients.

Key Features:
- Request rate limiting
- Traffic shaping
- Burst handling
- Adaptive throttling
- Fair resource allocation
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ThrottlingState(TypedDict):
    """State for throttling pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    service_name: str
    throttle_strategy: str  # "token_bucket", "leaky_bucket", "fixed_window", "sliding_window"
    rate_limit: int  # requests per time unit
    time_unit: str  # "second", "minute", "hour"
    burst_size: int  # max burst capacity
    current_tokens: int
    requests_allowed: int
    requests_throttled: int
    client_limits: Dict[str, int]  # per-client limits
    adaptive_enabled: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Throttle Manager
def throttle_manager(state: ThrottlingState) -> ThrottlingState:
    """Manages throttling configuration and strategy"""
    service_name = state.get("service_name", "")
    throttle_strategy = state.get("throttle_strategy", "token_bucket")
    rate_limit = state.get("rate_limit", 100)
    time_unit = state.get("time_unit", "second")
    burst_size = state.get("burst_size", 10)
    
    system_message = SystemMessage(content="""You are a throttle manager.
    Configure and manage rate limiting strategies.""")
    
    user_message = HumanMessage(content=f"""Configure throttling:

Service: {service_name}
Strategy: {throttle_strategy}
Rate Limit: {rate_limit}/{time_unit}
Burst Size: {burst_size}

Set up throttling mechanism.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Initialize tokens based on strategy
    current_tokens = burst_size if throttle_strategy == "token_bucket" else 0
    
    report = f"""
    🚦 Throttle Management:
    
    Configuration:
    • Service: {service_name}
    • Strategy: {throttle_strategy.upper()}
    • Rate Limit: {rate_limit} requests/{time_unit}
    • Burst Capacity: {burst_size}
    • Current Tokens: {current_tokens}
    
    Throttling Strategies:
    
    1. Token Bucket:
       • Tokens added at fixed rate
       • Burst traffic supported
       • Tokens consumed per request
       • Most flexible strategy
       • Example: AWS API Gateway
    
    2. Leaky Bucket:
       • Fixed output rate
       • Queue requests
       • Smooth traffic flow
       • No burst support
       • Example: Network traffic shaping
    
    3. Fixed Window:
       • Counter resets at intervals
       • Simple implementation
       • Burst at boundaries
       • Example: GitHub API (5000/hour)
    
    4. Sliding Window:
       • Rolling time window
       • More accurate than fixed
       • Higher computation cost
       • Example: Twitter API v2
    
    When to Use Each:
    
    Token Bucket:
    ✓ Need burst handling
    ✓ Variable request sizes
    ✓ Most APIs and services
    ✓ Cloud service throttling
    
    Leaky Bucket:
    ✓ Smooth output required
    ✓ Network traffic control
    ✓ Video streaming
    ✓ Queue-based systems
    
    Fixed Window:
    ✓ Simple requirements
    ✓ Hour/day quotas
    ✓ Low traffic services
    ✓ Billing periods
    
    Sliding Window:
    ✓ Precise rate limiting
    ✓ High traffic services
    ✓ SLA enforcement
    ✓ DDoS protection
    
    Industry Examples:
    
    AWS:
    • API Gateway: Token bucket
    • DynamoDB: Adaptive throttling
    • Lambda: Concurrent execution limits
    • CloudWatch: API throttling
    
    Google Cloud:
    • Rate limiting per 100 seconds
    • Per-user rate limits
    • Burst quotas
    • Daily quotas
    
    GitHub API:
    • 5,000 requests/hour (authenticated)
    • 60 requests/hour (unauthenticated)
    • Separate GraphQL limits
    • Conditional requests optimization
    
    Twitter API:
    • App rate limit
    • User rate limit
    • Endpoint-specific limits
    • 15-minute windows
    """
    
    return {
        "messages": [AIMessage(content=f"🚦 Throttle Manager:\n{response.content}\n{report}")],
        "current_tokens": current_tokens
    }


# Rate Limiter
def rate_limiter(state: ThrottlingState) -> ThrottlingState:
    """Enforces rate limits on incoming requests"""
    throttle_strategy = state.get("throttle_strategy", "token_bucket")
    rate_limit = state.get("rate_limit", 100)
    current_tokens = state.get("current_tokens", 0)
    burst_size = state.get("burst_size", 10)
    
    system_message = SystemMessage(content="""You are a rate limiter.
    Enforce rate limits and manage request throughput.""")
    
    # Simulate incoming requests
    incoming_requests = 15
    
    user_message = HumanMessage(content=f"""Process requests with rate limiting:

Strategy: {throttle_strategy}
Rate Limit: {rate_limit}
Current Tokens: {current_tokens}
Burst Size: {burst_size}
Incoming Requests: {incoming_requests}

Apply rate limiting.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate rate limiting
    requests_allowed = min(incoming_requests, current_tokens)
    requests_throttled = incoming_requests - requests_allowed
    current_tokens = max(0, current_tokens - requests_allowed)
    
    report = f"""
    ⚡ Rate Limiting:
    
    Request Processing:
    • Incoming Requests: {incoming_requests}
    • Requests Allowed: {requests_allowed}
    • Requests Throttled: {requests_throttled}
    • Remaining Tokens: {current_tokens}
    • Throttle Rate: {(requests_throttled/incoming_requests*100):.1f}%
    
    Rate Limiting Algorithms:
    
    Token Bucket Implementation:
    ```
    class TokenBucket:
        def __init__(self, rate, capacity):
            self.rate = rate          # tokens/second
            self.capacity = capacity  # max tokens
            self.tokens = capacity
            self.last_update = time()
        
        def allow_request(self, tokens=1):
            self.refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
        
        def refill(self):
            now = time()
            elapsed = now - self.last_update
            new_tokens = elapsed * self.rate
            self.tokens = min(
                self.capacity,
                self.tokens + new_tokens
            )
            self.last_update = now
    ```
    
    Sliding Window Log:
    ```
    class SlidingWindowLog:
        def __init__(self, rate, window):
            self.rate = rate
            self.window = window  # seconds
            self.requests = []
        
        def allow_request(self):
            now = time()
            cutoff = now - self.window
            
            # Remove old requests
            self.requests = [
                t for t in self.requests
                if t > cutoff
            ]
            
            if len(self.requests) < self.rate:
                self.requests.append(now)
                return True
            return False
    ```
    
    Fixed Window Counter:
    ```
    class FixedWindow:
        def __init__(self, rate, window):
            self.rate = rate
            self.window = window
            self.counter = 0
            self.window_start = time()
        
        def allow_request(self):
            now = time()
            
            # Reset window if expired
            if now - self.window_start >= self.window:
                self.counter = 0
                self.window_start = now
            
            if self.counter < self.rate:
                self.counter += 1
                return True
            return False
    ```
    
    Response Headers:
    • X-RateLimit-Limit: {rate_limit}
    • X-RateLimit-Remaining: {current_tokens}
    • X-RateLimit-Reset: {int(time.time() + 60)}
    • Retry-After: {60 if requests_throttled > 0 else 0}
    
    HTTP Status Codes:
    • 200 OK: Request allowed
    • 429 Too Many Requests: Throttled
    • 503 Service Unavailable: Overload
    
    Client Best Practices:
    • Respect rate limit headers
    • Implement exponential backoff
    • Use Retry-After header
    • Batch requests when possible
    • Cache responses
    • Use webhooks vs polling
    """
    
    return {
        "messages": [AIMessage(content=f"⚡ Rate Limiter:\n{response.content}\n{report}")],
        "current_tokens": current_tokens,
        "requests_allowed": requests_allowed,
        "requests_throttled": requests_throttled
    }


# Adaptive Controller
def adaptive_controller(state: ThrottlingState) -> ThrottlingState:
    """Adjusts throttling based on system load"""
    adaptive_enabled = state.get("adaptive_enabled", True)
    rate_limit = state.get("rate_limit", 100)
    requests_throttled = state.get("requests_throttled", 0)
    
    system_message = SystemMessage(content="""You are an adaptive throttling controller.
    Adjust rate limits dynamically based on system conditions.""")
    
    user_message = HumanMessage(content=f"""Adaptive throttling:

Enabled: {adaptive_enabled}
Base Rate: {rate_limit}
Throttled: {requests_throttled}

Adjust throttling based on load.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate adaptive adjustment
    if adaptive_enabled and requests_throttled > 0:
        adjustment = "Decrease rate (high load detected)"
        new_rate = int(rate_limit * 0.8)
    else:
        adjustment = "Maintain rate (system healthy)"
        new_rate = rate_limit
    
    report = f"""
    🎯 Adaptive Throttling:
    
    System Analysis:
    • Adaptive Mode: {'Enabled ✅' if adaptive_enabled else 'Disabled'}
    • Current Rate: {rate_limit}
    • Suggested Rate: {new_rate}
    • Adjustment: {adjustment}
    • Throttled Requests: {requests_throttled}
    
    Adaptive Strategies:
    
    1. Load-Based:
       • Monitor CPU/Memory
       • Adjust limits dynamically
       • Prevent overload
       • Gradual rate changes
       • DynamoDB approach
    
    2. Response Time:
       • Track latency metrics
       • Reduce rate if slow
       • Maintain SLA targets
       • P95/P99 monitoring
    
    3. Error Rate:
       • Monitor error percentage
       • Throttle on high errors
       • Circuit breaker integration
       • Automatic recovery
    
    4. Queue Depth:
       • Track pending requests
       • Limit queue growth
       • Prevent memory issues
       • Backpressure signal
    
    Adaptive Algorithms:
    
    AIMD (Additive Increase, Multiplicative Decrease):
    • Increase: rate = rate + 1 (gradual)
    • Decrease: rate = rate * 0.5 (aggressive)
    • TCP congestion control style
    • Stable convergence
    
    PID Controller:
    • Proportional-Integral-Derivative
    • Smooth rate adjustments
    • Minimize oscillation
    • Engineering control theory
    
    Machine Learning:
    • Predict optimal rate
    • Historical pattern analysis
    • Anomaly detection
    • Seasonal adjustments
    
    Implementation Example:
    ```python
    class AdaptiveThrottle:
        def __init__(self, base_rate):
            self.base_rate = base_rate
            self.current_rate = base_rate
            self.min_rate = base_rate * 0.1
            self.max_rate = base_rate * 2.0
        
        def adjust(self, metrics):
            cpu = metrics['cpu_percent']
            latency = metrics['p95_latency_ms']
            error_rate = metrics['error_rate']
            
            # Decrease if overloaded
            if cpu > 80 or latency > 1000 or error_rate > 0.05:
                self.current_rate *= 0.9
            # Increase if underutilized
            elif cpu < 50 and latency < 100 and error_rate < 0.01:
                self.current_rate *= 1.1
            
            # Clamp to limits
            self.current_rate = max(
                self.min_rate,
                min(self.max_rate, self.current_rate)
            )
            
            return self.current_rate
    ```
    
    Monitoring Metrics:
    • Throttle rate
    • Request acceptance rate
    • System resource utilization
    • Response time distribution
    • Error rate trends
    • Queue length
    
    Auto-Scaling Integration:
    • Scale out when throttling high
    • Scale in when capacity excess
    • Coordinate with load balancer
    • Health check integration
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Adaptive Controller:\n{response.content}\n{report}")]
    }


# Throttle Monitor
def throttle_monitor(state: ThrottlingState) -> ThrottlingState:
    """Monitors throttling metrics and provides insights"""
    service_name = state.get("service_name", "")
    throttle_strategy = state.get("throttle_strategy", "")
    rate_limit = state.get("rate_limit", 0)
    requests_allowed = state.get("requests_allowed", 0)
    requests_throttled = state.get("requests_throttled", 0)
    client_limits = state.get("client_limits", {})
    
    total_requests = requests_allowed + requests_throttled
    throttle_percentage = (requests_throttled / total_requests * 100) if total_requests > 0 else 0
    
    summary = f"""
    📊 THROTTLING COMPLETE
    
    Service Status:
    • Service: {service_name}
    • Strategy: {throttle_strategy.upper()}
    • Rate Limit: {rate_limit}
    • Requests Allowed: {requests_allowed}
    • Requests Throttled: {requests_throttled}
    • Throttle Rate: {throttle_percentage:.1f}%
    
    Throttling Pattern Process:
    1. Throttle Manager → Configure strategy
    2. Rate Limiter → Enforce limits
    3. Adaptive Controller → Dynamic adjustments
    4. Monitor → Track performance
    
    Common Throttling Use Cases:
    
    API Rate Limiting:
    • Prevent abuse
    • Fair usage enforcement
    • Cost control
    • Infrastructure protection
    • SLA compliance
    
    DDoS Protection:
    • Block attack traffic
    • Preserve service availability
    • Geographic rate limits
    • IP-based throttling
    • Challenge-response (CAPTCHA)
    
    Resource Protection:
    • Database connection limits
    • CPU/Memory protection
    • Disk I/O throttling
    • Network bandwidth control
    • Thread pool limits
    
    Cost Optimization:
    • Cloud API call limits
    • Third-party API costs
    • Bandwidth costs
    • Compute costs
    • Storage operations
    
    Multi-Tier Throttling:
    
    Global Limits:
    • Overall service capacity
    • Infrastructure limits
    • Total throughput
    
    Per-User Limits:
    • Fair resource sharing
    • Prevent single user monopoly
    • Tier-based limits (free/paid)
    
    Per-Endpoint Limits:
    • Expensive operations
    • Different resource costs
    • Granular control
    
    Geographic Limits:
    • Region-specific capacity
    • Compliance requirements
    • Network proximity
    
    Real-World Examples:
    
    Stripe API:
    • Default: 100 req/sec
    • Burst: Up to 150 req/sec
    • Per-endpoint limits
    • Webhook retry with backoff
    
    AWS Lambda:
    • Concurrent executions: 1000 (default)
    • Burst capacity: 3000
    • Per-region limits
    • Reserved concurrency
    
    Cloudflare:
    • 1200 req/5min (Free)
    • 2400 req/5min (Pro)
    • Enterprise: Custom
    • Rate Limiting Rules
    
    Redis:
    • Command throttling
    • Slow log tracking
    • Client output buffer limits
    • Memory-based eviction
    
    Best Practices:
    
    Design:
    • Choose appropriate strategy
    • Set reasonable defaults
    • Document limits clearly
    • Provide quota endpoints
    • Version your limits
    
    Implementation:
    • Use proven libraries
    • Distributed rate limiting (Redis)
    • Atomic operations
    • Accurate timestamps
    • Handle clock skew
    
    Client Communication:
    • Clear error messages
    • Rate limit headers
    • Retry-After header
    • Documentation
    • Status page
    
    Monitoring:
    • Track throttle rates
    • Alert on high throttling
    • Client-specific metrics
    • Trend analysis
    • Capacity planning
    
    Testing:
    • Load testing with limits
    • Burst traffic scenarios
    • Client retry behavior
    • Distributed coordination
    • Clock edge cases
    
    Key Insight:
    Throttling protects your service from overload while
    ensuring fair resource allocation. Choose the right
    strategy for your use case and monitor continuously.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Throttle Monitor:\n{summary}")]
    }


# Build the graph
def build_throttling_graph():
    """Build the throttling pattern graph"""
    workflow = StateGraph(ThrottlingState)
    
    workflow.add_node("throttle_mgr", throttle_manager)
    workflow.add_node("rate_limiter", rate_limiter)
    workflow.add_node("adaptive", adaptive_controller)
    workflow.add_node("monitor", throttle_monitor)
    
    workflow.add_edge(START, "throttle_mgr")
    workflow.add_edge("throttle_mgr", "rate_limiter")
    workflow.add_edge("rate_limiter", "adaptive")
    workflow.add_edge("adaptive", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_throttling_graph()
    
    print("=== Throttling MCP Pattern ===\n")
    
    # Test Case: API rate limiting with token bucket
    print("\n" + "="*70)
    print("TEST CASE: API Rate Limiting")
    print("="*70)
    
    state = {
        "messages": [],
        "service_name": "user_api",
        "throttle_strategy": "token_bucket",
        "rate_limit": 100,
        "time_unit": "second",
        "burst_size": 10,
        "current_tokens": 0,
        "requests_allowed": 0,
        "requests_throttled": 0,
        "client_limits": {"client_1": 50, "client_2": 30},
        "adaptive_enabled": True
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nRequests Allowed: {result.get('requests_allowed', 0)}")
    print(f"Requests Throttled: {result.get('requests_throttled', 0)}")
    total = result.get('requests_allowed', 0) + result.get('requests_throttled', 0)
    if total > 0:
        print(f"Throttle Rate: {(result.get('requests_throttled', 0)/total*100):.1f}%")
