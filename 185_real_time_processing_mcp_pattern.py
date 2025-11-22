"""
Pattern 185: Real-Time Processing MCP Pattern

This pattern demonstrates processing data with minimal latency, providing immediate
responses and enabling time-sensitive decision making. Real-time processing is critical
for applications requiring instant feedback and low-latency operations.

Key Concepts:
1. Low Latency: Millisecond to sub-second response times
2. Immediate Processing: Process data as soon as it arrives
3. In-Memory Computation: Minimize disk I/O for speed
4. Event-Driven: React to events instantly
5. Streaming: Continuous data flow processing
6. Hot Path: Optimized code path for speed
7. Time-Sensitive: Results must be delivered quickly

Real-Time Characteristics:
- Ultra-Low Latency: < 100ms typical, < 1s maximum
- Continuous Availability: 24/7 processing
- In-Memory: Data processed in RAM (fast)
- Immediate Feedback: Results returned immediately
- High Concurrency: Handle many simultaneous requests
- Predictable Performance: Consistent latency

Real-Time vs Near Real-Time:
- Real-Time: < 100ms (hard requirement)
- Near Real-Time: < 1 second (soft requirement)
- Mini-Batch: 1-10 seconds (micro-batching)
- Batch: Minutes to hours (periodic)

Real-Time Patterns:
1. Request-Response: Synchronous immediate response
2. Event Processing: Asynchronous event handling
3. Stream Processing: Continuous flow processing
4. Complex Event Processing (CEP): Pattern detection

Benefits:
- Instant Insights: Immediate data availability
- Better UX: Fast response improves user experience
- Timely Actions: Enable time-sensitive decisions
- Competitive Advantage: React faster than competitors
- Operational Efficiency: Immediate problem detection

Trade-offs:
- Complexity: More complex than batch processing
- Cost: Requires always-on infrastructure
- Resource Usage: Constant resource consumption
- Limited Processing: Less time for complex operations
- Harder Testing: Timing-dependent behavior

Use Cases:
- Trading Systems: Execute trades in milliseconds
- Fraud Detection: Block suspicious transactions immediately
- Gaming: Real-time player interactions
- Monitoring/Alerting: Instant problem detection
- Recommendations: Show relevant suggestions immediately
- Chat Applications: Instant message delivery
- IoT: Real-time sensor data processing
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import time
from collections import deque

# Define the state for real-time processing
class RealTimeProcessingState(TypedDict):
    """State for real-time processing"""
    request: Dict[str, Any]
    processing_start: Optional[str]
    cache_result: Optional[Dict[str, Any]]
    processed_result: Optional[Dict[str, Any]]
    response: Optional[Dict[str, Any]]
    latency_metrics: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# REAL-TIME PROCESSING COMPONENTS
# ============================================================================

class LatencyRequirement(Enum):
    """Latency requirements for real-time systems"""
    HARD_REAL_TIME = 0.001  # 1ms - trading systems
    ULTRA_LOW = 0.010       # 10ms - gaming, HFT
    LOW = 0.100             # 100ms - user interfaces
    SOFT_REAL_TIME = 1.0    # 1s - monitoring, alerts

@dataclass
class LatencyMetrics:
    """Track latency metrics"""
    operation: str
    start_time: datetime
    end_time: datetime
    
    def get_latency_ms(self) -> float:
        """Get latency in milliseconds"""
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    def meets_sla(self, sla_ms: float) -> bool:
        """Check if meets SLA"""
        return self.get_latency_ms() <= sla_ms

class InMemoryCache:
    """
    In-Memory Cache: Ultra-fast data access
    
    Real-time systems use in-memory storage for speed:
    - RAM access: ~100 nanoseconds
    - SSD access: ~100 microseconds
    - HDD access: ~10 milliseconds
    
    Cache Hit: Instant response (< 1ms)
    Cache Miss: Need to fetch from source (slower)
    """
    
    def __init__(self, ttl_seconds: int = 60):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if expired
            if datetime.now() - entry["timestamp"] < self.ttl:
                return entry["value"]
            else:
                # Expired, remove
                del self.cache[key]
        
        return None
    
    def put(self, key: str, value: Dict[str, Any]):
        """Put value in cache"""
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now()
        }
    
    def invalidate(self, key: str):
        """Invalidate cache entry"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()

class HotPathProcessor:
    """
    Hot Path: Optimized code path for speed
    
    Optimization Techniques:
    - In-memory data structures
    - Minimal I/O operations
    - Pre-computed results
    - Connection pooling
    - Circuit breakers for fast failures
    """
    
    def __init__(self):
        self.cache = InMemoryCache(ttl_seconds=60)
        self.processing_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0
        }
    
    def process_with_cache(self, request_key: str, 
                          compute_fn) -> Dict[str, Any]:
        """
        Process with caching (hot path optimization)
        
        Fast Path: Cache hit (< 1ms)
        Slow Path: Cache miss, compute result (10-100ms)
        """
        start_time = datetime.now()
        
        # Try cache first (hot path)
        cached = self.cache.get(request_key)
        
        if cached is not None:
            # Cache hit - instant response
            self.processing_stats["cache_hits"] += 1
            self.processing_stats["total_requests"] += 1
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                "result": cached,
                "source": "cache",
                "latency_ms": latency_ms,
                "cache_hit": True
            }
        
        # Cache miss - compute result (cold path)
        self.processing_stats["cache_misses"] += 1
        self.processing_stats["total_requests"] += 1
        
        result = compute_fn()
        
        # Store in cache for future requests
        self.cache.put(request_key, result)
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            "result": result,
            "source": "computed",
            "latency_ms": latency_ms,
            "cache_hit": False
        }
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.processing_stats["total_requests"]
        if total == 0:
            return 0.0
        
        hits = self.processing_stats["cache_hits"]
        return (hits / total) * 100

class RealTimeAnalyzer:
    """
    Real-Time Analyzer: Immediate data analysis
    
    Features:
    - Streaming analytics
    - Sliding windows
    - Anomaly detection
    - Instant aggregations
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
    
    def analyze_realtime(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data point in real-time
        
        Real-time analysis over sliding window
        """
        # Add to sliding window
        self.data_window.append(data_point)
        
        # Compute real-time statistics
        values = [d.get("value", 0) for d in self.data_window]
        
        if not values:
            return {"error": "No data"}
        
        avg = sum(values) / len(values)
        current_value = data_point.get("value", 0)
        
        # Anomaly detection (simple threshold)
        is_anomaly = abs(current_value - avg) > (avg * 0.5)  # 50% deviation
        
        return {
            "current_value": current_value,
            "window_avg": avg,
            "window_size": len(self.data_window),
            "is_anomaly": is_anomaly,
            "anomaly_score": abs(current_value - avg) / avg if avg != 0 else 0
        }

class CircuitBreaker:
    """
    Circuit Breaker: Fast failure for real-time systems
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing fast (immediate error)
    - HALF_OPEN: Testing if recovered
    
    Prevents cascading failures in real-time systems
    """
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            # Check if timeout expired
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                # Fail fast - don't even try
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if in half-open state
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            # Open circuit if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

# Create instances
hot_path = HotPathProcessor()
analyzer = RealTimeAnalyzer(window_size=100)
circuit_breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=30)

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def receive_request(state: RealTimeProcessingState) -> RealTimeProcessingState:
    """Receive real-time request"""
    request = state["request"]
    
    return {
        "processing_start": datetime.now().isoformat(),
        "messages": [f"[Receive] Request received: {request.get('type', 'unknown')}"]
    }

def check_cache(state: RealTimeProcessingState) -> RealTimeProcessingState:
    """Check cache for instant response (hot path)"""
    request = state["request"]
    start_time = datetime.now()
    
    request_key = f"{request.get('type', 'unknown')}_{request.get('id', 'default')}"
    
    # Try cache
    cached = hot_path.cache.get(request_key)
    
    if cached:
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            "cache_result": {
                "hit": True,
                "data": cached,
                "latency_ms": latency_ms
            },
            "messages": [f"[Cache] HIT - Instant response ({latency_ms:.2f}ms)"]
        }
    else:
        return {
            "cache_result": {
                "hit": False,
                "latency_ms": 0
            },
            "messages": ["[Cache] MISS - Computing result"]
        }

def process_realtime(state: RealTimeProcessingState) -> RealTimeProcessingState:
    """Process request in real-time"""
    request = state["request"]
    cache_result = state.get("cache_result", {})
    start_time = datetime.now()
    
    # If cache hit, use cached result
    if cache_result.get("hit"):
        result = cache_result["data"]
    else:
        # Compute result (fast processing)
        result = {
            "request_id": request.get("id"),
            "type": request.get("type"),
            "value": request.get("value", 0) * 2,  # Simple transformation
            "processed_at": datetime.now().isoformat()
        }
        
        # Store in cache
        request_key = f"{request.get('type', 'unknown')}_{request.get('id', 'default')}"
        hot_path.cache.put(request_key, result)
    
    end_time = datetime.now()
    latency_ms = (end_time - start_time).total_seconds() * 1000
    
    return {
        "processed_result": result,
        "messages": [f"[Process] Processed in {latency_ms:.2f}ms"]
    }

def build_response(state: RealTimeProcessingState) -> RealTimeProcessingState:
    """Build real-time response"""
    processing_start = datetime.fromisoformat(state["processing_start"])
    processed_result = state.get("processed_result", {})
    cache_result = state.get("cache_result", {})
    
    # Calculate total latency
    end_time = datetime.now()
    total_latency_ms = (end_time - processing_start).total_seconds() * 1000
    
    response = {
        "status": "success",
        "data": processed_result,
        "metadata": {
            "cache_hit": cache_result.get("hit", False),
            "total_latency_ms": total_latency_ms,
            "timestamp": end_time.isoformat(),
            "sla_met": total_latency_ms < 100  # 100ms SLA
        }
    }
    
    return {
        "response": response,
        "latency_metrics": {
            "total_latency_ms": total_latency_ms,
            "cache_hit": cache_result.get("hit", False),
            "sla_met": total_latency_ms < 100
        },
        "messages": [f"[Response] Total latency: {total_latency_ms:.2f}ms"]
    }

# ============================================================================
# BUILD THE REAL-TIME PROCESSING GRAPH
# ============================================================================

def create_realtime_processing_graph():
    """
    Create a StateGraph demonstrating real-time processing pattern.
    
    Flow (optimized for minimal latency):
    1. Receive: Accept request
    2. Cache Check: Try hot path (instant response)
    3. Process: Compute if cache miss
    4. Response: Return result with latency metrics
    """
    
    workflow = StateGraph(RealTimeProcessingState)
    
    # Add real-time processing nodes
    workflow.add_node("receive", receive_request)
    workflow.add_node("cache", check_cache)
    workflow.add_node("process", process_realtime)
    workflow.add_node("respond", build_response)
    
    # Define real-time flow
    workflow.add_edge(START, "receive")
    workflow.add_edge("receive", "cache")
    workflow.add_edge("cache", "process")
    workflow.add_edge("process", "respond")
    workflow.add_edge("respond", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Real-Time Processing MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Real-Time Processing with Caching
    print("\n" + "=" * 80)
    print("Example 1: Real-Time Request Processing")
    print("=" * 80)
    
    realtime_graph = create_realtime_processing_graph()
    
    # First request (cache miss)
    print("\nFirst Request (Cache Miss):")
    initial_state: RealTimeProcessingState = {
        "request": {"id": "req_1", "type": "get_data", "value": 100},
        "processing_start": None,
        "cache_result": None,
        "processed_result": None,
        "response": None,
        "latency_metrics": {},
        "messages": []
    }
    
    result1 = realtime_graph.invoke(initial_state)
    
    for msg in result1["messages"]:
        print(f"  {msg}")
    
    print(f"\nLatency Metrics:")
    print(f"  Total Latency: {result1['latency_metrics']['total_latency_ms']:.2f}ms")
    print(f"  Cache Hit: {result1['latency_metrics']['cache_hit']}")
    print(f"  SLA Met (< 100ms): {result1['latency_metrics']['sla_met']}")
    
    # Second request (cache hit)
    print("\nSecond Request (Cache Hit - Instant!):")
    result2 = realtime_graph.invoke(initial_state)
    
    for msg in result2["messages"]:
        print(f"  {msg}")
    
    print(f"\nLatency Metrics:")
    print(f"  Total Latency: {result2['latency_metrics']['total_latency_ms']:.2f}ms")
    print(f"  Cache Hit: {result2['latency_metrics']['cache_hit']}")
    print(f"  SLA Met (< 100ms): {result2['latency_metrics']['sla_met']}")
    
    print(f"\nCache Hit Rate: {hot_path.get_cache_hit_rate():.1f}%")
    
    # Example 2: Real-Time Analytics
    print("\n" + "=" * 80)
    print("Example 2: Real-Time Streaming Analytics")
    print("=" * 80)
    
    # Simulate real-time data stream
    data_stream = [
        {"timestamp": "2024-01-01T10:00:00", "value": 100},
        {"timestamp": "2024-01-01T10:00:01", "value": 105},
        {"timestamp": "2024-01-01T10:00:02", "value": 102},
        {"timestamp": "2024-01-01T10:00:03", "value": 150},  # Anomaly
        {"timestamp": "2024-01-01T10:00:04", "value": 103},
    ]
    
    print("\nReal-Time Analysis Results:")
    for data_point in data_stream:
        analysis = analyzer.analyze_realtime(data_point)
        
        anomaly_marker = " ⚠️  ANOMALY!" if analysis["is_anomaly"] else ""
        print(f"  Value: {analysis['current_value']}, "
              f"Window Avg: {analysis['window_avg']:.1f}, "
              f"Anomaly Score: {analysis['anomaly_score']:.2f}{anomaly_marker}")
    
    # Example 3: Latency Requirements
    print("\n" + "=" * 80)
    print("Example 3: Latency Requirements by Use Case")
    print("=" * 80)
    
    print("\nHard Real-Time (< 1ms):")
    print("  • High-Frequency Trading (HFT)")
    print("  • Industrial Control Systems")
    print("  • Medical Devices")
    
    print("\nUltra-Low Latency (< 10ms):")
    print("  • Online Gaming")
    print("  • Video Conferencing")
    print("  • Autonomous Vehicles")
    
    print("\nLow Latency (< 100ms):")
    print("  • Web Applications")
    print("  • Mobile Apps")
    print("  • APIs")
    print("  • Search Engines")
    
    print("\nSoft Real-Time (< 1s):")
    print("  • Monitoring Systems")
    print("  • Alerting")
    print("  • Real-Time Dashboards")
    
    # Example 4: Optimization Techniques
    print("\n" + "=" * 80)
    print("Example 4: Real-Time Optimization Techniques")
    print("=" * 80)
    
    print("\n1. In-Memory Caching:")
    print("   Cache frequently accessed data in RAM")
    print("   Example: Redis, Memcached")
    
    print("\n2. Connection Pooling:")
    print("   Reuse database/API connections")
    print("   Avoid connection overhead")
    
    print("\n3. Hot Path Optimization:")
    print("   Optimize most frequent code path")
    print("   Minimize I/O operations")
    
    print("\n4. Circuit Breakers:")
    print("   Fail fast on downstream failures")
    print("   Prevent cascading failures")
    
    print("\n5. Asynchronous Processing:")
    print("   Non-blocking I/O")
    print("   Event-driven architecture")
    
    print("\n6. CDN/Edge Computing:")
    print("   Serve content from nearest location")
    print("   Reduce network latency")
    
    print("\n" + "=" * 80)
    print("Real-Time Processing Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Real-time processing requires < 100ms latency
  • Use in-memory caching for instant responses
  • Hot path optimization minimizes latency
  • Circuit breakers prevent cascading failures
  • Trade complexity for speed
  • Common techniques: caching, connection pooling, async I/O
  • Tools: Redis, Memcached, Apache Flink, Kafka
  • Use for: APIs, web apps, fraud detection, gaming, trading
    """)
