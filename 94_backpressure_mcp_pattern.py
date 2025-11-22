"""
Backpressure MCP Pattern

This pattern handles situations where downstream systems cannot keep up
with the rate of incoming requests, preventing system overload.

Key Features:
- Flow control mechanisms
- Queue management
- Load shedding strategies
- Graceful degradation
- Producer-consumer balancing
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class BackpressureState(TypedDict):
    """State for backpressure pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    system_name: str
    queue_capacity: int
    current_queue_size: int
    processing_rate: int  # items/second
    arrival_rate: int  # items/second
    backpressure_strategy: str  # "drop_oldest", "drop_newest", "block", "reject"
    items_processed: int
    items_dropped: int
    system_health: str  # "healthy", "warning", "overloaded"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Backpressure Detector
def backpressure_detector(state: BackpressureState) -> BackpressureState:
    """Detects backpressure conditions"""
    system_name = state.get("system_name", "")
    queue_capacity = state.get("queue_capacity", 1000)
    current_queue_size = state.get("current_queue_size", 0)
    processing_rate = state.get("processing_rate", 100)
    arrival_rate = state.get("arrival_rate", 150)
    
    system_message = SystemMessage(content="""You are a backpressure detector.
    Monitor system capacity and detect overload conditions.""")
    
    user_message = HumanMessage(content=f"""Detect backpressure:

System: {system_name}
Queue: {current_queue_size}/{queue_capacity}
Processing: {processing_rate}/s
Arrival: {arrival_rate}/s

Analyze system capacity.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Detect backpressure
    utilization = (current_queue_size / queue_capacity * 100) if queue_capacity > 0 else 0
    rate_imbalance = arrival_rate - processing_rate
    
    if utilization > 80 or rate_imbalance > processing_rate * 0.5:
        system_health = "overloaded"
    elif utilization > 50 or rate_imbalance > 0:
        system_health = "warning"
    else:
        system_health = "healthy"
    
    report = f"""
    🔍 Backpressure Detection:
    
    System Metrics:
    • Queue Utilization: {utilization:.1f}%
    • Processing Rate: {processing_rate}/s
    • Arrival Rate: {arrival_rate}/s
    • Rate Imbalance: {rate_imbalance:+d}/s
    • System Health: {system_health.upper()}
    
    Backpressure Indicators:
    
    Queue-Based:
    • Queue depth increasing
    • Queue near capacity
    • Sustained high utilization
    • Growing wait times
    
    Rate-Based:
    • Arrival > Processing
    • Sustained imbalance
    • Throughput degradation
    • Latency increase
    
    Resource-Based:
    • CPU saturation
    • Memory pressure
    • Disk I/O bottleneck
    • Network congestion
    
    Backpressure Patterns:
    
    Reactive Streams (Java):
    ```java
    Publisher<T> publisher;
    Subscriber<T> subscriber;
    
    subscriber.onSubscribe(subscription -> {{
        subscription.request(10);  // Request 10 items
    }});
    
    // Publisher respects subscriber pace
    // Subscriber controls flow
    ```
    
    Golang Channels:
    ```go
    ch := make(chan int, 100)  // Buffered
    
    // Producer blocks when full
    ch <- item
    
    // Consumer controls pace
    item := <-ch
    ```
    
    Akka Streams:
    ```scala
    Source(items)
      .buffer(100, OverflowStrategy.backpressure)
      .throttle(10, 1.second)
      .to(Sink)
    ```
    
    RxJS:
    ```javascript
    observable
      .pipe(
        bufferTime(1000),
        map(buffer => process(buffer))
      )
      .subscribe();
    ```
    
    Detection Strategies:
    
    Queue Monitoring:
    • Track queue depth over time
    • Alert on sustained growth
    • Monitor queue age
    • Detect hot spots
    
    Rate Comparison:
    • Input vs output rates
    • Capacity vs demand
    • Trend analysis
    • Predictive alerting
    
    Latency Tracking:
    • End-to-end latency
    • Queue wait time
    • Processing time
    • P95/P99 metrics
    
    Resource Utilization:
    • CPU, Memory, Disk, Network
    • Saturation metrics
    • Contention detection
    • Bottleneck identification
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Backpressure Detector:\n{response.content}\n{report}")],
        "system_health": system_health
    }


# Flow Controller
def flow_controller(state: BackpressureState) -> BackpressureState:
    """Controls flow based on backpressure"""
    system_health = state.get("system_health", "healthy")
    backpressure_strategy = state.get("backpressure_strategy", "drop_oldest")
    queue_capacity = state.get("queue_capacity", 1000)
    current_queue_size = state.get("current_queue_size", 0)
    arrival_rate = state.get("arrival_rate", 150)
    processing_rate = state.get("processing_rate", 100)
    
    system_message = SystemMessage(content="""You are a flow controller.
    Apply backpressure strategies to manage system load.""")
    
    user_message = HumanMessage(content=f"""Control flow:

Health: {system_health}
Strategy: {backpressure_strategy}
Queue: {current_queue_size}/{queue_capacity}
Rates: {arrival_rate}/s in, {processing_rate}/s out

Apply flow control.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate flow control
    if system_health == "overloaded":
        if backpressure_strategy == "drop_oldest":
            items_dropped = 20
            items_processed = processing_rate
        elif backpressure_strategy == "drop_newest":
            items_dropped = 20
            items_processed = processing_rate
        elif backpressure_strategy == "reject":
            items_dropped = arrival_rate - processing_rate
            items_processed = processing_rate
        else:  # block
            items_dropped = 0
            items_processed = processing_rate
    else:
        items_dropped = 0
        items_processed = min(arrival_rate, processing_rate)
    
    report = f"""
    🎛️ Flow Control:
    
    Control Actions:
    • Strategy: {backpressure_strategy.upper()}
    • Items Processed: {items_processed}
    • Items Dropped: {items_dropped}
    • System Health: {system_health}
    
    Backpressure Strategies:
    
    1. Drop Oldest (FIFO Eviction):
       • Remove oldest items first
       • Favor recent data
       • Good for real-time data
       • Example: Live metrics
    
    2. Drop Newest (LIFO Eviction):
       • Remove newest items
       • Preserve historical data
       • Good for audit logs
       • Example: Compliance data
    
    3. Reject New:
       • Refuse incoming requests
       • Return error to client
       • Client handles retry
       • Example: HTTP 503
    
    4. Block Producer:
       • Pause incoming flow
       • Wait for capacity
       • Synchronous backpressure
       • Example: Blocking queues
    
    5. Shed Load:
       • Drop percentage of traffic
       • Priority-based dropping
       • Preserve critical requests
       • Example: Load shedding
    
    6. Throttle Input:
       • Slow down producers
       • Rate limiting
       • Smooth flow
       • Example: Token bucket
    
    Implementation Examples:
    
    RabbitMQ:
    ```python
    # Consumer prefetch
    channel.basic_qos(prefetch_count=10)
    
    # Memory alarm triggers
    rabbitmqctl set_vm_memory_high_watermark 0.4
    
    # Flow control automatically applied
    ```
    
    Kafka:
    ```python
    # Producer config
    producer = KafkaProducer(
        buffer_memory=33554432,  # 32MB
        max_block_ms=60000,      # Block 60s
        # Then throw exception
    )
    
    # Consumer controls pace
    consumer.poll(max_records=100)
    ```
    
    gRPC:
    ```python
    # Streaming with backpressure
    def stream_data(request_iterator):
        for request in request_iterator:
            # Process at consumer pace
            yield process(request)
    
    # Client controls flow
    ```
    
    TCP Flow Control:
    • Window size advertisement
    • Receiver window (RWND)
    • Congestion window (CWND)
    • ACK pacing
    • Automatic backpressure
    
    Circuit Breaker Integration:
    • Detect repeated failures
    • Open circuit on overload
    • Fail fast
    • Reduce cascading failures
    • Automatic recovery
    
    Load Shedding:
    • Priority queues
    • Drop low-priority first
    • Preserve critical traffic
    • Graceful degradation
    • Partial service
    """
    
    return {
        "messages": [AIMessage(content=f"🎛️ Flow Controller:\n{response.content}\n{report}")],
        "items_processed": items_processed,
        "items_dropped": items_dropped
    }


# Backpressure Monitor
def backpressure_monitor(state: BackpressureState) -> BackpressureState:
    """Monitors backpressure metrics"""
    system_name = state.get("system_name", "")
    queue_capacity = state.get("queue_capacity", 0)
    current_queue_size = state.get("current_queue_size", 0)
    processing_rate = state.get("processing_rate", 0)
    arrival_rate = state.get("arrival_rate", 0)
    items_processed = state.get("items_processed", 0)
    items_dropped = state.get("items_dropped", 0)
    system_health = state.get("system_health", "healthy")
    
    drop_rate = (items_dropped / (items_processed + items_dropped) * 100) if (items_processed + items_dropped) > 0 else 0
    
    summary = f"""
    📊 BACKPRESSURE COMPLETE
    
    System Status:
    • System: {system_name}
    • Health: {system_health.upper()}
    • Queue: {current_queue_size}/{queue_capacity}
    • Processing: {processing_rate}/s
    • Arrival: {arrival_rate}/s
    • Items Processed: {items_processed}
    • Items Dropped: {items_dropped}
    • Drop Rate: {drop_rate:.1f}%
    
    Backpressure Pattern Process:
    1. Detector → Identify overload conditions
    2. Flow Controller → Apply mitigation strategies
    3. Monitor → Track effectiveness
    
    Real-World Examples:
    
    Netflix:
    • Hystrix circuit breaker
    • Thread pool isolation
    • Semaphore isolation
    • Adaptive concurrency limits
    • Graceful degradation
    
    Uber:
    • QALM (Queue Adaptive Limiting)
    • Adaptive queue sizing
    • Request prioritization
    • Load shedding
    • Regional backpressure
    
    Google:
    • Adaptive throttling
    • Global rate limiting
    • Per-user quotas
    • Load shedding
    • Graceful degradation
    
    Kafka:
    • Producer buffering
    • Broker flow control
    • Consumer backpressure
    • Partition rebalancing
    • Quota management
    
    Backpressure Propagation:
    
    Upstream:
    • Signal to producers
    • Slow down input rate
    • Apply rate limiting
    • Queue at source
    
    Downstream:
    • Consumer pace control
    • Pull-based consumption
    • Batch processing
    • Acknowledgment control
    
    Horizontal:
    • Load balancing
    • Request routing
    • Circuit breaking
    • Failover
    
    Best Practices:
    
    Design:
    • Choose appropriate strategy
    • Implement multiple levels
    • Consider data importance
    • Plan for degradation
    • Test failure modes
    
    Monitoring:
    • Queue depth
    • Processing latency
    • Drop rate
    • System resources
    • End-to-end latency
    
    Alerting:
    • Sustained backpressure
    • High drop rates
    • Queue near capacity
    • Performance degradation
    • Resource saturation
    
    Recovery:
    • Automatic scaling
    • Circuit breaker recovery
    • Queue draining
    • Priority processing
    • Graceful restart
    
    Testing:
    • Load testing
    • Stress testing
    • Chaos engineering
    • Failure injection
    • Capacity planning
    
    Key Insight:
    Backpressure is essential for system stability under load.
    Implement multiple strategies and propagate signals
    throughout your distributed system.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Backpressure Monitor:\n{summary}")]
    }


# Build the graph
def build_backpressure_graph():
    """Build the backpressure pattern graph"""
    workflow = StateGraph(BackpressureState)
    
    workflow.add_node("detector", backpressure_detector)
    workflow.add_node("controller", flow_controller)
    workflow.add_node("monitor", backpressure_monitor)
    
    workflow.add_edge(START, "detector")
    workflow.add_edge("detector", "controller")
    workflow.add_edge("controller", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_backpressure_graph()
    
    print("=== Backpressure MCP Pattern ===\n")
    
    # Test Case: Message processing system under load
    print("\n" + "="*70)
    print("TEST CASE: Message Queue Backpressure")
    print("="*70)
    
    state = {
        "messages": [],
        "system_name": "message_processor",
        "queue_capacity": 1000,
        "current_queue_size": 850,
        "processing_rate": 100,
        "arrival_rate": 150,
        "backpressure_strategy": "drop_oldest",
        "items_processed": 0,
        "items_dropped": 0,
        "system_health": "healthy"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nSystem Health: {result.get('system_health', 'unknown')}")
    print(f"Items Processed: {result.get('items_processed', 0)}")
    print(f"Items Dropped: {result.get('items_dropped', 0)}")
