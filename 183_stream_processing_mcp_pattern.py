"""
Pattern 183: Stream Processing MCP Pattern

This pattern demonstrates continuous processing of unbounded data streams in real-time.
Unlike batch processing, stream processing handles data as it arrives, enabling
low-latency analytics and real-time decision making.

Key Concepts:
1. Unbounded Data: Continuous flow of events (no end)
2. Event Time vs Processing Time: When event occurred vs when processed
3. Windowing: Group events into time-based windows
4. Stateful Processing: Maintain state across events
5. Watermarks: Track progress in event time
6. Late Data Handling: Process delayed events
7. Exactly-Once Semantics: Guarantee each event processed once

Stream Characteristics:
- Continuous Processing: Process events as they arrive
- Low Latency: Near real-time results (milliseconds to seconds)
- Unbounded: Infinite stream of events
- Ordered/Unordered: Events may arrive out of order
- Stateful: Maintain state across events (counts, aggregations)
- Scalable: Horizontal scaling for high throughput

Windowing Types:
1. Tumbling Window: Fixed-size, non-overlapping (every 5 minutes)
2. Sliding Window: Fixed-size, overlapping (last 5 minutes, slide every 1 minute)
3. Session Window: Dynamic size based on inactivity gap
4. Global Window: All events in single window

Stream Operations:
- Map: Transform each event
- Filter: Select events matching predicate
- FlatMap: Transform one event to multiple
- Reduce: Aggregate events (sum, count, average)
- Join: Combine multiple streams
- Window: Group events by time

Benefits:
- Real-Time: Immediate insights and actions
- Low Latency: Process events in milliseconds
- Continuous: Always on, processing 24/7
- Scalable: Handle millions of events per second
- Event-Driven: React to events as they happen

Trade-offs:
- Complexity: Harder than batch processing
- Ordering: Out-of-order events require careful handling
- State Management: Stateful operations need careful design
- Exactly-Once: Hard to guarantee in distributed systems
- Debugging: Harder to debug continuous processing

Use Cases:
- Real-Time Analytics: Dashboard metrics, KPIs
- Fraud Detection: Detect suspicious transactions immediately
- IoT Monitoring: Process sensor data streams
- Log Processing: Real-time log analysis and alerting
- Recommendation Systems: Update recommendations in real-time
- Stock Trading: Process market data and execute trades
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Deque
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import time

# Define the state for stream processing
class StreamProcessingState(TypedDict):
    """State for stream processing"""
    events: List[Dict[str, Any]]
    filtered_events: Optional[List[Dict[str, Any]]]
    windowed_aggregates: Optional[List[Dict[str, Any]]]
    enriched_events: Optional[List[Dict[str, Any]]]
    alerts: Optional[List[Dict[str, Any]]]
    stream_metadata: Dict[str, Any]
    processing_metrics: Annotated[List[Dict[str, Any]], operator.add]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# STREAM EVENT MODEL
# ============================================================================

@dataclass
class StreamEvent:
    """Event in data stream"""
    event_id: str
    event_type: str
    event_time: datetime  # When event occurred
    processing_time: datetime  # When event processed
    payload: Dict[str, Any]
    
    def get_delay(self) -> float:
        """Get processing delay in seconds"""
        return (self.processing_time - self.event_time).total_seconds()

class WindowType(Enum):
    """Types of time windows"""
    TUMBLING = "tumbling"  # Non-overlapping fixed windows
    SLIDING = "sliding"    # Overlapping fixed windows
    SESSION = "session"    # Dynamic windows based on activity

@dataclass
class TimeWindow:
    """Time-based window for grouping events"""
    window_start: datetime
    window_end: datetime
    window_type: WindowType
    events: List[StreamEvent]
    
    def is_event_in_window(self, event: StreamEvent) -> bool:
        """Check if event belongs to this window"""
        return self.window_start <= event.event_time < self.window_end
    
    def add_event(self, event: StreamEvent):
        """Add event to window"""
        if self.is_event_in_window(event):
            self.events.append(event)

# ============================================================================
# STREAM PROCESSORS
# ============================================================================

class StreamFilter:
    """
    Filter Stream: Select events matching predicate
    
    Example: Filter high-value transactions
    """
    
    def __init__(self, predicate_fn):
        self.predicate_fn = predicate_fn
    
    def filter(self, events: List[StreamEvent]) -> List[StreamEvent]:
        """Filter events"""
        return [event for event in events if self.predicate_fn(event)]

class StreamMapper:
    """
    Map Stream: Transform each event
    
    Example: Extract fields, convert formats
    """
    
    def __init__(self, map_fn):
        self.map_fn = map_fn
    
    def map(self, events: List[StreamEvent]) -> List[StreamEvent]:
        """Map events"""
        return [self.map_fn(event) for event in events]

class StreamAggregator:
    """
    Aggregate Stream: Compute aggregates over windows
    
    Aggregations: COUNT, SUM, AVG, MIN, MAX
    """
    
    def __init__(self, window_duration_seconds: int):
        self.window_duration = timedelta(seconds=window_duration_seconds)
        self.windows: Dict[str, TimeWindow] = {}
    
    def create_tumbling_windows(self, events: List[StreamEvent]) -> List[TimeWindow]:
        """
        Create tumbling windows (non-overlapping)
        
        Example: [0-5s], [5-10s], [10-15s]
        """
        if not events:
            return []
        
        # Find time range
        min_time = min(e.event_time for e in events)
        max_time = max(e.event_time for e in events)
        
        # Create windows
        windows = []
        current_start = min_time
        
        while current_start <= max_time:
            window_end = current_start + self.window_duration
            window = TimeWindow(
                window_start=current_start,
                window_end=window_end,
                window_type=WindowType.TUMBLING,
                events=[]
            )
            
            # Add events to window
            for event in events:
                if window.is_event_in_window(event):
                    window.add_event(event)
            
            windows.append(window)
            current_start = window_end
        
        return windows
    
    def aggregate_windows(self, windows: List[TimeWindow]) -> List[Dict[str, Any]]:
        """Compute aggregates for each window"""
        aggregates = []
        
        for window in windows:
            if not window.events:
                continue
            
            # Compute aggregations
            values = [event.payload.get("value", 0) for event in window.events]
            
            aggregate = {
                "window_start": window.window_start.isoformat(),
                "window_end": window.window_end.isoformat(),
                "count": len(window.events),
                "sum": sum(values),
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "event_types": list(set(e.event_type for e in window.events))
            }
            
            aggregates.append(aggregate)
        
        return aggregates

class StreamJoiner:
    """
    Join Streams: Combine events from multiple streams
    
    Join Types:
    - Inner Join: Events from both streams
    - Left Join: All from left, matching from right
    - Window Join: Join events within time window
    """
    
    def __init__(self, join_key: str, time_window_seconds: int = 5):
        self.join_key = join_key
        self.time_window = timedelta(seconds=time_window_seconds)
    
    def window_join(self, left_events: List[StreamEvent], 
                    right_events: List[StreamEvent]) -> List[Dict[str, Any]]:
        """
        Window Join: Join events within time window
        
        Example: Join clicks with purchases within 5 seconds
        """
        joined = []
        
        for left_event in left_events:
            for right_event in right_events:
                # Check if join key matches
                left_key = left_event.payload.get(self.join_key)
                right_key = right_event.payload.get(self.join_key)
                
                if left_key != right_key:
                    continue
                
                # Check if within time window
                time_diff = abs((left_event.event_time - right_event.event_time).total_seconds())
                
                if time_diff <= self.time_window.total_seconds():
                    # Join events
                    joined_event = {
                        "join_key": left_key,
                        "left_event": left_event.payload,
                        "right_event": right_event.payload,
                        "time_diff_seconds": time_diff
                    }
                    joined.append(joined_event)
        
        return joined

class StreamStateMachine:
    """
    Stateful Stream Processing: Maintain state across events
    
    Example: Track user session state, fraud detection patterns
    """
    
    def __init__(self):
        self.state: Dict[str, Any] = defaultdict(dict)
    
    def update_state(self, key: str, event: StreamEvent) -> Dict[str, Any]:
        """Update state based on event"""
        if key not in self.state:
            self.state[key] = {
                "first_seen": event.event_time,
                "last_seen": event.event_time,
                "event_count": 0,
                "total_value": 0
            }
        
        # Update state
        self.state[key]["last_seen"] = event.event_time
        self.state[key]["event_count"] += 1
        self.state[key]["total_value"] += event.payload.get("value", 0)
        
        return self.state[key]
    
    def get_state(self, key: str) -> Dict[str, Any]:
        """Get current state for key"""
        return self.state.get(key, {})

# ============================================================================
# STREAM PROCESSING ENGINE
# ============================================================================

class StreamProcessor:
    """
    Stream Processing Engine: Orchestrate stream operations
    
    Operations:
    1. Ingest events from stream
    2. Filter events
    3. Window and aggregate
    4. Detect patterns/anomalies
    5. Generate alerts
    """
    
    def __init__(self):
        self.state_machine = StreamStateMachine()
    
    def process_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process stream of events"""
        # Convert to StreamEvent objects
        stream_events = []
        for event_data in events:
            event = StreamEvent(
                event_id=event_data.get("id", ""),
                event_type=event_data.get("type", ""),
                event_time=datetime.fromisoformat(event_data.get("event_time", datetime.now().isoformat())),
                processing_time=datetime.now(),
                payload=event_data.get("payload", {})
            )
            stream_events.append(event)
        
        # 1. Filter high-value events
        filter_high_value = StreamFilter(lambda e: e.payload.get("value", 0) > 100)
        filtered = filter_high_value.filter(stream_events)
        
        # 2. Window and aggregate (5-second tumbling windows)
        aggregator = StreamAggregator(window_duration_seconds=5)
        windows = aggregator.create_tumbling_windows(stream_events)
        aggregates = aggregator.aggregate_windows(windows)
        
        # 3. Detect anomalies (high event rate)
        alerts = []
        for agg in aggregates:
            if agg["count"] > 3:  # Threshold
                alerts.append({
                    "alert_type": "HIGH_EVENT_RATE",
                    "window": f"{agg['window_start']} to {agg['window_end']}",
                    "event_count": agg["count"],
                    "severity": "HIGH"
                })
        
        return {
            "total_events": len(stream_events),
            "filtered_events": filtered,
            "windows": windows,
            "aggregates": aggregates,
            "alerts": alerts
        }

# Create processor instance
processor = StreamProcessor()

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def ingest_events(state: StreamProcessingState) -> StreamProcessingState:
    """Ingest events from stream"""
    events = state["events"]
    
    return {
        "stream_metadata": {
            "total_events": len(events),
            "ingest_time": datetime.now().isoformat()
        },
        "processing_metrics": [{
            "stage": "Ingest",
            "events_processed": len(events)
        }],
        "messages": [f"[Ingest] Received {len(events)} events from stream"]
    }

def filter_events(state: StreamProcessingState) -> StreamProcessingState:
    """Filter high-value events"""
    events = state["events"]
    
    # Convert to StreamEvent
    stream_events = []
    for event_data in events:
        event = StreamEvent(
            event_id=event_data.get("id", ""),
            event_type=event_data.get("type", ""),
            event_time=datetime.fromisoformat(event_data.get("event_time", datetime.now().isoformat())),
            processing_time=datetime.now(),
            payload=event_data.get("payload", {})
        )
        stream_events.append(event)
    
    # Filter high-value
    filter_proc = StreamFilter(lambda e: e.payload.get("value", 0) > 100)
    filtered = filter_proc.filter(stream_events)
    
    # Convert back to dict
    filtered_dicts = [
        {
            "id": e.event_id,
            "type": e.event_type,
            "event_time": e.event_time.isoformat(),
            "payload": e.payload
        }
        for e in filtered
    ]
    
    return {
        "filtered_events": filtered_dicts,
        "processing_metrics": [{
            "stage": "Filter",
            "events_in": len(stream_events),
            "events_out": len(filtered),
            "filtered_out": len(stream_events) - len(filtered)
        }],
        "messages": [f"[Filter] Filtered to {len(filtered)} high-value events"]
    }

def window_aggregate(state: StreamProcessingState) -> StreamProcessingState:
    """Window and aggregate events"""
    events = state["events"]
    
    # Convert to StreamEvent
    stream_events = []
    for event_data in events:
        event = StreamEvent(
            event_id=event_data.get("id", ""),
            event_type=event_data.get("type", ""),
            event_time=datetime.fromisoformat(event_data.get("event_time", datetime.now().isoformat())),
            processing_time=datetime.now(),
            payload=event_data.get("payload", {})
        )
        stream_events.append(event)
    
    # Create windows and aggregate
    aggregator = StreamAggregator(window_duration_seconds=5)
    windows = aggregator.create_tumbling_windows(stream_events)
    aggregates = aggregator.aggregate_windows(windows)
    
    return {
        "windowed_aggregates": aggregates,
        "processing_metrics": [{
            "stage": "Windowing",
            "windows_created": len(windows),
            "aggregates_computed": len(aggregates)
        }],
        "messages": [f"[Window] Created {len(windows)} windows, computed {len(aggregates)} aggregates"]
    }

def detect_anomalies(state: StreamProcessingState) -> StreamProcessingState:
    """Detect anomalies and generate alerts"""
    aggregates = state.get("windowed_aggregates", [])
    
    # Detect high event rate
    alerts = []
    for agg in aggregates:
        if agg["count"] > 3:
            alerts.append({
                "alert_type": "HIGH_EVENT_RATE",
                "window": f"{agg['window_start']} to {agg['window_end']}",
                "event_count": agg["count"],
                "severity": "HIGH"
            })
        
        # Detect high average value
        if agg["avg"] > 150:
            alerts.append({
                "alert_type": "HIGH_AVG_VALUE",
                "window": f"{agg['window_start']} to {agg['window_end']}",
                "average_value": agg["avg"],
                "severity": "MEDIUM"
            })
    
    return {
        "alerts": alerts,
        "processing_metrics": [{
            "stage": "Anomaly Detection",
            "alerts_generated": len(alerts)
        }],
        "messages": [f"[Anomaly Detection] Generated {len(alerts)} alerts"]
    }

# ============================================================================
# BUILD THE STREAM PROCESSING GRAPH
# ============================================================================

def create_stream_processing_graph():
    """
    Create a StateGraph demonstrating stream processing pattern.
    
    Flow:
    1. Ingest: Receive events from stream
    2. Filter: Select relevant events
    3. Window: Group into time windows and aggregate
    4. Detect: Find anomalies and generate alerts
    """
    
    workflow = StateGraph(StreamProcessingState)
    
    # Add stream processing stages
    workflow.add_node("ingest", ingest_events)
    workflow.add_node("filter", filter_events)
    workflow.add_node("window", window_aggregate)
    workflow.add_node("detect", detect_anomalies)
    
    # Define stream flow
    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "filter")
    workflow.add_edge("filter", "window")
    workflow.add_edge("window", "detect")
    workflow.add_edge("detect", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Stream Processing MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Real-Time Stream Processing
    print("\n" + "=" * 80)
    print("Example 1: Real-Time Event Stream Processing")
    print("=" * 80)
    
    stream_graph = create_stream_processing_graph()
    
    # Simulate event stream (events arrive over time)
    base_time = datetime.now()
    
    initial_state: StreamProcessingState = {
        "events": [
            {"id": "e1", "type": "transaction", "event_time": base_time.isoformat(), 
             "payload": {"value": 150, "user_id": "user1"}},
            {"id": "e2", "type": "transaction", "event_time": (base_time + timedelta(seconds=1)).isoformat(),
             "payload": {"value": 200, "user_id": "user2"}},
            {"id": "e3", "type": "transaction", "event_time": (base_time + timedelta(seconds=2)).isoformat(),
             "payload": {"value": 50, "user_id": "user1"}},
            {"id": "e4", "type": "transaction", "event_time": (base_time + timedelta(seconds=3)).isoformat(),
             "payload": {"value": 180, "user_id": "user3"}},
            {"id": "e5", "type": "transaction", "event_time": (base_time + timedelta(seconds=6)).isoformat(),
             "payload": {"value": 120, "user_id": "user2"}},
            {"id": "e6", "type": "transaction", "event_time": (base_time + timedelta(seconds=7)).isoformat(),
             "payload": {"value": 90, "user_id": "user1"}},
        ],
        "filtered_events": None,
        "windowed_aggregates": None,
        "enriched_events": None,
        "alerts": None,
        "stream_metadata": {},
        "processing_metrics": [],
        "messages": []
    }
    
    result = stream_graph.invoke(initial_state)
    
    print("\nStream Processing Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nProcessing Metrics:")
    for metric in result["processing_metrics"]:
        print(f"  {metric}")
    
    print("\nFiltered Events (value > 100):")
    for event in result["filtered_events"]:
        print(f"  {event['id']}: ${event['payload']['value']}")
    
    print("\nWindow Aggregates (5-second tumbling windows):")
    for agg in result["windowed_aggregates"]:
        print(f"  Window: {agg['window_start'][:19]} to {agg['window_end'][:19]}")
        print(f"    Count: {agg['count']}, Sum: ${agg['sum']}, Avg: ${agg['avg']:.2f}")
    
    print("\nAlerts:")
    for alert in result["alerts"]:
        print(f"  [{alert['severity']}] {alert['alert_type']}: {alert}")
    
    # Example 2: Windowing Types
    print("\n" + "=" * 80)
    print("Example 2: Window Types")
    print("=" * 80)
    
    print("\n1. Tumbling Window (Non-Overlapping):")
    print("   [0-5s], [5-10s], [10-15s]")
    print("   Use case: Count events per 5-minute interval")
    
    print("\n2. Sliding Window (Overlapping):")
    print("   [0-5s], [1-6s], [2-7s], [3-8s]")
    print("   Use case: Last 5 minutes of data, updated every minute")
    
    print("\n3. Session Window (Dynamic):")
    print("   Group events with < 5 min gap")
    print("   Use case: User sessions, activity bursts")
    
    # Example 3: Stream vs Batch
    print("\n" + "=" * 80)
    print("Example 3: Stream Processing vs Batch Processing")
    print("=" * 80)
    
    print("\nStream Processing:")
    print("  Data: Unbounded (continuous stream)")
    print("  Latency: Milliseconds to seconds")
    print("  Processing: Event-by-event or micro-batches")
    print("  Use case: Real-time dashboards, fraud detection")
    print("  Tools: Apache Flink, Kafka Streams, Apache Storm")
    
    print("\nBatch Processing:")
    print("  Data: Bounded (finite dataset)")
    print("  Latency: Minutes to hours")
    print("  Processing: Process entire dataset")
    print("  Use case: Daily reports, data warehousing")
    print("  Tools: Apache Spark, Hadoop MapReduce")
    
    print("\n" + "=" * 80)
    print("Stream Processing Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Stream processing handles continuous, unbounded data
  • Windows group events for aggregation (tumbling, sliding, session)
  • Stateful processing maintains state across events
  • Low latency enables real-time analytics and alerts
  • Handle out-of-order events with watermarks
  • Common operations: filter, map, reduce, join, window
  • Tools: Apache Flink, Kafka Streams, Apache Storm, Spark Streaming
  • Use for: real-time analytics, fraud detection, IoT monitoring
    """)
