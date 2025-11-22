"""
Metrics Collection MCP Pattern

This pattern collects, aggregates, and reports quantitative metrics
about system performance, health, and business operations.

Key Features:
- Multi-source metric collection
- Real-time aggregation
- Time-series data handling
- Metric types (counter, gauge, histogram, summary)
- Export and visualization
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class MetricsState(TypedDict):
    """State for metrics collection pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    metric_types: List[str]  # "counter", "gauge", "histogram", "summary"
    metrics: Dict[str, Dict]  # metric_name -> {type, value, labels, timestamp}
    collection_interval: int  # seconds
    retention_period: int  # days
    aggregations: Dict[str, float]  # metric_name -> aggregated_value
    export_format: str  # "prometheus", "statsd", "influxdb"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Metric Collector
def metric_collector(state: MetricsState) -> MetricsState:
    """Collects metrics from various sources"""
    metric_types = state.get("metric_types", ["counter", "gauge", "histogram"])
    collection_interval = state.get("collection_interval", 60)
    
    system_message = SystemMessage(content="""You are a metric collector.
    Collect quantitative metrics from system components.""")
    
    user_message = HumanMessage(content=f"""Collect metrics:

Types: {', '.join(metric_types)}
Interval: {collection_interval}s

Gather performance and business metrics.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate metric collection
    metrics = {
        "http_requests_total": {
            "type": "counter",
            "value": 15234,
            "labels": {"method": "GET", "status": "200"},
            "timestamp": int(time.time())
        },
        "cpu_usage_percent": {
            "type": "gauge",
            "value": 67.5,
            "labels": {"host": "server-1"},
            "timestamp": int(time.time())
        },
        "request_duration_seconds": {
            "type": "histogram",
            "value": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
            "buckets": {"0.01": 100, "0.05": 250, "0.1": 400, "0.25": 150, "0.5": 50, "1.0": 10},
            "labels": {"endpoint": "/api/users"},
            "timestamp": int(time.time())
        },
        "memory_usage_bytes": {
            "type": "gauge",
            "value": 1073741824,  # 1GB
            "labels": {"host": "server-1"},
            "timestamp": int(time.time())
        },
        "database_connections_active": {
            "type": "gauge",
            "value": 45,
            "labels": {"database": "postgres", "pool": "main"},
            "timestamp": int(time.time())
        },
        "cache_hits_total": {
            "type": "counter",
            "value": 8567,
            "labels": {"cache": "redis"},
            "timestamp": int(time.time())
        },
        "cache_misses_total": {
            "type": "counter",
            "value": 234,
            "labels": {"cache": "redis"},
            "timestamp": int(time.time())
        }
    }
    
    report = f"""
    📊 Metric Collection:
    
    Collection Summary:
    • Total Metrics: {len(metrics)}
    • Types: {', '.join(set(m['type'] for m in metrics.values()))}
    • Collection Interval: {collection_interval}s
    • Timestamp: {int(time.time())}
    
    Metric Types Explained:
    
    Counter:
    • Monotonically increasing value
    • Always goes up (or resets to 0)
    • Examples: requests_total, errors_total
    • Rate calculation: increase per second
    • Prometheus: counter
    
    Gauge:
    • Current value (can go up or down)
    • Point-in-time measurement
    • Examples: cpu_usage, memory_bytes, queue_size
    • Direct value representation
    • Prometheus: gauge
    
    Histogram:
    • Distribution of values
    • Bucketed observations
    • Examples: request_duration, response_size
    • Calculates: sum, count, buckets
    • Percentiles: p50, p95, p99
    • Prometheus: histogram
    
    Summary:
    • Similar to histogram
    • Client-side percentiles
    • Streaming quantiles
    • Lower server load
    • Examples: request_latency
    • Prometheus: summary
    
    Metric Naming Conventions:
    
    Prometheus Style:
    ```
    <namespace>_<subsystem>_<name>_<unit>
    
    Examples:
    http_requests_total
    process_cpu_seconds_total
    http_request_duration_seconds
    node_memory_bytes
    ```
    
    Base Units:
    • Time: seconds (not milliseconds)
    • Size: bytes (not MB/GB)
    • Percentage: ratio (0.0-1.0, not 0-100)
    • Temperature: celsius
    
    Labels (Dimensions):
    ```python
    http_requests_total{{
        method="GET",
        endpoint="/api/users",
        status="200"
    }}
    ```
    
    Label Best Practices:
    • Low cardinality (< 100 unique values)
    • Avoid user IDs, timestamps
    • Use for filtering/grouping
    • Consistent naming
    • Don't change dynamically
    
    Collection Methods:
    
    Pull-Based (Prometheus):
    • Metrics endpoint (/metrics)
    • Server scrapes periodically
    • Service discovery
    • Simple architecture
    • Missed scrapes = data loss
    
    Push-Based (StatsD):
    • Send metrics to collector
    • Fire-and-forget UDP
    • Short-lived processes
    • Higher network load
    • Buffering required
    
    Libraries by Language:
    
    Python:
    ```python
    from prometheus_client import Counter, Gauge, Histogram
    
    # Counter
    requests = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint']
    )
    requests.labels(method='GET', endpoint='/api').inc()
    
    # Gauge
    cpu = Gauge('cpu_usage_percent', 'CPU usage')
    cpu.set(67.5)
    
    # Histogram
    latency = Histogram(
        'request_duration_seconds',
        'Request latency',
        buckets=[.01, .05, .1, .5, 1, 5]
    )
    latency.observe(0.234)
    ```
    
    Go:
    ```go
    import "github.com/prometheus/client_golang/prometheus"
    
    var (
        requests = prometheus.NewCounterVec(
            prometheus.CounterOpts{{
                Name: "http_requests_total",
                Help: "Total HTTP requests",
            }},
            []string{{"method", "endpoint"}},
        )
    )
    
    requests.WithLabelValues("GET", "/api").Inc()
    ```
    
    Java:
    ```java
    import io.prometheus.client.Counter;
    
    Counter requests = Counter.build()
        .name("http_requests_total")
        .help("Total HTTP requests")
        .labelNames("method", "endpoint")
        .register();
    
    requests.labels("GET", "/api").inc();
    ```
    
    Common Metrics to Collect:
    
    System Metrics:
    • CPU usage (%)
    • Memory usage (bytes/%)
    • Disk I/O (bytes, iops)
    • Network I/O (bytes, packets)
    • File descriptors
    • Thread count
    
    Application Metrics:
    • Request rate (req/sec)
    • Error rate (errors/sec, %)
    • Response time (p50, p95, p99)
    • Active connections
    • Queue length
    • Cache hit ratio
    
    Business Metrics:
    • Transactions completed
    • Revenue generated
    • User signups
    • Items sold
    • Conversion rate
    • Active users
    
    Database Metrics:
    • Query duration
    • Connection pool usage
    • Slow queries
    • Lock waits
    • Replication lag
    • Table sizes
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Metric Collector:\n{response.content}\n{report}")],
        "metrics": metrics
    }


# Metric Aggregator
def metric_aggregator(state: MetricsState) -> MetricsState:
    """Aggregates metrics over time windows"""
    metrics = state.get("metrics", {})
    
    system_message = SystemMessage(content="""You are a metric aggregator.
    Aggregate metrics and compute statistical summaries.""")
    
    user_message = HumanMessage(content=f"""Aggregate metrics:

Total Metrics: {len(metrics)}

Compute aggregations and statistics.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate aggregations
    aggregations = {
        "avg_cpu_usage": 65.3,
        "p95_request_duration": 0.42,
        "total_requests": 15234,
        "error_rate_percent": 0.5,
        "cache_hit_rate": 97.3
    }
    
    report = f"""
    🔢 Metric Aggregation:
    
    Aggregation Results:
    • Metrics Processed: {len(metrics)}
    • Aggregations Computed: {len(aggregations)}
    
    Key Aggregations:
    {chr(10).join(f"• {k}: {v}" for k, v in aggregations.items())}
    
    Aggregation Functions:
    
    Basic Statistics:
    • SUM: Total value
    • COUNT: Number of observations
    • AVG/MEAN: Average value
    • MIN: Minimum value
    • MAX: Maximum value
    
    Percentiles:
    • P50 (Median): 50th percentile
    • P90: 90th percentile
    • P95: 95th percentile
    • P99: 99th percentile
    • P99.9: 99.9th percentile
    
    Rates:
    • rate(): Per-second rate
    • irate(): Instant rate
    • increase(): Total increase
    • delta(): Difference
    
    Moving Averages:
    • Simple Moving Average (SMA)
    • Exponential Moving Average (EMA)
    • Weighted Moving Average (WMA)
    
    Prometheus Query Examples:
    
    Rate Calculation:
    ```promql
    # Requests per second
    rate(http_requests_total[5m])
    
    # Error rate
    rate(http_errors_total[5m]) / 
    rate(http_requests_total[5m])
    ```
    
    Percentiles:
    ```promql
    # 95th percentile latency
    histogram_quantile(0.95,
      rate(request_duration_seconds_bucket[5m])
    )
    ```
    
    Aggregation:
    ```promql
    # Total CPU across all hosts
    sum(cpu_usage) by (cluster)
    
    # Average memory per pod
    avg(memory_bytes) by (pod)
    ```
    
    Time Windows:
    ```promql
    # Last 5 minutes
    avg_over_time(cpu_usage[5m])
    
    # Last hour
    max_over_time(memory_bytes[1h])
    ```
    
    InfluxDB Queries:
    ```sql
    SELECT mean("cpu_usage")
    FROM "system"
    WHERE time > now() - 1h
    GROUP BY time(5m), "host"
    
    SELECT percentile("duration", 95)
    FROM "requests"
    WHERE time > now() - 1h
    ```
    
    Aggregation Strategies:
    
    Time-Based:
    • 1-minute windows
    • 5-minute windows  
    • Hourly summaries
    • Daily roll-ups
    
    Dimension-Based:
    • Group by host
    • Group by service
    • Group by region
    • Group by environment
    
    Downsampling:
    • Raw data: 10s resolution
    • 1 hour: 1m resolution
    • 1 day: 5m resolution
    • 1 week: 15m resolution
    • 1 month: 1h resolution
    
    Storage Optimization:
    • Retention policies
    • Compression
    • Aggregated views
    • Sampling
    • Drop old data
    
    Common Patterns:
    
    RED Method (Requests):
    • Rate: Requests per second
    • Errors: Error rate
    • Duration: Response time
    
    USE Method (Resources):
    • Utilization: % busy
    • Saturation: Queue depth
    • Errors: Error count
    
    Four Golden Signals:
    • Latency: Response time
    • Traffic: Request rate
    • Errors: Error rate
    • Saturation: Resource usage
    """
    
    return {
        "messages": [AIMessage(content=f"🔢 Metric Aggregator:\n{response.content}\n{report}")],
        "aggregations": aggregations
    }


# Metric Reporter
def metric_reporter(state: MetricsState) -> MetricsState:
    """Reports metrics in various formats"""
    metrics = state.get("metrics", {})
    aggregations = state.get("aggregations", {})
    export_format = state.get("export_format", "prometheus")
    retention_period = state.get("retention_period", 30)
    
    summary = f"""
    📊 METRICS COLLECTION COMPLETE
    
    Metrics Summary:
    • Total Metrics: {len(metrics)}
    • Aggregations: {len(aggregations)}
    • Export Format: {export_format.upper()}
    • Retention: {retention_period} days
    
    Metrics Collection Pattern Process:
    1. Metric Collector → Gather metrics
    2. Metric Aggregator → Compute statistics
    3. Metric Reporter → Export and visualize
    
    Export Formats:
    
    Prometheus Format:
    ```
    # HELP http_requests_total Total HTTP requests
    # TYPE http_requests_total counter
    http_requests_total{{method="GET",status="200"}} 15234
    
    # HELP cpu_usage_percent CPU usage percentage
    # TYPE cpu_usage_percent gauge
    cpu_usage_percent{{host="server-1"}} 67.5
    
    # HELP request_duration_seconds Request duration
    # TYPE request_duration_seconds histogram
    request_duration_seconds_bucket{{le="0.1"}} 250
    request_duration_seconds_bucket{{le="0.5"}} 450
    request_duration_seconds_bucket{{le="+Inf"}} 500
    request_duration_seconds_sum 125.5
    request_duration_seconds_count 500
    ```
    
    StatsD Format:
    ```
    # Counter
    http.requests:1|c|@0.1
    
    # Gauge
    cpu.usage:67.5|g
    
    # Timer
    request.duration:234|ms
    
    # Histogram
    response.size:1024|h
    ```
    
    InfluxDB Line Protocol:
    ```
    cpu_usage,host=server-1 value=67.5 1609459200000000000
    http_requests,method=GET,status=200 count=15234 1609459200000000000
    ```
    
    JSON Format:
    ```json
    {{
      "metrics": [
        {{
          "name": "http_requests_total",
          "type": "counter",
          "value": 15234,
          "labels": {{"method": "GET", "status": "200"}},
          "timestamp": 1609459200
        }}
      ]
    }}
    ```
    
    Monitoring Stack Options:
    
    Prometheus + Grafana:
    • Open source
    • Pull-based metrics
    • PromQL query language
    • Alert manager
    • Service discovery
    • Grafana dashboards
    
    Datadog:
    • SaaS platform
    • APM + Infrastructure
    • Custom metrics
    • Alerting
    • Dashboards
    • Log correlation
    
    New Relic:
    • Full-stack observability
    • APM focus
    • Real user monitoring
    • Distributed tracing
    • Custom dashboards
    
    CloudWatch:
    • AWS native
    • Metric math
    • Alarms
    • Dashboards
    • Log insights integration
    
    Elastic Stack:
    • Metricbeat collectors
    • Elasticsearch storage
    • Kibana visualization
    • Machine learning
    
    Best Practices:
    
    Naming:
    • Descriptive names
    • Consistent format
    • Include units
    • Use underscores
    • Namespace properly
    
    Labels:
    • Low cardinality
    • Static dimensions
    • Filterable attributes
    • Consistent keys
    • Avoid high-cardinality values
    
    Collection:
    • Appropriate intervals (10s-60s)
    • Buffer metrics
    • Handle failures
    • Retry logic
    • Monitor collector health
    
    Storage:
    • Retention policies
    • Compression
    • Downsampling
    • Backup strategy
    • Capacity planning
    
    Performance:
    • Batch exports
    • Async processing
    • Connection pooling
    • Rate limiting
    • Resource limits
    
    Alerting Integration:
    • Define thresholds
    • Alert on trends
    • Anomaly detection
    • Alert fatigue prevention
    • Runbook links
    
    Real-World Metrics:
    
    E-commerce:
    • Orders per minute
    • Checkout completion rate
    • Cart abandonment rate
    • Revenue per hour
    • Inventory levels
    
    SaaS Application:
    • Active users
    • API calls per user
    • Feature usage
    • Signup rate
    • Churn rate
    
    Infrastructure:
    • CPU/Memory/Disk
    • Network throughput
    • Connection counts
    • Queue depths
    • Cache hit rates
    
    Key Insight:
    Metrics provide quantitative visibility into system
    health and business performance. Choose appropriate
    metric types and aggregations for your use case.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Metric Reporter:\n{summary}")]
    }


# Build the graph
def build_metrics_graph():
    """Build the metrics collection pattern graph"""
    workflow = StateGraph(MetricsState)
    
    workflow.add_node("collector", metric_collector)
    workflow.add_node("aggregator", metric_aggregator)
    workflow.add_node("reporter", metric_reporter)
    
    workflow.add_edge(START, "collector")
    workflow.add_edge("collector", "aggregator")
    workflow.add_edge("aggregator", "reporter")
    workflow.add_edge("reporter", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_metrics_graph()
    
    print("=== Metrics Collection MCP Pattern ===\n")
    
    # Test Case: Application metrics collection
    print("\n" + "="*70)
    print("TEST CASE: Application Metrics Collection")
    print("="*70)
    
    state = {
        "messages": [],
        "metric_types": ["counter", "gauge", "histogram", "summary"],
        "metrics": {},
        "collection_interval": 60,
        "retention_period": 30,
        "aggregations": {},
        "export_format": "prometheus"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    metrics = result.get('metrics', {})
    print(f"\nMetrics Collected: {len(metrics)}")
    print(f"Aggregations Computed: {len(result.get('aggregations', {}))}")
    print(f"Export Format: {result.get('export_format', 'unknown')}")
