"""
Telemetry MCP Pattern

This pattern implements comprehensive telemetry collection across
distributed systems for observability and monitoring.

Key Features:
- Multi-source telemetry collection
- Metrics, logs, and traces integration
- Real-time data aggregation
- Telemetry export and storage
- Observability correlation
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class TelemetryState(TypedDict):
    """State for telemetry pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    telemetry_data: Dict  # {metrics, logs, traces, events}
    sources: List[str]
    collection_interval_seconds: int
    exporters: List[str]  # ["prometheus", "jaeger", "elasticsearch"]
    correlation_data: Dict


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Telemetry Collector
def telemetry_collector(state: TelemetryState) -> TelemetryState:
    """Collects telemetry from multiple sources"""
    sources = state.get("sources", [])
    
    system_message = SystemMessage(content="""You are a telemetry collector.
    Gather metrics, logs, and traces from distributed systems.""")
    
    user_message = HumanMessage(content=f"""Collect telemetry:

Sources: {len(sources) if sources else 'Auto-discover'}

Initialize telemetry collection.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Define sources if not provided
    if not sources:
        sources = ["api-gateway", "user-service", "database", "cache", "message-queue"]
    
    # Collect comprehensive telemetry
    current_time = int(time.time())
    
    telemetry_data = {
        "metrics": {
            "api-gateway": {
                "http_requests_total": 15234,
                "http_request_duration_seconds": {"p50": 0.045, "p95": 0.125, "p99": 0.342},
                "http_errors_total": 87,
                "active_connections": 234
            },
            "user-service": {
                "rpc_calls_total": 8934,
                "rpc_duration_seconds": {"p50": 0.023, "p95": 0.089, "p99": 0.234},
                "cpu_usage_percent": 67.5,
                "memory_usage_mb": 512.3
            },
            "database": {
                "query_duration_seconds": {"p50": 0.012, "p95": 0.056, "p99": 0.145},
                "active_connections": 45,
                "connection_pool_usage_percent": 75.0,
                "slow_queries_total": 12
            }
        },
        "logs": [
            {
                "timestamp": current_time,
                "level": "ERROR",
                "source": "api-gateway",
                "message": "Failed to connect to user-service",
                "trace_id": "abc123",
                "span_id": "span-1"
            },
            {
                "timestamp": current_time + 10,
                "level": "WARN",
                "source": "database",
                "message": "High connection pool utilization",
                "trace_id": None,
                "span_id": None
            }
        ],
        "traces": [
            {
                "trace_id": "abc123",
                "spans": [
                    {"span_id": "span-1", "service": "api-gateway", "duration_ms": 245},
                    {"span_id": "span-2", "service": "user-service", "duration_ms": 180},
                    {"span_id": "span-3", "service": "database", "duration_ms": 165}
                ],
                "total_duration_ms": 245
            }
        ],
        "events": [
            {
                "timestamp": current_time,
                "type": "deployment",
                "source": "ci-cd",
                "message": "Deployed version 1.2.3 to production",
                "metadata": {"version": "1.2.3", "environment": "production"}
            }
        ]
    }
    
    exporters = ["prometheus", "jaeger", "elasticsearch", "datadog"]
    collection_interval = 15
    
    report = f"""
    📡 Telemetry Collector:
    
    Collection Overview:
    • Sources: {len(sources)}
    • Metrics Collected: {sum(len(v) for v in telemetry_data['metrics'].values())}
    • Log Entries: {len(telemetry_data['logs'])}
    • Traces: {len(telemetry_data['traces'])}
    • Events: {len(telemetry_data['events'])}
    • Interval: {collection_interval}s
    
    Telemetry Concepts:
    
    Three Pillars of Observability:
    
    Metrics:
    • Quantitative measurements
    • Time-series data
    • Aggregatable
    • Low cardinality
    • Examples: CPU %, request rate, error count
    
    Logs:
    • Discrete events
    • Detailed context
    • High cardinality
    • Searchable
    • Examples: Errors, warnings, audit trail
    
    Traces:
    • Request flow
    • Service dependencies
    • Timing breakdown
    • Distributed context
    • Examples: End-to-end request path
    
    Additional Signals:
    
    Events:
    • State changes
    • Deployments
    • Configuration changes
    • Infrastructure events
    • Business events
    
    Profiles:
    • CPU flamegraphs
    • Memory allocation
    • Continuous profiling
    • Code-level performance
    
    OpenTelemetry:
    
    Overview:
    • Vendor-neutral standard
    • CNCF graduated project
    • Single SDK for all signals
    • Auto-instrumentation
    • Multiple exporters
    
    Components:
    • API: Instrumentation interface
    • SDK: Implementation
    • Collector: Processing pipeline
    • Exporters: Backend integration
    
    Python Example:
    ```python
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    
    # Setup resource (service identification)
    resource = Resource(attributes={{
        "service.name": "my-service",
        "service.version": "1.2.3",
        "deployment.environment": "production"
    }})
    
    # Setup tracing
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    
    # Setup span exporter
    span_exporter = OTLPSpanExporter(endpoint="http://collector:4317")
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(span_exporter)
    )
    
    # Setup metrics
    metrics.set_meter_provider(MeterProvider(resource=resource))
    meter = metrics.get_meter(__name__)
    
    # Create metrics
    request_counter = meter.create_counter(
        "http_requests_total",
        description="Total HTTP requests"
    )
    
    request_duration = meter.create_histogram(
        "http_request_duration_seconds",
        description="HTTP request duration"
    )
    
    # Use in application
    @tracer.start_as_current_span("process_request")
    def process_request(request):
        request_counter.add(1, {{"method": request.method}})
        
        start = time.time()
        result = handle_request(request)
        duration = time.time() - start
        
        request_duration.record(duration, {{"endpoint": request.path}})
        
        return result
    ```
    
    Telemetry Collection Patterns:
    
    Push-based:
    • Application pushes data
    • StatsD, OTLP
    • Fire-and-forget
    • Good for short-lived processes
    
    Pull-based:
    • Collector scrapes endpoints
    • Prometheus /metrics
    • Service discovery
    • Good for long-lived services
    
    Agent-based:
    • Sidecar collector
    • DaemonSet (Kubernetes)
    • Local aggregation
    • Reduced network traffic
    
    Telemetry Pipeline:
    
    Collection → Processing → Export → Storage → Visualization
    
    Processing Steps:
    • Filtering: Remove noise
    • Sampling: Reduce volume
    • Enrichment: Add context
    • Aggregation: Summarize data
    • Correlation: Link signals
    
    OpenTelemetry Collector:
    ```yaml
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318
      
      prometheus:
        config:
          scrape_configs:
          - job_name: 'services'
            scrape_interval: 15s
            static_configs:
            - targets: ['service:8080']
    
    processors:
      batch:
        timeout: 10s
        send_batch_size: 1024
      
      attributes:
        actions:
        - key: environment
          value: production
          action: insert
      
      resource:
        attributes:
        - key: cluster
          value: us-west-2
          action: insert
    
    exporters:
      prometheus:
        endpoint: "0.0.0.0:8889"
      
      jaeger:
        endpoint: "jaeger:14250"
        tls:
          insecure: true
      
      elasticsearch:
        endpoints: ["http://elasticsearch:9200"]
        index: "telemetry"
      
      otlp:
        endpoint: "backend:4317"
    
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch, attributes]
          exporters: [jaeger, otlp]
        
        metrics:
          receivers: [otlp, prometheus]
          processors: [batch, resource]
          exporters: [prometheus, otlp]
        
        logs:
          receivers: [otlp]
          processors: [batch, attributes]
          exporters: [elasticsearch]
    ```
    
    Sampling Strategies:
    
    Head Sampling (Before trace complete):
    • Probabilistic: Sample N%
    • Rate limiting: Max traces/sec
    • Always sample errors
    • Deterministic across services
    
    Tail Sampling (After trace complete):
    • Sample based on full context
    • Keep slow requests
    • Keep errors
    • Smart sampling decisions
    • Higher resource cost
    """
    
    return {
        "messages": [AIMessage(content=f"📡 Telemetry Collector:\n{response.content}\n{report}")],
        "telemetry_data": telemetry_data,
        "sources": sources,
        "collection_interval_seconds": collection_interval,
        "exporters": exporters
    }


# Correlation Engine
def correlation_engine(state: TelemetryState) -> TelemetryState:
    """Correlates metrics, logs, and traces"""
    telemetry_data = state.get("telemetry_data", {})
    
    system_message = SystemMessage(content="""You are a correlation engine.
    Link metrics, logs, and traces to provide unified observability.""")
    
    user_message = HumanMessage(content=f"""Correlate telemetry:

Metrics: {len(telemetry_data.get('metrics', {}))}
Logs: {len(telemetry_data.get('logs', []))}
Traces: {len(telemetry_data.get('traces', []))}

Build correlation map.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Build correlation data
    correlation_data = {
        "trace_log_correlation": {},
        "metric_trace_correlation": {},
        "anomaly_correlation": []
    }
    
    # Correlate logs with traces
    for log in telemetry_data.get("logs", []):
        if log.get("trace_id"):
            if log["trace_id"] not in correlation_data["trace_log_correlation"]:
                correlation_data["trace_log_correlation"][log["trace_id"]] = []
            correlation_data["trace_log_correlation"][log["trace_id"]].append(log)
    
    # Detect correlated anomalies
    metrics = telemetry_data.get("metrics", {})
    if metrics.get("api-gateway", {}).get("http_errors_total", 0) > 50:
        correlation_data["anomaly_correlation"].append({
            "type": "high_error_rate",
            "source": "api-gateway",
            "metric": "http_errors_total",
            "value": metrics["api-gateway"]["http_errors_total"],
            "related_logs": [log for log in telemetry_data.get("logs", []) if log["source"] == "api-gateway"]
        })
    
    summary = f"""
    📊 TELEMETRY COMPLETE
    
    Telemetry Summary:
    • Sources: {len(state.get('sources', []))}
    • Metrics Categories: {len(telemetry_data.get('metrics', {}))}
    • Log Entries: {len(telemetry_data.get('logs', []))}
    • Traces: {len(telemetry_data.get('traces', []))}
    • Events: {len(telemetry_data.get('events', []))}
    • Exporters: {len(state.get('exporters', []))}
    
    Correlations Found:
    • Trace-Log Links: {len(correlation_data['trace_log_correlation'])}
    • Anomalies: {len(correlation_data['anomaly_correlation'])}
    
    Telemetry Pattern Process:
    1. Telemetry Collector → Gather all signals
    2. Correlation Engine → Link related data
    
    Correlation Benefits:
    
    Unified View:
    • Single pane of glass
    • Context switching reduced
    • Faster troubleshooting
    • Better insights
    
    Root Cause Analysis:
    • Error logs → traces
    • Slow requests → metrics
    • Anomalies → events
    • Full context
    
    Correlation Techniques:
    
    Trace ID Propagation:
    • Generate at entry point
    • Pass in headers (traceparent)
    • Include in all logs
    • Link spans to logs
    
    Temporal Correlation:
    • Time window matching
    • Before/after analysis
    • Event causation
    • Pattern detection
    
    Service Correlation:
    • Service dependency graph
    • Cross-service analysis
    • Cascade detection
    • Impact radius
    
    Metadata Correlation:
    • User ID
    • Session ID
    • Request ID
    • Custom attributes
    
    Telemetry Best Practices:
    
    Standardization:
    • Consistent naming
    • Common labels
    • Standard formats
    • Schema validation
    
    Context Propagation:
    • W3C Trace Context
    • Baggage for metadata
    • Consistent IDs
    • Cross-service correlation
    
    Resource Attribution:
    • Service name
    • Version
    • Environment
    • Cluster/region
    
    Cardinality Control:
    • Limit label values
    • Avoid high-cardinality IDs
    • Use exemplars
    • Smart sampling
    
    Privacy and Security:
    • Redact PII
    • Encrypt in transit
    • Access controls
    • Retention policies
    
    Cost Optimization:
    • Sampling strategies
    • Data retention tiers
    • Aggregation
    • Efficient storage
    
    Key Insight:
    Comprehensive telemetry with correlation across
    metrics, logs, and traces provides deep system
    understanding and enables rapid issue resolution.
    """
    
    return {
        "messages": [AIMessage(content=f"🔗 Correlation Engine:\n{response.content}\n{summary}")],
        "correlation_data": correlation_data
    }


# Build the graph
def build_telemetry_graph():
    """Build the telemetry pattern graph"""
    workflow = StateGraph(TelemetryState)
    
    workflow.add_node("telemetry_collector", telemetry_collector)
    workflow.add_node("correlation_engine", correlation_engine)
    
    workflow.add_edge(START, "telemetry_collector")
    workflow.add_edge("telemetry_collector", "correlation_engine")
    workflow.add_edge("correlation_engine", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_telemetry_graph()
    
    print("=== Telemetry MCP Pattern ===\n")
    
    # Test Case: Comprehensive telemetry collection
    print("\n" + "="*70)
    print("TEST CASE: Multi-Signal Telemetry Collection")
    print("="*70)
    
    state = {
        "messages": [],
        "telemetry_data": {},
        "sources": [],
        "collection_interval_seconds": 15,
        "exporters": [],
        "correlation_data": {}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nTelemetry Results:")
    print(f"Sources: {len(result.get('sources', []))}")
    print(f"Metrics: {len(result.get('telemetry_data', {}).get('metrics', {}))}")
    print(f"Logs: {len(result.get('telemetry_data', {}).get('logs', []))}")
    print(f"Traces: {len(result.get('telemetry_data', {}).get('traces', []))}")
