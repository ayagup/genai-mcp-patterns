"""
Tracing MCP Pattern

This pattern implements distributed tracing to track requests across
multiple services and identify performance bottlenecks.

Key Features:
- Distributed request tracing
- Span collection and correlation
- Trace assembly and visualization
- Performance bottleneck detection
- Service dependency mapping
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class TracingState(TypedDict):
    """State for tracing pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tracing_backend: str  # "jaeger", "zipkin", "xray", "datadog"
    trace_id: str
    spans: List[Dict]  # [{span_id, parent_span_id, service, operation, duration_ms, tags}]
    services_involved: List[str]
    total_duration_ms: float
    bottlenecks: List[Dict]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Tracer
def tracer(state: TracingState) -> TracingState:
    """Creates and manages trace spans"""
    tracing_backend = state.get("tracing_backend", "jaeger")
    
    system_message = SystemMessage(content="""You are a distributed tracer.
    Create and track spans across multiple services.""")
    
    user_message = HumanMessage(content=f"""Initialize tracing:

Backend: {tracing_backend}

Set up distributed tracing.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate trace with multiple spans
    import uuid
    trace_id = str(uuid.uuid4())
    
    spans = [
        {
            "span_id": "span-1",
            "parent_span_id": None,
            "service": "api-gateway",
            "operation": "HTTP GET /api/users/123",
            "start_time": int(time.time() * 1000),
            "duration_ms": 245.5,
            "tags": {"http.method": "GET", "http.status_code": 200, "http.url": "/api/users/123"}
        },
        {
            "span_id": "span-2",
            "parent_span_id": "span-1",
            "service": "auth-service",
            "operation": "validateToken",
            "start_time": int(time.time() * 1000) + 5,
            "duration_ms": 15.2,
            "tags": {"auth.user_id": "123", "auth.method": "jwt"}
        },
        {
            "span_id": "span-3",
            "parent_span_id": "span-1",
            "service": "user-service",
            "operation": "getUserById",
            "start_time": int(time.time() * 1000) + 25,
            "duration_ms": 180.3,
            "tags": {"db.system": "postgresql", "db.statement": "SELECT * FROM users WHERE id=?"}
        },
        {
            "span_id": "span-4",
            "parent_span_id": "span-3",
            "service": "database",
            "operation": "SELECT query",
            "start_time": int(time.time() * 1000) + 30,
            "duration_ms": 165.8,
            "tags": {"db.type": "sql", "db.instance": "users-db"}
        },
        {
            "span_id": "span-5",
            "parent_span_id": "span-1",
            "service": "cache-service",
            "operation": "get",
            "start_time": int(time.time() * 1000) + 220,
            "duration_ms": 2.1,
            "tags": {"cache.key": "user:123", "cache.hit": "true"}
        }
    ]
    
    services_involved = list(set(span["service"] for span in spans))
    total_duration_ms = max(span["start_time"] + span["duration_ms"] for span in spans) - min(span["start_time"] for span in spans)
    
    report = f"""
    🔍 Distributed Tracing:
    
    Trace Overview:
    • Trace ID: {trace_id[:8]}...
    • Backend: {tracing_backend.upper()}
    • Total Spans: {len(spans)}
    • Services: {len(services_involved)}
    • Total Duration: {total_duration_ms:.2f}ms
    
    Distributed Tracing Concepts:
    
    Trace:
    • End-to-end request journey
    • Unique trace ID
    • Multiple spans
    • Service dependencies
    • Performance view
    
    Span:
    • Single operation unit
    • Unique span ID
    • Parent-child relationship
    • Start time + duration
    • Tags and logs
    • Service name
    
    Context Propagation:
    • Trace context in headers
    • W3C Trace Context standard
    • traceparent header
    • tracestate header
    • Baggage items
    
    Tracing Backends:
    
    Jaeger (CNCF):
    • OpenTelemetry compatible
    • Uber origin
    • Service dependency graph
    • Root cause analysis
    • Adaptive sampling
    
    Zipkin:
    • Twitter origin
    • Simple architecture
    • Multiple transports
    • B3 propagation
    • Web UI
    
    AWS X-Ray:
    • AWS native
    • Service map
    • Trace analytics
    • Lambda integration
    • Sampling rules
    
    Datadog APM:
    • Full observability
    • Auto-instrumentation
    • Service catalog
    • Analytics
    • Alerting
    
    OpenTelemetry:
    • Vendor-neutral standard
    • Traces, metrics, logs
    • Auto-instrumentation
    • Multiple exporters
    • CNCF graduated
    
    Span Types:
    
    Client Span:
    • Outbound RPC call
    • HTTP request
    • Database query
    • Cache lookup
    
    Server Span:
    • Inbound RPC
    • HTTP handler
    • Message consumer
    • Service entry
    
    Internal Span:
    • Function call
    • Business logic
    • Computation
    • Internal operation
    
    Instrumentation:
    
    Automatic (Python):
    ```python
    from opentelemetry import trace
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    
    # Setup
    trace.set_tracer_provider(TracerProvider())
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831
    )
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
    # Auto-instrument Flask
    FlaskInstrumentor().instrument_app(app)
    ```
    
    Manual (Python):
    ```python
    from opentelemetry import trace
    
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("operation") as span:
        span.set_attribute("user.id", "123")
        span.add_event("Processing started")
        
        # Do work
        result = process_data()
        
        span.set_attribute("result.count", len(result))
    ```
    
    Context Propagation (HTTP):
    ```python
    # Inject trace context into headers
    from opentelemetry.propagate import inject
    
    headers = {{}}
    inject(headers)
    response = requests.get(url, headers=headers)
    
    # Extract trace context from headers
    from opentelemetry.propagate import extract
    
    ctx = extract(request.headers)
    with tracer.start_as_current_span("handler", context=ctx):
        handle_request()
    ```
    
    Trace Sampling:
    
    Always Sample:
    • All traces collected
    • High overhead
    • Complete visibility
    • Expensive at scale
    
    Probabilistic:
    • Sample N% of traces
    • Consistent sampling
    • Scalable
    • May miss rare issues
    
    Rate Limiting:
    • Max traces per second
    • Prevents overload
    • Predictable cost
    • May miss bursts
    
    Adaptive:
    • Adjust based on load
    • Always sample errors
    • Increase for slow requests
    • ML-driven decisions
    
    Tags and Annotations:
    
    Standard Tags:
    • span.kind: client/server/producer/consumer
    • component: framework/library name
    • error: true/false
    • http.method: GET/POST/etc
    • http.status_code: 200/404/etc
    • db.system: postgresql/mongodb
    • messaging.system: kafka/rabbitmq
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Tracer:\n{response.content}\n{report}")],
        "trace_id": trace_id,
        "spans": spans,
        "services_involved": services_involved,
        "total_duration_ms": total_duration_ms
    }


# Trace Analyzer
def trace_analyzer(state: TracingState) -> TracingState:
    """Analyzes traces to find bottlenecks"""
    spans = state.get("spans", [])
    total_duration_ms = state.get("total_duration_ms", 0.0)
    
    system_message = SystemMessage(content="""You are a trace analyzer.
    Identify performance bottlenecks and optimization opportunities.""")
    
    user_message = HumanMessage(content=f"""Analyze trace:

Total Spans: {len(spans)}
Duration: {total_duration_ms:.2f}ms

Identify bottlenecks.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Identify bottlenecks (spans taking > 50% of total time)
    bottlenecks = []
    for span in spans:
        if span["duration_ms"] > total_duration_ms * 0.5:
            bottlenecks.append({
                "service": span["service"],
                "operation": span["operation"],
                "duration_ms": span["duration_ms"],
                "percentage": (span["duration_ms"] / total_duration_ms * 100)
            })
    
    summary = f"""
    📊 TRACING COMPLETE
    
    Trace Analysis:
    • Trace ID: {state.get('trace_id', 'N/A')[:8]}...
    • Total Spans: {len(spans)}
    • Services: {len(state.get('services_involved', []))}
    • Total Duration: {total_duration_ms:.2f}ms
    • Bottlenecks Found: {len(bottlenecks)}
    
    {chr(10).join(f"• {b['service']} - {b['operation']}: {b['duration_ms']:.2f}ms ({b['percentage']:.1f}%)" for b in bottlenecks) if bottlenecks else "• No major bottlenecks detected"}
    
    Tracing Pattern Process:
    1. Tracer → Create and collect spans
    2. Trace Analyzer → Identify bottlenecks
    
    Trace Visualization:
    
    Waterfall View:
    ```
    api-gateway [HTTP GET]          |████████████████| 245ms
      ├─ auth-service [validate]    |█|               15ms
      ├─ user-service [getUserById] |    ████████████| 180ms
      │   └─ database [SELECT]      |     ███████████| 165ms
      └─ cache-service [get]        |               █| 2ms
    ```
    
    Service Graph:
    ```
    [api-gateway] → [auth-service]
                  → [user-service] → [database]
                  → [cache-service]
    ```
    
    Use Cases:
    
    Performance Debugging:
    • Identify slow services
    • Find N+1 query problems
    • Detect serial processing
    • Optimize critical paths
    
    Dependency Analysis:
    • Service dependencies
    • Call patterns
    • Circular dependencies
    • Coupling detection
    
    Error Tracking:
    • Error propagation
    • Failure points
    • Cascading failures
    • Root cause analysis
    
    Capacity Planning:
    • Service load
    • Resource usage
    • Scaling needs
    • Cost optimization
    
    Best Practices:
    
    Instrumentation:
    • Instrument at boundaries
    • Add business context
    • Include error details
    • Avoid sensitive data
    
    Sampling:
    • Always trace errors
    • Sample high-traffic paths
    • Adjust based on volume
    • Monitor sampling rate
    
    Performance:
    • Async span export
    • Batch processing
    • Resource limits
    • Sampling strategies
    
    Privacy:
    • Redact PII
    • Filter sensitive headers
    • Compliance requirements
    • Data retention policies
    
    Key Insight:
    Distributed tracing provides end-to-end visibility
    across microservices, enabling rapid troubleshooting
    and performance optimization.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Trace Analyzer:\n{response.content}\n{summary}")],
        "bottlenecks": bottlenecks
    }


# Build the graph
def build_tracing_graph():
    """Build the tracing pattern graph"""
    workflow = StateGraph(TracingState)
    
    workflow.add_node("tracer", tracer)
    workflow.add_node("analyzer", trace_analyzer)
    
    workflow.add_edge(START, "tracer")
    workflow.add_edge("tracer", "analyzer")
    workflow.add_edge("analyzer", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_tracing_graph()
    
    print("=== Tracing MCP Pattern ===\n")
    
    # Test Case: Distributed request tracing
    print("\n" + "="*70)
    print("TEST CASE: Multi-Service Request Trace")
    print("="*70)
    
    state = {
        "messages": [],
        "tracing_backend": "jaeger",
        "trace_id": "",
        "spans": [],
        "services_involved": [],
        "total_duration_ms": 0.0,
        "bottlenecks": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nTrace ID: {result.get('trace_id', 'N/A')[:16]}...")
    print(f"Total Spans: {len(result.get('spans', []))}")
    print(f"Services: {len(result.get('services_involved', []))}")
    print(f"Bottlenecks: {len(result.get('bottlenecks', []))}")
