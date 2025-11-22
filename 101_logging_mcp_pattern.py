"""
Logging MCP Pattern

This pattern provides structured logging capabilities for distributed systems,
collecting, formatting, and analyzing log data from multiple sources.

Key Features:
- Structured logging with levels
- Log aggregation from multiple sources
- Log formatting and enrichment
- Log analysis and search
- Correlation and context tracking
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class LoggingState(TypedDict):
    """State for logging pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    log_level: str  # "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
    log_format: str  # "JSON", "TEXT", "STRUCTURED"
    log_entries: List[Dict]  # [{timestamp, level, message, context, source}]
    sources: List[str]  # List of log sources
    total_logs: int
    error_count: int
    warning_count: int
    correlation_ids: Dict[str, List[int]]  # correlation_id -> log indices


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Log Collector
def log_collector(state: LoggingState) -> LoggingState:
    """Collects logs from multiple sources"""
    sources = state.get("sources", ["app-server", "database", "cache"])
    log_level = state.get("log_level", "INFO")
    
    system_message = SystemMessage(content="""You are a log collector.
    Collect log entries from multiple sources in a distributed system.""")
    
    user_message = HumanMessage(content=f"""Collect logs:

Sources: {', '.join(sources)}
Minimum Level: {log_level}

Gather log entries from all sources.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate log collection
    import uuid
    log_entries = [
        {
            "timestamp": int(time.time() * 1000),
            "level": "INFO",
            "message": "Application started successfully",
            "context": {"service": "app-server", "version": "1.2.3"},
            "source": "app-server",
            "correlation_id": str(uuid.uuid4())
        },
        {
            "timestamp": int(time.time() * 1000) + 100,
            "level": "DEBUG",
            "message": "Database connection established",
            "context": {"service": "database", "pool_size": 10},
            "source": "database",
            "correlation_id": str(uuid.uuid4())
        },
        {
            "timestamp": int(time.time() * 1000) + 200,
            "level": "WARN",
            "message": "Cache miss for key: user_123",
            "context": {"service": "cache", "key": "user_123"},
            "source": "cache",
            "correlation_id": str(uuid.uuid4())
        },
        {
            "timestamp": int(time.time() * 1000) + 300,
            "level": "ERROR",
            "message": "Failed to connect to external API",
            "context": {"service": "app-server", "api": "payment-gateway", "retry": 3},
            "source": "app-server",
            "correlation_id": str(uuid.uuid4())
        },
        {
            "timestamp": int(time.time() * 1000) + 400,
            "level": "INFO",
            "message": "Request processed successfully",
            "context": {"service": "app-server", "duration_ms": 45, "status": 200},
            "source": "app-server",
            "correlation_id": str(uuid.uuid4())
        }
    ]
    
    total_logs = len(log_entries)
    error_count = sum(1 for log in log_entries if log["level"] == "ERROR")
    warning_count = sum(1 for log in log_entries if log["level"] == "WARN")
    
    report = f"""
    📝 Log Collection:
    
    Collection Summary:
    • Sources: {len(sources)}
    • Total Logs: {total_logs}
    • Errors: {error_count}
    • Warnings: {warning_count}
    • Format: Structured JSON
    
    Log Levels (Severity):
    
    TRACE/DEBUG:
    • Detailed diagnostic information
    • Development and troubleshooting
    • High volume
    • Usually disabled in production
    
    INFO:
    • General informational messages
    • Application lifecycle events
    • Normal operations
    • Audit trail
    
    WARN:
    • Warning messages
    • Potential issues
    • Degraded functionality
    • Requires attention
    
    ERROR:
    • Error conditions
    • Failed operations
    • Exceptions caught
    • Requires investigation
    
    FATAL/CRITICAL:
    • Severe errors
    • Application crash
    • Data corruption
    • Immediate action required
    
    Structured Logging Best Practices:
    
    JSON Format:
    ```json
    {{
      "timestamp": "2024-01-01T12:00:00Z",
      "level": "ERROR",
      "message": "Database connection failed",
      "context": {{
        "service": "user-service",
        "database": "postgres",
        "error": "connection timeout",
        "retry_count": 3
      }},
      "trace_id": "abc-123",
      "span_id": "xyz-789",
      "source": "user-service-pod-1"
    }}
    ```
    
    Python Structured Logging:
    ```python
    import structlog
    
    logger = structlog.get_logger()
    
    logger.info(
        "user_login",
        user_id="123",
        ip_address="192.168.1.1",
        success=True,
        duration_ms=45
    )
    ```
    
    Log Aggregation Tools:
    
    ELK Stack:
    • Elasticsearch: Search and analytics
    • Logstash: Log processing pipeline
    • Kibana: Visualization and dashboards
    
    Splunk:
    • Enterprise log management
    • Machine learning analytics
    • Security monitoring
    • Compliance reporting
    
    Datadog:
    • Cloud-native logging
    • APM integration
    • Real-time analytics
    • Alerting
    
    Loki (Grafana):
    • Prometheus-inspired
    • Label-based indexing
    • Cost-effective
    • Kubernetes-native
    
    CloudWatch Logs:
    • AWS native
    • Insights queries
    • Metric filters
    • Log groups and streams
    
    Log Collection Methods:
    
    Agent-Based:
    • Filebeat, Fluentd, Logstash
    • Tail log files
    • Forward to aggregator
    • Low application overhead
    
    Library-Based:
    • Direct logging to service
    • Application integration
    • Structured from source
    • Network dependency
    
    Sidecar Pattern:
    • Container per pod
    • Log extraction
    • Kubernetes common
    • Resource overhead
    
    Log Enrichment:
    • Add metadata (host, pod, region)
    • Correlation IDs
    • User context
    • Environment info
    • Timestamps normalization
    """
    
    return {
        "messages": [AIMessage(content=f"📝 Log Collector:\n{response.content}\n{report}")],
        "log_entries": log_entries,
        "total_logs": total_logs,
        "error_count": error_count,
        "warning_count": warning_count
    }


# Log Formatter
def log_formatter(state: LoggingState) -> LoggingState:
    """Formats and enriches log entries"""
    log_entries = state.get("log_entries", [])
    log_format = state.get("log_format", "JSON")
    
    system_message = SystemMessage(content="""You are a log formatter.
    Format and enrich log entries for consistency and searchability.""")
    
    user_message = HumanMessage(content=f"""Format logs:

Total Entries: {len(log_entries)}
Format: {log_format}

Standardize and enrich log data.""")
    
    response = llm.invoke([system_message, user_message])
    
    report = f"""
    🎨 Log Formatting:
    
    Formatting Applied:
    • Format: {log_format}
    • Entries Processed: {len(log_entries)}
    • Timestamp: ISO 8601
    • Enrichment: Added metadata
    
    Log Format Standards:
    
    Common Log Format (CLF):
    ```
    127.0.0.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234
    ```
    
    JSON Format:
    ```json
    {{
      "@timestamp": "2024-01-01T12:00:00.000Z",
      "level": "INFO",
      "logger": "com.example.UserService",
      "message": "User login successful",
      "thread": "http-nio-8080-exec-1",
      "context": {{
        "user_id": "123",
        "session_id": "abc-xyz",
        "ip": "192.168.1.1"
      }}
    }}
    ```
    
    Logfmt:
    ```
    ts=2024-01-01T12:00:00Z level=info msg="User login" user_id=123 ip=192.168.1.1
    ```
    
    CEF (Common Event Format):
    ```
    CEF:0|Vendor|Product|Version|EventID|Name|Severity|Extension
    ```
    
    Field Standardization:
    
    Timestamp:
    • ISO 8601 format
    • UTC timezone
    • Millisecond precision
    • Consistent parsing
    
    Level Mapping:
    • TRACE → 0
    • DEBUG → 1
    • INFO → 2
    • WARN → 3
    • ERROR → 4
    • FATAL → 5
    
    Contextual Fields:
    • user_id: User identifier
    • session_id: Session tracking
    • request_id: Request correlation
    • trace_id: Distributed tracing
    • span_id: Trace span
    • service: Service name
    • environment: prod/staging/dev
    • version: Application version
    • host: Hostname/IP
    • pod: Kubernetes pod name
    
    Log Enrichment Techniques:
    
    Automatic:
    • Timestamp injection
    • Hostname/IP addition
    • Process ID
    • Thread ID
    • Stack trace (errors)
    
    Contextual:
    • Request headers
    • User information
    • Session data
    • Correlation IDs
    • Business context
    
    Derived:
    • Geolocation from IP
    • User agent parsing
    • URL parsing
    • Duration calculation
    • Status categorization
    
    Performance Considerations:
    
    Async Logging:
    • Non-blocking I/O
    • Background threads
    • Ring buffer
    • Batching
    
    Sampling:
    • Log only N% of requests
    • Adaptive sampling
    • Priority-based
    • Error always logged
    
    Filtering:
    • Level-based
    • Source-based
    • Pattern-based
    • Rate limiting
    """
    
    return {
        "messages": [AIMessage(content=f"🎨 Log Formatter:\n{response.content}\n{report}")]
    }


# Log Aggregator
def log_aggregator(state: LoggingState) -> LoggingState:
    """Aggregates and correlates log entries"""
    log_entries = state.get("log_entries", [])
    
    system_message = SystemMessage(content="""You are a log aggregator.
    Aggregate logs and identify correlated events.""")
    
    user_message = HumanMessage(content=f"""Aggregate logs:

Total Entries: {len(log_entries)}

Group and correlate related log entries.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Build correlation map
    correlation_ids = {}
    for idx, log in enumerate(log_entries):
        corr_id = log.get("correlation_id", "unknown")
        if corr_id not in correlation_ids:
            correlation_ids[corr_id] = []
        correlation_ids[corr_id].append(idx)
    
    report = f"""
    🔗 Log Aggregation:
    
    Aggregation Results:
    • Total Entries: {len(log_entries)}
    • Unique Correlations: {len(correlation_ids)}
    • Sources: {len(set(log.get('source', '') for log in log_entries))}
    
    Log Aggregation Patterns:
    
    Time-Based:
    • Group by time windows
    • 1-minute, 5-minute intervals
    • Trending analysis
    • Rate calculation
    
    Source-Based:
    • Group by service
    • Group by host
    • Group by environment
    • Cross-service view
    
    Correlation-Based:
    • Request ID tracking
    • Trace ID grouping
    • Session grouping
    • Transaction flows
    
    Level-Based:
    • Errors only
    • Warnings and above
    • Debug excluded
    • Severity filtering
    
    Correlation Techniques:
    
    Request Tracing:
    ```python
    import uuid
    
    # Generate correlation ID
    correlation_id = str(uuid.uuid4())
    
    # Add to all logs in request
    logger.info("Processing request", 
                correlation_id=correlation_id)
    
    # Pass to downstream services
    headers = {{'X-Correlation-ID': correlation_id}}
    ```
    
    Distributed Tracing:
    • OpenTelemetry
    • Jaeger
    • Zipkin
    • AWS X-Ray
    • Google Cloud Trace
    
    Log Aggregation Architecture:
    
    Centralized:
    • Single aggregation point
    • Simple architecture
    • Single point of failure
    • Scalability limit
    
    Distributed:
    • Multiple aggregators
    • Regional deployment
    • High availability
    • Complex coordination
    
    Hierarchical:
    • Local aggregation
    • Regional roll-up
    • Global view
    • Reduces network traffic
    
    Query Patterns:
    
    Full-Text Search:
    ```
    message:"database connection failed"
    ```
    
    Field Search:
    ```
    level:ERROR AND service:user-api
    ```
    
    Time Range:
    ```
    @timestamp:[2024-01-01 TO 2024-01-02]
    ```
    
    Aggregations:
    ```
    COUNT(*) GROUP BY level
    AVG(duration_ms) GROUP BY endpoint
    ```
    
    Best Practices:
    
    Retention:
    • Hot: 7-30 days (fast query)
    • Warm: 30-90 days (slower)
    • Cold: 90+ days (archive)
    • Compliance-based retention
    
    Indexing:
    • Index important fields
    • Full-text search fields
    • Time-series optimization
    • Balance query speed vs storage
    
    Security:
    • Encrypt in transit
    • Encrypt at rest
    • Access control
    • PII masking/redaction
    • Audit log access
    """
    
    return {
        "messages": [AIMessage(content=f"🔗 Log Aggregator:\n{response.content}\n{report}")],
        "correlation_ids": correlation_ids
    }


# Log Analyzer
def log_analyzer(state: LoggingState) -> LoggingState:
    """Analyzes logs for patterns and insights"""
    log_entries = state.get("log_entries", [])
    error_count = state.get("error_count", 0)
    warning_count = state.get("warning_count", 0)
    total_logs = state.get("total_logs", 0)
    correlation_ids = state.get("correlation_ids", {})
    
    error_rate = (error_count / total_logs * 100) if total_logs > 0 else 0
    
    summary = f"""
    📊 LOGGING COMPLETE
    
    Log Analysis Summary:
    • Total Logs: {total_logs}
    • Errors: {error_count} ({error_rate:.1f}%)
    • Warnings: {warning_count}
    • Correlations: {len(correlation_ids)}
    • Sources: {len(state.get('sources', []))}
    
    Logging Pattern Process:
    1. Log Collector → Gather logs from sources
    2. Log Formatter → Standardize and enrich
    3. Log Aggregator → Correlate related logs
    4. Log Analyzer → Extract insights
    
    Log Analysis Techniques:
    
    Pattern Detection:
    • Error patterns
    • Anomaly detection
    • Trend analysis
    • Seasonality
    • Correlation discovery
    
    Machine Learning:
    • Log clustering
    • Anomaly detection
    • Failure prediction
    • Root cause analysis
    • Auto-categorization
    
    Statistical Analysis:
    • Error rate trends
    • Response time distribution
    • Request volume patterns
    • Resource utilization
    • SLA compliance
    
    Real-World Use Cases:
    
    Debugging:
    • Trace request flow
    • Identify error source
    • Reproduce issues
    • Performance bottlenecks
    
    Security:
    • Failed login attempts
    • Suspicious patterns
    • Data access audit
    • Compliance reporting
    
    Performance:
    • Slow queries
    • High latency endpoints
    • Resource contention
    • Optimization opportunities
    
    Business Intelligence:
    • User behavior
    • Feature usage
    • Conversion funnels
    • A/B test results
    
    Log Analysis Tools:
    
    Elasticsearch Queries:
    ```json
    {{
      "query": {{
        "bool": {{
          "must": [
            {{"match": {{"level": "ERROR"}}}},
            {{"range": {{"@timestamp": {{"gte": "now-1h"}}}}}}
          ]
        }}
      }},
      "aggs": {{
        "errors_by_service": {{
          "terms": {{"field": "service.keyword"}}
        }}
      }}
    }}
    ```
    
    Splunk SPL:
    ```
    index=production level=ERROR
    | stats count by service, error_type
    | where count > 10
    ```
    
    CloudWatch Insights:
    ```
    fields @timestamp, level, message
    | filter level = "ERROR"
    | stats count() by service
    | sort count desc
    ```
    
    Key Metrics to Track:
    
    Availability:
    • Error rate
    • Success rate
    • Uptime percentage
    
    Performance:
    • Response time (p50, p95, p99)
    • Throughput (requests/sec)
    • Resource usage
    
    Quality:
    • Error types distribution
    • Warning trends
    • Exception patterns
    
    Business:
    • User actions
    • Transaction volume
    • Conversion rates
    
    Best Practices:
    
    Log What Matters:
    • Business events
    • Errors and exceptions
    • Performance metrics
    • Security events
    • State changes
    
    Don't Log:
    • Passwords, tokens, secrets
    • PII without masking
    • Excessive debug data in prod
    • Binary data
    • High-frequency loops
    
    Log Consistently:
    • Standard format across services
    • Consistent field names
    • Correlation IDs everywhere
    • Same timezone (UTC)
    
    Monitor Your Logs:
    • Log volume trends
    • Error rate alerts
    • Missing logs (gaps)
    • Storage utilization
    • Query performance
    
    Key Insight:
    Effective logging is essential for observability.
    Use structured logging, correlate events, and
    analyze patterns to maintain system health.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Log Analyzer:\n{summary}")]
    }


# Build the graph
def build_logging_graph():
    """Build the logging pattern graph"""
    workflow = StateGraph(LoggingState)
    
    workflow.add_node("collector", log_collector)
    workflow.add_node("formatter", log_formatter)
    workflow.add_node("aggregator", log_aggregator)
    workflow.add_node("analyzer", log_analyzer)
    
    workflow.add_edge(START, "collector")
    workflow.add_edge("collector", "formatter")
    workflow.add_edge("formatter", "aggregator")
    workflow.add_edge("aggregator", "analyzer")
    workflow.add_edge("analyzer", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_logging_graph()
    
    print("=== Logging MCP Pattern ===\n")
    
    # Test Case: Distributed system logging
    print("\n" + "="*70)
    print("TEST CASE: Multi-Service Log Aggregation")
    print("="*70)
    
    state = {
        "messages": [],
        "log_level": "DEBUG",
        "log_format": "JSON",
        "log_entries": [],
        "sources": ["app-server", "database", "cache", "message-queue"],
        "total_logs": 0,
        "error_count": 0,
        "warning_count": 0,
        "correlation_ids": {}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nTotal Logs Collected: {result.get('total_logs', 0)}")
    print(f"Errors: {result.get('error_count', 0)}")
    print(f"Warnings: {result.get('warning_count', 0)}")
    print(f"Unique Correlations: {len(result.get('correlation_ids', {}))}")
