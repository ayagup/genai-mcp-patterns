"""
Analytics MCP Pattern

This pattern implements advanced analytics capabilities for logs,
metrics, and traces to extract insights and detect patterns.

Key Features:
- Pattern detection in logs
- Anomaly detection in metrics
- Trend analysis
- Predictive analytics
- Business intelligence insights
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class AnalyticsState(TypedDict):
    """State for analytics pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    data_source: str  # "logs", "metrics", "traces"
    analysis_type: str  # "pattern", "anomaly", "trend", "prediction"
    input_data: Dict
    patterns_detected: List[Dict]
    anomalies: List[Dict]
    insights: List[str]
    recommendations: List[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Pattern Analyzer
def pattern_analyzer(state: AnalyticsState) -> AnalyticsState:
    """Analyzes data to detect patterns and anomalies"""
    data_source = state.get("data_source", "logs")
    analysis_type = state.get("analysis_type", "pattern")
    
    system_message = SystemMessage(content="""You are an analytics engine.
    Detect patterns, anomalies, and trends in observability data.""")
    
    user_message = HumanMessage(content=f"""Analyze data:

Source: {data_source}
Type: {analysis_type}

Perform analytics.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate input data
    input_data = {
        "logs": {
            "total_entries": 125000,
            "time_range": "last_24h",
            "error_count": 3456,
            "warning_count": 8923,
            "sources": ["api-gateway", "user-service", "database", "cache"]
        },
        "metrics": {
            "request_rate": [1234, 1245, 1298, 1256, 1287, 1345, 1423],
            "error_rate": [0.5, 0.6, 0.5, 0.7, 2.3, 1.8, 0.6],
            "response_time_p95": [145, 148, 152, 156, 245, 198, 151]
        },
        "traces": {
            "total_traces": 45000,
            "avg_duration_ms": 234.5,
            "slow_traces": 234,
            "error_traces": 89
        }
    }
    
    # Detect patterns
    patterns_detected = [
        {
            "type": "error_spike",
            "description": "Sudden increase in error rate at 14:30 UTC",
            "affected_service": "api-gateway",
            "time_window": "14:30-14:45 UTC",
            "severity": "high",
            "details": {
                "normal_rate": 0.5,
                "spike_rate": 2.3,
                "increase_percentage": 360
            }
        },
        {
            "type": "slow_query_pattern",
            "description": "Recurring slow database queries during peak hours",
            "affected_service": "database",
            "time_window": "12:00-14:00 daily",
            "severity": "medium",
            "details": {
                "query_type": "SELECT with JOIN",
                "avg_duration_ms": 1250,
                "frequency": "~50/hour"
            }
        },
        {
            "type": "cascade_failure",
            "description": "Service failures propagating from database to API",
            "affected_services": ["database", "user-service", "api-gateway"],
            "time_window": "14:30-14:32 UTC",
            "severity": "critical",
            "details": {
                "root_cause": "database connection pool exhaustion",
                "propagation_time": "~2 minutes",
                "services_affected": 3
            }
        }
    ]
    
    # Detect anomalies
    anomalies = [
        {
            "metric": "response_time_p95",
            "timestamp": "2024-01-15 14:30:00",
            "value": 245.0,
            "expected_range": [140, 160],
            "deviation": 153.125,  # percentage
            "confidence": 0.95
        },
        {
            "metric": "error_rate",
            "timestamp": "2024-01-15 14:30:00",
            "value": 2.3,
            "expected_range": [0.4, 0.7],
            "deviation": 328.57,  # percentage
            "confidence": 0.98
        }
    ]
    
    report = f"""
    🔍 Pattern Analyzer:
    
    Analysis Overview:
    • Data Source: {data_source.capitalize()}
    • Analysis Type: {analysis_type.capitalize()}
    • Time Range: {input_data['logs']['time_range']}
    • Patterns Found: {len(patterns_detected)}
    • Anomalies Detected: {len(anomalies)}
    
    Analytics Concepts:
    
    Pattern Detection:
    
    Time-based Patterns:
    • Daily/weekly cycles
    • Peak hour traffic
    • Seasonal trends
    • Maintenance windows
    • Business hours vs off-hours
    
    Error Patterns:
    • Recurring errors
    • Error correlation
    • Cascade failures
    • Retry storms
    • Circuit breaker trips
    
    Performance Patterns:
    • Slow queries
    • Memory leaks
    • CPU spikes
    • N+1 query problems
    • Cache miss patterns
    
    User Behavior Patterns:
    • Usage patterns
    • Feature adoption
    • Conversion funnels
    • Churn indicators
    • Geographic distribution
    
    Anomaly Detection Methods:
    
    Statistical:
    • Standard deviation (Z-score)
    • Interquartile range (IQR)
    • Moving averages
    • Seasonal decomposition
    • Grubbs' test
    
    Python Example (Z-score):
    ```python
    import numpy as np
    from scipy import stats
    
    def detect_anomalies_zscore(data, threshold=3):
        mean = np.mean(data)
        std = np.std(data)
        z_scores = [(x - mean) / std for x in data]
        
        anomalies = []
        for i, z in enumerate(z_scores):
            if abs(z) > threshold:
                anomalies.append({{
                    "index": i,
                    "value": data[i],
                    "z_score": z,
                    "deviation": abs((data[i] - mean) / mean * 100)
                }})
        
        return anomalies
    
    # Usage
    response_times = [145, 148, 152, 156, 245, 198, 151]
    anomalies = detect_anomalies_zscore(response_times)
    print(f"Anomalies: {{anomalies}}")
    ```
    
    Machine Learning:
    • Isolation Forest
    • One-class SVM
    • LSTM Autoencoders
    • DBSCAN clustering
    • Local Outlier Factor (LOF)
    
    Scikit-learn Example:
    ```python
    from sklearn.ensemble import IsolationForest
    import numpy as np
    
    # Training data (normal behavior)
    X_train = np.array([[1234], [1245], [1256], [1287]])
    
    # Test data (includes anomaly)
    X_test = np.array([[1298], [3456], [1312]])
    
    # Train model
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_train)
    
    # Predict (-1 for anomaly, 1 for normal)
    predictions = clf.predict(X_test)
    anomaly_scores = clf.score_samples(X_test)
    
    for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
        if pred == -1:
            print(f"Anomaly detected: {{X_test[i][0]}} (score: {{score:.3f}})")
    ```
    
    Time-Series Anomaly Detection (Prophet):
    ```python
    from prophet import Prophet
    import pandas as pd
    
    # Prepare data
    df = pd.DataFrame({{
        'ds': pd.date_range('2024-01-01', periods=100, freq='H'),
        'y': response_times  # Your metric values
    }})
    
    # Train model
    model = Prophet(interval_width=0.95)
    model.fit(df)
    
    # Make predictions
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    
    # Detect anomalies
    df['predicted'] = forecast['yhat'][:len(df)]
    df['upper_bound'] = forecast['yhat_upper'][:len(df)]
    df['lower_bound'] = forecast['yhat_lower'][:len(df)]
    
    df['anomaly'] = ((df['y'] > df['upper_bound']) | 
                     (df['y'] < df['lower_bound']))
    
    anomalies = df[df['anomaly']]
    print(f"Detected {{len(anomalies)}} anomalies")
    ```
    
    Trend Analysis:
    
    Linear Regression:
    ```python
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # Time series data
    X = np.array(range(len(metrics))).reshape(-1, 1)
    y = np.array(metrics)
    
    model = LinearRegression()
    model.fit(X, y)
    
    trend = model.coef_[0]
    if trend > 0:
        print(f"Upward trend: +{{trend:.2f}} per period")
    else:
        print(f"Downward trend: {{trend:.2f}} per period")
    ```
    
    Moving Averages:
    ```python
    def simple_moving_average(data, window=5):
        sma = []
        for i in range(len(data)):
            if i < window - 1:
                sma.append(None)
            else:
                sma.append(sum(data[i-window+1:i+1]) / window)
        return sma
    
    def exponential_moving_average(data, alpha=0.3):
        ema = [data[0]]
        for i in range(1, len(data)):
            ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
        return ema
    ```
    
    Log Analytics:
    
    Elasticsearch Aggregations:
    ```json
    {{
      "aggs": {{
        "errors_over_time": {{
          "date_histogram": {{
            "field": "@timestamp",
            "interval": "1h"
          }},
          "aggs": {{
            "error_count": {{
              "filter": {{
                "term": {{"level": "ERROR"}}
              }}
            }}
          }}
        }},
        "top_error_messages": {{
          "terms": {{
            "field": "message.keyword",
            "size": 10
          }}
        }},
        "error_rate_by_service": {{
          "terms": {{
            "field": "service.keyword"
          }},
          "aggs": {{
            "error_percentage": {{
              "bucket_script": {{
                "buckets_path": {{
                  "errors": "errors>_count",
                  "total": "_count"
                }},
                "script": "params.errors / params.total * 100"
              }}
            }}
          }}
        }}
      }}
    }}
    ```
    
    Metric Analytics:
    
    PromQL Advanced Queries:
    ```promql
    # Anomaly detection (deviation from average)
    abs(
      rate(http_requests_total[5m])
      - 
      avg_over_time(rate(http_requests_total[5m])[1h:5m])
    ) > 
    stddev_over_time(rate(http_requests_total[5m])[1h:5m]) * 2
    
    # Predict future values (linear extrapolation)
    predict_linear(
      http_requests_total[1h], 
      3600  # 1 hour into future
    )
    
    # Correlation between metrics
    (
      rate(http_errors_total[5m])
      / 
      rate(http_requests_total[5m])
    ) > 0.01
    and
    deriv(http_response_time_seconds[5m]) > 0
    ```
    
    Trace Analytics:
    
    Service Dependency Analysis:
    • Call graph generation
    • Hotpath identification
    • Bottleneck detection
    • Critical path analysis
    
    Error Propagation:
    • Trace error rate
    • Failure correlation
    • Root cause service
    • Impact radius
    
    Business Intelligence:
    
    KPI Tracking:
    • Revenue metrics
    • User engagement
    • Conversion rates
    • Feature adoption
    • Customer satisfaction
    
    Cohort Analysis:
    • User retention
    • Feature usage over time
    • Churn prediction
    • Lifetime value
    
    A/B Test Analysis:
    • Statistical significance
    • Confidence intervals
    • Sample size calculation
    • P-value computation
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Pattern Analyzer:\n{response.content}\n{report}")],
        "input_data": input_data,
        "patterns_detected": patterns_detected,
        "anomalies": anomalies
    }


# Insight Generator
def insight_generator(state: AnalyticsState) -> AnalyticsState:
    """Generates insights and recommendations from analytics"""
    patterns = state.get("patterns_detected", [])
    anomalies = state.get("anomalies", [])
    
    system_message = SystemMessage(content="""You are an insight generator.
    Transform analytical findings into actionable insights.""")
    
    user_message = HumanMessage(content=f"""Generate insights:

Patterns: {len(patterns)}
Anomalies: {len(anomalies)}

Create actionable insights.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate insights
    insights = [
        "Error rate spiked by 360% at 14:30 UTC, correlating with database connection pool exhaustion",
        "Daily slow query pattern from 12:00-14:00 suggests need for query optimization or indexing",
        "Cascade failure pattern indicates tight coupling between services - consider circuit breakers",
        "Response time anomaly (245ms vs expected 140-160ms) suggests performance degradation",
        "Database is the single point of failure affecting 3 downstream services"
    ]
    
    # Generate recommendations
    recommendations = [
        "Increase database connection pool size from current limit",
        "Add indexes to frequently queried tables to reduce slow query frequency",
        "Implement circuit breakers between user-service and database to prevent cascade failures",
        "Set up alerts for error rate >1% and response time >200ms",
        "Consider read replicas or caching to reduce database load during peak hours",
        "Investigate root cause of 14:30 UTC incident with detailed trace analysis",
        "Implement auto-scaling for database connections based on traffic patterns"
    ]
    
    summary = f"""
    📊 ANALYTICS COMPLETE
    
    Analytics Summary:
    • Data Source: {state.get('data_source', 'N/A').capitalize()}
    • Patterns Detected: {len(patterns)}
    • Anomalies Found: {len(anomalies)}
    • Insights Generated: {len(insights)}
    • Recommendations: {len(recommendations)}
    
    Key Patterns:
    {chr(10).join(f"  {i+1}. {p['type'].replace('_', ' ').title()}: {p['description']} (Severity: {p['severity'].upper()})" for i, p in enumerate(patterns))}
    
    Critical Anomalies:
    {chr(10).join(f"  • {a['metric']}: {a['value']:.2f} (Expected: {a['expected_range']}, Deviation: {a['deviation']:.1f}%)" for a in anomalies)}
    
    Analytics Pattern Process:
    1. Pattern Analyzer → Detect patterns and anomalies
    2. Insight Generator → Generate actionable insights
    
    Top Insights:
    {chr(10).join(f"  • {insight}" for insight in insights[:3])}
    
    Priority Recommendations:
    {chr(10).join(f"  {i+1}. {rec}" for i, rec in enumerate(recommendations[:5]))}
    
    Advanced Analytics Techniques:
    
    Root Cause Analysis:
    • 5 Whys method
    • Fishbone diagrams
    • Fault tree analysis
    • Trace correlation
    • Timeline analysis
    
    Predictive Analytics:
    • Time series forecasting
    • Capacity planning
    • Failure prediction
    • Trend extrapolation
    • Seasonality modeling
    
    Prescriptive Analytics:
    • Automated remediation
    • Optimization recommendations
    • What-if scenarios
    • Decision support
    • Action triggers
    
    Real-time Analytics:
    • Stream processing (Kafka, Flink)
    • Sliding windows
    • Continuous aggregation
    • Event correlation
    • Complex event processing
    
    Analytics Tools & Platforms:
    
    ELK Stack:
    • Elasticsearch: Search & analytics
    • Logstash: Data processing
    • Kibana: Visualization
    • Machine learning features
    
    Splunk:
    • SPL (Search Processing Language)
    • Real-time search
    • Alerting
    • ML Toolkit
    • IT Service Intelligence
    
    Datadog:
    • APM analytics
    • Log analytics
    • Anomaly detection
    • Watchdog (AI)
    • Correlations
    
    New Relic:
    • Query language (NRQL)
    • Applied Intelligence
    • Anomaly detection
    • Alert quality
    • Incident intelligence
    
    Custom Analytics (Python):
    • Pandas: Data manipulation
    • NumPy: Numerical computing
    • SciPy: Scientific computing
    • Scikit-learn: Machine learning
    • Statsmodels: Statistical models
    
    Analytics Best Practices:
    
    Data Quality:
    • Validate inputs
    • Handle missing data
    • Outlier treatment
    • Data normalization
    • Time synchronization
    
    Model Selection:
    • Understand data distribution
    • Choose appropriate algorithm
    • Validate assumptions
    • Test multiple approaches
    • Cross-validation
    
    Actionability:
    • Clear insights
    • Specific recommendations
    • Priority ranking
    • Expected impact
    • Implementation steps
    
    Continuous Improvement:
    • Track recommendation outcomes
    • Refine models
    • Update thresholds
    • Incorporate feedback
    • Automate learnings
    
    Key Insight:
    Analytics transforms raw observability data into
    actionable insights, enabling proactive problem
    resolution and continuous system improvement.
    
    🎯 OBSERVABILITY PATTERNS (101-110) COMPLETE! 🎯
    
    Category Summary:
    • 10 comprehensive observability patterns implemented
    • Coverage: Logging, Metrics, Tracing, Monitoring, Alerting,
               Debugging, Profiling, Telemetry, Dashboards, Analytics
    • Three Pillars: Metrics ✓ Logs ✓ Traces ✓
    • Full observability stack demonstrated
    
    This completes the Observability Patterns category,
    providing a complete toolkit for system monitoring,
    troubleshooting, and optimization!
    """
    
    return {
        "messages": [AIMessage(content=f"💡 Insight Generator:\n{response.content}\n{summary}")],
        "insights": insights,
        "recommendations": recommendations
    }


# Build the graph
def build_analytics_graph():
    """Build the analytics pattern graph"""
    workflow = StateGraph(AnalyticsState)
    
    workflow.add_node("pattern_analyzer", pattern_analyzer)
    workflow.add_node("insight_generator", insight_generator)
    
    workflow.add_edge(START, "pattern_analyzer")
    workflow.add_edge("pattern_analyzer", "insight_generator")
    workflow.add_edge("insight_generator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_analytics_graph()
    
    print("=== Analytics MCP Pattern ===\n")
    
    # Test Case: Comprehensive data analytics
    print("\n" + "="*70)
    print("TEST CASE: Observability Data Analytics")
    print("="*70)
    
    state = {
        "messages": [],
        "data_source": "logs",
        "analysis_type": "pattern",
        "input_data": {},
        "patterns_detected": [],
        "anomalies": [],
        "insights": [],
        "recommendations": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("OBSERVABILITY PATTERNS COMPLETE!")
    print(f"{'='*70}")
    print(f"Patterns: {len(result.get('patterns_detected', []))}")
    print(f"Anomalies: {len(result.get('anomalies', []))}")
    print(f"Insights: {len(result.get('insights', []))}")
    print(f"Recommendations: {len(result.get('recommendations', []))}")
