"""
Dashboard MCP Pattern

This pattern implements visualization dashboards with real-time
data display, interactive widgets, and customizable layouts.

Key Features:
- Multi-widget dashboards
- Real-time data visualization
- Interactive charts and graphs
- Custom layout management
- Alert visualization
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class DashboardState(TypedDict):
    """State for dashboard pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    widgets: List[Dict]  # [{type, title, data, config}]
    layout: Dict  # {rows, columns, widget_positions}
    data_sources: List[str]
    refresh_interval_seconds: int
    theme: str  # "light", "dark"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Dashboard Builder
def dashboard_builder(state: DashboardState) -> DashboardState:
    """Creates and configures dashboard widgets"""
    widgets = state.get("widgets", [])
    
    system_message = SystemMessage(content="""You are a dashboard builder.
    Create comprehensive visualization dashboards for system monitoring.""")
    
    user_message = HumanMessage(content=f"""Build dashboard:

Existing Widgets: {len(widgets) if widgets else 'None'}

Create monitoring dashboard.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Define dashboard widgets if not provided
    if not widgets:
        widgets = [
            {
                "id": "widget-1",
                "type": "metric_card",
                "title": "Request Rate",
                "data": {
                    "current_value": 1234,
                    "unit": "req/s",
                    "trend": "+12.5%",
                    "trend_direction": "up"
                },
                "config": {"color": "green", "icon": "activity"}
            },
            {
                "id": "widget-2",
                "type": "line_chart",
                "title": "Response Time (P95)",
                "data": {
                    "labels": ["00:00", "00:15", "00:30", "00:45", "01:00"],
                    "datasets": [
                        {"name": "API Gateway", "values": [145, 152, 148, 156, 151]},
                        {"name": "User Service", "values": [98, 102, 95, 105, 99]}
                    ]
                },
                "config": {"y_axis_label": "ms", "show_legend": True}
            },
            {
                "id": "widget-3",
                "type": "gauge",
                "title": "CPU Usage",
                "data": {
                    "value": 67.5,
                    "max": 100,
                    "unit": "%",
                    "thresholds": {"warning": 70, "critical": 85}
                },
                "config": {"color_ranges": [
                    {"from": 0, "to": 70, "color": "green"},
                    {"from": 70, "to": 85, "color": "yellow"},
                    {"from": 85, "to": 100, "color": "red"}
                ]}
            },
            {
                "id": "widget-4",
                "type": "bar_chart",
                "title": "Requests by Status Code",
                "data": {
                    "labels": ["200", "201", "400", "404", "500"],
                    "values": [14523, 234, 123, 45, 12]
                },
                "config": {"horizontal": False, "show_values": True}
            },
            {
                "id": "widget-5",
                "type": "table",
                "title": "Recent Alerts",
                "data": {
                    "columns": ["Time", "Severity", "Source", "Message"],
                    "rows": [
                        ["12:34:56", "Critical", "API Gateway", "High error rate"],
                        ["12:33:12", "Warning", "Database", "Connection pool high"],
                        ["12:31:45", "Info", "Cache", "Hit rate low"]
                    ]
                },
                "config": {"page_size": 10, "sortable": True}
            },
            {
                "id": "widget-6",
                "type": "heatmap",
                "title": "Service Health Matrix",
                "data": {
                    "rows": ["API Gateway", "User Service", "Database"],
                    "columns": ["Availability", "Latency", "Error Rate"],
                    "values": [
                        [99.9, 98.5, 99.2],
                        [92.3, 85.4, 96.7],
                        [95.6, 88.9, 91.2]
                    ]
                },
                "config": {"color_scale": "green_to_red"}
            }
        ]
    
    # Define layout
    layout = {
        "rows": 3,
        "columns": 3,
        "widget_positions": {
            "widget-1": {"row": 0, "col": 0, "width": 1, "height": 1},
            "widget-2": {"row": 0, "col": 1, "width": 2, "height": 1},
            "widget-3": {"row": 1, "col": 0, "width": 1, "height": 1},
            "widget-4": {"row": 1, "col": 1, "width": 1, "height": 1},
            "widget-5": {"row": 1, "col": 2, "width": 1, "height": 1},
            "widget-6": {"row": 2, "col": 0, "width": 3, "height": 1}
        }
    }
    
    data_sources = ["prometheus", "elasticsearch", "jaeger", "postgresql"]
    refresh_interval = 15
    theme = "dark"
    
    report = f"""
    📊 Dashboard Builder:
    
    Dashboard Overview:
    • Widgets: {len(widgets)}
    • Layout: {layout['rows']}x{layout['columns']} grid
    • Data Sources: {len(data_sources)}
    • Refresh: Every {refresh_interval}s
    • Theme: {theme.capitalize()}
    
    Dashboard Concepts:
    
    Widget Types:
    
    Metric Cards:
    • Single value display
    • Trend indicator
    • Comparison to baseline
    • Quick status check
    • Examples: Request rate, error count
    
    Line Charts:
    • Time-series data
    • Multiple datasets
    • Trend visualization
    • Historical analysis
    • Examples: Response time, throughput
    
    Bar Charts:
    • Categorical comparison
    • Distribution view
    • Top N analysis
    • Horizontal/vertical
    • Examples: Status codes, endpoints
    
    Pie/Donut Charts:
    • Proportional data
    • Part-to-whole
    • Category breakdown
    • Percentage view
    • Examples: Traffic by region
    
    Gauges:
    • Current value vs threshold
    • Visual status indicator
    • Color-coded ranges
    • Capacity monitoring
    • Examples: CPU usage, disk space
    
    Tables:
    • Detailed data view
    • Sortable columns
    • Pagination
    • Action buttons
    • Examples: Recent events, alerts
    
    Heatmaps:
    • Matrix visualization
    • Multi-dimensional data
    • Pattern detection
    • Correlation view
    • Examples: Service health, hourly traffic
    
    Dashboard Frameworks:
    
    Grafana:
    • Open source
    • Multi-datasource
    • Rich plugin ecosystem
    • Alerting integration
    • Template variables
    
    ```json
    {{
      "dashboard": {{
        "title": "System Metrics",
        "panels": [
          {{
            "id": 1,
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {{
                "expr": "rate(http_requests_total[5m])",
                "legendFormat": "{{{{service}}}}"
              }}
            ],
            "gridPos": {{"x": 0, "y": 0, "w": 12, "h": 8}}
          }}
        ],
        "refresh": "30s",
        "time": {{"from": "now-1h", "to": "now"}}
      }}
    }}
    ```
    
    Kibana:
    • Elasticsearch UI
    • Log visualization
    • Lens (drag-drop)
    • Canvas for reports
    • Alerting
    
    Datadog:
    • SaaS platform
    • Real-time streaming
    • APM integration
    • AI-powered insights
    • Mobile dashboards
    
    Custom (React):
    ```jsx
    import {{ LineChart, BarChart, MetricCard }} from 'recharts';
    
    function Dashboard() {{
      const [metrics, setMetrics] = useState({{}});
      
      useEffect(() => {{
        // Fetch data every 15s
        const interval = setInterval(() => {{
          fetch('/api/metrics')
            .then(res => res.json())
            .then(setMetrics);
        }}, 15000);
        
        return () => clearInterval(interval);
      }}, []);
      
      return (
        <div className="dashboard-grid">
          <MetricCard
            title="Request Rate"
            value={{metrics.requestRate}}
            unit="req/s"
            trend={{metrics.trend}}
          />
          
          <LineChart
            data={{metrics.timeSeries}}
            width={{600}}
            height={{300}}
          >
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Line dataKey="value" stroke="#8884d8" />
          </LineChart>
        </div>
      );
    }}
    ```
    
    Dashboard Design Principles:
    
    Information Hierarchy:
    • Most important metrics first
    • Top-left prominence
    • Logical grouping
    • Visual weight
    
    Glanceability:
    • Quick understanding
    • Clear labels
    • Appropriate scale
    • Color coding
    
    Actionable Insights:
    • Link to details
    • Drill-down capability
    • Filter/scope controls
    • Alert integration
    
    Responsive Design:
    • Mobile-friendly
    • Auto-scaling
    • Adaptive layout
    • Touch-friendly
    
    Layout Patterns:
    
    Overview + Detail:
    • High-level summary
    • Drill-down available
    • Progressive disclosure
    • Context preservation
    
    Multi-page:
    • Separate dashboards
    • Service-specific views
    • Team dashboards
    • Executive summaries
    
    Real-time Streaming:
    • Live updates
    • WebSocket connection
    • Auto-refresh
    • Animation for changes
    
    Data Refresh Strategies:
    
    Polling:
    • Periodic HTTP requests
    • Simple implementation
    • Server overhead
    • Delayed updates
    
    WebSocket:
    • Bi-directional
    • Real-time push
    • Efficient
    • Connection management
    
    Server-Sent Events:
    • One-way streaming
    • Auto-reconnect
    • HTTP-based
    • Simpler than WebSocket
    
    Visualization Best Practices:
    
    Choose Right Chart:
    • Time-series → Line chart
    • Comparison → Bar chart
    • Part-whole → Pie chart
    • Distribution → Histogram
    • Correlation → Scatter plot
    
    Color Usage:
    • Consistent meaning
    • Accessibility (colorblind)
    • Limited palette
    • Semantic colors (red=bad)
    
    Performance:
    • Lazy loading
    • Data aggregation
    • Client-side caching
    • Efficient queries
    • Virtualization for tables
    
    Interactivity:
    • Tooltips for details
    • Click to filter
    • Time range selector
    • Zoom capability
    • Export to CSV/PDF
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Dashboard Builder:\n{response.content}\n{report}")],
        "widgets": widgets,
        "layout": layout,
        "data_sources": data_sources,
        "refresh_interval_seconds": refresh_interval,
        "theme": theme
    }


# Data Visualizer
def data_visualizer(state: DashboardState) -> DashboardState:
    """Renders and updates dashboard visualizations"""
    widgets = state.get("widgets", [])
    layout = state.get("layout", {})
    
    system_message = SystemMessage(content="""You are a data visualizer.
    Render dashboard widgets with real-time data updates.""")
    
    user_message = HumanMessage(content=f"""Render dashboard:

Widgets: {len(widgets)}
Layout: {layout.get('rows', 0)}x{layout.get('columns', 0)}

Visualize data.""")
    
    response = llm.invoke([system_message, user_message])
    
    summary = f"""
    📊 DASHBOARD COMPLETE
    
    Dashboard Summary:
    • Total Widgets: {len(widgets)}
    • Layout: {layout.get('rows', 0)} rows × {layout.get('columns', 0)} columns
    • Data Sources: {len(state.get('data_sources', []))}
    • Refresh Rate: {state.get('refresh_interval_seconds', 0)}s
    • Theme: {state.get('theme', 'default').capitalize()}
    
    Widget Breakdown:
    {chr(10).join(f"  • {w['type'].replace('_', ' ').title()}: {w['title']}" for w in widgets)}
    
    Dashboard Pattern Process:
    1. Dashboard Builder → Create widget configuration
    2. Data Visualizer → Render and update displays
    
    Advanced Dashboard Features:
    
    Template Variables:
    • Dynamic filters
    • Environment selector
    • Time range picker
    • Service dropdown
    • Query parameterization
    
    Annotations:
    • Deployment markers
    • Incident timeline
    • Release versions
    • Maintenance windows
    • Business events
    
    Alerting Integration:
    • Visual indicators
    • Alert panels
    • Threshold lines
    • Status overlays
    • Alert history
    
    Drill-Down:
    • Click to expand
    • Link to traces
    • Filter propagation
    • Context preservation
    • Breadcrumb navigation
    
    Dashboard as Code:
    
    Terraform (Datadog):
    ```hcl
    resource "datadog_dashboard" "system_metrics" {{
      title       = "System Metrics"
      layout_type = "ordered"
      
      widget {{
        timeseries_definition {{
          title = "Request Rate"
          request {{
            q = "sum:http.requests{{*}}.as_rate()"
          }}
        }}
      }}
      
      widget {{
        query_value_definition {{
          title = "Error Rate"
          request {{
            q = "sum:http.errors{{*}}.as_rate()"
          }}
          precision = 2
        }}
      }}
    }}
    ```
    
    Grafana Provisioning:
    ```yaml
    apiVersion: 1
    
    providers:
      - name: 'default'
        orgId: 1
        folder: ''
        type: file
        options:
          path: /var/lib/grafana/dashboards
    ```
    
    Dashboard Testing:
    
    Visual Regression:
    • Screenshot comparison
    • Automated testing
    • CI/CD integration
    • Percy, BackstopJS
    
    Data Validation:
    • Query correctness
    • Value ranges
    • Update frequency
    • Missing data handling
    
    Performance Testing:
    • Load time
    • Render speed
    • Memory usage
    • Network efficiency
    
    Accessibility Testing:
    • Screen reader support
    • Keyboard navigation
    • Color contrast
    • WCAG compliance
    
    Dashboard Metrics:
    
    Usage Analytics:
    • View count
    • Time spent
    • User interactions
    • Popular widgets
    
    Performance Metrics:
    • Load time
    • Query duration
    • Refresh overhead
    • Client-side performance
    
    Data Quality:
    • Staleness
    • Gaps/missing data
    • Error rates
    • Data latency
    
    Key Insight:
    Effective dashboards provide at-a-glance
    understanding of system health with drill-down
    capabilities for detailed investigation.
    """
    
    return {
        "messages": [AIMessage(content=f"📈 Data Visualizer:\n{response.content}\n{summary}")]
    }


# Build the graph
def build_dashboard_graph():
    """Build the dashboard pattern graph"""
    workflow = StateGraph(DashboardState)
    
    workflow.add_node("dashboard_builder", dashboard_builder)
    workflow.add_node("data_visualizer", data_visualizer)
    
    workflow.add_edge(START, "dashboard_builder")
    workflow.add_edge("dashboard_builder", "data_visualizer")
    workflow.add_edge("data_visualizer", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_dashboard_graph()
    
    print("=== Dashboard MCP Pattern ===\n")
    
    # Test Case: Multi-widget monitoring dashboard
    print("\n" + "="*70)
    print("TEST CASE: System Monitoring Dashboard")
    print("="*70)
    
    state = {
        "messages": [],
        "widgets": [],
        "layout": {},
        "data_sources": [],
        "refresh_interval_seconds": 15,
        "theme": "dark"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nDashboard Configuration:")
    print(f"Widgets: {len(result.get('widgets', []))}")
    print(f"Layout: {result.get('layout', {}).get('rows', 0)}x{result.get('layout', {}).get('columns', 0)}")
    print(f"Refresh: {result.get('refresh_interval_seconds', 0)}s")
