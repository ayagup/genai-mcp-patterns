"""
Pattern 213: Sidecar MCP Pattern

Sidecar pattern deploys helper container alongside main application:
- Extends functionality without modifying main app
- Shares same lifecycle and resources
- Handles cross-cutting concerns
- Enhances isolation and modularity

Common Sidecar Functions:
- Logging and monitoring
- Configuration management
- Service mesh proxy
- Security and authentication
- Data synchronization

Benefits:
- Separation of concerns
- Language-agnostic
- Independent deployment
- Resource sharing
- Simplified main application

Use Cases:
- Kubernetes pod sidecars
- Service mesh data plane
- Log collection and forwarding
- Configuration hot-reloading
- Health checking
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
from datetime import datetime


class SidecarState(TypedDict):
    """State for sidecar pattern operations"""
    sidecar_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class MainApplication:
    """Main application container"""
    app_name: str
    version: str
    
    request_count: int = 0
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, int] = field(default_factory=lambda: {'requests': 0, 'errors': 0})
    
    def process_request(self, request_id: str) -> Dict[str, Any]:
        """Process business logic"""
        self.request_count += 1
        self.metrics['requests'] += 1
        
        # Log request
        log_entry = f"[{datetime.now().isoformat()}] Processing request: {request_id}"
        self.logs.append(log_entry)
        
        # Simulate processing
        time.sleep(0.01)
        
        return {
            'request_id': request_id,
            'status': 'success',
            'processed_by': self.app_name
        }


@dataclass
class LoggingSidecar:
    """Sidecar for log collection and forwarding"""
    name: str = "logging-sidecar"
    
    collected_logs: List[str] = field(default_factory=list)
    forwarded_count: int = 0
    
    def collect_logs(self, app: MainApplication) -> int:
        """Collect logs from main application"""
        new_logs = app.logs[len(self.collected_logs):]
        self.collected_logs.extend(new_logs)
        return len(new_logs)
    
    def forward_logs(self, destination: str = "log-server") -> int:
        """Forward logs to external system"""
        # Simulate forwarding
        count = len(self.collected_logs) - self.forwarded_count
        self.forwarded_count = len(self.collected_logs)
        return count


@dataclass
class MonitoringSidecar:
    """Sidecar for metrics collection"""
    name: str = "monitoring-sidecar"
    
    metrics_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    
    def scrape_metrics(self, app: MainApplication) -> Dict[str, Any]:
        """Scrape metrics from main application"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'requests': app.metrics['requests'],
            'errors': app.metrics['errors'],
            'request_count': app.request_count
        }
        self.metrics_snapshots.append(snapshot)
        return snapshot
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export metrics to monitoring system"""
        return self.metrics_snapshots.copy()


@dataclass
class ConfigSidecar:
    """Sidecar for configuration management"""
    name: str = "config-sidecar"
    
    config: Dict[str, Any] = field(default_factory=dict)
    reload_count: int = 0
    
    def load_config(self, source: str = "config-server") -> Dict[str, Any]:
        """Load configuration from external source"""
        # Simulate loading config
        self.config = {
            'max_connections': 100,
            'timeout_ms': 5000,
            'feature_flags': {
                'new_feature': True,
                'beta_feature': False
            },
            'log_level': 'INFO'
        }
        self.reload_count += 1
        return self.config
    
    def hot_reload(self) -> bool:
        """Hot reload configuration without restarting main app"""
        self.load_config()
        return True


@dataclass
class ProxySidecar:
    """Sidecar for network proxy (service mesh)"""
    name: str = "proxy-sidecar"
    
    intercepted_requests: int = 0
    cached_responses: Dict[str, Any] = field(default_factory=dict)
    
    def intercept_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Intercept and process outgoing request"""
        self.intercepted_requests += 1
        
        # Check cache
        cache_key = request.get('url', '')
        if cache_key in self.cached_responses:
            return {**self.cached_responses[cache_key], 'cached': True}
        
        # Add headers
        request['headers'] = {
            **request.get('headers', {}),
            'X-Request-ID': f"req-{self.intercepted_requests}",
            'X-Proxy': self.name
        }
        
        return request
    
    def cache_response(self, url: str, response: Dict[str, Any]):
        """Cache response"""
        self.cached_responses[url] = response


class PodWithSidecars:
    """
    Kubernetes-like Pod with main app and sidecars
    """
    
    def __init__(self, pod_name: str):
        self.pod_name = pod_name
        self.main_app = MainApplication("webapp", "v1.0")
        
        # Initialize sidecars
        self.logging_sidecar = LoggingSidecar()
        self.monitoring_sidecar = MonitoringSidecar()
        self.config_sidecar = ConfigSidecar()
        self.proxy_sidecar = ProxySidecar()
        
        # Load initial config
        self.config_sidecar.load_config()
    
    def handle_request(self, request_id: str) -> Dict[str, Any]:
        """Handle request with sidecar support"""
        # 1. Proxy intercepts request
        proxied_request = self.proxy_sidecar.intercept_request({'request_id': request_id})
        
        # 2. Main app processes
        response = self.main_app.process_request(request_id)
        
        # 3. Logging sidecar collects logs
        self.logging_sidecar.collect_logs(self.main_app)
        
        # 4. Monitoring sidecar scrapes metrics
        self.monitoring_sidecar.scrape_metrics(self.main_app)
        
        return response
    
    def get_pod_status(self) -> Dict[str, Any]:
        """Get pod status including all sidecars"""
        return {
            'pod_name': self.pod_name,
            'main_app': {
                'name': self.main_app.app_name,
                'version': self.main_app.version,
                'request_count': self.main_app.request_count
            },
            'sidecars': {
                'logging': {
                    'collected_logs': len(self.logging_sidecar.collected_logs),
                    'forwarded': self.logging_sidecar.forwarded_count
                },
                'monitoring': {
                    'snapshots': len(self.monitoring_sidecar.metrics_snapshots)
                },
                'config': {
                    'reload_count': self.config_sidecar.reload_count
                },
                'proxy': {
                    'intercepted': self.proxy_sidecar.intercepted_requests,
                    'cached_items': len(self.proxy_sidecar.cached_responses)
                }
            }
        }


def setup_pod_agent(state: SidecarState):
    """Agent to set up pod with sidecars"""
    operations = []
    results = []
    
    pod = PodWithSidecars("webapp-pod-1")
    
    operations.append("Sidecar Pattern Setup:")
    operations.append(f"\nPod: {pod.pod_name}")
    operations.append(f"  Main Application: {pod.main_app.app_name} v{pod.main_app.version}")
    operations.append("\n  Sidecars:")
    operations.append(f"    - {pod.logging_sidecar.name}: Log collection & forwarding")
    operations.append(f"    - {pod.monitoring_sidecar.name}: Metrics scraping")
    operations.append(f"    - {pod.config_sidecar.name}: Configuration management")
    operations.append(f"    - {pod.proxy_sidecar.name}: Network proxy")
    
    results.append("✓ Pod with 4 sidecars initialized")
    
    # Store in state
    state['_pod'] = pod
    
    return {
        "sidecar_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Pod setup complete"]
    }


def request_handling_agent(state: SidecarState):
    """Agent to demonstrate request handling"""
    pod = state['_pod']
    operations = []
    results = []
    
    operations.append("\n📨 Request Handling Demo:")
    
    # Process requests
    operations.append("\nProcessing 5 requests:")
    for i in range(5):
        response = pod.handle_request(f"req-{i+1}")
        operations.append(f"  Request {i+1}: {response['status']}")
    
    results.append("✓ All requests processed with sidecar support")
    
    return {
        "sidecar_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Request handling complete"]
    }


def sidecar_functions_agent(state: SidecarState):
    """Agent to demonstrate sidecar functions"""
    pod = state['_pod']
    operations = []
    results = []
    
    operations.append("\n⚙️ Sidecar Functions Demo:")
    
    # Logging sidecar
    forwarded = pod.logging_sidecar.forward_logs()
    operations.append(f"\nLogging Sidecar:")
    operations.append(f"  Collected logs: {len(pod.logging_sidecar.collected_logs)}")
    operations.append(f"  Forwarded: {forwarded} logs")
    
    # Monitoring sidecar
    metrics = pod.monitoring_sidecar.export_metrics()
    operations.append(f"\nMonitoring Sidecar:")
    operations.append(f"  Metrics snapshots: {len(metrics)}")
    if metrics:
        latest = metrics[-1]
        operations.append(f"  Latest: {latest['requests']} requests, {latest['errors']} errors")
    
    # Config sidecar
    operations.append(f"\nConfig Sidecar:")
    operations.append(f"  Current config: {pod.config_sidecar.config.get('log_level', 'N/A')}")
    operations.append(f"  Reload count: {pod.config_sidecar.reload_count}")
    pod.config_sidecar.hot_reload()
    operations.append(f"  Hot reload: SUCCESS")
    
    # Proxy sidecar
    operations.append(f"\nProxy Sidecar:")
    operations.append(f"  Intercepted requests: {pod.proxy_sidecar.intercepted_requests}")
    
    results.append("✓ All sidecars functioning correctly")
    
    return {
        "sidecar_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Sidecar functions demonstrated"]
    }


def statistics_agent(state: SidecarState):
    """Agent to show statistics"""
    pod = state['_pod']
    operations = []
    results = []
    metrics = []
    
    status = pod.get_pod_status()
    
    operations.append("\n" + "="*60)
    operations.append("POD STATUS")
    operations.append("="*60)
    
    operations.append(f"\nPod: {status['pod_name']}")
    operations.append(f"\nMain Application:")
    operations.append(f"  Name: {status['main_app']['name']}")
    operations.append(f"  Version: {status['main_app']['version']}")
    operations.append(f"  Requests: {status['main_app']['request_count']}")
    
    operations.append(f"\nSidecars:")
    for name, stats in status['sidecars'].items():
        operations.append(f"  {name}:")
        for key, value in stats.items():
            operations.append(f"    {key}: {value}")
    
    metrics.append("\n📊 Sidecar Pattern Benefits:")
    metrics.append("  ✓ Separation of concerns")
    metrics.append("  ✓ Independent scaling")
    metrics.append("  ✓ Language agnostic")
    metrics.append("  ✓ Shared lifecycle")
    metrics.append("  ✓ Resource efficiency")
    
    results.append("✓ Sidecar pattern demonstrated successfully")
    
    return {
        "sidecar_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_sidecar_graph():
    """Create the sidecar workflow graph"""
    workflow = StateGraph(SidecarState)
    
    # Add nodes
    workflow.add_node("setup", setup_pod_agent)
    workflow.add_node("requests", request_handling_agent)
    workflow.add_node("functions", sidecar_functions_agent)
    workflow.add_node("statistics", statistics_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "requests")
    workflow.add_edge("requests", "functions")
    workflow.add_edge("functions", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 213: Sidecar MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_sidecar_graph()
    
    # Initialize state
    initial_state = {
        "sidecar_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 80)
    print("SIDECAR OPERATIONS")
    print("=" * 80)
    for op in final_state["sidecar_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("OPERATION RESULTS")
    print("=" * 80)
    for result in final_state["operation_results"]:
        print(result)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Sidecar Pattern: Helper container alongside main application

Common Sidecar Types:
1. Logging Sidecar: Collect and forward logs
2. Monitoring Sidecar: Scrape and export metrics
3. Config Sidecar: Manage configuration
4. Proxy Sidecar: Service mesh data plane
5. Security Sidecar: Authentication, encryption
6. Sync Sidecar: Data synchronization

Real-World Examples:
- Envoy proxy (Istio service mesh)
- Fluentd/Fluent Bit (log forwarding)
- Prometheus exporters
- Consul agent
- Ambassador

Benefits:
✓ Modular architecture
✓ Independent updates
✓ Language agnostic
✓ Simplified main app
✓ Shared resources (network, storage)
""")


if __name__ == "__main__":
    main()
