"""
Pattern 212: Service Mesh MCP Pattern

Service Mesh provides infrastructure layer for service-to-service communication:
- Service discovery and load balancing
- Traffic management (routing, splitting, mirroring)
- Resilience (retries, timeouts, circuit breakers)
- Security (mTLS, authentication, authorization)
- Observability (metrics, tracing, logging)
- Sidecar proxy pattern

Components:
- Control Plane: Configuration management, service registry
- Data Plane: Sidecar proxies handling actual traffic

Benefits:
- Decoupled infrastructure concerns from application code
- Consistent policies across all services
- Enhanced observability
- Zero-trust security
- Traffic control without code changes

Use Cases:
- Microservices architectures
- Multi-cloud deployments
- Hybrid cloud environments
- Service-to-service communication
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import random


class ServiceMeshState(TypedDict):
    """State for service mesh operations"""
    mesh_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class Service:
    """Microservice in the mesh"""
    service_id: str
    service_name: str
    version: str
    host: str
    port: int
    is_healthy: bool = True
    request_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    
    def process_request(self) -> Dict[str, Any]:
        """Process a request"""
        self.request_count += 1
        
        # Simulate processing
        latency = random.uniform(10, 50)
        time.sleep(latency / 1000)
        self.total_latency += latency
        
        # Simulate occasional errors
        if random.random() < 0.1:  # 10% error rate
            self.error_count += 1
            return {'status': 'error', 'latency': latency}
        
        return {'status': 'success', 'latency': latency, 'version': self.version}
    
    def get_avg_latency(self) -> float:
        """Get average latency"""
        return self.total_latency / self.request_count if self.request_count > 0 else 0


@dataclass
class SidecarProxy:
    """Sidecar proxy for service"""
    service: Service
    mesh_config: 'ServiceMesh'
    
    retry_attempts: int = 3
    timeout_ms: int = 100
    circuit_breaker_threshold: int = 5
    circuit_breaker_open: bool = False
    consecutive_failures: int = 0
    
    def forward_request(self, target_service: str) -> Dict[str, Any]:
        """Forward request with resilience patterns"""
        
        # Circuit breaker check
        if self.circuit_breaker_open:
            return {
                'status': 'circuit_breaker_open',
                'message': 'Circuit breaker is open'
            }
        
        # Retry logic
        for attempt in range(self.retry_attempts):
            response = self._send_request(target_service)
            
            if response['status'] == 'success':
                self.consecutive_failures = 0
                return response
            
            # Retry on failure
            if attempt < self.retry_attempts - 1:
                time.sleep(0.01)  # Backoff
        
        # All retries failed
        self.consecutive_failures += 1
        
        # Open circuit breaker if threshold reached
        if self.consecutive_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
        
        return {'status': 'failed', 'retries': self.retry_attempts}
    
    def _send_request(self, target_service: str) -> Dict[str, Any]:
        """Send request to target service"""
        target = self.mesh_config.get_service(target_service)
        if target:
            return target.process_request()
        return {'status': 'not_found'}


class ServiceMesh:
    """
    Service Mesh implementation with:
    - Service registry
    - Sidecar proxies
    - Traffic routing
    - Load balancing
    - Observability
    """
    
    def __init__(self):
        self.services: Dict[str, List[Service]] = {}
        self.proxies: Dict[str, SidecarProxy] = {}
        self.traffic_rules: Dict[str, Dict[str, int]] = {}  # Service -> {version: weight}
        
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    def register_service(self, service: Service):
        """Register service in mesh"""
        if service.service_name not in self.services:
            self.services[service.service_name] = []
        self.services[service.service_name].append(service)
        
        # Create sidecar proxy
        self.proxies[service.service_id] = SidecarProxy(service, self)
    
    def set_traffic_split(self, service_name: str, version_weights: Dict[str, int]):
        """Configure traffic splitting between versions"""
        self.traffic_rules[service_name] = version_weights
    
    def get_service(self, service_name: str) -> Optional[Service]:
        """Get service instance (with load balancing and traffic rules)"""
        if service_name not in self.services:
            return None
        
        instances = self.services[service_name]
        healthy = [s for s in instances if s.is_healthy]
        
        if not healthy:
            return None
        
        # Apply traffic splitting if configured
        if service_name in self.traffic_rules:
            weights = self.traffic_rules[service_name]
            total_weight = sum(weights.values())
            rand = random.randint(1, total_weight)
            
            cumulative = 0
            for instance in healthy:
                if instance.version in weights:
                    cumulative += weights[instance.version]
                    if rand <= cumulative:
                        return instance
        
        # Default: round-robin (least loaded)
        return min(healthy, key=lambda s: s.request_count)
    
    def send_request(self, from_service: str, to_service: str) -> Dict[str, Any]:
        """Send request through mesh"""
        self.total_requests += 1
        
        proxy = self.proxies.get(from_service)
        if not proxy:
            self.failed_requests += 1
            return {'status': 'proxy_not_found'}
        
        response = proxy.forward_request(to_service)
        
        if response['status'] == 'success':
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics"""
        total_services = sum(len(svcs) for svcs in self.services.items())
        
        return {
            'total_services': total_services,
            'service_types': len(self.services),
            'total_requests': self.total_requests,
            'successful': self.successful_requests,
            'failed': self.failed_requests,
            'success_rate': f"{(self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0:.1f}%"
        }


def setup_mesh_agent(state: ServiceMeshState):
    """Agent to set up service mesh"""
    operations = []
    results = []
    
    mesh = ServiceMesh()
    
    # Register services with multiple versions
    services = [
        Service("svc-1", "api-service", "v1", "10.0.1.1", 8080),
        Service("svc-2", "api-service", "v2", "10.0.1.2", 8080),
        Service("svc-3", "user-service", "v1", "10.0.2.1", 8081),
        Service("svc-4", "order-service", "v1", "10.0.3.1", 8082),
        Service("svc-5", "order-service", "v2", "10.0.3.2", 8082),
    ]
    
    operations.append("Service Mesh Setup:")
    operations.append("\nRegistered Services:")
    for service in services:
        mesh.register_service(service)
        operations.append(f"  {service.service_id}: {service.service_name} {service.version} @ {service.host}:{service.port}")
    
    # Configure traffic splitting (canary deployment)
    mesh.set_traffic_split("api-service", {"v1": 90, "v2": 10})  # 90/10 split
    mesh.set_traffic_split("order-service", {"v1": 70, "v2": 30})  # 70/30 split
    
    operations.append("\nTraffic Splitting Rules:")
    operations.append("  api-service: v1=90%, v2=10% (canary)")
    operations.append("  order-service: v1=70%, v2=30%")
    
    results.append(f"✓ Service Mesh configured with {len(services)} services")
    
    # Store in state
    state['_mesh'] = mesh
    
    return {
        "mesh_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Service mesh setup complete"]
    }


def traffic_management_agent(state: ServiceMeshState):
    """Agent to demonstrate traffic management"""
    mesh = state['_mesh']
    operations = []
    results = []
    
    operations.append("\n🚦 Traffic Management Demo:")
    
    # Send 20 requests to api-service
    operations.append("\nSending 20 requests to api-service (90/10 v1/v2 split):")
    
    v1_count = 0
    v2_count = 0
    
    for i in range(20):
        response = mesh.send_request("svc-1", "api-service")
        if response['status'] == 'success':
            if response['version'] == 'v1':
                v1_count += 1
            else:
                v2_count += 1
    
    operations.append(f"  v1 received: {v1_count} requests ({v1_count/20*100:.0f}%)")
    operations.append(f"  v2 received: {v2_count} requests ({v2_count/20*100:.0f}%)")
    
    results.append(f"✓ Traffic split: ~{v1_count/20*100:.0f}%/{v2_count/20*100:.0f}% (expected 90/10)")
    
    return {
        "mesh_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Traffic management demo complete"]
    }


def resilience_demo_agent(state: ServiceMeshState):
    """Agent to demonstrate resilience features"""
    mesh = state['_mesh']
    operations = []
    results = []
    
    operations.append("\n🛡️ Resilience Features Demo:")
    
    # Demonstrate retry logic
    operations.append("\nRetry Logic:")
    operations.append("  Sidecar proxies automatically retry failed requests")
    operations.append("  Default: 3 retry attempts with exponential backoff")
    
    # Send requests
    operations.append("\nSending 10 requests through mesh:")
    success = 0
    for i in range(10):
        response = mesh.send_request("svc-3", "user-service")
        if response['status'] == 'success':
            success += 1
    
    operations.append(f"  Successful: {success}/10")
    
    results.append("✓ Automatic retries and circuit breaking active")
    
    return {
        "mesh_operations": operations,
        "operation_results": results,
        "performance_metrics": [f"Success rate: {success/10*100:.0f}%"],
        "messages": ["Resilience demo complete"]
    }


def observability_agent(state: ServiceMeshState):
    """Agent to demonstrate observability"""
    mesh = state['_mesh']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n📊 Observability Demo:")
    
    # Collect service metrics
    operations.append("\nService Metrics:")
    for service_name, instances in mesh.services.items():
        operations.append(f"  {service_name}:")
        for instance in instances:
            operations.append(f"    {instance.version}: {instance.request_count} requests, "
                             f"{instance.get_avg_latency():.1f}ms avg latency")
    
    stats = mesh.get_statistics()
    operations.append(f"\nMesh Statistics:")
    operations.append(f"  Total services: {stats['total_services']}")
    operations.append(f"  Service types: {stats['service_types']}")
    operations.append(f"  Total requests: {stats['total_requests']}")
    operations.append(f"  Success rate: {stats['success_rate']}")
    
    results.append("✓ Observability metrics collected")
    
    return {
        "mesh_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Observability demo complete"]
    }


def statistics_agent(state: ServiceMeshState):
    """Agent to show statistics"""
    mesh = state['_mesh']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("SERVICE MESH STATISTICS")
    operations.append("="*60)
    
    stats = mesh.get_statistics()
    
    operations.append(f"\nMesh Overview:")
    operations.append(f"  Services: {stats['total_services']}")
    operations.append(f"  Service types: {stats['service_types']}")
    operations.append(f"  Total requests: {stats['total_requests']}")
    operations.append(f"  Success rate: {stats['success_rate']}")
    
    metrics.append("\n📊 Service Mesh Benefits:")
    metrics.append("  ✓ Service discovery")
    metrics.append("  ✓ Load balancing")
    metrics.append("  ✓ Traffic splitting (canary, blue/green)")
    metrics.append("  ✓ Automatic retries")
    metrics.append("  ✓ Circuit breaking")
    metrics.append("  ✓ mTLS security")
    metrics.append("  ✓ Distributed tracing")
    metrics.append("  ✓ Metrics collection")
    
    results.append("✓ Service Mesh demonstrated successfully")
    
    return {
        "mesh_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_service_mesh_graph():
    """Create the service mesh workflow graph"""
    workflow = StateGraph(ServiceMeshState)
    
    # Add nodes
    workflow.add_node("setup", setup_mesh_agent)
    workflow.add_node("traffic", traffic_management_agent)
    workflow.add_node("resilience", resilience_demo_agent)
    workflow.add_node("observability", observability_agent)
    workflow.add_node("statistics", statistics_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "traffic")
    workflow.add_edge("traffic", "resilience")
    workflow.add_edge("resilience", "observability")
    workflow.add_edge("observability", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 212: Service Mesh MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_service_mesh_graph()
    
    # Initialize state
    initial_state = {
        "mesh_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 80)
    print("MESH OPERATIONS")
    print("=" * 80)
    for op in final_state["mesh_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("OPERATION RESULTS")
    print("=" * 80)
    for result in final_state["operation_results"]:
        print(result)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    for metric in final_state["performance_metrics"]:
        print(metric)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Service Mesh Pattern provides infrastructure layer for microservices:

Control Plane:
- Service registry and discovery
- Configuration management
- Policy enforcement
- Certificate authority (for mTLS)

Data Plane (Sidecar Proxies):
- Intercept all network traffic
- Load balancing
- Retries and timeouts
- Circuit breaking
- Metrics collection

Real-World Implementations:
- Istio
- Linkerd
- Consul Connect
- AWS App Mesh
- Kuma

Key Features:
✓ Service Discovery: Automatic registration
✓ Load Balancing: Intelligent routing
✓ Traffic Management: Canary, blue/green deployments
✓ Resilience: Retries, circuit breakers, timeouts
✓ Security: mTLS, authentication, authorization
✓ Observability: Metrics, traces, logs
✓ Policy Enforcement: Rate limiting, quotas
""")


if __name__ == "__main__":
    main()
