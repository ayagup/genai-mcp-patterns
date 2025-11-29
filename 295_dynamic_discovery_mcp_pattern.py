"""
Pattern 295: Dynamic Discovery MCP Pattern

This pattern demonstrates dynamic discovery where services and resources
are discovered at runtime based on current conditions and requirements.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import random


class DynamicDiscoveryPattern(TypedDict):
    """State for dynamic discovery"""
    messages: Annotated[List[str], add]
    runtime_registry: Dict[str, Any]
    discovery_events: List[Dict[str, Any]]
    dynamic_routes: Dict[str, str]
    runtime_metrics: Dict[str, Any]


class DynamicService:
    """Service with dynamic properties"""
    
    def __init__(self, service_id: str, service_type: str):
        self.service_id = service_id
        self.service_type = service_type
        self.endpoint = None
        self.status = "running"
        self.last_seen = time.time()
        self.response_time_ms = 0
        self.success_rate = 1.0
        self.load = 0.0
    
    def update_metrics(self):
        """Update dynamic metrics"""
        self.last_seen = time.time()
        self.response_time_ms = random.randint(10, 200)
        self.success_rate = random.uniform(0.85, 1.0)
        self.load = random.uniform(0.1, 0.9)
    
    def is_healthy(self):
        """Check if service is healthy"""
        time_since_seen = time.time() - self.last_seen
        return (self.status == "running" and 
                time_since_seen < 30 and 
                self.success_rate > 0.8)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "service_id": self.service_id,
            "service_type": self.service_type,
            "endpoint": self.endpoint,
            "status": self.status,
            "last_seen": self.last_seen,
            "response_time_ms": self.response_time_ms,
            "success_rate": self.success_rate,
            "load": self.load,
            "healthy": self.is_healthy()
        }


class DynamicRegistry:
    """Registry for dynamic discovery"""
    
    def __init__(self):
        self.services = {}
        self.discovery_listeners = []
    
    def register(self, service: DynamicService):
        """Register or update service"""
        is_new = service.service_id not in self.services
        self.services[service.service_id] = service
        
        event = {
            "type": "registration" if is_new else "update",
            "service_id": service.service_id,
            "timestamp": time.time()
        }
        return event
    
    def deregister(self, service_id: str):
        """Deregister service"""
        if service_id in self.services:
            del self.services[service_id]
            return {
                "type": "deregistration",
                "service_id": service_id,
                "timestamp": time.time()
            }
        return None
    
    def discover_best(self, service_type: str):
        """Discover best service dynamically"""
        candidates = [s for s in self.services.values() 
                     if s.service_type == service_type and s.is_healthy()]
        
        if not candidates:
            return None
        
        # Score based on response time, success rate, and load
        def score_service(service):
            response_score = 1.0 - (service.response_time_ms / 1000.0)
            load_score = 1.0 - service.load
            return (service.success_rate * 0.5 + 
                   response_score * 0.3 + 
                   load_score * 0.2)
        
        best_service = max(candidates, key=score_service)
        return best_service.to_dict(), score_service(best_service)
    
    def discover_all_healthy(self):
        """Discover all healthy services"""
        return [s.to_dict() for s in self.services.values() if s.is_healthy()]
    
    def cleanup_stale(self, timeout: int = 30):
        """Remove stale services"""
        current_time = time.time()
        stale_ids = []
        
        for service_id, service in self.services.items():
            if current_time - service.last_seen > timeout:
                stale_ids.append(service_id)
        
        for service_id in stale_ids:
            del self.services[service_id]
        
        return stale_ids


def initialize_dynamic_registry_agent(state: DynamicDiscoveryPattern) -> DynamicDiscoveryPattern:
    """Initialize dynamic registry"""
    print("\n🔄 Initializing Dynamic Discovery Registry...")
    
    registry = DynamicRegistry()
    
    print(f"  Registry: Ready")
    print(f"  Mode: Dynamic (Runtime)")
    print(f"  Features:")
    print(f"    • Real-time registration")
    print(f"    • Health-based selection")
    print(f"    • Performance monitoring")
    print(f"    • Auto cleanup")
    
    return {
        **state,
        "runtime_registry": {},
        "discovery_events": [],
        "dynamic_routes": {},
        "runtime_metrics": {},
        "messages": ["✓ Dynamic registry initialized"]
    }


def simulate_dynamic_registration_agent(state: DynamicDiscoveryPattern) -> DynamicDiscoveryPattern:
    """Simulate dynamic service registration"""
    print("\n📝 Simulating Dynamic Service Registration...")
    
    registry = DynamicRegistry()
    events = []
    
    # Simulate services coming online dynamically
    service_configs = [
        ("api_v1", "api", "http://10.0.1.10:8080"),
        ("api_v2", "api", "http://10.0.1.11:8080"),
        ("db_primary", "database", "postgres://10.0.2.10:5432"),
        ("db_replica", "database", "postgres://10.0.2.11:5432"),
        ("cache_1", "cache", "redis://10.0.3.10:6379")
    ]
    
    for service_id, service_type, endpoint in service_configs:
        service = DynamicService(service_id, service_type)
        service.endpoint = endpoint
        service.update_metrics()
        
        event = registry.register(service)
        events.append(event)
        
        print(f"  ✓ Registered: {service_id}")
        print(f"    Type: {service_type}")
        print(f"    Endpoint: {endpoint}")
        print(f"    Response Time: {service.response_time_ms}ms")
        print(f"    Success Rate: {service.success_rate:.2%}")
        
        time.sleep(0.05)  # Simulate time passing
    
    # Simulate metric updates
    print(f"\n  Simulating metric updates...")
    for service_id in list(registry.services.keys())[:3]:
        registry.services[service_id].update_metrics()
        event = {
            "type": "metrics_update",
            "service_id": service_id,
            "timestamp": time.time()
        }
        events.append(event)
        print(f"    Updated: {service_id}")
    
    print(f"\n  Total Services: {len(registry.services)}")
    print(f"  Total Events: {len(events)}")
    
    registry_state = {sid: svc.to_dict() for sid, svc in registry.services.items()}
    
    return {
        **state,
        "runtime_registry": registry_state,
        "discovery_events": events,
        "messages": [f"✓ Registered {len(service_configs)} services dynamically"]
    }


def perform_dynamic_discovery_agent(state: DynamicDiscoveryPattern) -> DynamicDiscoveryPattern:
    """Perform dynamic discovery"""
    print("\n🔍 Performing Dynamic Discovery...")
    
    registry = DynamicRegistry()
    
    # Recreate registry
    for service_id, service_data in state["runtime_registry"].items():
        service = DynamicService(service_data["service_id"], service_data["service_type"])
        service.endpoint = service_data["endpoint"]
        service.status = service_data["status"]
        service.last_seen = service_data["last_seen"]
        service.response_time_ms = service_data["response_time_ms"]
        service.success_rate = service_data["success_rate"]
        service.load = service_data["load"]
        registry.register(service)
    
    # Discover best services for different types
    service_types = ["api", "database", "cache"]
    dynamic_routes = {}
    
    for stype in service_types:
        result = registry.discover_best(stype)
        
        if result:
            best_service, score = result
            dynamic_routes[stype] = best_service["service_id"]
            
            print(f"\n  Best {stype} service:")
            print(f"    Service: {best_service['service_id']}")
            print(f"    Endpoint: {best_service['endpoint']}")
            print(f"    Score: {score:.3f}")
            print(f"    Response Time: {best_service['response_time_ms']}ms")
            print(f"    Success Rate: {best_service['success_rate']:.2%}")
            print(f"    Load: {best_service['load']:.2%}")
    
    # Discover all healthy services
    healthy_services = registry.discover_all_healthy()
    print(f"\n  Total Healthy Services: {len(healthy_services)}")
    
    return {
        **state,
        "dynamic_routes": dynamic_routes,
        "messages": [f"✓ Discovered routes for {len(service_types)} service types"]
    }


def monitor_runtime_metrics_agent(state: DynamicDiscoveryPattern) -> DynamicDiscoveryPattern:
    """Monitor runtime metrics"""
    print("\n📊 Monitoring Runtime Metrics...")
    
    services = state["runtime_registry"].values()
    
    # Calculate runtime metrics
    metrics = {
        "total_services": len(services),
        "healthy_services": sum(1 for s in services if s["healthy"]),
        "avg_response_time": sum(s["response_time_ms"] for s in services) / max(len(services), 1),
        "avg_success_rate": sum(s["success_rate"] for s in services) / max(len(services), 1),
        "avg_load": sum(s["load"] for s in services) / max(len(services), 1),
        "event_count": len(state["discovery_events"])
    }
    
    # Event distribution
    event_types = {}
    for event in state["discovery_events"]:
        etype = event["type"]
        event_types[etype] = event_types.get(etype, 0) + 1
    
    metrics["event_distribution"] = event_types
    
    print(f"  Total Services: {metrics['total_services']}")
    print(f"  Healthy Services: {metrics['healthy_services']}")
    print(f"  Health Rate: {(metrics['healthy_services']/max(metrics['total_services'],1))*100:.1f}%")
    print(f"  Avg Response Time: {metrics['avg_response_time']:.1f}ms")
    print(f"  Avg Success Rate: {metrics['avg_success_rate']:.2%}")
    print(f"  Avg Load: {metrics['avg_load']:.2%}")
    
    print(f"\n  Event Distribution:")
    for etype, count in event_types.items():
        print(f"    {etype}: {count}")
    
    return {
        **state,
        "runtime_metrics": metrics,
        "messages": ["✓ Runtime metrics collected"]
    }


def generate_dynamic_discovery_report_agent(state: DynamicDiscoveryPattern) -> DynamicDiscoveryPattern:
    """Generate dynamic discovery report"""
    print("\n" + "="*70)
    print("DYNAMIC DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n🔄 Dynamic Registry:")
    print(f"  Total Services: {len(state['runtime_registry'])}")
    
    print(f"\n🔧 Registered Services:")
    for service_data in list(state['runtime_registry'].values()):
        status_icon = "✅" if service_data["healthy"] else "⚠️"
        print(f"\n  {status_icon} {service_data['service_id']}:")
        print(f"      Type: {service_data['service_type']}")
        print(f"      Endpoint: {service_data['endpoint']}")
        print(f"      Response Time: {service_data['response_time_ms']}ms")
        print(f"      Success Rate: {service_data['success_rate']:.2%}")
        print(f"      Load: {service_data['load']:.2%}")
        print(f"      Healthy: {service_data['healthy']}")
    
    print(f"\n🎯 Dynamic Routes:")
    for stype, service_id in state['dynamic_routes'].items():
        endpoint = state['runtime_registry'][service_id]['endpoint']
        print(f"  {stype} → {service_id}")
        print(f"    Endpoint: {endpoint}")
    
    print(f"\n📊 Runtime Metrics:")
    metrics = state['runtime_metrics']
    if metrics:
        print(f"  Health Rate: {(metrics['healthy_services']/max(metrics['total_services'],1))*100:.1f}%")
        print(f"  Avg Response Time: {metrics['avg_response_time']:.1f}ms")
        print(f"  Avg Success Rate: {metrics['avg_success_rate']:.2%}")
        print(f"  Avg Load: {metrics['avg_load']:.2%}")
        print(f"  Total Events: {metrics['event_count']}")
    
    print(f"\n📝 Discovery Events:")
    print(f"  Total Events: {len(state['discovery_events'])}")
    for event in state['discovery_events'][:5]:
        print(f"    • {event['type']}: {event['service_id']}")
    
    print(f"\n💡 Dynamic Discovery Benefits:")
    print("  ✓ Real-time adaptation")
    print("  ✓ Performance-based routing")
    print("  ✓ Auto-failover")
    print("  ✓ Health-aware selection")
    print("  ✓ Zero-downtime updates")
    print("  ✓ Load-aware distribution")
    
    print(f"\n🔧 Dynamic Mechanisms:")
    print("  • Runtime registration")
    print("  • Health monitoring")
    print("  • Performance scoring")
    print("  • Automatic cleanup")
    print("  • Event-driven updates")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Microservices")
    print("  • Auto-scaling systems")
    print("  • Cloud-native apps")
    print("  • Service mesh")
    print("  • Container orchestration")
    print("  • Dynamic load balancing")
    
    print("\n" + "="*70)
    print("✅ Dynamic Discovery Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_dynamic_discovery_graph():
    """Create dynamic discovery workflow"""
    workflow = StateGraph(DynamicDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_dynamic_registry_agent)
    workflow.add_node("register", simulate_dynamic_registration_agent)
    workflow.add_node("discover", perform_dynamic_discovery_agent)
    workflow.add_node("monitor", monitor_runtime_metrics_agent)
    workflow.add_node("report", generate_dynamic_discovery_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "register")
    workflow.add_edge("register", "discover")
    workflow.add_edge("discover", "monitor")
    workflow.add_edge("monitor", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 295: Dynamic Discovery MCP Pattern")
    print("="*70)
    
    app = create_dynamic_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "runtime_registry": {},
        "discovery_events": [],
        "dynamic_routes": {},
        "runtime_metrics": {}
    })
    
    print("\n✅ Dynamic Discovery Pattern Complete!")


if __name__ == "__main__":
    main()
