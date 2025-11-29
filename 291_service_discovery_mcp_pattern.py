"""
Pattern 291: Service Discovery MCP Pattern

This pattern demonstrates service discovery where agents and services
dynamically register and discover each other in a distributed system.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class ServiceDiscoveryPattern(TypedDict):
    """State for service discovery"""
    messages: Annotated[List[str], add]
    service_registry: Dict[str, Any]
    registered_services: List[Dict[str, Any]]
    discovered_services: List[Dict[str, Any]]
    health_status: Dict[str, Any]


class ServiceRegistry:
    """Central registry for service discovery"""
    
    def __init__(self):
        self.services = {}
        self.health_checks = {}
    
    def register(self, service_id: str, service_info: Dict[str, Any]):
        """Register a service"""
        self.services[service_id] = {
            **service_info,
            "registered_at": time.time(),
            "last_heartbeat": time.time(),
            "status": "active"
        }
    
    def deregister(self, service_id: str):
        """Deregister a service"""
        if service_id in self.services:
            self.services[service_id]["status"] = "inactive"
            self.services[service_id]["deregistered_at"] = time.time()
    
    def discover(self, criteria: Dict[str, Any] = None):
        """Discover services matching criteria"""
        if not criteria:
            return list(self.services.values())
        
        results = []
        for service_id, service_info in self.services.items():
            matches = True
            
            # Check service type
            if "service_type" in criteria:
                if service_info.get("service_type") != criteria["service_type"]:
                    matches = False
            
            # Check tags
            if "tags" in criteria and matches:
                required_tags = set(criteria["tags"])
                service_tags = set(service_info.get("tags", []))
                if not required_tags.issubset(service_tags):
                    matches = False
            
            # Check status
            if "status" in criteria and matches:
                if service_info.get("status") != criteria["status"]:
                    matches = False
            
            if matches:
                results.append(service_info)
        
        return results
    
    def heartbeat(self, service_id: str):
        """Update service heartbeat"""
        if service_id in self.services:
            self.services[service_id]["last_heartbeat"] = time.time()
            return True
        return False
    
    def check_health(self):
        """Check health of all services"""
        current_time = time.time()
        health_timeout = 30  # seconds
        
        health_status = {
            "healthy": 0,
            "unhealthy": 0,
            "total": len(self.services)
        }
        
        for service_id, service_info in self.services.items():
            if service_info["status"] == "active":
                time_since_heartbeat = current_time - service_info["last_heartbeat"]
                if time_since_heartbeat > health_timeout:
                    service_info["status"] = "unhealthy"
                    health_status["unhealthy"] += 1
                else:
                    health_status["healthy"] += 1
        
        return health_status


def initialize_registry_agent(state: ServiceDiscoveryPattern) -> ServiceDiscoveryPattern:
    """Initialize service registry"""
    print("\n📋 Initializing Service Registry...")
    
    registry = ServiceRegistry()
    
    print(f"  Registry: Ready")
    print(f"  Features:")
    print(f"    • Service registration")
    print(f"    • Service discovery")
    print(f"    • Health monitoring")
    print(f"    • Automatic cleanup")
    
    return {
        **state,
        "service_registry": {},
        "registered_services": [],
        "discovered_services": [],
        "health_status": {},
        "messages": ["✓ Service registry initialized"]
    }


def register_services_agent(state: ServiceDiscoveryPattern) -> ServiceDiscoveryPattern:
    """Register multiple services"""
    print("\n📝 Registering Services...")
    
    registry = ServiceRegistry()
    
    # Register various services
    services_to_register = [
        {
            "service_id": "api-gateway-1",
            "service_type": "gateway",
            "name": "API Gateway",
            "host": "10.0.1.10",
            "port": 8080,
            "protocol": "http",
            "tags": ["public", "production", "load-balanced"]
        },
        {
            "service_id": "auth-service-1",
            "service_type": "authentication",
            "name": "Auth Service",
            "host": "10.0.1.20",
            "port": 9000,
            "protocol": "grpc",
            "tags": ["internal", "production", "critical"]
        },
        {
            "service_id": "auth-service-2",
            "service_type": "authentication",
            "name": "Auth Service Replica",
            "host": "10.0.1.21",
            "port": 9000,
            "protocol": "grpc",
            "tags": ["internal", "production", "replica"]
        },
        {
            "service_id": "data-processor-1",
            "service_type": "processor",
            "name": "Data Processor",
            "host": "10.0.2.10",
            "port": 5000,
            "protocol": "http",
            "tags": ["internal", "production", "analytics"]
        },
        {
            "service_id": "cache-service-1",
            "service_type": "cache",
            "name": "Redis Cache",
            "host": "10.0.3.10",
            "port": 6379,
            "protocol": "redis",
            "tags": ["internal", "production", "cache"]
        }
    ]
    
    for service in services_to_register:
        service_id = service.pop("service_id")
        registry.register(service_id, service)
        print(f"  ✓ Registered: {service['name']}")
        print(f"    ID: {service_id}")
        print(f"    Type: {service['service_type']}")
        print(f"    Endpoint: {service['host']}:{service['port']}")
    
    print(f"\n  Total Services Registered: {len(registry.services)}")
    
    registered = [
        {"service_id": sid, **info}
        for sid, info in registry.services.items()
    ]
    
    return {
        **state,
        "service_registry": registry.services.copy(),
        "registered_services": registered,
        "messages": [f"✓ Registered {len(services_to_register)} services"]
    }


def discover_services_agent(state: ServiceDiscoveryPattern) -> ServiceDiscoveryPattern:
    """Discover services based on criteria"""
    print("\n🔍 Discovering Services...")
    
    registry = ServiceRegistry()
    registry.services = state["service_registry"].copy()
    
    # Perform different discovery queries
    discovery_queries = [
        {"name": "All Authentication Services", "criteria": {"service_type": "authentication"}},
        {"name": "Production Services", "criteria": {"tags": ["production"]}},
        {"name": "Internal Services", "criteria": {"tags": ["internal"]}},
        {"name": "Critical Services", "criteria": {"tags": ["critical"]}},
        {"name": "All Services", "criteria": None}
    ]
    
    all_discoveries = []
    
    for query in discovery_queries:
        results = registry.discover(query["criteria"])
        
        print(f"\n  Query: {query['name']}")
        print(f"  Results: {len(results)} service(s)")
        
        for service in results[:3]:  # Show first 3
            print(f"    • {service['name']} ({service['service_type']})")
            print(f"      Endpoint: {service['host']}:{service['port']}")
        
        all_discoveries.extend(results)
    
    return {
        **state,
        "discovered_services": all_discoveries,
        "messages": [f"✓ Executed {len(discovery_queries)} discovery queries"]
    }


def monitor_health_agent(state: ServiceDiscoveryPattern) -> ServiceDiscoveryPattern:
    """Monitor service health"""
    print("\n💚 Monitoring Service Health...")
    
    registry = ServiceRegistry()
    registry.services = state["service_registry"].copy()
    
    # Simulate heartbeats for some services
    active_services = ["api-gateway-1", "auth-service-1", "auth-service-2", "cache-service-1"]
    
    print(f"  Processing heartbeats...")
    for service_id in active_services:
        registry.heartbeat(service_id)
        print(f"    ✓ Heartbeat: {service_id}")
    
    # Check overall health
    health_status = registry.check_health()
    
    print(f"\n  Health Status:")
    print(f"    Total Services: {health_status['total']}")
    print(f"    Healthy: {health_status['healthy']}")
    print(f"    Unhealthy: {health_status['unhealthy']}")
    
    if health_status['total'] > 0:
        health_percentage = (health_status['healthy'] / health_status['total']) * 100
        print(f"    Health Rate: {health_percentage:.1f}%")
    
    return {
        **state,
        "health_status": health_status,
        "service_registry": registry.services.copy(),
        "messages": ["✓ Health monitoring complete"]
    }


def generate_service_discovery_report_agent(state: ServiceDiscoveryPattern) -> ServiceDiscoveryPattern:
    """Generate service discovery report"""
    print("\n" + "="*70)
    print("SERVICE DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n📋 Service Registry:")
    print(f"  Total Services: {len(state['service_registry'])}")
    
    # Group by service type
    by_type = {}
    for service_id, service_info in state['service_registry'].items():
        stype = service_info.get('service_type', 'unknown')
        by_type[stype] = by_type.get(stype, 0) + 1
    
    print(f"\n  Services by Type:")
    for stype, count in by_type.items():
        print(f"    {stype}: {count}")
    
    print(f"\n📝 Registered Services:")
    for service in state['registered_services'][:5]:
        print(f"\n  • {service['name']}:")
        print(f"      ID: {service['service_id']}")
        print(f"      Type: {service['service_type']}")
        print(f"      Endpoint: {service['host']}:{service['port']}")
        print(f"      Protocol: {service['protocol']}")
        print(f"      Tags: {', '.join(service.get('tags', []))}")
        print(f"      Status: {service['status']}")
    
    print(f"\n🔍 Discovery Statistics:")
    print(f"  Services Discovered: {len(state['discovered_services'])}")
    print(f"  Unique Services: {len(set(s.get('name') for s in state['discovered_services']))}")
    
    print(f"\n💚 Health Status:")
    health = state['health_status']
    if health:
        print(f"  Total: {health.get('total', 0)}")
        print(f"  Healthy: {health.get('healthy', 0)}")
        print(f"  Unhealthy: {health.get('unhealthy', 0)}")
        
        if health.get('total', 0) > 0:
            health_rate = (health['healthy'] / health['total']) * 100
            status_icon = "✅" if health_rate >= 80 else "⚠️" if health_rate >= 50 else "❌"
            print(f"  Health Rate: {health_rate:.1f}% {status_icon}")
    
    print(f"\n💡 Service Discovery Benefits:")
    print("  ✓ Dynamic service location")
    print("  ✓ Load balancing support")
    print("  ✓ Failover capabilities")
    print("  ✓ Zero-downtime updates")
    print("  ✓ Health monitoring")
    print("  ✓ Auto-scaling support")
    
    print(f"\n🔧 Discovery Methods:")
    print("  • Type-based discovery")
    print("  • Tag-based filtering")
    print("  • Status-based queries")
    print("  • Multi-criteria search")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Microservices architecture")
    print("  • Dynamic scaling")
    print("  • Load balancing")
    print("  • Service mesh")
    print("  • Container orchestration")
    print("  • Cloud-native apps")
    
    print(f"\n🎯 Registration Patterns:")
    print("  • Self-registration")
    print("  • Third-party registration")
    print("  • Client-side discovery")
    print("  • Server-side discovery")
    
    print("\n" + "="*70)
    print("✅ Service Discovery Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_service_discovery_graph():
    """Create service discovery workflow"""
    workflow = StateGraph(ServiceDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_registry_agent)
    workflow.add_node("register", register_services_agent)
    workflow.add_node("discover", discover_services_agent)
    workflow.add_node("health", monitor_health_agent)
    workflow.add_node("report", generate_service_discovery_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "register")
    workflow.add_edge("register", "discover")
    workflow.add_edge("discover", "health")
    workflow.add_edge("health", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 291: Service Discovery MCP Pattern")
    print("="*70)
    
    app = create_service_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "service_registry": {},
        "registered_services": [],
        "discovered_services": [],
        "health_status": {}
    })
    
    print("\n✅ Service Discovery Pattern Complete!")


if __name__ == "__main__":
    main()
