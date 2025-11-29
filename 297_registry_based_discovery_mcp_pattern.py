"""
Pattern 297: Registry-Based Discovery MCP Pattern

This pattern demonstrates discovery using a central registry where services
register themselves and clients query the registry to find services.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class RegistryBasedDiscoveryPattern(TypedDict):
    """State for registry-based discovery"""
    messages: Annotated[List[str], add]
    central_registry: Dict[str, Any]
    registration_log: List[Dict[str, Any]]
    query_log: List[Dict[str, Any]]
    registry_statistics: Dict[str, Any]


class ServiceRegistration:
    """Service registration entry"""
    
    def __init__(self, service_id: str, name: str, endpoint: str):
        self.service_id = service_id
        self.name = name
        self.endpoint = endpoint
        self.registered_at = time.time()
        self.last_heartbeat = time.time()
        self.metadata = {}
        self.ttl = 60  # seconds
    
    def is_alive(self):
        """Check if registration is still alive"""
        return (time.time() - self.last_heartbeat) < self.ttl
    
    def heartbeat(self):
        """Update heartbeat"""
        self.last_heartbeat = time.time()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "service_id": self.service_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "ttl": self.ttl,
            "alive": self.is_alive(),
            "metadata": self.metadata
        }


class CentralRegistry:
    """Central service registry"""
    
    def __init__(self):
        self.registrations = {}
        self.name_index = {}
        self.registration_count = 0
        self.query_count = 0
    
    def register(self, registration: ServiceRegistration):
        """Register a service"""
        self.registrations[registration.service_id] = registration
        self.registration_count += 1
        
        # Index by name
        if registration.name not in self.name_index:
            self.name_index[registration.name] = []
        if registration.service_id not in self.name_index[registration.name]:
            self.name_index[registration.name].append(registration.service_id)
        
        return {
            "service_id": registration.service_id,
            "status": "registered",
            "timestamp": time.time()
        }
    
    def deregister(self, service_id: str):
        """Deregister a service"""
        if service_id in self.registrations:
            registration = self.registrations[service_id]
            
            # Remove from name index
            if registration.name in self.name_index:
                self.name_index[registration.name].remove(service_id)
                if not self.name_index[registration.name]:
                    del self.name_index[registration.name]
            
            del self.registrations[service_id]
            
            return {
                "service_id": service_id,
                "status": "deregistered",
                "timestamp": time.time()
            }
        return None
    
    def lookup_by_id(self, service_id: str):
        """Lookup service by ID"""
        self.query_count += 1
        if service_id in self.registrations:
            reg = self.registrations[service_id]
            if reg.is_alive():
                return reg.to_dict()
        return None
    
    def lookup_by_name(self, name: str):
        """Lookup services by name"""
        self.query_count += 1
        service_ids = self.name_index.get(name, [])
        results = []
        
        for service_id in service_ids:
            if service_id in self.registrations:
                reg = self.registrations[service_id]
                if reg.is_alive():
                    results.append(reg.to_dict())
        
        return results
    
    def get_all_services(self):
        """Get all registered services"""
        return [reg.to_dict() for reg in self.registrations.values() if reg.is_alive()]
    
    def cleanup_expired(self):
        """Remove expired registrations"""
        expired = []
        for service_id, reg in list(self.registrations.items()):
            if not reg.is_alive():
                expired.append(service_id)
                self.deregister(service_id)
        return expired


def initialize_central_registry_agent(state: RegistryBasedDiscoveryPattern) -> RegistryBasedDiscoveryPattern:
    """Initialize central registry"""
    print("\n🏛️ Initializing Central Registry...")
    
    registry = CentralRegistry()
    
    print(f"  Registry: Ready")
    print(f"  Type: Centralized")
    print(f"  Features:")
    print(f"    • Service registration")
    print(f"    • Name-based lookup")
    print(f"    • TTL management")
    print(f"    • Auto-cleanup")
    
    return {
        **state,
        "central_registry": {},
        "registration_log": [],
        "query_log": [],
        "registry_statistics": {},
        "messages": ["✓ Central registry initialized"]
    }


def register_services_to_registry_agent(state: RegistryBasedDiscoveryPattern) -> RegistryBasedDiscoveryPattern:
    """Register services to central registry"""
    print("\n📝 Registering Services to Registry...")
    
    registry = CentralRegistry()
    registration_log = []
    
    # Services registering themselves
    services_to_register = [
        ("svc_api_001", "api-service", "http://10.0.1.10:8080"),
        ("svc_api_002", "api-service", "http://10.0.1.11:8080"),
        ("svc_auth_001", "auth-service", "https://10.0.2.10:9000"),
        ("svc_db_001", "database-service", "postgresql://10.0.3.10:5432"),
        ("svc_cache_001", "cache-service", "redis://10.0.4.10:6379"),
        ("svc_queue_001", "queue-service", "amqp://10.0.5.10:5672")
    ]
    
    for service_id, name, endpoint in services_to_register:
        registration = ServiceRegistration(service_id, name, endpoint)
        registration.metadata = {
            "version": "1.0.0",
            "protocol": endpoint.split("://")[0] if "://" in endpoint else "http"
        }
        
        log_entry = registry.register(registration)
        registration_log.append(log_entry)
        
        print(f"  ✓ Registered: {name}")
        print(f"    ID: {service_id}")
        print(f"    Endpoint: {endpoint}")
        print(f"    TTL: {registration.ttl}s")
    
    print(f"\n  Total Registrations: {registry.registration_count}")
    print(f"  Unique Service Names: {len(registry.name_index)}")
    
    registry_state = {sid: reg.to_dict() for sid, reg in registry.registrations.items()}
    
    return {
        **state,
        "central_registry": registry_state,
        "registration_log": registration_log,
        "messages": [f"✓ Registered {len(services_to_register)} services"]
    }


def query_registry_agent(state: RegistryBasedDiscoveryPattern) -> RegistryBasedDiscoveryPattern:
    """Query the central registry"""
    print("\n🔍 Querying Central Registry...")
    
    registry = CentralRegistry()
    
    # Recreate registry
    for service_id, reg_data in state["central_registry"].items():
        registration = ServiceRegistration(
            reg_data["service_id"],
            reg_data["name"],
            reg_data["endpoint"]
        )
        registration.registered_at = reg_data["registered_at"]
        registration.last_heartbeat = reg_data["last_heartbeat"]
        registration.ttl = reg_data["ttl"]
        registration.metadata = reg_data["metadata"]
        registry.register(registration)
    
    # Perform various queries
    queries = [
        ("By Name: api-service", "name", "api-service"),
        ("By Name: auth-service", "name", "auth-service"),
        ("By ID: svc_db_001", "id", "svc_db_001"),
        ("All Services", "all", None)
    ]
    
    query_log = []
    
    for query_name, query_type, query_value in queries:
        print(f"\n  Query: {query_name}")
        
        if query_type == "name":
            results = registry.lookup_by_name(query_value)
        elif query_type == "id":
            result = registry.lookup_by_id(query_value)
            results = [result] if result else []
        else:
            results = registry.get_all_services()
        
        print(f"  Results: {len(results)} service(s)")
        
        for result in results[:3]:
            print(f"    • {result['name']}")
            print(f"      Endpoint: {result['endpoint']}")
        
        query_log.append({
            "query": query_name,
            "type": query_type,
            "value": query_value,
            "result_count": len(results),
            "timestamp": time.time()
        })
    
    print(f"\n  Total Queries: {registry.query_count}")
    
    return {
        **state,
        "query_log": query_log,
        "messages": [f"✓ Executed {len(queries)} queries"]
    }


def monitor_registry_health_agent(state: RegistryBasedDiscoveryPattern) -> RegistryBasedDiscoveryPattern:
    """Monitor registry health"""
    print("\n💚 Monitoring Registry Health...")
    
    registry = CentralRegistry()
    
    # Recreate registry
    for service_id, reg_data in state["central_registry"].items():
        registration = ServiceRegistration(
            reg_data["service_id"],
            reg_data["name"],
            reg_data["endpoint"]
        )
        registration.registered_at = reg_data["registered_at"]
        registration.last_heartbeat = reg_data["last_heartbeat"]
        registration.ttl = reg_data["ttl"]
        registry.register(registration)
    
    # Simulate heartbeats for some services
    active_services = ["svc_api_001", "svc_api_002", "svc_auth_001", "svc_cache_001"]
    
    print(f"  Processing heartbeats...")
    for service_id in active_services:
        if service_id in registry.registrations:
            registry.registrations[service_id].heartbeat()
            print(f"    ✓ Heartbeat: {service_id}")
    
    # Cleanup expired
    expired = registry.cleanup_expired()
    if expired:
        print(f"\n  Cleaned up {len(expired)} expired registrations")
    
    # Calculate statistics
    all_services = registry.get_all_services()
    statistics = {
        "total_registrations": registry.registration_count,
        "active_services": len(all_services),
        "service_types": len(registry.name_index),
        "total_queries": registry.query_count,
        "expired_services": len(expired),
        "queries_per_service": registry.query_count / max(len(all_services), 1)
    }
    
    print(f"\n  Registry Statistics:")
    print(f"    Total Registrations: {statistics['total_registrations']}")
    print(f"    Active Services: {statistics['active_services']}")
    print(f"    Service Types: {statistics['service_types']}")
    print(f"    Total Queries: {statistics['total_queries']}")
    print(f"    Queries/Service: {statistics['queries_per_service']:.2f}")
    
    return {
        **state,
        "registry_statistics": statistics,
        "messages": ["✓ Registry health monitored"]
    }


def generate_registry_discovery_report_agent(state: RegistryBasedDiscoveryPattern) -> RegistryBasedDiscoveryPattern:
    """Generate registry-based discovery report"""
    print("\n" + "="*70)
    print("REGISTRY-BASED DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n🏛️ Central Registry:")
    print(f"  Registered Services: {len(state['central_registry'])}")
    
    print(f"\n🔧 Registered Services:")
    for reg_data in list(state['central_registry'].values()):
        status_icon = "✅" if reg_data["alive"] else "⚠️"
        print(f"\n  {status_icon} {reg_data['name']}:")
        print(f"      ID: {reg_data['service_id']}")
        print(f"      Endpoint: {reg_data['endpoint']}")
        print(f"      TTL: {reg_data['ttl']}s")
        print(f"      Alive: {reg_data['alive']}")
    
    print(f"\n📝 Registration Log:")
    for log_entry in state['registration_log']:
        print(f"  • {log_entry['service_id']}: {log_entry['status']}")
    
    print(f"\n🔍 Query Log:")
    for query in state['query_log']:
        print(f"  • {query['query']}: {query['result_count']} result(s)")
    
    print(f"\n📊 Registry Statistics:")
    stats = state['registry_statistics']
    if stats:
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\n💡 Registry-Based Discovery Benefits:")
    print("  ✓ Centralized management")
    print("  ✓ Dynamic registration")
    print("  ✓ Service discovery")
    print("  ✓ Health monitoring")
    print("  ✓ Load balancing support")
    print("  ✓ Auto cleanup")
    
    print(f"\n🔧 Registry Features:")
    print("  • Self-registration")
    print("  • Heartbeat monitoring")
    print("  • TTL-based expiration")
    print("  • Name-based lookup")
    print("  • ID-based lookup")
    print("  • Batch queries")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Microservices")
    print("  • Service mesh")
    print("  • Cloud platforms")
    print("  • Container orchestration")
    print("  • API gateways")
    print("  • Load balancers")
    
    print(f"\n🎯 Popular Implementations:")
    print("  • Consul")
    print("  • Eureka")
    print("  • ZooKeeper")
    print("  • etcd")
    print("  • Kubernetes Service Registry")
    
    print("\n" + "="*70)
    print("✅ Registry-Based Discovery Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_registry_based_discovery_graph():
    """Create registry-based discovery workflow"""
    workflow = StateGraph(RegistryBasedDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_central_registry_agent)
    workflow.add_node("register", register_services_to_registry_agent)
    workflow.add_node("query", query_registry_agent)
    workflow.add_node("monitor", monitor_registry_health_agent)
    workflow.add_node("report", generate_registry_discovery_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "register")
    workflow.add_edge("register", "query")
    workflow.add_edge("query", "monitor")
    workflow.add_edge("monitor", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 297: Registry-Based Discovery MCP Pattern")
    print("="*70)
    
    app = create_registry_based_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "central_registry": {},
        "registration_log": [],
        "query_log": [],
        "registry_statistics": {}
    })
    
    print("\n✅ Registry-Based Discovery Pattern Complete!")


if __name__ == "__main__":
    main()
