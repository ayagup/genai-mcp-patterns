"""
Pattern 299: Heartbeat Discovery MCP Pattern

This pattern demonstrates service discovery using heartbeat mechanisms
where services send periodic heartbeat signals to indicate they are alive
and available.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class HeartbeatDiscoveryPattern(TypedDict):
    """State for heartbeat-based discovery"""
    messages: Annotated[List[str], add]
    services: Dict[str, Any]
    heartbeat_log: List[Dict[str, Any]]
    health_status: Dict[str, Any]
    monitoring_statistics: Dict[str, Any]


class ServiceHeartbeat:
    """Service with heartbeat monitoring"""
    
    def __init__(self, service_id: str, name: str, endpoint: str):
        self.service_id = service_id
        self.name = name
        self.endpoint = endpoint
        self.first_seen = time.time()
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 10  # seconds
        self.timeout_threshold = 30  # seconds
        self.heartbeat_count = 0
        self.missed_heartbeats = 0
        self.status = "healthy"
        self.metadata = {}
    
    def send_heartbeat(self):
        """Send heartbeat signal"""
        current_time = time.time()
        time_since_last = current_time - self.last_heartbeat
        
        # Check if heartbeat is late
        if time_since_last > self.heartbeat_interval * 1.5:
            self.missed_heartbeats += 1
        
        self.last_heartbeat = current_time
        self.heartbeat_count += 1
        
        # Update status
        if self.is_healthy():
            self.status = "healthy"
        elif self.is_degraded():
            self.status = "degraded"
        else:
            self.status = "unhealthy"
        
        return {
            "service_id": self.service_id,
            "timestamp": current_time,
            "status": self.status,
            "heartbeat_number": self.heartbeat_count
        }
    
    def is_healthy(self):
        """Check if service is healthy"""
        time_since_last = time.time() - self.last_heartbeat
        return time_since_last < self.heartbeat_interval * 2
    
    def is_degraded(self):
        """Check if service is degraded"""
        time_since_last = time.time() - self.last_heartbeat
        return self.heartbeat_interval * 2 <= time_since_last < self.timeout_threshold
    
    def is_alive(self):
        """Check if service is still alive"""
        time_since_last = time.time() - self.last_heartbeat
        return time_since_last < self.timeout_threshold
    
    def get_uptime(self):
        """Get service uptime"""
        return time.time() - self.first_seen
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "service_id": self.service_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "heartbeat_count": self.heartbeat_count,
            "missed_heartbeats": self.missed_heartbeats,
            "uptime": self.get_uptime(),
            "is_alive": self.is_alive(),
            "is_healthy": self.is_healthy()
        }


class HeartbeatMonitor:
    """Monitor service heartbeats"""
    
    def __init__(self):
        self.services = {}
        self.heartbeat_history = []
        self.alert_threshold = 3
    
    def register_service(self, service: ServiceHeartbeat):
        """Register a service for monitoring"""
        self.services[service.service_id] = service
        return {
            "service_id": service.service_id,
            "registered_at": time.time()
        }
    
    def process_heartbeat(self, service_id: str):
        """Process heartbeat from service"""
        if service_id in self.services:
            service = self.services[service_id]
            heartbeat_result = service.send_heartbeat()
            self.heartbeat_history.append(heartbeat_result)
            return heartbeat_result
        return None
    
    def check_health(self):
        """Check health of all services"""
        health_report = {
            "healthy": [],
            "degraded": [],
            "unhealthy": [],
            "dead": []
        }
        
        for service in self.services.values():
            if service.is_healthy():
                health_report["healthy"].append(service.service_id)
            elif service.is_degraded():
                health_report["degraded"].append(service.service_id)
            elif service.is_alive():
                health_report["unhealthy"].append(service.service_id)
            else:
                health_report["dead"].append(service.service_id)
        
        return health_report
    
    def get_available_services(self):
        """Get list of available services"""
        return [
            service.to_dict()
            for service in self.services.values()
            if service.is_healthy()
        ]
    
    def cleanup_dead_services(self):
        """Remove dead services"""
        dead_services = [
            sid for sid, svc in self.services.items()
            if not svc.is_alive()
        ]
        
        for service_id in dead_services:
            del self.services[service_id]
        
        return dead_services


def initialize_heartbeat_monitor_agent(state: HeartbeatDiscoveryPattern) -> HeartbeatDiscoveryPattern:
    """Initialize heartbeat monitoring system"""
    print("\n💓 Initializing Heartbeat Monitor...")
    
    print(f"  Heartbeat Interval: 10s")
    print(f"  Timeout Threshold: 30s")
    print(f"  Alert Threshold: 3 missed")
    
    print(f"\n  Features:")
    print(f"    • Automatic health detection")
    print(f"    • Missed heartbeat tracking")
    print(f"    • Service status (healthy/degraded/unhealthy)")
    print(f"    • Dead service cleanup")
    
    return {
        **state,
        "services": {},
        "heartbeat_log": [],
        "health_status": {},
        "monitoring_statistics": {},
        "messages": ["✓ Heartbeat monitor initialized"]
    }


def register_services_with_heartbeat_agent(state: HeartbeatDiscoveryPattern) -> HeartbeatDiscoveryPattern:
    """Register services with heartbeat monitoring"""
    print("\n📝 Registering Services...")
    
    monitor = HeartbeatMonitor()
    
    # Services to register
    services_to_register = [
        ("svc_web_001", "web-server", "http://10.0.1.10:80"),
        ("svc_web_002", "web-server", "http://10.0.1.11:80"),
        ("svc_api_001", "api-server", "http://10.0.2.10:8080"),
        ("svc_api_002", "api-server", "http://10.0.2.11:8080"),
        ("svc_db_001", "database", "postgresql://10.0.3.10:5432"),
        ("svc_cache_001", "cache", "redis://10.0.4.10:6379")
    ]
    
    for service_id, name, endpoint in services_to_register:
        service = ServiceHeartbeat(service_id, name, endpoint)
        service.metadata = {
            "version": "1.0.0",
            "region": "us-east-1"
        }
        
        monitor.register_service(service)
        
        print(f"  ✓ Registered: {name}")
        print(f"    ID: {service_id}")
        print(f"    Endpoint: {endpoint}")
        print(f"    Interval: {service.heartbeat_interval}s")
    
    print(f"\n  Total Services: {len(monitor.services)}")
    
    # Convert to state
    services_dict = {sid: svc.to_dict() for sid, svc in monitor.services.items()}
    
    return {
        **state,
        "services": services_dict,
        "messages": [f"✓ Registered {len(services_to_register)} services"]
    }


def simulate_heartbeat_signals_agent(state: HeartbeatDiscoveryPattern) -> HeartbeatDiscoveryPattern:
    """Simulate heartbeat signals from services"""
    print("\n💓 Simulating Heartbeat Signals...")
    
    monitor = HeartbeatMonitor()
    
    # Recreate services
    for service_id, service_data in state["services"].items():
        service = ServiceHeartbeat(
            service_data["service_id"],
            service_data["name"],
            service_data["endpoint"]
        )
        service.first_seen = service_data.get("last_heartbeat", time.time())
        service.last_heartbeat = service_data.get("last_heartbeat", time.time())
        monitor.register_service(service)
    
    # Simulate multiple heartbeat cycles
    heartbeat_log = []
    
    # Cycle 1: All services send heartbeat
    print(f"\n  Cycle 1: All services healthy")
    for service_id in list(monitor.services.keys()):
        result = monitor.process_heartbeat(service_id)
        if result:
            heartbeat_log.append(result)
            print(f"    ✓ {service_id}: {result['status']}")
    
    # Cycle 2: One service misses heartbeat
    print(f"\n  Cycle 2: One service degraded")
    for service_id in list(monitor.services.keys()):
        if service_id != "svc_api_002":  # This one misses
            result = monitor.process_heartbeat(service_id)
            if result:
                heartbeat_log.append(result)
                print(f"    ✓ {service_id}: {result['status']}")
        else:
            print(f"    ⚠️ {service_id}: missed heartbeat")
    
    # Cycle 3: All services recover
    print(f"\n  Cycle 3: Services recover")
    for service_id in list(monitor.services.keys()):
        result = monitor.process_heartbeat(service_id)
        if result:
            heartbeat_log.append(result)
            print(f"    ✓ {service_id}: {result['status']}")
    
    print(f"\n  Total Heartbeats: {len(heartbeat_log)}")
    
    # Update services state
    services_dict = {sid: svc.to_dict() for sid, svc in monitor.services.items()}
    
    return {
        **state,
        "services": services_dict,
        "heartbeat_log": heartbeat_log,
        "messages": [f"✓ Processed {len(heartbeat_log)} heartbeats"]
    }


def monitor_service_health_agent(state: HeartbeatDiscoveryPattern) -> HeartbeatDiscoveryPattern:
    """Monitor service health"""
    print("\n🏥 Monitoring Service Health...")
    
    monitor = HeartbeatMonitor()
    
    # Recreate services
    for service_id, service_data in state["services"].items():
        service = ServiceHeartbeat(
            service_data["service_id"],
            service_data["name"],
            service_data["endpoint"]
        )
        service.last_heartbeat = service_data["last_heartbeat"]
        service.heartbeat_count = service_data["heartbeat_count"]
        service.missed_heartbeats = service_data["missed_heartbeats"]
        service.status = service_data["status"]
        monitor.register_service(service)
    
    # Check health
    health_report = monitor.check_health()
    
    print(f"  Healthy: {len(health_report['healthy'])} services")
    for service_id in health_report['healthy']:
        print(f"    ✅ {service_id}")
    
    print(f"\n  Degraded: {len(health_report['degraded'])} services")
    for service_id in health_report['degraded']:
        print(f"    ⚠️ {service_id}")
    
    print(f"\n  Unhealthy: {len(health_report['unhealthy'])} services")
    for service_id in health_report['unhealthy']:
        print(f"    ❌ {service_id}")
    
    print(f"\n  Dead: {len(health_report['dead'])} services")
    for service_id in health_report['dead']:
        print(f"    💀 {service_id}")
    
    # Get available services
    available = monitor.get_available_services()
    print(f"\n  Available for Discovery: {len(available)} services")
    
    return {
        **state,
        "health_status": health_report,
        "messages": [f"✓ Health check complete"]
    }


def generate_heartbeat_report_agent(state: HeartbeatDiscoveryPattern) -> HeartbeatDiscoveryPattern:
    """Generate heartbeat discovery report"""
    print("\n" + "="*70)
    print("HEARTBEAT DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n💓 Registered Services:")
    for service_id, service_data in state["services"].items():
        status_icon = "✅" if service_data["is_healthy"] else "⚠️"
        print(f"\n  {status_icon} {service_data['name']}:")
        print(f"      ID: {service_data['service_id']}")
        print(f"      Endpoint: {service_data['endpoint']}")
        print(f"      Status: {service_data['status']}")
        print(f"      Heartbeats: {service_data['heartbeat_count']}")
        print(f"      Missed: {service_data['missed_heartbeats']}")
        print(f"      Uptime: {service_data['uptime']:.1f}s")
    
    print(f"\n📊 Heartbeat Log:")
    for i, heartbeat in enumerate(state['heartbeat_log'][:10], 1):
        print(f"  {i}. {heartbeat['service_id']}")
        print(f"     Status: {heartbeat['status']}")
        print(f"     Heartbeat #: {heartbeat['heartbeat_number']}")
    
    print(f"\n🏥 Health Status:")
    health = state['health_status']
    if health:
        print(f"  Healthy: {len(health['healthy'])} services")
        print(f"  Degraded: {len(health['degraded'])} services")
        print(f"  Unhealthy: {len(health['unhealthy'])} services")
        print(f"  Dead: {len(health['dead'])} services")
    
    # Calculate statistics
    total_services = len(state['services'])
    total_heartbeats = len(state['heartbeat_log'])
    
    statistics = {
        "total_services": total_services,
        "total_heartbeats": total_heartbeats,
        "avg_heartbeats_per_service": total_heartbeats / max(total_services, 1),
        "healthy_percentage": len(health.get('healthy', [])) / max(total_services, 1) if health else 0
    }
    
    print(f"\n📈 Statistics:")
    print(f"  Total Services: {statistics['total_services']}")
    print(f"  Total Heartbeats: {statistics['total_heartbeats']}")
    print(f"  Avg Heartbeats/Service: {statistics['avg_heartbeats_per_service']:.1f}")
    print(f"  Health Rate: {statistics['healthy_percentage']:.1%}")
    
    print(f"\n💡 Heartbeat Discovery Benefits:")
    print("  ✓ Real-time health monitoring")
    print("  ✓ Automatic failure detection")
    print("  ✓ No polling overhead")
    print("  ✓ Simple implementation")
    print("  ✓ Lightweight protocol")
    print("  ✓ Fast failover")
    
    print(f"\n🔧 Heartbeat Mechanism:")
    print("  • Periodic signals (10s)")
    print("  • Timeout detection (30s)")
    print("  • Status tracking (healthy/degraded/unhealthy)")
    print("  • Missed heartbeat counting")
    print("  • Automatic cleanup")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Cluster management")
    print("  • Load balancer health checks")
    print("  • Container orchestration")
    print("  • Distributed systems")
    print("  • High availability systems")
    print("  • Service mesh")
    
    print(f"\n🎯 Popular Implementations:")
    print("  • Kubernetes Liveness Probes")
    print("  • Docker Health Checks")
    print("  • Consul Health Checks")
    print("  • Nagios")
    print("  • Zabbix")
    
    print("\n" + "="*70)
    print("✅ Heartbeat Discovery Pattern Complete!")
    print("="*70)
    
    return {
        **state,
        "monitoring_statistics": statistics,
        "messages": ["✓ Report generated"]
    }


def create_heartbeat_discovery_graph():
    """Create heartbeat discovery workflow"""
    workflow = StateGraph(HeartbeatDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_heartbeat_monitor_agent)
    workflow.add_node("register", register_services_with_heartbeat_agent)
    workflow.add_node("simulate", simulate_heartbeat_signals_agent)
    workflow.add_node("monitor", monitor_service_health_agent)
    workflow.add_node("report", generate_heartbeat_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "register")
    workflow.add_edge("register", "simulate")
    workflow.add_edge("simulate", "monitor")
    workflow.add_edge("monitor", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 299: Heartbeat Discovery MCP Pattern")
    print("="*70)
    
    app = create_heartbeat_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "services": {},
        "heartbeat_log": [],
        "health_status": {},
        "monitoring_statistics": {}
    })
    
    print("\n✅ Heartbeat Discovery Pattern Complete!")


if __name__ == "__main__":
    main()
