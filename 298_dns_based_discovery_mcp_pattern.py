"""
Pattern 298: DNS-Based Discovery MCP Pattern

This pattern demonstrates service discovery using DNS (Domain Name System)
where services are registered as DNS records and clients resolve services
using DNS queries.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class DNSBasedDiscoveryPattern(TypedDict):
    """State for DNS-based discovery"""
    messages: Annotated[List[str], add]
    dns_zones: Dict[str, Any]
    dns_records: List[Dict[str, Any]]
    resolution_log: List[Dict[str, Any]]
    dns_statistics: Dict[str, Any]


class DNSRecord:
    """DNS record entry"""
    
    def __init__(self, hostname: str, record_type: str, value: str):
        self.hostname = hostname
        self.record_type = record_type  # A, SRV, CNAME, TXT
        self.value = value
        self.ttl = 300  # seconds
        self.created_at = time.time()
    
    def is_valid(self):
        """Check if record is still valid"""
        return (time.time() - self.created_at) < self.ttl
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "hostname": self.hostname,
            "type": self.record_type,
            "value": self.value,
            "ttl": self.ttl,
            "created_at": self.created_at,
            "valid": self.is_valid()
        }


class DNSZone:
    """DNS zone management"""
    
    def __init__(self, zone_name: str):
        self.zone_name = zone_name
        self.records = {}
        self.record_count = 0
        self.query_count = 0
    
    def add_record(self, record: DNSRecord):
        """Add a DNS record"""
        key = f"{record.hostname}:{record.record_type}"
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(record)
        self.record_count += 1
        
        return {
            "hostname": record.hostname,
            "type": record.record_type,
            "status": "added",
            "timestamp": time.time()
        }
    
    def resolve(self, hostname: str, record_type: str = "A"):
        """Resolve DNS query"""
        self.query_count += 1
        key = f"{hostname}:{record_type}"
        
        if key in self.records:
            valid_records = [r for r in self.records[key] if r.is_valid()]
            return [r.to_dict() for r in valid_records]
        
        return []
    
    def get_all_records(self):
        """Get all valid records"""
        all_records = []
        for record_list in self.records.values():
            for record in record_list:
                if record.is_valid():
                    all_records.append(record.to_dict())
        return all_records


class ServiceRegistry:
    """Service registry using DNS"""
    
    def __init__(self):
        self.zones = {}
    
    def get_zone(self, zone_name: str) -> DNSZone:
        """Get or create zone"""
        if zone_name not in self.zones:
            self.zones[zone_name] = DNSZone(zone_name)
        return self.zones[zone_name]
    
    def register_service(self, service_name: str, zone: str, ip_address: str, port: int = None):
        """Register a service"""
        dns_zone = self.get_zone(zone)
        
        # Add A record
        hostname = f"{service_name}.{zone}"
        a_record = DNSRecord(hostname, "A", ip_address)
        dns_zone.add_record(a_record)
        
        # Add SRV record if port provided
        if port:
            srv_value = f"0 10 {port} {hostname}"
            srv_record = DNSRecord(f"_{service_name}._tcp.{zone}", "SRV", srv_value)
            dns_zone.add_record(srv_record)
        
        return {
            "service": service_name,
            "zone": zone,
            "hostname": hostname,
            "ip": ip_address,
            "port": port
        }
    
    def discover_service(self, service_name: str, zone: str):
        """Discover service by name"""
        dns_zone = self.get_zone(zone)
        hostname = f"{service_name}.{zone}"
        
        # Try A record
        a_records = dns_zone.resolve(hostname, "A")
        srv_records = dns_zone.resolve(f"_{service_name}._tcp.{zone}", "SRV")
        
        return {
            "hostname": hostname,
            "a_records": a_records,
            "srv_records": srv_records
        }


def initialize_dns_system_agent(state: DNSBasedDiscoveryPattern) -> DNSBasedDiscoveryPattern:
    """Initialize DNS-based discovery system"""
    print("\n🌐 Initializing DNS-Based Discovery System...")
    
    zones = [
        "services.local",
        "api.internal",
        "backend.cluster"
    ]
    
    print(f"  DNS Zones: {len(zones)}")
    for zone in zones:
        print(f"    • {zone}")
    
    print(f"\n  Features:")
    print(f"    • A Records (IP resolution)")
    print(f"    • SRV Records (Service + Port)")
    print(f"    • CNAME Records (Aliases)")
    print(f"    • TTL Management")
    
    return {
        **state,
        "dns_zones": {},
        "dns_records": [],
        "resolution_log": [],
        "dns_statistics": {},
        "messages": ["✓ DNS system initialized"]
    }


def register_services_to_dns_agent(state: DNSBasedDiscoveryPattern) -> DNSBasedDiscoveryPattern:
    """Register services to DNS"""
    print("\n📝 Registering Services to DNS...")
    
    registry = ServiceRegistry()
    dns_records = []
    
    # Services to register
    services = [
        ("api-gateway", "services.local", "10.0.1.10", 8080),
        ("api-gateway", "services.local", "10.0.1.11", 8080),
        ("auth-service", "services.local", "10.0.2.10", 9000),
        ("user-service", "api.internal", "10.0.3.10", 8001),
        ("order-service", "api.internal", "10.0.3.11", 8002),
        ("payment-service", "api.internal", "10.0.3.12", 8003),
        ("database", "backend.cluster", "10.0.4.10", 5432),
        ("cache", "backend.cluster", "10.0.4.20", 6379)
    ]
    
    for service_name, zone, ip, port in services:
        result = registry.register_service(service_name, zone, ip, port)
        dns_records.append(result)
        
        print(f"  ✓ Registered: {result['hostname']}")
        print(f"    IP: {result['ip']}")
        print(f"    Port: {result['port']}")
    
    # Convert zones to dict
    zones_dict = {}
    for zone_name, zone in registry.zones.items():
        zones_dict[zone_name] = {
            "zone_name": zone_name,
            "record_count": zone.record_count,
            "records": zone.get_all_records()
        }
    
    print(f"\n  Total Services: {len(services)}")
    print(f"  DNS Zones: {len(registry.zones)}")
    print(f"  Total Records: {sum(z.record_count for z in registry.zones.values())}")
    
    return {
        **state,
        "dns_zones": zones_dict,
        "dns_records": dns_records,
        "messages": [f"✓ Registered {len(services)} services to DNS"]
    }


def resolve_dns_queries_agent(state: DNSBasedDiscoveryPattern) -> DNSBasedDiscoveryPattern:
    """Resolve DNS queries"""
    print("\n🔍 Resolving DNS Queries...")
    
    registry = ServiceRegistry()
    
    # Recreate zones from state
    for zone_name, zone_data in state["dns_zones"].items():
        zone = registry.get_zone(zone_name)
        for record_data in zone_data["records"]:
            record = DNSRecord(
                record_data["hostname"],
                record_data["type"],
                record_data["value"]
            )
            record.created_at = record_data["created_at"]
            record.ttl = record_data["ttl"]
            zone.add_record(record)
    
    # Perform DNS queries
    queries = [
        ("api-gateway", "services.local"),
        ("auth-service", "services.local"),
        ("user-service", "api.internal"),
        ("database", "backend.cluster")
    ]
    
    resolution_log = []
    
    for service_name, zone in queries:
        print(f"\n  Query: {service_name}.{zone}")
        
        result = registry.discover_service(service_name, zone)
        
        print(f"  A Records: {len(result['a_records'])}")
        for a_record in result['a_records']:
            print(f"    → {a_record['value']}")
        
        print(f"  SRV Records: {len(result['srv_records'])}")
        for srv_record in result['srv_records']:
            print(f"    → {srv_record['value']}")
        
        resolution_log.append({
            "query": f"{service_name}.{zone}",
            "a_records": len(result['a_records']),
            "srv_records": len(result['srv_records']),
            "timestamp": time.time()
        })
    
    print(f"\n  Total Queries: {len(queries)}")
    
    return {
        **state,
        "resolution_log": resolution_log,
        "messages": [f"✓ Resolved {len(queries)} DNS queries"]
    }


def analyze_dns_performance_agent(state: DNSBasedDiscoveryPattern) -> DNSBasedDiscoveryPattern:
    """Analyze DNS performance"""
    print("\n📊 Analyzing DNS Performance...")
    
    # Calculate statistics
    total_zones = len(state["dns_zones"])
    total_records = sum(zone["record_count"] for zone in state["dns_zones"].values())
    total_queries = len(state["resolution_log"])
    
    # Analyze record types
    record_types = {}
    for zone_data in state["dns_zones"].values():
        for record in zone_data["records"]:
            rtype = record["type"]
            record_types[rtype] = record_types.get(rtype, 0) + 1
    
    # Analyze query success
    successful_queries = sum(
        1 for log in state["resolution_log"]
        if log["a_records"] > 0 or log["srv_records"] > 0
    )
    
    statistics = {
        "total_zones": total_zones,
        "total_records": total_records,
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "success_rate": successful_queries / max(total_queries, 1),
        "record_types": record_types,
        "avg_records_per_zone": total_records / max(total_zones, 1)
    }
    
    print(f"  Total Zones: {statistics['total_zones']}")
    print(f"  Total Records: {statistics['total_records']}")
    print(f"  Total Queries: {statistics['total_queries']}")
    print(f"  Success Rate: {statistics['success_rate']:.1%}")
    
    print(f"\n  Record Type Distribution:")
    for rtype, count in record_types.items():
        print(f"    {rtype}: {count}")
    
    return {
        **state,
        "dns_statistics": statistics,
        "messages": ["✓ DNS performance analyzed"]
    }


def generate_dns_discovery_report_agent(state: DNSBasedDiscoveryPattern) -> DNSBasedDiscoveryPattern:
    """Generate DNS-based discovery report"""
    print("\n" + "="*70)
    print("DNS-BASED DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n🌐 DNS Zones:")
    for zone_name, zone_data in state["dns_zones"].items():
        print(f"\n  Zone: {zone_name}")
        print(f"    Records: {zone_data['record_count']}")
        
        # Show sample records
        for record in zone_data["records"][:3]:
            print(f"      • {record['hostname']} ({record['type']})")
            print(f"        → {record['value']}")
    
    print(f"\n📝 Registered Services:")
    for service in state["dns_records"]:
        print(f"\n  {service['hostname']}:")
        print(f"    IP: {service['ip']}")
        print(f"    Port: {service['port']}")
        print(f"    Zone: {service['zone']}")
    
    print(f"\n🔍 DNS Resolutions:")
    for resolution in state["resolution_log"]:
        print(f"  • {resolution['query']}")
        print(f"    A Records: {resolution['a_records']}")
        print(f"    SRV Records: {resolution['srv_records']}")
    
    print(f"\n📊 DNS Statistics:")
    stats = state["dns_statistics"]
    if stats:
        print(f"  Total Zones: {stats['total_zones']}")
        print(f"  Total Records: {stats['total_records']}")
        print(f"  Total Queries: {stats['total_queries']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Avg Records/Zone: {stats['avg_records_per_zone']:.1f}")
    
    print(f"\n💡 DNS-Based Discovery Benefits:")
    print("  ✓ Standard protocol")
    print("  ✓ Wide support")
    print("  ✓ Built-in caching")
    print("  ✓ Load balancing")
    print("  ✓ Failover support")
    print("  ✓ Hierarchical structure")
    
    print(f"\n🔧 DNS Record Types:")
    print("  • A: IP address mapping")
    print("  • SRV: Service + port + priority")
    print("  • CNAME: Canonical name alias")
    print("  • TXT: Text metadata")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Kubernetes DNS")
    print("  • Cloud load balancers")
    print("  • Microservice discovery")
    print("  • Container orchestration")
    print("  • Geographic routing")
    print("  • Service failover")
    
    print(f"\n🎯 Popular Implementations:")
    print("  • Kubernetes CoreDNS")
    print("  • Amazon Route 53")
    print("  • Azure DNS")
    print("  • Google Cloud DNS")
    print("  • Consul DNS")
    
    print("\n" + "="*70)
    print("✅ DNS-Based Discovery Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_dns_based_discovery_graph():
    """Create DNS-based discovery workflow"""
    workflow = StateGraph(DNSBasedDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_dns_system_agent)
    workflow.add_node("register", register_services_to_dns_agent)
    workflow.add_node("resolve", resolve_dns_queries_agent)
    workflow.add_node("analyze", analyze_dns_performance_agent)
    workflow.add_node("report", generate_dns_discovery_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "register")
    workflow.add_edge("register", "resolve")
    workflow.add_edge("resolve", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 298: DNS-Based Discovery MCP Pattern")
    print("="*70)
    
    app = create_dns_based_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "dns_zones": {},
        "dns_records": [],
        "resolution_log": [],
        "dns_statistics": {}
    })
    
    print("\n✅ DNS-Based Discovery Pattern Complete!")


if __name__ == "__main__":
    main()
