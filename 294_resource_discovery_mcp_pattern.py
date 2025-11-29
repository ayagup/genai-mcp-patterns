"""
Pattern 294: Resource Discovery MCP Pattern

This pattern demonstrates discovering and locating resources such as
data sources, APIs, tools, and computational resources in a system.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class ResourceDiscoveryPattern(TypedDict):
    """State for resource discovery"""
    messages: Annotated[List[str], add]
    resource_inventory: Dict[str, Any]
    discovered_resources: List[Dict[str, Any]]
    resource_metrics: Dict[str, Any]
    allocation_recommendations: List[Dict[str, Any]]


class Resource:
    """Represents a resource"""
    
    def __init__(self, resource_id: str, name: str, resource_type: str):
        self.resource_id = resource_id
        self.name = name
        self.resource_type = resource_type
        self.location = None
        self.capacity = 100
        self.used = 0
        self.available = True
        self.tags = []
        self.metadata = {}
    
    def add_tag(self, tag: str):
        """Add tag"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_usage(self, used: int):
        """Set resource usage"""
        self.used = min(used, self.capacity)
    
    def get_availability_percent(self):
        """Get availability percentage"""
        return ((self.capacity - self.used) / self.capacity) * 100
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "resource_id": self.resource_id,
            "name": self.name,
            "resource_type": self.resource_type,
            "location": self.location,
            "capacity": self.capacity,
            "used": self.used,
            "available": self.available,
            "availability_percent": self.get_availability_percent(),
            "tags": self.tags,
            "metadata": self.metadata
        }


class ResourceInventory:
    """Inventory of resources"""
    
    def __init__(self):
        self.resources = {}
        self.type_index = {}
        self.tag_index = {}
    
    def register_resource(self, resource: Resource):
        """Register a resource"""
        self.resources[resource.resource_id] = resource
        
        # Index by type
        if resource.resource_type not in self.type_index:
            self.type_index[resource.resource_type] = []
        self.type_index[resource.resource_type].append(resource.resource_id)
        
        # Index by tags
        for tag in resource.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(resource.resource_id)
    
    def discover_by_type(self, resource_type: str):
        """Discover resources by type"""
        resource_ids = self.type_index.get(resource_type, [])
        return [self.resources[rid].to_dict() for rid in resource_ids if rid in self.resources]
    
    def discover_by_tag(self, tag: str):
        """Discover resources by tag"""
        resource_ids = self.tag_index.get(tag, [])
        return [self.resources[rid].to_dict() for rid in resource_ids if rid in self.resources]
    
    def discover_available(self, resource_type: str = None, min_availability: float = 0):
        """Discover available resources"""
        results = []
        for resource in self.resources.values():
            if resource.available:
                if resource_type is None or resource.resource_type == resource_type:
                    if resource.get_availability_percent() >= min_availability:
                        results.append(resource.to_dict())
        return results
    
    def find_best_resource(self, resource_type: str):
        """Find best resource based on availability"""
        candidates = self.discover_by_type(resource_type)
        
        if not candidates:
            return None
        
        # Sort by availability
        candidates.sort(key=lambda r: r["availability_percent"], reverse=True)
        return candidates[0] if candidates else None


def initialize_resource_inventory_agent(state: ResourceDiscoveryPattern) -> ResourceDiscoveryPattern:
    """Initialize resource inventory"""
    print("\n📦 Initializing Resource Inventory...")
    
    inventory = ResourceInventory()
    
    print(f"  Inventory: Ready")
    print(f"  Features:")
    print(f"    • Resource registration")
    print(f"    • Type-based indexing")
    print(f"    • Tag-based discovery")
    print(f"    • Availability tracking")
    
    return {
        **state,
        "resource_inventory": {},
        "discovered_resources": [],
        "resource_metrics": {},
        "allocation_recommendations": [],
        "messages": ["✓ Resource inventory initialized"]
    }


def register_resources_agent(state: ResourceDiscoveryPattern) -> ResourceDiscoveryPattern:
    """Register various resources"""
    print("\n📝 Registering Resources...")
    
    inventory = ResourceInventory()
    
    # Define resources
    resources_config = [
        {"id": "db_primary", "name": "Primary Database", "type": "database", 
         "location": "us-east-1a", "capacity": 1000, "used": 650, "tags": ["production", "critical", "postgres"]},
        
        {"id": "db_replica_1", "name": "Database Replica 1", "type": "database",
         "location": "us-east-1b", "capacity": 1000, "used": 450, "tags": ["production", "replica", "postgres"]},
        
        {"id": "cache_primary", "name": "Redis Cache", "type": "cache",
         "location": "us-east-1a", "capacity": 500, "used": 200, "tags": ["production", "cache", "redis"]},
        
        {"id": "compute_node_1", "name": "Compute Node 1", "type": "compute",
         "location": "us-west-2a", "capacity": 100, "used": 75, "tags": ["production", "cpu", "ml"]},
        
        {"id": "compute_node_2", "name": "Compute Node 2", "type": "compute",
         "location": "us-west-2b", "capacity": 100, "used": 30, "tags": ["production", "cpu", "ml"]},
        
        {"id": "storage_s3", "name": "S3 Storage", "type": "storage",
         "location": "us-east-1", "capacity": 10000, "used": 3500, "tags": ["production", "object-storage", "s3"]},
        
        {"id": "api_gateway", "name": "API Gateway", "type": "api",
         "location": "global", "capacity": 10000, "used": 4200, "tags": ["production", "gateway", "rest"]},
        
        {"id": "message_queue", "name": "Message Queue", "type": "queue",
         "location": "us-east-1", "capacity": 5000, "used": 1800, "tags": ["production", "messaging", "sqs"]}
    ]
    
    for config in resources_config:
        resource = Resource(config["id"], config["name"], config["type"])
        resource.location = config["location"]
        resource.capacity = config["capacity"]
        resource.used = config["used"]
        
        for tag in config["tags"]:
            resource.add_tag(tag)
        
        inventory.register_resource(resource)
        
        availability = resource.get_availability_percent()
        print(f"  ✓ Registered: {resource.name}")
        print(f"    Type: {resource.resource_type}")
        print(f"    Location: {resource.location}")
        print(f"    Availability: {availability:.1f}%")
    
    print(f"\n  Total Resources: {len(inventory.resources)}")
    print(f"  Resource Types: {len(inventory.type_index)}")
    
    resource_dict = {rid: res.to_dict() for rid, res in inventory.resources.items()}
    
    return {
        **state,
        "resource_inventory": resource_dict,
        "messages": [f"✓ Registered {len(resources_config)} resources"]
    }


def discover_resources_agent(state: ResourceDiscoveryPattern) -> ResourceDiscoveryPattern:
    """Discover resources by various criteria"""
    print("\n🔍 Discovering Resources...")
    
    inventory = ResourceInventory()
    
    # Recreate inventory
    for res_id, res_data in state["resource_inventory"].items():
        resource = Resource(res_data["resource_id"], res_data["name"], res_data["resource_type"])
        resource.location = res_data["location"]
        resource.capacity = res_data["capacity"]
        resource.used = res_data["used"]
        resource.tags = res_data["tags"]
        inventory.register_resource(resource)
    
    # Discovery queries
    discovery_queries = [
        ("Type: Database", inventory.discover_by_type("database")),
        ("Type: Compute", inventory.discover_by_type("compute")),
        ("Tag: Production", inventory.discover_by_tag("production")),
        ("Tag: Critical", inventory.discover_by_tag("critical")),
        ("Available (>50%)", inventory.discover_available(min_availability=50))
    ]
    
    all_discovered = []
    
    for query_name, results in discovery_queries:
        print(f"\n  Query: {query_name}")
        print(f"  Results: {len(results)} resource(s)")
        
        for res in results[:3]:
            print(f"    • {res['name']}")
            print(f"      Availability: {res['availability_percent']:.1f}%")
        
        all_discovered.extend(results)
    
    return {
        **state,
        "discovered_resources": all_discovered,
        "messages": [f"✓ Executed {len(discovery_queries)} queries"]
    }


def analyze_resource_metrics_agent(state: ResourceDiscoveryPattern) -> ResourceDiscoveryPattern:
    """Analyze resource metrics"""
    print("\n📊 Analyzing Resource Metrics...")
    
    resources = state["resource_inventory"].values()
    
    # Calculate metrics
    metrics = {
        "total_resources": len(resources),
        "total_capacity": sum(r["capacity"] for r in resources),
        "total_used": sum(r["used"] for r in resources),
        "by_type": {},
        "by_location": {},
        "high_utilization": [],
        "low_utilization": []
    }
    
    # Overall utilization
    if metrics["total_capacity"] > 0:
        metrics["overall_utilization"] = (metrics["total_used"] / metrics["total_capacity"]) * 100
    
    # By type
    for res in resources:
        rtype = res["resource_type"]
        if rtype not in metrics["by_type"]:
            metrics["by_type"][rtype] = {"count": 0, "capacity": 0, "used": 0}
        
        metrics["by_type"][rtype]["count"] += 1
        metrics["by_type"][rtype]["capacity"] += res["capacity"]
        metrics["by_type"][rtype]["used"] += res["used"]
    
    # By location
    for res in resources:
        loc = res["location"]
        if loc not in metrics["by_location"]:
            metrics["by_location"][loc] = 0
        metrics["by_location"][loc] += 1
    
    # Utilization categories
    for res in resources:
        avail_pct = res["availability_percent"]
        if avail_pct < 30:  # Less than 30% available = high utilization
            metrics["high_utilization"].append(res["resource_id"])
        elif avail_pct > 70:  # More than 70% available = low utilization
            metrics["low_utilization"].append(res["resource_id"])
    
    print(f"  Total Resources: {metrics['total_resources']}")
    print(f"  Overall Utilization: {metrics.get('overall_utilization', 0):.1f}%")
    
    print(f"\n  Utilization by Type:")
    for rtype, data in metrics["by_type"].items():
        util = (data["used"] / data["capacity"]) * 100 if data["capacity"] > 0 else 0
        print(f"    {rtype}: {util:.1f}% ({data['count']} resources)")
    
    print(f"\n  Resources by Location:")
    for loc, count in metrics["by_location"].items():
        print(f"    {loc}: {count}")
    
    print(f"\n  High Utilization (>70%): {len(metrics['high_utilization'])}")
    print(f"  Low Utilization (<30%): {len(metrics['low_utilization'])}")
    
    return {
        **state,
        "resource_metrics": metrics,
        "messages": ["✓ Resource metrics analyzed"]
    }


def generate_resource_discovery_report_agent(state: ResourceDiscoveryPattern) -> ResourceDiscoveryPattern:
    """Generate resource discovery report"""
    print("\n" + "="*70)
    print("RESOURCE DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n📦 Resource Inventory:")
    print(f"  Total Resources: {len(state['resource_inventory'])}")
    
    print(f"\n🔧 Registered Resources:")
    for res_data in list(state['resource_inventory'].values())[:6]:
        print(f"\n  • {res_data['name']}:")
        print(f"      ID: {res_data['resource_id']}")
        print(f"      Type: {res_data['resource_type']}")
        print(f"      Location: {res_data['location']}")
        print(f"      Capacity: {res_data['capacity']}")
        print(f"      Used: {res_data['used']}")
        print(f"      Available: {res_data['availability_percent']:.1f}%")
        print(f"      Tags: {', '.join(res_data['tags'])}")
    
    print(f"\n📊 Resource Metrics:")
    metrics = state['resource_metrics']
    if metrics:
        print(f"  Overall Utilization: {metrics.get('overall_utilization', 0):.1f}%")
        print(f"  Total Capacity: {metrics.get('total_capacity', 0):,}")
        print(f"  Total Used: {metrics.get('total_used', 0):,}")
        
        print(f"\n  By Type:")
        for rtype, data in metrics.get('by_type', {}).items():
            util = (data['used'] / data['capacity']) * 100 if data['capacity'] > 0 else 0
            print(f"    {rtype}: {util:.1f}% utilization")
        
        print(f"\n  Resource Health:")
        print(f"    High Utilization: {len(metrics.get('high_utilization', []))}")
        print(f"    Low Utilization: {len(metrics.get('low_utilization', []))}")
    
    print(f"\n💡 Resource Discovery Benefits:")
    print("  ✓ Dynamic resource location")
    print("  ✓ Capacity planning")
    print("  ✓ Load distribution")
    print("  ✓ Resource optimization")
    print("  ✓ Availability monitoring")
    print("  ✓ Cost management")
    
    print(f"\n🔧 Discovery Methods:")
    print("  • Type-based")
    print("  • Tag-based")
    print("  • Location-based")
    print("  • Availability-based")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Resource allocation")
    print("  • Load balancing")
    print("  • Capacity planning")
    print("  • Cost optimization")
    print("  • Multi-cloud management")
    print("  • Infrastructure monitoring")
    
    print("\n" + "="*70)
    print("✅ Resource Discovery Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_resource_discovery_graph():
    """Create resource discovery workflow"""
    workflow = StateGraph(ResourceDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_resource_inventory_agent)
    workflow.add_node("register", register_resources_agent)
    workflow.add_node("discover", discover_resources_agent)
    workflow.add_node("analyze", analyze_resource_metrics_agent)
    workflow.add_node("report", generate_resource_discovery_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "register")
    workflow.add_edge("register", "discover")
    workflow.add_edge("discover", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 294: Resource Discovery MCP Pattern")
    print("="*70)
    
    app = create_resource_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "resource_inventory": {},
        "discovered_resources": [],
        "resource_metrics": {},
        "allocation_recommendations": []
    })
    
    print("\n✅ Resource Discovery Pattern Complete!")


if __name__ == "__main__":
    main()
