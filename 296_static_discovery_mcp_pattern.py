"""
Pattern 296: Static Discovery MCP Pattern

This pattern demonstrates static discovery where service locations and
configurations are predefined and loaded from configuration files.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import json


class StaticDiscoveryPattern(TypedDict):
    """State for static discovery"""
    messages: Annotated[List[str], add]
    static_config: Dict[str, Any]
    service_endpoints: Dict[str, str]
    resolved_dependencies: Dict[str, List[str]]
    configuration_metadata: Dict[str, Any]


class StaticConfiguration:
    """Static service configuration"""
    
    def __init__(self):
        self.services = {}
        self.dependencies = {}
        self.metadata = {
            "version": "1.0.0",
            "environment": "production",
            "last_updated": "2024-01-15"
        }
    
    def load_from_dict(self, config: Dict[str, Any]):
        """Load configuration from dictionary"""
        self.services = config.get("services", {})
        self.dependencies = config.get("dependencies", {})
        self.metadata.update(config.get("metadata", {}))
    
    def get_service_endpoint(self, service_name: str):
        """Get service endpoint"""
        return self.services.get(service_name, {}).get("endpoint")
    
    def get_service_config(self, service_name: str):
        """Get full service configuration"""
        return self.services.get(service_name, {})
    
    def resolve_dependencies(self, service_name: str):
        """Resolve service dependencies"""
        if service_name not in self.dependencies:
            return []
        
        deps = self.dependencies[service_name]
        resolved = []
        
        for dep in deps:
            if dep in self.services:
                resolved.append({
                    "service": dep,
                    "endpoint": self.services[dep].get("endpoint"),
                    "available": True
                })
            else:
                resolved.append({
                    "service": dep,
                    "endpoint": None,
                    "available": False
                })
        
        return resolved
    
    def get_all_endpoints(self):
        """Get all service endpoints"""
        return {
            name: config.get("endpoint")
            for name, config in self.services.items()
        }


def initialize_static_config_agent(state: StaticDiscoveryPattern) -> StaticDiscoveryPattern:
    """Initialize static configuration"""
    print("\n📋 Initializing Static Configuration...")
    
    config = StaticConfiguration()
    
    print(f"  Configuration: Ready")
    print(f"  Type: Static (Predefined)")
    print(f"  Features:")
    print(f"    • Fixed endpoints")
    print(f"    • Configuration files")
    print(f"    • Dependency resolution")
    print(f"    • Version control")
    
    return {
        **state,
        "static_config": {},
        "service_endpoints": {},
        "resolved_dependencies": {},
        "configuration_metadata": {},
        "messages": ["✓ Static configuration initialized"]
    }


def load_static_config_agent(state: StaticDiscoveryPattern) -> StaticDiscoveryPattern:
    """Load static configuration"""
    print("\n📥 Loading Static Configuration...")
    
    config = StaticConfiguration()
    
    # Simulated configuration (normally loaded from file)
    static_config_data = {
        "metadata": {
            "version": "2.0.0",
            "environment": "production",
            "region": "us-east-1",
            "last_updated": "2024-11-29"
        },
        "services": {
            "auth_service": {
                "endpoint": "https://auth.example.com:8443",
                "protocol": "https",
                "timeout": 5000,
                "retry_policy": "exponential_backoff",
                "health_check": "/health"
            },
            "api_gateway": {
                "endpoint": "https://api.example.com:443",
                "protocol": "https",
                "timeout": 3000,
                "retry_policy": "simple",
                "health_check": "/status"
            },
            "database": {
                "endpoint": "postgresql://db.example.com:5432/maindb",
                "protocol": "postgresql",
                "timeout": 10000,
                "pool_size": 20,
                "ssl": True
            },
            "cache": {
                "endpoint": "redis://cache.example.com:6379",
                "protocol": "redis",
                "timeout": 1000,
                "ttl": 3600
            },
            "message_queue": {
                "endpoint": "amqp://mq.example.com:5672",
                "protocol": "amqp",
                "timeout": 5000,
                "queue_size": 1000
            },
            "storage": {
                "endpoint": "s3://storage.example.com/bucket",
                "protocol": "s3",
                "region": "us-east-1",
                "access_type": "private"
            }
        },
        "dependencies": {
            "api_gateway": ["auth_service", "cache"],
            "auth_service": ["database", "cache"],
            "database": [],
            "cache": [],
            "message_queue": [],
            "storage": []
        }
    }
    
    config.load_from_dict(static_config_data)
    
    print(f"  Loaded Services: {len(config.services)}")
    print(f"  Configuration Version: {config.metadata['version']}")
    print(f"  Environment: {config.metadata['environment']}")
    
    print(f"\n  Service Endpoints:")
    for service_name, service_config in config.services.items():
        print(f"    • {service_name}: {service_config['endpoint']}")
    
    return {
        **state,
        "static_config": static_config_data,
        "service_endpoints": config.get_all_endpoints(),
        "configuration_metadata": config.metadata,
        "messages": [f"✓ Loaded {len(config.services)} services"]
    }


def resolve_service_dependencies_agent(state: StaticDiscoveryPattern) -> StaticDiscoveryPattern:
    """Resolve service dependencies"""
    print("\n🔗 Resolving Service Dependencies...")
    
    config = StaticConfiguration()
    config.load_from_dict(state["static_config"])
    
    all_resolved = {}
    
    for service_name in config.services.keys():
        dependencies = config.resolve_dependencies(service_name)
        
        if dependencies:
            all_resolved[service_name] = dependencies
            
            print(f"\n  {service_name}:")
            for dep in dependencies:
                status = "✓" if dep["available"] else "✗"
                print(f"    {status} {dep['service']}: {dep.get('endpoint', 'N/A')}")
        else:
            print(f"\n  {service_name}: No dependencies")
    
    print(f"\n  Total Dependencies Resolved: {sum(len(deps) for deps in all_resolved.values())}")
    
    return {
        **state,
        "resolved_dependencies": all_resolved,
        "messages": [f"✓ Resolved dependencies for {len(all_resolved)} services"]
    }


def validate_static_config_agent(state: StaticDiscoveryPattern) -> StaticDiscoveryPattern:
    """Validate static configuration"""
    print("\n✅ Validating Static Configuration...")
    
    config = StaticConfiguration()
    config.load_from_dict(state["static_config"])
    
    validation_results = {
        "total_services": len(config.services),
        "valid_endpoints": 0,
        "invalid_endpoints": 0,
        "missing_dependencies": [],
        "circular_dependencies": []
    }
    
    # Validate endpoints
    for service_name, service_config in config.services.items():
        endpoint = service_config.get("endpoint")
        if endpoint and len(endpoint) > 0:
            validation_results["valid_endpoints"] += 1
        else:
            validation_results["invalid_endpoints"] += 1
    
    # Check for missing dependencies
    for service_name, deps in config.dependencies.items():
        for dep in deps:
            if dep not in config.services:
                validation_results["missing_dependencies"].append({
                    "service": service_name,
                    "missing": dep
                })
    
    # Validation summary
    print(f"  Total Services: {validation_results['total_services']}")
    print(f"  Valid Endpoints: {validation_results['valid_endpoints']}")
    print(f"  Invalid Endpoints: {validation_results['invalid_endpoints']}")
    
    if validation_results["missing_dependencies"]:
        print(f"\n  ⚠️ Missing Dependencies: {len(validation_results['missing_dependencies'])}")
        for issue in validation_results["missing_dependencies"][:3]:
            print(f"    {issue['service']} → {issue['missing']}")
    else:
        print(f"\n  ✓ All dependencies satisfied")
    
    is_valid = (validation_results["invalid_endpoints"] == 0 and 
                len(validation_results["missing_dependencies"]) == 0)
    
    status = "✅ VALID" if is_valid else "⚠️ ISSUES FOUND"
    print(f"\n  Configuration Status: {status}")
    
    return {
        **state,
        "messages": [f"✓ Validation complete: {status}"]
    }


def generate_static_discovery_report_agent(state: StaticDiscoveryPattern) -> StaticDiscoveryPattern:
    """Generate static discovery report"""
    print("\n" + "="*70)
    print("STATIC DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n📋 Configuration Metadata:")
    metadata = state["configuration_metadata"]
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔧 Service Endpoints:")
    print(f"  Total Services: {len(state['service_endpoints'])}")
    for service_name, endpoint in state["service_endpoints"].items():
        print(f"  • {service_name}:")
        print(f"      Endpoint: {endpoint}")
    
    print(f"\n🔗 Dependency Graph:")
    config_data = state["static_config"]
    dependencies = config_data.get("dependencies", {})
    
    for service, deps in dependencies.items():
        if deps:
            print(f"  {service} depends on:")
            for dep in deps:
                print(f"    → {dep}")
    
    print(f"\n📊 Service Details:")
    services = config_data.get("services", {})
    for service_name, service_config in list(services.items())[:4]:
        print(f"\n  {service_name}:")
        for key, value in service_config.items():
            if key != "endpoint":
                print(f"    {key}: {value}")
    
    print(f"\n💡 Static Discovery Benefits:")
    print("  ✓ Predictable endpoints")
    print("  ✓ Version controlled")
    print("  ✓ No runtime lookup")
    print("  ✓ Fast resolution")
    print("  ✓ Simple configuration")
    print("  ✓ Easy debugging")
    
    print(f"\n🔧 Configuration Management:")
    print("  • Configuration files")
    print("  • Environment variables")
    print("  • Hardcoded values")
    print("  • Config maps")
    print("  • Version control")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Small deployments")
    print("  • Monolithic apps")
    print("  • Fixed infrastructure")
    print("  • Development environments")
    print("  • Simple architectures")
    print("  • Legacy systems")
    
    print(f"\n⚠️ Limitations:")
    print("  • No runtime discovery")
    print("  • Manual updates required")
    print("  • Not suitable for dynamic scaling")
    print("  • Configuration drift risk")
    print("  • Harder to manage at scale")
    
    print(f"\n✨ Best Practices:")
    print("  • Use environment-specific configs")
    print("  • Version control configurations")
    print("  • Validate on deployment")
    print("  • Document dependencies")
    print("  • Regular config audits")
    
    print("\n" + "="*70)
    print("✅ Static Discovery Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_static_discovery_graph():
    """Create static discovery workflow"""
    workflow = StateGraph(StaticDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_static_config_agent)
    workflow.add_node("load", load_static_config_agent)
    workflow.add_node("resolve", resolve_service_dependencies_agent)
    workflow.add_node("validate", validate_static_config_agent)
    workflow.add_node("report", generate_static_discovery_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load")
    workflow.add_edge("load", "resolve")
    workflow.add_edge("resolve", "validate")
    workflow.add_edge("validate", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 296: Static Discovery MCP Pattern")
    print("="*70)
    
    app = create_static_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "static_config": {},
        "service_endpoints": {},
        "resolved_dependencies": {},
        "configuration_metadata": {}
    })
    
    print("\n✅ Static Discovery Pattern Complete!")


if __name__ == "__main__":
    main()
