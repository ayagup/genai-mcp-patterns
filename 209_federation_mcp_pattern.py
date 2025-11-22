"""
Pattern 209: Federation MCP Pattern

Federation connects multiple independent systems into a unified whole:
- Independent systems maintain autonomy
- Unified access layer
- Cross-system queries and operations
- Distributed governance

Use Cases:
- Multi-region deployments
- Mergers & acquisitions
- Partner integrations
- Microservices coordination
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END


class FederationState(TypedDict):
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class FederatedSystem:
    system_id: str
    region: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def query(self, key: str) -> Any:
        return self.data.get(key)


class FederationLayer:
    def __init__(self):
        self.systems: Dict[str, FederatedSystem] = {}
    
    def register_system(self, system: FederatedSystem):
        self.systems[system.system_id] = system
    
    def federated_query(self, key: str) -> List[Dict[str, Any]]:
        """Query across all federated systems"""
        results = []
        for system in self.systems.values():
            value = system.query(key)
            if value:
                results.append({
                    'system': system.system_id,
                    'region': system.region,
                    'data': value
                })
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_systems': len(self.systems),
            'regions': list(set(s.region for s in self.systems.values()))
        }


def demo_agent(state: FederationState):
    operations = []
    results = []
    
    federation = FederationLayer()
    
    # Register systems from different regions
    systems = [
        FederatedSystem("us-east", "US-East"),
        FederatedSystem("us-west", "US-West"),
        FederatedSystem("eu-central", "Europe"),
        FederatedSystem("ap-south", "Asia")
    ]
    
    operations.append("Federation Setup:")
    for system in systems:
        federation.register_system(system)
        operations.append(f"  Registered: {system.system_id} ({system.region})")
    
    # Add data to each system
    systems[0].data["customer:123"] = {"name": "Alice", "region": "US"}
    systems[2].data["customer:123"] = {"name": "Alice", "region": "EU"}
    
    # Federated query
    operations.append("\nFederated Query for customer:123:")
    results_list = federation.federated_query("customer:123")
    for result in results_list:
        operations.append(f"  {result['system']} ({result['region']}): {result['data']}")
    
    stats = federation.get_stats()
    results.append(f"✓ Federation across {stats['total_systems']} systems")
    results.append(f"✓ Regions: {', '.join(stats['regions'])}")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": [f"Federated systems: {stats['total_systems']}"],
        "messages": ["Federation demo complete"]
    }


def create_federation_graph():
    workflow = StateGraph(FederationState)
    workflow.add_node("demo", demo_agent)
    workflow.add_edge(START, "demo")
    workflow.add_edge("demo", END)
    return workflow.compile()


def main():
    print("="*80)
    print("Pattern 209: Federation MCP Pattern")
    print("="*80)
    
    app = create_federation_graph()
    final_state = app.invoke({
        "scaling_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    })
    
    for op in final_state["scaling_operations"]:
        print(op)
    
    print("\n" + "="*80)
    print("""
Federation Benefits:
- Autonomy: Systems remain independent
- Unified Access: Single query interface
- Scalability: Add new systems easily
- Flexibility: Different technologies

Examples: Federated identity (OAuth/SAML), Federated databases, Kubernetes federation
""")


if __name__ == "__main__":
    main()
